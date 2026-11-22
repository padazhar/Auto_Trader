from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from execution.position_sizing import compute_position_plan
from core.market_io import latest_symbol_price, latest_symbol_price_status, normalize_price_components
from features.feature_store_contract import validate_fused_frame_freshness
from features.timeframe_fusion import build_fused_frame
from inference.expected_value import ActionEVSummary, adjust_ev_with_dqn, compute_action_evs, compute_cost_rate, estimate_payoff_metrics
from inference.model_artifact_loader import ModelArtifactLoader
from inference.regime_filter import evaluate_regime
from inference.prediction_engines import predict_dqn, predict_gbm, predict_rf, quality_gate_from_metrics
from inference.hmm_regime_confirmer import predict_hmm_regime
from inference.readiness import anchor_model_readiness




def _coerce_positive_price(value) -> float | None:
    try:
        price = float(value)
    except Exception:
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def _resolve_price_from_row(symbol: str, interval: str, row: pd.Series, paths) -> tuple[float | None, str, int | None]:
    runtime_status = latest_symbol_price_status(paths, symbol)
    runtime_source = str(runtime_status.get("reason") or runtime_status.get("price_source") or "snapshot_missing")
    if runtime_status.get("price_available") and runtime_status.get("price_fresh"):
        return float(runtime_status["price"]), str(runtime_status.get("price_source") or "runtime_market_price"), runtime_status.get("price_ts_ms")
    if runtime_status.get("price_available") and not runtime_status.get("price_fresh"):
        # Keep stale runtime prices as a fallback reference for the pure signal
        # decision path. Feature freshness is validated separately above, and
        # execution still re-prices orders through the execution layer. Returning
        # the stale price here preserves decision/risk-plan traceability instead
        # of collapsing the decision into an opaque price_unavailable HOLD.
        stale_source = str(runtime_status.get("price_source") or runtime_status.get("reason") or "runtime_market_price_stale")
        if stale_source == "market_latest_snapshot":
            stale_source = f"market_latest_snapshot:{interval}"
        return float(runtime_status["price"]), stale_source, runtime_status.get("price_ts_ms")

    interval_key = str(interval)
    candidates = [
        (f"{interval_key}_close", "fused_anchor_close"),
        ("close", "fused_close"),
        (f"{interval_key}_open", "fused_anchor_open"),
    ]
    for column, source in candidates:
        if column in row.index:
            price = _coerce_positive_price(row.get(column))
            if price is not None:
                return price, source, None

    db_path_getter = getattr(paths, "market_db_path", None)
    if callable(db_path_getter):
        db_path = Path(db_path_getter(symbol))
        table_name = f'klines_{interval_key}'
        if db_path.exists():
            try:
                with sqlite3.connect(db_path) as conn:
                    latest = conn.execute(
                        f'SELECT close FROM "{table_name}" WHERE close IS NOT NULL ORDER BY timestamp DESC LIMIT 1'
                    ).fetchone()
                if latest:
                    price = _coerce_positive_price(latest[0])
                    if price is not None:
                        return price, "market_db_latest_close", None
            except Exception:
                pass
    return None, runtime_source if runtime_source != "snapshot_missing" else "unavailable", None


@dataclass(slots=True)
class TradeDecision:
    """Typed result returned by the signal decision engine."""

    symbol: str
    interval: str
    action: str
    score: float
    reason: str
    components: dict[str, Any] = field(default_factory=dict)


class TradeDecisionLogic:
    """Generate pure trading decisions from approved inference artifacts."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.loader = ModelArtifactLoader(paths)
        self.cfg = paths.config.inference
        self.trading_cfg = getattr(paths.config, "trading", None)
        self.risk_cfg = getattr(paths.config, "risk", None)

    def _hold(self, symbol: str, interval: str, reason: str, components: dict[str, Any] | None = None) -> TradeDecision:
        payload = dict(components or {})
        if "price" in payload or "latest_price" in payload or "price_source" in payload:
            price_payload = normalize_price_components(
                payload.get("latest_price", payload.get("price")),
                price_source=str(payload.get("price_source") or reason),
                price_ts_ms=payload.get("price_ts_ms"),
                fallback_reason=reason if reason == "price_unavailable" else None,
            )
            payload.update(price_payload)
        payload = self._mark_decision_state(payload, action="HOLD", reason=reason, executable=False, blocker=reason)
        return TradeDecision(symbol=symbol, interval=interval, action="HOLD", score=0.0, reason=reason, components=payload)

    def _quality_gate(self, metrics: dict[str, Any] | None):
        return quality_gate_from_metrics(
            metrics,
            min_rows=self.cfg.quality.min_rows,
            min_precision_macro=self.cfg.quality.min_precision_macro,
            min_roc_auc_ovr=self.cfg.quality.min_roc_auc_ovr,
            min_balanced_accuracy=self.cfg.quality.min_balanced_accuracy,
        )

    def _trading_value(self, key: str, default: float, *, mode: str | None = None) -> float:
        """Return a trading threshold, optionally overridden by an operator mode profile.

        Mode profiles let the signal layer become more or less selective without
        changing model artifacts. Missing profile keys deliberately fall back to
        the base trading config so a partial profile cannot accidentally zero a
        safety threshold.
        """
        base_value = float(getattr(self.trading_cfg, key, default) or default)
        mode_key = str(mode or getattr(self.cfg, "active_mode", "") or "").strip()
        profiles = getattr(self.cfg, "mode_profiles", {}) or {}
        profile = profiles.get(mode_key) if isinstance(profiles, dict) else None
        if isinstance(profile, dict) and key in profile:
            try:
                return float(profile[key])
            except Exception:
                return base_value
        return base_value

    @staticmethod
    def _manifest_summary(artifact: Any, fallback_role: str | None = None) -> dict[str, Any]:
        manifest = dict(getattr(artifact, "manifest", {}) or {})
        lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
        feature_set = manifest.get("feature_set") if isinstance(manifest.get("feature_set"), dict) else {}
        timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
        return {
            "model_type": manifest.get("model_type"),
            "model_role": manifest.get("model_role") or fallback_role,
            "schema_version": manifest.get("schema_version"),
            "candidate_id": lifecycle.get("candidate_id"),
            "stage": lifecycle.get("stage"),
            "promoted_at": lifecycle.get("promoted_at"),
            "feature_set_version": feature_set.get("version"),
            "feature_set_signature": feature_set.get("signature"),
            "created_at": timestamps.get("created_at"),
            "updated_at": timestamps.get("updated_at"),
        }

    @staticmethod
    def _ev_components(summary: ActionEVSummary) -> dict[str, Any]:
        return {
            "long": {
                "value": summary.long.value,
                "success_prob": summary.long.success_prob,
                "failure_prob": summary.long.failure_prob,
                "abstain_prob": summary.long.abstain_prob,
            },
            "hold": {
                "value": summary.hold.value,
                "success_prob": summary.hold.success_prob,
                "failure_prob": summary.hold.failure_prob,
                "abstain_prob": summary.hold.abstain_prob,
            },
            "short": {
                "value": summary.short.value,
                "success_prob": summary.short.success_prob,
                "failure_prob": summary.short.failure_prob,
                "abstain_prob": summary.short.abstain_prob,
            },
            "best_directional_action": summary.best_directional.action,
            "best_directional_value": summary.best_directional.value,
            "best_overall_action": summary.best_overall.action,
            "best_overall_value": summary.best_overall.value,
        }

    @staticmethod
    def _direction_from_action(action: str) -> int:
        normalized = str(action or "").upper()
        if normalized == "LONG":
            return 1
        if normalized == "SHORT":
            return -1
        return 0

    @staticmethod
    def _mark_decision_state(
        components: dict[str, Any],
        *,
        action: str,
        reason: str,
        executable: bool,
        blocker: str | None = None,
    ) -> dict[str, Any]:
        """Attach the canonical decision/execution state to decision components."""
        payload = dict(components or {})
        normalized_action = str(action or "HOLD").upper()
        normalized_reason = str(reason or "unspecified")
        normalized_blocker = str(blocker or (normalized_reason if not executable else ""))
        payload["decision_state"] = {
            "action": normalized_action,
            "reason": normalized_reason,
            "execution_eligible": bool(executable),
            "blocker": normalized_blocker,
        }
        payload["execution_eligible"] = bool(executable)
        payload["decision_blocker"] = normalized_blocker
        return payload

    def _runtime_execution_mode(self) -> str:
        runtime_cfg = getattr(self.paths.config, "runtime", None)
        return str(getattr(runtime_cfg, "execution_mode", "paper") or "paper").strip().lower()

    def _hmm_paper_gate_active(self) -> bool:
        """Return True only for the approved Step 12 paper-entry HMM gate.

        HMM may be a mandatory confirmer for paper entries after Step 11
        approval, but it is not allowed to influence live trading, EV maths,
        sizing, leverage, exits, or risk.
        """
        hmm_cfg = getattr(self.cfg, "hmm", None)
        if hmm_cfg is None:
            return False
        if not bool(getattr(hmm_cfg, "enabled", False)):
            return False
        if bool(getattr(hmm_cfg, "live_influence_enabled", False)):
            return False
        return bool(getattr(hmm_cfg, "mandatory_paper_gate_enabled", False))

    def _hmm_shadow_component(self, symbol: str, interval: str, row: pd.Series | None) -> dict[str, Any]:
        """Return HMM telemetry and, when approved, paper-entry gate evidence.

        Step 12 permits HMM to act only as a mandatory paper-entry confirmer.
        This component is still excluded from RF/GBM EV scoring and risk sizing.
        It may only block new paper entries/flips after the RF/GBM decision has
        already been computed.
        """
        hmm_cfg = getattr(self.cfg, "hmm", None)
        mandatory_gate_enabled = bool(getattr(hmm_cfg, "mandatory_paper_gate_enabled", False)) if hmm_cfg else False
        live_influence_enabled = bool(getattr(hmm_cfg, "live_influence_enabled", False)) if hmm_cfg else False
        inference_enabled = bool(getattr(hmm_cfg, "enabled", False)) if hmm_cfg else False
        shadow_enabled = bool(getattr(hmm_cfg, "candidate_shadow_enabled", False)) if hmm_cfg else False
        candidate_inference_enabled = bool(
            inference_enabled
            and not live_influence_enabled
            and (shadow_enabled or mandatory_gate_enabled)
        )
        if candidate_inference_enabled and row is not None:
            hmm_artifact = None
            load_candidate_hmm = getattr(self.loader, "load_candidate_hmm", None)
            if callable(load_candidate_hmm):
                hmm_artifact = load_candidate_hmm(symbol, interval)
            hmm_prediction = predict_hmm_regime(
                hmm_artifact,
                row,
                anchor_interval=self.paths.config.frames.anchor_interval,
                min_confidence=float(getattr(hmm_cfg, "min_posterior_confidence", 0.60) or 0.60),
            )
            payload = hmm_prediction.as_dict()
        else:
            payload = {
                "available": False,
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_gate_enabled": mandatory_gate_enabled,
                "live_influence_enabled": live_influence_enabled,
                "reason": "hmm_shadow_disabled" if not candidate_inference_enabled else "hmm_feature_row_unavailable",
            }
        # Keep artifact governance and runtime gate policy separate.  HMM
        # artifacts remain shadow candidates; the runtime may nevertheless use
        # a valid candidate as a mandatory PAPER entry/flip confirmer.
        payload["artifact_governance_role"] = str((payload.get("artifact") or {}).get("governance_role") or "shadow_candidate")
        payload["artifact_shadow_only"] = bool((payload.get("artifact") or {}).get("shadow_only", payload.get("shadow_only", True)))
        payload["artifact_decision_influence_allowed"] = bool((payload.get("artifact") or {}).get("decision_influence_allowed", False))
        payload["runtime_gate_policy"] = {
            "scope": "paper_entry_or_flip",
            "mandatory_enabled": mandatory_gate_enabled,
            "require_valid_artifact": bool(getattr(hmm_cfg, "require_valid_artifact_for_entry", True)) if hmm_cfg else True,
            "live_influence_enabled": live_influence_enabled,
        }
        payload["used"] = False
        payload["gate_applies"] = False
        payload["gate_used"] = False
        payload["gate_result"] = "not_applicable"
        payload["gate_reason"] = "hmm_gate_not_applied"
        payload["decision_effect"] = "none"
        payload["decision_influence_allowed"] = False
        payload["mandatory_gate_enabled"] = mandatory_gate_enabled
        payload["live_influence_enabled"] = live_influence_enabled
        payload["runtime_execution_mode"] = self._runtime_execution_mode()
        return payload

    def _hmm_gate_block_reason(self, action: str, components: dict[str, Any]) -> str | None:
        """Return a fail-closed HOLD reason when Step 12 HMM gate rejects an entry.

        The gate applies only to new entries and flips. Exits/protective actions
        are never blocked by HMM. Passing the gate does not change EV, direction,
        sizing, leverage, or risk.
        """
        normalized_action = str(action or "HOLD").upper()
        hmm_component = components.setdefault("hmm", {})
        applies = normalized_action.startswith("ENTER") or normalized_action.startswith("FLIP")
        hmm_component["gate_applies"] = bool(applies)
        hmm_component["gate_used"] = False
        if not applies:
            hmm_component["gate_result"] = "not_applicable"
            hmm_component["gate_reason"] = "not_entry_or_flip"
            return None

        hmm_cfg = getattr(self.cfg, "hmm", None)
        mandatory_enabled = bool(getattr(hmm_cfg, "mandatory_paper_gate_enabled", False)) if hmm_cfg else False
        live_influence_enabled = bool(getattr(hmm_cfg, "live_influence_enabled", False)) if hmm_cfg else False
        runtime_mode = self._runtime_execution_mode()
        hmm_component["mandatory_gate_enabled"] = mandatory_enabled
        hmm_component["live_influence_enabled"] = live_influence_enabled
        hmm_component["runtime_execution_mode"] = runtime_mode

        if not mandatory_enabled:
            hmm_component["gate_result"] = "not_applicable"
            hmm_component["gate_reason"] = "mandatory_paper_gate_disabled"
            return None
        hmm_component["gate_used"] = True
        hmm_component["used"] = True
        hmm_component["decision_effect"] = "paper_entry_gate"

        if live_influence_enabled:
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = "hmm_live_influence_forbidden"
            return "hmm_gate_blocked:live_influence_forbidden"
        if runtime_mode != "paper":
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = "hmm_paper_gate_live_mode_forbidden"
            return "hmm_gate_blocked:paper_gate_live_mode_forbidden"
        if not bool(getattr(hmm_cfg, "require_valid_artifact_for_entry", True)):
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = "hmm_valid_artifact_requirement_disabled"
            return "hmm_gate_blocked:valid_artifact_requirement_disabled"
        if not bool(hmm_component.get("available", False)):
            reason = str(hmm_component.get("reason") or "hmm_artifact_missing")
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = reason
            return f"hmm_gate_blocked:{reason}"
        if hmm_component.get("state") is None:
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = "hmm_state_missing"
            return "hmm_gate_blocked:state_missing"
        try:
            confidence = float(hmm_component.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        min_confidence = float(getattr(hmm_cfg, "min_posterior_confidence", 0.60) or 0.60)
        if confidence < min_confidence:
            hmm_component["gate_result"] = "blocked"
            hmm_component["gate_reason"] = f"hmm_low_confidence:{confidence:.6f}<{min_confidence:.6f}"
            return "hmm_gate_blocked:low_confidence"

        # Step 12 intentionally treats a valid, high-confidence HMM state as the
        # mandatory paper confirmer. No state-to-direction mapping is inferred
        # here, which avoids leaking arbitrary regime labels into EV maths.
        hmm_component["gate_result"] = "passed"
        hmm_component["gate_reason"] = "hmm_valid_high_confidence_paper_confirmer"
        return None

    def decide(self, symbol: str, interval: str, previous_position: int = 0, *, use_candidate_artifacts: bool = False, mode: str | None = None) -> TradeDecision:
        if not use_candidate_artifacts:
            readiness = anchor_model_readiness(self.paths, symbol=symbol, interval=interval)
        else:
            readiness = None

        fused = build_fused_frame(self.paths, symbol, interval, tail_rows=self.cfg.tail_rows)
        if fused.empty:
            return self._hold(symbol, interval, "no_features")

        freshness = validate_fused_frame_freshness(self.paths, fused, symbol=symbol, anchor_interval=interval)
        if not bool(freshness.get("ok", False)):
            return self._hold(symbol, interval, str(freshness.get("reason") or "stale_features"), {"freshness": freshness})

        row = fused.iloc[-1]
        price, price_source, price_ts_ms = _resolve_price_from_row(symbol, interval, row, self.paths)
        price_payload = normalize_price_components(
            price,
            price_source=price_source,
            price_ts_ms=price_ts_ms,
            fallback_reason=price_source if price is None else None,
        )
        runtime_price_status = latest_symbol_price_status(self.paths, symbol)
        price_payload.update({
            "price_fresh": bool(runtime_price_status.get("price_fresh")) if runtime_price_status.get("price_available") else False,
            "price_age_seconds": runtime_price_status.get("price_age_seconds"),
            "price_max_age_seconds": runtime_price_status.get("max_age_seconds"),
            "price_status_reason": runtime_price_status.get("reason"),
        })
        if price is None:
            return self._hold(symbol, interval, str(price_source or "price_unavailable"), {"freshness": freshness, **price_payload})
        atr_fraction = abs(float(row.get(f"{interval}_atr_pct_14", row.get("atr_pct_14", 0.0)) or 0.0))
        ev_cost_rate = compute_cost_rate(self.trading_cfg)
        regime = evaluate_regime(row, interval=interval, regime_cfg=getattr(self.paths.config, "regime", None))
        deployment_stage = "candidate_shadow" if use_candidate_artifacts else "incumbent"
        effective_mode = str(mode or getattr(self.cfg, "active_mode", "Medium") or "Medium").title()
        components: dict[str, Any] = {
            "regime": regime.as_dict(),
            "freshness": freshness,
            "deployment_stage": deployment_stage,
            "decision_mode": effective_mode,
            **price_payload,
        }
        components["hmm"] = self._hmm_shadow_component(symbol, interval, row)

        if readiness is not None and not readiness.ready:
            components["anchor_readiness"] = {k: v.reason for k, v in readiness.models.items()}
            return self._hold(symbol, interval, readiness.reason, components)

        rf_artifact = self.loader.load_candidate_rf(symbol, interval) if use_candidate_artifacts else self.loader.load_rf(symbol, interval)
        rf_gate = self._quality_gate(rf_artifact.metrics if rf_artifact else {}) if rf_artifact else None
        rf_pred = predict_rf(rf_artifact, row)
        components["rf"] = {
            "available": rf_pred.available,
            "gate": rf_gate.reasons if rf_gate else ["artifact_missing"],
            "p_short": rf_pred.short_probability,
            "p_hold": rf_pred.hold_probability,
            "p_long": rf_pred.long_probability,
            "direction_score": rf_pred.direction_score,
            "artifact": self._manifest_summary(rf_artifact, "direction") if rf_artifact else {},
        }

        gbm_artifact = self.loader.load_candidate_gbm(symbol, interval) if use_candidate_artifacts else self.loader.load_gbm(symbol, interval)
        gbm_direction_metrics = gbm_artifact.direction.metrics if gbm_artifact and gbm_artifact.direction else {}
        gbm_gate = self._quality_gate(gbm_direction_metrics) if gbm_artifact and gbm_artifact.direction else None
        gbm_pred = predict_gbm(gbm_artifact, row)
        components["gbm"] = {
            "available": gbm_pred.available,
            "gate": gbm_gate.reasons if gbm_gate else ["artifact_missing"],
            "p_short": gbm_pred.short_probability,
            "p_hold": gbm_pred.hold_probability,
            "p_long": gbm_pred.long_probability,
            "direction_score": gbm_pred.direction_score,
            "meta_probability": gbm_pred.meta_probability,
            "effective_score": gbm_pred.effective_score,
            "meta_available": gbm_pred.meta_available,
            "artifact": {
                "direction": self._manifest_summary(gbm_artifact.direction, "direction") if gbm_artifact and gbm_artifact.direction else {},
                "meta": self._manifest_summary(gbm_artifact.meta, "meta") if gbm_artifact and gbm_artifact.meta else {},
            },
        }

        weighted_scores: list[float] = []
        directional_signs: list[int] = []
        contributing_models = 0
        rf_metrics = dict(rf_artifact.metrics or {}) if rf_artifact else {}
        gbm_metrics = dict(gbm_artifact.direction.metrics or {}) if gbm_artifact and gbm_artifact.direction else {}

        anchor_strength = max(rf_pred.long_probability, rf_pred.short_probability) if rf_pred.available else 0.0
        lower_tf_scale = 1.0
        if interval != self.paths.config.frames.anchor_interval:
            if anchor_strength < self.cfg.ev.weak_anchor_probability:
                lower_tf_scale = self.cfg.ev.weak_anchor_lower_tf_scale
            elif anchor_strength >= self.cfg.ev.strong_anchor_probability:
                lower_tf_scale = 1.0

        if rf_pred.available and rf_gate and rf_gate.passed:
            avg_win, avg_loss = estimate_payoff_metrics(rf_metrics, atr_fraction=atr_fraction, cost_rate=ev_cost_rate)
            rf_summary = compute_action_evs(
                p_long=rf_pred.long_probability,
                p_hold=rf_pred.hold_probability,
                p_short=rf_pred.short_probability,
                avg_win=avg_win,
                avg_loss=avg_loss,
            )
            components["rf"]["action_evs"] = self._ev_components(rf_summary)
            directional_choice = rf_summary.best_directional
            if rf_summary.best_overall.action == "HOLD":
                weighted_scores.append(0.0)
            else:
                weighted_scores.append(directional_choice.value * self.cfg.weights.rf * lower_tf_scale)
            directional_sign = self._direction_from_action(rf_summary.best_overall.action)
            if directional_sign != 0:
                directional_signs.append(directional_sign)
            contributing_models += 1

        if gbm_pred.available and gbm_gate and gbm_gate.passed:
            if gbm_pred.meta_available and gbm_pred.meta_probability <= self.cfg.ev.meta_threshold:
                components["gbm"]["meta_blocked"] = True
                components["gbm"]["effective_ev"] = 0.0
                weighted_scores.append(0.0)
                contributing_models += 1
            else:
                avg_win, avg_loss = estimate_payoff_metrics(gbm_metrics, atr_fraction=atr_fraction, cost_rate=ev_cost_rate)
                gbm_summary = compute_action_evs(
                    p_long=gbm_pred.long_probability,
                    p_hold=gbm_pred.hold_probability,
                    p_short=gbm_pred.short_probability,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                )
                directional_choice = gbm_summary.best_directional
                gbm_effective_ev = directional_choice.value * (gbm_pred.meta_probability if gbm_pred.meta_available else 1.0)
                components["gbm"]["action_evs"] = self._ev_components(gbm_summary)
                components["gbm"]["effective_ev"] = gbm_effective_ev
                if gbm_summary.best_overall.action == "HOLD":
                    weighted_scores.append(0.0)
                else:
                    weighted_scores.append(gbm_effective_ev * self.cfg.weights.gbm * lower_tf_scale)
                directional_sign = self._direction_from_action(gbm_summary.best_overall.action)
                if directional_sign != 0:
                    directional_signs.append(directional_sign)
                contributing_models += 1

        dqn_artifact = self.loader.load_candidate_dqn(symbol, interval) if use_candidate_artifacts else self.loader.load_dqn(symbol, interval)
        dqn_used = False
        dqn_pred = predict_dqn(dqn_artifact, row)
        rf_confidence = max(rf_pred.short_probability, rf_pred.hold_probability, rf_pred.long_probability)
        rf_gray = self.cfg.dqn.gray_band_low <= rf_confidence <= self.cfg.dqn.gray_band_high
        dqn_execution_weight = float(getattr(self.cfg.weights, "dqn", 0.0) or 0.0)
        dqn_adjustment_strength = float(getattr(self.cfg.ev, "dqn_adjustment_strength", 0.0) or 0.0)
        dqn_influence_enabled = dqn_execution_weight > 0.0 and dqn_adjustment_strength > 0.0
        components["dqn"] = {
            "available": dqn_pred.available,
            "used": False,
            "role": "confirmatory_policy",
            "execution_weight": dqn_execution_weight,
            "adjustment_strength": dqn_adjustment_strength,
            "influence_enabled": dqn_influence_enabled,
            "action": dqn_pred.action,
            "confidence": dqn_pred.confidence,
            "score": dqn_pred.score,
            "gray_band": rf_gray,
            "artifact": self._manifest_summary(dqn_artifact, "confirmatory_policy") if dqn_artifact else {},
        }
        if dqn_influence_enabled and dqn_pred.available and rf_gray and contributing_models > 0:
            dqn_used = True
            components["dqn"]["used"] = True
            weighted_scores = [
                adjust_ev_with_dqn(score, dqn_score=dqn_pred.score, strength=dqn_adjustment_strength)
                for score in weighted_scores
            ]
            if abs(dqn_pred.score) > 0:
                directional_signs.append(1 if dqn_pred.score > 0 else -1)

        components["hmm"] = dict(components.get("hmm") or self._hmm_shadow_component(symbol, interval, row))
        if not bool(components["hmm"].get("gate_used", False)):
            components["hmm"]["used"] = False
            components["hmm"]["decision_effect"] = "none"
            components["hmm"]["decision_influence_allowed"] = False

        if contributing_models < self.cfg.minimum_models_required:
            components["consensus"] = {"contributing_models": contributing_models, "required": self.cfg.minimum_models_required}
            missing_reason = "candidate_artifact_missing" if use_candidate_artifacts else "insufficient_models"
            return self._hold(symbol, interval, missing_reason, components)

        non_zero_signs = [sign for sign in directional_signs if sign != 0]
        if self.cfg.consensus.require_agreement and non_zero_signs and len(set(non_zero_signs)) > 1:
            components["consensus"] = {
                "contributing_models": contributing_models,
                "agreement": False,
                "directional_signs": non_zero_signs,
            }
            return self._hold(symbol, interval, "model_disagreement", components)

        weight_sum = (self.cfg.weights.rf + self.cfg.weights.gbm) * lower_tf_scale
        # Regime output is shadow telemetry only under the approved governance scope.
        # It must not scale active decision scores, actions, risk, or execution eligibility.
        score = float(sum(weighted_scores) / max(weight_sum, 1e-9))
        direction_sign = 1 if sum(non_zero_signs) > 0 else (-1 if sum(non_zero_signs) < 0 else 0)

        min_ev_threshold = self._trading_value("ev_threshold", self.cfg.ev.threshold, mode=effective_mode)
        min_atr_fraction = self._trading_value("min_atr_fraction", 0.0, mode=effective_mode)
        min_short_prob = self._trading_value("short_probability_threshold", 0.60, mode=effective_mode)
        min_long_prob = self._trading_value("long_probability_threshold", 0.55, mode=effective_mode)
        effective_meta_probability = gbm_pred.meta_probability if gbm_pred.meta_available else 1.0
        short_strength = max(rf_pred.short_probability, gbm_pred.short_probability)
        long_strength = max(rf_pred.long_probability, gbm_pred.long_probability)

        components["consensus"] = {
            "contributing_models": contributing_models,
            "required": self.cfg.minimum_models_required,
            "ev_score": score,
            "directional_signs": non_zero_signs,
            "dqn_used": dqn_used,
            "lower_tf_scale": lower_tf_scale,
            "direction_sign": direction_sign,
            "atr_fraction": atr_fraction,
            "cost_rate": ev_cost_rate,
            "cost_bps": ev_cost_rate * 10000.0,
            "meta_probability": effective_meta_probability,
            "short_strength": short_strength,
            "long_strength": long_strength,
            "thresholds": {
                "ev_threshold": min_ev_threshold,
                "min_atr_fraction": min_atr_fraction,
                "short_probability_threshold": min_short_prob,
                "long_probability_threshold": min_long_prob,
            },
            "regime_influence_applied": False,
        }
        components["lineage"] = {
            "symbol": symbol,
            "interval": interval,
            "anchor_interval": self.paths.config.frames.anchor_interval,
            "rf": components["rf"].get("artifact", {}),
            "gbm": components["gbm"].get("artifact", {}),
            "dqn": components["dqn"].get("artifact", {}),
            "hmm": components.get("hmm", {}).get("artifact", {}),
            "deployment_stage": deployment_stage,
        }

        # HMM remains excluded from EV/direction/risk.  The approved paper-only
        # confirmer may block a new entry/flip after the RF/GBM action exists.
        if score < min_ev_threshold:
            return self._hold(symbol, interval, "negative_ev", components)
        if atr_fraction < min_atr_fraction:
            return self._hold(symbol, interval, "movement_filter", components)
        if direction_sign < 0 and short_strength < min_short_prob:
            return self._hold(symbol, interval, "weak_short_filter", components)
        if direction_sign > 0 and long_strength < min_long_prob:
            return self._hold(symbol, interval, "weak_long_filter", components)

        # Flips from an existing position may use a slightly lower threshold
        # than fresh entries, but never a negative threshold.
        flip_thr = max(min_ev_threshold * 0.85, 0.0)
        exit_band = max(min_ev_threshold * 0.45, 1e-9)

        if previous_position > 0:
            if direction_sign < 0 and score >= flip_thr:
                action, reason = "FLIP_TO_SHORT", "ev_flip_short"
            elif score <= exit_band:
                action, reason = "EXIT_LONG", "ev_exit_band"
            else:
                action, reason = "HOLD", "hold_long_bias"
        elif previous_position < 0:
            if direction_sign > 0 and score >= flip_thr:
                action, reason = "FLIP_TO_LONG", "ev_flip_long"
            elif score <= exit_band:
                action, reason = "EXIT_SHORT", "ev_exit_band"
            else:
                action, reason = "HOLD", "hold_short_bias"
        else:
            if direction_sign > 0 and score >= min_ev_threshold:
                action, reason = "ENTER_LONG", "ev_enter_long"
            elif direction_sign < 0 and score >= min_ev_threshold:
                action, reason = "ENTER_SHORT", "ev_enter_short"
            else:
                action, reason = "HOLD", "ev_hold"

        hmm_gate_blocker = self._hmm_gate_block_reason(action, components)
        if hmm_gate_blocker is not None:
            return self._hold(symbol, interval, hmm_gate_blocker, components)

        if action.startswith("ENTER") or action.startswith("FLIP"):
            side = "LONG" if "LONG" in action else "SHORT"
            components["risk_plan"] = compute_position_plan(
                balance=self.risk_cfg.account_balance_usd,
                risk_per_trade=self.risk_cfg.risk_per_trade,
                price=price,
                atr_fraction=max(atr_fraction, 1e-6),
                sl_atr_mult=self.risk_cfg.sl_atr_mult,
                tp_atr_mult=self.risk_cfg.tp_atr_mult,
                max_notional_usd=self.risk_cfg.max_notional_usd,
                min_notional_usd=self.risk_cfg.min_notional_usd,
                direction=side,
            ).__dict__

        components.update(price_payload)
        components["atr_fraction"] = atr_fraction
        executable = action.startswith("ENTER") or action.startswith("FLIP") or action.startswith("EXIT")
        blocker = "" if executable else reason
        components = self._mark_decision_state(
            components,
            action=action,
            reason=reason,
            executable=executable,
            blocker=blocker,
        )
        return TradeDecision(symbol=symbol, interval=interval, action=action, score=score, reason=reason, components=components)
