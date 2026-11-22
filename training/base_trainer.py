from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except Exception:  # pragma: no cover - older sklearn
    FrozenEstimator = None


from core.artifact_contract import (
    incumbent_pointer_path,
    load_manifest_metrics,
    promote_candidate_manifest,
    prune_related_candidate_artifacts,
    resolve_active_manifest_path,
    save_json,
    write_manifest,
)
from core.log_manager import get_logger
from core.result_contracts import TrainerResult
from features.dataset_preparation import build_feature_set_metadata
from features.feature_store_contract import resolve_feature_recipe_version, validate_feature_store_metadata
from features.timeframe_fusion import build_fused_frame


class BaseTrainer(ABC):
    """Base trainer with shared progress, manifest, calibration, and result helpers."""

    model_type: str = "base"

    def __init__(self, paths) -> None:
        self.paths = paths
        self.logger = get_logger(f"training.{self.model_type}", paths)

    def progress(self, symbol: str, interval: str, percent: float, message: str, **extra: Any) -> None:
        """Persist trainer progress for observability and UI consumption."""
        payload = {
            "model_type": self.model_type,
            "symbol": symbol,
            "interval": interval,
            "percent": max(0.0, min(100.0, float(percent))),
            "message": str(message),
        }
        if extra:
            payload["extra"] = dict(extra)
        self.logger.info(
            "progress symbol=%s interval=%s percent=%.1f message=%s extra=%s",
            symbol,
            interval,
            payload["percent"],
            message,
            extra or {},
        )
        try:
            path = self.paths.trainer_progress_path(self.model_type, symbol, interval)
            save_json(path, payload)
        except Exception:
            self.logger.exception("Failed to persist trainer progress symbol=%s interval=%s", symbol, interval)

    def load_fused_frame(self, symbol: str, interval: str, tail_rows: int) -> pd.DataFrame:
        """Load the canonical fused feature frame for a symbol/interval pair."""
        frame = build_fused_frame(self.paths, symbol, interval, tail_rows=tail_rows)
        self.logger.info(
            "loaded fused frame symbol=%s interval=%s rows=%s cols=%s tail_rows=%s",
            symbol,
            interval,
            len(frame),
            len(frame.columns),
            tail_rows,
        )
        return frame

    @staticmethod
    def new_candidate_id() -> str:
        """Create a unique candidate identifier for staged artifacts."""
        return uuid.uuid4().hex[:16]

    @staticmethod
    def fit_calibrated_classifier(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Fit a classifier and calibrate its probabilities on a trailing holdout slice."""
        fitted = clone(model)
        fitted.fit(X_train, y_train)

        if FrozenEstimator is not None:
            try:
                calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(fitted), method="sigmoid")
                calibrator.fit(X_cal, y_cal)
                return calibrator
            except Exception:
                pass

        try:
            calibrator = CalibratedClassifierCV(estimator=fitted, method="sigmoid", cv="prefit")
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=fitted, method="sigmoid", cv="prefit")
        calibrator.fit(X_cal, y_cal)
        return calibrator


    @staticmethod
    def directional_quality_report(y_true: Any, y_pred: Any, *, class_encoding: dict[int, str] | None = None) -> dict[str, Any]:
        """Build institutional-style directional quality metrics for LONG/SHORT usability.

        The report is intentionally descriptive only. It does not change promotion
        decisions in this step; Step 4 may later consume these fields for gates.
        """
        encoding = class_encoding or {0: "short", 1: "hold", 2: "long"}
        try:
            true_series = pd.Series(y_true).dropna().astype(int)
            pred_series = pd.Series(y_pred).dropna().astype(int)
        except Exception:
            return {
                "directional_quality": {
                    "available": False,
                    "reason": "invalid_label_or_prediction_vector",
                }
            }
        n = int(min(len(true_series), len(pred_series)))
        if n <= 0:
            return {
                "directional_quality": {
                    "available": False,
                    "reason": "empty_label_or_prediction_vector",
                }
            }
        true_values = true_series.iloc[:n].to_numpy(dtype=int)
        pred_values = pred_series.iloc[:n].to_numpy(dtype=int)
        labels = sorted(set(encoding.keys()) | set(int(v) for v in true_values.tolist()) | set(int(v) for v in pred_values.tolist()))

        confusion: dict[str, dict[str, int]] = {}
        per_class: dict[str, dict[str, Any]] = {}
        true_counts: dict[str, int] = {}
        pred_counts: dict[str, int] = {}

        for actual in labels:
            actual_name = str(encoding.get(actual, actual))
            actual_mask = true_values == actual
            true_count = int(actual_mask.sum())
            true_counts[actual_name] = true_count
            confusion[actual_name] = {}
            for predicted in labels:
                predicted_name = str(encoding.get(predicted, predicted))
                predicted_mask = pred_values == predicted
                if actual == labels[0]:
                    pred_counts[predicted_name] = int(predicted_mask.sum())
                confusion[actual_name][predicted_name] = int((actual_mask & predicted_mask).sum())

        for cls in labels:
            name = str(encoding.get(cls, cls))
            tp = int(((true_values == cls) & (pred_values == cls)).sum())
            fp = int(((true_values != cls) & (pred_values == cls)).sum())
            fn = int(((true_values == cls) & (pred_values != cls)).sum())
            precision = tp / float(tp + fp) if (tp + fp) else 0.0
            recall = tp / float(tp + fn) if (tp + fn) else 0.0
            per_class[name] = {
                "precision": BaseTrainer.safe_float(precision),
                "recall": BaseTrainer.safe_float(recall),
                "support": int((true_values == cls).sum()),
                "predicted": int((pred_values == cls).sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        hold_name = str(encoding.get(1, "hold"))
        long_name = str(encoding.get(2, "long"))
        short_name = str(encoding.get(0, "short"))
        hold_true_fraction = true_counts.get(hold_name, 0) / float(n)
        hold_pred_fraction = pred_counts.get(hold_name, 0) / float(n)
        directional_true_fraction = 1.0 - hold_true_fraction
        directional_pred_fraction = 1.0 - hold_pred_fraction

        return {
            "directional_quality": {
                "available": True,
                "rows_valid": n,
                "class_encoding": {str(k): str(v) for k, v in encoding.items()},
                "class_distribution_valid": true_counts,
                "prediction_distribution_valid": pred_counts,
                "confusion_matrix": confusion,
                "per_class": per_class,
                "long_precision": per_class.get(long_name, {}).get("precision"),
                "long_recall": per_class.get(long_name, {}).get("recall"),
                "short_precision": per_class.get(short_name, {}).get("precision"),
                "short_recall": per_class.get(short_name, {}).get("recall"),
                "hold_true_fraction": BaseTrainer.safe_float(hold_true_fraction),
                "hold_prediction_fraction": BaseTrainer.safe_float(hold_pred_fraction),
                "directional_true_fraction": BaseTrainer.safe_float(directional_true_fraction),
                "directional_prediction_fraction": BaseTrainer.safe_float(directional_pred_fraction),
                "hold_dominance_warning": bool(hold_pred_fraction >= 0.70 or hold_true_fraction >= 0.70),
            }
        }


    @staticmethod
    def calibration_quality_report(y_true: Any, y_prob: Any, *, class_labels: list[int] | None = None, bins: int = 10) -> dict[str, Any]:
        """Measure probability calibration quality on the validation slice.

        The report is model-agnostic and safe for both binary and multiclass
        estimators.  It intentionally uses the same validation probabilities
        consumed by EV logic so promotion can reject models whose confidence is
        not reliable enough for risk decisions.
        """
        try:
            true_values = pd.Series(y_true).dropna().astype(int).to_numpy(dtype=int)
            probabilities = np.asarray(y_prob, dtype=float)
        except Exception:
            return {
                "calibration": {
                    "available": False,
                    "reason": "invalid_probability_vector",
                }
            }
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        if probabilities.ndim != 2 or probabilities.shape[0] <= 0 or len(true_values) <= 0:
            return {
                "calibration": {
                    "available": False,
                    "reason": "empty_probability_matrix",
                }
            }
        n = int(min(len(true_values), probabilities.shape[0]))
        if n <= 0:
            return {
                "calibration": {
                    "available": False,
                    "reason": "empty_probability_matrix",
                }
            }
        probabilities = probabilities[:n].copy()
        true_values = true_values[:n]
        row_sums = probabilities.sum(axis=1)
        valid_rows = np.isfinite(probabilities).all(axis=1) & np.isfinite(row_sums) & (row_sums > 0.0)
        if not bool(valid_rows.any()):
            return {
                "calibration": {
                    "available": False,
                    "reason": "non_finite_probability_matrix",
                }
            }
        probabilities = probabilities[valid_rows]
        true_values = true_values[valid_rows]
        probabilities = np.clip(probabilities / probabilities.sum(axis=1, keepdims=True), 0.0, 1.0)
        n_classes = int(probabilities.shape[1])
        labels = list(class_labels or range(n_classes))
        if len(labels) != n_classes:
            labels = list(range(n_classes))
        label_to_index = {int(label): idx for idx, label in enumerate(labels)}
        target = np.zeros_like(probabilities, dtype=float)
        known = np.array([int(label) in label_to_index for label in true_values], dtype=bool)
        if not bool(known.any()):
            return {
                "calibration": {
                    "available": False,
                    "reason": "labels_not_aligned_to_probability_columns",
                }
            }
        probabilities = probabilities[known]
        true_values = true_values[known]
        target = np.zeros_like(probabilities, dtype=float)
        for row_idx, label in enumerate(true_values):
            target[row_idx, label_to_index[int(label)]] = 1.0
        predicted_index = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        correctness = (predicted_index == np.array([label_to_index[int(label)] for label in true_values], dtype=int)).astype(float)
        bin_count = max(1, int(bins or 10))
        edges = np.linspace(0.0, 1.0, bin_count + 1)
        ece = 0.0
        populated_bins = 0
        for idx in range(bin_count):
            lower = edges[idx]
            upper = edges[idx + 1]
            if idx == bin_count - 1:
                mask = (confidence >= lower) & (confidence <= upper)
            else:
                mask = (confidence >= lower) & (confidence < upper)
            if not bool(mask.any()):
                continue
            populated_bins += 1
            bin_accuracy = float(correctness[mask].mean())
            bin_confidence = float(confidence[mask].mean())
            ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
        brier = float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))
        return {
            "calibration": {
                "available": True,
                "rows_valid": int(len(true_values)),
                "class_labels": [int(label) for label in labels],
                "method": "sigmoid_prefit",
                "expected_calibration_error": BaseTrainer.safe_float(ece),
                "brier_score": BaseTrainer.safe_float(brier),
                "mean_confidence": BaseTrainer.safe_float(float(confidence.mean())),
                "realized_accuracy_at_confidence": BaseTrainer.safe_float(float(correctness.mean())),
                "bins": int(bin_count),
                "populated_bins": int(populated_bins),
            }
        }

    def _promotion_cfg(self):
        """Return the shared training promotion policy configuration when available."""
        training_cfg = getattr(getattr(self.paths, "config", None), "training", None)
        return getattr(training_cfg, "promotion", None)

    @staticmethod
    def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
        """Extract a numeric metric value when present and finite."""
        return BaseTrainer.safe_float((metrics or {}).get(key))

    def _promotion_metric_keys(self, metrics: dict[str, Any]) -> tuple[str, ...]:
        """Select the appropriate promotion metrics for the artifact under evaluation."""
        metrics = dict(metrics or {})
        if any(key in metrics for key in ("roc_auc_ovr", "precision_macro", "balanced_accuracy")):
            return ("roc_auc_ovr", "precision_macro", "balanced_accuracy")
        return ("roc_auc", "precision", "balanced_accuracy")

    def _promotion_score(self, metrics: dict[str, Any]) -> float | None:
        """Compute a compact promotion score from the shared metric set."""
        values = [
            self._metric_value(metrics, key)
            for key in self._promotion_metric_keys(metrics)
        ]
        filtered = [value for value in values if value is not None]
        if not filtered:
            return None
        return self.safe_float(sum(filtered) / float(len(filtered)))


    def _load_recent_paper_trade_rows(self, symbol: str, *, lookback: int) -> list[dict[str, Any]]:
        """Load recent authoritative paper-execution closes for promotion gates.

        Paper promotion evidence is sourced only from the portfolio snapshot
        written by the paper execution authority. Historical backtest databases
        and CSV ledgers are deliberately excluded from this contract.
        """
        target = str(symbol or "").upper()
        limit = max(1, int(lookback or 1))
        rows: list[dict[str, Any]] = []

        portfolio_path = getattr(self.paths, "portfolio_state_path", None)
        try:
            if portfolio_path and Path(portfolio_path).exists():
                payload = json.loads(Path(portfolio_path).read_text(encoding="utf-8")) or {}
                closed = payload.get("closed_trades", []) if isinstance(payload, dict) else []
                if isinstance(closed, list):
                    for item in closed:
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("symbol") or "").upper() != target:
                            continue
                        row = dict(item)
                        row["outcome_source"] = "paper_execution_ledger"
                        rows.append(row)
        except Exception:
            self.logger.exception("failed to read authoritative paper execution trades for promotion gate symbol=%s", target)

        def _row_ts(row: dict[str, Any]) -> str:
            return str(row.get("closed_at") or row.get("exit_time") or row.get("created_at") or "")

        rows.sort(key=_row_ts, reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (str(row.get("symbol") or target), str(row.get("side") or ""), str(row.get("closed_at") or row.get("exit_time") or row.get("created_at") or len(deduped)))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= limit:
                break
        return deduped

    def _paper_expectancy_snapshot(self, symbol: str, *, lookback: int) -> dict[str, Any]:
        rows = self._load_recent_paper_trade_rows(symbol, lookback=lookback)
        pnls: list[float] = []
        wins = 0
        for row in rows:
            value = self.safe_float(row.get("pnl_usd"))
            if value is None:
                continue
            pnls.append(float(value))
            if value > 0.0:
                wins += 1
        total = float(sum(pnls)) if pnls else 0.0
        count = len(pnls)
        expectancy = total / float(count) if count else 0.0
        win_rate = float(wins) / float(count) if count else 0.0
        running = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in reversed(pnls):
            running += pnl
            peak = max(peak, running)
            max_drawdown = min(max_drawdown, running - peak)
        return {
            "available": bool(count),
            "trade_count": count,
            "wins": wins,
            "losses": sum(1 for pnl in pnls if pnl < 0.0),
            "total_pnl_usd": self.safe_float(total),
            "expectancy_usd": self.safe_float(expectancy),
            "win_rate": self.safe_float(win_rate),
            "max_drawdown_usd": self.safe_float(max_drawdown),
            "lookback_trades": int(max(1, lookback)),
            "outcome_source": "paper_execution_ledger",
            "authoritative_for_paper_promotion": True,
        }

    def _promotion_floor_reasons(self, metrics: dict[str, Any], *, model_role: str | None = None, symbol: str | None = None) -> list[str]:
        """Evaluate shared minimum-quality floors for promotion eligibility."""
        cfg = self._promotion_cfg()
        if cfg is None:
            return []

        reasons: list[str] = []
        label_validation = (metrics or {}).get("label_validation")
        if isinstance(label_validation, dict) and label_validation.get("promotion_eligible") is False:
            for reason in label_validation.get("reasons", []) or ["invalid_label_support"]:
                reasons.append(f"label_validation:{reason}")
        fold_count = int((metrics or {}).get("fold_count") or 0)
        if fold_count < int(getattr(cfg, "min_fold_count", 1)):
            reasons.append(f"insufficient_fold_count:{fold_count}")

        roc_auc_ovr = self._metric_value(metrics, "roc_auc_ovr")
        precision_macro = self._metric_value(metrics, "precision_macro")
        balanced_accuracy = self._metric_value(metrics, "balanced_accuracy")
        roc_auc = self._metric_value(metrics, "roc_auc")
        precision = self._metric_value(metrics, "precision")

        if roc_auc_ovr is not None and roc_auc_ovr < float(getattr(cfg, "min_roc_auc_ovr", 0.0)):
            reasons.append("below_min_roc_auc_ovr")
        if precision_macro is not None and precision_macro < float(getattr(cfg, "min_precision_macro", 0.0)):
            reasons.append("below_min_precision_macro")
        if balanced_accuracy is not None and balanced_accuracy < float(getattr(cfg, "min_balanced_accuracy", 0.0)):
            reasons.append("below_min_balanced_accuracy")
        if roc_auc is not None and roc_auc < float(getattr(cfg, "min_roc_auc", 0.0)):
            reasons.append("below_min_roc_auc")
        if precision is not None and precision < float(getattr(cfg, "min_precision", 0.0)):
            reasons.append("below_min_precision")

        if bool(getattr(cfg, "require_paper_expectancy", False)) and symbol:
            paper = self._paper_expectancy_snapshot(symbol, lookback=int(getattr(cfg, "paper_lookback_trades", 25) or 25))
            metrics["paper_expectancy"] = paper
            trade_count = int(paper.get("trade_count") or 0)
            min_trades = int(getattr(cfg, "min_paper_trades", 0) or 0)
            evidence_promotion_without_trades_requested = bool(
                getattr(cfg, "allow_initial_promotion_without_paper_trades", False)
            )
            paper_bootstrap_requested = bool(
                getattr(cfg, "allow_initial_paper_bootstrap_execution", False)
            )
            paper_reason_count_before = len(reasons)
            paper_governance: dict[str, Any] = {
                "status": "paper_evidence_sufficient" if trade_count >= min_trades else "insufficient_paper_evidence",
                "trade_count": trade_count,
                "min_paper_trades": min_trades,
                "evidence_sufficient": bool(trade_count >= min_trades),
                "paper_gate_passed": False,
                "bootstrap_requested": paper_bootstrap_requested,
                "paper_bootstrap_only": False,
                "paper_execution_authorized": False,
                "execution_authorized": False,
                "live_eligible": False,
                "evidence_promotion_without_trades_requested": evidence_promotion_without_trades_requested,
            }

            # Evidence-backed incumbent promotion remains gated by realized paper
            # results. A separate explicitly configured initial paper-bootstrap
            # path is allowed to break the otherwise circular dependency:
            # no active model -> no paper trades -> no promotable model. The
            # bootstrap path is finalized only after we confirm there is no
            # incumbent and it never grants live authority.
            if trade_count < min_trades:
                if paper_bootstrap_requested and trade_count == 0:
                    paper_governance.update(
                        {
                            "status": "paper_bootstrap_pending_initial_activation",
                            "paper_bootstrap_only": True,
                        }
                    )
                    metrics.setdefault("promotion_deferrals", []).append(
                        f"paper_bootstrap_requires_initial_slot:{trade_count}/{min_trades}"
                    )
                elif evidence_promotion_without_trades_requested and trade_count == 0:
                    # Preserve the explicit Step 4 governance contract: the old
                    # bypass flag never grants execution authority or live status.
                    paper_governance.update(
                        {
                            "status": "paper_bootstrap_not_execution_ready",
                            "paper_bootstrap_only": True,
                            "paper_execution_authorized": False,
                            "execution_authorized": False,
                            "live_eligible": False,
                        }
                    )
                    metrics.setdefault("promotion_deferrals", []).append(
                        f"paper_bootstrap_not_execution_ready:{trade_count}/{min_trades}"
                    )
                    reasons.append("paper_bootstrap_not_authorized_for_execution")
                else:
                    reasons.append(f"insufficient_paper_trades:{trade_count}")
            else:
                expectancy = self._metric_value(paper, "expectancy_usd")
                win_rate = self._metric_value(paper, "win_rate")
                drawdown = self._metric_value(paper, "max_drawdown_usd")
                if expectancy is not None and expectancy < float(getattr(cfg, "min_paper_expectancy_usd", 0.0) or 0.0):
                    reasons.append("below_min_paper_expectancy")
                if win_rate is not None and win_rate < float(getattr(cfg, "min_paper_win_rate", 0.0) or 0.0):
                    reasons.append("below_min_paper_win_rate")
                if drawdown is not None and drawdown < float(getattr(cfg, "max_paper_drawdown_usd", -1e18) or -1e18):
                    reasons.append("paper_drawdown_exceeded")
                if len(reasons) == paper_reason_count_before:
                    paper_governance["paper_gate_passed"] = True
                    paper_governance["live_eligible"] = True
                else:
                    paper_governance["status"] = "paper_evidence_rejected"
            metrics["paper_evidence_governance"] = paper_governance

        if model_role in (None, "direction"):
            directional_quality = (metrics or {}).get("directional_quality")
            if not isinstance(directional_quality, dict) or directional_quality.get("available") is False:
                if bool(getattr(cfg, "require_directional_quality", True)):
                    reasons.append("directional_quality_missing")
            else:
                directional_checks = (
                    ("long_precision", "min_long_precision", "below_min_long_precision"),
                    ("short_precision", "min_short_precision", "below_min_short_precision"),
                    ("long_recall", "min_long_recall", "below_min_long_recall"),
                    ("short_recall", "min_short_recall", "below_min_short_recall"),
                )
                for metric_key, cfg_key, reason in directional_checks:
                    value = self._metric_value(directional_quality, metric_key)
                    threshold = float(getattr(cfg, cfg_key, 0.0) or 0.0)
                    if value is not None and value < threshold:
                        reasons.append(reason)
                hold_fraction = self._metric_value(directional_quality, "hold_prediction_fraction")
                max_hold = float(getattr(cfg, "max_hold_prediction_fraction", 1.0) or 1.0)
                if hold_fraction is not None and hold_fraction > max_hold:
                    reasons.append("hold_prediction_dominates")
                directional_fraction = self._metric_value(directional_quality, "directional_prediction_fraction")
                min_directional = float(getattr(cfg, "min_directional_prediction_fraction", 0.0) or 0.0)
                if directional_fraction is not None and directional_fraction < min_directional:
                    reasons.append("directional_prediction_too_low")

        calibration = (metrics or {}).get("calibration")
        if bool(getattr(cfg, "require_calibration_quality", False)):
            if not isinstance(calibration, dict) or calibration.get("available") is False:
                reasons.append("calibration_quality_missing")
            else:
                ece = self._metric_value(calibration, "expected_calibration_error")
                brier = self._metric_value(calibration, "brier_score")
                max_ece = float(getattr(cfg, "max_expected_calibration_error", 1.0) or 1.0)
                max_brier = float(getattr(cfg, "max_brier_score", 1.0) or 1.0)
                if ece is not None and ece > max_ece:
                    reasons.append("calibration_ece_exceeded")
                if brier is not None and brier > max_brier:
                    reasons.append("calibration_brier_exceeded")

        return reasons

    def _promotion_comparison_reasons(self, candidate_metrics: dict[str, Any], incumbent_metrics: dict[str, Any]) -> list[str]:
        """Compare candidate metrics against the incumbent under the shared policy."""
        cfg = self._promotion_cfg()
        if cfg is None:
            return []
        reasons: list[str] = []
        candidate_score = self._promotion_score(candidate_metrics)
        incumbent_score = self._promotion_score(incumbent_metrics)
        min_score_improvement = float(getattr(cfg, "min_score_improvement", 0.0))
        min_metric_improvement = float(getattr(cfg, "min_metric_improvement", 0.0))
        max_metric_regression = float(getattr(cfg, "max_metric_regression", 0.0))

        if candidate_score is None:
            reasons.append("candidate_score_missing")
        if incumbent_score is None:
            reasons.append("incumbent_score_missing")
        if reasons:
            return reasons

        if candidate_score < (incumbent_score + min_score_improvement):
            reasons.append("candidate_score_not_improved")

        metric_improved = False
        for key in self._promotion_metric_keys(candidate_metrics):
            candidate_value = self._metric_value(candidate_metrics, key)
            incumbent_value = self._metric_value(incumbent_metrics, key)
            if candidate_value is None or incumbent_value is None:
                continue
            if candidate_value < (incumbent_value - max_metric_regression):
                reasons.append(f"metric_regressed:{key}")
            if candidate_value >= (incumbent_value + min_metric_improvement):
                metric_improved = True

        if not metric_improved:
            reasons.append("no_metric_improvement")
        return reasons

    def evaluate_candidate_promotion(
        self,
        *,
        symbol: str,
        interval: str,
        metrics: dict[str, Any],
        manifest_path,
        model_role: str | None = None,
        promote_requested: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a candidate artifact against the shared promotion policy."""
        cfg = self._promotion_cfg()
        state = {
            "attempted": False,
            "approved": False,
            "promoted": False,
            "manifest_path": str(manifest_path),
            "candidate_id": None,
            "model_role": model_role,
            "has_incumbent": False,
            "candidate_score": self._promotion_score(metrics),
            "incumbent_score": None,
            "reasons": [],
            "paper_evidence_status": "not_evaluated",
            "paper_bootstrap_only": False,
            "paper_execution_only": False,
            "deployment_stage": "candidate",
            "live_eligible": False,
            "paper_evidence_sufficient": None,
        }

        if not promote_requested:
            state["reasons"].append("promotion_not_requested")
            return state
        if cfg is not None and not bool(getattr(cfg, "enabled", True)):
            state["reasons"].append("promotion_disabled_by_config")
            return state

        state["attempted"] = True
        state["reasons"].extend(self._promotion_floor_reasons(metrics, model_role=model_role, symbol=symbol))
        if isinstance(metrics, dict) and isinstance(metrics.get("paper_expectancy"), dict):
            state["paper_expectancy"] = dict(metrics.get("paper_expectancy") or {})
        paper_governance = (metrics or {}).get("paper_evidence_governance")
        if isinstance(paper_governance, dict):
            state["paper_evidence_governance"] = dict(paper_governance)
            state["paper_evidence_status"] = str(paper_governance.get("status") or "not_evaluated")
            state["paper_bootstrap_only"] = bool(paper_governance.get("paper_bootstrap_only", False))
            state["paper_evidence_sufficient"] = bool(paper_governance.get("evidence_sufficient", False))

        active_manifest = resolve_active_manifest_path(self.paths, self.model_type, symbol, interval, model_role=model_role)
        state["has_incumbent"] = active_manifest is not None
        if active_manifest is None:
            if cfg is not None and not bool(getattr(cfg, "allow_initial_promotion", True)):
                state["reasons"].append("initial_promotion_disabled")
        elif state["paper_bootstrap_only"]:
            # Bootstrap is an initial paper-evidence mechanism only. It must
            # never replace an incumbent without completed paper outcome gates.
            state["reasons"].append("paper_bootstrap_initial_slot_only")
        elif bool(getattr(cfg, "require_outperform_incumbent", True)):
            incumbent_metrics = load_manifest_metrics(self.paths, active_manifest)
            state["incumbent_score"] = self._promotion_score(incumbent_metrics)
            state["reasons"].extend(self._promotion_comparison_reasons(metrics, incumbent_metrics))

        state["approved"] = len(state["reasons"]) == 0
        if state["approved"] and state["paper_bootstrap_only"] and not state["has_incumbent"]:
            governance = dict(state.get("paper_evidence_governance") or {})
            governance.update(
                {
                    "status": "paper_bootstrap_execution_ready",
                    "paper_bootstrap_only": True,
                    "paper_execution_authorized": True,
                    "execution_authorized": True,
                    "live_eligible": False,
                }
            )
            state.update(
                {
                    "paper_evidence_governance": governance,
                    "paper_evidence_status": "paper_bootstrap_execution_ready",
                    "paper_execution_only": True,
                    "deployment_stage": "paper_bootstrap",
                    "live_eligible": False,
                }
            )
        elif state["approved"]:
            state["deployment_stage"] = "incumbent"
            state["live_eligible"] = True
        return state

    def save_training_bundle(
        self,
        *,
        symbol: str,
        interval: str,
        rows_used: int,
        feature_columns: list[str],
        artifacts: dict[str, Any],
        metrics: dict[str, Any],
        candidate_id: str,
        model_role: str | None = None,
        promote: bool = True,
        extra_manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a candidate bundle, then optionally promote it through the shared policy."""
        manifest_extra = dict(extra_manifest or {})
        explicit_feature_set = manifest_extra.pop("feature_set", None)
        recipe_version = resolve_feature_recipe_version(getattr(self.paths, "config", None))
        anchor_interval = None
        optional_context_fields: list[str] = []
        if isinstance(explicit_feature_set, dict):
            anchor_interval = explicit_feature_set.get("anchor_interval")
            optional_context_fields = [str(col) for col in (explicit_feature_set.get("optional_context_fields") or [])]
        if anchor_interval is None:
            anchor_interval = interval
        feature_set_payload = explicit_feature_set or build_feature_set_metadata(
            feature_columns,
            recipe_version=recipe_version,
            anchor_interval=anchor_interval,
            optional_context_fields=optional_context_fields,
        )
        if isinstance(feature_set_payload, dict):
            feature_set_payload = dict(feature_set_payload)
            feature_set_payload.setdefault("anchor_interval", anchor_interval)
            feature_set_payload.setdefault("optional_context_fields", list(optional_context_fields))

        compatible, reasons, feature_set = validate_feature_store_metadata(
            feature_set_payload,
            feature_columns=feature_columns,
            recipe_version=recipe_version,
            anchor_interval=anchor_interval,
            optional_context_fields=optional_context_fields,
        )
        if explicit_feature_set and not compatible:
            non_fatal_reasons = {"feature_signature_missing", "feature_signature_mismatch"}
            fatal_reasons = [str(reason) for reason in reasons if str(reason) not in non_fatal_reasons]
            if fatal_reasons:
                raise ValueError("incompatible_feature_set_metadata:" + ",".join(fatal_reasons))
        validation = dict(manifest_extra.pop("validation", {}))
        training = dict(manifest_extra.pop("training", {}))
        lifecycle = dict(manifest_extra.pop("lifecycle", {}))
        lifecycle.update({"stage": "candidate", "candidate_id": candidate_id})

        save_json(artifacts["feature_columns"], {"feature_columns": list(feature_columns)})
        manifest = write_manifest(
            self.paths,
            model_type=self.model_type,
            symbol=symbol,
            interval=interval,
            rows_used=rows_used,
            feature_count=len(feature_columns),
            artifacts=artifacts,
            metrics=metrics,
            extra={
                "model_role": model_role,
                "feature_set": feature_set,
                "validation": validation,
                "training": training,
                "lifecycle": lifecycle,
                **manifest_extra,
            },
        )

        promotion_state = self.evaluate_candidate_promotion(
            symbol=symbol,
            interval=interval,
            metrics=metrics,
            manifest_path=artifacts["manifest"],
            model_role=model_role,
            promote_requested=promote,
        )
        promotion_state["candidate_id"] = str(candidate_id)

        promoted = None
        if promotion_state["approved"]:
            promoted = promote_candidate_manifest(
                self.paths,
                model_type=self.model_type,
                symbol=symbol,
                interval=interval,
                manifest_path=artifacts["manifest"],
                candidate_id=candidate_id,
                model_role=model_role,
                metrics=metrics,
                deployment_stage=str(promotion_state.get("deployment_stage") or "incumbent"),
                paper_execution_only=bool(promotion_state.get("paper_execution_only", False)),
                paper_evidence_status=str(promotion_state.get("paper_evidence_status") or "not_evaluated"),
            )
            promotion_state.update(
                {
                    "promoted": True,
                    "pointer_path": str(incumbent_pointer_path(self.paths, self.model_type, symbol, interval, model_role=model_role)),
                    "pointer": dict(promoted or {}),
                }
            )

        prune_results: list[dict[str, Any]] = []
        try:
            prune_results = prune_related_candidate_artifacts(
                self.paths,
                model_type=self.model_type,
                symbol=symbol,
                interval=interval,
                model_role=model_role,
            )
        except Exception:
            self.logger.exception(
                "artifact lifecycle pruning failed model=%s symbol=%s interval=%s role=%s",
                self.model_type,
                symbol,
                interval,
                model_role,
            )

        self.logger.info(
            "saved training bundle symbol=%s interval=%s rows_used=%s feature_count=%s candidate_id=%s role=%s promotion_attempted=%s promoted=%s reasons=%s prune_deleted=%s",
            symbol,
            interval,
            rows_used,
            len(feature_columns),
            candidate_id,
            model_role,
            promotion_state["attempted"],
            bool(promoted),
            promotion_state.get("reasons", []),
            sum(len(result.get("deleted", [])) for result in prune_results if isinstance(result, dict)),
        )
        return {"manifest": manifest, "promoted": promoted, "promotion": promotion_state, "artifact_pruning": prune_results}

    def ok_result(
        self,
        *,
        symbol: str,
        interval: str,
        rows_used: int,
        artifacts: dict[str, Any],
        metrics: dict[str, Any],
        model_role: str | None = None,
    ) -> TrainerResult:
        """Build a success result payload."""
        self.logger.info(
            "training success model=%s symbol=%s interval=%s rows_used=%s role=%s",
            self.model_type,
            symbol,
            interval,
            rows_used,
            model_role,
        )
        return TrainerResult(
            ok=True,
            model_type=self.model_type,
            symbol=symbol,
            interval=interval,
            rows_used=int(rows_used),
            artifacts={k: str(v) for k, v in artifacts.items()},
            metrics=dict(metrics or {}),
            error=None,
            model_role=model_role,
        )

    def fail_result(self, *, symbol: str, interval: str, error: str, rows_used: int = 0, model_role: str | None = None) -> TrainerResult:
        """Build a failure result payload."""
        self.logger.error(
            "training failure model=%s symbol=%s interval=%s rows_used=%s role=%s error=%s",
            self.model_type,
            symbol,
            interval,
            rows_used,
            model_role,
            error,
        )
        return TrainerResult(
            ok=False,
            model_type=self.model_type,
            symbol=symbol,
            interval=interval,
            rows_used=int(rows_used),
            artifacts={},
            metrics={},
            error=str(error),
            model_role=model_role,
        )

    @staticmethod
    def safe_float(value: Any) -> float | None:
        """Safely coerce an arbitrary value to float."""
        try:
            if value is None:
                return None
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return None
            return value
        except Exception:
            return None

    @abstractmethod
    def train(self, symbol: str, interval: str) -> TrainerResult:
        """Run the model-specific training pipeline."""
        raise NotImplementedError
