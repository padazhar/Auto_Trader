from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.artifact_contract import load_json, save_json, validate_artifact_bundle
from core.path_manager import PathManager


@dataclass(slots=True)
class HMMArtifactEvidence:
    symbol: str
    interval: str
    candidate_id: str
    manifest_path: str
    valid: bool
    validation_reason: str
    shadow_only: bool
    decision_influence_allowed: bool
    mandatory_paper_gate_enabled: bool
    live_influence_enabled: bool
    observation_schema_version: str | None = None
    n_states: int | None = None


@dataclass(slots=True)
class HMMShadowDecisionEvidence:
    total_decisions: int = 0
    decisions_with_hmm: int = 0
    available_decisions: int = 0
    high_confidence_decisions: int = 0
    actionable_shadow_decisions: int = 0
    counterfactual_entry_decisions: int = 0
    decision_influence_violations: int = 0
    live_or_mandatory_violations: int = 0
    missing_or_unavailable_decisions: int = 0
    confidence_min: float | None = None
    confidence_mean: float | None = None
    confidence_max: float | None = None
    state_counts: dict[str, int] = field(default_factory=dict)
    symbol_counts: dict[str, int] = field(default_factory=dict)
    interval_counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    counterfactual_action_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class HMMPaperLedgerEvidence:
    present: bool
    authoritative: bool
    authority_mode: str
    reconciliation_status: str
    state_domain: str
    state_owner: str
    closed_trade_count: int
    open_position_count: int
    realized_pnl_usd: float
    reason: str


@dataclass(slots=True)
class HMMReadinessEvidence:
    monitoring_state_present: bool
    execution_eligible_count: int
    gbm_ready_symbol_count: int
    rf_ready_symbol_count: int
    blockers: list[str]
    anchor_blockers: dict[str, int]
    reason: str


@dataclass(slots=True)
class HMMShadowEvaluationReport:
    schema_version: str
    evaluation_type: str
    approval_recommendation: str
    ready_for_mandatory_paper_gate_review: bool
    activation_performed: bool
    config_mutation_performed: bool
    pointer_mutation_performed: bool
    live_candidate_validation_allowed: bool
    hmm_decision_effect_allowed: bool
    rf_anchor_preserved: bool
    ev_fusion_preserved: bool
    evaluated_at_source: str
    thresholds: dict[str, Any]
    config_state: dict[str, Any]
    artifact_summary: dict[str, Any]
    decision_summary: dict[str, Any]
    paper_ledger_summary: dict[str, Any]
    readiness_summary: dict[str, Any]
    reasons: list[str]
    artifacts: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows


def _candidate_id_from_manifest(path: Path, manifest: dict[str, Any]) -> str:
    lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
    candidate = lifecycle.get("candidate_id") or manifest.get("candidate_id")
    if candidate:
        return str(candidate)
    try:
        parts = list(path.parts)
        idx = parts.index("candidates")
        return str(parts[idx + 1])
    except Exception:
        return ""


def _manifest_symbol_interval(path: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    symbol = str(manifest.get("symbol") or "").upper()
    interval = str(manifest.get("interval") or "")
    if symbol and interval:
        return symbol, interval
    try:
        # .../hmm_regime_models/<symbol>/<interval>/candidates/<candidate>/manifest.json
        parts = list(path.parts)
        idx = parts.index("hmm_regime_models")
        symbol = str(parts[idx + 1]).upper()
        interval = str(parts[idx + 2])
    except Exception:
        pass
    return symbol, interval


def _discover_hmm_artifacts(paths: PathManager) -> list[HMMArtifactEvidence]:
    manifests = sorted(paths.hmm_models_dir.glob("*/*/candidates/*/manifest.json")) if paths.hmm_models_dir.exists() else []
    out: list[HMMArtifactEvidence] = []
    for manifest_path in manifests:
        payload = load_json(manifest_path) or {}
        symbol, interval = _manifest_symbol_interval(manifest_path, payload)
        contract = payload.get("hmm_contract") if isinstance(payload.get("hmm_contract"), dict) else {}
        valid, reason = validate_artifact_bundle(paths, manifest_path)
        out.append(
            HMMArtifactEvidence(
                symbol=symbol,
                interval=interval,
                candidate_id=_candidate_id_from_manifest(manifest_path, payload),
                manifest_path=manifest_path.relative_to(paths.project_root).as_posix() if manifest_path.is_relative_to(paths.project_root) else str(manifest_path),
                valid=bool(valid),
                validation_reason=str(reason),
                shadow_only=bool(contract.get("shadow_only", False)),
                decision_influence_allowed=bool(contract.get("decision_influence_allowed", False)),
                mandatory_paper_gate_enabled=bool(contract.get("mandatory_paper_gate_enabled", False)),
                live_influence_enabled=bool(contract.get("live_influence_enabled", False)),
                observation_schema_version=str(contract.get("schema_version") or payload.get("observation_schema_version") or "") or None,
                n_states=int(contract.get("n_states")) if str(contract.get("n_states", "")).isdigit() else None,
            )
        )
    return out


def _extract_hmm_shadow_component(row: dict[str, Any]) -> dict[str, Any] | None:
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    direct = components.get("hmm")
    if isinstance(direct, dict):
        return direct
    shadow_candidate = components.get("shadow_candidate")
    if isinstance(shadow_candidate, dict):
        shadow_components = shadow_candidate.get("components")
        if isinstance(shadow_components, dict):
            nested = shadow_components.get("hmm")
            if isinstance(nested, dict):
                return nested
    return None


def _counterfactual_action_from_row(row: dict[str, Any]) -> str:
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    shadow_candidate = components.get("shadow_candidate")
    if isinstance(shadow_candidate, dict):
        action = str(shadow_candidate.get("effective_action") or shadow_candidate.get("action") or "HOLD").upper()
        return action or "HOLD"
    return str(row.get("effective_action") or row.get("action") or "HOLD").upper()


def _decision_evidence(
    paths: PathManager,
    *,
    min_confidence: float,
    allow_step12_paper_gate_effect: bool = False,
) -> HMMShadowDecisionEvidence:
    confidence_values: list[float] = []
    state_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    interval_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    counterfactual_action_counts: Counter[str] = Counter()
    result = HMMShadowDecisionEvidence()
    for row in _iter_jsonl(paths.decision_journal_path):
        result.total_decisions += 1
        hmm = _extract_hmm_shadow_component(row)
        if hmm is None:
            continue
        result.decisions_with_hmm += 1
        symbol = str(row.get("symbol") or "").upper()
        interval = str(row.get("interval") or "")
        if symbol:
            symbol_counts[symbol] += 1
        if interval:
            interval_counts[interval] += 1
        reason = str(hmm.get("reason") or "unknown")
        reason_counts[reason] += 1
        used = bool(hmm.get("used", False))
        effect = str(hmm.get("decision_effect") or "none")
        decision_influence_allowed = bool(hmm.get("decision_influence_allowed", False))
        mandatory_gate_enabled = bool(hmm.get("mandatory_gate_enabled", False))
        live_influence_enabled = bool(hmm.get("live_influence_enabled", False))
        runtime_mode = str(hmm.get("runtime_execution_mode") or "paper").strip().lower()
        step12_paper_gate_effect = (
            allow_step12_paper_gate_effect
            and mandatory_gate_enabled
            and not live_influence_enabled
            and runtime_mode == "paper"
            and effect == "paper_entry_gate"
        )
        if not step12_paper_gate_effect and (used or effect != "none" or decision_influence_allowed):
            result.decision_influence_violations += 1
        if live_influence_enabled or (mandatory_gate_enabled and not allow_step12_paper_gate_effect):
            result.live_or_mandatory_violations += 1
        available = bool(hmm.get("available", False))
        if not available:
            result.missing_or_unavailable_decisions += 1
            continue
        result.available_decisions += 1
        confidence = _safe_float(hmm.get("confidence"))
        if confidence is not None:
            confidence_values.append(confidence)
            if confidence >= float(min_confidence):
                result.high_confidence_decisions += 1
        state = hmm.get("state")
        if state is not None:
            state_counts[str(state)] += 1
        counterfactual_action = _counterfactual_action_from_row(row)
        counterfactual_action_counts[counterfactual_action] += 1
        if counterfactual_action != "HOLD":
            result.actionable_shadow_decisions += 1
        if counterfactual_action.startswith("ENTER_") or counterfactual_action.startswith("FLIP_"):
            result.counterfactual_entry_decisions += 1
    if confidence_values:
        result.confidence_min = min(confidence_values)
        result.confidence_mean = sum(confidence_values) / len(confidence_values)
        result.confidence_max = max(confidence_values)
    result.state_counts = dict(sorted(state_counts.items()))
    result.symbol_counts = dict(sorted(symbol_counts.items()))
    result.interval_counts = dict(sorted(interval_counts.items()))
    result.reason_counts = dict(sorted(reason_counts.items()))
    result.counterfactual_action_counts = dict(sorted(counterfactual_action_counts.items()))
    return result





def _paper_ledger_evidence(paths: PathManager) -> HMMPaperLedgerEvidence:
    payload = load_json(paths.portfolio_state_path) or {}
    if not isinstance(payload, dict) or not payload:
        return HMMPaperLedgerEvidence(
            present=False,
            authoritative=False,
            authority_mode="",
            reconciliation_status="",
            state_domain="",
            state_owner="",
            closed_trade_count=0,
            open_position_count=0,
            realized_pnl_usd=0.0,
            reason="portfolio_state_missing",
        )
    closed = payload.get("closed_trades") if isinstance(payload.get("closed_trades"), list) else []
    positions = payload.get("positions") if isinstance(payload.get("positions"), dict) else {}
    authority_mode = str(payload.get("authority_mode") or "")
    reconciliation_status = str(payload.get("reconciliation_status") or "")
    state_domain = str(payload.get("state_domain") or "")
    state_owner = str(payload.get("state_owner") or "")
    authoritative = (
        state_domain == "execution_portfolio"
        and state_owner in {"live_trading_controller", "paper_trading_controller"}
        and (authority_mode.startswith("paper") or reconciliation_status.startswith("paper"))
    )
    reason = "authoritative_paper_execution_ledger" if authoritative else "portfolio_state_not_authoritative_paper_execution_ledger"
    return HMMPaperLedgerEvidence(
        present=True,
        authoritative=bool(authoritative),
        authority_mode=authority_mode,
        reconciliation_status=reconciliation_status,
        state_domain=state_domain,
        state_owner=state_owner,
        closed_trade_count=len(closed),
        open_position_count=len(positions),
        realized_pnl_usd=float(_safe_float(payload.get("realized_pnl_usd")) or 0.0),
        reason=reason,
    )


def _readiness_evidence(paths: PathManager) -> HMMReadinessEvidence:
    payload = load_json(paths.monitoring_state_path) or {}
    if not isinstance(payload, dict) or not payload:
        return HMMReadinessEvidence(
            monitoring_state_present=False,
            execution_eligible_count=0,
            gbm_ready_symbol_count=0,
            rf_ready_symbol_count=0,
            blockers=[],
            anchor_blockers={},
            reason="monitoring_readiness_evidence_missing",
        )
    readiness = payload.get("trading_readiness") if isinstance(payload.get("trading_readiness"), dict) else {}
    rows = readiness.get("execution_readiness_matrix") if isinstance(readiness.get("execution_readiness_matrix"), list) else []
    gbm_ready = 0
    rf_ready = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        models = row.get("models") if isinstance(row.get("models"), dict) else {}
        gbm = models.get("gbm") if isinstance(models.get("gbm"), dict) else {}
        rf = models.get("rf") if isinstance(models.get("rf"), dict) else {}
        if bool(gbm.get("ready") or gbm.get("execution_ready")):
            gbm_ready += 1
        if bool(rf.get("ready") or rf.get("execution_ready")):
            rf_ready += 1
    blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
    anchor_blockers = readiness.get("anchor_blockers") if isinstance(readiness.get("anchor_blockers"), dict) else {}
    execution_eligible = int(readiness.get("execution_eligible_count") or readiness.get("ready_count") or 0)
    reason = "accepted_anchor_readiness_present" if execution_eligible > 0 and gbm_ready > 0 else "accepted_anchor_readiness_missing_or_blocked"
    return HMMReadinessEvidence(
        monitoring_state_present=True,
        execution_eligible_count=execution_eligible,
        gbm_ready_symbol_count=gbm_ready,
        rf_ready_symbol_count=rf_ready,
        blockers=[str(item) for item in blockers],
        anchor_blockers={str(k): int(v) for k, v in anchor_blockers.items() if str(v).isdigit() or isinstance(v, int)},
        reason=reason,
    )


def _artifact_key_counts(artifacts: list[HMMArtifactEvidence]) -> tuple[Counter[str], Counter[str], set[tuple[str, str]]]:
    symbol_counts: Counter[str] = Counter()
    interval_counts: Counter[str] = Counter()
    keys: set[tuple[str, str]] = set()
    for artifact in artifacts:
        symbol = str(artifact.symbol or "").upper()
        interval = str(artifact.interval or "")
        if symbol:
            symbol_counts[symbol] += 1
        if interval:
            interval_counts[interval] += 1
        if symbol and interval:
            keys.add((symbol, interval))
    return symbol_counts, interval_counts, keys


def _decision_keys_from_summary(decisions: HMMShadowDecisionEvidence) -> set[tuple[str, str]]:
    symbols = {str(symbol).upper() for symbol, count in (decisions.symbol_counts or {}).items() if count}
    intervals = {str(interval) for interval, count in (decisions.interval_counts or {}).items() if count}
    if len(intervals) == 1:
        interval = next(iter(intervals))
        return {(symbol, interval) for symbol in symbols if symbol and interval}
    return set()


def _missing_decision_artifact_keys(
    decisions: HMMShadowDecisionEvidence,
    shadow_valid_artifacts: list[HMMArtifactEvidence],
) -> list[str]:
    _symbol_counts, _interval_counts, artifact_keys = _artifact_key_counts(shadow_valid_artifacts)
    missing: list[str] = []
    for symbol, interval in sorted(_decision_keys_from_summary(decisions)):
        if (symbol, interval) not in artifact_keys:
            missing.append(f"{symbol}:{interval}")
    return missing


def run_hmm_shadow_evaluation(
    project_root: str | Path = ".",
    *,
    write_report: bool = False,
    min_samples: int = 100,
    min_available_ratio: float = 0.50,
    min_high_confidence_ratio: float = 0.35,
    min_paper_closed_trades: int = 5,
    min_execution_eligible_symbols: int = 1,
    min_counterfactual_entries: int = 1,
) -> HMMShadowEvaluationReport:
    paths = PathManager.from_project_root(Path(project_root))
    cfg = paths.config
    training_hmm = getattr(getattr(cfg, "training", None), "hmm", None)
    inference_hmm = getattr(getattr(cfg, "inference", None), "hmm", None)
    min_confidence = float(getattr(inference_hmm, "min_posterior_confidence", 0.60) or 0.60)
    runtime_mode = str(getattr(getattr(cfg, "runtime", None), "execution_mode", "paper") or "paper").strip().lower()
    training_mandatory_enabled = bool(getattr(training_hmm, "mandatory_paper_gate_enabled", False))
    inference_mandatory_enabled = bool(getattr(inference_hmm, "mandatory_paper_gate_enabled", False))
    training_live_influence_enabled = bool(getattr(training_hmm, "live_influence_enabled", False))
    inference_live_influence_enabled = bool(getattr(inference_hmm, "live_influence_enabled", False))
    approved_step12_paper_gate_active = (
        runtime_mode == "paper"
        and inference_mandatory_enabled
        and not training_mandatory_enabled
        and not training_live_influence_enabled
        and not inference_live_influence_enabled
    )
    artifacts = _discover_hmm_artifacts(paths)
    decisions = _decision_evidence(
        paths,
        min_confidence=min_confidence,
        allow_step12_paper_gate_effect=approved_step12_paper_gate_active,
    )
    paper_ledger = _paper_ledger_evidence(paths)
    readiness = _readiness_evidence(paths)
    valid_artifacts = [a for a in artifacts if a.valid]
    shadow_valid_artifacts = [
        a for a in valid_artifacts
        if a.shadow_only and not a.decision_influence_allowed and not a.mandatory_paper_gate_enabled and not a.live_influence_enabled
    ]
    artifact_symbol_counts, artifact_interval_counts, _artifact_keys = _artifact_key_counts(shadow_valid_artifacts)
    missing_decision_artifact_keys = _missing_decision_artifact_keys(decisions, shadow_valid_artifacts)
    reasons: list[str] = []
    if not bool(getattr(training_hmm, "enabled", False)) or not bool(getattr(training_hmm, "candidate_training_enabled", False)):
        reasons.append("hmm_candidate_training_not_enabled")
    if not bool(getattr(inference_hmm, "enabled", False)) or not bool(getattr(inference_hmm, "candidate_shadow_enabled", False)):
        reasons.append("hmm_shadow_inference_not_enabled")
    if training_mandatory_enabled or (inference_mandatory_enabled and not approved_step12_paper_gate_active):
        reasons.append("mandatory_paper_gate_already_enabled")
    if training_live_influence_enabled or inference_live_influence_enabled:
        reasons.append("hmm_live_influence_enabled_forbidden")
    if decisions.decisions_with_hmm < int(min_samples):
        reasons.append(f"insufficient_hmm_shadow_samples:{decisions.decisions_with_hmm}/{int(min_samples)}")
    available_ratio = (decisions.available_decisions / decisions.decisions_with_hmm) if decisions.decisions_with_hmm else 0.0
    high_confidence_ratio = (decisions.high_confidence_decisions / decisions.decisions_with_hmm) if decisions.decisions_with_hmm else 0.0
    if available_ratio < float(min_available_ratio):
        reasons.append(f"hmm_available_ratio_below_threshold:{available_ratio:.6f}<{float(min_available_ratio):.6f}")
    if high_confidence_ratio < float(min_high_confidence_ratio):
        reasons.append(f"hmm_high_confidence_ratio_below_threshold:{high_confidence_ratio:.6f}<{float(min_high_confidence_ratio):.6f}")
    if decisions.decision_influence_violations:
        reasons.append(f"hmm_decision_influence_violations:{decisions.decision_influence_violations}")
    if decisions.live_or_mandatory_violations:
        reasons.append(f"hmm_live_or_mandatory_violations:{decisions.live_or_mandatory_violations}")
    if decisions.counterfactual_entry_decisions < int(min_counterfactual_entries):
        reasons.append(f"insufficient_hmm_counterfactual_entry_support:{decisions.counterfactual_entry_decisions}/{int(min_counterfactual_entries)}")
    if not artifacts:
        reasons.append("hmm_candidate_artifacts_missing")
    elif not valid_artifacts:
        reasons.append("hmm_valid_candidate_artifacts_missing")
    elif not shadow_valid_artifacts:
        reasons.append("hmm_shadow_only_candidate_artifacts_missing")
    if decisions.decisions_with_hmm == 0 and shadow_valid_artifacts:
        reasons.append("hmm_shadow_telemetry_missing_from_decision_journal")
    if missing_decision_artifact_keys:
        reasons.append(f"hmm_valid_candidate_artifacts_missing_for_decision_keys:{len(missing_decision_artifact_keys)}")
    if not paper_ledger.present:
        reasons.append("authoritative_paper_ledger_missing")
    elif not paper_ledger.authoritative:
        reasons.append(f"paper_ledger_not_authoritative:{paper_ledger.reason}")
    if paper_ledger.closed_trade_count < int(min_paper_closed_trades):
        reasons.append(f"insufficient_authoritative_paper_closed_trades:{paper_ledger.closed_trade_count}/{int(min_paper_closed_trades)}")
    if not readiness.monitoring_state_present:
        reasons.append("readiness_evidence_missing")
    if readiness.execution_eligible_count < int(min_execution_eligible_symbols):
        reasons.append(f"insufficient_execution_eligible_symbols:{readiness.execution_eligible_count}/{int(min_execution_eligible_symbols)}")
    if readiness.gbm_ready_symbol_count < int(min_execution_eligible_symbols):
        reasons.append(f"accepted_gbm_estimator_missing_or_blocked:{readiness.gbm_ready_symbol_count}/{int(min_execution_eligible_symbols)}")

    ready = not reasons
    if ready and approved_step12_paper_gate_active:
        recommendation = "mandatory_paper_gate_active_validated"
    else:
        recommendation = "paper_gate_review_ready" if ready else "keep_hmm_shadow_only"
    thresholds = {
        "min_samples": int(min_samples),
        "min_available_ratio": float(min_available_ratio),
        "min_high_confidence_ratio": float(min_high_confidence_ratio),
        "min_posterior_confidence": min_confidence,
        "min_paper_closed_trades": int(min_paper_closed_trades),
        "min_execution_eligible_symbols": int(min_execution_eligible_symbols),
        "min_counterfactual_entries": int(min_counterfactual_entries),
    }
    report = HMMShadowEvaluationReport(
        schema_version="hmm_shadow_evaluation_v1",
        evaluation_type="paper_gate_post_activation_validation" if approved_step12_paper_gate_active else "evidence_only_no_activation",
        approval_recommendation=recommendation,
        ready_for_mandatory_paper_gate_review=bool(ready),
        activation_performed=bool(approved_step12_paper_gate_active),
        config_mutation_performed=False,
        pointer_mutation_performed=False,
        live_candidate_validation_allowed=False,
        hmm_decision_effect_allowed=bool(approved_step12_paper_gate_active),
        rf_anchor_preserved=True,
        ev_fusion_preserved=True,
        evaluated_at_source="decision_journal_and_hmm_candidate_manifests",
        thresholds=thresholds,
        config_state={
            "training_hmm_enabled": bool(getattr(training_hmm, "enabled", False)),
            "training_candidate_enabled": bool(getattr(training_hmm, "candidate_training_enabled", False)),
            "training_mandatory_paper_gate_enabled": training_mandatory_enabled,
            "training_live_influence_enabled": training_live_influence_enabled,
            "inference_hmm_enabled": bool(getattr(inference_hmm, "enabled", False)),
            "inference_candidate_shadow_enabled": bool(getattr(inference_hmm, "candidate_shadow_enabled", False)),
            "inference_mandatory_paper_gate_enabled": inference_mandatory_enabled,
            "inference_live_influence_enabled": inference_live_influence_enabled,
            "require_valid_artifact_for_entry": bool(getattr(inference_hmm, "require_valid_artifact_for_entry", False)),
            "runtime_execution_mode": runtime_mode,
            "approved_step12_paper_gate_active": approved_step12_paper_gate_active,
        },
        artifact_summary={
            "candidate_artifact_count": len(artifacts),
            "valid_candidate_artifact_count": len(valid_artifacts),
            "shadow_only_candidate_artifact_count": len(shadow_valid_artifacts),
            "shadow_only_artifact_symbol_counts": dict(sorted(artifact_symbol_counts.items())),
            "shadow_only_artifact_interval_counts": dict(sorted(artifact_interval_counts.items())),
            "missing_decision_artifact_keys": missing_decision_artifact_keys,
        },
        decision_summary=asdict(decisions) | {
            "available_ratio": available_ratio,
            "high_confidence_ratio": high_confidence_ratio,
        },
        paper_ledger_summary=asdict(paper_ledger),
        readiness_summary=asdict(readiness),
        reasons=reasons,
        artifacts=[asdict(a) for a in artifacts],
    )
    if write_report:
        save_json(paths.runtime_dir / "hmm_shadow_evaluation_report.json", report.to_dict())
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HMM shadow evidence without enabling the mandatory gate.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--min-available-ratio", type=float, default=0.50)
    parser.add_argument("--min-high-confidence-ratio", type=float, default=0.35)
    parser.add_argument("--min-paper-closed-trades", type=int, default=5)
    parser.add_argument("--min-execution-eligible-symbols", type=int, default=1)
    parser.add_argument("--min-counterfactual-entries", type=int, default=1)
    args = parser.parse_args()
    report = run_hmm_shadow_evaluation(
        args.project_root,
        write_report=args.write_report,
        min_samples=args.min_samples,
        min_available_ratio=args.min_available_ratio,
        min_high_confidence_ratio=args.min_high_confidence_ratio,
        min_paper_closed_trades=args.min_paper_closed_trades,
        min_execution_eligible_symbols=args.min_execution_eligible_symbols,
        min_counterfactual_entries=args.min_counterfactual_entries,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
