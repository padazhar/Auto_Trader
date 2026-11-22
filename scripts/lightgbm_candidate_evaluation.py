from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.artifact_contract import load_json, resolve_active_manifest_path, save_json, validate_artifact_bundle
from core.path_manager import PathManager

REPORT_SCHEMA_VERSION = "lightgbm_candidate_evaluation.v1"
REPORT_FILENAME = "lightgbm_candidate_evaluation_report.json"


@dataclass(slots=True)
class LightGbmCandidateEvaluationItem:
    """Single HistGBM-vs-LightGBM candidate comparison item."""

    symbol: str
    interval: str
    status: str
    paper_approval_recommended: bool
    activation_performed: bool
    baseline_manifest_path: str | None
    baseline_candidate_id: str | None
    baseline_estimator_family: str | None
    baseline_score: float | None
    candidate_manifest_path: str | None
    candidate_id: str | None
    candidate_estimator_family: str | None
    candidate_score: float | None
    score_delta: float | None
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LightGbmCandidateEvaluationReport:
    """Governed Step 9 LightGBM candidate evaluation report.

    The report is deliberately evidence-only: it never changes configuration,
    active pointers, manifests, promotion state, or live readiness.
    """

    project_root: str
    schema_version: str
    generated_at: str
    overall_status: str
    ready_for_lightgbm_paper_approval: bool
    activation_performed: bool
    current_configured_estimator: str
    baseline_family: str
    candidate_family: str
    items: list[LightGbmCandidateEvaluationItem]
    counts: dict[str, int]
    requirements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status,
            "ready_for_lightgbm_paper_approval": bool(self.ready_for_lightgbm_paper_approval),
            "activation_performed": bool(self.activation_performed),
            "current_configured_estimator": self.current_configured_estimator,
            "baseline_family": self.baseline_family,
            "candidate_family": self.candidate_family,
            "items": [asdict(item) for item in self.items],
            "counts": dict(self.counts),
            "requirements": dict(self.requirements),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    if out != out:  # NaN
        return None
    return out


def _normalize_estimator_family(value: Any) -> str:
    raw = str(value or "histgbm").strip().lower().replace("-", "_")
    aliases = {
        "histgbm": "histgbm",
        "hist_gradient_boosting": "histgbm",
        "histgradientboosting": "histgbm",
        "lightgbm": "lightgbm",
        "lgbm": "lightgbm",
    }
    return aliases.get(raw, raw)


def _manifest_estimator_family(payload: dict[str, Any]) -> str:
    raw = payload.get("estimator_family")
    if raw is None and isinstance(payload.get("estimator_contract"), dict):
        contract = payload["estimator_contract"]
        raw = contract.get("family") or contract.get("estimator_family") or contract.get("direction_family")
    if raw is None and isinstance(payload.get("training"), dict):
        raw = payload["training"].get("estimator_family")
    return _normalize_estimator_family(raw)


def _candidate_id_from_manifest(path: Path, payload: dict[str, Any]) -> str | None:
    lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
    raw = lifecycle.get("candidate_id") or payload.get("candidate_id")
    if raw:
        return str(raw)
    if path.parent.name and path.parent.name != "candidates":
        return path.parent.name
    return None


def _score_from_payload(payload: dict[str, Any]) -> float | None:
    """Return the best available comparable offline score from a GBM bundle.

    The function intentionally uses manifest-local validation/promotion metrics
    only. It does not inspect live trades, does not read backtest outcome files,
    and does not infer paper performance.
    """
    promotion = payload.get("promotion") if isinstance(payload.get("promotion"), dict) else {}
    candidates: list[Any] = [
        promotion.get("candidate_score"),
    ]
    direction = promotion.get("direction") if isinstance(promotion.get("direction"), dict) else {}
    candidates.extend([
        direction.get("candidate_score"),
        direction.get("balanced_accuracy"),
        direction.get("roc_auc_ovr"),
        direction.get("precision_macro"),
    ])
    metrics_summary = payload.get("metrics_summary") if isinstance(payload.get("metrics_summary"), dict) else {}
    if isinstance(metrics_summary.get("direction"), dict):
        direction_summary = metrics_summary["direction"]
        candidates.extend([
            direction_summary.get("candidate_score"),
            direction_summary.get("balanced_accuracy"),
            direction_summary.get("roc_auc_ovr"),
            direction_summary.get("precision_macro"),
        ])
    candidates.extend([
        payload.get("candidate_score"),
        payload.get("balanced_accuracy"),
        payload.get("roc_auc_ovr"),
        payload.get("precision_macro"),
    ])
    for value in candidates:
        score = _safe_float(value)
        if score is not None:
            return score
    return None


def _manifest_row(paths: PathManager, path: Path) -> dict[str, Any]:
    payload = load_json(path) or {}
    valid, reason = validate_artifact_bundle(paths, path)
    return {
        "path": path,
        "payload": payload,
        "valid": bool(valid),
        "validation_reason": str(reason),
        "family": _manifest_estimator_family(payload),
        "candidate_id": _candidate_id_from_manifest(path, payload),
        "score": _score_from_payload(payload),
        "created_at": str(((payload.get("timestamps") or {}) if isinstance(payload.get("timestamps"), dict) else {}).get("created_at") or ""),
    }


def _candidate_manifest_paths(paths: PathManager, symbol: str, interval: str) -> list[Path]:
    base = paths.gbm_models_dir / symbol.upper() / str(interval)
    try:
        return sorted(base.glob("candidates/*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    except Exception:
        return []


def _discover_symbol_intervals(paths: PathManager) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    base = paths.gbm_models_dir
    if base.exists():
        for symbol_dir in base.iterdir():
            if not symbol_dir.is_dir():
                continue
            for interval_dir in symbol_dir.iterdir():
                if interval_dir.is_dir():
                    pairs.add((symbol_dir.name.upper(), interval_dir.name))
    registry = load_json(paths.training_registry_path) or {}
    if isinstance(registry, dict):
        for entry in registry.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("model_type") or "").lower() != "gbm":
                continue
            symbol = str(entry.get("symbol") or "").upper()
            interval = str(entry.get("interval") or "")
            if symbol and interval:
                pairs.add((symbol, interval))
    return sorted(pairs)


def _best_lightgbm_candidate(paths: PathManager, symbol: str, interval: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for path in _candidate_manifest_paths(paths, symbol, interval):
        row = _manifest_row(paths, path)
        if row["family"] == "lightgbm":
            candidates.append(row)
    if not candidates:
        return None
    # Prefer valid candidates with comparable scores; then highest score; then newest.
    return sorted(
        candidates,
        key=lambda row: (
            1 if row.get("valid") else 0,
            row.get("score") is not None,
            float(row.get("score") if row.get("score") is not None else -1e9),
            str(row.get("created_at") or ""),
        ),
        reverse=True,
    )[0]


def _baseline_histgbm(paths: PathManager, symbol: str, interval: str) -> dict[str, Any] | None:
    active = resolve_active_manifest_path(paths, "gbm", symbol, interval, model_role="bundle")
    if active is not None:
        row = _manifest_row(paths, active)
        if row["family"] == "histgbm":
            return row
    # Fallback for candidate-only fixture/reporting contexts: use newest valid HistGBM candidate as baseline evidence.
    candidates: list[dict[str, Any]] = []
    for path in _candidate_manifest_paths(paths, symbol, interval):
        row = _manifest_row(paths, path)
        if row["family"] == "histgbm":
            candidates.append(row)
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            1 if row.get("valid") else 0,
            row.get("score") is not None,
            float(row.get("score") if row.get("score") is not None else -1e9),
            str(row.get("created_at") or ""),
        ),
        reverse=True,
    )[0]


def _evaluate_pair(paths: PathManager, symbol: str, interval: str, *, min_score_improvement: float) -> LightGbmCandidateEvaluationItem:
    baseline = _baseline_histgbm(paths, symbol, interval)
    candidate = _best_lightgbm_candidate(paths, symbol, interval)
    reasons: list[str] = []
    status = "retain_histgbm"
    recommended = False
    if baseline is None:
        reasons.append("histgbm_baseline_missing")
    elif not baseline.get("valid"):
        reasons.append(f"histgbm_baseline_invalid:{baseline.get('validation_reason')}")
    if candidate is None:
        reasons.append("lightgbm_candidate_missing")
    elif not candidate.get("valid"):
        reasons.append(f"lightgbm_candidate_invalid:{candidate.get('validation_reason')}")
    baseline_score = baseline.get("score") if baseline else None
    candidate_score = candidate.get("score") if candidate else None
    delta: float | None = None
    if baseline_score is None and baseline is not None:
        reasons.append("histgbm_baseline_score_missing")
    if candidate_score is None and candidate is not None:
        reasons.append("lightgbm_candidate_score_missing")
    if baseline_score is not None and candidate_score is not None:
        delta = float(candidate_score) - float(baseline_score)
        if delta < float(min_score_improvement):
            reasons.append(f"lightgbm_score_delta_below_threshold:{delta:.6f}<{float(min_score_improvement):.6f}")
    if not reasons:
        status = "approved_for_paper_switch_review"
        recommended = True
    elif candidate is None:
        status = "candidate_missing"
    elif baseline is None:
        status = "baseline_missing"
    elif candidate is not None and not candidate.get("valid"):
        status = "candidate_invalid"
    elif baseline is not None and not baseline.get("valid"):
        status = "baseline_invalid"
    elif baseline_score is None or candidate_score is None:
        status = "insufficient_metric_evidence"

    return LightGbmCandidateEvaluationItem(
        symbol=symbol.upper(),
        interval=str(interval),
        status=status,
        paper_approval_recommended=bool(recommended),
        activation_performed=False,
        baseline_manifest_path=str(baseline["path"]) if baseline else None,
        baseline_candidate_id=str(baseline.get("candidate_id")) if baseline and baseline.get("candidate_id") else None,
        baseline_estimator_family=str(baseline.get("family")) if baseline else None,
        baseline_score=float(baseline_score) if baseline_score is not None else None,
        candidate_manifest_path=str(candidate["path"]) if candidate else None,
        candidate_id=str(candidate.get("candidate_id")) if candidate and candidate.get("candidate_id") else None,
        candidate_estimator_family=str(candidate.get("family")) if candidate else None,
        candidate_score=float(candidate_score) if candidate_score is not None else None,
        score_delta=float(delta) if delta is not None else None,
        reasons=reasons,
        details={
            "baseline_validation_reason": baseline.get("validation_reason") if baseline else None,
            "candidate_validation_reason": candidate.get("validation_reason") if candidate else None,
        },
    )


def run_lightgbm_candidate_evaluation(
    project_root: Path | str | None = None,
    *,
    write_report: bool = False,
    output_path: Path | str | None = None,
    min_score_improvement: float | None = None,
) -> LightGbmCandidateEvaluationReport:
    """Evaluate LightGBM candidate artefacts against the HistGBM baseline.

    This is Step 9's evidence gate. It never activates a LightGBM candidate and
    never mutates model pointers, manifests, config, or live readiness.
    """
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    paths = PathManager.from_project_root(root)
    promotion = getattr(paths.config.training, "promotion", None)
    threshold = float(min_score_improvement if min_score_improvement is not None else getattr(promotion, "min_score_improvement", 0.0) or 0.0)
    items = [_evaluate_pair(paths, symbol, interval, min_score_improvement=threshold) for symbol, interval in _discover_symbol_intervals(paths)]
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    approved = [item for item in items if item.paper_approval_recommended]
    missing_candidates = [item for item in items if item.status == "candidate_missing"]
    if approved:
        overall_status = "pass"
    elif items and len(missing_candidates) == len(items):
        overall_status = "warn"
    elif items:
        overall_status = "warn"
    else:
        overall_status = "warn"
        counts["no_gbm_evidence"] = 1
    configured = _normalize_estimator_family(getattr(paths.config.training.gbm, "estimator", "histgbm"))
    report = LightGbmCandidateEvaluationReport(
        project_root=str(root),
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at=_utc_now(),
        overall_status=overall_status,
        ready_for_lightgbm_paper_approval=bool(approved),
        activation_performed=False,
        current_configured_estimator=configured,
        baseline_family="histgbm",
        candidate_family="lightgbm",
        items=items,
        counts=counts,
        requirements={
            "min_score_improvement": threshold,
            "activation_policy": "evidence_report_only_no_pointer_or_config_mutation",
            "rf_anchor_preserved": True,
            "hmm_influence_allowed": False,
            "live_candidate_validation_allowed": False,
        },
    )
    if write_report:
        destination = Path(output_path) if output_path is not None else paths.runtime_dir / REPORT_FILENAME
        save_json(destination, report.to_dict())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LightGBM GBM candidates against the HistGBM rollback baseline without activating them.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write-report", action="store_true", help="Write the evaluation report under data/runtime by default.")
    parser.add_argument("--output", type=Path, default=None, help="Optional report output path.")
    parser.add_argument("--min-score-improvement", type=float, default=None)
    args = parser.parse_args(argv)
    report = run_lightgbm_candidate_evaluation(
        args.project_root,
        write_report=bool(args.write_report),
        output_path=args.output,
        min_score_improvement=args.min_score_improvement,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.overall_status in {"pass", "warn"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
