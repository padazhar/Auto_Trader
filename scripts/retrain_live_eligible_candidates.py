from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import re

from core.artifact_contract import resolve_artifact_path, validate_artifact_bundle
from core.path_manager import PathManager
from core.result_contracts import TrainerResult, coerce_result
from core.training_registry import TrainingRegistry
from scheduling.priority_manager import _load_symbols
from training.dqn_training_engine import train_dqn_model
from training.gbm_training_engine import train_gbm_model
from training.rf_training_engine import train_rf_model

RETRAINING_CONTRACT_VERSION = "cost_aware_labels_ev_v1"


@dataclass(slots=True)
class RetrainJob:
    model_type: str
    symbol: str
    interval: str


@dataclass(slots=True)
class RetrainJobResult:
    model_type: str
    symbol: str
    interval: str
    ok: bool
    rows_used: int
    manifest_path: str | None = None
    artifact_valid: bool = False
    artifact_validation_error: str | None = None
    candidate_id: str | None = None
    live_eligible: bool | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrainReport:
    project_root: str
    retraining_contract_version: str
    retraining_scope: str
    affected_anchor_jobs: list[dict[str, Any]]
    generated_at_utc: str
    cost_aware_label_ev_contract: dict[str, Any]
    jobs_requested: int
    jobs_succeeded: int
    jobs_failed: int
    report_status: str
    registry_before: dict[str, Any]
    registry_after: dict[str, Any]
    registry_execution_ready_delta: int
    results: list[RetrainJobResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "retraining_contract_version": self.retraining_contract_version,
            "retraining_scope": self.retraining_scope,
            "affected_anchor_jobs": list(self.affected_anchor_jobs),
            "generated_at_utc": self.generated_at_utc,
            "cost_aware_label_ev_contract": dict(self.cost_aware_label_ev_contract),
            "jobs_requested": self.jobs_requested,
            "jobs_succeeded": self.jobs_succeeded,
            "jobs_failed": self.jobs_failed,
            "report_status": self.report_status,
            "registry_before": dict(self.registry_before),
            "registry_after": dict(self.registry_after),
            "registry_execution_ready_delta": int(self.registry_execution_ready_delta),
            "results": [asdict(item) for item in self.results],
        }




def _registry_governance_summary(registry: TrainingRegistry) -> dict[str, Any]:
    """Return a compact, read-only governance summary for retraining reports.

    This is evidence only. It must not approve, demote, delete, or mutate model
    artifacts. The goal is to prove whether governed retraining increased the
    number of registry-approved execution-ready models.
    """
    try:
        entries = registry.snapshot().get("entries", {})
    except Exception:
        entries = {}
    status_counts: dict[str, int] = {}
    truth_status_counts: dict[str, int] = {}
    execution_ready_by_model: dict[str, int] = {}
    blocked_by_truth_status: dict[str, int] = {}
    total_entries = 0
    execution_ready_count = 0
    for item in entries.values():
        if not isinstance(item, dict):
            continue
        total_entries += 1
        status = str(item.get("status") or "unknown")
        truth_status = str(item.get("registry_truth_status") or "unknown")
        model_type = str(item.get("model_type") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        truth_status_counts[truth_status] = truth_status_counts.get(truth_status, 0) + 1
        if bool(item.get("execution_ready", False)):
            execution_ready_count += 1
            execution_ready_by_model[model_type] = execution_ready_by_model.get(model_type, 0) + 1
        else:
            blocked_by_truth_status[truth_status] = blocked_by_truth_status.get(truth_status, 0) + 1
    return {
        "total_entries": total_entries,
        "execution_ready_count": execution_ready_count,
        "status_counts": status_counts,
        "registry_truth_status_counts": truth_status_counts,
        "execution_ready_by_model": execution_ready_by_model,
        "blocked_by_truth_status": blocked_by_truth_status,
    }

def _config_float(cfg: Any | None, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(cfg, name, default) or default)
    except Exception:
        return float(default)


def collect_cost_aware_label_ev_contract(paths: PathManager) -> dict[str, Any]:
    """Return the retraining contract that must match Step 8 cost-aware labels/EV.

    The values are report-only governance metadata. They do not change model
    training, thresholds, risk sizing, or live enablement. Step 9 uses this
    metadata to reject stale retraining reports created before cost-aware
    labels/EV were introduced.
    """
    trading_cfg = getattr(paths.config, "trading", None)
    return {
        "contract_version": RETRAINING_CONTRACT_VERSION,
        "fees_bps": _config_float(trading_cfg, "fees_bps"),
        "slippage_bps": _config_float(trading_cfg, "slippage_bps"),
        "spread_bps": _config_float(trading_cfg, "spread_bps"),
        "funding_bps": _config_float(trading_cfg, "funding_bps"),
        "min_profit_atr_mult": _config_float(trading_cfg, "min_profit_atr_mult"),
    }

def _intervals_for_model(paths: PathManager, model_type: str) -> list[str]:
    scheduling_cfg = getattr(paths.config, "scheduling", None)
    frames_cfg = getattr(paths.config, "frames", None)
    priority_intervals = list(getattr(scheduling_cfg, "priority_intervals", []) or [])
    fusion_intervals = list(getattr(frames_cfg, "fusion_intervals", []) or [])
    if model_type == "dqn":
        allowed = list(getattr(paths.config.training.dqn, "allowed_intervals", []) or [])
        return [interval for interval in allowed if interval]
    if priority_intervals:
        eligible = [interval for interval in priority_intervals if interval in fusion_intervals or not fusion_intervals]
        if eligible:
            return eligible
    anchor = str(getattr(frames_cfg, "anchor_interval", "15m") or "15m")
    return [anchor]


def discover_live_eligible_jobs(paths: PathManager) -> list[RetrainJob]:
    jobs: list[RetrainJob] = []
    symbols = _load_symbols(paths)
    model_enabled = {
        "rf": bool(getattr(paths.config.training.rf, "enabled", False)),
        "gbm": bool(getattr(paths.config.training.gbm, "enabled", False)),
        "dqn": bool(getattr(paths.config.training.dqn, "enabled", False)),
    }
    for model_type in ("rf", "gbm", "dqn"):
        if not model_enabled.get(model_type, False):
            continue
        for symbol in symbols:
            for interval in _intervals_for_model(paths, model_type):
                jobs.append(RetrainJob(model_type=model_type, symbol=symbol, interval=interval))
    return jobs




def _load_symbol_state(paths: PathManager) -> dict[str, Any]:
    path = paths.runtime_dir / "symbol_state.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _parse_anchor_blockers(reason: str, *, default_interval: str = "15m") -> list[dict[str, str]]:
    """Extract anchor model blockers from readiness reason text.

    Expected fragments can come from either the older runtime reason contract::

        anchor_readiness_not_ready:rf:15m:registry_status:idle
        anchor_readiness_not_ready:gbm:15m:feature_incompatible:feature_set_missing

    or the hardened Step 1-6 readiness summary emitted by the signal layer::

        anchor_training_incomplete:rf:deployment_stage:candidate,gbm:feature_incompatible:feature_set_missing

    The newer summary intentionally omits the anchor interval because the
    runtime already treats the configured anchor frame as canonical. For
    recovery, resolve those compact fragments to ``default_interval`` so the
    retraining helper can repair stale/candidate RF and GBM anchors without
    weakening readiness gates. The parser is intentionally conservative and
    only creates retraining jobs for RF/GBM anchor blockers. It never approves
    models or changes readiness.
    """
    blockers: list[dict[str, str]] = []
    text = str(reason or "")
    normalized = text
    for prefix in ("anchor_readiness_not_ready:", "anchor_training_incomplete:"):
        normalized = normalized.replace(prefix, "")
    fallback_interval = str(default_interval or "15m")
    for fragment in normalized.split(","):
        raw = fragment.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) < 2:
            continue
        model_type = parts[0].strip().lower()
        if model_type not in {"rf", "gbm"}:
            continue
        if len(parts) >= 3 and re.fullmatch(r"\d+[mhdw]", str(parts[1]).strip().lower()):
            interval = str(parts[1]).strip()
            detail = ":".join(parts[2:]).strip() or "unknown"
        else:
            interval = fallback_interval
            detail = ":".join(parts[1:]).strip() or "unknown"
        blockers.append({"model_type": model_type, "interval": interval, "reason": detail})
    return blockers


def discover_affected_anchor_jobs(paths: PathManager, *, include_warming: bool = False) -> list[RetrainJob]:
    """Return RF/GBM anchor retraining jobs for active, non-ready symbols.

    This is a governed recovery helper for Step 5. It targets only symbols
    that are active but not execution-eligible and have explicit anchor model
    readiness blockers. Market-data-warming symbols are skipped by default so
    retraining does not try to force incomplete data into execution readiness.
    """
    state = _load_symbol_state(paths)
    symbols = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    frames_cfg = getattr(paths.config, "frames", None)
    anchor_interval = str(getattr(frames_cfg, "anchor_interval", "15m") or "15m")
    active = set(state.get("active_symbols") or [])
    eligible = set(state.get("execution_eligible_symbols") or [])
    jobs: list[RetrainJob] = []
    seen: set[tuple[str, str, str]] = set()
    for symbol in sorted(active):
        if symbol in eligible:
            continue
        entry = symbols.get(symbol) if isinstance(symbols.get(symbol), dict) else {}
        data_reason = str(entry.get("data_ready_reason") or "")
        if not include_warming and data_reason.startswith("market_data_warming"):
            continue
        for blocker in _parse_anchor_blockers(str(entry.get("model_ready_reason") or entry.get("readiness_last_reason") or ""), default_interval=anchor_interval):
            model_type = blocker["model_type"]
            interval = blocker["interval"]
            key = (model_type, symbol, interval)
            if key in seen:
                continue
            seen.add(key)
            jobs.append(RetrainJob(model_type=model_type, symbol=symbol, interval=interval))
    return jobs


def _affected_anchor_job_details(paths: PathManager, jobs: list[RetrainJob]) -> list[dict[str, Any]]:
    state = _load_symbol_state(paths)
    symbols = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    details: list[dict[str, Any]] = []
    for job in jobs:
        entry = symbols.get(job.symbol) if isinstance(symbols.get(job.symbol), dict) else {}
        details.append({
            "model_type": job.model_type,
            "symbol": job.symbol,
            "interval": job.interval,
            "data_ready_reason": entry.get("data_ready_reason"),
            "model_ready_reason": entry.get("model_ready_reason"),
            "readiness_last_reason": entry.get("readiness_last_reason"),
        })
    return details

def _trainer_map() -> dict[str, Callable[[Any, str, str], TrainerResult]]:
    return {
        "rf": train_rf_model,
        "gbm": train_gbm_model,
        "dqn": train_dqn_model,
    }


def _result_live_eligible(result: TrainerResult) -> bool | None:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    governance = metrics.get("governance") if isinstance(metrics.get("governance"), dict) else {}
    if "live_eligible" in governance:
        return bool(governance.get("live_eligible"))
    if "live_eligible" in metrics:
        try:
            return bool(metrics.get("live_eligible"))
        except Exception:
            return None
    return None


def _result_candidate_id(result: TrainerResult) -> str | None:
    metrics = result.metrics if isinstance(result.metrics, dict) else {}
    promotion = metrics.get("promotion") if isinstance(metrics.get("promotion"), dict) else {}
    candidate_id = promotion.get("candidate_id") or metrics.get("candidate_id")
    return str(candidate_id) if candidate_id else None


def _validate_manifest(paths: PathManager, result: TrainerResult) -> tuple[bool, str | None, str | None]:
    artifacts = result.artifacts if isinstance(result.artifacts, dict) else {}
    manifest_raw = artifacts.get("manifest") or artifacts.get("direction_manifest")
    if not manifest_raw:
        return False, "manifest_missing", None
    manifest_path = resolve_artifact_path(paths, manifest_raw)
    ok, error = validate_artifact_bundle(paths, manifest_path)
    return bool(ok), (None if ok else str(error)), str(manifest_path)


def run_retraining(paths: PathManager, jobs: list[RetrainJob] | None = None, *, retraining_scope: str = "all_live_eligible") -> RetrainReport:
    registry = TrainingRegistry(paths)
    registry_before = _registry_governance_summary(registry)
    jobs = discover_live_eligible_jobs(paths) if jobs is None else list(jobs)
    affected_anchor_details = _affected_anchor_job_details(paths, jobs) if retraining_scope == "affected_anchor" else []
    results: list[RetrainJobResult] = []
    trainers = _trainer_map()

    for job in jobs:
        registry.mark_running(job.model_type, job.symbol, job.interval)
        trainer = trainers[job.model_type]
        try:
            result = coerce_result(trainer(paths, job.symbol, job.interval))
        except Exception as exc:
            registry.mark_failed(job.model_type, job.symbol, job.interval, str(exc))
            results.append(
                RetrainJobResult(
                    model_type=job.model_type,
                    symbol=job.symbol,
                    interval=job.interval,
                    ok=False,
                    rows_used=0,
                    error=str(exc),
                )
            )
            continue

        if not result.ok:
            registry.mark_failed(job.model_type, job.symbol, job.interval, str(result.error or "trainer_failed"))
            results.append(
                RetrainJobResult(
                    model_type=job.model_type,
                    symbol=job.symbol,
                    interval=job.interval,
                    ok=False,
                    rows_used=int(result.rows_used),
                    error=str(result.error or "trainer_failed"),
                    metrics=dict(result.metrics or {}),
                )
            )
            continue

        artifact_valid, artifact_error, manifest_path = _validate_manifest(paths, result)
        if not artifact_valid:
            registry.mark_failed(job.model_type, job.symbol, job.interval, str(artifact_error or "artifact_validation_failed"))
            results.append(
                RetrainJobResult(
                    model_type=job.model_type,
                    symbol=job.symbol,
                    interval=job.interval,
                    ok=False,
                    rows_used=int(result.rows_used),
                    manifest_path=manifest_path,
                    artifact_valid=False,
                    artifact_validation_error=str(artifact_error or "artifact_validation_failed"),
                    candidate_id=_result_candidate_id(result),
                    live_eligible=_result_live_eligible(result),
                    error=str(artifact_error or "artifact_validation_failed"),
                    metrics=dict(result.metrics or {}),
                )
            )
            continue

        registry.mark_success(
            model_type=job.model_type,
            symbol=job.symbol,
            interval=job.interval,
            metrics=dict(result.metrics or {}),
            artifacts=dict(result.artifacts or {}),
            rows_used=int(result.rows_used),
        )
        results.append(
            RetrainJobResult(
                model_type=job.model_type,
                symbol=job.symbol,
                interval=job.interval,
                ok=True,
                rows_used=int(result.rows_used),
                manifest_path=manifest_path,
                artifact_valid=True,
                artifact_validation_error=None,
                candidate_id=_result_candidate_id(result),
                live_eligible=_result_live_eligible(result),
                error=None,
                metrics=dict(result.metrics or {}),
            )
        )

    succeeded = sum(1 for item in results if item.ok)
    failed = len(results) - succeeded
    registry_after = _registry_governance_summary(registry)
    report_status = "pass" if len(jobs) > 0 and failed == 0 else "fail"
    report = RetrainReport(
        project_root=str(paths.project_root),
        retraining_contract_version=RETRAINING_CONTRACT_VERSION,
        retraining_scope=str(retraining_scope),
        affected_anchor_jobs=affected_anchor_details,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        cost_aware_label_ev_contract=collect_cost_aware_label_ev_contract(paths),
        jobs_requested=len(jobs),
        jobs_succeeded=succeeded,
        jobs_failed=failed,
        report_status=report_status,
        registry_before=registry_before,
        registry_after=registry_after,
        registry_execution_ready_delta=int(registry_after.get("execution_ready_count", 0)) - int(registry_before.get("execution_ready_count", 0)),
        results=results,
    )
    report_path = paths.runtime_dir / "retrain_live_eligible_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain all live-eligible model families under the current governance contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--affected-anchor-only", action="store_true", help="Retrain only active non-ready RF/GBM anchor blockers from symbol_state.json.")
    parser.add_argument("--include-warming", action="store_true", help="Include market-data-warming symbols in affected-anchor discovery.")
    args = parser.parse_args(argv)
    paths = PathManager.from_project_root(args.project_root)
    jobs = discover_affected_anchor_jobs(paths, include_warming=bool(args.include_warming)) if args.affected_anchor_only else None
    report = run_retraining(paths, jobs=jobs, retraining_scope="affected_anchor" if args.affected_anchor_only else "all_live_eligible")
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.report_status == "pass" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
