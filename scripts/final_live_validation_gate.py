from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.market_io import market_latest_snapshot_diagnostics
from core.path_manager import PathManager
from scripts.live_candidate_guard_check import run_live_candidate_guard_check
from scripts.live_candidate_operational_check import run_live_candidate_operational_check
from scripts.live_trading_readiness_check import run_live_trading_readiness_check
from scripts.paper_24h_validation import validate_paper_24h_report
from scripts.post_paper_attribution_analysis import analyze_post_paper_attribution
from scripts.registry_readiness_verification import run_registry_readiness_verification
from scripts.repository_readiness_check import run_readiness_check
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION
from scripts.runtime_self_check import run_runtime_self_check


@dataclass(slots=True)
class FinalGateItem:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FinalLiveValidationReport:
    project_root: str
    overall_status: str
    controlled_live_candidate_ready: bool
    live_trading_ready: bool
    items: list[FinalGateItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "controlled_live_candidate_ready": self.controlled_live_candidate_ready,
            "live_trading_ready": self.live_trading_ready,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[FinalGateItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(FinalGateItem(name=name, status=status, message=message, details=dict(details)))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _latest_report_path(paths: PathManager) -> str | None:
    try:
        candidates = sorted(paths.reports_dir.glob("run_report_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    except Exception:
        candidates = []
    return str(candidates[0]) if candidates else None


def _validate_runtime_evidence_completeness(paths: PathManager) -> tuple[bool, dict[str, Any]]:
    latest_report = _latest_report_path(paths)
    required_paths: dict[str, Path | None] = {
        "training_registry": paths.runtime_dir / "training_registry.json",
        "symbol_state": paths.runtime_dir / "symbol_state.json",
        "retrain_live_eligible_report": paths.runtime_dir / "retrain_live_eligible_report.json",
        "paper_24h_validation_report": paths.runtime_dir / "paper_24h_validation_report.json",
        "latest_run_report": Path(latest_report) if latest_report else None,
        "decision_journal": paths.runtime_dir / "decision_journal.jsonl",
        "execution_journal": paths.runtime_dir / "execution_journal.jsonl",
        "portfolio_state": paths.runtime_dir / "portfolio_state.json",
        "live_backtest_trades": paths.data_dir / "live_backtest_trades.csv",
    }
    checks: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, path in required_paths.items():
        exists = bool(path is not None and path.exists())
        non_empty = bool(exists and path is not None and path.stat().st_size > 0)
        checks[name] = {"path": str(path) if path is not None else None, "exists": exists, "non_empty": non_empty}
        if not non_empty:
            missing.append(name)
    return not missing, {"required_evidence": checks, "missing_evidence": missing}


def _validate_governed_retraining_evidence(paths: PathManager) -> tuple[bool, dict[str, Any]]:
    report_path = paths.runtime_dir / "retrain_live_eligible_report.json"
    payload = _load_json(report_path)
    if not payload:
        return False, {"report_path": str(report_path), "report_present": False}

    contract_version = str(payload.get("retraining_contract_version") or "")
    jobs_requested = int(payload.get("jobs_requested") or 0)
    jobs_failed = int(payload.get("jobs_failed") or 0)
    report_status = str(payload.get("report_status") or "").lower()
    contract_ok = contract_version == RETRAINING_CONTRACT_VERSION
    ok = bool(report_status == "pass" and jobs_requested > 0 and jobs_failed == 0 and contract_ok)
    return ok, {
        "report_path": str(report_path),
        "report_present": True,
        "report_status": report_status,
        "retraining_contract_version": contract_version,
        "expected_retraining_contract_version": RETRAINING_CONTRACT_VERSION,
        "contract_ok": contract_ok,
        "jobs_requested": jobs_requested,
        "jobs_succeeded": payload.get("jobs_succeeded"),
        "jobs_failed": jobs_failed,
    }


def run_final_live_validation_gate(project_root: Path | str | None = None) -> FinalLiveValidationReport:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[FinalGateItem] = []

    repo = run_readiness_check(root)
    runtime = run_runtime_self_check(root)
    registry = run_registry_readiness_verification(root)
    live_guard = run_live_candidate_guard_check(root)
    operational = run_live_candidate_operational_check(root)
    live_readiness = run_live_trading_readiness_check(root)
    paths = PathManager.from_project_root(root)
    runtime_evidence_complete, runtime_evidence_details = _validate_runtime_evidence_completeness(paths)
    retraining_ok, retraining_details = _validate_governed_retraining_evidence(paths)
    paper_24h = validate_paper_24h_report(root)
    post_paper = analyze_post_paper_attribution(root)
    price_cache = market_latest_snapshot_diagnostics(paths, stale_after_seconds=900.0)
    price_cache_ready = bool(price_cache.get("ready"))

    _record(
        items,
        "repository_readiness",
        repo.overall_status,
        "Repository readiness check completed.",
        paper_trading_ready=bool(repo.paper_trading_ready),
        live_candidate_ready=bool(repo.live_candidate_ready),
        live_candidate_reason=str(repo.live_candidate_reason),
    )
    _record(
        items,
        "runtime_self_check",
        runtime.overall_status,
        "Runtime self-check completed.",
        runtime_ready=bool(runtime.runtime_ready),
    )
    _record(
        items,
        "registry_readiness_verification",
        str(registry.overall_status),
        "Registry/readiness verification completed. Live-candidate status requires registry-derived execution eligibility with no false-ready symbols.",
        registry_gate_consistent=bool(registry.registry_gate_consistent),
        false_ready_symbols=list(registry.false_ready_symbols),
        registry_execution_ready_count=int(registry.registry_execution_ready_count),
        symbol_state_execution_eligible_count=int(registry.symbol_state_execution_eligible_count),
    )
    _record(
        items,
        "runtime_evidence_completeness",
        "pass" if runtime_evidence_complete else "warn",
        "All required runtime evidence for live-candidate review is present." if runtime_evidence_complete else "Required runtime evidence for live-candidate review is missing; live-candidate approval remains blocked.",
        **runtime_evidence_details,
    )
    _record(
        items,
        "governed_retraining_evidence",
        "pass" if retraining_ok else "warn",
        "Governed retraining evidence satisfies the current cost-aware label/EV contract." if retraining_ok else "Governed retraining evidence is missing, stale, empty, or failed; live-candidate approval remains blocked.",
        **retraining_details,
    )
    _record(
        items,
        "paper_24h_validation",
        "pass" if paper_24h.paper_24h_passed else "warn",
        "24-hour paper validation evidence was inspected for final live-candidate gating.",
        paper_24h_passed=bool(paper_24h.paper_24h_passed),
        raw_overall_status=str(paper_24h.overall_status),
        report_path=str(paper_24h.report_path),
        validation_items=[asdict(item) for item in paper_24h.items],
    )
    _record(
        items,
        "post_paper_attribution_analysis",
        "pass" if post_paper.attribution_analysis_passed else "warn",
        "Post-paper attribution evidence was inspected for closed trades, MAE/MFE, risk rejections, and cost-aware PnL.",
        attribution_analysis_passed=bool(post_paper.attribution_analysis_passed),
        raw_overall_status=str(post_paper.overall_status),
        report_path=post_paper.report_path,
        validation_items=[asdict(item) for item in post_paper.items],
    )
    _record(
        items,
        "live_candidate_guard",
        live_guard.overall_status,
        "Live candidate guard check completed.",
        live_guard_ready=bool(live_guard.live_guard_ready),
    )
    _record(
        items,
        "live_candidate_operational",
        operational.overall_status,
        "Live candidate operational check completed.",
        live_candidate_ready=bool(operational.live_candidate_ready),
    )
    _record(
        items,
        "live_trading_readiness",
        "pass" if live_readiness.live_trading_ready else "warn",
        "Final live trading readiness gate completed.",
        raw_overall_status=str(live_readiness.overall_status),
        live_trading_ready=bool(live_readiness.live_trading_ready),
    )
    _record(
        items,
        "runtime_price_cache",
        "pass" if price_cache_ready else "warn",
        "Runtime latest-price cache is complete and fresh." if price_cache_ready else "Runtime latest-price cache is stale in the static repository snapshot; refresh during a real candidate run before live enablement.",
        diagnostics=price_cache,
    )

    hard_fail_names = {
        "repository_readiness",
        "runtime_self_check",
        "registry_readiness_verification",
        "live_candidate_guard",
        "live_candidate_operational",
    }
    hard_fail = any(item.status == "fail" and item.name in hard_fail_names for item in items)
    controlled_live_candidate_ready = all((
        bool(repo.paper_trading_ready),
        bool(runtime.runtime_ready),
        bool(registry.registry_gate_consistent),
        bool(runtime_evidence_complete),
        bool(retraining_ok),
        bool(paper_24h.paper_24h_passed),
        bool(post_paper.attribution_analysis_passed),
        bool(live_guard.live_guard_ready),
        bool(operational.live_candidate_ready),
    ))
    live_trading_ready = bool(controlled_live_candidate_ready and live_readiness.live_trading_ready)

    if hard_fail:
        overall_status = "fail"
    elif controlled_live_candidate_ready and live_trading_ready:
        overall_status = "pass"
    else:
        overall_status = "warn"

    return FinalLiveValidationReport(
        project_root=str(root),
        overall_status=overall_status,
        controlled_live_candidate_ready=controlled_live_candidate_ready,
        live_trading_ready=live_trading_ready,
        items=items,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Final controlled live validation gate")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    report = run_final_live_validation_gate(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.controlled_live_candidate_ready and report.live_trading_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
