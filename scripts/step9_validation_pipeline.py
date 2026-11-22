from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.path_manager import PathManager
from scripts.final_live_validation_gate import run_final_live_validation_gate
from scripts.paper_mode_smoke_check import run_paper_mode_smoke_check
from scripts.paper_mode_soak_check import run_paper_mode_soak_check
from scripts.paper_24h_validation import validate_paper_24h_report
from scripts.post_paper_attribution_analysis import analyze_post_paper_attribution
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION, run_retraining
from scripts.registry_readiness_verification import run_registry_readiness_verification
from scripts.runtime_self_check import run_runtime_self_check


@dataclass(slots=True)
class Step9CheckItem:
    """Single Step 9 validation item."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Step9ValidationReport:
    """Step 9 retrain/smoke/paper-soak/live-candidate validation report."""

    project_root: str
    overall_status: str
    ready_for_paper_soak: bool
    ready_for_live_candidate: bool
    live_trading_ready: bool
    retraining_required: bool
    full_paper_run_required: bool
    live_candidate_validation_required: bool
    items: list[Step9CheckItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "ready_for_paper_soak": self.ready_for_paper_soak,
            "ready_for_live_candidate": self.ready_for_live_candidate,
            "live_trading_ready": self.live_trading_ready,
            "retraining_required": self.retraining_required,
            "full_paper_run_required": self.full_paper_run_required,
            "live_candidate_validation_required": self.live_candidate_validation_required,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[Step9CheckItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(Step9CheckItem(name=name, status=status, message=message, details=dict(details)))


def _status_from_bool(ok: bool, *, warn_when_false: bool = False) -> str:
    if ok:
        return "pass"
    return "warn" if warn_when_false else "fail"


def _latest_report_path(paths: PathManager) -> str | None:
    try:
        candidates = sorted(paths.reports_dir.glob("run_report_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    except Exception:
        candidates = []
    return str(candidates[0]) if candidates else None


def _runtime_evidence_completeness(paths: PathManager) -> tuple[bool, dict[str, Any]]:
    """Return whether all runtime evidence needed for live-candidate review exists.

    This check is deliberately evidence-only. It does not mark any model ready,
    does not change registry state, and does not start retraining or paper runs.
    Missing future-run evidence must keep Step 9/final live gates fail-closed.
    """
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
        non_empty = bool(exists and path is not None and (path.stat().st_size > 0))
        checks[name] = {
            "path": str(path) if path is not None else None,
            "exists": exists,
            "non_empty": non_empty,
        }
        if not non_empty:
            missing.append(name)
    return not missing, {"required_evidence": checks, "missing_evidence": missing}


def run_step9_validation_pipeline(
    project_root: Path | str | None = None,
    *,
    run_retrain: bool = False,
) -> Step9ValidationReport:
    """Run the Step 9 validation sequence without enabling live trading.

    Step 9 is the first phase after cost-aware labels/EV, so retraining is
    mandatory before any live-candidate conclusion can be trusted. By default,
    this function does not start retraining because retraining may be long and
    should be operator-triggered. Pass ``run_retrain=True`` to perform the
    repository's governed retraining job sequence and persist its report.
    """
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    paths = PathManager.from_project_root(root)
    items: list[Step9CheckItem] = []

    runtime_evidence_complete, runtime_evidence_details = _runtime_evidence_completeness(paths)
    _record(
        items,
        "runtime_evidence_completeness",
        "pass" if runtime_evidence_complete else "warn",
        "All required runtime evidence for live-candidate review is present." if runtime_evidence_complete else "Required runtime evidence for live-candidate review is missing; validation must remain fail-closed until generated by retraining and paper validation runs.",
        **runtime_evidence_details,
    )

    runtime = run_runtime_self_check(root)
    _record(
        items,
        "runtime_self_check",
        str(runtime.overall_status),
        "Runtime self-check completed before Step 9 validation.",
        runtime_ready=bool(runtime.runtime_ready),
    )

    retrain_ok = False
    if run_retrain:
        try:
            retrain_report = run_retraining(paths)
            retrain_ok = retrain_report.report_status == "pass"
            _record(
                items,
                "governed_retraining",
                str(retrain_report.report_status),
                "Governed retraining completed for current live-eligible job set.",
                jobs_requested=int(retrain_report.jobs_requested),
                jobs_succeeded=int(retrain_report.jobs_succeeded),
                jobs_failed=int(retrain_report.jobs_failed),
            )
        except Exception as exc:
            _record(items, "governed_retraining", "fail", "Governed retraining failed.", error=str(exc))
    else:
        report_path = paths.runtime_dir / "retrain_live_eligible_report.json"
        if report_path.exists():
            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                contract_version = str(payload.get("retraining_contract_version") or "")
                contract_ok = contract_version == RETRAINING_CONTRACT_VERSION
                retrain_ok = (
                    str(payload.get("report_status") or "").lower() == "pass"
                    and int(payload.get("jobs_requested") or 0) > 0
                    and int(payload.get("jobs_failed") or 0) == 0
                    and contract_ok
                )
                _record(
                    items,
                    "governed_retraining",
                    "pass" if retrain_ok else "warn",
                    "Existing retraining report found and inspected.",
                    report_path=str(report_path),
                    report_status=payload.get("report_status"),
                    retraining_contract_version=payload.get("retraining_contract_version"),
                    expected_retraining_contract_version=RETRAINING_CONTRACT_VERSION,
                    contract_ok=contract_ok,
                    jobs_requested=payload.get("jobs_requested"),
                    jobs_succeeded=payload.get("jobs_succeeded"),
                    jobs_failed=payload.get("jobs_failed"),
                )
            except Exception as exc:
                _record(items, "governed_retraining", "fail", "Existing retraining report could not be parsed.", report_path=str(report_path), error=str(exc))
        else:
            _record(
                items,
                "governed_retraining",
                "warn",
                "Retraining is required after Step 8 cost-aware label/EV changes; no retraining report was found. Run this script with --run-retrain or run scripts/retrain_live_eligible_candidates.py before trusting live-candidate results.",
                report_path=str(report_path),
            )

    registry_verification = run_registry_readiness_verification(root)
    _record(
        items,
        "registry_readiness_verification",
        str(registry_verification.overall_status),
        "Registry/readiness verification completed. Execution eligibility must remain derived from registry-approved active manifests.",
        registry_gate_consistent=bool(registry_verification.registry_gate_consistent),
        active_symbols_count=int(registry_verification.active_symbols_count),
        symbol_state_execution_eligible_count=int(registry_verification.symbol_state_execution_eligible_count),
        registry_execution_ready_count=int(registry_verification.registry_execution_ready_count),
        false_ready_symbols=list(registry_verification.false_ready_symbols),
        registry_ready_not_in_symbol_state=list(registry_verification.registry_ready_not_in_symbol_state),
    )

    smoke = run_paper_mode_smoke_check(root)
    _record(
        items,
        "paper_mode_smoke",
        str(smoke.overall_status),
        "Paper-mode smoke check completed.",
        paper_mode_ready=bool(smoke.paper_mode_ready),
    )

    soak = run_paper_mode_soak_check(root)
    _record(
        items,
        "paper_mode_soak_contract",
        str(soak.overall_status),
        "Paper-mode persistence soak contract completed. This is not a substitute for the required 24-hour paper run.",
        paper_mode_soak_ready=bool(soak.paper_mode_soak_ready),
    )

    paper_24h = validate_paper_24h_report(root)
    full_paper_run_ok = bool(paper_24h.paper_24h_passed)
    _record(
        items,
        "paper_24h_validation",
        str(paper_24h.overall_status),
        "24-hour paper validation evidence was inspected. A Boolean flag alone is not sufficient; duration, paper mode, runtime stability, governed retraining, registry approval, and cost realism must all be proven.",
        report_path=str(paper_24h.report_path),
        paper_24h_passed=bool(paper_24h.paper_24h_passed),
        validation_items=[asdict(item) for item in paper_24h.items],
    )

    post_paper = analyze_post_paper_attribution(root)
    _record(
        items,
        "post_paper_attribution_analysis",
        str(post_paper.overall_status),
        "Post-paper attribution evidence was inspected. Live-candidate review requires closed-trade attribution, MAE/MFE, risk rejection attribution, and cost-aware PnL fields after the 24-hour paper run.",
        report_path=post_paper.report_path,
        attribution_analysis_passed=bool(post_paper.attribution_analysis_passed),
        validation_items=[asdict(item) for item in post_paper.items],
    )
    post_paper_ok = bool(post_paper.attribution_analysis_passed)

    final_gate = run_final_live_validation_gate(root)
    _record(
        items,
        "final_live_validation_gate",
        str(final_gate.overall_status),
        "Final live validation gate completed. Live trading must remain disabled/fail-closed unless this gate explicitly passes under operator-controlled live credentials and reconciliation.",
        controlled_live_candidate_ready=bool(final_gate.controlled_live_candidate_ready),
        live_trading_ready=bool(final_gate.live_trading_ready),
    )

    hard_fail = any(item.status == "fail" for item in items)
    ready_for_paper_soak = bool(
        not hard_fail
        and runtime.runtime_ready
        and smoke.paper_mode_ready
        and soak.paper_mode_soak_ready
        and retrain_ok
    )
    ready_for_live_candidate = bool(ready_for_paper_soak and runtime_evidence_complete and full_paper_run_ok and post_paper_ok and final_gate.controlled_live_candidate_ready)
    live_trading_ready = bool(ready_for_live_candidate and final_gate.live_trading_ready)

    if hard_fail:
        overall_status = "fail"
    elif ready_for_live_candidate and live_trading_ready:
        overall_status = "pass"
    else:
        overall_status = "warn"

    report = Step9ValidationReport(
        project_root=str(root),
        overall_status=overall_status,
        ready_for_paper_soak=ready_for_paper_soak,
        ready_for_live_candidate=ready_for_live_candidate,
        live_trading_ready=live_trading_ready,
        retraining_required=not retrain_ok,
        full_paper_run_required=not full_paper_run_ok,
        live_candidate_validation_required=not live_trading_ready,
        items=items,
    )

    output_path = paths.runtime_dir / "step9_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Step 9 retrain, smoke, paper-soak, and live-candidate validation pipeline")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    parser.add_argument("--run-retrain", action="store_true", help="Run governed retraining before smoke/soak/live-candidate checks")
    args = parser.parse_args(argv)

    report = run_step9_validation_pipeline(args.project_root, run_retrain=bool(args.run_retrain))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready_for_paper_soak and not any(item.status == "fail" for item in report.items) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
