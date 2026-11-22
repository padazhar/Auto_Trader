from __future__ import annotations

import json

from scripts.paper_24h_validation import MIN_PAPER_VALIDATION_SECONDS, validate_paper_24h_report
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION
from scripts.step9_validation_pipeline import run_step9_validation_pipeline


class _Report:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _runtime_dir(project_root):
    path = project_root / "data" / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_paper_24h_validation_warns_when_report_missing(project_root):
    report_path = _runtime_dir(project_root) / "paper_24h_validation_report.json"
    report_path.unlink(missing_ok=True)

    report = validate_paper_24h_report(project_root)

    assert report.overall_status == "warn"
    assert report.paper_24h_passed is False
    assert any(item.name == "report_present" and item.status == "warn" for item in report.items)


def test_paper_24h_validation_rejects_boolean_only_report(project_root):
    report_path = _runtime_dir(project_root) / "paper_24h_validation_report.json"
    report_path.write_text(json.dumps({"paper_24h_passed": True}), encoding="utf-8")

    report = validate_paper_24h_report(project_root)

    assert report.paper_24h_passed is False
    assert report.overall_status in {"fail", "warn"}
    names = {item.name for item in report.items}
    assert {"duration", "paper_mode", "registry_approval", "cost_realism"}.issubset(names)


def test_paper_24h_validation_accepts_complete_evidence(project_root):
    report_path = _runtime_dir(project_root) / "paper_24h_validation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "paper_24h_passed": True,
                "duration_seconds": MIN_PAPER_VALIDATION_SECONDS,
                "execution_mode": "paper",
                "runtime_stability_clean": True,
                "governed_retraining_completed": True,
                "registry_gate_consistent": True,
                "fees_slippage_spread_included": True,
            }
        ),
        encoding="utf-8",
    )

    report = validate_paper_24h_report(project_root)

    assert report.overall_status == "pass"
    assert report.paper_24h_passed is True


def test_step9_pipeline_requires_complete_24h_paper_evidence(project_root, monkeypatch):
    runtime_dir = _runtime_dir(project_root)
    (runtime_dir / "retrain_live_eligible_report.json").write_text(
        json.dumps(
            {
                "report_status": "pass",
                "retraining_contract_version": RETRAINING_CONTRACT_VERSION,
                "jobs_requested": 1,
                "jobs_succeeded": 1,
                "jobs_failed": 0,
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "paper_24h_validation_report.json").write_text(json.dumps({"paper_24h_passed": True}), encoding="utf-8")

    monkeypatch.setattr("scripts.step9_validation_pipeline.run_runtime_self_check", lambda root: _Report(overall_status="pass", runtime_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_registry_readiness_verification", lambda root: _Report(overall_status="pass", registry_gate_consistent=True, active_symbols_count=10, symbol_state_execution_eligible_count=10, registry_execution_ready_count=10, false_ready_symbols=[], registry_ready_not_in_symbol_state=[]))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_paper_mode_smoke_check", lambda root: _Report(overall_status="pass", paper_mode_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_paper_mode_soak_check", lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_final_live_validation_gate", lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False))

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["ready_for_paper_soak"] is False
    assert payload["ready_for_live_candidate"] is False
    assert payload["full_paper_run_required"] is True
    paper_item = next(item for item in payload["items"] if item["name"] == "paper_24h_validation")
    assert paper_item["status"] in {"fail", "warn"}
    assert paper_item["details"]["paper_24h_passed"] is False
