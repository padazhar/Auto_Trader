from __future__ import annotations

import json
from pathlib import Path

from scripts.final_live_validation_gate import run_final_live_validation_gate
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION


def test_final_live_validation_gate_blocks_live_candidate_without_step_evidence(project_root: Path, monkeypatch) -> None:
    monkeypatch.delenv("AUTO_TRADER_ENABLE_LIVE_MODE", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_SECRET", raising=False)

    report = run_final_live_validation_gate(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] in {"warn", "fail"}
    assert payload["controlled_live_candidate_ready"] is False
    assert payload["live_trading_ready"] is False

    by_name = {item["name"]: item for item in payload["items"]}
    assert "repository_readiness" in by_name
    assert "runtime_self_check" in by_name
    assert "registry_readiness_verification" in by_name
    assert "governed_retraining_evidence" in by_name
    assert "paper_24h_validation" in by_name
    assert "post_paper_attribution_analysis" in by_name
    assert "live_candidate_guard" in by_name
    assert "live_candidate_operational" in by_name
    assert "live_trading_readiness" in by_name


def test_final_live_validation_gate_rejects_stale_retraining_contract(project_root: Path, monkeypatch) -> None:
    monkeypatch.delenv("AUTO_TRADER_ENABLE_LIVE_MODE", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_SECRET", raising=False)

    runtime_dir = project_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "retrain_live_eligible_report.json").write_text(
        json.dumps({
            "report_status": "pass",
            "retraining_contract_version": "legacy_contract",
            "jobs_requested": 3,
            "jobs_succeeded": 3,
            "jobs_failed": 0,
        }),
        encoding="utf-8",
    )

    report = run_final_live_validation_gate(project_root)
    payload = report.to_dict()
    retraining = {item["name"]: item for item in payload["items"]}["governed_retraining_evidence"]

    assert payload["controlled_live_candidate_ready"] is False
    assert retraining["status"] == "warn"
    assert retraining["details"]["contract_ok"] is False
    assert retraining["details"]["expected_retraining_contract_version"] == RETRAINING_CONTRACT_VERSION


def test_final_live_gate_blocks_when_runtime_evidence_is_missing(project_root, monkeypatch):
    from scripts.final_live_validation_gate import run_final_live_validation_gate

    runtime_dir = project_root / "data" / "runtime"
    (runtime_dir / "paper_24h_validation_report.json").unlink(missing_ok=True)
    (runtime_dir / "retrain_live_eligible_report.json").unlink(missing_ok=True)

    class Report:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr("scripts.final_live_validation_gate.run_readiness_check", lambda root: Report(overall_status="pass", paper_trading_ready=True, live_candidate_ready=True, live_candidate_reason="ok"))
    monkeypatch.setattr("scripts.final_live_validation_gate.run_runtime_self_check", lambda root: Report(overall_status="pass", runtime_ready=True))
    monkeypatch.setattr("scripts.final_live_validation_gate.run_registry_readiness_verification", lambda root: Report(overall_status="pass", registry_gate_consistent=True, false_ready_symbols=[], registry_execution_ready_count=10, symbol_state_execution_eligible_count=10))
    monkeypatch.setattr("scripts.final_live_validation_gate.run_live_candidate_guard_check", lambda root: Report(overall_status="pass", live_guard_ready=True))
    monkeypatch.setattr("scripts.final_live_validation_gate.run_live_candidate_operational_check", lambda root: Report(overall_status="pass", live_candidate_ready=True))
    monkeypatch.setattr("scripts.final_live_validation_gate.run_live_trading_readiness_check", lambda root: Report(overall_status="pass", live_trading_ready=True))
    monkeypatch.setattr("scripts.final_live_validation_gate.market_latest_snapshot_diagnostics", lambda paths, stale_after_seconds=900.0: {"ready": True})

    report = run_final_live_validation_gate(project_root)
    payload = report.to_dict()
    item = next(item for item in payload["items"] if item["name"] == "runtime_evidence_completeness")

    assert payload["controlled_live_candidate_ready"] is False
    assert payload["live_trading_ready"] is False
    assert item["status"] == "warn"
    assert "paper_24h_validation_report" in item["details"]["missing_evidence"]
