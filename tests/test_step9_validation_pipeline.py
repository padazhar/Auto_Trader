from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION
from scripts.step9_validation_pipeline import run_step9_validation_pipeline


class _Report(SimpleNamespace):
    pass


def _clear_step9_runtime_reports(project_root: Path) -> None:
    runtime_dir = project_root / "data" / "runtime"
    for name in ("retrain_live_eligible_report.json", "paper_24h_validation_report.json", "step9_validation_report.json"):
        try:
            (runtime_dir / name).unlink()
        except FileNotFoundError:
            pass


def test_step9_pipeline_warns_until_retraining_and_24h_paper_validation_exist(project_root: Path, monkeypatch) -> None:
    _clear_step9_runtime_reports(project_root)
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["overall_status"] == "warn"
    assert payload["ready_for_paper_soak"] is False
    assert payload["ready_for_live_candidate"] is False
    assert payload["live_trading_ready"] is False
    assert payload["retraining_required"] is True
    assert payload["full_paper_run_required"] is True
    names = {item["name"] for item in payload["items"]}
    assert "governed_retraining" in names
    assert "paper_24h_validation" in names
    assert (project_root / "data" / "runtime" / "step9_validation_report.json").exists()


def test_step9_pipeline_allows_paper_soak_after_retraining_report(project_root: Path, monkeypatch) -> None:
    _clear_step9_runtime_reports(project_root)
    runtime_dir = project_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "retrain_live_eligible_report.json").write_text(
        json.dumps({"report_status": "pass", "retraining_contract_version": RETRAINING_CONTRACT_VERSION, "jobs_requested": 2, "jobs_succeeded": 2, "jobs_failed": 0}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["overall_status"] == "warn"
    assert payload["ready_for_paper_soak"] is True
    assert payload["ready_for_live_candidate"] is False
    assert payload["live_trading_ready"] is False
    assert payload["retraining_required"] is False
    assert payload["full_paper_run_required"] is True


def test_step9_pipeline_can_run_governed_retraining_when_requested(project_root: Path, monkeypatch) -> None:
    _clear_step9_runtime_reports(project_root)
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_retraining",
        lambda paths: _Report(report_status="pass", jobs_requested=1, jobs_succeeded=1, jobs_failed=0),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=True)
    payload = report.to_dict()

    assert payload["ready_for_paper_soak"] is True
    assert payload["retraining_required"] is False
    assert payload["live_trading_ready"] is False


def test_step9_pipeline_rejects_stale_retraining_report_without_cost_aware_contract(project_root: Path, monkeypatch) -> None:
    _clear_step9_runtime_reports(project_root)
    runtime_dir = project_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "retrain_live_eligible_report.json").write_text(
        json.dumps({"report_status": "pass", "jobs_requested": 2, "jobs_succeeded": 2, "jobs_failed": 0}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["retraining_required"] is True
    item = next(item for item in payload["items"] if item["name"] == "governed_retraining")
    assert item["status"] == "warn"
    assert item["details"]["contract_ok"] is False


def test_step9_pipeline_reports_missing_runtime_evidence(project_root: Path, monkeypatch) -> None:
    _clear_step9_runtime_reports(project_root)
    runtime_dir = project_root / "data" / "runtime"
    for name in ("execution_journal.jsonl", "portfolio_state.json"):
        (runtime_dir / name).unlink(missing_ok=True)

    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=False, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    item = next(item for item in report.to_dict()["items"] if item["name"] == "runtime_evidence_completeness")

    assert item["status"] == "warn"
    assert "paper_24h_validation_report" in item["details"]["missing_evidence"]
    assert "retrain_live_eligible_report" in item["details"]["missing_evidence"]
