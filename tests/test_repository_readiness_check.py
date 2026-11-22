from __future__ import annotations

from pathlib import Path

from scripts.repository_readiness_check import run_readiness_check


def test_repository_readiness_check_reports_paper_ready_but_not_live_candidate_ready(project_root: Path) -> None:
    """Verify the repository readiness check reflects the current staged rollout truth."""
    report = run_readiness_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["paper_trading_ready"] is True
    assert payload["live_candidate_ready"] is False
    assert "reconciliation" in payload["live_candidate_reason"].lower()

    names = {item["name"] for item in payload["items"]}
    assert "config_schema" in names
    assert "launcher_modes" in names
    assert "training_governance" in names
    assert "feature_compatibility" in names
    assert "canonical_futures_context" in names
    assert "exchange_reconciliation" in names
    assert "control_plane_and_live_guard" in names
    assert "operator_naming" in names

    launcher_item = next(item for item in payload["items"] if item["name"] == "launcher_modes")
    assert launcher_item["status"] == "pass"
    assert "runtime" in launcher_item["details"]["choices"]
    assert "trainer" in launcher_item["details"]["choices"]
    assert launcher_item["details"]["isolated_process_roles"] == ["runtime", "trainer"]
