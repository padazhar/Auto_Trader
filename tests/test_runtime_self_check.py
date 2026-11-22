from __future__ import annotations

from pathlib import Path

from scripts.runtime_self_check import run_runtime_self_check


def test_runtime_self_check_reports_paper_baseline_ready(project_root: Path) -> None:
    """Verify the runtime self-check covers the paper baseline and passes."""
    report = run_runtime_self_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["runtime_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert "config_check" in names
    assert "paper_mode_guard_check" in names
    assert "registry_check" in names
    assert "web_app_check" in names
    assert "signal_engine_paused_path_check" in names
    assert "runtime_cleanup_contract_check" in names
    assert "bundled_manifest_check" in names
    assert "rf_training_contract_check" in names
