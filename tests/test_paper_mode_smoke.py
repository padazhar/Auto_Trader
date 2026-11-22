from __future__ import annotations

from pathlib import Path

from scripts.paper_mode_smoke_check import run_paper_mode_smoke_check


def test_paper_mode_smoke_check_reports_ready(project_root: Path) -> None:
    """Verify the repository is paper-safe at the lightweight startup/construction layer."""
    report = run_paper_mode_smoke_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["paper_mode_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert "config_bootstrap" in names
    assert "module_imports" in names
    assert "launcher_guard_all_mode" in names
    assert "constructor_training_registry" in names
    assert "constructor_control_plane_store" in names
    assert "constructor_signal_engine" in names
    assert "constructor_live_trading_controller" in names
