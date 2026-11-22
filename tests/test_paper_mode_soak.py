from __future__ import annotations

from pathlib import Path

from scripts.paper_mode_soak_check import run_paper_mode_soak_check


def test_paper_mode_soak_check_reports_ready(project_root: Path) -> None:
    """Verify the repository persistence foundations survive a paper-mode soak-style lifecycle."""
    report = run_paper_mode_soak_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["paper_mode_soak_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert "control_plane_roundtrip" in names
    assert "portfolio_roundtrip" in names
    assert "monitoring_persistence" in names
