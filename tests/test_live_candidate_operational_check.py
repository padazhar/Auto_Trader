from __future__ import annotations

from pathlib import Path

from scripts.live_candidate_operational_check import run_live_candidate_operational_check


def test_live_candidate_operational_check_reports_ready(project_root: Path) -> None:
    """Verify the repository exposes a coherent live-candidate operational gate."""
    report = run_live_candidate_operational_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["live_candidate_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert "startup_guard_contract" in names
    assert "authority_state_transitions" in names
    assert "control_plane_live_visibility" in names
    assert "pause_roundtrip" in names
