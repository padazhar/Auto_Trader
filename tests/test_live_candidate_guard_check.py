from __future__ import annotations

from pathlib import Path

from scripts.live_candidate_guard_check import run_live_candidate_guard_check


def test_live_candidate_guard_check_reports_ready(project_root: Path) -> None:
    """Verify live-only reconciliation and fail-closed authority guards are wired coherently."""
    report = run_live_candidate_guard_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["live_guard_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert "disabled_reconciliation_guard" in names
    assert "reconciliation_error_guard" in names
    assert "reconciliation_mismatch_guard" in names
    assert "reconciliation_stale_guard" in names
    assert "unreconciled_override_visibility" in names
