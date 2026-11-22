from pathlib import Path


def test_dashboard_frontend_retains_last_valid_snapshot_on_empty_refresh():
    app_js = Path("monitoring/static/app.js").read_text(encoding="utf-8")

    assert "lastGoodSnapshot" in app_js
    assert "isMeaningfulSnapshotPayload" in app_js
    assert "mergeSnapshotForDisplay" in app_js
    assert "Latest snapshot was empty/partial; holding last valid UI state." in app_js
    assert "state.lastGoodSnapshot = state.snapshot" in app_js


def test_dashboard_events_refresh_does_not_clear_previous_events_on_fetch_failure():
    app_js = Path("monitoring/static/app.js").read_text(encoding="utf-8")

    assert "lastGoodEvents" in app_js
    assert "events refresh failed" in app_js
    assert "renderEvents(state.lastGoodEvents)" in app_js
