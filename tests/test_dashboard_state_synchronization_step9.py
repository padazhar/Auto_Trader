from pathlib import Path


APP_JS = Path("monitoring/static/app.js")


def test_dashboard_uses_centralized_merge_for_all_snapshot_sources():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "function applyDashboardSnapshot" in app_js
    assert "Always merge into last known-good state" in app_js
    assert "renderSnapshot(snapshot, source = \"snapshot\")" in app_js
    assert "renderSnapshot(payload.payload || {}, \"websocket_snapshot\")" in app_js
    assert "loadInitialState(\"poll_snapshot\")" in app_js


def test_dashboard_polling_is_not_started_when_websocket_is_connected():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "socket.onopen = () =>" in app_js
    assert "stopPolling();" in app_js
    assert "connectSocket();\n}" in app_js
    assert "connectSocket();\n  startPolling(false);" not in app_js


def test_dashboard_prevents_overlapping_poll_refreshes():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "pollInFlight" in app_js
    assert "if (state.pollInFlight) return;" in app_js
    assert "state.pollInFlight = true;" in app_js
    assert "state.pollInFlight = false;" in app_js


def test_dashboard_preserves_previous_rows_for_partial_empty_payloads():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "function shouldPreserveEmptyArray" in app_js
    assert "Keep the last known-good rows" in app_js
    assert "execution_runtime.open_positions" in app_js
    assert "paper_execution_runtime.trade_history" in app_js


def test_dashboard_preserves_regressive_default_sections_from_standalone_process():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "function shouldPreserveRegressiveSection" in app_js
    assert "A standalone dashboard process can briefly read a skeleton/default section" in app_js
    assert "previousScore >= 3 && incomingScore <= 1" in app_js
    assert "shouldPreserveRegressiveSection(nextPath, previous[key], value)" in app_js


def test_dashboard_formats_invalid_model_dates_as_blank_placeholder():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "if (!Number.isFinite(raw)) return \"--\";" in app_js
    assert "Number.isNaN(date.getTime()) ? \"--\"" in app_js


def test_dashboard_updates_dynamic_panels_atomically_without_blank_repaint():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "function replaceHtmlStable" in app_js
    assert "if (cacheKey && state.renderedHtml.get(cacheKey) === nextHtml) return false;" in app_js
    assert "scrollHost.scrollTop" in app_js
    assert 'body.innerHTML = "";' not in app_js


def test_dashboard_has_websocket_heartbeat_watchdog_and_popup_resync():
    app_js = APP_JS.read_text(encoding="utf-8")

    assert "lastSocketMessageAt" in app_js
    assert "function startSocketWatchdog" in app_js
    assert 'payload.type === "heartbeat"' in app_js
    assert "function syncOpenPopup" in app_js
    assert "syncOpenPopup();" in app_js
