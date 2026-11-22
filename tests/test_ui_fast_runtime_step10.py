from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from monitoring.publisher import MonitorPublisher
from monitoring.web_app import create_app


def _paths(tmp_path: Path):
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    return SimpleNamespace(
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
        monitoring_runtime_dir=tmp_path / "runtime",
        monitoring_state_path=tmp_path / "runtime" / "monitoring_state.json",
        monitoring_events_path=tmp_path / "runtime" / "monitoring_events.json",
        monitor_snapshot_path=tmp_path / "runtime" / "monitoring_state.json",
        monitor_events_path=tmp_path / "runtime" / "monitoring_events.json",
        monitoring_static_dir=static_dir,
        decision_journal_path=tmp_path / "runtime" / "decision_journal.jsonl",
        symbol_state_path=tmp_path / "runtime" / "symbol_state.json",
        symbol_actions_path=tmp_path / "runtime" / "symbol_actions.json",
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                monitoring_fast_snapshot_interval_sec=1.0,
                monitoring_heavy_snapshot_interval_sec=5.0,
                execution_mode="paper",
            ),
            monitoring=SimpleNamespace(auth_token=""),
            system=SimpleNamespace(environment="dev"),
            symbols=SimpleNamespace(target_active_symbols=10),
        ),
    )


def test_fast_runtime_snapshot_excludes_heavy_registry_payload(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    publisher.update_paper_execution_runtime(
        execution_mode="paper",
        source_name="test",
        open_positions=[{"symbol": "BTCUSDT", "last_price": 100.0, "unrealized_pnl_usd": 1.5}],
        recent_decisions=[{"symbol": "BTCUSDT", "action": "HOLD"}],
        summary={"open_positions": 1, "unrealized_pnl_usd": 1.5},
        trade_history=[],
        authority={"authority_mode": "paper_local"},
        runtime_gate={"blocked": False},
    )
    publisher.update_training_runtime(running_jobs=[{"symbol": "BTCUSDT", "model_type": "rf"}])

    payload = publisher.fast_runtime_snapshot()

    assert payload["type"] == "fast_runtime"
    assert payload["execution_runtime"]["open_position_count"] == 1
    assert payload["market_data"].keys() == {"runtime_price_snapshot"}
    assert "training_registry" not in payload
    assert "model_versions" not in payload


def test_fast_runtime_api_preserves_execution_contract(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    publisher.update_paper_execution_runtime(
        execution_mode="paper",
        source_name="test",
        open_positions=[{"symbol": "ETHUSDT", "last_price": 2000.0, "unrealized_pnl_usd": -2.0}],
        recent_decisions=[],
        summary={"open_positions": 1, "unrealized_pnl_usd": -2.0},
        trade_history=[],
        authority={"authority_mode": "paper_local"},
        runtime_gate={"blocked": False},
    )

    import monitoring.web_app as web_app
    monkeypatch.setattr(web_app, "validate_runtime_secrets", lambda config: None)
    client = TestClient(create_app(paths))
    response = client.get("/api/runtime/fast")

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_runtime"]["open_position_count"] == 1
    assert payload["execution_runtime"]["open_positions"][0]["symbol"] == "ETHUSDT"
    assert "training_registry" not in payload
    assert payload["market_data"]["runtime_price_snapshot"] is not None


def test_ui_config_exposes_refresh_contract(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from monitoring.web_app import create_app

    paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(dashboard_refresh_ms=500, monitoring_fast_snapshot_interval_sec=0.5, monitoring_heavy_snapshot_interval_sec=3.0),
            monitoring=SimpleNamespace(embedded_in_all_mode=False, auth_token=""),
        ),
        monitoring_static_dir=tmp_path,
        decision_journal_path=tmp_path / "decision_journal.jsonl",
        project_root=tmp_path,
    )
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    monkeypatch.setattr("monitoring.web_app.validate_runtime_secrets", lambda config: None)
    monkeypatch.setattr("monitoring.web_app.token_required", lambda request, paths: None)
    monkeypatch.setattr("monitoring.web_app._write_monitoring_owner_marker", lambda paths: None)
    monkeypatch.setattr("monitoring.web_app._clear_monitoring_owner_marker", lambda paths: None)
    client = TestClient(create_app(paths))

    payload = client.get("/api/ui-config").json()

    assert payload["dashboard_refresh_ms"] == 500
    assert payload["fast_snapshot_interval_sec"] == 0.5
    assert payload["heavy_snapshot_interval_sec"] == 3.0
    assert payload["standalone_process"] is True
    assert payload["snapshot_only"] is True


def test_monitoring_websocket_watch_publishes_non_rendering_heartbeat():
    source = Path("monitoring/web_app.py").read_text(encoding="utf-8")

    assert '"type": "heartbeat"' in source
    assert "last_heartbeat_sent_at" in source
    assert "without repainting data" in source
