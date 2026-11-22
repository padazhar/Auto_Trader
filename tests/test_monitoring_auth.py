from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from monitoring.auth import authenticate_monitoring_token, resolve_monitoring_token, websocket_token_required
from monitoring.web_app import create_app


def _paths(tmp_path: Path, token: str = "secret-token") -> SimpleNamespace:
    project_root = tmp_path / "repo"
    runtime_dir = project_root / "data" / "runtime"
    model_storage_dir = project_root / "data" / "model_storage"
    monitoring_static_dir = project_root / "monitoring_static"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    monitoring_static_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "app.js", "styles.css"):
        (monitoring_static_dir / name).write_text("stub", encoding="utf-8")
    return SimpleNamespace(
        project_root=project_root,
        runtime_dir=runtime_dir,
        model_storage_dir=model_storage_dir,
        monitoring_static_dir=monitoring_static_dir,
        snapshot_path=runtime_dir / "monitor_snapshot.json",
        events_path=runtime_dir / "monitor_events.json",
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        symbol_state_path=runtime_dir / "symbol_state.json",
        config=SimpleNamespace(
            monitoring=SimpleNamespace(auth_token=token, host="127.0.0.1", port=9000),
            runtime=SimpleNamespace(execution_mode="paper"),
            exchange=SimpleNamespace(position_reconciliation_enabled=False, require_reconciliation_for_live=True),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            system=SimpleNamespace(environment="development"),
            notifications=SimpleNamespace(telegram=SimpleNamespace(enabled=False, bot_token="", chat_id="")),
            telegram=SimpleNamespace(enabled=False, bot_token="", chat_id=""),
        ),
    )


class _DummyWebSocket:
    def __init__(self, headers=None, query_params=None):
        self.headers = headers or {}
        self.query_params = query_params or {}


def test_monitoring_auth_is_disabled_for_all_modes(tmp_path: Path) -> None:
    paths = _paths(tmp_path, token="1268118")
    assert resolve_monitoring_token(paths) is None
    assert authenticate_monitoring_token("bad", "anything") is True


def test_monitoring_api_is_accessible_without_any_token(tmp_path: Path) -> None:
    paths = _paths(tmp_path, token="1268118")
    app = create_app(paths)
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200


def test_monitoring_login_endpoint_reports_auth_disabled(tmp_path: Path) -> None:
    paths = _paths(tmp_path, token="1268118")
    app = create_app(paths)
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"token": "wrong"})
    assert response.status_code == 200
    assert response.json()["auth_required"] is False


def test_websocket_auth_bypass_returns_without_error(tmp_path: Path) -> None:
    paths = _paths(tmp_path, token="1268118")
    websocket = _DummyWebSocket()
    import asyncio
    asyncio.run(websocket_token_required(websocket, paths))
