from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from execution.portfolio_state import PortfolioStateStore
from monitoring.control_plane_store import ControlPlaneStore
from monitoring.web_app import create_app


def _paths(tmp_path: Path) -> SimpleNamespace:
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
            monitoring=SimpleNamespace(auth_token="", host="127.0.0.1", port=9000),
            runtime=SimpleNamespace(execution_mode="paper"),
            exchange=SimpleNamespace(position_reconciliation_enabled=False, require_reconciliation_for_live=True),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            system=SimpleNamespace(environment="development"),
            notifications=SimpleNamespace(telegram=SimpleNamespace(enabled=False, bot_token="", chat_id="")),
            telegram=SimpleNamespace(enabled=False, bot_token="", chat_id=""),
        ),
    )


def _seed_decision_journal(paths) -> None:
    rows = [
        {
            "ts": 1710000200.0,
            "symbol": "BTCUSDT",
            "interval": "15m",
            "action": "ENTER_LONG",
            "reason": "edge_positive",
            "score": 0.82,
            "components": {
                "freshness": {"ok": True, "age_seconds": 12.0},
                "shadow_candidate": {"available": True, "different_action": True, "action": "HOLD", "reason": "candidate_guarded"},
                "lineage": {"rf": {"candidate_id": "rf_incumbent_1"}},
            },
        },
        {
            "ts": 1710000100.0,
            "symbol": "ETHUSDT",
            "interval": "15m",
            "action": "HOLD",
            "reason": "stale_market_data",
            "score": 0.11,
            "components": {
                "freshness": {"ok": False, "reason": "stale_market_data", "age_seconds": 420.0},
                "shadow_candidate": {"available": True, "different_action": False, "action": "HOLD", "reason": "candidate_match"},
                "lineage": {"rf": {"candidate_id": "rf_incumbent_1"}},
            },
        },
    ]
    with paths.decision_journal_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_control_plane_summary_exposes_runtime_observability(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_decision_journal(paths)

    portfolio_store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    portfolio = portfolio_store.load()
    portfolio.realized_pnl_usd = 12.5
    portfolio.daily_realized_pnl_usd = 5.5
    portfolio.last_equity_usd = 1012.5
    portfolio_store.save(portfolio)

    store = ControlPlaneStore(paths)
    summary = store.summary()
    observability = summary["runtime_observability"]

    assert observability["freshness"]["fresh_entries"] == 1
    assert observability["freshness"]["stale_entries"] == 1
    assert observability["prediction_drift"]["shadow_available"] == 2
    assert observability["prediction_drift"]["shadow_divergence_rate"] == 0.5
    assert observability["runtime_performance"]["portfolio_runtime"]["equity_usd"] == 1012.5


def test_runtime_observability_api_returns_summary(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_decision_journal(paths)
    app = create_app(paths)
    client = TestClient(app)

    response = client.get("/api/runtime-observability")
    assert response.status_code == 200
    payload = response.json()
    assert "runtime_observability" in payload
    assert payload["runtime_observability"]["freshness"]["stale_entries"] == 1
    assert "runtime_gate" in payload


def test_normalize_snapshot_separates_backtest_from_authoritative_paper_execution() -> None:
    from monitoring.web_app import _normalize_snapshot

    snapshot = {
        "system": {"execution_mode": "paper"},
        "inference": {
            "open_positions": {
                "BTCUSDT": {"side": "LONG", "quantity": 0.25, "entry_price": 101.5},
            },
            "decisions": [{"symbol": "BTCUSDT", "action": "ENTER_LONG", "reason": "unit_test"}],
            "pnl_summary": {"equity_usd": 1001.0},
        },
        "backtest_runtime": {
            "open_positions": [],
            "recent_messages": [],
            "summary": {},
        },
    }

    normalized = _normalize_snapshot(snapshot)

    assert normalized["backtest"]["execution_surface"] == "historical_backtest"
    assert normalized["backtest"]["authoritative_for_runtime_positions"] is False
    assert normalized["backtest"]["open_positions"] == []

    assert normalized["paper_execution"]["execution_surface"] == "paper_forward_execution"
    assert normalized["paper_execution"]["authoritative_for_runtime_positions"] is True
    # Step 1 made paper execution state canonical; inference positions must not
    # resurrect open positions when paper_execution_runtime is absent/empty.
    assert normalized["paper_execution"]["open_positions"] == []
    assert normalized["execution_runtime"]["execution_surface"] == "paper_forward_execution"
    assert normalized["execution_runtime"]["authoritative_for_runtime_positions"] is True
    assert normalized["execution_runtime"]["open_position_count"] == 0
    assert normalized["execution_runtime"]["open_order_count"] == 0
    assert normalized["decision_journal"]["entries"] == 0
    assert normalized["decision_journal"]["writers"] == {}


def test_snapshot_api_exposes_authoritative_paper_execution_surface(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    app = create_app(paths)
    client = TestClient(app)

    response = client.get("/api/snapshot")
    assert response.status_code == 200
    payload = response.json()

    assert "paper_execution" in payload
    assert payload["paper_execution"]["execution_surface"] == "paper_forward_execution"
    assert payload["execution_runtime"]["execution_surface"] == "paper_forward_execution"


def test_normalize_snapshot_builds_live_execution_runtime_from_inference_and_authority() -> None:
    from monitoring.web_app import _normalize_snapshot

    snapshot = {
        "system": {"execution_mode": "live"},
        "control_plane": {
            "live_runtime": {
                "authoritative": True,
                "authority_mode": "live_exchange_reconciled",
                "authority_reason": "reconciled",
                "open_position_count": 1,
                "open_order_count": 2,
                "last_recovery_sources": ["exchange_open_orders"],
            }
        },
        "decision_journal": {
            "entries": 2,
            "latest_ts": 200.0,
            "latest_writer": "live_trading_controller",
            "latest_stage": "execution",
            "writers": {"live_trading_controller": 1, "signal_engine": 1},
            "stages": {"execution": 1, "signal_generation": 1},
            "recent": [{"ts": 200.0, "journal_writer": "live_trading_controller", "decision_stage": "execution"}],
        },
        "inference": {
            "open_positions": {
                "ETHUSDT": {"side": "SHORT", "quantity": 1.0, "entry_price": 2500.0},
            },
            "decisions": [{"symbol": "ETHUSDT", "action": "EXIT_SHORT", "reason": "unit_test"}],
            "pnl_summary": {"equity_usd": 1510.0},
        },
    }

    normalized = _normalize_snapshot(snapshot)

    assert normalized["execution_runtime"]["execution_surface"] == "live_forward_execution"
    assert normalized["execution_runtime"]["authoritative_for_runtime_positions"] is True
    assert normalized["execution_runtime"]["open_positions"][0]["symbol"] == "ETHUSDT"
    assert normalized["execution_runtime"]["open_position_count"] == 1
    assert normalized["execution_runtime"]["open_order_count"] == 2
    assert normalized["execution_runtime"]["last_recovery_sources"] == ["exchange_open_orders"]
    assert normalized["execution_runtime"]["summary"]["equity_usd"] == 1510.0
    assert normalized["execution_runtime"]["authority"]["authority_mode"] == "live_exchange_reconciled"
    assert normalized["decision_journal"]["latest_writer"] == "live_trading_controller"
    assert normalized["decision_journal"]["stages"]["execution"] == 1


def test_normalize_snapshot_preserves_signal_runtime_gate_and_recent_decisions() -> None:
    from monitoring.web_app import _normalize_snapshot

    snapshot = {
        "system": {"execution_mode": "paper"},
        "signal_runtime": {
            "latest_mode": "paper",
            "latest_symbol": "BTCUSDT",
            "latest_action": "HOLD",
            "recent_signals": [{"symbol": "BTCUSDT", "action": "HOLD"}],
            "recent_decisions": [{"symbol": "BTCUSDT", "action": "HOLD", "reason": "paused"}],
            "runtime_gate": {"paused": True, "pause_reason": "manual_pause"},
            "last_error": None,
        },
    }

    normalized = _normalize_snapshot(snapshot)

    assert normalized["signal_runtime"]["recent_decisions"][0]["reason"] == "paused"
    assert normalized["signal_runtime"]["runtime_gate"]["paused"] is True
    assert normalized["signal_generator"]["runtime_gate"]["pause_reason"] == "manual_pause"


def test_monitor_publisher_accepts_extended_signal_runtime_payload(tmp_path: Path) -> None:
    from monitoring.publisher import MonitorPublisher

    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    publisher.update_signal_runtime(
        latest_mode="paper",
        latest_symbol="BTCUSDT",
        latest_action="HOLD",
        recent_signals=[{"symbol": "BTCUSDT", "action": "HOLD"}],
        recent_decisions=[{"symbol": "BTCUSDT", "action": "HOLD", "reason": "paused"}],
        runtime_gate={"paused": True, "pause_reason": "manual_pause"},
        last_error="boom",
        source_priority=100,
        source_name="signal_generator_controller",
    )

    signal_runtime = publisher.snapshot()["signal_runtime"]
    assert signal_runtime["recent_decisions"][0]["reason"] == "paused"
    assert signal_runtime["runtime_gate"]["paused"] is True
    assert signal_runtime["last_error"] == "boom"


def test_normalize_snapshot_prefers_training_runtime_for_progress_fields() -> None:
    from monitoring.web_app import _normalize_snapshot

    snapshot = {
        "system": {"execution_mode": "paper"},
        "training": {"rf": {"status": "idle"}},
        "training_runtime": {
            "rf_progress": 55.0,
            "gbm_progress": 20.0,
            "dqn_progress": 0.0,
            "current_rf": "BTCUSDT",
            "current_gbm": "ETHUSDT",
            "current_dqn": None,
        },
    }

    normalized = _normalize_snapshot(snapshot)

    assert normalized["training"]["rf_progress"] == 55.0
    assert normalized["training"]["current_rf"] == "BTCUSDT"
    assert normalized["training"]["current_gbm"] == "ETHUSDT"


def test_monitor_publisher_tracks_cumulative_errors_and_controller_recovery(tmp_path: Path) -> None:
    from monitoring.publisher import MonitorPublisher

    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    publisher.publish_event(level="WARNING", component="unit", category="warn", message="warning")
    publisher.controller_error("training", "training failed")

    snapshot = publisher.snapshot()
    assert snapshot["runtime_event_counters"]["warnings_since_start"] >= 1
    assert snapshot["runtime_event_counters"]["errors_since_start"] >= 1
    assert snapshot["error_state"]["active_controller_errors"]["training"]["error"] == "training failed"

    publisher.controller_success("training")
    recovered = publisher.snapshot()["error_state"]
    assert "training" not in recovered["active_controller_errors"]
    assert recovered["recovered_controller_errors"][-1]["controller"] == "training"
