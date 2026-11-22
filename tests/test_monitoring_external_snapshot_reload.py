from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from monitoring.publisher import MonitorPublisher


def _paths(tmp_path: Path) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=tmp_path,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        live_backtest_trades_path=tmp_path / "data" / "live_backtest_trades.csv",
        runtime_dir=runtime_dir,
        monitoring_state_path=runtime_dir / "monitoring_state.json",
        monitoring_events_path=runtime_dir / "monitoring_events.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        registry_path=runtime_dir / "training_registry.json",
        symbol_state_path=runtime_dir / "symbol_state.json",
        market_latest_snapshot_path=runtime_dir / "market" / "latest.json",
    )


def test_snapshot_and_events_reload_external_disk_state(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)

    paths.monitoring_state_path.write_text(json.dumps({
        "updated_at": 123.0,
        "system": {"launcher_mode": "all"},
        "health": {"status": "ok"},
        "controllers": {"live_trading": {"status": "running"}},
    }), encoding="utf-8")
    paths.monitoring_events_path.write_text(json.dumps([{
        "ts": 123.0, "level": "INFO", "component": "test", "category": "external", "message": "hello", "data": {}, "alert": False
    }]), encoding="utf-8")

    snapshot = publisher.snapshot()
    events = publisher.events()

    assert snapshot["system"]["launcher_mode"] == "all"
    assert snapshot["controllers"]["live_trading"]["status"] == "running"
    assert events[0]["message"] == "hello"


def test_publisher_summarizes_decision_journal_contract_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)

    rows = [
        {"ts": 200.0, "symbol": "BTCUSDT", "journal_writer": "live_trading_controller", "decision_stage": "execution", "action": "ENTER_LONG"},
        {"ts": 100.0, "symbol": "ETHUSDT", "journal_writer": "signal_engine", "decision_stage": "signal_generation", "action": "HOLD"},
    ]
    with paths.decision_journal_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    snapshot = publisher.snapshot()
    summary = snapshot["decision_journal"]

    assert summary["entries"] == 2
    assert summary["latest_writer"] == "live_trading_controller"
    assert summary["latest_stage"] == "execution"
    assert summary["writers"]["live_trading_controller"] == 1
    assert summary["writers"]["signal_engine"] == 1
    assert summary["stages"]["execution"] == 1
    assert summary["stages"]["signal_generation"] == 1


def _write_ready_price_snapshot(paths: SimpleNamespace) -> None:
    paths.market_latest_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    paths.market_latest_snapshot_path.write_text(json.dumps({
        "created_at_ms": 1,
        "active_symbols": ["BTCUSDT"],
        "price_symbols": ["BTCUSDT"],
        "expected_symbol_count": 1,
        "priced_symbol_count": 1,
        "missing_symbols": [],
        "all_symbols_priced": True,
        "symbols": {"BTCUSDT": {"price": 100.0, "price_source": "test", "price_ts_ms": 9999999999999}},
    }), encoding="utf-8")


def test_step6_persisted_health_is_not_green_when_trading_readiness_is_blocked(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.symbol_state_path.write_text(json.dumps({
        "active_symbols": ["BTCUSDT"],
        "execution_eligible_symbols": [],
        "target_active_symbols": 1,
        "active_symbols_source": "symbol_state",
    }), encoding="utf-8")
    _write_ready_price_snapshot(paths)
    publisher = MonitorPublisher(paths)

    publisher.update_signal_runtime(recent_readiness={
        "execution_eligible_count": 0,
        "blocked_count": 1,
        "rows": [],
    })
    publisher.heartbeat("live_trading", status="healthy", details={})

    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    assert persisted["health"]["status"] == "degraded"
    assert persisted["health"]["controller_status"] == "ok"
    assert persisted["trading_readiness"]["status"] == "blocked"
    assert "execution_eligible_symbols:0/1" in persisted["trading_readiness"]["blockers"]
    assert persisted["market_data"]["symbol_counts"]["ACTIVE"] == 1


def test_step6_execution_integrity_failure_persists_blocked_health(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    reason = "execution_state_integrity_failure:BTCUSDT:duplicate_entry"

    publisher.update_paper_execution_runtime(
        open_positions=[],
        recent_decisions=[],
        summary={"execution_integrity_status": "blocked", "execution_integrity_error": reason},
        runtime_gate={
            "blocked": True,
            "status": "blocked",
            "execution_integrity_status": "blocked",
            "execution_integrity_error": reason,
        },
    )
    publisher.heartbeat("live_trading", status="healthy", details={})

    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    assert persisted["health"]["status"] == "blocked"
    assert persisted["health"]["controller_status"] == "ok"
    assert persisted["health"]["errors"] >= 1
    assert persisted["trading_readiness"]["execution_integrity"]["blocked"] is True
    assert "execution_integrity_blocked" in persisted["trading_readiness"]["blockers"][0]


def test_step6_empty_monitor_registry_restores_canonical_registry_truth(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.registry_path.write_text(json.dumps({
        "rf:BTCUSDT:15m": {
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "status": "success",
            "execution_ready": True,
            "registry_truth_status": "execution_ready",
        }
    }), encoding="utf-8")
    publisher = MonitorPublisher(paths)

    snapshot = publisher.snapshot()

    assert snapshot["training_registry"]["projection_source"] == "canonical_registry_fallback"
    assert snapshot["training_registry"]["counts"]["success"] == 1
    assert snapshot["training_registry"]["diagnostic_counts"]["promoted"] == 1
    assert snapshot["training_registry"]["entries"][0]["symbol"] == "BTCUSDT"


def test_step6_control_plane_integrity_gate_also_blocks_overall_health(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    publisher = MonitorPublisher(paths)
    reason = "execution_state_integrity_failure:ETHUSDT:ledger_mismatch"

    publisher.update_control_plane({
        "runtime_gate": {
            "blocked": True,
            "status": "blocked",
            "execution_integrity_status": "blocked",
            "execution_integrity_error": reason,
        }
    })
    publisher.heartbeat("live_trading", status="healthy", details={})

    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    assert persisted["health"]["status"] == "blocked"
    assert persisted["trading_readiness"]["execution_integrity"]["source"] == "control_plane.runtime_gate"


def _independent_publisher(paths: SimpleNamespace) -> MonitorPublisher:
    """Return a publisher with independent in-memory state for process-role tests."""
    key = MonitorPublisher._instance_key(paths)
    with MonitorPublisher._instances_lock:
        MonitorPublisher._instances.pop(key, None)
    return MonitorPublisher(paths)


def test_step62_cross_process_writers_preserve_runtime_and_training_sections(tmp_path: Path) -> None:
    """Trainer updates must not overwrite runtime sections and vice versa."""
    paths = _paths(tmp_path)
    runtime_publisher = _independent_publisher(paths)
    trainer_publisher = _independent_publisher(paths)

    runtime_publisher.heartbeat("market_data", status="running", details={"mode": "fast"})
    runtime_publisher.update_market_data(symbols_total=2, symbol_counts={"ACTIVE": 2, "BLOCKED": 0, "TIMED_OUT": 0})
    runtime_publisher.update_paper_execution_runtime(
        open_positions=[{"symbol": "BTCUSDT", "side": "LONG", "entry_price": 100.0}],
        recent_decisions=[],
        summary={"open_position_count": 1, "unrealized_pnl_usd": 1.25},
    )

    # This instance was created before runtime published.  Without a serialised
    # disk reload before mutation, its full-state write erases runtime truth.
    trainer_publisher.heartbeat("tech_rev", status="running", details={"queued_jobs": 1})
    trainer_publisher.update_training_runtime(current_rf={"symbol": "ETHUSDT", "interval": "15m"})
    trainer_publisher.update_registry(
        counts={"success": 1, "running": 0},
        entries=[{"model_type": "rf", "symbol": "ETHUSDT", "interval": "15m", "status": "success"}],
    )

    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    assert persisted["controllers"]["market_data"]["status"] == "running"
    assert persisted["controllers"]["tech_rev"]["status"] == "running"
    assert persisted["paper_execution_runtime"]["open_position_count"] == 1
    assert persisted["training_runtime"]["current_rf"]["symbol"] == "ETHUSDT"
    assert persisted["training_registry"]["counts"]["success"] == 1

    # The stale runtime-side instance must likewise retain trainer truth on a
    # later market update instead of overwriting it from its old local copy.
    runtime_publisher.update_market_connectivity(status="connected", source="websocket")
    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    assert persisted["training_runtime"]["current_rf"]["symbol"] == "ETHUSDT"
    assert persisted["training_registry"]["counts"]["success"] == 1
    assert persisted["market_data"]["connectivity"]["status"] == "connected"


def test_step62_cross_process_event_writers_do_not_drop_prior_events(tmp_path: Path) -> None:
    """Events from separate runtime roles must survive alternating writes."""
    paths = _paths(tmp_path)
    runtime_publisher = _independent_publisher(paths)
    trainer_publisher = _independent_publisher(paths)

    runtime_publisher.publish_event(
        level="INFO", component="runtime", category="heartbeat", message="runtime-ready"
    )
    trainer_publisher.publish_event(
        level="INFO", component="trainer", category="training", message="trainer-ready"
    )
    runtime_publisher.publish_event(
        level="INFO", component="runtime", category="market", message="price-ready"
    )

    events = json.loads(paths.monitoring_events_path.read_text(encoding="utf-8"))
    assert [event["message"] for event in events] == ["runtime-ready", "trainer-ready", "price-ready"]


def test_step2_publisher_uses_paper_execution_ledger_and_keeps_backtest_reference_separate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.portfolio_state_path.write_text(json.dumps({
        "realized_pnl_usd": -4.0,
        "closed_trades": [{"symbol": "BTCUSDT", "side": "LONG", "pnl_usd": -4.0, "closed_at": 10}],
    }), encoding="utf-8")
    paths.live_backtest_trades_path.parent.mkdir(parents=True, exist_ok=True)
    paths.live_backtest_trades_path.write_text(
        "symbol,side,mode,entry_time,entry_price,exit_time,exit_price,pnl_usd,pnl_pct,notional_usd,exit_reason\n"
        "ETHUSDT,LONG,paper,1,100,2,102,50.0,1,500,tp\n",
        encoding="utf-8",
    )
    publisher = MonitorPublisher(paths)

    publisher.update_paper_execution_runtime(
        open_positions=[], recent_decisions=[],
        summary={"source": "csv_closed_trade_ledger", "completed_count": 1, "total_pnl": 50.0},
        trade_history=[{"symbol": "ETHUSDT", "pnl_usd": 50.0}],
    )

    persisted = json.loads(paths.monitoring_state_path.read_text(encoding="utf-8"))
    runtime = persisted["paper_execution_runtime"]
    assert runtime["execution_summary"]["source"] == "paper_execution_ledger"
    assert runtime["execution_summary"]["total_pnl"] == -4.0
    assert runtime["trade_history"][0]["symbol"] == "BTCUSDT"
    assert runtime["trade_result_source"]["backtest_reference"]["authoritative_for_paper_execution"] is False


def test_symbol_universe_projection_uses_signal_readiness_when_lifecycle_is_stale(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.symbol_state_path.write_text(json.dumps({
        "active_symbols": ["BTCUSDT", "ETHUSDT"],
        "execution_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
        "symbols": {
            "BTCUSDT": {"status": "ACTIVE", "activation_stage": "execution_active", "data_ready": True, "execution_eligible": True},
            "ETHUSDT": {"status": "ACTIVE", "activation_stage": "execution_active", "data_ready": True, "execution_eligible": True},
        },
        "target_active_symbols": 2,
        "ranked_candidates": ["BTCUSDT", "ETHUSDT"],
        "ranked_universe_size": 20,
        "updated_at": 100.0,
    }), encoding="utf-8")
    paths.monitoring_state_path.write_text(json.dumps({
        "signal_runtime": {
            "readiness": {
                "ready_count": 1,
                "blocked_count": 1,
                "execution_eligible_count": 1,
                "execution_eligible_symbols": ["BTCUSDT"],
                "execution_readiness_matrix": [
                    {"symbol": "BTCUSDT", "ready": True, "reason": "anchor_ready"},
                    {"symbol": "ETHUSDT", "ready": False, "reason": "anchor_training_incomplete:gbm:registry_status:idle"},
                ],
            }
        },
        "market_data": {"runtime_price_snapshot": {"ready": True}},
        "health": {"status": "ok"},
    }), encoding="utf-8")

    snapshot = MonitorPublisher(paths).snapshot()
    universe = snapshot["market_data"]["symbol_universe"]

    assert universe["active_symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert universe["execution_eligible_symbols"] == ["BTCUSDT"]
    assert universe["execution_eligible_count"] == 1
    assert snapshot["trading_readiness"]["execution_eligible_count"] == 1
    assert "execution_eligible_symbols:1/2" in snapshot["trading_readiness"]["blockers"]
