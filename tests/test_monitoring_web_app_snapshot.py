import json
from types import SimpleNamespace

from monitoring.web_app import _normalize_snapshot


def test_normalize_snapshot_derives_health_uptime_training_and_execution_messages():
    snapshot = {
        "system": {
            "execution_mode": "paper",
            "launcher_mode": "all",
            "started_at": 1000.0,
        },
        "controllers": {
            "market_data": {"status": "running", "details": {"mode": "slow"}},
        },
        "training_registry": {
            "entries": [
                {
                    "model_type": "rf",
                    "symbol": "BTCUSDT",
                    "interval": "15m",
                    "status": "running",
                    "started_at": 1010.0,
                }
            ],
            "counts_by_model": {"rf": {"running": 1}},
        },
        "paper_execution_runtime": {
            "recent_decisions": [
                {"symbol": "BTCUSDT", "reason": "price_unavailable", "mode": "paper"},
                {"symbol": "ETHUSDT", "reason": "price_unavailable", "mode": "paper"},
            ],
            "summary": {},
            "open_positions": [],
            "trade_history": [],
        },
    }

    normalized = _normalize_snapshot(snapshot)

    assert normalized["status"] == "running"
    assert normalized["health"]["status"] == "running"
    assert normalized["system"]["uptime_sec"] >= 0.0
    assert normalized["training"]["rf_running"] == 1
    assert normalized["training"]["current_rf"] == "BTCUSDT 15m"
    assert normalized["training"]["rf_progress"] == 1.0
    assert normalized["execution_runtime"]["recent_messages"]
    assert "price_unavailable" in normalized["execution_runtime"]["recent_messages"][0] or any(
        "price_unavailable" in message for message in normalized["execution_runtime"]["recent_messages"]
    )


def test_normalize_snapshot_does_not_use_inference_as_paper_execution_authority():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "inference": {
                "open_positions": {
                    "APEUSDT": {
                        "side": "LONG",
                        "entry_price": 0.17935,
                        "latest_price": 0.18100,
                        "price_source": "runtime_market_price",
                        "updated_at": 1234.0,
                        "unrealized_pnl_usd": 1.23,
                    }
                },
                "pnl_summary": {"unrealized_pnl_usd": 1.23},
            },
        }
    )

    assert normalized["execution_runtime"]["open_positions"] == []
    assert normalized["execution_runtime"]["execution_summary"]["completed_count"] == 0
    assert normalized["execution_runtime"]["execution_summary"]["total_pnl"] == 0.0


def test_normalize_snapshot_prefers_paper_execution_runtime_over_inference_fallback():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "paper_execution_runtime": {
                "open_positions": [],
                "trade_history": [
                    {"symbol": "BTCUSDT", "side": "LONG", "entry_price": 100.0, "exit_price": 105.0, "pnl_usd": 5.0}
                ],
                "summary": {"completed_count": 1, "wins": 1, "losses": 0, "total_pnl": 5.0},
            },
            "inference": {
                "open_positions": {
                    "BTCUSDT": {"side": "LONG", "entry_price": 100.0, "latest_price": 104.0}
                },
                "pnl_summary": {"completed_count": 0, "total_pnl": 0.0},
            },
        }
    )

    assert normalized["execution_runtime"]["open_positions"] == []
    assert normalized["execution_runtime"]["summary"]["completed_count"] == 1
    assert normalized["execution_runtime"]["summary"]["total_pnl"] == 5.0
    assert normalized["execution_runtime"]["trade_history"][0]["symbol"] == "BTCUSDT"


def test_step12_trading_readiness_downgrades_false_green_health_and_filters_active_universe():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "controllers": {"market_data": {"status": "running", "details": {}}},
            "health": {"status": "ok", "warnings": 0, "errors": 0},
            "trading_readiness": {
                "status": "blocked",
                "ready": False,
                "summary": "anchor_model_readiness_blockers:3",
                "blockers": ["anchor_model_readiness_blockers:3"],
            },
            "market_data": {
                "universe": {"symbols": ["BTCUSDT", "ETHUSDT", "OLDUSDT"], "source": "raw"},
                "symbol_universe": {
                    "active_symbols": ["BTCUSDT", "ETHUSDT"],
                    "target_active_symbols": 2,
                    "source": "symbol_state",
                },
            },
        }
    )

    assert normalized["health"]["status"] == "degraded"
    assert normalized["health"]["warnings"] >= 1
    assert normalized["market_data"]["universe"] == ["BTCUSDT", "ETHUSDT"]
    assert normalized["market_data"]["universe_source"] == "symbol_state"


def test_step2_execution_summary_contract_derives_from_trade_history_when_summary_missing():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "paper_execution_runtime": {
                "open_positions": [{"symbol": "BTCUSDT", "side": "LONG"}],
                "trade_history": [
                    {"symbol": "BTCUSDT", "side": "LONG", "pnl_usd": 4.0},
                    {"symbol": "ETHUSDT", "side": "SHORT", "pnl_usd": -1.5},
                ],
                "summary": {},
            },
        }
    )

    summary = normalized["execution_runtime"]["execution_summary"]
    assert normalized["execution_runtime"]["summary_contract_version"] == "execution_summary.v1"
    assert summary["open_positions"] == 1
    assert summary["completed_count"] == 2
    assert summary["trades"] == 2
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["total_pnl"] == 2.5
    assert summary["win_rate"] == 0.5


def test_step2_execution_card_does_not_fall_back_to_inference_when_runtime_empty():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "paper_execution_runtime": {"open_positions": [], "trade_history": [], "summary": {}},
            "inference": {"pnl_summary": {"completed_count": 99, "wins": 99, "total_pnl": 999.0}},
        }
    )

    summary = normalized["execution_runtime"]["execution_summary"]
    assert summary["completed_count"] == 0
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["total_pnl"] == 0.0


def test_step5_normalize_snapshot_exposes_control_plane_lineage_at_root():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "control_plane": {
                "decision_lineage_preview": [
                    {"symbol": "BTCUSDT", "action": "ENTER_LONG", "lineage": {"decision_id": "dec-1"}}
                ]
            },
        }
    )

    assert normalized["decision_lineage_preview"][0]["lineage"]["decision_id"] == "dec-1"
    assert normalized["control_plane"]["decision_lineage_preview"][0]["lineage"]["decision_id"] == "dec-1"


def test_step6_execution_integrity_failure_blocks_overall_health_while_controllers_remain_live():
    normalized = _normalize_snapshot(
        {
            "system": {"execution_mode": "paper"},
            "controllers": {"live_trading": {"status": "healthy", "details": {}}},
            "health": {"status": "ok", "warnings": 0, "errors": 0},
            "paper_execution_runtime": {
                "open_positions": [],
                "trade_history": [],
                "summary": {
                    "execution_integrity_status": "blocked",
                    "execution_integrity_error": "execution_state_integrity_failure:BTCUSDT:duplicate_entry",
                },
                "runtime_gate": {
                    "blocked": True,
                    "execution_integrity_status": "blocked",
                    "execution_integrity_error": "execution_state_integrity_failure:BTCUSDT:duplicate_entry",
                },
            },
        }
    )

    assert normalized["health"]["status"] == "blocked"
    assert normalized["health"]["errors"] >= 1
    assert normalized["health"]["controller_liveness_status"] == "running"
    assert normalized["health"]["execution_integrity"]["blocked"] is True


def test_step2_web_snapshot_never_overlays_backtest_pnl_on_paper_execution(tmp_path):
    runtime_dir = tmp_path / "data" / "runtime"
    runtime_dir.mkdir(parents=True)
    portfolio_path = runtime_dir / "portfolio_state.json"
    portfolio_path.write_text(json.dumps({
        "realized_pnl_usd": 3.5,
        "closed_trades": [{"symbol": "BTCUSDT", "side": "SHORT", "pnl_usd": 3.5, "closed_at": 10}],
    }), encoding="utf-8")
    backtest_path = tmp_path / "data" / "live_backtest_trades.csv"
    backtest_path.write_text(
        "symbol,side,mode,entry_time,entry_price,exit_time,exit_price,pnl_usd,pnl_pct,notional_usd,exit_reason\n"
        "ETHUSDT,LONG,paper,1,100,2,102,-99.0,-1,500,sl\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(project_root=tmp_path, portfolio_state_path=portfolio_path, live_backtest_trades_path=backtest_path)
    normalized = _normalize_snapshot({
        "system": {"execution_mode": "paper"},
        "paper_execution_runtime": {
            "open_positions": [],
            "trade_history": [{"symbol": "ETHUSDT", "pnl_usd": -99.0}],
            "summary": {"source": "csv_closed_trade_ledger", "completed_count": 1, "total_pnl": -99.0},
        },
    }, paths=paths)

    runtime = normalized["execution_runtime"]
    assert runtime["execution_summary"]["source"] == "paper_execution_ledger"
    assert runtime["execution_summary"]["total_pnl"] == 3.5
    assert runtime["trade_history"][0]["symbol"] == "BTCUSDT"
