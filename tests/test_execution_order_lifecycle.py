from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.exchange_adapter import PaperExchangeAdapter
from execution.fill_handler import FillHandler
from execution.order_models import OrderIntent
from execution.order_router import OrderRouter
from execution.order_store import OrderStore


def test_order_router_paper_path_records_filled_order(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "order_state.json", tmp_path / "execution_journal.jsonl")
    router = OrderRouter(store, PaperExchangeAdapter(), FillHandler(store))

    result = router.route_intent(
        OrderIntent(
            symbol="BTCUSDT",
            action="ENTER_LONG",
            side="LONG",
            quantity=0.25,
            signal_reason="unit_test",
            metadata={"reference_price": 101.5},
        ),
        ts=123.0,
        previous_position=0,
    )

    assert result["accepted"] is True
    assert result["status"] == "filled"
    assert result["fill"]["price"] == 101.5

    orders = store.list_orders()
    assert len(orders) == 1
    assert orders[0].status == "filled"
    assert orders[0].filled_quantity == 0.25
    assert orders[0].remaining_quantity == 0.0


def test_live_controller_initializes_order_subsystem(tmp_path: Path) -> None:
    from runners.live_trading_controller import LiveTradingController
    import runners.live_trading_controller as live_module

    class _DummyMonitor:
        def __init__(self, paths):
            pass

    class _DummyControlStore:
        def __init__(self, paths):
            pass
        def pause_state(self):
            return {"paused": False, "pause_reason": ""}

    class _DummyRiskEngine:
        def __init__(self, paths):
            pass

    class _DummyLogic:
        def __init__(self, paths):
            pass

    live_module.MonitorPublisher = _DummyMonitor
    live_module.ControlPlaneStore = _DummyControlStore
    live_module.RiskEngine = _DummyRiskEngine
    live_module.TradeDecisionLogic = _DummyLogic

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths = SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", market_poll_seconds=1.0),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            exchange=SimpleNamespace(position_reconciliation_enabled=False),
        ),
        runtime_dir=runtime_dir,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        tradable_symbols_path=runtime_dir / "tradable_symbols.yaml",
    )

    controller = LiveTradingController(paths, registry=SimpleNamespace(), signal_engine=SimpleNamespace(load_symbols=lambda: [], logic=None))
    assert controller.order_store is not None
    assert controller.fill_handler is not None
    assert controller.order_router is not None


def test_monitor_publisher_tracks_authoritative_paper_execution_surface(tmp_path: Path) -> None:
    from monitoring.publisher import MonitorPublisher

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    paths = SimpleNamespace(runtime_dir=runtime_dir, project_root=tmp_path / "repo")

    publisher = MonitorPublisher(paths)
    publisher.update_paper_execution_runtime(
        open_positions=[{"symbol": "BTCUSDT", "side": "LONG", "quantity": 0.25}],
        recent_decisions=[{"symbol": "BTCUSDT", "action": "ENTER_LONG"}],
        summary={"equity_usd": 1000.5},
    )

    snapshot = publisher.snapshot()
    assert snapshot["paper_execution_runtime"]["execution_surface"] == "paper_forward_execution"
    assert snapshot["paper_execution_runtime"]["authoritative_for_runtime_positions"] is True
    assert snapshot["paper_execution_runtime"]["open_positions"][0]["symbol"] == "BTCUSDT"


def test_portfolio_state_execution_authority_tracks_closed_trade_summary(tmp_path: Path) -> None:
    from execution.portfolio_state import PortfolioStateStore

    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()

    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100.0,
        ts=1000.0,
        position_plan={"quantity": 2.0, "notional_usd": 200.0, "stop_loss": 95.0, "take_profit": 110.0},
    )
    assert "BTCUSDT" in state.positions

    report = store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="EXIT_LONG",
        price=106.0,
        ts=1100.0,
        position_plan={"exit_reason": "unit_test_take_profit"},
    )

    assert report["realized_pnl_usd"] == 12.0
    assert "BTCUSDT" not in state.positions
    assert len(state.closed_trades) == 1
    assert state.closed_trades[0].exit_reason == "unit_test_take_profit"

    summary = store.execution_summary(state)
    assert summary["open_positions"] == 0
    assert summary["completed_count"] == 1
    assert summary["wins"] == 1
    assert summary["losses"] == 0
    assert summary["total_pnl"] == 12.0

    store.save(state)
    reloaded = store.load()
    assert len(reloaded.closed_trades) == 1
    assert store.execution_summary(reloaded)["completed_count"] == 1


def test_portfolio_state_load_does_not_resurrect_closed_positions(tmp_path: Path) -> None:
    from execution.portfolio_state import PortfolioStateStore
    import json

    state_path = tmp_path / "portfolio_state.json"
    state_path.write_text(
        json.dumps(
            {
                "positions": {
                    "ORCAUSDT": {
                        "symbol": "ORCAUSDT",
                        "side": "LONG",
                        "quantity": 10.0,
                        "notional_usd": 100.0,
                        "entry_price": 1.37,
                        "stop_loss": 1.31,
                        "take_profit": 1.45,
                        "opened_at": 1000.0,
                        "updated_at": 1010.0,
                        "unrealized_pnl_usd": 0.0,
                    },
                    "BTCUSDT": {
                        "symbol": "BTCUSDT",
                        "side": "LONG",
                        "quantity": 0.01,
                        "notional_usd": 700.0,
                        "entry_price": 70000.0,
                        "stop_loss": 69000.0,
                        "take_profit": 72000.0,
                        "opened_at": 2000.0,
                        "updated_at": 2010.0,
                        "unrealized_pnl_usd": 0.0,
                    },
                },
                "closed_trades": [
                    {
                        "symbol": "ORCAUSDT",
                        "side": "LONG",
                        "quantity": 10.0,
                        "notional_usd": 100.0,
                        "entry_price": 1.37,
                        "exit_price": 1.31,
                        "opened_at": 1000.0,
                        "closed_at": 1500.0,
                        "pnl_usd": -6.0,
                        "exit_reason": "SL_HIT",
                        "mode": "paper",
                    }
                ],
                "realized_pnl_usd": -6.0,
                "daily_realized_pnl_usd": -6.0,
                "last_equity_usd": 994.0,
                "peak_equity_usd": 1000.0,
            }
        ),
        encoding="utf-8",
    )

    store = PortfolioStateStore(state_path, starting_equity_usd=1000.0)
    state = store.load()

    assert "ORCAUSDT" not in state.positions
    assert "BTCUSDT" in state.positions
    assert len(state.closed_trades) == 1
    assert store.execution_summary(state)["open_positions"] == 1
    assert store.execution_summary(state)["completed_count"] == 1
