from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from execution.portfolio_state import ClosedTradeState, PortfolioState, PortfolioStateStore, PositionState, PositionStateInvariantError
from runners.live_trading_controller import LiveTradingController


def test_portfolio_store_removes_position_closed_after_open(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = PortfolioState(
        positions={
            "BTCUSDT": PositionState(
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                notional_usd=100.0,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                opened_at=10.0,
                updated_at=10.0,
            )
        },
        closed_trades=[
            ClosedTradeState(
                symbol="BTCUSDT",
                side="LONG",
                quantity=1.0,
                notional_usd=100.0,
                entry_price=100.0,
                exit_price=101.0,
                opened_at=10.0,
                closed_at=20.0,
                pnl_usd=1.0,
                exit_reason="test_close",
                mode="paper_local",
            )
        ],
    )

    store.save(state)
    loaded = store.load()

    assert loaded.positions == {}
    assert len(loaded.closed_trades) == 1


def test_open_position_snapshot_uses_portfolio_authority(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = PortfolioState(
        positions={
            "ETHUSDT": PositionState(
                symbol="ETHUSDT",
                side="SHORT",
                quantity=2.0,
                notional_usd=250.0,
                entry_price=2500.0,
                stop_loss=2600.0,
                take_profit=2400.0,
                opened_at=30.0,
                updated_at=31.0,
            )
        },
        authority_mode="paper_local",
    )

    snapshot = store.open_position_snapshot(state)

    assert snapshot["ETHUSDT"]["side"] == "SHORT"
    assert snapshot["ETHUSDT"]["source"] == "portfolio_state"
    assert snapshot["ETHUSDT"]["state_owner"] == "live_trading_controller"



def test_live_controller_passes_authoritative_open_position_snapshot_to_signal_engine() -> None:
    captured: dict[str, object] = {}

    class _SignalEngine:
        def compute_decision(self, symbol, open_snapshot, *, paused=False, pause_reason=""):
            captured["symbol"] = symbol
            captured["snapshot"] = open_snapshot
            captured["paused"] = paused
            return SimpleNamespace(action="HOLD")

    controller = LiveTradingController.__new__(LiveTradingController)
    controller.signal_engine = _SignalEngine()
    controller.logic = SimpleNamespace(decide=lambda *args, **kwargs: None)
    controller._log_debug = lambda *args, **kwargs: None

    authoritative_snapshot = {"ETHUSDT": {"side": "SHORT", "source": "portfolio_state"}}
    controller._compute_decision("ETHUSDT", "15m", -1, authoritative_snapshot)

    assert captured["symbol"] == "ETHUSDT"
    assert captured["snapshot"] == authoritative_snapshot
    assert captured["paused"] is False


def test_portfolio_store_rejects_duplicate_direct_entry_without_overwriting_position(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()

    store.apply_execution(
        state,
        symbol="ZECUSDT",
        action="ENTER_SHORT",
        price=40.0,
        ts=100.0,
        position_plan={"quantity": 5.0, "notional_usd": 200.0, "stop_loss": 41.0, "take_profit": 38.0},
    )
    original = state.positions["ZECUSDT"]

    with pytest.raises(PositionStateInvariantError, match="direct_entry_rejected_open_position:ZECUSDT:SHORT"):
        store.apply_execution(
            state,
            symbol="ZECUSDT",
            action="ENTER_SHORT",
            price=39.0,
            ts=110.0,
            position_plan={"quantity": 5.0, "notional_usd": 195.0, "stop_loss": 40.0, "take_profit": 37.0},
        )

    retained = state.positions["ZECUSDT"]
    assert retained.quantity == original.quantity
    assert retained.notional_usd == original.notional_usd
    assert retained.entry_price == original.entry_price
    assert retained.opened_at == original.opened_at


def test_portfolio_store_persists_execution_integrity_failure_status(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()

    store.record_execution_integrity_failure(
        state,
        reason="execution_state_integrity_failure:ZECUSDT:filled_without_state",
        ts=125.0,
    )
    store.save(state)

    restored = store.load()
    assert restored.execution_integrity_status == "blocked"
    assert restored.execution_integrity_error == "execution_state_integrity_failure:ZECUSDT:filled_without_state"
    assert restored.execution_integrity_failure_at == 125.0


def test_live_controller_runtime_gate_stays_blocked_after_execution_integrity_failure() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.control_store = SimpleNamespace(
        runtime_gate=lambda **kwargs: {"blocked": False, "status": "running", "trading_enabled": True}
    )
    state = PortfolioState(
        execution_integrity_status="blocked",
        execution_integrity_error="execution_state_integrity_failure:ZECUSDT:filled_without_state",
        execution_integrity_failure_at=125.0,
    )

    gate = controller._runtime_gate(state, SimpleNamespace(triggered=False), {"authoritative": True})

    assert gate["blocked"] is True
    assert gate["status"] == "blocked"
    assert gate["trading_enabled"] is False
    assert gate["reason"] == "execution_state_integrity_failure:ZECUSDT:filled_without_state"


def test_live_controller_integrity_failure_persists_pause_and_runtime_block(tmp_path: Path) -> None:
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()
    pauses: list[dict[str, object]] = []
    events: list[tuple[object, ...]] = []

    controller = LiveTradingController.__new__(LiveTradingController)
    controller.portfolio_store = store
    controller.control_store = SimpleNamespace(
        set_pause=lambda **kwargs: pauses.append(kwargs),
        runtime_gate=lambda **kwargs: {"blocked": False, "status": "running", "trading_enabled": True},
    )
    controller._call_monitor = lambda *args, **kwargs: events.append((args, kwargs))
    controller.logger = SimpleNamespace(error=lambda *args, **kwargs: None)

    gate = controller._block_on_execution_integrity_failure(
        state,
        symbol="ZECUSDT",
        exc=PositionStateInvariantError("direct_entry_rejected_open_position:ZECUSDT:SHORT"),
        ts=135.0,
        circuit_state=SimpleNamespace(triggered=False),
        authority={"authoritative": True},
    )

    restored = store.load()
    assert restored.execution_integrity_status == "blocked"
    assert pauses and pauses[0]["paused"] is True
    assert pauses[0]["updated_by"] == "live_trading_controller"
    assert gate["blocked"] is True
    assert gate["trading_enabled"] is False
    assert events and events[0][0][1] == "execution_state_integrity_failure"
