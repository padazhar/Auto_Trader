from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from execution.portfolio_state import PositionState
from runners.live_trading_controller import LiveTradingController


def test_open_position_row_uses_latest_mark_price_for_pnl() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.execution_mode = "paper"
    controller.paths = SimpleNamespace()

    position = PositionState(
        symbol="APEUSDT",
        side="LONG",
        quantity=100.0,
        notional_usd=17.0,
        entry_price=0.1700,
        stop_loss=0.1600,
        take_profit=0.1900,
        opened_at=1.0,
        updated_at=1.0,
    )

    row = controller._build_open_position_row(
        "APEUSDT",
        position,
        {"APEUSDT": 0.1800},
        {"APEUSDT": 1_700_000_000_000},
        {"APEUSDT": "runtime_market_price"},
        mode="paper",
    )

    assert row["last_price"] == 0.1800
    assert row["latest_price"] == 0.1800
    assert row["reference_price"] == 0.1800
    assert row["price_source"] == "runtime_market_price"
    assert round(row["estimated_pnl"], 8) == 1.0


def test_protective_exit_action_long_and_short() -> None:
    long_pos = PositionState(
        symbol="APEUSDT",
        side="LONG",
        quantity=1.0,
        notional_usd=10.0,
        entry_price=10.0,
        stop_loss=9.0,
        take_profit=12.0,
        opened_at=1.0,
        updated_at=1.0,
    )
    short_pos = PositionState(
        symbol="BTCUSDT",
        side="SHORT",
        quantity=1.0,
        notional_usd=10.0,
        entry_price=10.0,
        stop_loss=11.0,
        take_profit=8.0,
        opened_at=1.0,
        updated_at=1.0,
    )

    assert LiveTradingController._protective_exit_action(long_pos, 8.9) == ("EXIT_LONG", "protective_stop_loss")
    assert LiveTradingController._protective_exit_action(long_pos, 12.1) == ("EXIT_LONG", "protective_take_profit")
    assert LiveTradingController._protective_exit_action(short_pos, 11.1) == ("EXIT_SHORT", "protective_stop_loss")
    assert LiveTradingController._protective_exit_action(short_pos, 7.9) == ("EXIT_SHORT", "protective_take_profit")


def test_order_side_for_exit_actions_is_reduce_direction() -> None:
    assert LiveTradingController._order_side_for_action("EXIT_LONG", "LONG") == "SHORT"
    assert LiveTradingController._order_side_for_action("EXIT_SHORT", "SHORT") == "LONG"
    assert LiveTradingController._order_side_for_action("ENTER_LONG") == "LONG"
    assert LiveTradingController._order_side_for_action("ENTER_SHORT") == "SHORT"


def test_fast_execution_poll_used_only_when_positions_are_open() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                market_poll_seconds=10.0,
                execution_price_refresh_seconds=1.0,
                execution_price_max_age_seconds=3.0,
            )
        )
    )
    controller._log_debug = lambda *args, **kwargs: None

    controller.portfolio_store = SimpleNamespace(load=lambda: SimpleNamespace(positions={}))
    assert controller._next_loop_wait_seconds() == 10.0

    controller.portfolio_store = SimpleNamespace(load=lambda: SimpleNamespace(positions={"BTCUSDT": object()}))
    assert controller._next_loop_wait_seconds() == 1.0


def test_execution_price_max_age_prefers_tight_runtime_setting() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                price_max_age_seconds=180.0,
                execution_price_max_age_seconds=3.0,
            )
        )
    )
    assert controller._execution_price_max_age_seconds() == 3.0


def test_active_price_wait_does_not_block_open_position_protection() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    open_state = SimpleNamespace(positions={"BTCUSDT": object()})
    empty_state = SimpleNamespace(positions={})
    not_ready_snapshot = {
        "ready": False,
        "reason": "market_price_snapshot_not_ready",
        "missing_symbols": ["ETHUSDT"],
    }

    assert controller._should_wait_for_market_prices(["ETHUSDT"], not_ready_snapshot, empty_state) is True
    assert controller._should_wait_for_market_prices(["ETHUSDT"], not_ready_snapshot, open_state) is False
    assert controller._should_wait_for_market_prices([], not_ready_snapshot, open_state) is False
    assert controller._should_wait_for_market_prices(["ETHUSDT"], {"ready": True}, open_state) is False


def test_full_signal_cycle_is_throttled_while_open_position_marking_is_fast() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                market_poll_seconds=10.0,
                execution_price_refresh_seconds=1.0,
            )
        )
    )

    open_state = SimpleNamespace(positions={"BREVUSDT": object()})
    flat_state = SimpleNamespace(positions={})

    controller._last_full_signal_cycle_ts = 100.0

    assert controller._should_run_full_signal_cycle(ts=101.0, portfolio_state=open_state) is False
    assert controller._should_run_full_signal_cycle(ts=111.0, portfolio_state=open_state) is True
    assert controller._should_run_full_signal_cycle(ts=101.0, portfolio_state=flat_state) is True


def test_fast_mark_publish_due_uses_execution_refresh_interval() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                execution_price_refresh_seconds=1.0,
            )
        )
    )
    controller._last_fast_mark_publish_ts = 100.0

    assert controller._fast_mark_publish_due(now_ts=100.5) is False
    assert controller._fast_mark_publish_due(now_ts=101.0) is True


def test_fast_mark_publish_is_available_before_slow_signal_cycle() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                execution_price_refresh_seconds=1.0,
            )
        )
    )
    controller._last_fast_mark_publish_ts = 0.0
    open_state = SimpleNamespace(positions={"BREVUSDT": object()})

    assert controller._has_open_positions(open_state) is True
    assert controller._fast_mark_publish_due(now_ts=1.0) is True


def _price_gate_controller(tmp_path: Path) -> LiveTradingController:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace(
        runtime_dir=tmp_path,
        config=SimpleNamespace(
            runtime=SimpleNamespace(price_max_age_seconds=180.0, execution_price_max_age_seconds=2.0)
        ),
    )
    (tmp_path / "market").mkdir(parents=True, exist_ok=True)
    return controller


def test_execution_order_price_gate_accepts_fresh_entry_price_and_reprices_order_input(tmp_path: Path) -> None:
    controller = _price_gate_controller(tmp_path)
    (tmp_path / "market" / "latest.json").write_text(
        json.dumps({"symbols": {"BTCUSDT": {"mark_price": 101.5, "price_ts_ms": 1_700_000_009_999, "price_source": "runtime_market_mark"}}}),
        encoding="utf-8",
    )

    gate = controller._execution_order_price_gate("BTCUSDT", "ENTER_LONG", now_ms=1_700_000_010_000)

    assert gate["required"] is True
    assert gate["ok"] is True
    assert gate["price"] == 101.5
    assert gate["price_age_seconds"] == 0.001
    assert gate["max_age_seconds"] == 2.0


def test_execution_order_price_gate_rejects_stale_entry_and_flip_prices(tmp_path: Path) -> None:
    controller = _price_gate_controller(tmp_path)
    (tmp_path / "market" / "latest.json").write_text(
        json.dumps({"symbols": {"BTCUSDT": {"mark_price": 101.5, "price_ts_ms": 1_700_000_001_000, "price_source": "runtime_market_mark"}}}),
        encoding="utf-8",
    )

    entry_gate = controller._execution_order_price_gate("BTCUSDT", "ENTER_SHORT", now_ms=1_700_000_010_000)
    flip_gate = controller._execution_order_price_gate("BTCUSDT", "FLIP_TO_LONG", now_ms=1_700_000_010_000)

    assert entry_gate["ok"] is False
    assert entry_gate["reason"] == "execution_price_stale"
    assert entry_gate["observed_price"] == 101.5
    assert entry_gate["price"] is None
    assert entry_gate["price_age_seconds"] == 9.0
    assert flip_gate["ok"] is False
    assert flip_gate["reason"] == "execution_price_stale"


def test_execution_order_price_gate_does_not_expand_to_exit_actions(tmp_path: Path) -> None:
    controller = _price_gate_controller(tmp_path)

    gate = controller._execution_order_price_gate("BTCUSDT", "EXIT_LONG", now_ms=1_700_000_010_000)

    assert gate["required"] is False
    assert gate["ok"] is True
    assert gate["reason"] == "execution_price_gate_not_required"
