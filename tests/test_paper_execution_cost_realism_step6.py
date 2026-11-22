from __future__ import annotations

import pytest

from execution.portfolio_state import PortfolioStateStore


def test_portfolio_closed_trade_uses_net_pnl_after_fill_fees(tmp_path):
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()

    entry_fill = {
        "fee_usd": 0.30,
        "metadata": {
            "reference_price": 100.0,
            "estimated_slippage_usd": 0.20,
            "effective_slippage_bps": 3.0,
            "latency_bps": 1.0,
            "spread_bps": 1.0,
        },
    }
    exit_fill = {
        "fee_usd": 0.33,
        "metadata": {
            "reference_price": 106.0,
            "estimated_slippage_usd": 0.22,
            "effective_slippage_bps": 3.0,
            "latency_bps": 1.0,
            "spread_bps": 1.0,
        },
    }

    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100.0,
        ts=1000.0,
        position_plan={
            "quantity": 2.0,
            "notional_usd": 200.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "execution_fill": entry_fill,
        },
    )
    report = store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="EXIT_LONG",
        price=106.0,
        ts=1100.0,
        position_plan={"exit_reason": "unit_test_take_profit", "lineage": {"execution_fill": exit_fill}},
    )

    assert report["realized_pnl_usd"] == 11.37
    trade = state.closed_trades[0]
    assert trade.gross_pnl_usd == 12.0
    assert trade.fees_usd == 0.63
    assert trade.slippage_usd == pytest.approx(0.42)
    assert trade.pnl_usd == 11.37
    assert trade.entry_reference_price == 100.0
    assert trade.exit_reference_price == 106.0

    summary = store.execution_summary(state)
    assert summary["gross_pnl_usd"] == 12.0
    assert summary["fees_usd"] == 0.63
    assert summary["slippage_usd"] == pytest.approx(0.42)
    assert summary["total_pnl"] == 11.37
