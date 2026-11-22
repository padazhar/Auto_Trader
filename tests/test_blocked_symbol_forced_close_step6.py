from __future__ import annotations

from execution.portfolio_state import PortfolioState, PortfolioStateStore, PositionState


def test_paper_blocked_symbol_position_is_closed() -> None:
    store = PortfolioStateStore('/tmp/portfolio_state_step6.json', starting_equity_usd=1000.0)
    state = PortfolioState(
        positions={
            'BTCUSDT': PositionState(
                symbol='BTCUSDT',
                side='LONG',
                quantity=2.0,
                notional_usd=100.0,
                entry_price=10.0,
                stop_loss=9.0,
                take_profit=12.0,
                opened_at=100.0,
                updated_at=100.0,
            )
        },
        authority_mode='paper_local',
    )

    report = store.close_blocked_positions(
        state,
        blocked_symbols=['BTCUSDT'],
        price_by_symbol={'BTCUSDT': 11.0},
        ts=120.0,
        execution_mode='paper',
    )

    assert 'BTCUSDT' not in state.positions
    assert report['closed'][0]['symbol'] == 'BTCUSDT'
    assert state.closed_trades[-1].exit_reason == 'blocked_symbol_forced_close'


def test_live_blocked_symbol_close_is_fail_closed() -> None:
    store = PortfolioStateStore('/tmp/portfolio_state_step6.json', starting_equity_usd=1000.0)
    state = PortfolioState(
        positions={
            'ETHUSDT': PositionState(
                symbol='ETHUSDT',
                side='SHORT',
                quantity=1.0,
                notional_usd=100.0,
                entry_price=20.0,
                stop_loss=21.0,
                take_profit=18.0,
                opened_at=100.0,
                updated_at=100.0,
            )
        },
        authority_mode='live_exchange_required',
    )

    report = store.close_blocked_positions(
        state,
        blocked_symbols=['ETHUSDT'],
        price_by_symbol={'ETHUSDT': 19.0},
        ts=120.0,
        execution_mode='live',
    )

    assert 'ETHUSDT' in state.positions
    assert report['fail_closed'] is True
    assert report['failed'][0]['reason'] == 'live_blocked_symbol_close_requires_exchange_reconciliation'
