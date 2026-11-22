from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.exchange_reconciliation import BinanceFuturesReconciler
from execution.portfolio_state import PortfolioStateStore


def _paths(tmp_path: Path) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        exchange=SimpleNamespace(
            futures_rest_base_url="https://fapi.binance.com",
            api_key="key",
            api_secret="secret",
            recv_window_ms=5000,
            use_testnet=False,
            position_reconciliation_enabled=True,
            require_reconciliation_for_live=True,
            max_reconciliation_age_sec=30,
        )
    )
    return SimpleNamespace(
        config=config,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
    )


def test_exchange_reconciliation_success(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = store.load()

    ts = 1710000000.0
    state.positions["BTCUSDT"] = SimpleNamespace(
        symbol="BTCUSDT",
        side="LONG",
        quantity=0.01,
        notional_usd=1000.0,
        entry_price=100000.0,
        stop_loss=0.0,
        take_profit=0.0,
        opened_at=ts,
        updated_at=ts,
        unrealized_pnl_usd=0.0,
    )

    reconciler = BinanceFuturesReconciler(paths)
    monkeypatch.setattr(
        reconciler,
        "_signed_get_json",
        lambda path, params: [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "entryPrice": "100000",
                "markPrice": "100100",
                "notional": "1001",
                "unRealizedProfit": "1.0",
            }
        ],
    )

    result = reconciler.reconcile(state, ts=ts)
    assert result.success is True
    assert result.reconciled is True
    assert result.status == "reconciled"
    assert result.mismatch_count == 0

    store.record_reconciliation(state, result, ts=ts)
    assert state.authority_mode == "live_exchange_reconciled"
    assert state.reconciliation_status == "reconciled"
    assert state.exchange_position_count == 1
    assert state.last_reconciled_at == ts
    assert state.positions["BTCUSDT"].side == "LONG"
    assert state.positions["BTCUSDT"].quantity == 0.01
    assert state.positions["BTCUSDT"].unrealized_pnl_usd == 1.0


def test_exchange_reconciliation_detects_mismatch(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = store.load()

    ts = 1710000000.0
    state.positions["BTCUSDT"] = SimpleNamespace(
        symbol="BTCUSDT",
        side="SHORT",
        quantity=0.01,
        notional_usd=1000.0,
        entry_price=100000.0,
        stop_loss=0.0,
        take_profit=0.0,
        opened_at=ts,
        updated_at=ts,
        unrealized_pnl_usd=0.0,
    )

    reconciler = BinanceFuturesReconciler(paths)
    monkeypatch.setattr(
        reconciler,
        "_signed_get_json",
        lambda path, params: [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.01",
                "entryPrice": "100000",
                "markPrice": "100100",
                "notional": "1001",
                "unRealizedProfit": "1.0",
            }
        ],
    )

    result = reconciler.reconcile(state, ts=ts)
    assert result.success is True
    assert result.reconciled is False
    assert result.status == "mismatch"
    assert result.mismatch_count == 1

    store.record_reconciliation(state, result, ts=ts)
    assert state.authority_mode == "live_exchange_required"
    assert state.reconciliation_status == "mismatch"
    assert state.last_reconciled_at is None
    assert state.positions["BTCUSDT"].side == "SHORT"
