from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.exchange_adapter import PaperExchangeAdapter
from execution.order_models import OrderRequest
from services.backtest_engine import BacktestEngine, ClosedTrade, ParsedSignal


class _StaticPriceClient:
    def __init__(self, price: float) -> None:
        self.price = float(price)

    def get_mark_price(self, symbol: str, *, force: bool = False):
        return self.price, 1234567890.0, False


def _paths(tmp_path: Path) -> SimpleNamespace:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        signal_events_log_path=runtime / "signal_events.log",
        backtest_open_trades_path=runtime / "backtest_open_trades.json",
        live_backtest_trades_path=runtime / "live_backtest_trades.csv",
        trade_history_db_path=runtime / "trade_history.db",
        config=SimpleNamespace(
            market=SimpleNamespace(futures_base_url="https://example.test"),
            inference=SimpleNamespace(consensus_enter_threshold=0.6, consensus_flip_threshold=0.5),
            trading=SimpleNamespace(fees_bps=3.0, slippage_bps=2.0),
            runtime=SimpleNamespace(market_poll_seconds=10.0),
        ),
    )


def test_backtest_engine_applies_round_trip_costs_on_flat_price(tmp_path: Path) -> None:
    engine = BacktestEngine(_paths(tmp_path))
    engine.price_client = _StaticPriceClient(100.0)
    open_trades = {}
    messages: list[str] = []

    engine.on_signal(
        ParsedSignal(
            ts_epoch=1710000000,
            symbol="BTCUSDT",
            side="LONG",
            confidence=0.8,
            mode="Medium",
            raw_line="unit_test",
            notional_usd=1000.0,
        ),
        open_trades,
        messages,
    )
    trade = open_trades["BTCUSDT"]
    closed = engine._close_trade(trade, 100.0, "2024-03-09 00:05:00", "LOG_EXIT")

    assert trade.entry_price > 100.0
    assert trade.entry_fee_usd > 0.0
    assert trade.entry_effective_slippage_bps > 2.0
    assert closed.exit_price < 100.0
    assert closed.gross_pnl_usd < 0.0
    assert closed.fees_usd > 0.0
    assert closed.slippage_usd > 0.0
    assert closed.pnl_usd < closed.gross_pnl_usd
    assert closed.pnl_usd < 0.0


def test_paper_exchange_adapter_applies_configured_fee_and_slippage(tmp_path: Path) -> None:
    adapter = PaperExchangeAdapter(_paths(tmp_path))
    result = adapter.place_order(
        OrderRequest(
            order_id="ord-1",
            client_order_id="cli-1",
            symbol="BTCUSDT",
            side="LONG",
            quantity=1.0,
            order_type="MARKET",
            metadata={"reference_price": 100.0},
        ),
        ts=123.0,
    )

    assert result.accepted is True
    assert result.fill is not None
    assert result.fill.price > 100.0
    assert result.fill.fee_usd > 0.0
    assert result.fill.metadata["effective_slippage_bps"] > 2.0


def test_completed_trade_csv_export_accepts_full_closed_trade_contract(tmp_path: Path) -> None:
    engine = BacktestEngine(_paths(tmp_path))
    closed = ClosedTrade(
        symbol="BTCUSDT",
        side="LONG",
        mode="Medium",
        entry_time="2024-03-09 00:00:00",
        entry_price=100.0,
        exit_time="2024-03-09 00:05:00",
        exit_price=101.0,
        pnl_usd=8.5,
        pnl_pct=0.85,
        notional_usd=1000.0,
        exit_reason="unit_test",
        gross_pnl_usd=10.0,
        fees_usd=0.6,
        slippage_usd=0.9,
        latency_bps=1.0,
        spread_bps=1.5,
        entry_reference_price=99.95,
        exit_reference_price=101.05,
    )

    exported = engine.export_completed_trades_csv([closed])

    assert exported == [closed]
    header = engine.completed_csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert set(closed.to_csv_row()).issubset(set(header))
    assert "spread_bps" in header
