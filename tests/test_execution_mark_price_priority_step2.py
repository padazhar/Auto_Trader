from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from core.market_io import latest_symbol_price_status
from runners.live_trading_controller import LiveTradingController
from services.backtest_engine import MarkPriceClient


def _paths(tmp_path: Path) -> SimpleNamespace:
    market_latest = tmp_path / "runtime" / "market" / "latest.json"
    market_latest.parent.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        market_latest_snapshot_path=market_latest,
        config=SimpleNamespace(
            runtime=SimpleNamespace(
                price_max_age_seconds=30.0,
                execution_price_max_age_seconds=5.0,
            )
        ),
    )


def _write_snapshot(paths: SimpleNamespace, *, symbol: str = "BTCUSDT") -> None:
    now_ms = int(time.time() * 1000)
    payload = {
        "created_at_ms": now_ms,
        "active_symbols": [symbol],
        "price_symbols": [symbol],
        "expected_symbol_count": 1,
        "priced_symbol_count": 1,
        "missing_symbols": [],
        "all_symbols_priced": True,
        "symbols": {
            symbol: {
                "price": 99.0,
                "latest_price": 99.0,
                "reference_price": 99.0,
                "primary_price": 100.0,
                "mark_price": 101.0,
                "ticker_price": 98.0,
                "price_source": "mark_price_websocket_cached",
                "price_ts_ms": now_ms,
            }
        },
    }
    paths.market_latest_snapshot_path.write_text(json.dumps(payload), encoding="utf-8")


def test_latest_symbol_price_status_prefers_mark_price(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_snapshot(paths)

    status = latest_symbol_price_status(paths, "BTCUSDT", max_age_seconds=5.0)

    assert status["price"] == 101.0
    assert status["latest_price"] == 101.0
    assert status["selected_price_field"] == "mark_price"
    assert status["price_source"] == "mark_price_websocket_cached"
    assert status["price_fresh"] is True


def test_live_controller_decision_price_prefers_mark_price() -> None:
    controller = LiveTradingController.__new__(LiveTradingController)
    controller.paths = SimpleNamespace()

    price, components = controller._normalize_decision_price(
        "BTCUSDT",
        {
            "price": 99.0,
            "latest_price": 99.0,
            "reference_price": 99.0,
            "primary_price": 100.0,
            "mark_price": 101.0,
            "price_source": "mark_price_websocket_cached",
            "price_ts_ms": 1_700_000_000_000,
        },
    )

    assert price == 101.0
    assert components["latest_price"] == 101.0
    assert components["reference_price"] == 101.0
    assert components["price_source"] == "mark_price_websocket_cached"


def test_backtest_mark_price_client_prefers_runtime_snapshot(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_snapshot(paths, symbol="ETHUSDT")
    client = MarkPriceClient("https://invalid.local", paths=paths)

    price, ts, stale = client.get_mark_price("ETHUSDT", force=True)

    assert price == 101.0
    assert ts is not None
    assert stale is False
