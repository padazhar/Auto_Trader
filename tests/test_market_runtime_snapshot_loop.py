from __future__ import annotations

import asyncio

from core.market_io import load_market_latest_snapshot
from core.path_manager import PathManager
from runners.market_data_controller import MarketDataController


def _make_paths(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text(
        """
market:
  use_binance_http_updater: true
  refresh_universe_enabled: false
  refresh_universe_sec: 60
  top_n: 10
  fast_frames: [1m]
  slow_frames: [15m]
  sem_fast: 1
  sem_slow: 1
  request_timeout_sec: 1
  request_retries: 1
  bar_close_delay_ms: 0
  server_time_cache_ms: 1000
  lookback_days: {1m: 1, 15m: 1}
feature_store:
  include_optional_futures_context: false
frames:
  anchor_interval: 15m
symbols:
  target_active_symbols: 4
""",
        encoding="utf-8",
    )
    (project / "config" / "tradable_symbols.yaml").write_text("symbols: [BTCUSDT, ETHUSDT]\n", encoding="utf-8")
    return PathManager.from_project_root(project)


def test_refresh_runtime_price_snapshot_writes_latest_file(tmp_path, monkeypatch):
    paths = _make_paths(tmp_path)
    controller = MarketDataController(paths)

    async def fake_fetch(symbols):
        return {
            "source": "binance_futures_http",
            "created_at_ms": 123456,
            "symbols": {
                symbol: {
                    "price": 101.0 + idx,
                    "primary_price": 101.0 + idx,
                    "mark_price": 101.0 + idx,
                    "ticker_price": 101.0 + idx,
                    "db_close": None,
                    "source": "mark_price",
                    "price_source": "mark_price",
                    "timestamp": 123456,
                    "price_ts_ms": 123456,
                    "event_time_ms": 123456,
                }
                for idx, symbol in enumerate(symbols)
            },
        }

    monkeypatch.setattr(controller, "fetch_runtime_price_snapshot", fake_fetch)
    snapshot = asyncio.run(controller.refresh_runtime_price_snapshot(reason="test"))

    assert snapshot["reason"] == "test"
    loaded = load_market_latest_snapshot(paths)
    assert loaded["symbols"]["BTCUSDT"]["primary_price"] == 101.0
    assert loaded["symbols"]["ETHUSDT"]["source"] == "mark_price"
    assert loaded["active_symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert loaded["expected_symbol_count"] == 2
    assert loaded["priced_symbol_count"] == 2
    assert loaded["missing_symbols"] == []
    assert loaded["all_symbols_priced"] is True
    assert paths.market_latest_snapshot_path.exists()


def test_refresh_runtime_price_snapshot_writes_empty_snapshot_when_no_symbols(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text("{}\n", encoding="utf-8")
    (project / "config" / "tradable_symbols.yaml").write_text("symbols: []\n", encoding="utf-8")
    paths = PathManager.from_project_root(project)
    controller = MarketDataController(paths)

    snapshot = asyncio.run(controller.refresh_runtime_price_snapshot(reason="empty"))

    assert snapshot["symbols"] == {}
    loaded = load_market_latest_snapshot(paths)
    assert loaded["symbols"] == {}
    assert loaded["reason"] == "empty"
