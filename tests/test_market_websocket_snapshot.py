from __future__ import annotations

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
  runtime_price_websocket_enabled: true
  runtime_price_websocket_url: wss://fstream.binance.com/ws/!markPrice@arr@1s
  runtime_price_websocket_reconnect_sec: 1
  lookback_days: {1m: 1, 15m: 1}
feature_store:
  include_optional_futures_context: false
frames:
  anchor_interval: 15m
symbols:
  target_active_symbols: 2
""",
        encoding="utf-8",
    )
    (project / "config" / "tradable_symbols.yaml").write_text("symbols: [BTCUSDT, ETHUSDT]\n", encoding="utf-8")
    return PathManager.from_project_root(project)


def test_websocket_mark_price_snapshot_uses_active_symbols_only(tmp_path):
    paths = _make_paths(tmp_path)
    controller = MarketDataController(paths)
    payload = [
        {"e": "markPriceUpdate", "E": 123000, "s": "BTCUSDT", "p": "11185.50", "ap": "11185.40", "i": "11184.00", "r": "0.0001", "T": 456000},
        {"e": "markPriceUpdate", "E": 124000, "s": "ETHUSDT", "p": "2222.25", "ap": "2222.20", "i": "2221.00", "r": "0.0002", "T": 456000},
        {"e": "markPriceUpdate", "E": 125000, "s": "IGNOREDUSDT", "p": "1.0"},
    ]

    snapshot = controller._build_mark_price_snapshot_from_ws(payload, ["BTCUSDT", "ETHUSDT"], created_at_ms=130000)

    assert snapshot["source"] == "binance_futures_ws_mark_price"
    assert snapshot["stream"] == "!markPrice@arr@1s"
    assert snapshot["expected_symbol_count"] == 2
    assert snapshot["priced_symbol_count"] == 2
    assert snapshot["missing_symbols"] == []
    assert snapshot["all_symbols_priced"] is True
    assert set(snapshot["symbols"].keys()) == {"BTCUSDT", "ETHUSDT"}
    assert snapshot["symbols"]["BTCUSDT"]["primary_price"] == 11185.5
    assert snapshot["symbols"]["BTCUSDT"]["price_source"] == "mark_price_websocket"
    assert snapshot["symbols"]["BTCUSDT"]["price_ts_ms"] == 123000


def test_websocket_mark_price_config_parses(tmp_path):
    paths = _make_paths(tmp_path)

    assert paths.config.market.runtime_price_websocket_enabled is True
    assert paths.config.market.runtime_price_websocket_url.endswith("!markPrice@arr@1s")
    assert paths.config.market.runtime_price_websocket_reconnect_sec == 1.0


def test_runtime_price_symbols_include_open_positions_after_rotation(tmp_path):
    paths = _make_paths(tmp_path)
    paths.portfolio_state_path.write_text(
        '{"positions": {"ORCAUSDT": {"symbol": "ORCAUSDT", "side": "LONG"}}}',
        encoding="utf-8",
    )
    controller = MarketDataController(paths)

    assert controller._runtime_active_symbols() == ["BTCUSDT", "ETHUSDT"]
    assert controller._runtime_open_position_symbols() == ["ORCAUSDT"]
    assert controller._runtime_price_symbols() == ["BTCUSDT", "ETHUSDT", "ORCAUSDT"]


def test_websocket_mark_price_snapshot_prices_active_and_open_position_symbols(tmp_path):
    paths = _make_paths(tmp_path)
    paths.portfolio_state_path.write_text(
        '{"positions": {"ORCAUSDT": {"symbol": "ORCAUSDT", "side": "LONG"}}}',
        encoding="utf-8",
    )
    controller = MarketDataController(paths)
    payload = [
        {"e": "markPriceUpdate", "E": 123000, "s": "BTCUSDT", "p": "11185.50"},
        {"e": "markPriceUpdate", "E": 124000, "s": "ETHUSDT", "p": "2222.25"},
        {"e": "markPriceUpdate", "E": 125000, "s": "ORCAUSDT", "p": "1.37"},
    ]

    snapshot = controller._build_mark_price_snapshot_from_ws(
        payload,
        controller._runtime_price_symbols(),
        created_at_ms=130000,
    )

    assert snapshot["active_symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert snapshot["open_position_symbols"] == ["ORCAUSDT"]
    assert snapshot["price_symbols"] == ["BTCUSDT", "ETHUSDT", "ORCAUSDT"]
    assert snapshot["expected_symbol_count"] == 3
    assert snapshot["priced_symbol_count"] == 3
    assert set(snapshot["symbols"].keys()) == {"BTCUSDT", "ETHUSDT", "ORCAUSDT"}
    assert snapshot["symbols"]["ORCAUSDT"]["primary_price"] == 1.37


def test_websocket_snapshot_reuses_fresh_persisted_runtime_price_for_partial_payload(tmp_path):
    from core.market_io import write_market_latest_snapshot

    paths = _make_paths(tmp_path)
    controller = MarketDataController(paths)
    write_market_latest_snapshot(
        paths,
        {
            "source": "binance_futures_http",
            "created_at_ms": 125000,
            "symbols": {
                "ETHUSDT": {
                    "primary_price": 2222.25,
                    "mark_price": 2222.25,
                    "price_source": "mark_price",
                    "price_ts_ms": 126000,
                }
            },
        },
    )

    snapshot = controller._build_mark_price_snapshot_from_ws(
        [{"e": "markPriceUpdate", "E": 130000, "s": "BTCUSDT", "p": "11185.50"}],
        ["BTCUSDT", "ETHUSDT"],
        created_at_ms=131000,
    )

    assert snapshot["expected_symbol_count"] == 2
    assert snapshot["priced_symbol_count"] == 2
    assert snapshot["missing_symbols"] == []
    assert snapshot["symbols"]["BTCUSDT"]["price_source"] == "mark_price_websocket"
    assert snapshot["symbols"]["ETHUSDT"]["price_source"] == "runtime_price_snapshot_cached"
    assert snapshot["symbols"]["ETHUSDT"]["cache_origin_price_source"] == "mark_price"


def test_websocket_snapshot_does_not_treat_db_close_cache_as_live_price(tmp_path):
    from core.market_io import write_market_latest_snapshot

    paths = _make_paths(tmp_path)
    controller = MarketDataController(paths)
    write_market_latest_snapshot(
        paths,
        {
            "source": "binance_futures_http",
            "created_at_ms": 125000,
            "symbols": {
                "ETHUSDT": {
                    "primary_price": 2222.25,
                    "db_close": 2222.25,
                    "price_source": "db_close",
                    "price_ts_ms": 126000,
                }
            },
        },
    )

    snapshot = controller._build_mark_price_snapshot_from_ws(
        [{"e": "markPriceUpdate", "E": 130000, "s": "BTCUSDT", "p": "11185.50"}],
        ["BTCUSDT", "ETHUSDT"],
        created_at_ms=131000,
    )

    assert "ETHUSDT" in snapshot["missing_symbols"]
    assert snapshot["symbols"]["ETHUSDT"]["price_source"] in {"unavailable", "db_close"}


def test_websocket_http_fallback_is_throttled_when_latest_snapshot_exists(tmp_path):
    from core.market_io import write_market_latest_snapshot

    paths = _make_paths(tmp_path)
    controller = MarketDataController(paths)
    write_market_latest_snapshot(paths, {"created_at_ms": 1, "symbols": {}})

    assert controller._should_write_websocket_http_fallback(now_ms=100_000) is True
    assert controller._should_write_websocket_http_fallback(now_ms=105_000) is False
    assert controller._ws_http_fallback_events == 1
    assert controller._ws_retained_snapshot_events == 1
