from __future__ import annotations

import pandas as pd

from core.market_io import latest_symbol_price, load_market_latest_snapshot, load_open_position_symbols, load_runtime_price_symbols, write_market_latest_snapshot
from core.path_manager import PathManager
from inference.trade_decision_logic import _resolve_price_from_row


def test_market_latest_snapshot_round_trip(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text("{}\n", encoding="utf-8")
    (project / "config" / "tradable_symbols.yaml").write_text("symbols: [BTCUSDT]\n", encoding="utf-8")
    paths = PathManager.from_project_root(project)

    write_market_latest_snapshot(
        paths,
        {
            "source": "binance_futures_http",
            "created_at_ms": 123,
            "symbols": {
                "BTCUSDT": {
                    "primary_price": 101.5,
                    "mark_price": 101.4,
                    "ticker_price": 101.6,
                    "price_source": "mark_price",
                    "price_ts_ms": 456,
                }
            },
        },
    )

    snapshot = load_market_latest_snapshot(paths)
    assert snapshot["symbols"]["BTCUSDT"]["primary_price"] == 101.5
    price, source, ts = latest_symbol_price(paths, "BTCUSDT")
    assert price == 101.4
    assert source == "mark_price"
    assert ts == 456


def test_resolve_price_prefers_runtime_market_snapshot(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text("{}\n", encoding="utf-8")
    (project / "config" / "tradable_symbols.yaml").write_text("symbols: [BTCUSDT]\n", encoding="utf-8")
    paths = PathManager.from_project_root(project)
    import time
    now_ms = int(time.time() * 1000)
    write_market_latest_snapshot(paths, {"symbols": {"BTCUSDT": {"primary_price": 222.25, "price_ts_ms": now_ms}}})

    row = pd.Series({"close": None})
    price, source, ts = _resolve_price_from_row("BTCUSDT", "15m", row, paths)
    assert price == 222.25
    assert source == "runtime_market_primary"
    assert ts == now_ms


def test_active_universe_prefers_symbol_state_then_tradable_before_db(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text(
        "symbols:\n  target_active_symbols: 2\n", encoding="utf-8"
    )
    (project / "config" / "tradable_symbols.yaml").write_text(
        "symbols: [BTCUSDT, ETHUSDT, SOLUSDT]\n", encoding="utf-8"
    )
    paths = PathManager.from_project_root(project)
    (paths.market_storage_dir / "DOGEUSDT.db").touch()

    from core.market_io import load_active_universe_symbols

    assert load_active_universe_symbols(paths) == ["BTCUSDT", "ETHUSDT"]

    paths.symbol_state_path.write_text(
        '{"active_symbols": ["XRPUSDT", "BNBUSDT", "ADAUSDT"]}', encoding="utf-8"
    )
    assert load_active_universe_symbols(paths) == ["XRPUSDT", "BNBUSDT"]


def test_runtime_price_symbols_union_active_and_open_positions(tmp_path):
    project = tmp_path / "repo"
    (project / "config").mkdir(parents=True)
    (project / "config" / "system_settings.yaml").write_text(
        "symbols:\n  target_active_symbols: 2\n", encoding="utf-8"
    )
    (project / "config" / "tradable_symbols.yaml").write_text(
        "symbols: [BTCUSDT, ETHUSDT, SOLUSDT]\n", encoding="utf-8"
    )
    paths = PathManager.from_project_root(project)
    paths.portfolio_state_path.parent.mkdir(parents=True, exist_ok=True)
    paths.portfolio_state_path.write_text(
        '{"positions": {"ORCAUSDT": {"symbol": "ORCAUSDT"}, "ETHUSDT": {"symbol": "ETHUSDT"}}}',
        encoding="utf-8",
    )

    assert load_open_position_symbols(paths) == ["ORCAUSDT", "ETHUSDT"]
    assert load_runtime_price_symbols(paths) == ["BTCUSDT", "ETHUSDT", "ORCAUSDT"]
