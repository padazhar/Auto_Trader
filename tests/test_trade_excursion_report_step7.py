from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.path_manager import PathManager
from execution.portfolio_state import build_trade_excursion_attribution
from features.market_loader import load_market_window, save_market_frame
from reporting.run_report import build_run_report


def test_build_trade_excursion_attribution_for_long_trade():
    trade = {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 100.0,
        "exit_price": 98.0,
        "pnl_usd": -2.0,
        "exit_reason": "SL_HIT",
        "risk_plan": {"stop_loss": 98.0, "take_profit": 105.0},
    }
    candles = [
        {"timestamp": 1, "open": 100.0, "high": 103.0, "low": 99.0, "close": 101.0, "volume": 1.0},
        {"timestamp": 2, "open": 101.0, "high": 104.0, "low": 97.5, "close": 98.0, "volume": 1.0},
    ]

    payload = build_trade_excursion_attribution(trade=trade, candles=candles)

    assert payload["schema_version"] == "trade_excursion_attribution.v1"
    assert payload["analysis_status"] == "ok"
    assert payload["max_favorable_excursion_usd"] == 4.0
    assert payload["max_adverse_excursion_usd"] == 2.5
    assert payload["stop_loss_reachable"] is True
    assert payload["take_profit_reachable"] is False
    assert payload["sl_too_tight_candidate"] is True


def test_load_market_window_reads_bounded_ohlcv(tmp_path: Path):
    root = tmp_path
    (root / "config").mkdir(parents=True)
    (root / "config" / "system_settings.yaml").write_text(
        "system:\n  environment: test\nmarket:\n  top_n: 1\nframes:\n  fusion_intervals: ['1m']\n",
        encoding="utf-8",
    )
    paths = PathManager.from_project_root(root)
    frame = pd.DataFrame([
        {"timestamp": 1000, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1.0},
        {"timestamp": 2000, "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.0, "volume": 1.0},
        {"timestamp": 3000, "open": 3.0, "high": 3.1, "low": 2.9, "close": 3.0, "volume": 1.0},
    ])
    save_market_frame(paths, "BTCUSDT", "1m", frame)

    loaded = load_market_window(paths, "BTCUSDT", "1m", 1500, 2500)

    assert list(loaded["timestamp"]) == [2000]


def test_run_report_includes_trade_excursion_section(tmp_path: Path):
    root = tmp_path
    (root / "config").mkdir(parents=True)
    (root / "config" / "system_settings.yaml").write_text(
        "system:\n  environment: test\nmarket:\n  top_n: 1\nframes:\n  fusion_intervals: ['1m']\ntrading:\n  starting_equity_usd: 1000\n",
        encoding="utf-8",
    )
    paths = PathManager.from_project_root(root)
    save_market_frame(paths, "BTCUSDT", "1m", pd.DataFrame([
        {"timestamp": 1_700_000_000_000, "open": 100.0, "high": 103.0, "low": 99.0, "close": 101.0, "volume": 1.0},
        {"timestamp": 1_700_000_060_000, "open": 101.0, "high": 104.0, "low": 97.5, "close": 98.0, "volume": 1.0},
    ]))
    paths.portfolio_state_path.write_text(
        '{"closed_trades":[{"symbol":"BTCUSDT","side":"LONG","entry_price":100,"exit_price":98,"opened_at":1700000000,"closed_at":1700000060,"pnl_usd":-2,"exit_reason":"SL_HIT","risk_plan":{"stop_loss":98,"take_profit":105}}],"positions":{}}',
        encoding="utf-8",
    )

    report = build_run_report(paths, monitor=SimpleNamespace(snapshot=lambda: {}))

    excursion = report["trade_excursion_attribution"]
    assert excursion["schema_version"] == "trade_excursion_report.v1"
    assert excursion["closed_trade_count"] == 1
    assert excursion["status_counts"]["ok"] == 1
    assert excursion["sl_too_tight_candidate_count"] == 1
