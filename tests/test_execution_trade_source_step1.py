from __future__ import annotations

import json
from types import SimpleNamespace

from execution.portfolio_state import (
    apply_execution_trade_source,
    execution_trade_source_from_paths,
    load_closed_trade_csv,
    paper_execution_trade_source_from_paths,
)


CSV_HEADER = (
    "symbol,side,mode,entry_time,entry_price,exit_time,exit_price,pnl_usd,pnl_pct,"
    "notional_usd,exit_reason,gross_pnl_usd,fees_usd,slippage_usd,latency_bps,"
    "entry_reference_price,exit_reference_price\n"
)


def test_header_only_backtest_csv_does_not_replace_runtime_summary(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "live_backtest_trades.csv"
    csv_path.write_text(CSV_HEADER, encoding="utf-8")

    rows, metadata = load_closed_trade_csv(csv_path)
    assert rows == []
    assert metadata["usable"] is False
    assert metadata["reason"] == "header_only"
    assert metadata["authoritative_for_paper_execution"] is False

    runtime = {"summary": {"completed_count": 3, "trades": 3, "wins": 2, "losses": 1, "total_pnl": 7.0}}
    updated = apply_execution_trade_source(runtime, paths=SimpleNamespace(project_root=tmp_path))

    assert updated["summary"]["completed_count"] == 3
    assert updated["summary"]["total_pnl"] == 7.0
    assert updated["trade_result_source"]["selected_source"] == "runtime_execution_payload"


def test_populated_backtest_csv_never_overrides_paper_runtime_summary(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_path = data_dir / "live_backtest_trades.csv"
    csv_path.write_text(
        CSV_HEADER
        + "BTCUSDT,LONG,paper,1,100,2,101,5.0,1.0,500,tp,5.4,0.2,0.2,0.1,100,101\n"
        + "ETHUSDT,SHORT,paper,3,200,4,210,-3.0,-1.0,300,sl,-2.8,0.1,0.1,0.1,200,210\n",
        encoding="utf-8",
    )
    runtime = {
        "summary": {"completed_count": 1, "trades": 1, "wins": 1, "losses": 0, "total_pnl": 9.0},
        "trade_history": [{"symbol": "SOLUSDT", "pnl_usd": 9.0}],
    }

    updated = apply_execution_trade_source(runtime, paths=SimpleNamespace(project_root=tmp_path))
    backtest = execution_trade_source_from_paths(SimpleNamespace(project_root=tmp_path))

    assert updated["summary"]["completed_count"] == 1
    assert updated["summary"]["total_pnl"] == 9.0
    assert updated["trade_history"][0]["symbol"] == "SOLUSDT"
    assert updated["trade_result_source"]["backtest_reference"]["selected_source"] == "historical_backtest_ledger"
    assert backtest["authoritative_for_paper_execution"] is False


def test_authoritative_portfolio_closed_trades_replace_any_legacy_overlaid_surface(tmp_path):
    runtime_dir = tmp_path / "data" / "runtime"
    runtime_dir.mkdir(parents=True)
    portfolio_path = runtime_dir / "portfolio_state.json"
    portfolio_path.write_text(json.dumps({
        "realized_pnl_usd": -2.0,
        "closed_trades": [
            {"symbol": "BTCUSDT", "side": "LONG", "pnl_usd": 4.0, "closed_at": 1},
            {"symbol": "ETHUSDT", "side": "SHORT", "pnl_usd": -6.0, "closed_at": 2},
        ],
    }), encoding="utf-8")
    data_dir = tmp_path / "data"
    (data_dir / "live_backtest_trades.csv").write_text(
        CSV_HEADER + "XRPUSDT,LONG,paper,1,1,2,2,99.0,1,1,tp,99,0,0,0,1,2\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(project_root=tmp_path, portfolio_state_path=portfolio_path)
    legacy = {"summary": {"source": "csv_closed_trade_ledger", "total_pnl": 99.0}, "trade_history": [{"symbol": "XRPUSDT", "pnl_usd": 99.0}]}

    source = paper_execution_trade_source_from_paths(paths)
    updated = apply_execution_trade_source(legacy, paths=paths)

    assert source["selected_source"] == "paper_execution_ledger"
    assert updated["summary"]["source"] == "paper_execution_ledger"
    assert updated["summary"]["completed_count"] == 2
    assert updated["summary"]["total_pnl"] == -2.0
    assert [row["symbol"] for row in updated["trade_history"]] == ["BTCUSDT", "ETHUSDT"]
    assert updated["trade_history_source"] == "paper_execution_ledger"
