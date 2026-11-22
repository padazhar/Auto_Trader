from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from monitoring.publisher import MonitorPublisher
from monitoring.web_app import _normalize_snapshot


def _paths(tmp_path: Path) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    market_dir = runtime_dir / "market"
    market_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        runtime_dir=runtime_dir,
        monitoring_state_path=runtime_dir / "monitoring_state.json",
        monitoring_events_path=runtime_dir / "monitoring_events.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        market_latest_snapshot_path=market_dir / "latest.json",
    )


def test_monitoring_snapshot_projects_market_price_cache_diagnostics(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now_ms = int(time.time() * 1000)
    paths.market_latest_snapshot_path.write_text(
        json.dumps(
            {
                "source": "binance_futures_http",
                "reason": "loop",
                "created_at_ms": now_ms,
                "active_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "expected_symbol_count": 3,
                "priced_symbol_count": 2,
                "missing_symbols": ["SOLUSDT"],
                "all_symbols_priced": False,
                "symbols": {
                    "BTCUSDT": {"price": 100.0, "price_source": "mark_price", "price_ts_ms": now_ms},
                    "ETHUSDT": {"primary_price": 200.0, "price_source": "ticker_price", "price_ts_ms": now_ms - 10_000},
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = MonitorPublisher(paths).snapshot()
    diagnostics = snapshot["market_data"]["runtime_price_snapshot"]

    assert diagnostics["exists"] is True
    assert diagnostics["expected_symbol_count"] == 3
    assert diagnostics["priced_symbol_count"] == 2
    assert diagnostics["missing_symbol_count"] == 1
    assert diagnostics["missing_symbols"] == ["SOLUSDT"]
    assert diagnostics["price_source_counts"]["mark_price"] == 1
    assert diagnostics["price_source_counts"]["ticker_price"] == 1
    assert snapshot["market_data"]["price_cache_status"] == "degraded"


def test_web_snapshot_normalization_preserves_price_diagnostics() -> None:
    normalized = _normalize_snapshot(
        {
            "market_data": {
                "universe": {"symbols": ["BTCUSDT"], "source": "config", "last_refresh_ts": 123},
                "runtime_price_snapshot": {
                    "exists": True,
                    "expected_symbol_count": 1,
                    "priced_symbol_count": 1,
                    "missing_symbol_count": 0,
                    "stale_symbol_count": 0,
                    "all_symbols_priced": True,
                },
            }
        }
    )

    market = normalized["market_data"]
    assert market["universe"] == ["BTCUSDT"]
    assert market["price_cache_status"] == "ready"
    assert market["runtime_price_snapshot"]["priced_symbol_count"] == 1


def test_monitoring_diagnostics_uses_price_symbols_for_open_positions(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now_ms = int(time.time() * 1000)
    paths.market_latest_snapshot_path.write_text(
        json.dumps(
            {
                "source": "binance_futures_ws_mark_price",
                "reason": "websocket",
                "created_at_ms": now_ms,
                "active_symbols": ["BTCUSDT", "ETHUSDT"],
                "open_position_symbols": ["ORCAUSDT"],
                "price_symbols": ["BTCUSDT", "ETHUSDT", "ORCAUSDT"],
                "expected_symbol_count": 3,
                "priced_symbol_count": 3,
                "missing_symbols": [],
                "all_symbols_priced": True,
                "symbols": {
                    "BTCUSDT": {"mark_price": 100.0, "price_source": "mark_price_websocket", "price_ts_ms": now_ms},
                    "ETHUSDT": {"mark_price": 200.0, "price_source": "mark_price_websocket", "price_ts_ms": now_ms},
                    "ORCAUSDT": {"mark_price": 1.25, "price_source": "mark_price_websocket", "price_ts_ms": now_ms},
                },
            }
        ),
        encoding="utf-8",
    )

    diagnostics = MonitorPublisher(paths).snapshot()["market_data"]["runtime_price_snapshot"]

    assert diagnostics["expected_symbol_count"] == 3
    assert diagnostics["priced_symbol_count"] == 3
    assert diagnostics["missing_symbol_count"] == 0
    assert diagnostics["ready"] is True


def test_monitoring_price_diagnostics_exposes_websocket_fallback_counters(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    now_ms = int(time.time() * 1000)
    paths.market_latest_snapshot_path.write_text(
        json.dumps(
            {
                "source": "binance_futures_ws_mark_price",
                "reason": "websocket",
                "created_at_ms": now_ms,
                "active_symbols": ["BTCUSDT"],
                "price_symbols": ["BTCUSDT"],
                "expected_symbol_count": 1,
                "priced_symbol_count": 1,
                "missing_symbols": [],
                "all_symbols_priced": True,
                "websocket_cache_status": "ready",
                "websocket_partial_snapshot_events": 7,
                "websocket_http_fallback_events": 2,
                "websocket_retained_snapshot_events": 5,
                "symbols": {
                    "BTCUSDT": {"mark_price": 100.0, "price_source": "mark_price_websocket", "price_ts_ms": now_ms},
                },
            }
        ),
        encoding="utf-8",
    )

    diagnostics = MonitorPublisher(paths).snapshot()["market_data"]["runtime_price_snapshot"]

    assert diagnostics["websocket_cache_status"] == "ready"
    assert diagnostics["websocket_partial_snapshot_events"] == 7
    assert diagnostics["websocket_http_fallback_events"] == 2
    assert diagnostics["websocket_retained_snapshot_events"] == 5
