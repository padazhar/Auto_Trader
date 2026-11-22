from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from features.market_loader import (
    load_funding_rate_frame,
    load_open_interest_frame,
    refresh_funding_rate_data,
    refresh_open_interest_data,
)
from runners.market_data_controller import MarketDataController


class _DummyResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _Paths(SimpleNamespace):
    def market_db_path(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{str(symbol).upper()}.db"

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _make_paths(tmp_path: Path) -> _Paths:
    project_root = tmp_path / "repo"
    config_dir = project_root / "config"
    market_storage_dir = project_root / "data" / "market_storage"
    logs_dir = project_root / "data" / "logs"
    runtime_dir = project_root / "data" / "runtime"
    config_dir.mkdir(parents=True, exist_ok=True)
    market_storage_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "system_settings.yaml").write_text("{}", encoding="utf-8")
    (config_dir / "tradable_symbols.yaml").write_text("symbols: ['BTCUSDT']\n", encoding="utf-8")
    config = SimpleNamespace(
        market=SimpleNamespace(
            futures_base_url="https://fapi.binance.com",
            fetch_timeout_sec=1.0,
            max_retries=1,
            backoff_base_sec=0.0,
            klines_limit=1000,
            use_binance_http_updater=True,
            refresh_universe_enabled=False,
            refresh_universe_sec=300,
            top_n=1,
            fast_frames=["15m"],
            slow_frames=[],
            sem_fast=1,
            sem_slow=1,
            request_timeout_sec=1,
            request_retries=1,
            bar_close_delay_ms=0,
            server_time_cache_ms=0,
            lookback_days={"15m": 3},
        ),
        symbols=SimpleNamespace(target_active_symbols=1),
        frames=SimpleNamespace(anchor_interval="15m", fusion_intervals=["15m"], lookback_rows={"15m": 100}),
        feature_store=SimpleNamespace(include_optional_futures_context=True),
    )
    return _Paths(
        project_root=project_root,
        config_dir=config_dir,
        market_storage_dir=market_storage_dir,
        logs_dir=logs_dir,
        runtime_dir=runtime_dir,
        config=config,
    )


def test_refresh_funding_and_open_interest_writes_canonical_tables(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)

    funding_payload = [
        {"fundingTime": 1710000000000, "fundingRate": "0.0001"},
        {"fundingTime": 1710028800000, "fundingRate": "0.0002"},
    ]
    oi_payload = [
        {"timestamp": 1710027900000, "sumOpenInterest": "12345.0"},
        {"timestamp": 1710028800000, "sumOpenInterest": "12375.0"},
    ]

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith("/fapi/v1/fundingRate"):
            return _DummyResponse(funding_payload)
        if url.endswith("/futures/data/openInterestHist"):
            return _DummyResponse(oi_payload)
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("features.market_loader.requests.get", fake_get)

    funding_rows = refresh_funding_rate_data(paths, "BTCUSDT")
    oi_rows = refresh_open_interest_data(paths, "BTCUSDT", "15m")

    assert funding_rows == 2
    assert oi_rows == 2

    funding_frame = load_funding_rate_frame(paths, "BTCUSDT")
    open_interest_frame = load_open_interest_frame(paths, "BTCUSDT", "15m")

    assert funding_frame["funding_rate"].tolist() == [0.0001, 0.0002]
    assert open_interest_frame["open_interest"].tolist() == [12345.0, 12375.0]


def test_market_data_controller_refreshes_optional_context_on_anchor_interval(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr("runners.market_data_controller.refresh_funding_rate_data", lambda paths, symbol: calls.append(("funding", symbol)) or 1)
    monkeypatch.setattr("runners.market_data_controller.refresh_open_interest_data", lambda paths, symbol, period: calls.append((period, symbol)) or 2)
    monkeypatch.setattr("runners.market_data_controller.get_logger", lambda name, paths: SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, exception=lambda *a, **k: None))
    monkeypatch.setattr(
        "runners.market_data_controller.MonitorPublisher",
        lambda paths: SimpleNamespace(
            heartbeat=lambda *a, **k: None,
            update_market_connectivity=lambda *a, **k: None,
            update_market_universe=lambda *a, **k: None,
            update_market_result=lambda *a, **k: None,
            publish_event=lambda *a, **k: None,
            controller_error=lambda *a, **k: None,
            flush=lambda: None,
        ),
    )

    controller = MarketDataController(paths)
    result = asyncio.run(controller.update_optional_context_for_symbol("BTCUSDT", "15m"))
    skipped = asyncio.run(controller.update_optional_context_for_symbol("BTCUSDT", "5m"))

    assert result == {"funding_rate_rows": 1, "open_interest_rows": 2}
    assert skipped == {"funding_rate_rows": 0, "open_interest_rows": 0}
    assert calls == [("funding", "BTCUSDT"), ("15m", "BTCUSDT")]


def test_refresh_open_interest_replaces_existing_timestamps_without_integrity_error(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)

    first_payload = [
        {"timestamp": 1710027900000, "sumOpenInterest": "12345.0"},
        {"timestamp": 1710028800000, "sumOpenInterest": "12375.0"},
    ]
    second_payload = [
        {"timestamp": 1710028800000, "sumOpenInterest": "12399.0"},
        {"timestamp": 1710029700000, "sumOpenInterest": "12410.0"},
    ]
    payloads = [first_payload, second_payload]

    def fake_get(url, params=None, timeout=None, headers=None):
        if url.endswith("/futures/data/openInterestHist"):
            return _DummyResponse(payloads.pop(0))
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("features.market_loader.requests.get", fake_get)

    assert refresh_open_interest_data(paths, "BTCUSDT", "15m") == 2
    assert refresh_open_interest_data(paths, "BTCUSDT", "15m", lookback_ms=10_000_000) == 2

    open_interest_frame = load_open_interest_frame(paths, "BTCUSDT", "15m")
    assert open_interest_frame["timestamp"].tolist() == [1710027900000, 1710028800000, 1710029700000]
    assert open_interest_frame["open_interest"].tolist() == [12345.0, 12399.0, 12410.0]
