from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml

from core.symbol_management import load_symbol_state
from runners import market_data_controller as mdc


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.universe_updates: list[tuple[list[str], str]] = []

    def update_market_universe(self, symbols, source, last_refresh_ts=None) -> None:
        self.universe_updates.append((list(symbols), str(source)))

    def heartbeat(self, *args, **kwargs) -> None:
        pass

    def update_market_connectivity(self, *args, **kwargs) -> None:
        pass

    def update_market_result(self, *args, **kwargs) -> None:
        pass

    def publish_event(self, *args, **kwargs) -> None:
        pass


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _build_paths(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    config_dir = project_root / "config"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    market_storage_dir = project_root / "data" / "market"

    config_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    market_storage_dir.mkdir(parents=True, exist_ok=True)

    _write_yaml(config_dir / "system_settings.yaml", {})
    _write_yaml(
        config_dir / "tradable_symbols.yaml",
        {
            "symbols": ["BTCUSDT"],
            "pinned_symbols": ["ETHUSDT"],
            "skip_symbols": ["XRPUSDT"],
        },
    )

    config = SimpleNamespace(
        market=SimpleNamespace(
            use_binance_http_updater=True,
            refresh_universe_enabled=True,
            refresh_universe_sec=300,
            top_n=15,
            fast_frames=["1m", "3m", "5m", "15m"],
            slow_frames=["30m", "1h", "4h", "1d"],
            sem_fast=4,
            sem_slow=2,
            request_timeout_sec=5,
            request_retries=1,
            bar_close_delay_ms=1000,
            server_time_cache_ms=180000,
            lookback_days={"1m": 5},
        ),
        symbols=SimpleNamespace(target_active_symbols=3),
        feature_store=SimpleNamespace(include_optional_futures_context=True),
        frames=SimpleNamespace(anchor_interval="15m", fusion_intervals=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"]),
        training=SimpleNamespace(rf=SimpleNamespace(min_rows=450)),
        monitoring=SimpleNamespace(
            max_events=10,
            max_alerts=10,
            max_market_updates=10,
            max_recent_training_results=10,
            max_recent_decisions=10,
            max_recent_signals=10,
            max_backtest_open_positions=10,
            max_backtest_recent_closed=10,
            max_registry_entries=10,
        ),
        risk=SimpleNamespace(account_balance_usd=1000.0),
        runtime=SimpleNamespace(execution_mode="paper", allow_unreconciled_live_mode=False),
        exchange=SimpleNamespace(position_reconciliation_enabled=False, require_reconciliation_for_live=True),
    )

    return SimpleNamespace(
        project_root=project_root,
        config_dir=config_dir,
        market_storage_dir=market_storage_dir,
        tradable_symbols_path=config_dir / "tradable_symbols.yaml",
        symbol_state_path=runtime_dir / "symbol_state.json",
        symbol_actions_path=runtime_dir / "symbol_actions.json",
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        monitoring_state_path=runtime_dir / "monitoring_state.json",
        monitoring_events_path=runtime_dir / "monitoring_events.json",
        model_storage_dir=project_root / "data" / "model_storage",
        logs_dir=logs_dir,
        config=config,
        log_file_path=lambda name: logs_dir / f"{name}.log",
    )


def test_refresh_universe_once_uses_current_symbol_management_contract(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)

    controller = mdc.MarketDataController(paths)

    async def _fake_ranked() -> list[str]:
        return ["BTCUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]

    monkeypatch.setattr(controller, "fetch_volume_ranked_symbols", _fake_ranked)

    ranked = asyncio.run(controller.refresh_universe_once())

    assert ranked == ["BTCUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]

    state = load_symbol_state(paths)
    assert state["ranked_candidates"] == ["BTCUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
    # MarketDataController owns ranked replacement candidates only.  It must not
    # overwrite active_symbols or tradable_symbols.yaml; TechRevController owns
    # active/blocked/timed-out lifecycle state and export alignment.
    assert state["active_symbols"] == []

    assert paths.tradable_symbols_path.exists()
    exported = yaml.safe_load(paths.tradable_symbols_path.read_text(encoding="utf-8"))
    assert exported["symbols"] == ["BTCUSDT"]
    assert exported["pinned_symbols"] == ["ETHUSDT"]
    assert exported["skip_symbols"] == ["XRPUSDT"]


def test_fetch_volume_ranked_symbols_filters_to_exchange_trading_symbols(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)
    controller = mdc.MarketDataController(paths)

    async def _fake_request(url, *args, **kwargs):
        if url == mdc.BINANCE_FUTURES_EXCHANGE_INFO_URL:
            return {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "onboardDate": 1_500_000_000_000},
                    {"symbol": "币安人生USDT", "status": "TRADING", "contractType": "PERPETUAL", "quoteAsset": "USDT", "onboardDate": 1_500_000_000_000},
                    {"symbol": "BADUSDT", "status": "BREAK", "contractType": "PERPETUAL", "quoteAsset": "USDT", "onboardDate": 1_500_000_000_000},
                ]
            }
        if url == mdc.BINANCE_FUTURES_24H_URL:
            return [
                {"symbol": "BTCUSDT", "quoteVolume": "100"},
                {"symbol": "BADUSDT", "quoteVolume": "200"},
                {"symbol": "币安人生USDT", "quoteVolume": "300"},
            ]
        return None

    monkeypatch.setattr(controller, "_request_json", _fake_request)
    ranked = asyncio.run(controller.fetch_volume_ranked_symbols())
    assert ranked == ["BTCUSDT"]


def test_fetch_volume_ranked_symbols_excludes_recent_listings_without_required_history(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)
    controller = mdc.MarketDataController(paths)
    now_ms = 1_800_000_000_000
    monkeypatch.setattr(controller, "now_ms_corrected", lambda: now_ms)

    required_age = controller.minimum_history_age_ms
    assert required_age == 60 * 24 * 60 * 60 * 1000

    async def _fake_request(url, *args, **kwargs):
        if url == mdc.BINANCE_FUTURES_EXCHANGE_INFO_URL:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                        "onboardDate": now_ms - required_age - 1,
                    },
                    {
                        "symbol": "NEWUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                        "onboardDate": now_ms - required_age + 1,
                    },
                    {
                        "symbol": "UNKNOWNUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                    },
                ]
            }
        if url == mdc.BINANCE_FUTURES_24H_URL:
            return [
                {"symbol": "NEWUSDT", "quoteVolume": "300"},
                {"symbol": "UNKNOWNUSDT", "quoteVolume": "200"},
                {"symbol": "BTCUSDT", "quoteVolume": "100"},
            ]
        return None

    monkeypatch.setattr(controller, "_request_json", _fake_request)
    ranked = asyncio.run(controller.fetch_volume_ranked_symbols())

    assert ranked == ["BTCUSDT"]


def test_refresh_universe_persists_only_history_eligible_ranked_candidates(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)
    controller = mdc.MarketDataController(paths)
    now_ms = 1_800_000_000_000
    monkeypatch.setattr(controller, "now_ms_corrected", lambda: now_ms)
    required_age = controller.minimum_history_age_ms

    async def _fake_request(url, *args, **kwargs):
        if url == mdc.BINANCE_FUTURES_EXCHANGE_INFO_URL:
            return {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                        "onboardDate": now_ms - required_age - 1,
                    },
                    {
                        "symbol": "NEWUSDT",
                        "status": "TRADING",
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                        "onboardDate": now_ms - (7 * 24 * 60 * 60 * 1000),
                    },
                ]
            }
        if url == mdc.BINANCE_FUTURES_24H_URL:
            return [
                {"symbol": "NEWUSDT", "quoteVolume": "1000"},
                {"symbol": "BTCUSDT", "quoteVolume": "100"},
            ]
        return None

    monkeypatch.setattr(controller, "_request_json", _fake_request)
    ranked = asyncio.run(controller.refresh_universe_once())

    assert ranked == ["BTCUSDT"]
    state = load_symbol_state(paths)
    assert state["ranked_candidates"] == ["BTCUSDT"]


def test_bootstrap_new_active_symbols_backfills_all_required_intervals_once(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)
    controller = mdc.MarketDataController(paths)

    from core.symbol_management import save_symbol_state
    state = load_symbol_state(paths)
    state["active_symbols"] = ["BTCUSDT"]
    save_symbol_state(paths, state)

    calls: list[tuple[str, str]] = []

    async def _fake_update(symbol: str, timeframe: str):
        calls.append((symbol, timeframe))
        return mdc.UpdateResult(symbol, timeframe, 1, 1_800_000_000_000)

    monkeypatch.setattr(controller, "update_symbol_timeframe", _fake_update)
    ready = {"value": False}
    monkeypatch.setattr(controller, "_symbol_history_ready", lambda symbol: ready["value"])

    original_publish = controller._publish_update
    monkeypatch.setattr(controller, "_publish_update", lambda result: None)

    # First attempt runs every required interval but remains pending until the
    # shared history contract is actually satisfied.
    first = asyncio.run(controller.bootstrap_new_active_symbols_once())
    assert first == []
    assert calls == [("BTCUSDT", interval) for interval in controller.history_row_requirements]

    # Retry is throttled rather than hammering the exchange.
    calls.clear()
    second = asyncio.run(controller.bootstrap_new_active_symbols_once())
    assert second == []
    assert calls == []

    # Once history is ready, the symbol is marked complete without another fetch.
    ready["value"] = True
    third = asyncio.run(controller.bootstrap_new_active_symbols_once())
    assert third == ["BTCUSDT"]
    assert calls == []


def test_required_history_contract_is_shared_with_techrev(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(mdc, "MonitorPublisher", _DummyMonitor)
    controller = mdc.MarketDataController(paths)

    assert controller.history_row_requirements == {
        "1m": 80,
        "3m": 80,
        "5m": 80,
        "15m": 530,
        "30m": 80,
        "1h": 80,
        "4h": 80,
        "1d": 60,
    }
