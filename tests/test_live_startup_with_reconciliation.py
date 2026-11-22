from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import main as main_module


class DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.system_events: list[dict[str, object]] = []

    def publish_system(self, **payload) -> None:
        self.system_events.append(payload)

    def controller_error(self, name: str, message: str) -> None:
        pass

    def flush(self) -> None:
        pass


class DummyRegistry:
    def __init__(self, paths) -> None:
        self.paths = paths


def _fake_paths(project_root: Path) -> SimpleNamespace:
    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42, is_dev_mode=True),
        runtime=SimpleNamespace(
            execution_mode="live",
            allow_unreconciled_live_mode=False,
            live_canary_enabled=True,
            live_canary_max_symbols=1,
            live_canary_max_notional_usd=50.0,
            live_canary_max_concurrent_positions=1,
            live_canary_require_testnet=True,
        ),
        exchange=SimpleNamespace(
            position_reconciliation_enabled=True,
            require_reconciliation_for_live=True,
            api_key="key",
            api_secret="secret",
            use_testnet=True,
        ),
        telegram=SimpleNamespace(enabled=False),
    )
    return SimpleNamespace(project_root=project_root, config=config)


def test_main_live_mode_allows_startup_when_reconciliation_is_configured(monkeypatch, project_root: Path) -> None:
    paths = _fake_paths(project_root)
    monkeypatch.setenv("AUTO_TRADER_ENABLE_LIVE_MODE", "true")
    monkeypatch.setenv("AUTO_TRADER_ENABLE_LIVE_CANARY", "true")
    started: list[str] = []

    monkeypatch.setattr(main_module, "build_paths", lambda: paths)
    monkeypatch.setattr(main_module, "get_logger", lambda name, paths: DummyLogger())
    monkeypatch.setattr(main_module, "MonitorPublisher", DummyMonitor)
    monkeypatch.setattr(main_module, "TrainingRegistry", DummyRegistry)
    monkeypatch.setattr(main_module, "_validate_startup_secrets", lambda paths, logger: None)
    monkeypatch.setattr(main_module, "_validate_live_trading_readiness", lambda paths, logger, mode: None)

    def fake_run_single(name, target, stop_event, logger, monitor):
        started.append(name)

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    class _DummyLiveController:
        def __init__(self, paths, registry) -> None:
            self.paths = paths
            self.registry = registry

        def run_forever(self, stop_event=None) -> None:
            return None

    monkeypatch.setattr(main_module, "LiveTradingController", _DummyLiveController)

    main_module.main(["--mode", "live"])

    assert started == ["live_trading"]
