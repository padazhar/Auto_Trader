from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import main as main_module


class DummyLogger:
    """Minimal logger used for launcher smoke tests."""

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class DummyMonitor:
    """Collect monitor lifecycle calls during launcher smoke tests."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.system_events: list[dict[str, object]] = []
        self.controller_errors: list[tuple[str, str]] = []
        self.flushed = False

    def publish_system(self, **payload) -> None:
        self.system_events.append(payload)

    def controller_error(self, name: str, message: str) -> None:
        self.controller_errors.append((name, message))

    def flush(self) -> None:
        self.flushed = True


class DummyRegistry:
    """Placeholder training registry for launcher smoke tests."""

    def __init__(self, paths) -> None:
        self.paths = paths


class _Runner:
    """Simple controller stub exposing the expected run methods."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run_forever(self, stop_event) -> None:
        stop_event.set()

    def run_file_backed(self, stop_event) -> None:
        stop_event.set()


class _ControllerFactory:
    """Controller constructor shim that ignores runtime args."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args, **kwargs):
        return _Runner(self.name)


def _fake_paths(project_root: Path) -> SimpleNamespace:
    """Build a minimal paths object for launcher smoke tests."""
    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42, is_dev_mode=True),
        runtime=SimpleNamespace(execution_mode="paper"),
        telegram=SimpleNamespace(enabled=False),
        monitoring=SimpleNamespace(embedded_in_all_mode=False),
    )
    return SimpleNamespace(project_root=project_root, config=config)


def _patch_common_launcher_dependencies(monkeypatch, project_root: Path) -> SimpleNamespace:
    """Patch shared launcher dependencies with non-destructive test doubles."""
    paths = _fake_paths(project_root)
    monkeypatch.setattr(main_module, "build_paths", lambda: paths)
    monkeypatch.setattr(main_module, "get_logger", lambda name, paths: DummyLogger())
    monkeypatch.setattr(main_module, "MonitorPublisher", DummyMonitor)
    monkeypatch.setattr(main_module, "TrainingRegistry", DummyRegistry)
    monkeypatch.setattr(main_module, "_validate_startup_secrets", lambda paths, logger: None)
    return paths


def test_main_trainer_mode_dispatches_tech_rev_controller(monkeypatch, project_root: Path) -> None:
    """Verify trainer alias dispatches the canonical tech_rev training controller."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("trainer"))

    calls: list[dict[str, object]] = []

    def fake_run_single(name, target, stop_event, logger, monitor) -> None:
        calls.append({"name": name, "target": target, "stop_event": stop_event, "monitor": monitor})

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    main_module.main(["--mode", "trainer"])

    assert len(calls) == 1
    assert calls[0]["name"] == "tech_rev"
    assert callable(calls[0]["target"])


def test_main_tech_rev_mode_remains_canonical_trainer_controller(monkeypatch, project_root: Path) -> None:
    """Verify tech_rev remains the canonical internal training-controller name."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("trainer"))

    calls: list[str] = []
    monkeypatch.setattr(main_module, "_run_single", lambda name, target, stop_event, logger, monitor: calls.append(name))

    main_module.main(["--mode", "tech_rev"])

    assert calls == ["tech_rev"]


def test_main_signal_mode_dispatches_signal_controller(monkeypatch, project_root: Path) -> None:
    """Verify signal mode dispatches the signal controller without starting other runtimes."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)
    monkeypatch.setattr(main_module, "SignalGeneratorController", _ControllerFactory("signal_generator"))

    calls: list[dict[str, object]] = []

    def fake_run_single(name, target, stop_event, logger, monitor) -> None:
        calls.append({"name": name, "target": target, "stop_event": stop_event, "monitor": monitor})

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    main_module.main(["--mode", "signal"])

    assert len(calls) == 1
    assert calls[0]["name"] == "signal_generator"
    assert callable(calls[0]["target"])


def test_main_monitor_mode_dispatches_monitoring_target(monkeypatch, project_root: Path) -> None:
    """Verify monitor mode dispatches the monitoring controller target."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)

    calls: list[dict[str, object]] = []

    def fake_run_single(name, target, stop_event, logger, monitor) -> None:
        calls.append({"name": name, "target": target, "stop_event": stop_event, "monitor": monitor})

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    main_module.main(["--mode", "monitor"])

    assert len(calls) == 1
    assert calls[0]["name"] == "monitoring"
    assert callable(calls[0]["target"])




def test_main_dashboard_mode_dispatches_monitoring_target(monkeypatch, project_root: Path) -> None:
    """Verify dashboard mode routes to the monitoring runtime instead of the passive dashboard shim."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)

    calls: list[dict[str, object]] = []

    def fake_run_single(name, target, stop_event, logger, monitor) -> None:
        calls.append({"name": name, "target": target, "stop_event": stop_event, "monitor": monitor})

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    main_module.main(["--mode", "dashboard"])

    assert len(calls) == 1
    assert calls[0]["name"] == "monitoring"
    assert callable(calls[0]["target"])


def test_main_monitoring_mode_dispatches_monitoring_target(monkeypatch, project_root: Path) -> None:
    """Verify the explicit monitoring alias dispatches the monitoring controller target."""
    _patch_common_launcher_dependencies(monkeypatch, project_root)

    calls: list[dict[str, object]] = []

    def fake_run_single(name, target, stop_event, logger, monitor) -> None:
        calls.append({"name": name, "target": target, "stop_event": stop_event, "monitor": monitor})

    monkeypatch.setattr(main_module, "_run_single", fake_run_single)

    main_module.main(["--mode", "monitoring"])

    assert len(calls) == 1
    assert calls[0]["name"] == "monitoring"
    assert callable(calls[0]["target"])

def test_main_runtime_mode_builds_runtime_only_controller_set(monkeypatch, project_root: Path) -> None:
    """Verify runtime mode excludes trainer/dashboard while retaining paper backtest publication."""
    paths = _patch_common_launcher_dependencies(monkeypatch, project_root)
    monkeypatch.setattr(main_module, "MarketDataController", _ControllerFactory("market_data"))
    monkeypatch.setattr(main_module, "SignalGeneratorController", _ControllerFactory("signal_generator"))
    monkeypatch.setattr(main_module, "LiveTradingController", _ControllerFactory("live_trading"))
    monkeypatch.setattr(main_module, "StateBridgeController", _ControllerFactory("state_bridge"))
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("trainer"))
    monkeypatch.setattr(main_module, "BacktestController", _ControllerFactory("backtest"))

    started: list[tuple[str, bool]] = []
    monkeypatch.setattr(main_module, "_start_controller", lambda runtime, stop_event, logger, monitor: started.append((runtime.name, runtime.critical)))
    monkeypatch.setattr(main_module, "_supervise_controllers", lambda runtimes, stop_event, monitor, logger: None)

    main_module.main(["--mode", "runtime"])

    assert paths.config.telegram.enabled is False
    assert started == [
        ("market_data", True),
        ("signal_generator", True),
        ("live_trading", True),
        ("backtest", False),
        ("state_bridge", True),
    ]


def test_main_all_mode_builds_expected_controller_set(monkeypatch, project_root: Path) -> None:
    """Verify all mode assembles the expected controller runtimes and criticality tiers."""
    paths = _patch_common_launcher_dependencies(monkeypatch, project_root)
    monkeypatch.setattr(main_module, "MarketDataController", _ControllerFactory("market_data"))
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("tech_rev"))
    monkeypatch.setattr(main_module, "SignalGeneratorController", _ControllerFactory("signal_generator"))
    monkeypatch.setattr(main_module, "LiveTradingController", _ControllerFactory("live_trading"))
    monkeypatch.setattr(main_module, "BacktestController", _ControllerFactory("backtest"))
    monkeypatch.setattr(main_module, "StateBridgeController", _ControllerFactory("state_bridge"))
    monkeypatch.setattr(main_module, "TerminalDashboard", _ControllerFactory("dashboard"))
    monkeypatch.setattr(main_module, "TelegramController", _ControllerFactory("telegram"))

    started: list[tuple[str, bool]] = []
    supervision: list[list[tuple[str, bool]]] = []

    def fake_start_controller(runtime, stop_event, logger, monitor) -> None:
        started.append((runtime.name, runtime.critical))

    def fake_supervise(runtimes, stop_event, monitor, logger) -> None:
        supervision.append([(runtime.name, runtime.critical) for runtime in runtimes])

    monkeypatch.setattr(main_module, "_start_controller", fake_start_controller)
    monkeypatch.setattr(main_module, "_supervise_controllers", fake_supervise)

    main_module.main(["--mode", "all"])

    assert paths.config.telegram.enabled is False
    assert started == [
        ("market_data", True),
        ("tech_rev", True),
        ("signal_generator", True),
        ("live_trading", True),
        ("backtest", False),
        ("state_bridge", True),
    ]
    assert supervision == [started]


def test_main_all_mode_can_opt_in_to_embedded_monitoring(monkeypatch, project_root: Path) -> None:
    paths = _patch_common_launcher_dependencies(monkeypatch, project_root)
    paths.config.monitoring.embedded_in_all_mode = True
    monkeypatch.setattr(main_module, "MarketDataController", _ControllerFactory("market_data"))
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("tech_rev"))
    monkeypatch.setattr(main_module, "SignalGeneratorController", _ControllerFactory("signal_generator"))
    monkeypatch.setattr(main_module, "LiveTradingController", _ControllerFactory("live_trading"))
    monkeypatch.setattr(main_module, "BacktestController", _ControllerFactory("backtest"))
    monkeypatch.setattr(main_module, "StateBridgeController", _ControllerFactory("state_bridge"))

    started: list[str] = []
    monkeypatch.setattr(main_module, "_start_controller", lambda runtime, stop_event, logger, monitor: started.append(runtime.name))
    monkeypatch.setattr(main_module, "_supervise_controllers", lambda runtimes, stop_event, monitor, logger: None)

    main_module.main(["--mode", "all"])

    assert "monitoring" in started


def test_main_all_mode_skips_live_trading_runtime_when_execution_mode_is_live(monkeypatch, project_root: Path) -> None:
    paths = _patch_common_launcher_dependencies(monkeypatch, project_root)
    paths.config.runtime.execution_mode = "live"
    monkeypatch.setattr(main_module, "MarketDataController", _ControllerFactory("market_data"))
    monkeypatch.setattr(main_module, "TechRevController", _ControllerFactory("tech_rev"))
    monkeypatch.setattr(main_module, "SignalGeneratorController", _ControllerFactory("signal_generator"))
    monkeypatch.setattr(main_module, "LiveTradingController", _ControllerFactory("live_trading"))
    monkeypatch.setattr(main_module, "BacktestController", _ControllerFactory("backtest"))
    monkeypatch.setattr(main_module, "StateBridgeController", _ControllerFactory("state_bridge"))

    started: list[str] = []

    monkeypatch.setattr(main_module, "_start_controller", lambda runtime, stop_event, logger, monitor: started.append(runtime.name))
    monkeypatch.setattr(main_module, "_supervise_controllers", lambda runtimes, stop_event, monitor, logger: None)

    main_module.main(["--mode", "all"])

    assert "live_trading" not in started
