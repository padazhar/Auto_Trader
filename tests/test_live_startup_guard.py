from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main as main_module


class DummyLogger:
    """Minimal logger used for startup guard tests."""

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class DummyMonitor:
    """Collect launcher monitor calls without side effects."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.system_events: list[dict[str, object]] = []
        self.flushed = False

    def publish_system(self, **payload) -> None:
        self.system_events.append(payload)

    def controller_error(self, name: str, message: str) -> None:
        pass

    def flush(self) -> None:
        self.flushed = True


class DummyRegistry:
    """Placeholder training registry for startup guard tests."""

    def __init__(self, paths) -> None:
        self.paths = paths


def _fake_paths(project_root: Path, *, execution_mode: str, allow_unreconciled_live_mode: bool) -> SimpleNamespace:
    """Build a minimal paths object for launcher guard tests."""
    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42, is_dev_mode=True),
        runtime=SimpleNamespace(
            execution_mode=execution_mode,
            allow_unreconciled_live_mode=allow_unreconciled_live_mode,
        ),
        telegram=SimpleNamespace(enabled=False),
        monitoring=SimpleNamespace(embedded_in_all_mode=False),
    )
    return SimpleNamespace(project_root=project_root, config=config)


def _patch_common_launcher_dependencies(
    monkeypatch,
    project_root: Path,
    *,
    execution_mode: str,
    allow_unreconciled_live_mode: bool,
) -> SimpleNamespace:
    """Patch shared launcher dependencies with non-destructive test doubles."""
    paths = _fake_paths(
        project_root,
        execution_mode=execution_mode,
        allow_unreconciled_live_mode=allow_unreconciled_live_mode,
    )
    monkeypatch.setattr(main_module, "build_paths", lambda: paths)
    monkeypatch.setattr(main_module, "get_logger", lambda name, paths: DummyLogger())
    monkeypatch.setattr(main_module, "MonitorPublisher", DummyMonitor)
    monkeypatch.setattr(main_module, "TrainingRegistry", DummyRegistry)
    monkeypatch.setattr(main_module, "_validate_startup_secrets", lambda paths, logger: None)
    monkeypatch.setattr(main_module, "_validate_live_trading_readiness", lambda paths, logger, mode: None)
    return paths


def test_main_live_mode_blocks_when_runtime_execution_mode_is_paper(monkeypatch, project_root: Path) -> None:
    """Verify live launcher mode is blocked unless runtime.execution_mode is explicitly live."""
    _patch_common_launcher_dependencies(
        monkeypatch,
        project_root,
        execution_mode="paper",
        allow_unreconciled_live_mode=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(["--mode", "live"])

    assert exc_info.value.code == 3


def test_main_live_mode_blocks_when_unreconciled_live_mode_is_disallowed(monkeypatch, project_root: Path) -> None:
    """Verify live launcher mode is blocked while unreconciled live mode remains disallowed."""
    _patch_common_launcher_dependencies(
        monkeypatch,
        project_root,
        execution_mode="live",
        allow_unreconciled_live_mode=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main(["--mode", "live"])

    assert exc_info.value.code == 4


def test_main_all_mode_is_allowed_in_paper_execution_mode(monkeypatch, project_root: Path) -> None:
    """Verify paper all-mode remains allowed with dashboard isolated by default."""
    _patch_common_launcher_dependencies(
        monkeypatch,
        project_root,
        execution_mode="paper",
        allow_unreconciled_live_mode=False,
    )

    started: list[tuple[str, bool]] = []
    supervision: list[list[tuple[str, bool]]] = []

    def fake_start_controller(runtime, stop_event, logger, monitor) -> None:
        started.append((runtime.name, runtime.critical))

    def fake_supervise(runtimes, stop_event, monitor, logger) -> None:
        supervision.append([(runtime.name, runtime.critical) for runtime in runtimes])

    monkeypatch.setattr(main_module, "_start_controller", fake_start_controller)
    monkeypatch.setattr(main_module, "_supervise_controllers", fake_supervise)

    main_module.main(["--mode", "all"])

    assert started == [
        ("market_data", True),
        ("tech_rev", True),
        ("signal_generator", True),
        ("live_trading", True),
        ("backtest", False),
        ("state_bridge", True),
    ]
    assert supervision == [started]
