from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main as main_module


class DummyLogger:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.infos.append(str(message))

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, message, *args, **kwargs) -> None:
        self.errors.append(str(message))

    def exception(self, *args, **kwargs) -> None:
        pass


class DummyReport:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_status": "pass" if self.ready else "fail",
            "live_trading_ready": self.ready,
            "items": [{"name": "stub_gate", "status": "pass" if self.ready else "fail"}],
        }


def _fake_paths(project_root: Path, execution_mode: str = "live") -> SimpleNamespace:
    config = SimpleNamespace(
        runtime=SimpleNamespace(execution_mode=execution_mode),
    )
    return SimpleNamespace(project_root=project_root, config=config)


def test_validate_live_trading_readiness_blocks_failed_live_gate(monkeypatch, project_root: Path) -> None:
    logger = DummyLogger()
    paths = _fake_paths(project_root, execution_mode="live")
    monkeypatch.setattr(main_module, "run_live_trading_readiness_check", lambda project_root: DummyReport(False))

    with pytest.raises(SystemExit) as exc_info:
        main_module._validate_live_trading_readiness(paths, logger, "live")

    assert exc_info.value.code == 6


def test_validate_live_trading_readiness_skips_non_live_execution_mode(monkeypatch, project_root: Path) -> None:
    logger = DummyLogger()
    paths = _fake_paths(project_root, execution_mode="paper")
    called: list[Path] = []

    def fake_check(project_root):
        called.append(Path(project_root))
        return DummyReport(True)

    monkeypatch.setattr(main_module, "run_live_trading_readiness_check", fake_check)

    main_module._validate_live_trading_readiness(paths, logger, "all")

    assert called == []


def test_validate_live_trading_readiness_skips_all_mode_even_when_runtime_execution_mode_is_live(monkeypatch, project_root: Path) -> None:
    logger = DummyLogger()
    paths = _fake_paths(project_root, execution_mode="live")
    called: list[Path] = []

    def fake_check(project_root):
        called.append(Path(project_root))
        return DummyReport(True)

    monkeypatch.setattr(main_module, "run_live_trading_readiness_check", fake_check)

    main_module._validate_live_trading_readiness(paths, logger, "all")

    assert called == []
    assert logger.errors == []


def test_validate_live_trading_readiness_allows_passed_live_gate(monkeypatch, project_root: Path) -> None:
    logger = DummyLogger()
    paths = _fake_paths(project_root, execution_mode="live")
    monkeypatch.setattr(main_module, "run_live_trading_readiness_check", lambda project_root: DummyReport(True))

    main_module._validate_live_trading_readiness(paths, logger, "live")

    assert logger.errors == []
