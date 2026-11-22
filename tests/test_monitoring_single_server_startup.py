from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import main as main_module
from interface.terminal_dashboard import TerminalDashboard


def _make_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=tmp_path,
        config=SimpleNamespace(monitoring=SimpleNamespace(host="127.0.0.1", port=8000)),
    )


def test_terminal_dashboard_delegates_to_monitoring_server(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr("interface.terminal_dashboard.run_monitoring_server", lambda paths, stop_event=None: calls.append((paths, stop_event)))

    dashboard = TerminalDashboard(_make_paths(tmp_path))
    stop_event = threading.Event()
    dashboard.run_file_backed(stop_event=stop_event)

    assert calls == [(dashboard.paths, stop_event)]


def test_dashboard_mode_aliases_monitoring_runtime(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(main_module, "build_paths", lambda: _make_paths(tmp_path))
    monkeypatch.setattr(main_module, "_validate_startup_secrets", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "_validate_live_trading_readiness", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_module, "get_logger", lambda *args, **kwargs: SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None, exception=lambda *a, **k: None))
    monkeypatch.setattr(main_module, "MonitorPublisher", lambda paths: SimpleNamespace(publish_system=lambda **k: None, controller_error=lambda *a, **k: None, close=lambda: None))
    monkeypatch.setattr(main_module, "TrainingRegistry", lambda path: object())
    monkeypatch.setattr(main_module, "_monitoring_target", lambda paths, monitor=None, logger=None: (lambda stop_event: calls.append("monitoring")))
    monkeypatch.setattr(main_module, "_run_single", lambda name, target, stop_event, logger, monitor: target(stop_event))
    monkeypatch.setattr(main_module.argparse.ArgumentParser, "parse_args", lambda self, *args, **kwargs: SimpleNamespace(mode="dashboard"))

    main_module.main()

    assert calls == ["monitoring"]
