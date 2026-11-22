from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import main as main_module


class _DummyMonitor:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.heartbeats: list[tuple[str, str, dict]] = []

    def publish_event(self, *, level: str, component: str, category: str, message: str, data=None, alert=False) -> None:
        self.events.append(
            {
                "level": level,
                "component": component,
                "category": category,
                "message": message,
                "data": dict(data or {}),
                "alert": bool(alert),
            }
        )

    def heartbeat(self, name: str, *, status: str, details=None) -> None:
        self.heartbeats.append((name, status, dict(details or {})))


class _DummyLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def warning(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.warnings.append(message)

    def info(self, message: str, *args) -> None:
        if args:
            message = message % args
        self.infos.append(message)

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def _make_paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            monitoring=SimpleNamespace(host="127.0.0.1", port=8000),
        )
    )


def test_monitoring_target_reuses_existing_listener_without_dying(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    monitor = _DummyMonitor()
    logger = _DummyLogger()

    monkeypatch.setattr(main_module, "_monitoring_bind_status", lambda paths: ("listening", "existing listener detected on 127.0.0.1:8000"))
    monkeypatch.setattr(main_module, "_monitoring_listener_is_compatible", lambda paths: (True, ""))

    target = main_module._monitoring_target(paths, monitor=monitor, logger=logger)
    stop_event = threading.Event()
    thread = threading.Thread(target=target, args=(stop_event,))
    thread.start()
    time.sleep(0.05)
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert any(event["category"] == "monitoring_external" for event in monitor.events)
    assert any(status == "external" for _, status, _ in monitor.heartbeats)
    assert any("already running externally" in info for info in logger.infos)


def test_monitoring_target_degrades_when_server_start_raises_oserror(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    monitor = _DummyMonitor()
    logger = _DummyLogger()

    states = iter([("available", ""), ("listening", "existing listener detected on 127.0.0.1:8000")])
    monkeypatch.setattr(main_module, "_monitoring_bind_status", lambda paths: next(states))
    monkeypatch.setattr(main_module, "_monitoring_listener_is_compatible", lambda paths: (True, ""))

    def _boom(paths, stop_event=None):
        raise OSError("[Errno 98] address already in use")

    monkeypatch.setattr(main_module, "run_monitoring_server", _boom)

    target = main_module._monitoring_target(paths, monitor=monitor, logger=logger)
    stop_event = threading.Event()
    thread = threading.Thread(target=target, args=(stop_event,))
    thread.start()
    time.sleep(0.05)
    stop_event.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert any(event["message"] == "monitoring_external_listener_reused" for event in monitor.events)
    assert any(status == "external" for _, status, _ in monitor.heartbeats)


def test_monitoring_target_flags_incompatible_existing_listener(monkeypatch, tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    monitor = _DummyMonitor()
    logger = _DummyLogger()
    monkeypatch.setattr(main_module, "_monitoring_bind_status", lambda paths: ("listening", "existing listener detected on 127.0.0.1:8000"))
    monkeypatch.setattr(main_module, "_monitoring_listener_is_compatible", lambda paths: (False, "existing listener belongs to a different project root"))

    target = main_module._monitoring_target(paths, monitor=monitor, logger=logger)
    stop_event = threading.Event()
    thread = threading.Thread(target=target, args=(stop_event,), daemon=True)
    thread.start()
    time.sleep(0.1)
    stop_event.set()
    thread.join(timeout=1.0)

    assert any(event["message"] == "monitoring_external_listener_incompatible" for event in monitor.events)
    assert any(status == "degraded" for _, status, _ in monitor.heartbeats)
