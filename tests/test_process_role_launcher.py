from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import scripts.start_process_roles as launcher


class DummyProcess:
    def __init__(self, pid: int = 1234, returncode=None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_role_command_uses_requested_mode_and_log_path(tmp_path: Path) -> None:
    log_path = tmp_path / "run_runtime.log"
    command = launcher._role_command("runtime", log_path)

    assert "main.py --mode runtime" in command
    assert "tee" in command
    assert str(log_path) in command


def test_wait_for_runtime_ready_accepts_fresh_snapshot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    ready_path = tmp_path / "data" / "runtime" / "market" / "latest.json"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text("{}", encoding="utf-8")

    launcher._wait_for_runtime_ready(DummyProcess(), timeout_sec=0.1, strict=True)


def test_startup_sequence_starts_runtime_trainer_dashboard(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    started: list[str] = []

    def fake_ensure_dirs():
        logs = tmp_path / "logs"
        pids = tmp_path / "pids"
        logs.mkdir(parents=True)
        pids.mkdir(parents=True)
        return logs, pids

    def fake_start_role(role, *, terminal_windows, logs_dir, pid_dir):
        started.append(role)
        return None

    monkeypatch.setattr(launcher, "_ensure_dirs", fake_ensure_dirs)
    monkeypatch.setattr(launcher, "_start_role", fake_start_role)
    monkeypatch.setattr(launcher, "_wait_for_runtime_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)

    rc = launcher.main(["--no-wait"])

    assert rc == 0
    assert started == ["runtime", "trainer", "dashboard"]


def test_combined_mode_starts_all_then_dashboard(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    started: list[str] = []
    monkeypatch.setattr(launcher, "_ensure_dirs", lambda: (tmp_path / "logs", tmp_path / "pids"))
    monkeypatch.setattr(launcher, "_start_role", lambda role, **kwargs: started.append(role) or None)
    monkeypatch.setattr(launcher, "_wait_for_runtime_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)

    rc = launcher.main(["--combined-runtime-trainer", "--no-wait"])

    assert rc == 0
    assert started == ["all", "dashboard"]
