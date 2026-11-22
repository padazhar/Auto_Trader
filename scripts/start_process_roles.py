#!/usr/bin/env python3
"""Start Auto_Trader process roles in the recommended order.

Default startup order:
1. runtime   -> market data, signal, execution, snapshots
2. trainer   -> training/promotion/requalification
3. dashboard -> UI only

The launcher can either open separate terminal windows or run detached child
processes with per-role logs and PID files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable or "python3"

RUNTIME_READY_FILES = (
    Path("data/runtime/market/latest.json"),
    Path("data/runtime/runtime_status.json"),
    Path("data/runtime/monitoring_snapshot.json"),
    Path("data/runtime/signal_readiness.json"),
    Path("data/runtime/paper_execution_runtime.json"),
)


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _role_command(role: str, log_path: Path) -> str:
    return f"cd {_shell_quote(str(PROJECT_ROOT))} && {PYTHON} -u main.py --mode {role} 2>&1 | tee {_shell_quote(str(log_path))}"


def _find_terminal() -> list[str] | None:
    candidates = (
        ("gnome-terminal", ["gnome-terminal", "--", "bash", "-lc"]),
        ("konsole", ["konsole", "--noclose", "-e", "bash", "-lc"]),
        ("xfce4-terminal", ["xfce4-terminal", "--hold", "-e"]),
        ("xterm", ["xterm", "-hold", "-e"]),
    )
    for binary, command in candidates:
        if shutil.which(binary):
            return command
    return None


def _ensure_dirs() -> tuple[Path, Path]:
    logs_dir = PROJECT_ROOT / "logs" / "process_roles"
    pid_dir = PROJECT_ROOT / "data" / "runtime" / "process_roles"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, pid_dir


def _write_pid(pid_dir: Path, role: str, pid: int | None) -> None:
    if pid is None:
        return
    (pid_dir / f"{role}.pid").write_text(str(pid), encoding="utf-8")


def _start_role(role: str, *, terminal_windows: bool, logs_dir: Path, pid_dir: Path) -> subprocess.Popen | None:
    log_path = logs_dir / f"run_{role}.log"
    command = _role_command(role, log_path)
    if terminal_windows:
        terminal = _find_terminal()
        if terminal is None:
            print("No supported terminal emulator found; falling back to detached background process.")
        else:
            subprocess.Popen(terminal + [command], cwd=PROJECT_ROOT, start_new_session=True)
            print(f"Started {role} in a terminal window. log={log_path}")
            return None

    log_handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        [PYTHON, "-u", "main.py", "--mode", role],
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid(pid_dir, role, process.pid)
    print(f"Started {role} pid={process.pid}. log={log_path}")
    return process


def _runtime_evidence_exists(paths: Iterable[Path]) -> tuple[bool, Path | None]:
    for rel_path in paths:
        path = PROJECT_ROOT / rel_path
        try:
            if path.exists():
                return True, path
        except OSError:
            continue
    return False, None


def _process_exited(process: subprocess.Popen | None) -> bool:
    return process is not None and process.poll() is not None


def _wait_for_runtime_ready(process: subprocess.Popen | None, *, timeout_sec: float, strict: bool) -> None:
    deadline = time.time() + max(0.0, timeout_sec)
    print(f"Waiting for runtime snapshots for up to {timeout_sec:.0f}s...")
    while time.time() <= deadline:
        if _process_exited(process):
            raise SystemExit(f"runtime process exited early with code {process.returncode}; check logs/process_roles/run_runtime.log")
        ready, ready_path = _runtime_evidence_exists(RUNTIME_READY_FILES)
        if ready:
            print(f"Runtime ready evidence detected: {ready_path}")
            return
        time.sleep(1.0)
    message = "runtime snapshot readiness evidence was not detected before timeout"
    if strict:
        raise SystemExit(message)
    print(f"WARNING: {message}; continuing because strict wait is disabled.")


def _install_signal_forwarding(processes: list[subprocess.Popen]) -> None:
    def _handler(signum, frame):  # noqa: ANN001
        print("Stopping child processes...")
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signum)
                except Exception:
                    process.terminate()
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start Auto_Trader runtime, trainer, and dashboard in order.")
    parser.add_argument("--terminal-windows", action="store_true", help="Open each role in its own terminal window when available.")
    parser.add_argument("--combined-runtime-trainer", action="store_true", help="Start --mode all plus dashboard instead of runtime + trainer + dashboard.")
    parser.add_argument("--runtime-ready-timeout-sec", type=float, default=120.0, help="Maximum wait for runtime snapshot evidence before trainer starts.")
    parser.add_argument("--trainer-delay-sec", type=float, default=5.0, help="Delay after trainer start before dashboard starts.")
    parser.add_argument("--strict-runtime-wait", action="store_true", help="Abort if runtime snapshots are not detected before timeout.")
    parser.add_argument("--no-wait", action="store_true", help="Start roles and exit instead of supervising background child processes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logs_dir, pid_dir = _ensure_dirs()
    roles = ["all"] if args.combined_runtime_trainer else ["runtime", "trainer"]

    processes: list[subprocess.Popen] = []
    runtime_start_role = roles[0]
    runtime_process = _start_role(runtime_start_role, terminal_windows=args.terminal_windows, logs_dir=logs_dir, pid_dir=pid_dir)
    if runtime_process is not None:
        processes.append(runtime_process)

    _wait_for_runtime_ready(runtime_process, timeout_sec=args.runtime_ready_timeout_sec, strict=args.strict_runtime_wait)

    if not args.combined_runtime_trainer:
        trainer_process = _start_role("trainer", terminal_windows=args.terminal_windows, logs_dir=logs_dir, pid_dir=pid_dir)
        if trainer_process is not None:
            processes.append(trainer_process)
        time.sleep(max(0.0, args.trainer_delay_sec))

    dashboard_process = _start_role("dashboard", terminal_windows=args.terminal_windows, logs_dir=logs_dir, pid_dir=pid_dir)
    if dashboard_process is not None:
        processes.append(dashboard_process)

    print("Startup order complete.")
    print("Logs: logs/process_roles/run_runtime.log, run_trainer.log, run_dashboard.log")
    if args.no_wait or args.terminal_windows:
        return 0

    _install_signal_forwarding(processes)
    while True:
        for process in processes:
            if process.poll() is not None:
                raise SystemExit(f"child process pid={process.pid} exited with code {process.returncode}")
        time.sleep(2.0)


if __name__ == "__main__":
    raise SystemExit(main())
