from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Iterable


@dataclass(slots=True)
class ShutdownState:
    """Small immutable-ish status payload for launcher shutdown coordination."""

    requested: bool = False
    reason: str = ""
    requested_at: float | None = None
    request_count: int = 0
    joined: list[str] = field(default_factory=list)
    still_alive: list[str] = field(default_factory=list)
    flushed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "request_count": self.request_count,
            "joined": list(self.joined),
            "still_alive": list(self.still_alive),
            "flushed": self.flushed,
        }


class ShutdownCoordinator:
    """Coordinate a paper-safe, fail-closed launcher shutdown.

    The coordinator intentionally does not close live exchange positions. Step 3
    only establishes the stop/flush contract. Live position handling remains
    fail-closed and is governed by later explicit validation gates.
    """

    def __init__(
        self,
        *,
        stop_event: threading.Event,
        monitor: Any | None = None,
        logger: Any | None = None,
        join_timeout_sec: float = 5.0,
    ) -> None:
        self.stop_event = stop_event
        self.monitor = monitor
        self.logger = logger
        self.join_timeout_sec = max(0.1, float(join_timeout_sec or 5.0))
        self.state = ShutdownState()
        self._lock = threading.RLock()
        self._installed_handlers: dict[int, Any] = {}

    def install_signal_handlers(self, signals: Iterable[int] | None = None) -> None:
        """Install SIGINT/SIGTERM handlers when running on the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        selected = tuple(signals or (signal.SIGINT, signal.SIGTERM))
        for signum in selected:
            try:
                previous = signal.getsignal(signum)
                self._installed_handlers[int(signum)] = previous
                signal.signal(signum, self._handle_signal)
            except (ValueError, OSError, RuntimeError):
                continue

    def restore_signal_handlers(self) -> None:
        """Restore previous signal handlers after launcher finalization."""
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, previous in list(self._installed_handlers.items()):
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, RuntimeError):
                pass
        self._installed_handlers.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        """Record OS shutdown signals without performing blocking I/O.

        Signal handlers execute on the main Python thread.  Calling the normal
        request path from here can block on logging or the cross-process
        monitoring file lock, preventing the launcher from ever reaching its
        normal ``finally`` shutdown path.  Keep the handler limited to local
        state mutation and the shared stop event; normal finalization records
        the shutdown reason in the run report.
        """
        signame = getattr(signal.Signals(signum), "name", str(signum))
        reason = f"signal:{signame}"
        with self._lock:
            self.state.request_count += 1
            if not self.state.requested:
                self.state.requested = True
                self.state.reason = reason
                self.state.requested_at = time.time()
            self.stop_event.set()

    def request_shutdown(self, reason: str = "requested") -> ShutdownState:
        """Idempotently request shutdown and stop new controller work."""
        reason = str(reason or "requested")
        first_request = False
        with self._lock:
            self.state.request_count += 1
            if not self.state.requested:
                first_request = True
                self.state.requested = True
                self.state.reason = reason
                self.state.requested_at = time.time()
            self.stop_event.set()

        if first_request:
            self._log("warning", "Shutdown requested: %s", reason)
            self._publish_event(
                level="WARNING",
                component="runtime",
                category="shutdown_requested",
                message="Runtime shutdown requested",
                data={"reason": reason},
                alert=False,
            )
        else:
            self._log("warning", "Shutdown request repeated: %s", reason)
        return self.state

    def finalize(self, runtimes: Iterable[Any] | None = None) -> ShutdownState:
        """Stop controllers, join threads, and flush monitoring snapshots."""
        self.request_shutdown(self.state.reason or "launcher_finally")
        current = threading.current_thread()
        joined: list[str] = []
        still_alive: list[str] = []

        for runtime in list(runtimes or []):
            name = str(getattr(runtime, "name", "controller") or "controller")
            thread = getattr(runtime, "thread", None)
            if thread is None or thread is current:
                continue
            join = getattr(thread, "join", None)
            is_alive = getattr(thread, "is_alive", None)
            if callable(join):
                try:
                    join(timeout=self.join_timeout_sec)
                except RuntimeError:
                    pass
            alive = bool(is_alive()) if callable(is_alive) else False
            if alive:
                still_alive.append(name)
                self._publish_event(
                    level="WARNING",
                    component="runtime",
                    category="shutdown_join_timeout",
                    message="Controller did not stop before shutdown join timeout",
                    data={"controller": name, "join_timeout_sec": self.join_timeout_sec},
                    alert=True,
                )
            else:
                joined.append(name)

        with self._lock:
            self.state.joined = joined
            self.state.still_alive = still_alive

        # A signal-triggered shutdown must never block process termination on
        # observability persistence. MonitorPublisher serialises writes with a
        # cross-process file lock shared by runtime/trainer/dashboard roles; the
        # failed 1.4 runtime demonstrated that this lock can remain contended
        # while the other roles are still active. Runtime state is already
        # persisted continuously and the final run report carries ShutdownState.
        signal_requested = str(self.state.reason or "").startswith("signal:")
        if not signal_requested:
            self._flush_monitor()
        self.restore_signal_handlers()
        return self.state

    def _flush_monitor(self) -> None:
        flush = getattr(self.monitor, "flush", None)
        if not callable(flush):
            return
        try:
            flush()
            with self._lock:
                self.state.flushed = True
        except Exception as exc:  # pragma: no cover - defensive logging only
            self._log("exception", "Monitoring flush failed during shutdown: %s", exc)

    def _publish_event(self, **payload: Any) -> None:
        publisher = getattr(self.monitor, "publish_event", None)
        if not callable(publisher):
            return
        try:
            publisher(**payload)
        except Exception:
            pass

    def _log(self, level: str, message: str, *args: Any) -> None:
        method = getattr(self.logger, level, None)
        if callable(method):
            method(message, *args)
