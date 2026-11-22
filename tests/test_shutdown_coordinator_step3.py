from __future__ import annotations

import asyncio
import signal
import threading
import time
from types import SimpleNamespace

import main as main_module
from runners.market_data_controller import MarketDataController
import runners.market_data_controller as market_data_module
from runners.signal_generator_controller import SignalGeneratorController
import runners.signal_generator_controller as signal_generator_module
from runtime.shutdown_coordinator import ShutdownCoordinator


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def warning(self, message, *args) -> None:
        self.messages.append(("warning", message % args if args else message))

    def exception(self, message, *args) -> None:
        self.messages.append(("exception", message % args if args else message))

    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class DummyMonitor:
    def __init__(self, paths=None) -> None:
        self.events: list[dict] = []
        self.flushed = False
        self.system_events: list[dict] = []

    def publish_event(self, **payload) -> None:
        self.events.append(payload)

    def publish_system(self, **payload) -> None:
        self.system_events.append(payload)

    def controller_error(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        self.flushed = True


def test_shutdown_coordinator_is_idempotent_and_flushes_monitor() -> None:
    stop_event = threading.Event()
    monitor = DummyMonitor()
    logger = DummyLogger()
    coordinator = ShutdownCoordinator(stop_event=stop_event, monitor=monitor, logger=logger, join_timeout_sec=0.1)

    first = coordinator.request_shutdown("unit_test")
    second = coordinator.request_shutdown("repeat")
    final = coordinator.finalize([])

    assert first is second
    assert stop_event.is_set()
    assert final.requested is True
    assert final.reason == "unit_test"
    assert final.request_count == 3
    assert final.flushed is True
    assert monitor.events[0]["category"] == "shutdown_requested"


def test_shutdown_coordinator_reports_thread_join_timeout() -> None:
    stop_event = threading.Event()
    monitor = DummyMonitor()
    coordinator = ShutdownCoordinator(stop_event=stop_event, monitor=monitor, join_timeout_sec=0.01)
    blocker = threading.Event()

    def target() -> None:
        blocker.wait(0.25)

    thread = threading.Thread(target=target, name="blocked-controller")
    thread.start()
    try:
        final = coordinator.finalize([SimpleNamespace(name="blocked", thread=thread)])
    finally:
        blocker.set()
        thread.join(timeout=1.0)

    assert "blocked" in final.still_alive
    assert any(event.get("category") == "shutdown_join_timeout" for event in monitor.events)


def test_main_uses_shutdown_coordinator_for_all_mode(monkeypatch, project_root) -> None:
    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42, is_dev_mode=True),
        runtime=SimpleNamespace(execution_mode="paper"),
        telegram=SimpleNamespace(enabled=False),
        monitoring=SimpleNamespace(embedded_in_all_mode=False),
    )
    paths = SimpleNamespace(project_root=project_root, config=config)
    monitor = DummyMonitor(paths)
    logger = DummyLogger()
    finalized: list[list[str]] = []

    monkeypatch.setattr(main_module, "build_paths", lambda: paths)
    monkeypatch.setattr(main_module, "get_logger", lambda name, paths: logger)
    monkeypatch.setattr(main_module, "MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr(main_module, "TrainingRegistry", lambda paths: object())
    monkeypatch.setattr(main_module, "_validate_startup_secrets", lambda paths, logger: None)
    monkeypatch.setattr(main_module.ShutdownCoordinator, "install_signal_handlers", lambda self: None)

    def fake_start_controller(runtime, stop_event, logger, monitor) -> None:
        runtime.thread = threading.Thread(target=lambda: None, name=runtime.name)
        runtime.thread.start()

    def fake_supervise(runtimes, stop_event, monitor, logger) -> None:
        stop_event.set()

    original_finalize = main_module.ShutdownCoordinator.finalize

    def recording_finalize(self, runtimes):
        finalized.append([runtime.name for runtime in runtimes])
        return original_finalize(self, runtimes)

    monkeypatch.setattr(main_module, "_start_controller", fake_start_controller)
    monkeypatch.setattr(main_module, "_supervise_controllers", fake_supervise)
    monkeypatch.setattr(main_module.ShutdownCoordinator, "finalize", recording_finalize)

    main_module.main(["--mode", "all"])

    assert finalized == [["market_data", "tech_rev", "signal_generator", "live_trading", "backtest", "state_bridge"]]
    assert monitor.flushed is True



class BlockingMonitor(DummyMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.publish_entered = threading.Event()
        self.release_publish = threading.Event()

    def publish_event(self, **payload) -> None:
        self.publish_entered.set()
        self.release_publish.wait(5.0)
        super().publish_event(**payload)


def test_signal_handler_does_not_block_on_monitoring_io() -> None:
    stop_event = threading.Event()
    monitor = BlockingMonitor()
    coordinator = ShutdownCoordinator(stop_event=stop_event, monitor=monitor, join_timeout_sec=0.1)

    started = time.monotonic()
    coordinator._handle_signal(signal.SIGINT, None)
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert stop_event.is_set()
    assert coordinator.state.requested is True
    assert coordinator.state.reason == "signal:SIGINT"
    assert monitor.publish_entered.is_set() is False


def test_signal_shutdown_skips_blocking_monitor_flush() -> None:
    class BlockingFlushMonitor(DummyMonitor):
        def flush(self) -> None:
            raise AssertionError("signal finalization must not call blocking monitor.flush")

    stop_event = threading.Event()
    monitor = BlockingFlushMonitor()
    coordinator = ShutdownCoordinator(stop_event=stop_event, monitor=monitor, join_timeout_sec=0.1)
    coordinator._handle_signal(signal.SIGINT, None)

    final = coordinator.finalize([])

    assert final.reason == "signal:SIGINT"
    assert final.flushed is False

def test_market_data_run_async_cancels_workers_promptly_on_stop_event() -> None:
    controller = object.__new__(MarketDataController)
    controller.enabled = True
    controller.logger = DummyLogger()
    controller.monitor = DummyMonitor()
    controller.refresh_universe_enabled = False
    controller.fast_frames = []
    controller.slow_frames = []

    cancelled = {"runtime": False, "bootstrap": False}

    async def _sync_server_time() -> None:
        return None

    async def _refresh_runtime_price_snapshot(*args, **kwargs):
        return {}

    async def _blocking_runtime_snapshot_loop(stop_event=None) -> None:
        try:
            await asyncio.sleep(60.0)
        finally:
            cancelled["runtime"] = True

    async def _blocking_bootstrap_loop(stop_event=None) -> None:
        try:
            await asyncio.sleep(60.0)
        finally:
            cancelled["bootstrap"] = True

    controller.sync_server_time = _sync_server_time
    controller._publish_connectivity = lambda: None
    controller.refresh_runtime_price_snapshot = _refresh_runtime_price_snapshot
    controller.runtime_snapshot_loop = _blocking_runtime_snapshot_loop
    controller.active_history_bootstrap_loop = _blocking_bootstrap_loop

    stop_event = threading.Event()

    async def _exercise() -> None:
        task = asyncio.create_task(controller.run_async(stop_event=stop_event))
        await asyncio.sleep(0.05)
        stop_event.set()
        await asyncio.wait_for(task, timeout=0.75)

    asyncio.run(_exercise())

    assert cancelled == {"runtime": True, "bootstrap": True}


def test_market_data_run_async_preserves_worker_failure_propagation() -> None:
    controller = object.__new__(MarketDataController)
    controller.enabled = True
    controller.logger = DummyLogger()
    controller.monitor = DummyMonitor()
    controller.refresh_universe_enabled = False
    controller.fast_frames = []
    controller.slow_frames = []

    async def _sync_server_time() -> None:
        return None

    async def _refresh_runtime_price_snapshot(*args, **kwargs):
        return {}

    async def _failing_runtime_snapshot_loop(stop_event=None) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("worker_failed")

    async def _idle_bootstrap_loop(stop_event=None) -> None:
        await asyncio.sleep(60.0)

    controller.sync_server_time = _sync_server_time
    controller._publish_connectivity = lambda: None
    controller.refresh_runtime_price_snapshot = _refresh_runtime_price_snapshot
    controller.runtime_snapshot_loop = _failing_runtime_snapshot_loop
    controller.active_history_bootstrap_loop = _idle_bootstrap_loop

    stop_event = threading.Event()

    async def _exercise() -> None:
        try:
            await controller.run_async(stop_event=stop_event)
        except RuntimeError as exc:
            assert str(exc) == "worker_failed"
            return
        raise AssertionError("worker failure was not propagated")

    asyncio.run(_exercise())


class BlockingMarketDataFlushMonitor(DummyMonitor):
    def __init__(self) -> None:
        super().__init__()
        self.flush_entered = threading.Event()

    def heartbeat(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        self.flush_entered.set()
        time.sleep(5.0)


def test_market_data_run_forever_skips_blocking_flush_on_coordinated_shutdown() -> None:
    controller = object.__new__(MarketDataController)
    controller.logger = DummyLogger()
    controller.monitor = BlockingMarketDataFlushMonitor()

    async def _wait_for_shutdown(stop_event=None) -> None:
        while not (stop_event and stop_event.is_set()):
            await asyncio.sleep(0.01)

    controller.run_async = _wait_for_shutdown
    stop_event = threading.Event()
    thread = threading.Thread(target=controller.run_forever, kwargs={"stop_event": stop_event})
    thread.start()
    time.sleep(0.05)

    stop_event.set()
    thread.join(timeout=0.75)

    assert thread.is_alive() is False
    assert controller.monitor.flush_entered.is_set() is False


class StopAwareSignalMonitor:
    def __init__(self) -> None:
        self.control_updates = 0
        self.decision_traces = 0

    def update_control_plane(self, payload) -> None:
        self.control_updates += 1

    def append_decision_trace(self, payload) -> None:
        self.decision_traces += 1


def test_signal_generator_aborts_current_batch_after_stop_requested(monkeypatch) -> None:
    controller = object.__new__(SignalGeneratorController)
    controller.logger = DummyLogger()
    controller.monitor = StopAwareSignalMonitor()
    controller.loop_sleep_sec = 5.0
    controller.messages = []
    controller._last_emitted = {}
    controller._emit_cooldown_sec = 45.0
    controller.paths = SimpleNamespace(
        config=SimpleNamespace(frames=SimpleNamespace(anchor_interval="15m"))
    )

    decision_entered = threading.Event()
    compute_calls = []

    class Engine:
        default_mode = "Medium"

        def load_control_state(self):
            return {"runtime_gate": {"paused": False, "blocked": False, "status": "running"}}

        def load_open_trade_snapshot(self):
            return {}

        def load_configured_symbols(self):
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        def load_symbols(self):
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        def _previous_position_from_snapshot(self, symbol, snapshot):
            return 0

        def compute_decision(self, symbol, snapshot, paused=False, pause_reason=""):
            compute_calls.append(symbol)
            decision_entered.set()
            # Model the real failure mode: SIGINT arrives while a symbol decision
            # is already in progress.  The current decision may finish, but the
            # remaining batch must not continue.
            time.sleep(0.10)
            return SimpleNamespace(
                symbol=symbol, action="HOLD", score=0.0, reason="test", mode="Medium", interval="15m", components={}
            )

        def append_decision_journal(self, *args, **kwargs):
            raise AssertionError("shutdown requested during compute must prevent post-decision journal work")

    controller.engine = Engine()
    monkeypatch.setattr(
        signal_generator_module,
        "signal_readiness_snapshot",
        lambda *args, **kwargs: {"rows": [], "ready_count": 3, "blocked_count": 0, "reason_counts": {}},
    )

    stop_event = threading.Event()
    thread = threading.Thread(target=controller.run_forever, kwargs={"stop_event": stop_event})
    thread.start()
    assert decision_entered.wait(0.5)

    stop_event.set()
    thread.join(timeout=0.75)

    assert thread.is_alive() is False
    assert compute_calls == ["BTCUSDT"]
    assert controller.monitor.decision_traces == 0


def test_market_data_request_retry_stops_after_shutdown_requested(monkeypatch) -> None:
    controller = object.__new__(MarketDataController)
    controller.request_timeout_sec = 10
    controller.request_retries = 3
    controller._runtime_stop_event = threading.Event()
    calls = []

    def _failing_get(*args, **kwargs):
        calls.append((args, kwargs))
        controller._runtime_stop_event.set()
        raise RuntimeError("request_failed_during_shutdown")

    monkeypatch.setattr(market_data_module.requests, "get", _failing_get)

    result = asyncio.run(controller._request_json("https://example.invalid"))

    assert result is None
    assert len(calls) == 1
