from __future__ import annotations

import threading

from core.log_manager import get_logger


class StateBridgeController:
    """
    Compatibility-only controller.

    After the integration refactor, BacktestController is the single owner of
    simulated trade state. This controller now only emits a heartbeat/log line
    and no longer mirrors or mutates backtest_open_trades.json.
    """

    def __init__(self, paths) -> None:
        self.paths = paths
        self.logger = get_logger("runners.state_bridge_controller", paths)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.logger.info("StateBridgeController running in compatibility-noop mode")
        while not stop_event.is_set():
            stop_event.wait(2.0)
