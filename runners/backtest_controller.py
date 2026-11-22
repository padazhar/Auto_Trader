from __future__ import annotations

import threading
from dataclasses import asdict

from core.log_manager import get_logger
from monitoring.publisher import MonitorPublisher
from services.backtest_engine import BacktestEngine, ClosedTrade, OpenTrade


class BacktestController:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.logger = get_logger("runners.backtest_controller", paths)
        self.monitor = MonitorPublisher(paths)
        self.engine = BacktestEngine(paths)
        self.open_trades: dict[str, OpenTrade] = self.engine.restore_open_trades()
        self.completed: list[ClosedTrade] = self.engine.restore_completed_trades()
        self.engine.export_completed_trades_csv(self.completed)
        self.last_error: str | None = None
        self.messages: list[str] = []
        self.refresh_sec = max(0.2, float(paths.config.runtime.dashboard_refresh_ms) / 1000.0)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.logger.info("BacktestController started")
        self.engine.persist_open_trades(self.open_trades)
        while not stop_event.is_set():
            try:
                for signal in self.engine.drain_signal_lines():
                    closed = self.engine.on_signal(signal, self.open_trades, self.messages)
                    if closed:
                        self.engine.append_completed_trades(closed)
                closed_risk = self.engine.risk_checks(self.open_trades, self.messages)
                if closed_risk:
                    self.engine.append_completed_trades(closed_risk)
                self.completed = self.engine.restore_completed_trades()
                self.engine.export_completed_trades_csv(self.completed)
                self.engine.persist_open_trades(self.open_trades)
                summary = self.engine.summarize(self.open_trades, self.completed)
                open_rows = [self.engine.estimate_open_trade_row(trade) for trade in self.open_trades.values()]
                closed_rows = [self.engine.trade_history_row_from_closed(trade) for trade in self.completed[:100]]
                summary_dict = asdict(summary)
                execution_messages = list(self.messages[-30:])[::-1]
                self.monitor.update_backtest_runtime(
                    execution_surface="historical_backtest",
                    authoritative_for_runtime_positions=False,
                    open_positions=open_rows,
                    recent_closed=closed_rows[:25],
                    trade_history=closed_rows,
                    recent_messages=execution_messages,
                    summary=summary_dict,
                )
                self.monitor.heartbeat("backtest", status="running", details=summary_dict)
            except Exception as exc:
                self.last_error = str(exc)
                self.logger.exception("Backtest loop failed")
                self.monitor.controller_error("backtest", str(exc))
            stop_event.wait(self.refresh_sec)
