# runners/signal_generator_controller.py
from __future__ import annotations

import threading
import time
from typing import Any

from core.log_manager import get_logger
from inference.readiness import signal_readiness_snapshot
from monitoring.publisher import MonitorPublisher
from services.signal_engine import SignalEngine


class SignalGeneratorController:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.logger = get_logger("runners.signal_generator_controller", paths)
        self.monitor = MonitorPublisher(paths)
        self.engine = SignalEngine(paths)
        self.loop_sleep_sec = max(0.5, float(paths.config.runtime.market_poll_seconds))
        self.messages: list[str] = []
        self._last_emitted: dict[str, tuple[str, str, float]] = {}
        self._emit_cooldown_sec = float(getattr(paths.config.trading, "signal_cooldown_sec", 45.0) or 45.0)

    def _row_to_monitor_signal(self, row: dict[str, Any], ts_value: float) -> dict[str, Any]:
        symbol = str(row.get("symbol", ""))
        action = str(row.get("action", "HOLD"))
        reason = str(row.get("reason", ""))
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        return {
            "ts": ts_value,
            "symbol": symbol,
            "mode": str(row.get("mode", getattr(self.engine, "default_mode", "Medium"))),
            "action": action,
            "reason": reason,
            "score": round(score, 6),
            "interval": str(row.get("interval", self.paths.config.frames.anchor_interval)),
            "headline": f"{symbol} {action} {score:+.2f}",
        }

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.logger.info("SignalGeneratorController started")
        while not stop_event.is_set():
            try:
                control_state = self.engine.load_control_state()
                runtime_gate = dict(control_state.get("runtime_gate") or {}) if isinstance(control_state, dict) else {}
                paused = bool(runtime_gate.get("paused", control_state.get("paused", False) if isinstance(control_state, dict) else False))
                pause_reason = str(runtime_gate.get("pause_reason") or (control_state.get("pause_reason") if isinstance(control_state, dict) else "") or "")
                blocked = bool(runtime_gate.get("blocked", paused))
                block_reason = str(runtime_gate.get("reason") or ("control_plane_paused" if paused else "") or "")
                runtime_status = str(runtime_gate.get("status") or ("paused" if paused else "running"))
                self.monitor.update_control_plane(control_state)
                if stop_event.is_set():
                    return

                open_snapshot = self.engine.load_open_trade_snapshot()
                recent_signals: list[dict[str, Any]] = []
                recent_decisions: list[dict[str, Any]] = []
                latest_payload: dict[str, Any] | None = None
                loop_ts = time.time()

                configured_symbols = self.engine.load_configured_symbols() if hasattr(self.engine, "load_configured_symbols") else self.engine.load_symbols()
                eligible_symbols = self.engine.load_symbols()
                anchor_interval = str(getattr(self.paths.config.frames, "anchor_interval", "15m"))
                try:
                    readiness_payload = signal_readiness_snapshot(self.paths, symbols=configured_symbols, interval=anchor_interval)
                except Exception as readiness_exc:
                    # Readiness diagnostics are observability-only. They must never stop the
                    # signal loop or overwrite the runtime-gate behavior. Some tests and
                    # lightweight smoke runs provide a minimal paths object without the full
                    # artifact/registry directory layout; in that case publish a conservative
                    # diagnostic payload and continue evaluating eligible symbols normally.
                    reason = f"readiness_diagnostic_unavailable:{type(readiness_exc).__name__}"
                    readiness_payload = {
                        "interval": anchor_interval,
                        "total_symbols": len(configured_symbols),
                        "ready_count": 0,
                        "blocked_count": len(configured_symbols),
                        "ready_symbols": [],
                        "blocked_symbols": [str(sym).upper() for sym in configured_symbols],
                        "reason_counts": {reason: len(configured_symbols)} if configured_symbols else {},
                        "model_blockers": {},
                        "last_error": str(readiness_exc),
                        "rows": [
                            {
                                "symbol": str(sym).upper(),
                                "interval": anchor_interval,
                                "ready": False,
                                "reason": reason,
                                "models": {},
                            }
                            for sym in configured_symbols
                        ],
                    }

                for symbol in eligible_symbols:
                    if stop_event.is_set():
                        return
                    prev_position = self.engine._previous_position_from_snapshot(symbol, open_snapshot)
                    decision = self.engine.compute_decision(symbol, open_snapshot, paused=blocked, pause_reason=block_reason or pause_reason)
                    # A shutdown can arrive while a symbol decision is being computed.
                    # Do not continue journal/monitoring work for the rest of the batch;
                    # cooperative cancellation keeps the controller inside the launcher
                    # join contract without changing decision semantics during normal runs.
                    if stop_event.is_set():
                        return
                    journal_row = self.engine.append_decision_journal(
                        decision,
                        previous_position=prev_position,
                        paused=blocked,
                        pause_reason=block_reason or pause_reason,
                    )
                    self.monitor.append_decision_trace(journal_row)
                    decision_components = decision.components if isinstance(decision.components, dict) else {}
                    decision_state = decision_components.get("decision_state") if isinstance(decision_components.get("decision_state"), dict) else {}
                    execution_contract = dict(journal_row.get("execution_contract") or {}) if isinstance(journal_row.get("execution_contract"), dict) else {}
                    if not execution_contract:
                        execution_contract = {
                            "execution_blocker": str(journal_row.get("execution_blocker") or ""),
                            "rejection_reason": str(journal_row.get("rejection_reason") or ""),
                        }
                    row = {
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "score": decision.score,
                        "reason": decision.reason,
                        "mode": decision.mode,
                        "interval": decision.interval,
                        "ts": loop_ts,
                        "execution_eligible": bool(decision_state.get("execution_eligible", decision_components.get("execution_eligible", False))),
                        "decision_blocker": str(decision_state.get("blocker") or decision_components.get("decision_blocker") or ""),
                        "execution_blocker": str(execution_contract.get("execution_blocker") or ""),
                        "rejection_reason": str(execution_contract.get("rejection_reason") or ""),
                    }
                    recent_decisions.append(row)
                    signal_payload = self._row_to_monitor_signal(row, loop_ts)
                    recent_signals.append(signal_payload)
                    latest_payload = signal_payload
                    if blocked:
                        continue
                    if decision.action != "HOLD":
                        previous = self._last_emitted.get(decision.symbol)
                        should_emit = True
                        if previous is not None:
                            prev_action, prev_reason, prev_ts = previous
                            if (loop_ts - prev_ts) < self._emit_cooldown_sec:
                                should_emit = False
                            elif prev_action == decision.action and prev_reason == decision.reason and (loop_ts - prev_ts) < (self._emit_cooldown_sec * 2.0):
                                should_emit = False
                        if should_emit:
                            line = decision.pretty_line()
                            self.engine.append_signal_line(line)
                            self.messages.append(line)
                            self._last_emitted[decision.symbol] = (decision.action, decision.reason, loop_ts)

                if stop_event.is_set():
                    return

                evaluated_symbols = {str(row.get("symbol", "")).upper() for row in recent_signals}
                for row in readiness_payload.get("rows", []):
                    symbol = str(row.get("symbol", "")).upper()
                    if not symbol or symbol in evaluated_symbols or bool(row.get("ready")):
                        continue
                    diagnostic_signal = {
                        "ts": loop_ts,
                        "symbol": symbol,
                        "mode": str(getattr(self.engine, "default_mode", "Medium")),
                        "action": "HOLD",
                        "reason": str(row.get("reason") or "anchor_training_incomplete"),
                        "score": 0.0,
                        "interval": str(row.get("interval") or anchor_interval),
                        "headline": f"{symbol} HOLD anchor readiness blocked",
                        "readiness_only": True,
                    }
                    recent_signals.append(diagnostic_signal)
                    if latest_payload is None:
                        latest_payload = diagnostic_signal

                if stop_event.is_set():
                    return

                try:
                    adaptive_mode_state = self.engine.update_adaptive_mode_state(recent_decisions, open_snapshot=open_snapshot)
                except Exception as adaptive_exc:
                    adaptive_mode_state = {
                        "enabled": False,
                        "current_mode": str(getattr(self.engine, "default_mode", "Medium")),
                        "transition_reason": f"adaptive_mode_update_failed:{type(adaptive_exc).__name__}",
                        "last_error": str(adaptive_exc),
                    }

                if stop_event.is_set():
                    return

                kwargs: dict[str, Any] = {
                    "recent_signals": recent_signals[:100],
                    "recent_readiness": readiness_payload,
                    "runtime_gate": runtime_gate,
                    "adaptive_mode": adaptive_mode_state,
                    "last_error": None,
                    "source_priority": 100,
                    "source_name": "signal_generator_controller",
                }
                if latest_payload is not None:
                    kwargs.update(
                        latest_mode=latest_payload.get("mode"),
                        latest_symbol=latest_payload.get("symbol"),
                        latest_action=latest_payload.get("action"),
                        latest_reason=latest_payload.get("reason"),
                        latest_ts=latest_payload.get("ts"),
                    )
                kwargs["recent_decisions"] = recent_decisions[:50]
                self.monitor.update_signal_runtime(**kwargs)
                details = {
                    "symbols": len(configured_symbols),
                    "eligible_symbols": len(eligible_symbols),
                    "evaluated_symbols": len(recent_decisions),
                    "paused": paused,
                    "blocked": blocked,
                    "runtime_gate": runtime_gate,
                    "anchor_ready_symbols": int(readiness_payload.get("ready_count", 0)),
                    "anchor_blocked_symbols": int(readiness_payload.get("blocked_count", 0)),
                    "anchor_readiness_reason_counts": dict(readiness_payload.get("reason_counts", {}) or {}),
                    "adaptive_mode": dict(adaptive_mode_state or {}),
                    "effective_mode": str((adaptive_mode_state or {}).get("current_mode") or getattr(self.engine, "default_mode", "Medium")),
                }
                if pause_reason:
                    details["pause_reason"] = pause_reason
                if block_reason:
                    details["block_reason"] = block_reason
                self.monitor.heartbeat("signal_generator", status=runtime_status if blocked else "running", details=details)
            except Exception as exc:
                self.logger.exception("Signal generator loop failed")
                self.monitor.update_signal_runtime(
                    recent_signals=[],
                    recent_decisions=[],
                    runtime_gate={},
                    recent_readiness={},
                    last_error=str(exc),
                    source_priority=100,
                    source_name="signal_generator_controller",
                )
                self.monitor.controller_error("signal_generator", str(exc))
            stop_event.wait(self.loop_sleep_sec)
