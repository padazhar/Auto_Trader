# tests/test_signal_generator_runtime_gate.py
from __future__ import annotations

import threading
from types import SimpleNamespace

from runners.signal_generator_controller import SignalGeneratorController


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.control_updates: list[dict] = []
        self.decision_traces: list[dict] = []
        self.signal_updates: list[dict] = []
        self.inference_updates: list[dict] = []
        self.heartbeats: list[dict] = []
        self.errors: list[tuple[str, str]] = []

    def update_control_plane(self, payload) -> None:
        self.control_updates.append(dict(payload))

    def append_decision_trace(self, payload) -> None:
        self.decision_traces.append(dict(payload))

    def update_signal_runtime(self, **kwargs) -> None:
        self.signal_updates.append(dict(kwargs))

    def update_inference(self, **kwargs) -> None:
        self.inference_updates.append(dict(kwargs))

    def heartbeat(self, name: str, *, status: str, details: dict) -> None:
        self.heartbeats.append({"name": name, "status": status, "details": dict(details)})

    def controller_error(self, name: str, message: str) -> None:
        self.errors.append((name, message))


class _DummyDecision(SimpleNamespace):
    def pretty_line(self) -> str:
        return "ignored"


class _DummyEngine:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.default_mode = "Medium"
        self.compute_calls: list[dict] = []
        self.signal_lines: list[str] = []

    def load_control_state(self) -> dict:
        return {
            "paused": False,
            "pause_reason": "",
            "runtime_gate": {
                "blocked": True,
                "reason": "circuit_breaker_triggered",
                "status": "blocked",
                "paused": False,
                "pause_reason": "",
                "trading_enabled": False,
            },
        }

    def load_open_trade_snapshot(self) -> dict:
        return {}

    def load_symbols(self) -> list[str]:
        return ["BTCUSDT"]

    def _previous_position_from_snapshot(self, symbol: str, snapshot: dict) -> int:
        return 0

    def compute_decision(self, symbol: str, open_snapshot: dict, *, paused: bool = False, pause_reason: str = ""):
        self.compute_calls.append({"symbol": symbol, "paused": paused, "pause_reason": pause_reason})
        return _DummyDecision(symbol=symbol, interval="15m", action="HOLD", score=0.0, reason="control_plane_paused", mode="Medium", components={})

    def append_decision_journal(self, decision, previous_position: int, *, paused: bool = False, pause_reason: str = "") -> dict:
        return {
            "symbol": decision.symbol,
            "interval": decision.interval,
            "action": decision.action,
            "reason": decision.reason,
            "ts": 1.0,
            "paused": paused,
            "pause_reason": pause_reason,
        }

    def append_signal_line(self, line: str) -> None:
        self.signal_lines.append(line)


def test_signal_generator_respects_runtime_gate(monkeypatch) -> None:
    monkeypatch.setattr("runners.signal_generator_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.signal_generator_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.signal_generator_controller.SignalEngine", _DummyEngine)

    paths = SimpleNamespace(config=SimpleNamespace(runtime=SimpleNamespace(market_poll_seconds=0.01), trading=SimpleNamespace(signal_cooldown_sec=45.0), frames=SimpleNamespace(anchor_interval="15m")))
    controller = SignalGeneratorController(paths)

    stop_event = threading.Event()
    def _stop_once(*args, **kwargs):
        stop_event.set()
    stop_event.wait = _stop_once  # type: ignore[assignment]

    controller.run_forever(stop_event)

    assert controller.engine.compute_calls == [{"symbol": "BTCUSDT", "paused": True, "pause_reason": "circuit_breaker_triggered"}]
    assert controller.engine.signal_lines == []
    assert controller.monitor.heartbeats[0]["status"] == "blocked"
    assert controller.monitor.inference_updates == []
    assert controller.monitor.signal_updates[0]["runtime_gate"]["reason"] == "circuit_breaker_triggered"


def test_signal_generator_failure_does_not_overwrite_inference_surface(monkeypatch) -> None:
    class _ExplodingEngine(_DummyEngine):
        def load_symbols(self) -> list[str]:
            raise RuntimeError("boom")

    monkeypatch.setattr("runners.signal_generator_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.signal_generator_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.signal_generator_controller.SignalEngine", _ExplodingEngine)

    paths = SimpleNamespace(config=SimpleNamespace(runtime=SimpleNamespace(market_poll_seconds=0.01), trading=SimpleNamespace(signal_cooldown_sec=45.0), frames=SimpleNamespace(anchor_interval="15m")))
    controller = SignalGeneratorController(paths)

    stop_event = threading.Event()
    def _stop_once(*args, **kwargs):
        stop_event.set()
    stop_event.wait = _stop_once  # type: ignore[assignment]

    controller.run_forever(stop_event)

    assert controller.monitor.inference_updates == []
    assert controller.monitor.signal_updates[-1]["last_error"] == "boom"
    assert controller.monitor.errors == [("signal_generator", "boom")]


def test_signal_generator_publishes_anchor_readiness_without_emitting_signals(monkeypatch) -> None:
    class _ReadinessEngine(_DummyEngine):
        def load_configured_symbols(self) -> list[str]:
            return ["BTCUSDT", "ETHUSDT"]

        def load_symbols(self) -> list[str]:
            return ["BTCUSDT"]

    monkeypatch.setattr("runners.signal_generator_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.signal_generator_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.signal_generator_controller.SignalEngine", _ReadinessEngine)
    monkeypatch.setattr(
        "runners.signal_generator_controller.signal_readiness_snapshot",
        lambda paths, symbols, interval: {
            "interval": interval,
            "total_symbols": 2,
            "ready_count": 1,
            "blocked_count": 1,
            "ready_symbols": ["BTCUSDT"],
            "blocked_symbols": ["ETHUSDT"],
            "reason_counts": {"anchor_training_incomplete:rf:registry_status:idle": 1},
            "model_blockers": {"rf": 1, "gbm": 0},
            "rows": [
                {"symbol": "BTCUSDT", "interval": interval, "ready": True, "reason": "anchor_ready", "models": {}},
                {"symbol": "ETHUSDT", "interval": interval, "ready": False, "reason": "anchor_training_incomplete:rf:registry_status:idle", "models": {}},
            ],
        },
    )

    paths = SimpleNamespace(config=SimpleNamespace(runtime=SimpleNamespace(market_poll_seconds=0.01), trading=SimpleNamespace(signal_cooldown_sec=45.0), frames=SimpleNamespace(anchor_interval="15m")))
    controller = SignalGeneratorController(paths)

    stop_event = threading.Event()
    def _stop_once(*args, **kwargs):
        stop_event.set()
    stop_event.wait = _stop_once  # type: ignore[assignment]

    controller.run_forever(stop_event)

    update = controller.monitor.signal_updates[0]
    assert update["recent_readiness"]["blocked_count"] == 1
    assert any(row["symbol"] == "ETHUSDT" and row["readiness_only"] for row in update["recent_signals"])
    assert controller.engine.signal_lines == []
    assert controller.monitor.heartbeats[0]["details"]["anchor_blocked_symbols"] == 1
