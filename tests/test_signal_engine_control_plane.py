from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from services.signal_engine import EngineDecision, SignalEngine


class DummyLogic:
    def decide(self, symbol: str, interval: str, previous_position: int = 0):
        raise AssertionError("logic.decide should not be called while paused")


def _paths(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "tradable_symbols.yaml").write_text("symbols: [BTCUSDT]\n", encoding="utf-8")
    runtime = root / "data" / "runtime"
    signal_logs = root / "data" / "logs" / "signal_logs"
    runtime.mkdir(parents=True, exist_ok=True)
    signal_logs.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=root,
        config=SimpleNamespace(inference=SimpleNamespace(active_mode="Medium"), frames=SimpleNamespace(anchor_interval="15m")),
        tradable_symbols_path=root / "config" / "tradable_symbols.yaml",
        backtest_open_trades_path=runtime / "backtest_open_trades.json",
        control_plane_state_path=runtime / "control_plane_state.json",
        decision_journal_path=runtime / "decision_journal.jsonl",
        signal_events_log_path=signal_logs / "signal_events.log",
    )


def test_signal_engine_respects_pause_and_writes_journal(tmp_path):
    paths = _paths(tmp_path)
    engine = SignalEngine(paths)
    engine.logic = DummyLogic()
    state = engine.save_control_state(paused=True, pause_reason="maintenance", updated_by="test")
    decision = engine.compute_decision("BTCUSDT", {}, paused=state["paused"], pause_reason=state["pause_reason"])
    assert decision.action == "HOLD"
    assert decision.reason == "control_plane_paused"
    record = engine.append_decision_journal(decision, previous_position=0, paused=True, pause_reason="maintenance")
    assert record["paused"] is True
    assert record["effective_action"] == "HOLD"
    assert record["approved"] is False
    assert record["decision_stage"] == "signal_generation"
    assert record["journal_writer"] == "signal_engine"
    assert record["execution_surface"] == "signal_generation_only"
    assert record["components"]["execution_result"]["status"] == "not_attempted"
    assert engine.recent_decision_journal(limit=1)[0]["reason"] == "control_plane_paused"


def test_signal_engine_writes_signal_log(tmp_path):
    paths = _paths(tmp_path)
    engine = SignalEngine(paths)
    line = EngineDecision(symbol="BTCUSDT", interval="15m", action="ENTER_LONG", score=0.12, reason="test", mode="Medium", components={}).pretty_line()
    engine.append_signal_line(line)
    saved = paths.signal_events_log_path.read_text(encoding="utf-8")
    assert "BTCUSDT" in saved
    assert "ENTER LONG" in saved


def test_signal_engine_distinguishes_hold_blocker_from_signal_generation_only(tmp_path):
    paths = _paths(tmp_path)
    engine = SignalEngine(paths)
    decision = EngineDecision(
        symbol="BTCUSDT",
        interval="15m",
        action="HOLD",
        score=0.0,
        reason="weak_long_filter",
        mode="Medium",
        components={
            "decision_state": {
                "action": "HOLD",
                "reason": "weak_long_filter",
                "execution_eligible": False,
                "blocker": "weak_long_filter",
            },
            "execution_eligible": False,
            "decision_blocker": "weak_long_filter",
        },
    )

    record = engine.append_decision_journal(decision, previous_position=0)

    assert record["execution_status"] == "not_attempted"
    assert record["execution_attempted"] is False
    assert record["execution_blocker_stage"] == "decision_logic"
    assert record["execution_blocker"] == "weak_long_filter"
    assert record["rejection_reason"] == "weak_long_filter"
    assert record["signal_actionable"] is False
    assert record["components"]["execution_contract"]["signal_contract_version"] == "signal_attempt_contract_v1"


def test_signal_engine_marks_enter_as_signal_generation_only_not_order_attempt(tmp_path):
    paths = _paths(tmp_path)
    engine = SignalEngine(paths)
    decision = EngineDecision(
        symbol="BTCUSDT",
        interval="15m",
        action="ENTER_LONG",
        score=0.1,
        reason="ev_enter_long",
        mode="Medium",
        components={
            "decision_state": {
                "action": "ENTER_LONG",
                "reason": "ev_enter_long",
                "execution_eligible": True,
                "blocker": "",
            },
            "execution_eligible": True,
            "decision_blocker": "",
        },
    )

    record = engine.append_decision_journal(decision, previous_position=0)

    assert record["execution_status"] == "not_attempted"
    assert record["execution_attempted"] is False
    assert record["execution_blocker_stage"] == "signal_generation"
    assert record["execution_blocker"] == "signal_generation_only"
    assert record["rejection_reason"] == "signal_generation_only"
    assert record["signal_actionable"] is True
