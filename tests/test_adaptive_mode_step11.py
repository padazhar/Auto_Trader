from __future__ import annotations

import json
from types import SimpleNamespace

from services.signal_engine import AdaptiveModeController


class DummyPaths:
    def __init__(self, tmp_path):
        self.runtime_dir = tmp_path
        self.portfolio_state_path = tmp_path / "portfolio_state.json"
        self.config = SimpleNamespace(
            inference=SimpleNamespace(active_mode="Medium"),
            adaptive_mode=SimpleNamespace(
                enabled=True,
                hold_minutes=0.01,
                cooldown_minutes=1.0,
                medium_to_light_when_no_trades=True,
                heavy_on_loss_streak=3,
                heavy_win_rate_lookback=4,
                heavy_on_win_rate_below=0.25,
            ),
        )


def test_adaptive_mode_switches_medium_to_light_after_prolonged_hold(tmp_path):
    paths = DummyPaths(tmp_path)
    paths.portfolio_state_path.write_text(json.dumps({"closed_trades": []}), encoding="utf-8")
    controller = AdaptiveModeController(paths)
    state = controller.load_state()
    state["hold_started_ts"] = controller._now() - 120.0
    state["current_mode"] = "Medium"
    state["cooldown_until_ts"] = 0.0
    controller.save_state(state)

    updated = controller.update_after_decisions(
        [{"symbol": "BTCUSDT", "action": "HOLD"}],
        open_snapshot={},
    )

    assert updated["current_mode"] == "Light"
    assert updated["transition_reason"] == "prolonged_hold_no_trades"


def test_adaptive_mode_switches_to_heavy_on_loss_streak(tmp_path):
    paths = DummyPaths(tmp_path)
    paths.portfolio_state_path.write_text(
        json.dumps({"closed_trades": [{"pnl_usd": -1.0}, {"pnl_usd": -2.0}, {"pnl_usd": -3.0}]}),
        encoding="utf-8",
    )
    controller = AdaptiveModeController(paths)

    updated = controller.update_after_decisions(
        [{"symbol": "BTCUSDT", "action": "ENTER_LONG"}],
        open_snapshot={},
    )

    assert updated["current_mode"] == "Heavy"
    assert updated["transition_reason"] == "loss_quality_guard"


def test_signal_engine_keeps_adaptive_mode_inert_for_step11_scope(tmp_path):
    from services.signal_engine import SignalEngine

    paths = DummyPaths(tmp_path)
    paths.portfolio_state_path.write_text(json.dumps({"closed_trades": []}), encoding="utf-8")
    engine = SignalEngine.__new__(SignalEngine)
    engine.default_mode = "Medium"
    engine.adaptive_mode = AdaptiveModeController(paths)

    state = engine.update_adaptive_mode_state(
        [{"symbol": "BTCUSDT", "action": "HOLD"}],
        open_snapshot={},
    )

    assert state["enabled"] is False
    assert state["current_mode"] == "Medium"
    assert state["transition_reason"] == "disabled_step11_scope_guard"
    assert engine.load_adaptive_mode_state()["current_mode"] == "Medium"
