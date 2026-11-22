from __future__ import annotations

from types import SimpleNamespace

from execution.portfolio_state import PortfolioStateStore
from execution.risk_engine import RiskEngine
from runners.live_trading_controller import LiveTradingController


class DummyDecision:
    def __init__(self, symbol: str = "BTCUSDT", action: str = "ENTER_LONG"):
        self.symbol = symbol
        self.action = action
        self.reason = "ev_enter_long"
        self.interval = "15m"
        self.score = 0.1
        self.components = {"price": 100.0, "atr_fraction": 0.01}


def _paths(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    risk_cfg = SimpleNamespace(
        account_balance_usd=1000.0,
        risk_per_trade=0.0002,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=5000.0,
        min_notional_usd=25.0,
        max_total_exposure_usd=7500.0,
        max_concurrent_positions=2,
        max_symbol_notional_usd=1000.0,
        max_same_direction_exposure_usd=1500.0,
        max_risk_distortion_fraction=0.35,
        daily_loss_limit_usd=50.0,
        max_drawdown_fraction=0.10,
        max_correlated_positions=10,
        correlation_groups={},
    )
    return SimpleNamespace(
        config=SimpleNamespace(risk=risk_cfg, runtime=SimpleNamespace(execution_mode="paper"), exchange=SimpleNamespace()),
        portfolio_state_path=runtime / "portfolio_state.json",
        circuit_breaker_state_path=runtime / "circuit_breaker_state.json",
    )


def test_risk_distortion_rejection_contains_operator_breakdown(tmp_path):
    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    decision = engine.evaluate_trade(decision=DummyDecision(), portfolio_state=portfolio, ts=1.0)

    assert decision.approved is False
    assert decision.reason == "risk_distortion_exceeded"
    breakdown = decision.components["risk_breakdown"]
    assert breakdown["entry_price"] == 100.0
    assert breakdown["stop_loss"] < 100.0
    assert breakdown["take_profit"] > 100.0
    assert breakdown["risk_distortion_fraction"] > breakdown["max_risk_distortion_fraction"]
    assert breakdown["risk_limit"] == breakdown["max_risk_distortion_fraction"]
    assert breakdown["requested_notional_usd"] == breakdown["intended_notional_usd"]
    assert breakdown["approved_notional_usd"] == breakdown["notional_usd"]
    assert breakdown["requested_notional_usd"] != breakdown["approved_notional_usd"]
    assert breakdown["clamp_reason"] == "min_notional"


def test_post_allocation_distortion_rejection_contains_allocation_and_breakdown(tmp_path):
    paths = _paths(tmp_path)
    paths.config.risk.risk_per_trade = 0.01
    paths.config.risk.max_same_direction_exposure_usd = 400.0
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    decision = engine.evaluate_trade(decision=DummyDecision(symbol="SOLUSDT"), portfolio_state=portfolio, ts=1.0)

    assert decision.approved is False
    assert decision.reason == "risk_distortion_exceeded"
    assert decision.components["allocation"]["approved"] is True
    breakdown = decision.components["risk_breakdown"]
    assert breakdown["requested_notional_usd"] > breakdown["approved_notional_usd"]
    assert breakdown["approved_notional_usd"] == decision.components["allocation"]["target_notional_usd"]
    assert breakdown["risk_distortion_fraction"] > breakdown["max_risk_distortion_fraction"]
    assert breakdown["clamp_reason"] == "max_notional"


def test_execution_message_includes_risk_distortion_breakdown():
    row = {
        "symbol": "BTCUSDT",
        "effective_action": "HOLD",
        "execution_status": "rejected_risk",
        "rejection_reason": "risk_distortion_exceeded",
        "risk_breakdown": {
            "risk_distortion_fraction": 0.75,
            "max_risk_distortion_fraction": 0.35,
            "clamp_reason": "min_notional",
        },
    }

    messages = LiveTradingController._execution_messages([row])

    assert "risk_distortion_exceeded" in messages[0]
    assert "distortion=0.7500" in messages[0]
    assert "limit=0.3500" in messages[0]
    assert "clamp=min_notional" in messages[0]
