from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.portfolio_allocator import allocate_position_plan
from execution.portfolio_state import PortfolioStateStore
from execution.position_sizing import compute_position_plan
from execution.risk_engine import RiskEngine


class DummyDecision:
    def __init__(self, symbol: str, action: str, score: float = 0.9, price: float = 100.0, atr_fraction: float = 0.01):
        self.symbol = symbol
        self.action = action
        self.score = score
        self.components = {"price": price, "atr_fraction": atr_fraction}


def _paths(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    risk_cfg = SimpleNamespace(
        account_balance_usd=1000.0,
        risk_per_trade=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=5000.0,
        min_notional_usd=25.0,
        max_total_exposure_usd=750.0,
        max_concurrent_positions=4,
        max_symbol_notional_usd=500.0,
        max_same_direction_exposure_usd=450.0,
        max_risk_distortion_fraction=0.35,
        daily_loss_limit_usd=50.0,
        max_drawdown_fraction=0.10,
        max_correlated_positions=2,
        correlation_groups={"BTCUSDT": "majors", "ETHUSDT": "majors"},
    )
    return SimpleNamespace(
        config=SimpleNamespace(risk=risk_cfg),
        portfolio_state_path=runtime / "portfolio_state.json",
        circuit_breaker_state_path=runtime / "circuit_breaker_state.json",
    )


def test_allocator_scales_plan_by_confidence_and_headroom():
    cfg = SimpleNamespace(
        max_total_exposure_usd=400.0,
        max_same_direction_exposure_usd=300.0,
        max_symbol_notional_usd=250.0,
        min_notional_usd=25.0,
    )
    plan = compute_position_plan(
        balance=1000.0,
        risk_per_trade=0.01,
        price=100.0,
        atr_fraction=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=1000.0,
        min_notional_usd=25.0,
        direction="LONG",
    )
    decision = DummyDecision("SOLUSDT", "ENTER_LONG", score=0.2)
    allocation = allocate_position_plan(
        decision=decision,
        plan=plan,
        exposure={"gross_exposure_usd": 200.0, "long_exposure_usd": 150.0, "short_exposure_usd": 0.0, "correlated_counts": {}, "correlation_exposure_usd": {}},
        cfg=cfg,
        existing_notional_usd=0.0,
        existing_same_side_notional_usd=0.0,
        correlation_group="",
    )
    assert allocation.approved is True
    assert allocation.target_notional_usd <= 150.0
    assert allocation.allocation_fraction < 1.0


def test_risk_engine_rejects_allocation_when_final_plan_exceeds_distortion_limit(tmp_path):
    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = store.load()
    existing_plan = compute_position_plan(
        balance=1000.0,
        risk_per_trade=0.01,
        price=100.0,
        atr_fraction=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=500.0,
        min_notional_usd=25.0,
        direction="LONG",
    )
    store.apply_execution(state, symbol="BTCUSDT", action="ENTER_LONG", price=100.0, ts=1.0, position_plan={
        "quantity": existing_plan.quantity,
        "notional_usd": 350.0,
        "stop_loss": existing_plan.stop_loss,
        "take_profit": existing_plan.take_profit,
    })
    store.save(state)
    portfolio = engine.load_portfolio()
    decision = DummyDecision("SOLUSDT", "ENTER_LONG", score=0.95)
    result = engine.evaluate_trade(decision=decision, portfolio_state=portfolio, ts=2.0)
    assert result.approved is False
    assert result.final_action == "HOLD"
    assert result.reason == "risk_distortion_exceeded"
    assert result.position_plan is not None
    assert result.position_plan.notional_usd <= 400.0
    assert result.position_plan.risk_distortion_fraction > paths.config.risk.max_risk_distortion_fraction
    assert result.components["allocation"]["allocation_fraction"] < 1.0
    assert result.components["risk_breakdown"]["approved_notional_usd"] == result.position_plan.notional_usd
