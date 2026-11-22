from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from execution.circuit_breaker import CircuitBreakerStore
from execution.portfolio_state import PortfolioStateStore
from execution.position_sizing import compute_position_plan
from execution.risk_engine import RiskEngine


class DummyDecision:
    def __init__(self, symbol: str, action: str, price: float = 100.0, atr_fraction: float = 0.01):
        self.symbol = symbol
        self.action = action
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
        max_total_exposure_usd=7500.0,
        max_concurrent_positions=2,
        max_symbol_notional_usd=1000.0,
        max_same_direction_exposure_usd=1500.0,
        max_risk_distortion_fraction=0.35,
        daily_loss_limit_usd=50.0,
        max_drawdown_fraction=0.10,
        max_correlated_positions=1,
        correlation_groups={"BTCUSDT": "majors", "ETHUSDT": "majors"},
    )
    return SimpleNamespace(
        config=SimpleNamespace(risk=risk_cfg),
        portfolio_state_path=runtime / "portfolio_state.json",
        circuit_breaker_state_path=runtime / "circuit_breaker_state.json",
    )


def test_position_plan_surfaces_risk_distortion_when_clamped_up():
    plan = compute_position_plan(
        balance=1000.0,
        risk_per_trade=0.0002,
        price=100.0,
        atr_fraction=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=5000.0,
        min_notional_usd=25.0,
        direction="LONG",
    )
    assert plan.clamp_applied is True
    assert plan.clamp_reason == "min_notional"
    assert plan.risk_distortion_fraction > 0.35


def test_portfolio_store_persists_and_realizes_pnl(tmp_path):
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()
    plan = compute_position_plan(
        balance=1000.0,
        risk_per_trade=0.01,
        price=100.0,
        atr_fraction=0.01,
        sl_atr_mult=1.5,
        tp_atr_mult=2.0,
        max_notional_usd=5000.0,
        min_notional_usd=25.0,
        direction="LONG",
    )
    store.apply_execution(state, symbol="BTCUSDT", action="ENTER_LONG", price=100.0, ts=1.0, position_plan=asdict(plan))
    summary = store.mark_to_market(state, {"BTCUSDT": 102.0}, ts=2.0)
    assert summary["unrealized_pnl_usd"] > 0
    store.apply_execution(state, symbol="BTCUSDT", action="EXIT_LONG", price=102.0, ts=3.0)
    store.save(state)
    reloaded = store.load()
    assert reloaded.realized_pnl_usd > 0
    assert "BTCUSDT" not in reloaded.positions


def test_risk_engine_rejects_post_allocation_risk_distortion(tmp_path):
    paths = _paths(tmp_path)
    paths.config.risk.max_same_direction_exposure_usd = 400.0
    paths.config.risk.max_correlated_positions = 10
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    decision = engine.evaluate_trade(
        decision=DummyDecision("SOLUSDT", "ENTER_LONG"),
        portfolio_state=portfolio,
        ts=1.0,
    )

    assert decision.approved is False
    assert decision.final_action == "HOLD"
    assert decision.reason == "risk_distortion_exceeded"
    assert decision.components["allocation"]["approved"] is True
    assert decision.position_plan is not None
    assert decision.position_plan.notional_usd == 400.0
    assert decision.position_plan.risk_distortion_fraction > paths.config.risk.max_risk_distortion_fraction
    assert decision.components["risk_breakdown"]["clamp_reason"] == "max_notional"


def test_risk_engine_approves_post_allocation_plan_within_distortion_limit(tmp_path):
    paths = _paths(tmp_path)
    paths.config.risk.max_same_direction_exposure_usd = 500.0
    paths.config.risk.max_correlated_positions = 10
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    decision = engine.evaluate_trade(
        decision=DummyDecision("SOLUSDT", "ENTER_LONG"),
        portfolio_state=portfolio,
        ts=1.0,
    )

    assert decision.approved is True
    assert decision.reason == "risk_approved"
    assert decision.position_plan is not None
    assert decision.position_plan.notional_usd == 500.0
    assert decision.position_plan.risk_distortion_fraction <= paths.config.risk.max_risk_distortion_fraction


def test_risk_engine_blocks_on_concentration_and_circuit_breaker(tmp_path):
    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    btc = engine.evaluate_trade(decision=DummyDecision("BTCUSDT", "ENTER_LONG"), portfolio_state=portfolio, ts=1.0)
    assert btc.approved is True
    engine.portfolio_store.apply_execution(portfolio, symbol="BTCUSDT", action=btc.final_action, price=100.0, ts=1.0, position_plan=asdict(btc.position_plan))

    eth = engine.evaluate_trade(decision=DummyDecision("ETHUSDT", "ENTER_LONG"), portfolio_state=portfolio, ts=2.0)
    assert eth.approved is False
    assert eth.reason == "max_correlated_positions"

    pnl_summary = engine.portfolio_store.mark_to_market(portfolio, {"BTCUSDT": 85.0}, ts=3.0)
    circuit = engine.refresh_circuit_breaker(portfolio, pnl_summary=pnl_summary, ts=3.0)
    assert circuit.triggered is True

    sol = engine.evaluate_trade(decision=DummyDecision("SOLUSDT", "ENTER_SHORT"), portfolio_state=portfolio, ts=4.0)
    assert sol.approved is False
    assert sol.reason.startswith("circuit_breaker_")

    persisted = CircuitBreakerStore(paths.circuit_breaker_state_path).load()
    assert persisted.triggered is True


def test_risk_engine_quarantines_symbol_with_bad_recent_expectancy(tmp_path):
    from execution.portfolio_state import ClosedTradeState

    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    for index in range(5):
        portfolio.closed_trades.append(
            ClosedTradeState(
                symbol="HYPEUSDT",
                side="LONG",
                quantity=1.0,
                notional_usd=100.0,
                entry_price=100.0,
                exit_price=90.0,
                opened_at=float(index),
                closed_at=10.0 + float(index),
                pnl_usd=-12.0,
                exit_reason="SL_HIT",
            )
        )

    decision = engine.evaluate_trade(
        decision=DummyDecision("HYPEUSDT", "ENTER_LONG"),
        portfolio_state=portfolio,
        ts=20.0,
    )

    assert decision.approved is False
    assert decision.reason == "symbol_expectancy_quarantine"
    guard = decision.components["symbol_expectancy_guard"]
    assert guard["quarantined"] is True
    assert guard["trades"] == 5
    assert guard["realized_pnl_usd"] == -60.0


def test_risk_engine_allows_symbol_without_minimum_expectancy_history(tmp_path):
    from execution.portfolio_state import ClosedTradeState

    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    portfolio.closed_trades.append(
        ClosedTradeState(
            symbol="NEARUSDT",
            side="SHORT",
            quantity=1.0,
            notional_usd=100.0,
            entry_price=100.0,
            exit_price=101.0,
            opened_at=1.0,
            closed_at=2.0,
            pnl_usd=-1.0,
            exit_reason="SL_HIT",
        )
    )

    decision = engine.evaluate_trade(
        decision=DummyDecision("NEARUSDT", "ENTER_SHORT"),
        portfolio_state=portfolio,
        ts=3.0,
    )

    assert decision.approved is True
    assert decision.components["symbol_expectancy_guard"]["reason"] == "insufficient_history"


def test_adaptive_stop_multiplier_uses_mae_mfe_when_history_is_sufficient():
    from types import SimpleNamespace
    from execution.position_sizing import AdaptiveStopConfig, compute_adaptive_stop_multipliers

    rows = [
        {
            "symbol": "SOLUSDT",
            "side": "LONG",
            "attribution": {
                "max_adverse_excursion_bps": 180.0,
                "max_favorable_excursion_bps": 260.0,
            },
        }
        for _ in range(8)
    ]
    cfg = AdaptiveStopConfig.from_config(
        SimpleNamespace(
            adaptive_stops={
                "enabled": True,
                "lookback_trades": 20,
                "min_trades": 8,
                "min_sl_atr_mult": 0.8,
                "max_sl_atr_mult": 2.2,
                "min_tp_atr_mult": 1.0,
                "max_tp_atr_mult": 3.2,
                "mae_safety_mult": 1.15,
                "mfe_capture_mult": 0.85,
                "min_reward_risk_ratio": 1.10,
            }
        )
    )

    adjustment = compute_adaptive_stop_multipliers(
        rows,
        symbol="SOLUSDT",
        side="LONG",
        atr_fraction=0.01,
        base_sl_atr_mult=1.0,
        base_tp_atr_mult=1.5,
        cfg=cfg,
    )

    assert adjustment.enabled is True
    assert adjustment.applied is True
    assert adjustment.trades_used == 8
    assert adjustment.sl_atr_mult > 1.0
    assert adjustment.tp_atr_mult > adjustment.sl_atr_mult


def test_risk_engine_carries_adaptive_stop_metadata_when_enabled(tmp_path):
    from execution.portfolio_state import ClosedTradeState

    paths = _paths(tmp_path)
    paths.config.risk.adaptive_stops = {
        "enabled": True,
        "lookback_trades": 20,
        "min_trades": 8,
        "min_sl_atr_mult": 0.8,
        "max_sl_atr_mult": 2.2,
        "min_tp_atr_mult": 1.0,
        "max_tp_atr_mult": 3.2,
        "mae_safety_mult": 1.15,
        "mfe_capture_mult": 0.85,
        "min_reward_risk_ratio": 1.10,
    }
    paths.config.risk.max_concurrent_positions = 10
    paths.config.risk.max_correlated_positions = 10
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    for index in range(8):
        portfolio.closed_trades.append(
            ClosedTradeState(
                symbol="SOLUSDT",
                side="LONG",
                quantity=1.0,
                notional_usd=100.0,
                entry_price=100.0,
                exit_price=101.0,
                opened_at=float(index),
                closed_at=10.0 + float(index),
                pnl_usd=2.0,
                exit_reason="TP_HIT",
                attribution={
                    "max_adverse_excursion_bps": 180.0,
                    "max_favorable_excursion_bps": 260.0,
                },
            )
        )

    decision = engine.evaluate_trade(
        decision=DummyDecision("SOLUSDT", "ENTER_LONG", price=100.0, atr_fraction=0.01),
        portfolio_state=portfolio,
        ts=30.0,
    )

    assert decision.approved is True
    assert decision.components["adaptive_stops"]["applied"] is True
    assert decision.position_plan.sl_atr_mult == decision.components["adaptive_stops"]["sl_atr_mult"]
    assert decision.position_plan.tp_atr_mult == decision.components["adaptive_stops"]["tp_atr_mult"]



def test_risk_engine_rejects_same_side_direct_entry_when_position_is_already_open(tmp_path):
    paths = _paths(tmp_path)
    paths.config.risk.max_correlated_positions = 10
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    first = engine.evaluate_trade(
        decision=DummyDecision("ZECUSDT", "ENTER_SHORT"),
        portfolio_state=portfolio,
        ts=1.0,
    )
    assert first.approved is True
    engine.portfolio_store.apply_execution(
        portfolio,
        symbol="ZECUSDT",
        action=first.final_action,
        price=100.0,
        ts=1.0,
        position_plan=asdict(first.position_plan),
    )

    repeated = engine.evaluate_trade(
        decision=DummyDecision("ZECUSDT", "ENTER_SHORT"),
        portfolio_state=portfolio,
        ts=2.0,
    )

    assert repeated.approved is False
    assert repeated.final_action == "HOLD"
    assert repeated.reason == "same_side_position_already_open"
    assert repeated.components["position_invariant"]["direct_entry_allowed_with_open_position"] is False


def test_risk_engine_requires_flip_for_direct_entry_opposite_existing_position(tmp_path):
    paths = _paths(tmp_path)
    paths.config.risk.max_correlated_positions = 10
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()

    first = engine.evaluate_trade(
        decision=DummyDecision("ZECUSDT", "ENTER_SHORT"),
        portfolio_state=portfolio,
        ts=1.0,
    )
    assert first.approved is True
    engine.portfolio_store.apply_execution(
        portfolio,
        symbol="ZECUSDT",
        action=first.final_action,
        price=100.0,
        ts=1.0,
        position_plan=asdict(first.position_plan),
    )

    opposite_direct_entry = engine.evaluate_trade(
        decision=DummyDecision("ZECUSDT", "ENTER_LONG"),
        portfolio_state=portfolio,
        ts=2.0,
    )

    assert opposite_direct_entry.approved is False
    assert opposite_direct_entry.final_action == "HOLD"
    assert opposite_direct_entry.reason == "position_open_requires_flip"
