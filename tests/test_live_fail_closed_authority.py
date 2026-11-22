from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.risk_engine import RiskEngine


class DummyDecision:
    def __init__(self, symbol: str, action: str, price: float = 100.0, atr_fraction: float = 0.01):
        self.symbol = symbol
        self.action = action
        self.components = {"price": price, "atr_fraction": atr_fraction}


def _paths(tmp_path: Path):
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        config=SimpleNamespace(
            risk=SimpleNamespace(
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
                correlation_groups={},
            ),
            runtime=SimpleNamespace(execution_mode="live", allow_unreconciled_live_mode=True),
            exchange=SimpleNamespace(
                position_reconciliation_enabled=True,
                require_reconciliation_for_live=True,
                max_reconciliation_age_sec=30,
            ),
        ),
        portfolio_state_path=runtime / "portfolio_state.json",
        circuit_breaker_state_path=runtime / "circuit_breaker_state.json",
    )


def test_risk_engine_blocks_non_authoritative_live_override(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    portfolio.reconciliation_status = "unreconciled"

    decision = engine.evaluate_trade(decision=DummyDecision("BTCUSDT", "ENTER_LONG"), portfolio_state=portfolio, ts=1.0)

    assert decision.approved is False
    assert decision.final_action == "HOLD"
    assert decision.reason == "live_unreconciled_override"
