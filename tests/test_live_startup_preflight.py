from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main as main_module
from execution.risk_engine import RiskEngine


class DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def _paths(project_root: Path, *, reconciliation_enabled: bool = True) -> SimpleNamespace:
    runtime_dir = project_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=project_root,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        config=SimpleNamespace(
            risk=SimpleNamespace(
                account_balance_usd=1000.0,
                risk_per_trade=0.01,
                sl_atr_mult=1.2,
                tp_atr_mult=1.8,
                max_notional_usd=5000.0,
                min_notional_usd=25.0,
                max_total_exposure_usd=7500.0,
                max_concurrent_positions=6,
                max_symbol_notional_usd=1500.0,
                max_same_direction_exposure_usd=4500.0,
                max_risk_distortion_fraction=0.35,
                daily_loss_limit_usd=75.0,
                max_drawdown_fraction=0.12,
                max_correlated_positions=2,
                correlation_groups={},
            ),
            runtime=SimpleNamespace(execution_mode="live", allow_unreconciled_live_mode=False),
            exchange=SimpleNamespace(
                futures_rest_base_url="https://fapi.binance.com",
                api_key="key",
                api_secret="secret",
                recv_window_ms=5000,
                use_testnet=False,
                position_reconciliation_enabled=reconciliation_enabled,
                require_reconciliation_for_live=True,
                max_reconciliation_age_sec=30,
            ),
        ),
    )


def test_live_startup_preflight_blocks_disabled_reconciliation(tmp_path: Path) -> None:
    paths = _paths(tmp_path, reconciliation_enabled=False)
    with pytest.raises(SystemExit) as exc_info:
        main_module._live_startup_preflight_if_applicable(paths, DummyLogger(), "live")
    assert exc_info.value.code == 5


def test_live_startup_preflight_blocks_mismatch(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path, reconciliation_enabled=True)
    engine = RiskEngine(paths)
    portfolio = engine.load_portfolio()
    engine.portfolio_store.apply_execution(
        portfolio,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100000.0,
        ts=1.0,
        position_plan={"quantity": 0.01, "notional_usd": 1000.0, "stop_loss": 0.0, "take_profit": 0.0},
    )
    engine.save_portfolio(portfolio)

    class _DummyReconciler:
        def __init__(self, paths) -> None:
            self.enabled = True

        def reconcile(self, portfolio_state, ts: float):
            return SimpleNamespace(
                success=True,
                reconciled=False,
                status="mismatch",
                exchange_position_count=0,
                mismatch_count=1,
                snapshot_hash="hash",
                error="",
            )

    monkeypatch.setattr("main.BinanceFuturesReconciler", _DummyReconciler, raising=False)
    # helper imports locally from execution.exchange_reconciliation, patch there too
    monkeypatch.setattr("execution.exchange_reconciliation.BinanceFuturesReconciler", _DummyReconciler)

    with pytest.raises(SystemExit) as exc_info:
        main_module._live_startup_preflight_if_applicable(paths, DummyLogger(), "live")
    assert exc_info.value.code == 5
