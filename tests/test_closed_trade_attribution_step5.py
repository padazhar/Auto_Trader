from __future__ import annotations

from types import SimpleNamespace

from execution.portfolio_state import PortfolioStateStore
from reporting.run_report import build_run_report


def test_closed_trade_carries_entry_model_risk_and_outcome_attribution(tmp_path):
    store = PortfolioStateStore(tmp_path / "portfolio_state.json", starting_equity_usd=1000.0)
    state = store.load()
    lineage = {
        "decision_id": "dec-step5",
        "contract_version": "trade_lineage_v1",
        "signal_action": "ENTER_LONG",
        "signal_reason": "ev_enter_long",
        "ev_score": 0.0123,
        "risk_approved": True,
        "model_candidates": {"rf": "rf-1", "gbm": "gbm-1", "policy": "policy-1"},
        "prediction_vector": {"rf": {"long": 0.72, "hold": 0.2, "short": 0.08}},
    }
    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100.0,
        ts=100.0,
        position_plan={
            "quantity": 1.0,
            "notional_usd": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "lineage": lineage,
        },
    )
    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="EXIT_LONG",
        price=106.0,
        ts=120.0,
        position_plan={"exit_reason": "protective_take_profit"},
    )

    closed = state.closed_trades[-1]
    assert closed.attribution["schema_version"] == "closed_trade_attribution.v1"
    assert closed.attribution["decision_id"] == "dec-step5"
    assert closed.attribution["entry_reason"] == "ev_enter_long"
    assert closed.attribution["rf_candidate_id"] == "rf-1"
    assert closed.attribution["gbm_candidate_id"] == "gbm-1"
    assert closed.attribution["policy_candidate_id"] == "policy-1"
    assert closed.attribution["risk_approved"] is True
    assert closed.attribution["prediction_matched_outcome"] is True


def test_run_report_summarizes_closed_trade_attribution(tmp_path):
    portfolio_path = tmp_path / "portfolio_state.json"
    store = PortfolioStateStore(portfolio_path, starting_equity_usd=1000.0)
    state = store.load()
    store.apply_execution(
        state,
        symbol="ETHUSDT",
        action="ENTER_SHORT",
        price=200.0,
        ts=10.0,
        position_plan={
            "quantity": 1.0,
            "notional_usd": 200.0,
            "stop_loss": 210.0,
            "take_profit": 180.0,
            "lineage": {"decision_id": "dec-short", "signal_action": "ENTER_SHORT", "signal_reason": "ev_enter_short", "risk_approved": True},
        },
    )
    store.apply_execution(
        state,
        symbol="ETHUSDT",
        action="EXIT_SHORT",
        price=190.0,
        ts=20.0,
        position_plan={"exit_reason": "protective_take_profit"},
    )
    store.save(state)

    paths = SimpleNamespace(
        project_root=tmp_path,
        portfolio_state_path=portfolio_path,
        live_backtest_trades_path=tmp_path / "missing.csv",
        decision_journal_path=tmp_path / "missing.jsonl",
        training_registry_path=tmp_path / "missing_registry.json",
        symbol_state_path=tmp_path / "missing_symbols.json",
        runtime_dir=tmp_path,
    )
    report = build_run_report(paths, monitor=SimpleNamespace(snapshot=lambda: {}))
    attribution = report["closed_trade_attribution"]
    assert attribution["schema_version"] == "closed_trade_attribution_report.v1"
    assert attribution["closed_trade_count"] == 1
    assert attribution["missing_decision_id_count"] == 0
    assert attribution["prediction_match_counts"]["True"] == 1
