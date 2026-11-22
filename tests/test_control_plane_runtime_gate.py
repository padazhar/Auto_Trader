# tests/test_control_plane_runtime_gate.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.portfolio_state import PortfolioStateStore
from monitoring.control_plane_store import ControlPlaneStore


class _Paths(SimpleNamespace):
    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _make_paths(tmp_path: Path, *, execution_mode: str = "live") -> _Paths:
    project_root = tmp_path / "repo"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir = project_root / "data" / "model_storage"

    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_storage_dir.mkdir(parents=True, exist_ok=True)

    return _Paths(
        project_root=project_root,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        model_storage_dir=model_storage_dir,
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        config=SimpleNamespace(
            risk=SimpleNamespace(account_balance_usd=1000.0),
            runtime=SimpleNamespace(execution_mode=execution_mode),
            exchange=SimpleNamespace(
                position_reconciliation_enabled=True,
                require_reconciliation_for_live=True,
            ),
        ),
    )


def test_runtime_gate_blocks_on_circuit_breaker(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, execution_mode="paper")
    store = ControlPlaneStore(paths)
    store.circuit_store.save(
        store.circuit_store.load().__class__(
            triggered=True,
            reason="daily_loss_limit",
            triggered_at=1.0,
            last_checked_at=2.0,
            peak_equity_usd=1000.0,
            current_equity_usd=900.0,
            drawdown_fraction=0.1,
            daily_realized_pnl_usd=-100.0,
            day_key="2026-04-13",
        )
    )

    gate = store.runtime_gate()
    assert gate["blocked"] is True
    assert gate["reason"] == "circuit_breaker_triggered"
    assert gate["status"] == "blocked"
    assert gate["trading_enabled"] is False


def test_runtime_gate_blocks_on_live_non_authoritative_state(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path, execution_mode="live")
    portfolio_store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = portfolio_store.load()
    state.authority_mode = "live_exchange_required"
    state.reconciliation_status = "mismatch"
    state.last_reconciled_at = None
    portfolio_store.save(state)

    store = ControlPlaneStore(paths)
    summary = store.summary()
    gate = summary["runtime_gate"]

    assert summary["live_runtime"]["authoritative"] is False
    assert summary["live_runtime"]["authority_reason"] == "mismatch"
    assert gate["blocked"] is True
    assert gate["reason"] == "mismatch"
    assert gate["status"] == "guarded"
    assert gate["trading_enabled"] is False
