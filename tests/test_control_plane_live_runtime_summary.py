from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.order_models import OrderIntent
from execution.order_store import OrderStore
from execution.portfolio_state import PortfolioStateStore, PositionState
from monitoring.control_plane_store import ControlPlaneStore


class _Paths(SimpleNamespace):
    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _make_paths(tmp_path: Path) -> _Paths:
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
        order_state_path=runtime_dir / "order_state.json",
        execution_journal_path=runtime_dir / "execution_journal.jsonl",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        config=SimpleNamespace(
            risk=SimpleNamespace(account_balance_usd=1000.0),
            runtime=SimpleNamespace(execution_mode="live"),
            exchange=SimpleNamespace(
                position_reconciliation_enabled=True,
                require_reconciliation_for_live=True,
            ),
        ),
    )


def test_control_plane_summary_exposes_live_runtime_reconciliation_state(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    portfolio_store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = portfolio_store.load()
    state.authority_mode = "live_exchange_required"
    state.reconciliation_status = "mismatch"
    state.last_reconciled_at = None
    state.exchange_position_count = 2
    state.reconciliation_mismatch_count = 1
    state.last_reconciliation_error = "quantity_mismatch"
    state.exchange_snapshot_hash = "abc123"
    portfolio_store.save(state)

    store = ControlPlaneStore(paths)
    summary = store.summary()
    live_runtime = summary["live_runtime"]

    assert live_runtime["execution_mode"] == "live"
    assert live_runtime["reconciliation_enabled"] is True
    assert live_runtime["reconciliation_required"] is True
    assert live_runtime["exchange_authority_required"] is True
    assert live_runtime["authority_mode"] == "live_exchange_required"
    assert live_runtime["reconciliation_status"] == "mismatch"
    assert live_runtime["exchange_position_count"] == 2
    assert live_runtime["reconciliation_mismatch_count"] == 1
    assert live_runtime["last_reconciliation_error"] == "quantity_mismatch"
    assert live_runtime["exchange_snapshot_hash"] == "abc123"


def test_control_plane_summary_exposes_paper_local_authority_without_live_requirement(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    paths.config.runtime.execution_mode = "paper"

    store = ControlPlaneStore(paths)
    summary = store.summary()
    live_runtime = summary["live_runtime"]

    assert live_runtime["execution_mode"] == "paper"
    assert live_runtime["exchange_authority_required"] is False
    assert live_runtime["authoritative"] is True
    assert live_runtime["authority_reason"] == "paper_local_authority"


def test_control_plane_summary_exposes_open_order_counts_and_recovery_sources(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    portfolio_store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
    state = portfolio_store.load()
    state.positions["BTCUSDT"] = state.positions.get("BTCUSDT") or PositionState(
        symbol="BTCUSDT", side="LONG", quantity=0.01, notional_usd=300.0, entry_price=30000.0,
        stop_loss=0.0, take_profit=0.0, opened_at=1.0, updated_at=1.0
    )
    portfolio_store.save(state)

    order_store = OrderStore(paths.order_state_path, paths.execution_journal_path)
    order = order_store.create_from_intent(
        OrderIntent(symbol="BTCUSDT", action="ENTER_LONG", side="LONG", quantity=0.01, signal_reason="unit_test"),
        ts=1.0,
    )
    submitted = order_store.update_submission(order.order_id, status="submitted", ts=2.0, exchange_order_id="ex-1", exchange_status="NEW")
    submitted.metadata["last_recovery_source"] = "exchange_open_orders"
    payload = order_store._load_payload()
    payload["orders"][submitted.order_id] = submitted.to_dict()
    order_store._save_payload(payload)

    summary = ControlPlaneStore(paths).summary()["live_runtime"]

    assert summary["open_position_count"] == 1
    assert summary["open_order_count"] == 1
    assert summary["last_recovery_sources"] == ["exchange_open_orders"]
