from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.shared_utils import now_utc_ts
from execution.portfolio_state import PortfolioStateStore
from monitoring.control_plane_store import ControlPlaneStore
from monitoring.publisher import MonitorPublisher


@dataclass(slots=True)
class SoakCheckItem:
    """Single paper-mode soak check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperModeSoakReport:
    """Structured paper-mode soak validation report."""

    project_root: str
    overall_status: str
    paper_mode_soak_ready: bool
    items: list[SoakCheckItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the soak report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "paper_mode_soak_ready": self.paper_mode_soak_ready,
            "items": [asdict(item) for item in self.items],
        }


class _SoakPaths(SimpleNamespace):
    """Minimal path/config bundle for isolated paper-mode soak validation."""

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _record(items: list[SoakCheckItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a soak-check item to the report."""
    items.append(SoakCheckItem(name=name, status=status, message=message, details=dict(details)))


def _build_isolated_paths(temp_root: Path) -> _SoakPaths:
    """Build an isolated runtime path bundle for persistence-oriented validation."""
    project_root = temp_root / "repo"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir = project_root / "data" / "model_storage"

    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_storage_dir.mkdir(parents=True, exist_ok=True)

    monitoring_cfg = SimpleNamespace(
        max_events=100,
        max_alerts=50,
        max_market_updates=50,
        max_recent_training_results=50,
        max_recent_decisions=50,
        max_recent_signals=50,
        max_backtest_open_positions=50,
        max_backtest_recent_closed=50,
        max_registry_entries=100,
    )
    risk_cfg = SimpleNamespace(account_balance_usd=1000.0)
    runtime_cfg = SimpleNamespace(execution_mode="paper", allow_unreconciled_live_mode=False)
    exchange_cfg = SimpleNamespace(
        position_reconciliation_enabled=False,
        require_reconciliation_for_live=True,
    )

    config = SimpleNamespace(
        monitoring=monitoring_cfg,
        risk=risk_cfg,
        runtime=runtime_cfg,
        exchange=exchange_cfg,
    )

    return _SoakPaths(
        project_root=project_root,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        model_storage_dir=model_storage_dir,
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        monitoring_state_path=runtime_dir / "monitoring_state.json",
        monitoring_events_path=runtime_dir / "monitoring_events.json",
        config=config,
    )


def _reset_monitor_singleton(paths: _SoakPaths) -> None:
    """Clear the per-project monitor singleton so reload behavior can be tested."""
    key = str(paths.project_root)
    with MonitorPublisher._instances_lock:
        MonitorPublisher._instances.pop(key, None)


def _check_control_plane_roundtrip(paths: _SoakPaths, items: list[SoakCheckItem]) -> None:
    """Verify control-plane state persists and reloads correctly."""
    try:
        store = ControlPlaneStore(paths)
        store.set_pause(paused=True, pause_reason="paper_soak_test", updated_by="test")
        reloaded = ControlPlaneStore(paths).summary()
        if reloaded.get("paused") is not True:
            _record(items, "control_plane_roundtrip", "fail", "Pause state was not persisted correctly.", summary=reloaded)
            return
        if str(reloaded.get("pause_reason") or "") != "paper_soak_test":
            _record(items, "control_plane_roundtrip", "fail", "Pause reason was not persisted correctly.", summary=reloaded)
            return
        _record(items, "control_plane_roundtrip", "pass", "Control-plane state persisted and reloaded successfully.")
    except Exception as exc:
        _record(items, "control_plane_roundtrip", "fail", "Control-plane persistence check failed.", error=str(exc))


def _check_portfolio_roundtrip(paths: _SoakPaths, items: list[SoakCheckItem]) -> None:
    """Verify portfolio state survives entry, mark-to-market, exit, and reload cycles."""
    try:
        store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=1000.0)
        state = store.load()
        ts_open = now_utc_ts()

        entry_report = store.apply_execution(
            state,
            symbol="BTCUSDT",
            action="ENTER_LONG",
            price=100.0,
            ts=ts_open,
            position_plan={
                "quantity": 2.0,
                "notional_usd": 200.0,
                "stop_loss": 95.0,
                "take_profit": 110.0,
            },
        )
        if "BTCUSDT" not in state.positions:
            _record(items, "portfolio_roundtrip", "fail", "Entered position was not recorded.", execution_report=entry_report)
            return

        pnl_summary_open = store.mark_to_market(state, {"BTCUSDT": 103.0}, ts=ts_open + 1.0)
        store.save(state)

        reloaded_after_open = store.load()
        if "BTCUSDT" not in reloaded_after_open.positions:
            _record(items, "portfolio_roundtrip", "fail", "Open position did not survive save/reload.", pnl_summary=pnl_summary_open)
            return

        exit_report = store.apply_execution(
            reloaded_after_open,
            symbol="BTCUSDT",
            action="EXIT",
            price=105.0,
            ts=ts_open + 2.0,
        )
        store.mark_to_market(reloaded_after_open, {}, ts=ts_open + 3.0)
        store.save(reloaded_after_open)

        reloaded_after_exit = store.load()
        if reloaded_after_exit.positions:
            _record(items, "portfolio_roundtrip", "fail", "Closed position remained after exit/save/reload.", positions=list(reloaded_after_exit.positions.keys()))
            return
        if float(reloaded_after_exit.realized_pnl_usd) <= 0.0:
            _record(
                items,
                "portfolio_roundtrip",
                "fail",
                "Realized PnL did not persist after exit.",
                exit_report=exit_report,
                realized_pnl_usd=reloaded_after_exit.realized_pnl_usd,
            )
            return

        _record(
            items,
            "portfolio_roundtrip",
            "pass",
            "Portfolio state persisted correctly across entry, MTM, exit, and reload.",
            realized_pnl_usd=float(reloaded_after_exit.realized_pnl_usd),
            equity_usd=float(reloaded_after_exit.last_equity_usd),
        )
    except Exception as exc:
        _record(items, "portfolio_roundtrip", "fail", "Portfolio persistence check failed.", error=str(exc))


def _check_monitoring_persistence(paths: _SoakPaths, items: list[SoakCheckItem]) -> None:
    """Verify monitoring state/events persist and reload cleanly."""
    try:
        paths.decision_journal_path.write_text(
            json.dumps(
                {
                    "ts": now_utc_ts(),
                    "symbol": "BTCUSDT",
                    "interval": "15m",
                    "action": "HOLD",
                    "reason": "paper_soak_test",
                    "score": 0.0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        _reset_monitor_singleton(paths)
        publisher = MonitorPublisher(paths)
        publisher.publish_system(launcher_mode="all", python_version="test")
        publisher.heartbeat("signal_generator", status="running", details={"phase": "soak"})
        publisher.controller_success("signal_generator", details={"phase": "steady"})
        publisher.flush()

        _reset_monitor_singleton(paths)
        reloaded = MonitorPublisher(paths)
        snapshot = reloaded.snapshot()
        events = reloaded.events()

        system = snapshot.get("system", {})
        controllers = snapshot.get("controllers", {})
        decision_journal = snapshot.get("decision_journal", {})

        if str(system.get("launcher_mode") or "") != "all":
            _record(items, "monitoring_persistence", "fail", "Monitoring system state did not persist correctly.", snapshot=snapshot)
            return
        if str((controllers.get("signal_generator") or {}).get("status") or "") != "healthy":
            _record(items, "monitoring_persistence", "fail", "Controller health did not persist correctly.", snapshot=snapshot)
            return
        if int(decision_journal.get("entries") or 0) < 1:
            _record(items, "monitoring_persistence", "fail", "Decision journal summary was not reflected into monitoring state.", snapshot=snapshot)
            return
        if not isinstance(events, list):
            _record(items, "monitoring_persistence", "fail", "Monitoring events payload did not reload as a list.")
            return

        _record(
            items,
            "monitoring_persistence",
            "pass",
            "Monitoring state and event persistence/reload succeeded.",
            event_count=len(events),
            decision_journal_entries=int(decision_journal.get("entries") or 0),
        )
    except Exception as exc:
        _record(items, "monitoring_persistence", "fail", "Monitoring persistence check failed.", error=str(exc))


def run_paper_mode_soak_check(project_root: Path | str | None = None) -> PaperModeSoakReport:
    """Run persistence-oriented paper-mode soak validation in an isolated temp environment."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[SoakCheckItem] = []

    with tempfile.TemporaryDirectory(prefix="paper_mode_soak_") as tmp_dir:
        paths = _build_isolated_paths(Path(tmp_dir))
        _check_control_plane_roundtrip(paths, items)
        _check_portfolio_roundtrip(paths, items)
        _check_monitoring_persistence(paths, items)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    paper_mode_soak_ready = overall_status == "pass"
    return PaperModeSoakReport(
        project_root=str(root),
        overall_status=overall_status,
        paper_mode_soak_ready=paper_mode_soak_ready,
        items=items,
    )


def main() -> int:
    """CLI entrypoint for paper-mode soak validation."""
    parser = argparse.ArgumentParser(description="Paper-mode soak check")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args()

    report = run_paper_mode_soak_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.paper_mode_soak_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
