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

from monitoring.control_plane_store import ControlPlaneStore


@dataclass(slots=True)
class OperationalCheckItem:
    """Single live-candidate operational check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OperationalCheckReport:
    """Structured report for live-candidate operational readiness."""

    project_root: str
    overall_status: str
    live_candidate_ready: bool
    items: list[OperationalCheckItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "live_candidate_ready": self.live_candidate_ready,
            "items": [asdict(item) for item in self.items],
        }


class _Paths(SimpleNamespace):
    """Minimal isolated paths/config bundle for operational-control validation."""


class _NullLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def _record(items: list[OperationalCheckItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(OperationalCheckItem(name=name, status=status, message=message, details=dict(details)))


def _build_paths(temp_root: Path) -> _Paths:
    project_root = temp_root / "repo"
    runtime_dir = project_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return _Paths(
        project_root=project_root,
        runtime_dir=runtime_dir,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        config=SimpleNamespace(
            runtime=SimpleNamespace(execution_mode="paper", allow_unreconciled_live_mode=False),
            exchange=SimpleNamespace(
                position_reconciliation_enabled=True,
                require_reconciliation_for_live=True,
            ),
            risk=SimpleNamespace(account_balance_usd=1000.0),
        ),
    )


def _check_startup_guard_contract(items: list[OperationalCheckItem]) -> None:
    try:
        import main as main_module

        required = [
            hasattr(main_module, "_validate_runtime_mode_guards"),
            hasattr(main_module, "_validate_live_trading_readiness"),
            hasattr(main_module, "_live_startup_preflight_if_applicable"),
        ]
        if not all(required):
            _record(items, "startup_guard_contract", "fail", "Startup guard contract is incomplete.")
            return
        _record(items, "startup_guard_contract", "pass", "Startup guard contract exposes runtime, preflight, and final live gates.")
    except Exception as exc:
        _record(items, "startup_guard_contract", "fail", "Startup guard contract import failed.", error=str(exc))


def _check_authority_state_transitions(items: list[OperationalCheckItem]) -> None:
    try:
        from execution.portfolio_state import PortfolioStateStore
        from execution.risk_engine import RiskEngine
        from runners.live_trading_controller import LiveTradingController

        required = [
            hasattr(PortfolioStateStore, "record_reconciliation"),
            hasattr(RiskEngine, "live_authority_summary"),
            hasattr(LiveTradingController, "_run_reconciliation") or hasattr(LiveTradingController, "_maybe_reconcile"),
        ]
        if not all(required):
            _record(items, "authority_state_transitions", "fail", "Authority-state transition hooks are incomplete.")
            return
        _record(items, "authority_state_transitions", "pass", "Authority-state transition hooks are present for reconciliation and runtime authority.")
    except Exception as exc:
        _record(items, "authority_state_transitions", "fail", "Authority-state transition import failed.", error=str(exc))


def _check_control_plane_live_visibility(items: list[OperationalCheckItem]) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="candidate_operational_visibility_") as tmp_dir:
            paths = _build_paths(Path(tmp_dir))
            store = ControlPlaneStore(paths)
            summary = store.summary()
            live_runtime = summary.get("live_runtime", {}) if isinstance(summary, dict) else {}
            runtime_gate = summary.get("runtime_gate", {}) if isinstance(summary, dict) else {}
            if not isinstance(live_runtime, dict) or not isinstance(runtime_gate, dict):
                _record(items, "control_plane_live_visibility", "fail", "Control-plane summary does not expose live visibility payloads.")
                return
            _record(items, "control_plane_live_visibility", "pass", "Control-plane summary exposes live runtime and runtime gate visibility.")
    except Exception as exc:
        _record(items, "control_plane_live_visibility", "fail", "Control-plane live visibility failed.", error=str(exc))


def _check_pause_roundtrip(items: list[OperationalCheckItem]) -> None:
    try:
        with tempfile.TemporaryDirectory(prefix="candidate_operational_pause_") as tmp_dir:
            paths = _build_paths(Path(tmp_dir))
            store = ControlPlaneStore(paths)
            paused = store.set_pause(paused=True, pause_reason="operational_check", updated_by="check")
            resumed = store.set_pause(paused=False, pause_reason="", updated_by="check")
            if paused.get("paused") is not True or resumed.get("paused") is not False:
                _record(items, "pause_roundtrip", "fail", "Pause state did not roundtrip through the control plane.", paused=paused, resumed=resumed)
                return
            _record(items, "pause_roundtrip", "pass", "Pause state roundtrips cleanly through the control plane store.")
    except Exception as exc:
        _record(items, "pause_roundtrip", "fail", "Pause roundtrip validation failed.", error=str(exc))




def _check_shadow_candidate_contract(items: list[OperationalCheckItem]) -> None:
    try:
        from monitoring.control_plane_store import ControlPlaneStore
        from runners.live_trading_controller import LiveTradingController
        with tempfile.TemporaryDirectory(prefix="candidate_operational_shadow_") as tmp_dir:
            paths = _build_paths(Path(tmp_dir))
            store = ControlPlaneStore(paths)
            summary = store.summary()
            has_preview = "shadow_candidate_preview" in summary
            has_hook = hasattr(LiveTradingController, "_compute_shadow_candidate_decision")
            if not has_preview or not has_hook:
                _record(items, "shadow_candidate_contract", "fail", "Shadow-candidate runtime contract is incomplete.", has_preview=has_preview, has_hook=has_hook)
                return
            _record(items, "shadow_candidate_contract", "pass", "Shadow-candidate runtime contract and monitoring preview are present.")
    except Exception as exc:
        _record(items, "shadow_candidate_contract", "fail", "Shadow-candidate contract validation failed.", error=str(exc))

def run_live_candidate_operational_check(project_root: Path | str | None = None) -> OperationalCheckReport:
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[OperationalCheckItem] = []

    _check_startup_guard_contract(items)
    _check_authority_state_transitions(items)
    _check_control_plane_live_visibility(items)
    _check_pause_roundtrip(items)
    _check_shadow_candidate_contract(items)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    return OperationalCheckReport(
        project_root=str(root),
        overall_status=overall_status,
        live_candidate_ready=(overall_status == "pass"),
        items=items,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live candidate operational check")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    report = run_live_candidate_operational_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.live_candidate_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
