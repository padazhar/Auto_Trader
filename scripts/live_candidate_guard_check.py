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
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from execution.exchange_reconciliation import BinanceFuturesReconciler
from execution.portfolio_state import PortfolioStateStore
from execution.risk_engine import RiskEngine


@dataclass(slots=True)
class LiveGuardCheckItem:
    """Single live-candidate guard validation result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiveCandidateGuardReport:
    """Structured report for live-only reconciliation and authority validation."""

    project_root: str
    overall_status: str
    live_guard_ready: bool
    items: list[LiveGuardCheckItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "live_guard_ready": self.live_guard_ready,
            "items": [asdict(item) for item in self.items],
        }


class _Paths(SimpleNamespace):
    """Minimal isolated paths/config bundle for live-guard validation."""

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _record(items: list[LiveGuardCheckItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a single validation result to the report."""
    items.append(LiveGuardCheckItem(name=name, status=status, message=message, details=dict(details)))


def _build_paths(temp_root: Path, *, reconciliation_enabled: bool = True, allow_unreconciled_live_mode: bool = False) -> _Paths:
    """Create an isolated path/config environment for live-only guard checks."""
    project_root = temp_root / "repo"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"

    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config = SimpleNamespace(
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
        runtime=SimpleNamespace(
            execution_mode="live",
            allow_unreconciled_live_mode=allow_unreconciled_live_mode,
        ),
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
    )
    return _Paths(
        project_root=project_root,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        config=config,
    )


class _PatchFetch:
    """Simple context manager for temporary fetch overriding."""

    def __init__(self, reconciler: BinanceFuturesReconciler, fake_fetch: Callable[[], list[Any]]) -> None:
        self.reconciler = reconciler
        self.fake_fetch = fake_fetch
        self.original = reconciler.fetch_open_positions

    def __enter__(self):
        self.reconciler.fetch_open_positions = self.fake_fetch  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.reconciler.fetch_open_positions = self.original  # type: ignore[assignment]


def _check_disabled_reconciliation(items: list[LiveGuardCheckItem]) -> None:
    """Verify live mode fails closed when reconciliation is disabled."""
    with tempfile.TemporaryDirectory(prefix="live_guard_disabled_") as tmp_dir:
        paths = _build_paths(Path(tmp_dir), reconciliation_enabled=False)
        engine = RiskEngine(paths)
        portfolio_state = engine.load_portfolio()
        authority = engine.live_authority_summary(portfolio_state)
        if authority.get("reason") != "live_reconciliation_required" or authority.get("guard_required") is not True:
            _record(
                items,
                "disabled_reconciliation_guard",
                "fail",
                "Live mode did not fail closed when reconciliation was disabled.",
                authority=authority,
            )
            return
        _record(items, "disabled_reconciliation_guard", "pass", "Live mode fails closed when reconciliation is disabled.")


def _check_reconciliation_error(items: list[LiveGuardCheckItem]) -> None:
    """Verify reconciliation fetch failures surface as guarded live authority."""
    with tempfile.TemporaryDirectory(prefix="live_guard_error_") as tmp_dir:
        paths = _build_paths(Path(tmp_dir), reconciliation_enabled=True)
        engine = RiskEngine(paths)
        portfolio_state = engine.load_portfolio()
        reconciler = BinanceFuturesReconciler(paths)

        def _boom() -> list[Any]:
            raise RuntimeError("exchange_unavailable")

        with _PatchFetch(reconciler, _boom):
            result = reconciler.reconcile(portfolio_state, ts=datetime.now(timezone.utc).timestamp())

        engine.portfolio_store.record_reconciliation(portfolio_state, result, ts=datetime.now(timezone.utc).timestamp())
        authority = engine.live_authority_summary(portfolio_state)
        if result.status != "error" or authority.get("reason") != "live_reconciliation_error":
            _record(
                items,
                "reconciliation_error_guard",
                "fail",
                "Live mode did not surface reconciliation errors as a guarded state.",
                reconciliation_result=asdict(result),
                authority=authority,
            )
            return
        _record(items, "reconciliation_error_guard", "pass", "Reconciliation fetch errors fail closed in live mode.")


def _check_reconciliation_mismatch(items: list[LiveGuardCheckItem]) -> None:
    """Verify exchange-vs-ledger mismatches fail closed."""
    with tempfile.TemporaryDirectory(prefix="live_guard_mismatch_") as tmp_dir:
        paths = _build_paths(Path(tmp_dir), reconciliation_enabled=True)
        engine = RiskEngine(paths)
        portfolio_state = engine.load_portfolio()
        ts = datetime.now(timezone.utc).timestamp()
        engine.portfolio_store.apply_execution(
            portfolio_state,
            symbol="BTCUSDT",
            action="ENTER_LONG",
            price=100000.0,
            ts=ts,
            position_plan={
                "quantity": 0.01,
                "notional_usd": 1000.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
            },
        )
        reconciler = BinanceFuturesReconciler(paths)
        with _PatchFetch(reconciler, lambda: []):
            result = reconciler.reconcile(portfolio_state, ts=ts)
        engine.portfolio_store.record_reconciliation(portfolio_state, result, ts=ts)
        authority = engine.live_authority_summary(portfolio_state)
        if result.status != "mismatch" or authority.get("reason") != "live_reconciliation_mismatch":
            _record(
                items,
                "reconciliation_mismatch_guard",
                "fail",
                "Live mode did not fail closed on reconciliation mismatch.",
                reconciliation_result=asdict(result),
                authority=authority,
            )
            return
        _record(items, "reconciliation_mismatch_guard", "pass", "Exchange-vs-ledger mismatches fail closed in live mode.")


def _check_reconciliation_stale(items: list[LiveGuardCheckItem]) -> None:
    """Verify stale reconciled state is rejected for live authority."""
    with tempfile.TemporaryDirectory(prefix="live_guard_stale_") as tmp_dir:
        paths = _build_paths(Path(tmp_dir), reconciliation_enabled=True)
        engine = RiskEngine(paths)
        portfolio_state = engine.load_portfolio()
        old_ts = datetime.now(timezone.utc).timestamp() - 120.0
        portfolio_state.reconciliation_status = "reconciled"
        portfolio_state.last_reconciled_at = old_ts
        authority = engine.live_authority_summary(portfolio_state)
        if authority.get("reason") != "live_reconciliation_stale" or authority.get("guard_required") is not True:
            _record(
                items,
                "reconciliation_stale_guard",
                "fail",
                "Stale reconciliation was not rejected by live authority.",
                authority=authority,
            )
            return
        _record(items, "reconciliation_stale_guard", "pass", "Stale reconciliation is rejected by live authority.")


def _check_unreconciled_override(items: list[LiveGuardCheckItem]) -> None:
    """Verify the explicit unreconciled-live override remains visible and non-authoritative."""
    with tempfile.TemporaryDirectory(prefix="live_guard_override_") as tmp_dir:
        paths = _build_paths(Path(tmp_dir), reconciliation_enabled=True, allow_unreconciled_live_mode=True)
        engine = RiskEngine(paths)
        portfolio_state = engine.load_portfolio()
        authority = engine.live_authority_summary(portfolio_state)
        if authority.get("reason") != "live_unreconciled_override" or authority.get("authoritative") is not False:
            _record(
                items,
                "unreconciled_override_visibility",
                "fail",
                "Unreconciled live override did not surface the expected non-authoritative state.",
                authority=authority,
            )
            return
        _record(items, "unreconciled_override_visibility", "pass", "Explicit unreconciled live override remains visible and non-authoritative.")


def run_live_candidate_guard_check(project_root: Path | str | None = None) -> LiveCandidateGuardReport:
    """Run live-only validation of reconciliation and fail-closed authority guards."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[LiveGuardCheckItem] = []

    _check_disabled_reconciliation(items)
    _check_reconciliation_error(items)
    _check_reconciliation_mismatch(items)
    _check_reconciliation_stale(items)
    _check_unreconciled_override(items)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    return LiveCandidateGuardReport(
        project_root=str(root),
        overall_status=overall_status,
        live_guard_ready=(overall_status == "pass"),
        items=items,
    )


def main() -> int:
    """CLI entrypoint for live-only reconciliation/authority guard validation."""
    parser = argparse.ArgumentParser(description="Live candidate guard check")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args()

    report = run_live_candidate_guard_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.live_guard_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
