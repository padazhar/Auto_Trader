from __future__ import annotations

import copy
import json
from collections import deque
from contextlib import contextmanager
import threading
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - POSIX runtime uses fcntl; fallback preserves portability.
    fcntl = None

from core.path_manager import PathManager
from execution.portfolio_state import apply_execution_trade_source
from execution.symbol_expectancy_guard import SymbolExpectancyGuardConfig, summarize_symbol_expectancy
from inference.readiness import readiness_blocker_report
from core.shared_utils import atomic_write_bytes, now_utc_ts
from core.symbol_management import load_symbol_state, normalize_symbol_list, reconcile_symbol_lifecycle_with_readiness_snapshot


_DEFAULT_LIMITS = {
    "events": 300,
    "alerts": 100,
    "market_updates": 100,
    "recent_training_results": 120,
    "recent_decisions": 50,
    "recent_signals": 100,
    "backtest_open_positions": 100,
    "backtest_recent_closed": 50,
    "backtest_trade_history": 120,
    "registry_entries": 300,
    "scheduler_jobs": 50,
    "training_jobs": 50,
    "control_actions": 100,
    "decision_trace": 100,
    "decision_journal_recent": 25,
    "decision_journal_scan_bytes": 2_000_000,
}

_COMPACT_DECISION_KEYS = (
    "ts",
    "symbol",
    "mode",
    "action",
    "effective_action",
    "reason",
    "decision_stage",
    "journal_writer",
    "score",
    "ev",
    "price",
    "execution_status",
    "execution_blocker_stage",
    "execution_blocker",
    "rejection_reason",
    "order_status",
    "risk_breakdown",
)


def _bounded_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = int(default)
    return max(minimum, min(parsed, maximum))


def _compact_row(row: Any, *, keys: tuple[str, ...] | None = None) -> Any:
    if not isinstance(row, dict):
        return row
    if keys is None:
        return dict(row)
    return {key: row.get(key) for key in keys if key in row}


def _nested_get(row: dict[str, Any], *path: str) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _classify_rejection_reason(reason: Any, *, stage: Any = None, status: Any = None) -> str:
    text = " ".join(
        str(value or "").lower()
        for value in (reason, stage, status)
        if value is not None
    )
    if not text.strip():
        return "unknown"
    if "model_disagreement" in text or "disagreement" in text:
        return "model_disagreement"
    if "movement_filter" in text or "movement-filter" in text or "movement" in text:
        return "movement_filter"
    if "negative_ev" in text or "expected_value" in text or " ev" in f" {text}" or "ev_" in text:
        return "expected_value"
    if "price_unavailable" in text or "price stale" in text or "stale_price" in text or "missing_price" in text or "no_price" in text:
        return "price"
    if "readiness" in text or "training_incomplete" in text or "registry_status" in text or "model_unavailable" in text or "missing_model" in text or "part" in text:
        return "readiness"
    if "regime" in text:
        return "regime"
    if "hold_long_bias" in text or "hold_short_bias" in text or "bias" in text:
        return "bias"
    if "risk" in text:
        return "risk"
    return "other"


def _decision_rejection_reason(row: dict[str, Any]) -> Any:
    return _first_non_empty(
        row.get("rejection_reason"),
        row.get("execution_blocker"),
        row.get("decision_blocker"),
        row.get("blocker"),
        _nested_get(row, "execution_contract", "rejection_reason"),
        _nested_get(row, "execution_contract", "execution_blocker"),
        _nested_get(row, "components", "execution_contract", "rejection_reason"),
        _nested_get(row, "components", "execution_contract", "execution_blocker"),
        _nested_get(row, "components", "decision_blocker"),
        _nested_get(row, "components", "decision_state", "blocker"),
        row.get("reason"),
    )


def _normalize_rejection_visibility_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    execution_contract = row.get("execution_contract")
    if not isinstance(execution_contract, dict):
        execution_contract = _nested_get(row, "components", "execution_contract")
    if not isinstance(execution_contract, dict):
        execution_contract = {}

    reason = _decision_rejection_reason(row)
    status = _first_non_empty(
        row.get("execution_status"),
        execution_contract.get("execution_status"),
        row.get("status"),
        row.get("order_status"),
    )
    stage = _first_non_empty(
        row.get("execution_blocker_stage"),
        execution_contract.get("execution_blocker_stage"),
        row.get("decision_stage"),
    )
    category = _classify_rejection_reason(reason, stage=stage, status=status)
    risk_breakdown = row.get("risk_breakdown")
    if not isinstance(risk_breakdown, dict):
        nested_breakdown = _nested_get(row, "components", "risk_breakdown")
        risk_breakdown = nested_breakdown if isinstance(nested_breakdown, dict) else {}
    return {
        "ts": row.get("ts") or row.get("updated_at") or row.get("created_at"),
        "symbol": str(row.get("symbol") or "--"),
        "mode": str(row.get("mode") or row.get("execution_mode") or "--"),
        "action": str(row.get("effective_action") or row.get("action") or "--"),
        "status": str(status or "--"),
        "stage": str(stage or "--"),
        "reason": str(reason or row.get("reason") or "--"),
        "category": category,
        "score": row.get("score"),
        "ev": row.get("ev"),
        "risk_breakdown": dict(risk_breakdown or {}),
        "source": str(row.get("journal_writer") or row.get("source") or "--"),
    }




def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_execution_summary(summary: Any, *, open_positions: list[Any], recent_decisions: list[Any], trade_history: list[Any] | None = None) -> dict[str, Any]:
    normalized = dict(summary or {}) if isinstance(summary, dict) else {}
    closed_rows = [row for row in (trade_history or []) if isinstance(row, dict)]
    open_count = _safe_int(
        normalized.get("open_positions", normalized.get("open_count", len(open_positions))),
        len(open_positions),
    )
    completed_count = _safe_int(
        normalized.get("completed_count", normalized.get("completed", normalized.get("trades", normalized.get("closed_count", len(closed_rows))))),
        len(closed_rows),
    )
    derived_wins = sum(1 for row in closed_rows if _safe_float(row.get("pnl_usd", row.get("realized_pnl_usd", 0.0)), 0.0) > 0.0)
    derived_losses = sum(1 for row in closed_rows if _safe_float(row.get("pnl_usd", row.get("realized_pnl_usd", 0.0)), 0.0) < 0.0)
    wins = _safe_int(normalized.get("wins", derived_wins), derived_wins)
    losses = _safe_int(normalized.get("losses", derived_losses), derived_losses)
    derived_pnl = sum(_safe_float(row.get("pnl_usd", row.get("realized_pnl_usd", 0.0)), 0.0) for row in closed_rows)
    total_pnl = _safe_float(
        normalized.get("total_pnl", normalized.get("realized_pnl_usd", normalized.get("pnl_usd", derived_pnl))),
        derived_pnl,
    )
    if "win_rate" in normalized:
        win_rate = _safe_float(normalized.get("win_rate"), 0.0)
    else:
        win_rate = float(wins) / float(completed_count) if completed_count > 0 else 0.0

    normalized.setdefault("open_positions", open_count)
    normalized.setdefault("open_count", open_count)
    normalized.setdefault("completed_count", completed_count)
    normalized.setdefault("trades", completed_count)
    normalized.setdefault("wins", wins)
    normalized.setdefault("losses", losses)
    normalized.setdefault("total_pnl", total_pnl)
    normalized.setdefault("win_rate", win_rate)
    normalized.setdefault("source", normalized.get("source") or ("trade_history" if closed_rows else "runtime_execution_payload"))
    if recent_decisions:
        normalized.setdefault("recent_decision_count", len(recent_decisions))
    return normalized


def _normalize_paper_execution_runtime_payload(
    *,
    open_positions: Any = None,
    recent_decisions: Any = None,
    recent_messages: Any = None,
    summary: Any = None,
    trade_history: Any = None,
    authority: Any = None,
    runtime_gate: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    positions = _as_list(open_positions)
    decisions = _as_list(recent_decisions)
    messages = _as_list(recent_messages)
    closed_rows = _as_list(trade_history)
    runtime = {
        "state_owner": "live_trading_controller",
        "execution_surface": "paper_forward_execution",
        "authoritative_for_runtime_positions": True,
        "open_positions": positions,
        "open_position_count": len(positions),
        "recent_decisions": decisions,
        "recent_messages": messages,
        "trade_history": closed_rows,
        "summary": _normalize_execution_summary(summary, open_positions=positions, recent_decisions=decisions, trade_history=closed_rows),
        "authority": dict(authority or {}) if isinstance(authority, dict) else {},
        "runtime_gate": dict(runtime_gate or {}) if isinstance(runtime_gate, dict) else {},
        "updated_at": now_utc_ts(),
    }
    runtime.update(dict(extra or {}))
    runtime["summary"] = _normalize_execution_summary(runtime.get("summary"), open_positions=positions, recent_decisions=decisions, trade_history=closed_rows)
    guard_cfg = SymbolExpectancyGuardConfig()
    runtime["symbol_expectancy_guard"] = summarize_symbol_expectancy(closed_rows, cfg=guard_cfg)
    runtime["summary"].setdefault("symbol_expectancy_guard", runtime["symbol_expectancy_guard"])
    runtime["execution_summary"] = dict(runtime["summary"])
    runtime["summary_contract_version"] = "execution_summary.v1"
    return runtime

def _tail_jsonl(path: Path, *, limit: int, max_scan_bytes: int) -> tuple[int | None, list[dict[str, Any]]]:
    if not path.exists():
        return 0, []
    file_size = path.stat().st_size
    scan_from = max(0, file_size - max_scan_bytes)
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, int(limit)))
    line_count: int | None = 0 if scan_from == 0 else None
    try:
        with path.open("rb") as handle:
            if scan_from:
                handle.seek(scan_from)
                handle.readline()
            for raw_line in handle:
                if line_count is not None:
                    line_count += 1
                try:
                    line = raw_line.decode("utf-8").strip()
                except Exception:
                    continue
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return None, []
    parsed_rows = list(rows)
    parsed_rows.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
    return line_count, parsed_rows[: max(1, int(limit))]


class MonitorPublisher:
    _instances: dict[str, "MonitorPublisher"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, paths):
        key = cls._instance_key(paths)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    @classmethod
    def _instance_key(cls, paths) -> str:
        project_root = getattr(paths, "project_root", None)
        if project_root is not None:
            return str(project_root)
        runtime_dir = getattr(paths, "runtime_dir", None)
        if runtime_dir is not None:
            return str(runtime_dir)
        if isinstance(paths, (str, Path)):
            return str(Path(paths).resolve())
        return f"ephemeral:{id(paths)}"

    def __init__(self, paths) -> None:
        if getattr(self, "_initialized", False):
            return
        self.paths = self._normalize_paths(paths)
        self.snapshot_path = Path(
            getattr(
                self.paths,
                "monitoring_state_path",
                getattr(self.paths, "monitor_snapshot_path", self.paths.runtime_dir / "monitoring_state.json"),
            )
        )
        self.events_path = Path(
            getattr(
                self.paths,
                "monitoring_events_path",
                getattr(self.paths, "monitor_events_path", self.paths.runtime_dir / "monitoring_events.json"),
            )
        )
        self._lock = threading.RLock()
        self._snapshot = self._default_snapshot()
        self._events: list[dict[str, Any]] = []
        self._limits = self._load_limits()
        self._last_journal_signature: tuple[int, int] | None = None
        self._load_existing_state()
        self._initialized = True

    def _normalize_paths(self, paths):
        if isinstance(paths, PathManager):
            return paths
        if isinstance(paths, (str, Path)):
            return PathManager.from_project_root(Path(paths).resolve())
        project_root = getattr(paths, "project_root", None)
        runtime_dir = getattr(paths, "runtime_dir", None)
        if runtime_dir is not None:
            return paths
        if project_root is not None:
            return PathManager.from_project_root(Path(project_root).resolve())
        raise TypeError("MonitorPublisher requires a PathManager-like object or project root path")

    def _default_snapshot(self) -> dict[str, Any]:
        return {
            "updated_at": None,
            "system": {
                "launcher_mode": None,
                "project_root": "",
                "python_version": "",
                "pid": None,
                "started_at": None,
                "execution_mode": "paper",
                "allow_unreconciled_live_mode": False,
                "reconciliation_enabled": False,
            },
            "health": {
                "status": "starting",
                "updated_at": None,
            },
            "controllers": {},
            "market_data": {
                "state_owner": "market_data_controller",
                "symbols_total": 0,
                "symbol_counts": {"ACTIVE": 0, "TIMED_OUT": 0, "BLOCKED": 0},
                "funding_refresh": None,
                "open_interest_refresh": None,
                "latest_errors": [],
                "runtime_price_snapshot": {
                    "path": "",
                    "exists": False,
                    "source": "",
                    "reason": "",
                    "expected_symbol_count": 0,
                    "priced_symbol_count": 0,
                    "missing_symbol_count": 0,
                    "stale_symbol_count": 0,
                    "all_symbols_priced": False,
                    "created_at_ms": None,
                    "age_seconds": None,
                    "max_price_age_seconds": None,
                    "missing_symbols": [],
                    "stale_symbols": [],
                    "price_source_counts": {},
                    "updated_at": None,
                },
            },
            "training": {
                "rf": {},
                "gbm": {},
                "dqn": {},
            },
            "training_runtime": {
                "queue_depth": 0,
                "inflight_count": 0,
                "rf_running": 0,
                "gbm_running": 0,
                "dqn_running": 0,
                "rf_pending": 0,
                "gbm_pending": 0,
                "dqn_pending": 0,
                "hmm_pending": 0,
                "current_rf": "--",
                "current_gbm": "--",
                "current_dqn": "--",
                "current_hmm": "--",
                "rf_progress": 0.0,
                "gbm_progress": 0.0,
                "dqn_progress": 0.0,
                "hmm_progress": 0.0,
                "last_rf_message": "--",
                "last_gbm_message": "--",
                "last_dqn_message": "--",
                "last_hmm_message": "--",
                "due_jobs_preview": [],
                "queued_jobs": [],
                "running_jobs": [],
                "succeeded_jobs": [],
                "failed_jobs": [],
                "failure_cache_size": 0,
                "updated_at": None,
            },
            "scheduler": {
                "due_jobs_count": 0,
                "queued_jobs_count": 0,
                "due_jobs": [],
                "queued_jobs": [],
                "updated_at": None,
            },
            "training_registry": {
                "counts": {},
                "counts_by_model": {},
                "diagnostic_counts": {},
                "diagnostic_counts_by_model": {},
                "entries": [],
            },
            "training_results": [],
            "backtest_runtime": {
                "state_owner": "backtest_controller",
                "execution_surface": "historical_backtest",
                "authoritative_for_runtime_positions": False,
                "open_positions": [],
                "recent_closed": [],
                "trade_history": [],
                "recent_messages": [],
                "summary": {},
                "updated_at": None,
            },
            "paper_execution_runtime": {
                "state_owner": "live_trading_controller",
                "execution_surface": "paper_forward_execution",
                "authoritative_for_runtime_positions": True,
                "open_positions": [],
                "open_position_count": 0,
                "recent_decisions": [],
                "recent_messages": [],
                "summary": {
                    "open_positions": 0,
                    "open_count": 0,
                    "completed_count": 0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                },
                "authority": {},
                "runtime_gate": {},
                "updated_at": None,
            },
            "paper_execution": {
                "state_owner": "live_trading_controller",
                "execution_surface": "paper_forward_execution",
                "authoritative_for_runtime_positions": True,
                "open_positions": [],
                "open_position_count": 0,
                "recent_decisions": [],
                "recent_messages": [],
                "summary": {
                    "open_positions": 0,
                    "open_count": 0,
                    "completed_count": 0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                },
                "authority": {},
                "runtime_gate": {},
                "updated_at": None,
            },
            "execution_runtime": {
                "state_owner": "live_trading_controller",
                "execution_surface": "paper_forward_execution",
                "authoritative_for_runtime_positions": True,
                "open_positions": [],
                "open_position_count": 0,
                "recent_decisions": [],
                "recent_messages": [],
                "summary": {
                    "open_positions": 0,
                    "open_count": 0,
                    "completed_count": 0,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                },
                "authority": {},
                "runtime_gate": {},
                "updated_at": None,
            },
            "inference": {
                "state_owner": "live_trading_controller",
                "decisions": [],
                "open_positions": {},
                "pnl_summary": {},
                "last_error": None,
                "risk_summary": {},
                "updated_at": None,
            },
            "signal_runtime": {
                "state_owner": "signal_generator_controller",
                "latest_mode": None,
                "latest_symbol": None,
                "latest_action": None,
                "latest_reason": None,
                "latest_ts": None,
                "recent_signals": [],
                "recent_decisions": [],
                "runtime_gate": {},
                "last_error": None,
                "source_priority": 0,
                "source_name": "",
            },
            "control_plane": {
                "state_owner": "control_plane_store",
                "paused": False,
                "pause_reason": "",
                "updated_at": None,
                "updated_by": "system",
                "circuit_breaker": {},
                "symbol_summary": {"counts": {"ACTIVE": 0, "TIMED_OUT": 0, "BLOCKED": 0}, "rows": []},
                "recent_actions": [],
                "decision_lineage_preview": [],
                "shadow_candidate_preview": [],
                "runtime_observability": {
                    "freshness": {"recent_entries": 0, "fresh_entries": 0, "stale_entries": 0, "max_age_seconds": None, "avg_age_seconds": None, "latest_reason": ""},
                    "prediction_drift": {"recent_action_counts": {}, "baseline_action_counts": {}, "recent_non_hold_rate": 0.0, "baseline_non_hold_rate": 0.0, "avg_score_recent": None, "shadow_available": 0, "shadow_divergence_rate": 0.0},
                    "model_state": {"active_model_count": 0, "candidate_model_count": 0, "latest_lineage": {}},
                    "runtime_performance": {"recent_expected_direction_balance": 0, "portfolio_runtime": {}, "live_runtime": {}},
                },
                "live_runtime": {
                    "execution_mode": "paper",
                    "reconciliation_enabled": False,
                    "reconciliation_required": True,
                    "authority_mode": "paper_local",
                    "reconciliation_status": "paper_local",
                    "last_reconciled_at": None,
                    "exchange_position_count": 0,
                    "reconciliation_mismatch_count": 0,
                    "last_reconciliation_error": "",
                    "exchange_snapshot_hash": "",
                },
                "runtime_gate": {
                    "blocked": False,
                    "reason": "",
                    "status": "running",
                    "trading_enabled": True,
                    "paused": False,
                    "pause_reason": "",
                    "circuit_triggered": False,
                    "circuit_reason": "",
                    "live_authoritative": True,
                    "live_authority_reason": "paper_local_authority",
                },
            },
            "model_versions": {},
            "decision_journal": {
                "entries": 0,
                "latest_ts": None,
                "latest_writer": "",
                "latest_stage": "",
                "writers": {},
                "stages": {},
                "recent": [],
            },
            "rejection_visibility": {
                "total_recent": 0,
                "by_status": {},
                "by_stage": {},
                "by_reason": {},
                "by_category": {},
                "recent": [],
                "updated_at": None,
            },
            "errors": [],
            "runtime_event_counters": {
                "warnings_since_start": 0,
                "errors_since_start": 0,
            },
            "error_state": {
                "active_controller_errors": {},
                "recovered_controller_errors": [],
            },
            "decision_trace": [],
        }

    def _load_limits(self) -> dict[str, int]:
        cfg = getattr(getattr(self.paths, "config", None), "monitoring", None)
        return {
            "events": _bounded_int(getattr(cfg, "max_events", None), _DEFAULT_LIMITS["events"], maximum=2000),
            "alerts": _bounded_int(getattr(cfg, "max_alerts", None), _DEFAULT_LIMITS["alerts"], maximum=1000),
            "market_updates": _bounded_int(getattr(cfg, "max_market_updates", None), _DEFAULT_LIMITS["market_updates"], maximum=1000),
            "recent_training_results": _bounded_int(getattr(cfg, "max_recent_training_results", None), _DEFAULT_LIMITS["recent_training_results"], maximum=1000),
            "recent_decisions": _bounded_int(getattr(cfg, "max_recent_decisions", None), _DEFAULT_LIMITS["recent_decisions"], maximum=500),
            "recent_signals": _bounded_int(getattr(cfg, "max_recent_signals", None), _DEFAULT_LIMITS["recent_signals"], maximum=1000),
            "backtest_open_positions": _bounded_int(getattr(cfg, "max_backtest_open_positions", None), _DEFAULT_LIMITS["backtest_open_positions"], maximum=1000),
            "backtest_recent_closed": _bounded_int(getattr(cfg, "max_backtest_recent_closed", None), _DEFAULT_LIMITS["backtest_recent_closed"], maximum=1000),
            "backtest_trade_history": _DEFAULT_LIMITS["backtest_trade_history"],
            "registry_entries": _bounded_int(getattr(cfg, "max_registry_entries", None), _DEFAULT_LIMITS["registry_entries"], maximum=2000),
            "scheduler_jobs": _DEFAULT_LIMITS["scheduler_jobs"],
            "training_jobs": _DEFAULT_LIMITS["training_jobs"],
            "control_actions": _DEFAULT_LIMITS["control_actions"],
            "decision_trace": _DEFAULT_LIMITS["decision_trace"],
            "decision_journal_recent": _DEFAULT_LIMITS["decision_journal_recent"],
            "decision_journal_scan_bytes": _DEFAULT_LIMITS["decision_journal_scan_bytes"],
        }

    def _limit_list(self, rows: Any, key: str) -> list[Any]:
        limit = self._limits.get(key, _DEFAULT_LIMITS.get(key, 100))
        return list(rows or [])[-limit:]

    def _compact_snapshot_locked(self) -> None:
        s = self._snapshot
        s["errors"] = self._limit_list(s.get("errors", []), "alerts")
        s["warnings"] = self._limit_list(s.get("warnings", []), "alerts")

        market = s.get("market_data")
        if isinstance(market, dict):
            market["recent_results"] = self._limit_list(market.get("recent_results", []), "market_updates")
            market["latest_updates"] = self._limit_list(market.get("latest_updates", []), "market_updates")
            market["latest_errors"] = self._limit_list(market.get("latest_errors", []), "alerts")
            price_diag = market.get("runtime_price_snapshot")
            if isinstance(price_diag, dict):
                price_diag["missing_symbols"] = self._limit_list(price_diag.get("missing_symbols", []), "recent_signals")
                price_diag["stale_symbols"] = self._limit_list(price_diag.get("stale_symbols", []), "recent_signals")

        training_runtime = s.get("training_runtime")
        if isinstance(training_runtime, dict):
            for field in ("due_jobs_preview", "queued_jobs", "running_jobs", "succeeded_jobs", "failed_jobs"):
                training_runtime[field] = self._limit_list(training_runtime.get(field, []), "training_jobs")

        scheduler = s.get("scheduler")
        if isinstance(scheduler, dict):
            scheduler["due_jobs"] = self._limit_list(scheduler.get("due_jobs", []), "scheduler_jobs")
            scheduler["queued_jobs"] = self._limit_list(scheduler.get("queued_jobs", []), "scheduler_jobs")

        registry = s.get("training_registry")
        if isinstance(registry, dict):
            registry["entries"] = self._limit_list(registry.get("entries", []), "registry_entries")
            registry["recent_results"] = self._limit_list(registry.get("recent_results", []), "recent_training_results")

        s["training_results"] = self._limit_list(s.get("training_results", []), "recent_training_results")

        rejection_visibility = s.get("rejection_visibility")
        if isinstance(rejection_visibility, dict):
            rejection_visibility["recent"] = self._limit_list(rejection_visibility.get("recent", []), "decision_trace")
        control_plane = s.get("control_plane")
        if isinstance(control_plane, dict):
            observability = control_plane.get("runtime_observability")
            if isinstance(observability, dict):
                visibility = observability.get("rejection_visibility")
                if isinstance(visibility, dict):
                    visibility["recent"] = self._limit_list(visibility.get("recent", []), "decision_trace")

        backtest = s.get("backtest_runtime")
        if isinstance(backtest, dict):
            backtest["open_positions"] = self._limit_list(backtest.get("open_positions", []), "backtest_open_positions")
            backtest["recent_closed"] = self._limit_list(backtest.get("recent_closed", []), "backtest_recent_closed")
            backtest["trade_history"] = self._limit_list(backtest.get("trade_history", []), "backtest_trade_history")
            backtest["recent_messages"] = self._limit_list(backtest.get("recent_messages", []), "recent_signals")

        for execution_key in ("paper_execution_runtime", "paper_execution", "execution_runtime"):
            execution_runtime = s.get(execution_key)
            if isinstance(execution_runtime, dict):
                execution_runtime["open_positions"] = self._limit_list(execution_runtime.get("open_positions", []), "backtest_open_positions")
                execution_runtime["open_position_count"] = len(execution_runtime.get("open_positions") or [])
                execution_runtime["recent_decisions"] = self._limit_list(execution_runtime.get("recent_decisions", []), "recent_decisions")
                execution_runtime["recent_messages"] = self._limit_list(execution_runtime.get("recent_messages", []), "recent_decisions")
                execution_runtime["summary"] = _normalize_execution_summary(
                    execution_runtime.get("summary"),
                    open_positions=list(execution_runtime.get("open_positions") or []),
                    recent_decisions=list(execution_runtime.get("recent_decisions") or []),
                    trade_history=list(execution_runtime.get("trade_history") or []),
                )

        inference = s.get("inference")
        if isinstance(inference, dict):
            inference["decisions"] = self._limit_list(inference.get("decisions", []), "recent_decisions")

        signal = s.get("signal_runtime")
        if isinstance(signal, dict):
            signal["recent_signals"] = self._limit_list(signal.get("recent_signals", []), "recent_signals")
            signal["recent_decisions"] = self._limit_list(signal.get("recent_decisions", []), "recent_decisions")

        control = s.get("control_plane")
        if isinstance(control, dict):
            control["recent_actions"] = self._limit_list(control.get("recent_actions", []), "control_actions")
            control["decision_lineage_preview"] = self._limit_list(control.get("decision_lineage_preview", []), "recent_decisions")
            control["shadow_candidate_preview"] = self._limit_list(control.get("shadow_candidate_preview", []), "recent_decisions")
            # Keep the UI/root snapshot aligned with the control-plane source of truth.
            # The frontend and API consumers historically checked both locations.
            s["decision_lineage_preview"] = copy.deepcopy(control.get("decision_lineage_preview") or [])
            summary = control.get("symbol_summary")
            if isinstance(summary, dict):
                summary["rows"] = self._limit_list(summary.get("rows", []), "registry_entries")

        journal = s.get("decision_journal")
        if isinstance(journal, dict):
            journal["recent"] = [
                _compact_row(row, keys=_COMPACT_DECISION_KEYS)
                for row in self._limit_list(journal.get("recent", []), "decision_journal_recent")
            ]

        s["decision_trace"] = self._limit_list(s.get("decision_trace", []), "decision_trace")

    @contextmanager
    def _serialized_mutation(self):
        """Serialise cross-process snapshot mutations and merge current persisted truth.

        Runtime, trainer and dashboard roles intentionally run in separate processes.
        Each role therefore owns an independent in-memory ``MonitorPublisher`` snapshot
        while sharing one persisted monitoring-state file.  A process must reload the
        most recent persisted snapshot *after* obtaining the inter-process lock and
        before applying its local section mutation, otherwise its next full-state
        write can erase updates published by another role and make the UI alternate
        between stale/empty and populated panels.
        """
        with self._lock:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.snapshot_path.with_name(f".{self.snapshot_path.name}.lock")
            with lock_path.open("a+b") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    self._load_existing_state()
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _load_existing_state(self) -> None:
        try:
            if self.snapshot_path.exists():
                payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    merged = self._default_snapshot()
                    merged.update(payload)
                    self._snapshot = merged
                    self._compact_snapshot_locked()
        except Exception:
            pass

        try:
            if self.events_path.exists():
                payload = json.loads(self.events_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    self._events = [dict(item) for item in payload if isinstance(item, dict)][-self._limits["events"]:]
        except Exception:
            pass

        try:
            self._refresh_decision_journal_summary_locked()
        except Exception:
            pass

    def _persist_locked(self) -> None:
        # Persist derived operational truth alongside raw controller payloads.
        # Without this projection, the last writer could leave monitoring_state.json
        # showing controller liveness as healthy while trading was blocked.
        self._restore_empty_registry_projection_locked()
        self._refresh_symbol_universe_locked()
        self._refresh_market_price_diagnostics_locked()
        self._refresh_rejection_visibility_locked()
        self._refresh_trading_readiness_locked()
        self._compact_snapshot_locked()
        self._snapshot["updated_at"] = now_utc_ts()
        self._snapshot["health"] = self._compute_health_locked()
        self._snapshot["health"]["updated_at"] = self._snapshot["updated_at"]

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

        atomic_write_bytes(
            self.snapshot_path,
            json.dumps(self._snapshot, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        )
        atomic_write_bytes(
            self.events_path,
            json.dumps(self._events[-self._limits["events"]:], separators=(",", ":"), sort_keys=True).encode("utf-8"),
        )

    def _compute_health_locked(self) -> dict[str, Any]:
        """Compute operator health without confusing liveness with readiness."""
        controllers = self._snapshot.get("controllers", {})
        statuses = [
            str((payload or {}).get("status") or "").lower()
            for payload in controllers.values()
            if isinstance(payload, dict)
        ]
        if any(status == "error" for status in statuses):
            controller_status = "error"
        elif any(status in {"degraded", "blocked", "guarded"} for status in statuses):
            controller_status = "degraded"
        elif any(status in {"running", "healthy", "mark_to_market"} for status in statuses):
            controller_status = "ok"
        else:
            controller_status = "starting"

        health = dict(self._snapshot.get("health") or {})
        readiness = self._snapshot.get("trading_readiness") if isinstance(self._snapshot.get("trading_readiness"), dict) else {}
        integrity = readiness.get("execution_integrity") if isinstance(readiness.get("execution_integrity"), dict) else self._execution_integrity_state_locked()
        trading_status = str(readiness.get("status") or "unknown").lower()
        trading_ready = bool(readiness.get("ready")) if readiness else None

        if controller_status == "error":
            status = "error"
        elif bool(integrity.get("blocked")):
            status = "blocked"
        elif readiness and not trading_ready:
            status = "degraded"
        else:
            status = controller_status

        health["status"] = status
        health["controller_status"] = controller_status
        health["trading_readiness_status"] = trading_status
        health["trading_ready"] = trading_ready
        health["readiness_blockers"] = list(readiness.get("blockers") or [])[:25]
        if readiness and not trading_ready:
            health["warnings"] = max(int(health.get("warnings") or 0), len(health["readiness_blockers"]), 1)
        if bool(integrity.get("blocked")):
            health["errors"] = max(int(health.get("errors") or 0), 1)
            health["execution_integrity"] = dict(integrity)
        else:
            health.setdefault("errors", int(health.get("errors") or 0))
        return health

    def _append_event_locked(
        self,
        *,
        level: str,
        component: str,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
        alert: bool = False,
    ) -> None:
        normalized_level = str(level).upper()
        counters = self._snapshot.setdefault("runtime_event_counters", {})
        counters.setdefault("warnings_since_start", 0)
        counters.setdefault("errors_since_start", 0)
        if normalized_level == "WARNING":
            counters["warnings_since_start"] = int(counters.get("warnings_since_start") or 0) + 1
        elif normalized_level == "ERROR":
            counters["errors_since_start"] = int(counters.get("errors_since_start") or 0) + 1
        self._events.append(
            {
                "ts": now_utc_ts(),
                "level": normalized_level,
                "component": str(component),
                "category": str(category),
                "message": str(message),
                "data": dict(data or {}),
                "alert": bool(alert),
            }
        )
        self._events = self._events[-self._limits["events"]:]

    def _refresh_decision_journal_summary_locked(self, limit: int | None = None) -> None:
        path = getattr(self.paths, "decision_journal_path", None)
        if path is None:
            return
        journal_path = Path(path)
        if not journal_path.exists():
            self._snapshot["decision_journal"] = {
                "entries": 0,
                "entries_estimated": False,
                "latest_ts": None,
                "latest_writer": "",
                "latest_stage": "",
                "writers": {},
                "stages": {},
                "recent": [],
            }
            self._last_journal_signature = None
            return

        try:
            stat = journal_path.stat()
            signature = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return
        if signature == self._last_journal_signature:
            return

        bounded_limit = max(1, int(limit or self._limits["decision_journal_recent"]))
        total, recent = _tail_jsonl(
            journal_path,
            limit=bounded_limit,
            max_scan_bytes=self._limits["decision_journal_scan_bytes"],
        )
        current = dict(self._snapshot.get("decision_journal") or {})
        previous_entries = int(current.get("entries") or 0)
        entries = total if total is not None else max(previous_entries, len(recent))
        latest_ts = recent[0].get("ts") if recent else None
        writers: dict[str, int] = {}
        stages: dict[str, int] = {}
        execution_statuses: dict[str, int] = {}
        execution_blockers: dict[str, int] = {}
        execution_blocker_stages: dict[str, int] = {}
        compact_recent: list[dict[str, Any]] = []
        for row in recent:
            writer = str(row.get("journal_writer") or "")
            stage = str(row.get("decision_stage") or "")
            execution_status = str(row.get("execution_status") or "")
            blocker_stage = str(row.get("execution_blocker_stage") or "")
            blocker = str(row.get("rejection_reason") or row.get("execution_blocker") or "")
            if writer:
                writers[writer] = writers.get(writer, 0) + 1
            if stage:
                stages[stage] = stages.get(stage, 0) + 1
            if execution_status:
                execution_statuses[execution_status] = execution_statuses.get(execution_status, 0) + 1
            if blocker_stage:
                execution_blocker_stages[blocker_stage] = execution_blocker_stages.get(blocker_stage, 0) + 1
            if blocker:
                execution_blockers[blocker] = execution_blockers.get(blocker, 0) + 1
            compact_recent.append(_compact_row(row, keys=_COMPACT_DECISION_KEYS))
        self._snapshot["decision_journal"] = {
            "entries": int(entries),
            "entries_estimated": total is None,
            "latest_ts": latest_ts,
            "latest_writer": str(recent[0].get("journal_writer") or "") if recent else "",
            "latest_stage": str(recent[0].get("decision_stage") or "") if recent else "",
            "writers": writers,
            "stages": stages,
            "execution_statuses": execution_statuses,
            "execution_blocker_stages": execution_blocker_stages,
            "execution_blockers": execution_blockers,
            "recent": compact_recent,
        }
        self._last_journal_signature = signature


    def _market_latest_snapshot_path(self) -> Path | None:
        candidate = getattr(self.paths, "market_latest_snapshot_path", None)
        if candidate is not None:
            return Path(candidate)
        runtime_dir = getattr(self.paths, "runtime_dir", None)
        if runtime_dir is None:
            return None
        return Path(runtime_dir) / "market" / "latest.json"

    def _registry_state_path(self) -> Path | None:
        candidate = getattr(self.paths, "registry_path", None)
        if candidate is not None:
            return Path(candidate)
        runtime_dir = getattr(self.paths, "runtime_dir", None)
        if runtime_dir is None:
            return None
        return Path(runtime_dir) / "training_registry.json"

    @staticmethod
    def _registry_fallback_category(row: dict[str, Any]) -> str:
        status = str(row.get("status") or "idle").lower()
        truth = str(row.get("registry_truth_status") or "").lower()
        if status in {"queued", "running"}:
            return status
        if bool(row.get("execution_ready")):
            return "promoted"
        if truth in {
            "candidate_only",
            "missing_active_manifest",
            "active_manifest_invalid",
            "active_without_registry_approval",
            "feature_incompatible",
        }:
            return truth
        if bool(row.get("latest_candidate_paper_bootstrap_only")):
            return "candidate_only"
        if status == "failed":
            return "failed"
        return "missing_active_manifest" if status == "idle" else "unknown"

    def _restore_empty_registry_projection_locked(self) -> None:
        """Restore an empty dashboard registry from persisted canonical truth.

        TrainingRegistry remains the sole authority that evaluates manifests and
        writes registry truth.  This read-through is intentionally limited to a
        missing/empty monitoring projection so a standalone dashboard process
        cannot replace a populated canonical registry with zero counts.
        """
        current = self._snapshot.get("training_registry") if isinstance(self._snapshot.get("training_registry"), dict) else {}
        if list(current.get("entries") or []):
            return
        path = self._registry_state_path()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict) or not raw:
            return

        entries: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        counts_by_model: dict[str, dict[str, int]] = {}
        diagnostic_counts: dict[str, int] = {}
        diagnostic_counts_by_model: dict[str, dict[str, int]] = {}
        for key, value in sorted(raw.items()):
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("registry_key", str(key))
            model_type = str(row.get("model_type") or "unknown").lower()
            status = str(row.get("status") or "idle").lower()
            category = self._registry_fallback_category(row)
            row.setdefault("diagnostic_category", category)
            row.setdefault("diagnostic_detail", str(row.get("registry_truth_status") or category))
            entries.append(row)
            counts[status] = counts.get(status, 0) + 1
            by_model = counts_by_model.setdefault(model_type, {})
            by_model[status] = by_model.get(status, 0) + 1
            diagnostic_counts[category] = diagnostic_counts.get(category, 0) + 1
            diag_by_model = diagnostic_counts_by_model.setdefault(model_type, {})
            diag_by_model[category] = diag_by_model.get(category, 0) + 1
        if not entries:
            return
        payload = self._snapshot.setdefault("training_registry", {})
        payload.update({
            "counts": counts,
            "counts_by_model": counts_by_model,
            "diagnostic_counts": diagnostic_counts,
            "diagnostic_counts_by_model": diagnostic_counts_by_model,
            "entries": self._limit_list(entries, "registry_entries"),
            "projection_source": "canonical_registry_fallback",
            "updated_at": now_utc_ts(),
        })

    def _refresh_market_price_diagnostics_locked(self) -> None:
        """Refresh read-only diagnostics for data/runtime/market/latest.json.

        This does not mutate market data or controller state. It only projects the
        latest price-cache contract into the monitoring snapshot so the UI can
        show whether execution blockers are caused by a missing, stale, or
        partially priced runtime price cache.
        """
        path = self._market_latest_snapshot_path()
        now_ms = int(now_utc_ts() * 1000)
        diagnostics: dict[str, Any] = {
            "path": str(path or ""),
            "exists": bool(path and path.exists()),
            "source": "",
            "reason": "",
            "expected_symbol_count": 0,
            "priced_symbol_count": 0,
            "missing_symbol_count": 0,
            "stale_symbol_count": 0,
            "all_symbols_priced": False,
            "created_at_ms": None,
            "age_seconds": None,
            "max_price_age_seconds": None,
            "missing_symbols": [],
            "stale_symbols": [],
            "price_source_counts": {},
            "websocket_cache_status": "",
            "websocket_partial_snapshot_events": 0,
            "websocket_http_fallback_events": 0,
            "websocket_retained_snapshot_events": 0,
            "ready": False,
            "updated_at": now_utc_ts(),
        }
        if path is None or not path.exists():
            self._snapshot.setdefault("market_data", {})["runtime_price_snapshot"] = diagnostics
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics["load_error"] = str(exc)
            self._snapshot.setdefault("market_data", {})["runtime_price_snapshot"] = diagnostics
            return
        if not isinstance(payload, dict):
            diagnostics["load_error"] = "snapshot_not_dict"
            self._snapshot.setdefault("market_data", {})["runtime_price_snapshot"] = diagnostics
            return

        symbols = payload.get("symbols") if isinstance(payload.get("symbols"), dict) else {}
        price_symbols = [str(symbol).upper() for symbol in list(payload.get("price_symbols") or []) if symbol]
        active_symbols = [str(symbol).upper() for symbol in list(payload.get("active_symbols") or []) if symbol]
        if not price_symbols:
            price_symbols = list(active_symbols)
        if not price_symbols:
            price_symbols = [str(symbol).upper() for symbol in symbols.keys()]
        expected = int(payload.get("expected_symbol_count") or len(price_symbols) or len(symbols))
        missing_symbols = [str(symbol).upper() for symbol in list(payload.get("missing_symbols") or []) if symbol]
        priced_symbols: list[str] = []
        stale_symbols: list[str] = []
        price_source_counts: dict[str, int] = {}
        max_price_age_seconds: float | None = None
        runtime_cfg = getattr(getattr(getattr(self, "paths", None), "config", None), "runtime", None)
        try:
            stale_after_seconds = float(getattr(runtime_cfg, "price_max_age_seconds", 180.0) or 180.0)
        except Exception:
            stale_after_seconds = 180.0

        for symbol in price_symbols:
            row = symbols.get(symbol) if isinstance(symbols, dict) else None
            if not isinstance(row, dict):
                if symbol not in missing_symbols:
                    missing_symbols.append(symbol)
                continue
            price = (
                row.get("mark_price")
                or row.get("mark_price_average")
                or row.get("primary_price")
                or row.get("latest_price")
                or row.get("price")
                or row.get("reference_price")
                or row.get("ticker_price")
                or row.get("index_price")
                or row.get("db_close")
            )
            try:
                has_price = float(price) > 0
            except Exception:
                has_price = False
            if not has_price:
                if symbol not in missing_symbols:
                    missing_symbols.append(symbol)
                continue
            priced_symbols.append(symbol)
            source = str(row.get("price_source") or row.get("source") or "unknown")
            price_source_counts[source] = price_source_counts.get(source, 0) + 1
            ts = row.get("price_ts_ms") or row.get("event_time_ms") or row.get("timestamp")
            try:
                ts_ms = int(ts) if ts is not None else None
            except Exception:
                ts_ms = None
            if ts_ms:
                age_seconds = max(0.0, (now_ms - ts_ms) / 1000.0)
                max_price_age_seconds = age_seconds if max_price_age_seconds is None else max(max_price_age_seconds, age_seconds)
                if age_seconds > stale_after_seconds:
                    stale_symbols.append(symbol)

        created_at_ms = payload.get("created_at_ms") or payload.get("created_ms")
        try:
            created_at_ms = int(created_at_ms) if created_at_ms is not None else None
        except Exception:
            created_at_ms = None
        diagnostics.update({
            "source": str(payload.get("source") or ""),
            "reason": str(payload.get("reason") or ""),
            "expected_symbol_count": expected,
            "priced_symbol_count": int(payload.get("priced_symbol_count") or len(priced_symbols)),
            "missing_symbol_count": len(missing_symbols),
            "stale_symbol_count": len(stale_symbols),
            "all_symbols_priced": bool(payload.get("all_symbols_priced")) if "all_symbols_priced" in payload else len(missing_symbols) == 0 and expected == len(priced_symbols),
            "created_at_ms": created_at_ms,
            "age_seconds": max(0.0, (now_ms - created_at_ms) / 1000.0) if created_at_ms else None,
            "max_price_age_seconds": max_price_age_seconds,
            "missing_symbols": missing_symbols[:100],
            "stale_symbols": stale_symbols[:100],
            "price_source_counts": price_source_counts,
            "websocket_cache_status": str(payload.get("websocket_cache_status") or ""),
            "websocket_partial_snapshot_events": int(payload.get("websocket_partial_snapshot_events") or 0),
            "websocket_http_fallback_events": int(payload.get("websocket_http_fallback_events") or 0),
            "websocket_retained_snapshot_events": int(payload.get("websocket_retained_snapshot_events") or 0),
            "ready": bool(
                expected > 0
                and len(missing_symbols) == 0
                and len(stale_symbols) == 0
                and len(priced_symbols) >= expected
                and (bool(payload.get("all_symbols_priced")) if "all_symbols_priced" in payload else True)
            ),
        })
        market_data = self._snapshot.setdefault("market_data", {})
        market_data["runtime_price_snapshot"] = diagnostics
        market_data["price_cache_status"] = "ready" if diagnostics.get("ready") else "degraded"
        market_data["priced_symbol_count"] = diagnostics["priced_symbol_count"]
        market_data["missing_price_symbols"] = diagnostics["missing_symbols"]
        market_data["stale_price_symbols"] = diagnostics["stale_symbols"]


    def _refresh_symbol_universe_locked(self) -> None:
        """Project symbol lifecycle buckets into the monitoring snapshot.

        The market controller may publish a raw universe and the training/registry
        views can contain historical symbols.  The dashboard must have one clear
        active-universe contract so signal/training tables do not mix active,
        candidate, blocked, and historical symbols.
        """
        try:
            state = load_symbol_state(self.paths)
        except Exception:
            state = {}

        signal_runtime = self._snapshot.get("signal_runtime") if isinstance(self._snapshot.get("signal_runtime"), dict) else {}
        signal_readiness = signal_runtime.get("readiness") if isinstance(signal_runtime.get("readiness"), dict) else {}
        if signal_readiness:
            try:
                state, _ = reconcile_symbol_lifecycle_with_readiness_snapshot(
                    state,
                    signal_readiness,
                    now_ts=now_utc_ts(),
                )
                ranked = normalize_symbol_list(state.get("ranked_candidates") or [])
                active_for_buckets = normalize_symbol_list(state.get("active_symbols") or [])
                from core.symbol_management import classify_symbol_universe  # local import avoids widening module import surface

                state.update(classify_symbol_universe(
                    ranked_candidates=ranked,
                    active_symbols=active_for_buckets,
                    state=state,
                    ranked_universe_size=int(state.get("ranked_universe_size") or len(ranked) or 20),
                ))
            except Exception:
                pass

        active = normalize_symbol_list(state.get("active_symbols") or [])
        candidates = normalize_symbol_list(state.get("candidate_symbols") or [])
        blocked = normalize_symbol_list(state.get("blocked_symbols") or [])
        historical = normalize_symbol_list(state.get("historical_symbols") or [])
        execution_eligible = normalize_symbol_list(state.get("execution_eligible_symbols") or [])
        symbols_state = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
        warming: list[str] = []
        training_required: list[str] = []
        for symbol, row in symbols_state.items():
            if not isinstance(row, dict):
                continue
            stage = str(row.get("activation_stage") or "").lower()
            normalized_symbol = str(symbol or "").upper()
            if stage == "warming_data":
                warming.append(normalized_symbol)
            elif stage == "training_required":
                training_required.append(normalized_symbol)

        target = state.get("target_active_symbols")
        if target is None:
            try:
                target = int(getattr(getattr(self.paths, "config", None), "symbols", None).target_active_symbols)
            except Exception:
                target = len(active)

        ranked = normalize_symbol_list(state.get("ranked_candidates") or [])
        market_data = self._snapshot.setdefault("market_data", {})
        market_data["symbol_universe"] = {
            "active_symbols": active,
            "candidate_symbols": candidates,
            "blocked_symbols": blocked,
            "historical_symbols": historical,
            "warming_symbols": sorted(set(warming)),
            "training_required_symbols": sorted(set(training_required)),
            "execution_eligible_symbols": execution_eligible,
            "ranked_candidates": ranked,
            "ranked_universe_size": int(state.get("ranked_universe_size") or len(ranked) or 0),
            "target_active_symbols": int(target or 0),
            "active_count": len(active),
            "execution_eligible_count": len(execution_eligible),
            "source": state.get("active_symbols_source") or "symbol_state",
            "updated_at": state.get("updated_at") or now_utc_ts(),
        }
        symbol_state_path = getattr(self.paths, "symbol_state_path", None)
        if symbol_state_path is not None and Path(symbol_state_path).exists():
            lifecycle_counts = dict(market_data.get("symbol_counts") or {})
            lifecycle_counts.update({"ACTIVE": len(active), "BLOCKED": len(blocked)})
            market_data["symbol_counts"] = lifecycle_counts
        if active:
            existing_universe = market_data.get("universe") if isinstance(market_data.get("universe"), dict) else {}
            existing_source = str(existing_universe.get("source") or "") if isinstance(existing_universe, dict) else ""
            if not existing_universe or existing_source == "symbol_state":
                market_data["universe"] = {
                    "symbols": active,
                    "source": market_data["symbol_universe"]["source"],
                    "last_refresh_ts": market_data["symbol_universe"]["updated_at"],
                }

    def _execution_integrity_state_locked(self) -> dict[str, Any]:
        """Return the fail-closed execution-integrity state exposed by Step 1."""
        candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for surface_name in ("execution_runtime", "paper_execution_runtime", "paper_execution"):
            surface = self._snapshot.get(surface_name)
            if isinstance(surface, dict):
                summary = surface.get("summary") if isinstance(surface.get("summary"), dict) else {}
                gate = surface.get("runtime_gate") if isinstance(surface.get("runtime_gate"), dict) else {}
                candidates.append((surface_name, summary, gate))
        inference = self._snapshot.get("inference")
        if isinstance(inference, dict):
            summary = inference.get("pnl_summary") if isinstance(inference.get("pnl_summary"), dict) else {}
            gate = summary.get("runtime_gate") if isinstance(summary.get("runtime_gate"), dict) else {}
            candidates.append(("inference.pnl_summary", summary, gate))
        control_plane = self._snapshot.get("control_plane")
        if isinstance(control_plane, dict):
            gate = control_plane.get("runtime_gate") if isinstance(control_plane.get("runtime_gate"), dict) else {}
            candidates.append(("control_plane.runtime_gate", {}, gate))

        for source, summary, gate in candidates:
            status = str(_first_non_empty(
                gate.get("execution_integrity_status"),
                summary.get("execution_integrity_status"),
                "ok",
            ) or "ok").lower()
            reason = str(_first_non_empty(
                gate.get("execution_integrity_error"),
                summary.get("execution_integrity_error"),
                gate.get("reason") if bool(gate.get("blocked")) else None,
                "",
            ) or "")
            blocked = status != "ok" or "execution_state_integrity_failure" in reason
            if blocked:
                return {
                    "status": status if status != "ok" else "blocked",
                    "blocked": True,
                    "reason": reason or "execution_state_integrity_failure",
                    "source": source,
                    "failure_at": _first_non_empty(gate.get("execution_integrity_failure_at"), summary.get("execution_integrity_failure_at")),
                }
        return {"status": "ok", "blocked": False, "reason": "", "source": None, "failure_at": None}

    def _refresh_trading_readiness_locked(self) -> None:
        """Derive business readiness separately from process liveness."""
        market_data = self._snapshot.get("market_data") if isinstance(self._snapshot.get("market_data"), dict) else {}
        universe = market_data.get("symbol_universe") if isinstance(market_data.get("symbol_universe"), dict) else {}
        price_diag = market_data.get("runtime_price_snapshot") if isinstance(market_data.get("runtime_price_snapshot"), dict) else {}
        registry = self._snapshot.get("training_registry") if isinstance(self._snapshot.get("training_registry"), dict) else {}
        rejection_visibility = self._snapshot.get("rejection_visibility") if isinstance(self._snapshot.get("rejection_visibility"), dict) else {}
        control_plane = self._snapshot.get("control_plane") if isinstance(self._snapshot.get("control_plane"), dict) else {}
        signal_runtime = self._snapshot.get("signal_runtime") if isinstance(self._snapshot.get("signal_runtime"), dict) else {}
        signal_readiness = signal_runtime.get("readiness") if isinstance(signal_runtime.get("readiness"), dict) else {}

        blockers: list[str] = []
        execution_integrity = self._execution_integrity_state_locked()
        if bool(execution_integrity.get("blocked")):
            blockers.append(f"execution_integrity_blocked:{execution_integrity.get('reason') or 'execution_state_integrity_failure'}")
        price_ready = bool(price_diag.get("ready") or (price_diag.get("all_symbols_priced") and not price_diag.get("stale_symbol_count") and not price_diag.get("missing_symbol_count")))
        if not price_ready:
            blockers.append("price_cache_not_ready")

        active_count = int(universe.get("active_count") or 0)
        target = int(universe.get("target_active_symbols") or active_count or 0)
        if target and active_count != target:
            blockers.append(f"active_universe_count_mismatch:{active_count}/{target}")

        blocked_symbols = list(universe.get("blocked_symbols") or [])
        if blocked_symbols:
            blockers.append(f"blocked_symbols:{len(blocked_symbols)}")

        exec_eligible_count = int(universe.get("execution_eligible_count") or 0)
        if signal_readiness:
            try:
                exec_eligible_count = int(signal_readiness.get("execution_eligible_count", signal_readiness.get("ready_count", exec_eligible_count)) or 0)
            except Exception:
                pass
        if active_count and exec_eligible_count < active_count:
            blockers.append(f"execution_eligible_symbols:{exec_eligible_count}/{active_count}")
        if signal_readiness:
            blocked_count = int(signal_readiness.get("blocked_count") or 0)
            if blocked_count:
                blockers.append(f"signal_readiness_blocked:{blocked_count}")

        diagnostic_counts = registry.get("diagnostic_counts") if isinstance(registry.get("diagnostic_counts"), dict) else {}
        # Registry diagnostics include candidate-only and optional/fast-timeframe
        # artifacts.  Missing optional manifests are operator diagnostics, not a
        # hard execution gate, once the signal runtime has published a clean
        # active-symbol RF/GBM anchor readiness matrix.  Invalid active manifests
        # and feature incompatibilities are always safety blockers because they
        # can invalidate an artifact the runtime may load.
        for key in ("active_manifest_invalid", "feature_incompatible"):
            value = int(diagnostic_counts.get(key) or 0)
            if value:
                blockers.append(f"registry_{key}:{value}")

        use_missing_manifest_as_hard_gate = not signal_readiness or int(signal_readiness.get("blocked_count") or 0) > 0
        if use_missing_manifest_as_hard_gate:
            value = int(diagnostic_counts.get("missing_active_manifest") or 0)
            if value:
                blockers.append(f"registry_missing_active_manifest:{value}")

        by_reason = rejection_visibility.get("by_reason") if isinstance(rejection_visibility.get("by_reason"), dict) else {}
        anchor_blockers = {str(k): int(v or 0) for k, v in by_reason.items() if "anchor_training_incomplete" in str(k) or "anchor_readiness_not_ready" in str(k)}
        if anchor_blockers:
            total_anchor = sum(anchor_blockers.values())
            blockers.append(f"anchor_model_readiness_blockers:{total_anchor}")

        paused = bool(control_plane.get("paused"))
        circuit = control_plane.get("circuit_breaker") if isinstance(control_plane.get("circuit_breaker"), dict) else {}
        if paused:
            blockers.append("control_plane_paused")
        if circuit.get("triggered"):
            blockers.append(f"circuit_breaker:{circuit.get('reason') or 'triggered'}")

        status = "ready" if not blockers else ("blocked" if any("execution_integrity" in b or "anchor" in b or "registry" in b or "execution_eligible" in b for b in blockers) else "degraded")
        blocker_report = readiness_blocker_report(signal_readiness, runtime_blockers=blockers)
        payload = {
            "status": status,
            "ready": status == "ready",
            "blockers": blockers[:25],
            "price_ready": price_ready,
            "active_symbols": list(universe.get("active_symbols") or []),
            "active_count": active_count,
            "target_active_symbols": target,
            "execution_eligible_count": exec_eligible_count,
            "blocked_symbol_count": len(blocked_symbols),
            "execution_integrity": copy.deepcopy(execution_integrity),
            "anchor_blockers": anchor_blockers,
            "signal_readiness": copy.deepcopy(signal_readiness) if signal_readiness else {},
            "execution_readiness_matrix": copy.deepcopy(signal_readiness.get("execution_readiness_matrix") or signal_readiness.get("rows") or []) if signal_readiness else [],
            "readiness_blocker_report": copy.deepcopy(blocker_report),
            "blocker_summary_by_layer": copy.deepcopy(blocker_report.get("summary", {})),
            "summary": "READY" if status == "ready" else "; ".join(blockers[:4]),
            "updated_at": now_utc_ts(),
        }
        self._snapshot["trading_readiness"] = payload
        health = self._snapshot.setdefault("health", {})
        if isinstance(health, dict) and status != "ready":
            health["status"] = "degraded"
            health["warnings"] = max(int(health.get("warnings") or 0), len(blockers))
        if status != "ready":
            self._snapshot.setdefault("control_plane", {}).setdefault("runtime_observability", {})["trading_readiness"] = copy.deepcopy(payload)

    def _collect_rejection_visibility_rows_locked(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add_many(source_rows: Any, source_name: str) -> None:
            for raw in _as_list(source_rows):
                if not isinstance(raw, dict):
                    continue
                enriched = dict(raw)
                enriched.setdefault("source", source_name)
                normalized = _normalize_rejection_visibility_row(enriched)
                if normalized is None:
                    continue
                reason = str(normalized.get("reason") or "")
                status = str(normalized.get("status") or "")
                action = str(normalized.get("action") or "")
                if action.upper() not in {"HOLD", "REJECT", "REJECTED", "SKIP", "SKIPPED"} and "reject" not in status.lower():
                    continue
                if not reason or reason == "--":
                    continue
                key = (
                    str(normalized.get("ts") or ""),
                    str(normalized.get("symbol") or ""),
                    status,
                    reason,
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(normalized)

        decision_journal = self._snapshot.get("decision_journal")
        if isinstance(decision_journal, dict):
            add_many(decision_journal.get("recent"), "decision_journal")

        execution_runtime = self._snapshot.get("execution_runtime") or self._snapshot.get("paper_execution_runtime")
        if isinstance(execution_runtime, dict):
            add_many(execution_runtime.get("recent_decisions"), "execution_runtime")

        paper_runtime = self._snapshot.get("paper_execution_runtime")
        if isinstance(paper_runtime, dict) and paper_runtime is not execution_runtime:
            add_many(paper_runtime.get("recent_decisions"), "paper_execution_runtime")

        signal_runtime = self._snapshot.get("signal_runtime")
        if isinstance(signal_runtime, dict):
            add_many(signal_runtime.get("recent_decisions"), "signal_runtime")
            add_many(signal_runtime.get("recent_signals"), "signal_runtime")

        add_many(self._snapshot.get("decision_trace"), "decision_trace")

        rows.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
        return self._limit_list(rows, "decision_trace")

    def _refresh_rejection_visibility_locked(self) -> None:
        rows = self._collect_rejection_visibility_rows_locked()
        by_status: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        by_reason: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for row in rows:
            status = str(row.get("status") or "--")
            stage = str(row.get("stage") or "--")
            reason = str(row.get("reason") or "--")
            category = str(row.get("category") or "unknown")
            by_status[status] = by_status.get(status, 0) + 1
            by_stage[stage] = by_stage.get(stage, 0) + 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1

        payload = {
            "total_recent": len(rows),
            "by_status": by_status,
            "by_stage": by_stage,
            "by_reason": dict(sorted(by_reason.items(), key=lambda item: item[1], reverse=True)[:20]),
            "by_category": by_category,
            "recent": rows,
            "updated_at": now_utc_ts(),
        }
        self._snapshot["rejection_visibility"] = payload
        control_plane = self._snapshot.get("control_plane")
        if isinstance(control_plane, dict):
            observability = control_plane.setdefault("runtime_observability", {})
            if isinstance(observability, dict):
                observability["rejection_visibility"] = copy.deepcopy(payload)


    def fast_runtime_snapshot(self) -> dict[str, Any]:
        """Return a lightweight runtime payload for high-frequency UI refreshes.

        This method intentionally avoids the expensive full snapshot refresh path
        used by ``snapshot()``.  It is limited to fast-changing execution, price,
        control, and inference fields so browser PnL/SL/TP displays can refresh
        quickly without rebuilding training registry, model versions, logs, and
        history payloads every second.
        """
        with self._lock:
            self._load_existing_state()
            execution_runtime = copy.deepcopy(self._snapshot.get("execution_runtime") or {})
            paper_execution_runtime = copy.deepcopy(self._snapshot.get("paper_execution_runtime") or {})
            paper_execution = copy.deepcopy(self._snapshot.get("paper_execution") or {})
            inference = copy.deepcopy(self._snapshot.get("inference") or {})
            market_data = self._snapshot.get("market_data") or {}
            price_snapshot = copy.deepcopy((market_data or {}).get("runtime_price_snapshot") or {})
            control_plane = self._snapshot.get("control_plane") or {}
            runtime_gate = copy.deepcopy((control_plane or {}).get("runtime_gate") or {})
            live_runtime = copy.deepcopy((control_plane or {}).get("live_runtime") or {})
            system = copy.deepcopy(self._snapshot.get("system") or {})
            updated_at = _first_non_empty(
                execution_runtime.get("updated_at"),
                paper_execution_runtime.get("updated_at"),
                inference.get("updated_at"),
                price_snapshot.get("updated_at"),
                self._snapshot.get("updated_at"),
                now_utc_ts(),
            )
            return {
                "type": "fast_runtime",
                "updated_at": updated_at,
                "system": {
                    "execution_mode": system.get("execution_mode", "paper"),
                    "launcher_mode": system.get("launcher_mode"),
                },
                "execution_runtime": execution_runtime,
                "paper_execution_runtime": paper_execution_runtime,
                "paper_execution": paper_execution,
                "inference": inference,
                "market_data": {"runtime_price_snapshot": price_snapshot},
                "control_plane": {
                    "runtime_gate": runtime_gate,
                    "live_runtime": live_runtime,
                    "paused": control_plane.get("paused", False),
                    "pause_reason": control_plane.get("pause_reason", ""),
                    "updated_at": control_plane.get("updated_at"),
                },
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._load_existing_state()
            self._refresh_decision_journal_summary_locked()
            self._restore_empty_registry_projection_locked()
            self._refresh_symbol_universe_locked()
            self._refresh_market_price_diagnostics_locked()
            self._refresh_rejection_visibility_locked()
            self._refresh_trading_readiness_locked()
            self._snapshot["health"] = self._compute_health_locked()
            self._compact_snapshot_locked()
            return copy.deepcopy(self._snapshot)

    def events(self, limit: int = 200) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 1000))
        with self._lock:
            self._load_existing_state()
            return copy.deepcopy(self._events[-bounded_limit:])

    def flush(self) -> None:
        with self._serialized_mutation():
            self._refresh_decision_journal_summary_locked()
            self._persist_locked()

    def publish_system(self, **payload) -> None:
        with self._serialized_mutation():
            self._snapshot["system"].update(dict(payload or {}))
            self._persist_locked()

    def publish_event(
        self,
        *,
        level: str,
        component: str,
        category: str,
        message: str,
        data: dict[str, Any] | None = None,
        alert: bool = False,
    ) -> None:
        with self._serialized_mutation():
            self._append_event_locked(
                level=level,
                component=component,
                category=category,
                message=message,
                data=data,
                alert=alert,
            )
            self._persist_locked()

    def heartbeat(self, controller_name: str, *, status: str = "running", details: dict[str, Any] | None = None) -> None:
        with self._serialized_mutation():
            self._snapshot["controllers"][controller_name] = {
                "status": status,
                "details": dict(details or {}),
                "updated_at": now_utc_ts(),
            }
            self._persist_locked()

    def controller_success(self, controller_name: str, *, details: dict[str, Any] | None = None) -> None:
        with self._serialized_mutation():
            now = now_utc_ts()
            self._snapshot["controllers"][controller_name] = {
                "status": "healthy",
                "details": dict(details or {}),
                "updated_at": now,
            }
            error_state = self._snapshot.setdefault("error_state", {})
            active = error_state.setdefault("active_controller_errors", {})
            recovered = error_state.setdefault("recovered_controller_errors", [])
            previous = active.pop(str(controller_name), None)
            if isinstance(previous, dict):
                recovered.append({**previous, "recovered_at": now})
                error_state["recovered_controller_errors"] = recovered[-50:]
            self._persist_locked()

    def controller_error(self, controller_name: str, error: str) -> None:
        with self._serialized_mutation():
            self._snapshot["controllers"][controller_name] = {
                "status": "error",
                "details": {"error": str(error)},
                "updated_at": now_utc_ts(),
            }
            error_row = {
                "controller": controller_name,
                "error": str(error),
                "ts": now_utc_ts(),
            }
            self._snapshot["errors"].append(error_row)
            error_state = self._snapshot.setdefault("error_state", {})
            active = error_state.setdefault("active_controller_errors", {})
            active[str(controller_name)] = dict(error_row)
            self._snapshot["errors"] = self._snapshot["errors"][-50:]
            self._append_event_locked(
                level="ERROR",
                component=controller_name,
                category="controller_error",
                message=str(error),
                data={"controller": controller_name, "error": str(error)},
                alert=True,
            )
            self._persist_locked()

    def update_market_connectivity(self, *, status: str, **details) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["market_data"]
            payload["connectivity"] = {
                "status": str(status),
                "updated_at": now_utc_ts(),
                **dict(details or {}),
            }
            self._persist_locked()

    def update_market_universe(self, *, symbols, source: str, last_refresh_ts=None, **extra) -> None:
        normalized = [str(symbol).upper() for symbol in list(symbols or [])]
        with self._serialized_mutation():
            payload = self._snapshot["market_data"]
            payload["symbols_total"] = len(normalized)
            payload["universe"] = {
                "symbols": normalized,
                "source": str(source),
                "last_refresh_ts": last_refresh_ts,
                **dict(extra or {}),
            }
            self._persist_locked()

    def update_market_result(self, *, symbol: str, timeframe: str, rows_added: int, last_ts: int, status: str, **extra) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["market_data"]
            results = list(payload.get("recent_results", []))
            results.append({
                "symbol": str(symbol).upper(),
                "timeframe": str(timeframe),
                "rows_added": int(rows_added),
                "last_ts": int(last_ts),
                "status": str(status),
                "updated_at": now_utc_ts(),
                **dict(extra or {}),
            })
            payload["recent_results"] = self._limit_list(results, "market_updates")
            self._persist_locked()

    def update_training_runtime(self, **payload) -> None:
        with self._serialized_mutation():
            current = dict(self._snapshot.get("training_runtime", {}))
            current.update(dict(payload or {}))
            current["updated_at"] = now_utc_ts()
            self._snapshot["training_runtime"] = current
            self._persist_locked()

    def update_scheduler(self, *, due_jobs, queued_jobs, **extra) -> None:
        def _normalize_job(job):
            if isinstance(job, dict):
                return dict(job)
            return {
                "model_type": str(getattr(job, "model_type", "") or ""),
                "symbol": str(getattr(job, "symbol", "") or "").upper(),
                "interval": str(getattr(job, "interval", "") or ""),
                "reason": str(getattr(job, "reason", "") or ""),
            }
        with self._serialized_mutation():
            due_rows = [_normalize_job(job) for job in list(due_jobs or [])]
            queued_rows = [_normalize_job(job) for job in list(queued_jobs or [])]
            self._snapshot["scheduler"] = {
                "due_jobs_count": len(due_rows),
                "queued_jobs_count": len(queued_rows),
                "due_jobs": self._limit_list(due_rows, "scheduler_jobs"),
                "queued_jobs": self._limit_list(queued_rows, "scheduler_jobs"),
                "updated_at": now_utc_ts(),
                **dict(extra or {}),
            }
            self._persist_locked()

    def update_backtest_runtime(
        self,
        *,
        open_positions=None,
        recent_closed=None,
        trade_history=None,
        recent_messages=None,
        summary=None,
        **extra,
    ) -> None:
        with self._serialized_mutation():
            self._snapshot["backtest_runtime"] = {
                "state_owner": "backtest_controller",
                "execution_surface": "historical_backtest",
                "authoritative_for_runtime_positions": False,
                "open_positions": self._limit_list(open_positions or [], "backtest_open_positions"),
                "recent_closed": self._limit_list(recent_closed or [], "backtest_recent_closed"),
                "trade_history": self._limit_list(trade_history or [], "backtest_trade_history"),
                "recent_messages": self._limit_list(recent_messages or [], "recent_signals"),
                "summary": dict(summary or {}),
                "updated_at": now_utc_ts(),
                **dict(extra or {}),
            }
            self._persist_locked()


    def update_paper_execution_runtime(
        self,
        *,
        open_positions=None,
        recent_decisions=None,
        summary=None,
        trade_history=None,
        authority=None,
        runtime_gate=None,
        **extra,
    ) -> None:
        with self._serialized_mutation():
            limited_positions = self._limit_list(open_positions or [], "backtest_open_positions")
            limited_decisions = self._limit_list(recent_decisions or [], "recent_decisions")
            limited_messages = self._limit_list(extra.pop("recent_messages", []), "recent_decisions")
            runtime = _normalize_paper_execution_runtime_payload(
                open_positions=limited_positions,
                recent_decisions=limited_decisions,
                recent_messages=limited_messages,
                summary=summary,
                trade_history=self._limit_list(trade_history or [], "backtest_trade_history"),
                authority=authority,
                runtime_gate=runtime_gate,
                extra=dict(extra or {}),
            )
            runtime = apply_execution_trade_source(
                runtime,
                paths=self.paths,
                history_limit=_bounded_int(self._limits.get("backtest_trade_history"), 120, minimum=1),
            )
            runtime["summary"] = _normalize_execution_summary(
                runtime.get("summary"),
                open_positions=list(runtime.get("open_positions") or []),
                recent_decisions=list(runtime.get("recent_decisions") or []),
                trade_history=list(runtime.get("trade_history") or []),
            )
            runtime["execution_summary"] = dict(runtime["summary"])
            runtime["summary_contract_version"] = "execution_summary.v1"
            self._snapshot["paper_execution_runtime"] = runtime
            self._snapshot["paper_execution"] = copy.deepcopy(runtime)
            self._snapshot["execution_runtime"] = copy.deepcopy(runtime)
            self._persist_locked()

    def update_market_data(
        self,
        *,
        symbols_total: int | None = None,
        symbol_counts: dict[str, int] | None = None,
        funding_refresh: dict[str, Any] | None = None,
        open_interest_refresh: dict[str, Any] | None = None,
        latest_errors: list[str] | None = None,
        runtime_price_snapshot: dict[str, Any] | None = None,
    ) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["market_data"]
            if symbols_total is not None:
                payload["symbols_total"] = int(symbols_total)
            if symbol_counts is not None:
                payload["symbol_counts"] = dict(symbol_counts)
            if funding_refresh is not None:
                payload["funding_refresh"] = dict(funding_refresh)
            if open_interest_refresh is not None:
                payload["open_interest_refresh"] = dict(open_interest_refresh)
            if latest_errors is not None:
                payload["latest_errors"] = self._limit_list(latest_errors, "alerts")
            if runtime_price_snapshot is not None:
                payload["runtime_price_snapshot"] = dict(runtime_price_snapshot)
            self._persist_locked()

    def update_training(self, model_type: str, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            bucket = self._snapshot["training"].setdefault(str(model_type), {})
            bucket.update(dict(payload or {}))
            self._persist_locked()

    def update_registry(
        self,
        *,
        counts: dict[str, Any] | None = None,
        counts_by_model: dict[str, Any] | None = None,
        diagnostic_counts: dict[str, Any] | None = None,
        diagnostic_counts_by_model: dict[str, Any] | None = None,
        entries: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["training_registry"]
            if counts is not None:
                payload["counts"] = dict(counts)
            if counts_by_model is not None:
                payload["counts_by_model"] = dict(counts_by_model)
            if diagnostic_counts is not None:
                payload["diagnostic_counts"] = dict(diagnostic_counts)
            if diagnostic_counts_by_model is not None:
                payload["diagnostic_counts_by_model"] = {
                    str(model): dict(model_counts or {})
                    for model, model_counts in dict(diagnostic_counts_by_model or {}).items()
                }
            if entries is not None:
                payload["entries"] = self._limit_list(entries, "registry_entries")
            self._persist_locked()

    def append_training_result(self, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            results = self._snapshot.setdefault("training_results", [])
            results.append(dict(payload or {}))
            self._snapshot["training_results"] = self._limit_list(results, "recent_training_results")
            self._persist_locked()

    def update_inference(
        self,
        *,
        decisions: list[dict[str, Any]] | None = None,
        open_positions: dict[str, Any] | None = None,
        pnl_summary: dict[str, Any] | None = None,
        last_error: str | None = None,
        risk_summary: dict[str, Any] | None = None,
    ) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["inference"]
            if decisions is not None:
                payload["decisions"] = self._limit_list(decisions, "recent_decisions")
            if open_positions is not None:
                payload["open_positions"] = dict(open_positions)
            if pnl_summary is not None:
                payload["pnl_summary"] = dict(pnl_summary)
            payload["last_error"] = last_error
            if risk_summary is not None:
                payload["risk_summary"] = dict(risk_summary)
            payload["updated_at"] = now_utc_ts()
            self._persist_locked()

    def update_control_plane(self, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            current = dict(self._snapshot.get("control_plane", {}))
            current.update(dict(payload or {}))
            self._snapshot["control_plane"] = current
            self._persist_locked()

    def append_control_action(self, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            control_plane = self._snapshot.setdefault("control_plane", {})
            actions = list(control_plane.get("recent_actions", []))
            actions.append(dict(payload or {}))
            control_plane["recent_actions"] = self._limit_list(actions, "control_actions")
            self._persist_locked()

    def update_model_versions(self, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            self._snapshot["model_versions"] = dict(payload or {})
            self._persist_locked()

    def append_decision_trace(self, payload: dict[str, Any]) -> None:
        with self._serialized_mutation():
            trace = self._snapshot.setdefault("decision_trace", [])
            trace.append(dict(payload or {}))
            self._snapshot["decision_trace"] = self._limit_list(trace, "decision_trace")
            self._persist_locked()

    def update_signal_runtime(
        self,
        *,
        latest_mode: str | None = None,
        latest_symbol: str | None = None,
        latest_action: str | None = None,
        latest_reason: str | None = None,
        latest_ts: float | None = None,
        recent_signals: list[dict[str, Any]] | None = None,
        recent_decisions: list[dict[str, Any]] | None = None,
        runtime_gate: dict[str, Any] | None = None,
        recent_readiness: dict[str, Any] | None = None,
        adaptive_mode: dict[str, Any] | None = None,
        last_error: str | None = None,
        source_priority: int | None = None,
        source_name: str | None = None,
    ) -> None:
        with self._serialized_mutation():
            payload = self._snapshot["signal_runtime"]
            if latest_mode is not None:
                payload["latest_mode"] = latest_mode
            if latest_symbol is not None:
                payload["latest_symbol"] = latest_symbol
            if latest_action is not None:
                payload["latest_action"] = latest_action
            if latest_reason is not None:
                payload["latest_reason"] = latest_reason
            if latest_ts is not None:
                payload["latest_ts"] = latest_ts
            if recent_signals is not None:
                payload["recent_signals"] = self._limit_list(recent_signals, "recent_signals")
            if recent_decisions is not None:
                payload["recent_decisions"] = self._limit_list(recent_decisions, "recent_decisions")
            if recent_readiness is not None:
                payload["readiness"] = dict(recent_readiness)
                payload["readiness_blocker_report"] = readiness_blocker_report(recent_readiness)
            if runtime_gate is not None:
                payload["runtime_gate"] = dict(runtime_gate)
            if adaptive_mode is not None:
                payload["adaptive_mode"] = dict(adaptive_mode)
                self._snapshot.setdefault("control_plane", {})["adaptive_mode"] = dict(adaptive_mode)
            payload["last_error"] = last_error
            if source_priority is not None:
                payload["source_priority"] = int(source_priority)
            if source_name is not None:
                payload["source_name"] = str(source_name)
            self._persist_locked()
