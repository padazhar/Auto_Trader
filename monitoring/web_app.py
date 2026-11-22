from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from contextlib import asynccontextmanager, suppress
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Request, Response, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.config_contract import SecretValidationError, validate_runtime_secrets
from core.shared_utils import now_utc_ts
from execution.portfolio_state import apply_execution_trade_source, summarize_closed_trade_rows
from core.symbol_management import load_symbol_actions, load_symbol_state, save_symbol_actions, save_symbol_state, symbol_entry
from monitoring.auth import DASHBOARD_TOKEN_COOKIE, token_required, websocket_token_required
from monitoring.control_plane_store import ControlPlaneStore
from monitoring.publisher import MonitorPublisher


class WebsocketHub:
    """Manage websocket clients connected to the monitoring server."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def add(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def remove(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            await self.remove(websocket)




class LoginRequest(BaseModel):
    token: str = ""


class LogoutRequest(BaseModel):
    clear_local_token: bool = True

class RetryRequest(BaseModel):
    symbols: list[str] = []
    retry_all_blocked: bool = False
    updated_by: str = "ui"


class PauseRequest(BaseModel):
    paused: bool
    reason: str = ""
    updated_by: str = "ui"


class SymbolControlRequest(BaseModel):
    symbol: str
    updated_by: str = "ui"


class CircuitResetRequest(BaseModel):
    reason: str = "manual_reset"
    updated_by: str = "ui"


class RollbackRequest(BaseModel):
    model_type: str
    symbol: str
    interval: str
    manifest_path: str
    model_role: str | None = None
    updated_by: str = "ui"


def _recent_decision_journal(paths, limit: int = 100) -> list[dict[str, Any]]:
    path = paths.decision_journal_path
    if not path.exists():
        return []
    bounded_limit = max(1, min(int(limit), 1000))
    max_scan_bytes = 2_000_000
    rows: deque[dict[str, Any]] = deque(maxlen=bounded_limit)
    try:
        file_size = path.stat().st_size
        scan_from = max(0, file_size - max_scan_bytes)
        with path.open("rb") as handle:
            if scan_from:
                handle.seek(scan_from)
                handle.readline()
            for raw_line in handle:
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
        return []
    result = list(rows)
    result.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
    return result[:bounded_limit]



def _canonical_execution_summary(summary: Any, *, open_positions: list[Any] | None = None, trade_history: list[Any] | None = None) -> dict[str, Any]:
    """Return the Execution card/report summary contract.

    This keeps the monitoring API and browser UI on one canonical payload:
    open positions come from the runtime authority; completed trades/PnL can be
    derived from closed trade rows when a summary is absent or incomplete.
    """
    normalized = dict(summary or {}) if isinstance(summary, dict) else {}
    positions = [row for row in (open_positions or []) if isinstance(row, dict)]
    closed_rows = [row for row in (trade_history or []) if isinstance(row, dict)]
    derived = summarize_closed_trade_rows(closed_rows) if closed_rows else {}

    def _int_value(*values: Any, default: int = 0) -> int:
        for value in values:
            if value is None or value == "":
                continue
            try:
                return int(float(value))
            except Exception:
                continue
        return int(default)

    def _float_value(*values: Any, default: float = 0.0) -> float:
        for value in values:
            if value is None or value == "":
                continue
            try:
                return float(value)
            except Exception:
                continue
        return float(default)

    open_count = _int_value(normalized.get("open_positions"), normalized.get("open_count"), len(positions), default=len(positions))
    completed_count = _int_value(normalized.get("completed_count"), normalized.get("trades"), normalized.get("closed_count"), derived.get("completed_count"), default=0)
    wins = _int_value(normalized.get("wins"), derived.get("wins"), default=0)
    losses = _int_value(normalized.get("losses"), derived.get("losses"), default=0)
    total_pnl = _float_value(normalized.get("total_pnl"), normalized.get("realized_pnl_usd"), normalized.get("pnl_usd"), derived.get("total_pnl"), default=0.0)
    win_rate = _float_value(normalized.get("win_rate"), default=(float(wins) / float(completed_count) if completed_count > 0 else 0.0))

    normalized.update({
        "open_positions": open_count,
        "open_count": open_count,
        "completed_count": completed_count,
        "trades": completed_count,
        "wins": wins,
        "losses": losses,
        "total_pnl": total_pnl,
        "realized_pnl_usd": _float_value(normalized.get("realized_pnl_usd"), total_pnl, default=total_pnl),
        "win_rate": win_rate,
    })
    normalized.setdefault("source", normalized.get("source") or ("trade_history" if closed_rows else "runtime_execution_payload"))
    return normalized


def _normalize_snapshot(snapshot: dict[str, Any] | None, paths: Any | None = None) -> dict[str, Any]:
    def _derive_training_runtime(
        raw_training_runtime: dict[str, Any],
        raw_training_registry: dict[str, Any],
    ) -> dict[str, Any]:
        training_runtime = dict(raw_training_runtime or {})
        registry = dict(raw_training_registry or {})
        entries = list(registry.get("entries") or [])
        counts_by_model = dict(registry.get("counts_by_model") or {})
        model_keys = ("rf", "gbm", "dqn")

        def _entry_ts(row: dict[str, Any]) -> float:
            for key in ("updated_at", "completed_at", "last_success_at", "started_at", "queued_at", "created_at"):
                value = row.get(key)
                if value is not None:
                    with suppress(Exception):
                        return float(value)
            return 0.0

        for model_key in model_keys:
            model_entries = [
                dict(row or {})
                for row in entries
                if str((row or {}).get("model_type") or "").lower() == model_key
            ]
            model_entries.sort(key=_entry_ts, reverse=True)
            counts = dict(counts_by_model.get(model_key) or {})
            running_rows = [row for row in model_entries if str(row.get("status") or "").lower() == "running"]
            queued_rows = [row for row in model_entries if str(row.get("status") or "").lower() == "queued"]
            latest_row = running_rows[0] if running_rows else (model_entries[0] if model_entries else {})

            running_key = f"{model_key}_running"
            pending_key = f"{model_key}_pending"
            current_key = f"current_{model_key}"
            progress_key = f"{model_key}_progress"
            message_key = f"last_{model_key}_message"

            training_runtime[running_key] = int(training_runtime.get(running_key) or counts.get("running") or len(running_rows))
            training_runtime[pending_key] = int(training_runtime.get(pending_key) or counts.get("queued") or len(queued_rows))

            if latest_row and (not training_runtime.get(current_key) or str(training_runtime.get(current_key)).strip() in {"", "--"}):
                symbol = str(latest_row.get("symbol") or "--").upper()
                interval = str(latest_row.get("interval") or "--")
                status = str(latest_row.get("status") or "--").upper()
                training_runtime[current_key] = f"{symbol} {interval}"
                training_runtime[message_key] = f"{status} {symbol} {interval}"

            progress_value = training_runtime.get(progress_key)
            if progress_value is None:
                progress_value = 0.0
            with suppress(Exception):
                progress_value = float(progress_value)
            if training_runtime[running_key] > 0 and float(progress_value or 0.0) <= 0.0:
                progress_value = 1.0
            training_runtime[progress_key] = max(0.0, min(float(progress_value or 0.0), 100.0))

        training_runtime.setdefault("running_jobs", [row for row in entries if str((row or {}).get("status") or "").lower() == "running"])
        training_runtime.setdefault("queued_jobs", [row for row in entries if str((row or {}).get("status") or "").lower() == "queued"])
        training_runtime["inflight_count"] = int(training_runtime.get("inflight_count") or len(training_runtime.get("running_jobs") or []))
        training_runtime["queue_depth"] = int(training_runtime.get("queue_depth") or len(training_runtime.get("queued_jobs") or []))
        training_runtime.setdefault("updated_at", now_utc_ts())
        return training_runtime

    def _derive_execution_messages(execution_runtime: dict[str, Any]) -> list[str]:
        existing_messages = execution_runtime.get("recent_messages")
        if isinstance(existing_messages, list) and existing_messages:
            return [str(item) for item in existing_messages[:25]]

        messages: list[str] = []
        authority = dict(execution_runtime.get("authority") or {})
        recent_decisions = list(execution_runtime.get("recent_decisions") or [])
        summary = dict(execution_runtime.get("summary") or {})
        runtime_gate = dict(summary.get("runtime_gate") or authority.get("runtime_gate") or {})

        if runtime_gate.get("blocked"):
            messages.append(f"runtime gate blocked: {runtime_gate.get('reason') or runtime_gate.get('circuit_reason') or 'unknown'}")
        elif authority.get("authoritative") is False:
            messages.append(f"execution authority unavailable: {authority.get('authority_reason') or authority.get('reason') or 'not_authoritative'}")

        reason_counts: dict[str, int] = {}
        for row in recent_decisions[-25:]:
            payload = row or {}
            reason = str(payload.get("rejection_reason") or payload.get("execution_blocker") or payload.get("reason") or "").strip()
            status = str(payload.get("execution_status") or "").strip()
            key = f"{status}: {reason}" if status and reason else (reason or status)
            if not key:
                continue
            reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:5]:
            messages.append(f"execution block: {reason} ({count})")

        if not messages and execution_runtime.get("open_position_count"):
            messages.append(f"{execution_runtime.get('open_position_count')} runtime position(s) active")
        if not messages:
            messages.append("execution is idle until fresh prices and approved non-HOLD decisions arrive")
        return messages

    def _derive_health(payload: dict[str, Any]) -> dict[str, Any]:
        health = dict(payload.get("health") or {})
        controllers = dict(payload.get("controllers") or {})
        system = dict(payload.get("system") or {})
        trading_readiness = dict(payload.get("trading_readiness") or {})
        execution_runtime = dict(payload.get("execution_runtime") or payload.get("paper_execution_runtime") or {})
        execution_summary = dict(execution_runtime.get("summary") or {})
        runtime_gate = dict(execution_runtime.get("runtime_gate") or execution_summary.get("runtime_gate") or {})
        integrity_status = str(
            runtime_gate.get("execution_integrity_status")
            or execution_summary.get("execution_integrity_status")
            or "ok"
        ).lower()
        integrity_reason = str(
            runtime_gate.get("execution_integrity_error")
            or execution_summary.get("execution_integrity_error")
            or (runtime_gate.get("reason") if runtime_gate.get("blocked") else "")
            or ""
        )
        integrity_blocked = integrity_status != "ok" or "execution_state_integrity_failure" in integrity_reason
        statuses = [str((row or {}).get("status") or "").lower() for row in controllers.values()]
        if any(status == "error" for status in statuses):
            health["status"] = "error"
        elif integrity_blocked:
            health["status"] = "blocked"
            health["execution_integrity"] = {"status": integrity_status if integrity_status != "ok" else "blocked", "blocked": True, "reason": integrity_reason or "execution_state_integrity_failure"}
        elif trading_readiness and not trading_readiness.get("ready"):
            health["status"] = "degraded"
        elif health.get("status") in (None, "", "unknown"):
            if any(status in {"running", "healthy", "mark_to_market"} for status in statuses):
                health["status"] = "running"
            else:
                health["status"] = "starting"
        if statuses:
            health["controller_liveness_status"] = "error" if any(status == "error" for status in statuses) else ("running" if any(status in {"running", "healthy", "mark_to_market"} for status in statuses) else "unknown")
        if trading_readiness:
            health["trading_readiness_status"] = str(trading_readiness.get("status") or "unknown")
            health["trading_ready"] = bool(trading_readiness.get("ready"))
        api = dict(health.get("api_connectivity") or {})
        api.setdefault("binance", "paper" if str(system.get("execution_mode") or "paper").lower() != "live" else "unknown")
        health["api_connectivity"] = api
        db = dict(health.get("db_connectivity") or {})
        db.setdefault("market_storage", "connected" if payload.get("market_data") else "unknown")
        health["db_connectivity"] = db
        warning_count = int(health.get("warnings") or 0)
        if trading_readiness and not trading_readiness.get("ready"):
            warning_count = max(warning_count, len(trading_readiness.get("blockers") or []), 1)
        health["warnings"] = warning_count
        event_counters = payload.get("runtime_event_counters") if isinstance(payload.get("runtime_event_counters"), dict) else {}
        error_state = payload.get("error_state") if isinstance(payload.get("error_state"), dict) else {}
        active_controller_errors = error_state.get("active_controller_errors") if isinstance(error_state.get("active_controller_errors"), dict) else {}
        recovered_controller_errors = error_state.get("recovered_controller_errors") if isinstance(error_state.get("recovered_controller_errors"), list) else []
        health["warnings_since_start"] = int(event_counters.get("warnings_since_start") or 0)
        health["errors_since_start"] = int(event_counters.get("errors_since_start") or 0)
        health["active_controller_errors"] = len(active_controller_errors)
        health["recovered_controller_errors"] = len(recovered_controller_errors)
        if integrity_blocked:
            health["errors"] = max(int(health.get("errors") or 0), 1, health["errors_since_start"])
        else:
            health["errors"] = max(int(health.get("errors") or 0), health["errors_since_start"])
        return health

    def _normalize_market_data(raw_market_data: dict[str, Any] | None) -> dict[str, Any]:
        market_data = dict(raw_market_data or {})
        universe_payload = market_data.get("universe") or []
        if isinstance(universe_payload, dict):
            universe_symbols = [str(symbol).upper() for symbol in list(universe_payload.get("symbols") or [])]
            market_data["universe"] = universe_symbols
            market_data.setdefault("universe_source", universe_payload.get("source"))
            market_data.setdefault("last_universe_refresh_ts", universe_payload.get("last_refresh_ts"))
        elif isinstance(universe_payload, list):
            market_data["universe"] = [str(symbol).upper() for symbol in universe_payload]
        else:
            market_data["universe"] = []
        market_data.setdefault("latest_updates", list(market_data.get("recent_results") or []))
        symbol_universe = market_data.get("symbol_universe") if isinstance(market_data.get("symbol_universe"), dict) else {}
        active_symbols = [str(symbol).upper() for symbol in list(symbol_universe.get("active_symbols") or []) if symbol]
        if active_symbols:
            market_data["universe"] = active_symbols
            market_data["universe_source"] = symbol_universe.get("source") or "symbol_state"
            market_data["last_universe_refresh_ts"] = symbol_universe.get("updated_at") or market_data.get("last_universe_refresh_ts")
        price_diag = dict(market_data.get("runtime_price_snapshot") or {})
        price_diag.setdefault("exists", False)
        price_diag.setdefault("expected_symbol_count", len(market_data.get("universe") or []))
        price_diag.setdefault("priced_symbol_count", int(market_data.get("priced_symbol_count") or 0))
        price_diag.setdefault("missing_symbol_count", len(market_data.get("missing_price_symbols") or []))
        price_diag.setdefault("stale_symbol_count", len(market_data.get("stale_price_symbols") or []))
        price_diag.setdefault("all_symbols_priced", False)
        price_diag.setdefault("missing_symbols", list(market_data.get("missing_price_symbols") or []))
        price_diag.setdefault("stale_symbols", list(market_data.get("stale_price_symbols") or []))
        price_diag.setdefault("price_source_counts", {})
        market_data["runtime_price_snapshot"] = price_diag
        market_data.setdefault("price_cache_status", "ready" if price_diag.get("all_symbols_priced") and not price_diag.get("stale_symbol_count") else "degraded")
        return market_data

    def _normalize_inference_open_positions(raw_open_positions: Any, *, mode: str) -> list[dict[str, Any]]:
        if isinstance(raw_open_positions, list):
            return [dict(row or {}) for row in raw_open_positions]
        if not isinstance(raw_open_positions, dict):
            return []
        rows: list[dict[str, Any]] = []
        for symbol, row in raw_open_positions.items():
            details = dict(row or {})
            last_price = details.get("last_price")
            if last_price is None:
                last_price = details.get("latest_price") if details.get("latest_price") is not None else details.get("reference_price")
            estimated_pnl = details.get("estimated_pnl")
            if estimated_pnl is None:
                estimated_pnl = details.get("pnl_usd") if details.get("pnl_usd") is not None else details.get("unrealized_pnl_usd")
            normalized = {**details}
            normalized.update(
                {
                    "symbol": str(symbol).upper(),
                    "side": details.get("side") or "OPEN",
                    "mode": details.get("mode") or mode,
                    "entry_price": details.get("entry_price"),
                    "last_price": last_price,
                    "latest_price": details.get("latest_price") if details.get("latest_price") is not None else last_price,
                    "reference_price": details.get("reference_price") if details.get("reference_price") is not None else last_price,
                    "last_price_ts": details.get("last_price_ts") or details.get("updated_at") or details.get("opened_at"),
                    "estimated_pnl": estimated_pnl if estimated_pnl is not None else 0.0,
                }
            )
            rows.append(normalized)
        return rows

    def _normalize_recent_signals(signal_generator: dict[str, Any], inference: dict[str, Any], *, mode: str) -> list[dict[str, Any]]:
        rows = list(signal_generator.get("recent_signals") or [])
        if rows:
            return rows
        normalized: list[dict[str, Any]] = []
        for row in list(inference.get("decisions") or [])[-100:]:
            normalized.append(
                {
                    "symbol": str(row.get("symbol") or "--").upper(),
                    "action": row.get("effective_action") or row.get("action") or "--",
                    "mode": row.get("mode") or mode,
                    "reason": row.get("reason") or "--",
                    "ts": row.get("ts"),
                    "raw": row.get("reason") or "--",
                }
            )
        return normalized

    payload = dict(snapshot or {})

    system = dict(payload.get("system") or {})
    inference = dict(payload.get("inference") or {})
    signal_runtime = dict(payload.get("signal_runtime") or {})
    backtest_runtime = dict(payload.get("backtest_runtime") or {})
    paper_execution_runtime = dict(payload.get("paper_execution_runtime") or {})
    training_registry = dict(payload.get("training_registry") or {})
    training_runtime = _derive_training_runtime(dict(payload.get("training_runtime") or payload.get("training") or {}), training_registry)
    control_plane = dict(payload.get("control_plane") or {})
    decision_journal = dict(payload.get("decision_journal") or {})
    live_runtime = dict(control_plane.get("live_runtime") or {})
    market_data = _normalize_market_data(payload.get("market_data") or {})
    execution_mode = str(system.get("execution_mode") or system.get("launcher_mode") or "paper")

    signal_generator = dict(payload.get("signal_generator") or signal_runtime)
    signal_generator["recent_signals"] = _normalize_recent_signals(signal_generator, inference, mode=execution_mode)
    signal_generator.setdefault("recent_decisions", list(signal_generator.get("recent_decisions") or []))
    signal_generator.setdefault("runtime_gate", dict(signal_generator.get("runtime_gate") or {}))
    signal_generator.setdefault("last_error", signal_generator.get("last_error"))
    payload["signal_generator"] = signal_generator
    payload["signal_runtime"] = signal_generator

    backtest = dict(payload.get("backtest") or backtest_runtime)
    backtest.setdefault("execution_surface", "historical_backtest")
    backtest.setdefault("authoritative_for_runtime_positions", False)
    backtest.setdefault("open_positions", list(backtest.get("open_positions") or []))
    backtest.setdefault("recent_messages", list(backtest.get("recent_messages") or []))
    backtest.setdefault("summary", dict(backtest.get("summary") or {}))
    payload["backtest"] = backtest

    paper_execution = dict(paper_execution_runtime or payload.get("paper_execution") or {})
    paper_execution.setdefault("execution_surface", "paper_forward_execution")
    paper_execution.setdefault("authoritative_for_runtime_positions", execution_mode != "live")
    paper_execution.setdefault("open_positions", list(paper_execution.get("open_positions") or []))
    paper_execution.setdefault("recent_decisions", list(paper_execution.get("recent_decisions") or []))
    paper_execution.setdefault("recent_messages", list(paper_execution.get("recent_messages") or []))
    paper_execution.setdefault("trade_history", list(paper_execution.get("trade_history") or []))
    paper_execution.setdefault("summary", dict(paper_execution.get("summary") or {}))
    payload["paper_execution"] = paper_execution

    if training_registry.get("recent_results") is None:
        training_registry["recent_results"] = list(payload.get("training_results") or [])
    training_registry.setdefault("entries", list(training_registry.get("entries") or []))
    payload["training_registry"] = training_registry

    decision_journal.setdefault("entries", int(decision_journal.get("entries") or 0))
    decision_journal.setdefault("latest_ts", decision_journal.get("latest_ts"))
    decision_journal.setdefault("latest_writer", str(decision_journal.get("latest_writer") or ""))
    decision_journal.setdefault("latest_stage", str(decision_journal.get("latest_stage") or ""))
    decision_journal.setdefault("writers", dict(decision_journal.get("writers") or {}))
    decision_journal.setdefault("stages", dict(decision_journal.get("stages") or {}))
    if decision_journal.get("recent") is None:
        decision_journal["recent"] = []
    payload["decision_journal"] = decision_journal

    execution_open_positions = _normalize_inference_open_positions(
        paper_execution.get("open_positions") if execution_mode != "live" else inference.get("open_positions"),
        mode=execution_mode,
    )
    payload["execution_runtime"] = {
        "execution_surface": "live_forward_execution" if execution_mode == "live" else "paper_forward_execution",
        "state_owner": "live_trading_controller",
        "authoritative_for_runtime_positions": bool(live_runtime.get("authoritative")) if execution_mode == "live" else True,
        "open_positions": execution_open_positions,
        "open_position_count": int(live_runtime.get("open_position_count")) if execution_mode == "live" and live_runtime.get("open_position_count") is not None else len(execution_open_positions),
        "open_order_count": int(live_runtime.get("open_order_count")) if live_runtime.get("open_order_count") is not None else int(paper_execution.get("open_order_count") or 0),
        "last_recovery_sources": list(live_runtime.get("last_recovery_sources") or paper_execution.get("last_recovery_sources") or []),
        "recent_decisions": list(paper_execution.get("recent_decisions") or ([] if execution_mode != "live" else inference.get("decisions") or [])),
        "summary": dict(paper_execution.get("summary") or ({} if execution_mode != "live" else inference.get("pnl_summary") or {})),
        "trade_history": list(paper_execution.get("trade_history") or []),
        "authority": dict(live_runtime or paper_execution.get("authority") or {}),
        "recent_messages": list(paper_execution_runtime.get("recent_messages") or paper_execution.get("recent_messages") or []),
    }
    if execution_mode != "live" and paths is not None:
        try:
            payload["execution_runtime"] = apply_execution_trade_source(payload["execution_runtime"], paths=paths)
            payload["paper_execution"] = apply_execution_trade_source(payload["paper_execution"], paths=paths)
            payload["paper_execution_runtime"] = apply_execution_trade_source(paper_execution_runtime, paths=paths)
        except Exception:
            pass
    for key in ("execution_runtime", "paper_execution_runtime", "paper_execution"):
        surface = payload.get(key)
        if isinstance(surface, dict):
            canonical_summary = _canonical_execution_summary(
                surface.get("summary"),
                open_positions=list(surface.get("open_positions") or []),
                trade_history=list(surface.get("trade_history") or []),
            )
            surface["summary"] = canonical_summary
            surface["execution_summary"] = dict(canonical_summary)
            surface.setdefault("summary_contract_version", "execution_summary.v1")
            surface.setdefault("trade_history", list(surface.get("trade_history") or []))
            payload[key] = surface

    payload["execution_runtime"]["recent_messages"] = _derive_execution_messages(payload["execution_runtime"])

    if system.get("started_at") is not None and system.get("uptime_sec") in (None, ""):
        with suppress(Exception):
            system["uptime_sec"] = max(0.0, float(now_utc_ts()) - float(system.get("started_at")))

    payload["system"] = system
    control_lineage = []
    if isinstance(control_plane, dict):
        control_lineage = list(control_plane.get("decision_lineage_preview") or [])
    if not control_lineage:
        control_lineage = list(payload.get("decision_lineage_preview") or [])
    payload["decision_lineage_preview"] = control_lineage
    if isinstance(control_plane, dict):
        control_plane["decision_lineage_preview"] = control_lineage
    payload["control_plane"] = control_plane
    payload["market_data"] = market_data
    payload["training"] = training_runtime
    payload.setdefault("controllers", {})
    payload.setdefault("scheduler", {})
    payload.setdefault("model_versions", {})
    payload["health"] = _derive_health(payload)
    payload["status"] = payload["health"].get("status", payload.get("status") or "starting")
    payload.setdefault("updated_at", now_utc_ts())
    return payload



def _monitoring_owner_payload(paths) -> dict[str, Any]:
    """Return a small payload that lets the launcher identify the monitoring owner."""
    import os

    return {
        "ok": True,
        "component": "monitoring",
        "pid": os.getpid(),
        "project_root": str(getattr(paths, "project_root", "")),
        "runtime_dir": str(getattr(paths, "runtime_dir", "")),
        "updated_at": now_utc_ts(),
    }


def _monitoring_owner_path(paths):
    """Return the local owner marker path for diagnostics."""
    runtime_dir = getattr(paths, "monitoring_runtime_dir", None) or getattr(paths, "runtime_dir", None)
    if runtime_dir is None:
        return None
    return runtime_dir / "monitoring_server_owner.json"


def _write_monitoring_owner_marker(paths) -> None:
    """Write a best-effort owner marker for the running monitoring server."""
    path = _monitoring_owner_path(paths)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_monitoring_owner_payload(paths), separators=(",", ":")), encoding="utf-8")
    except Exception:
        return


def _clear_monitoring_owner_marker(paths) -> None:
    """Remove the local owner marker on clean shutdown."""
    path = _monitoring_owner_path(paths)
    if path is None:
        return
    with suppress(Exception):
        path.unlink()

def create_app(paths) -> FastAPI:
    """Create the monitoring FastAPI application."""
    try:
        validate_runtime_secrets(paths.config)
    except SecretValidationError as exc:
        raise RuntimeError(f"Monitoring server startup blocked: {exc}") from exc

    publisher = MonitorPublisher(paths)
    control_store = ControlPlaneStore(paths)
    hub = WebsocketHub()
    static_dir = paths.monitoring_static_dir

    def _runtime_interval(name: str, default: float, *, minimum: float = 0.25, maximum: float = 60.0) -> float:
        runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
        try:
            value = float(getattr(runtime_cfg, name, default) if runtime_cfg is not None else default)
        except Exception:
            value = float(default)
        return max(float(minimum), min(float(value), float(maximum)))

    def _fast_runtime_payload() -> dict[str, Any]:
        payload = publisher.fast_runtime_snapshot()
        # Keep the execution card contract identical to the full snapshot surface,
        # then strip heavy/slow sections so this endpoint remains cheap for 1s UI refresh.
        normalized = _normalize_snapshot(payload, paths)
        for heavy_key in (
            "training_registry",
            "training_results",
            "model_versions",
            "scheduler",
            "events",
            "alerts",
            "logs",
        ):
            normalized.pop(heavy_key, None)
        return normalized

    async def _watch_files() -> None:
        last_snapshot_ts = None
        last_fast_ts = None
        last_event_count = None
        last_heavy_sent_at = 0.0
        last_heartbeat_sent_at = 0.0
        fast_interval = _runtime_interval("monitoring_fast_snapshot_interval_sec", 1.0, minimum=0.25, maximum=10.0)
        heavy_interval = _runtime_interval("monitoring_heavy_snapshot_interval_sec", 5.0, minimum=1.0, maximum=60.0)
        while True:
            now_ts = now_utc_ts()
            fast_payload = _fast_runtime_payload()
            fast_ts = fast_payload.get("updated_at")
            if fast_ts != last_fast_ts:
                last_fast_ts = fast_ts
                await hub.broadcast_json({"type": "fast_runtime", "payload": fast_payload})

            if last_heavy_sent_at <= 0.0 or (float(now_ts) - float(last_heavy_sent_at)) >= heavy_interval:
                snapshot = _normalize_snapshot(publisher.snapshot(), paths)
                if snapshot.get("updated_at") != last_snapshot_ts:
                    last_snapshot_ts = snapshot.get("updated_at")
                    await hub.broadcast_json({"type": "snapshot", "payload": snapshot})
                last_heavy_sent_at = float(now_ts)

            events = publisher.events(limit=50)
            if len(events) != last_event_count:
                last_event_count = len(events)
                await hub.broadcast_json({"type": "events", "payload": events})

            # A live socket that carries no changing snapshot can otherwise look frozen
            # in the browser. Heartbeats drive the client watchdog without repainting data.
            if last_heartbeat_sent_at <= 0.0 or (float(now_ts) - float(last_heartbeat_sent_at)) >= max(1.0, float(fast_interval)):
                await hub.broadcast_json({"type": "heartbeat", "payload": {"server_ts": float(now_ts)}})
                last_heartbeat_sent_at = float(now_ts)
            await asyncio.sleep(fast_interval)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _write_monitoring_owner_marker(paths)
        task = asyncio.create_task(_watch_files(), name="monitoring-file-watch")
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            _clear_monitoring_owner_marker(paths)

    app = FastAPI(title="Auto Trader Monitoring", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def _cache_control(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/api/") or request.url.path.endswith(".js") or request.url.path.endswith(".css"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _auth_dependency(request: Request) -> None:
        return token_required(request, paths)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/styles.css")
    async def styles() -> FileResponse:
        return FileResponse(static_dir / "styles.css")

    @app.get("/app.js")
    async def app_js() -> FileResponse:
        return FileResponse(static_dir / "app.js")

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/auth/status")
    async def auth_status(request: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "authenticated": True, "auth_required": False})

    @app.post("/api/auth/login")
    async def auth_login(payload: LoginRequest) -> JSONResponse:
        response = JSONResponse({"ok": True, "authenticated": True, "auth_required": False})
        response.delete_cookie(DASHBOARD_TOKEN_COOKIE, path="/")
        return response

    @app.post("/api/auth/logout")
    async def auth_logout() -> JSONResponse:
        response = JSONResponse({"ok": True, "authenticated": False, "auth_required": False})
        response.delete_cookie(DASHBOARD_TOKEN_COOKIE, path="/")
        return response

    @app.get("/api/monitoring-owner")
    async def monitoring_owner() -> JSONResponse:
        return JSONResponse(_monitoring_owner_payload(paths))

    @app.get("/api/health")
    async def health(_: None = Depends(_auth_dependency)) -> JSONResponse:
        snapshot = _normalize_snapshot(publisher.snapshot(), paths)
        payload = {
            "status": snapshot.get("health", {}).get("status", "unknown"),
            "updated_at": snapshot.get("updated_at"),
            "controllers": snapshot.get("controllers", {}),
            "control_plane": snapshot.get("control_plane", {}),
        }
        payload.update({key: value for key, value in _monitoring_owner_payload(paths).items() if key not in payload})
        return JSONResponse(payload)

    @app.get("/api/snapshot")
    async def api_snapshot(_: None = Depends(_auth_dependency)) -> JSONResponse:
        snapshot = _normalize_snapshot(publisher.snapshot(), paths)
        snapshot["control_plane"] = control_store.summary()
        snapshot["model_versions"] = control_store.list_model_versions()
        return JSONResponse(_normalize_snapshot(snapshot, paths))

    @app.get("/api/runtime/fast")
    async def api_runtime_fast(_: None = Depends(_auth_dependency)) -> JSONResponse:
        return JSONResponse(_fast_runtime_payload())

    @app.get("/api/ui-config")
    async def api_ui_config(_: None = Depends(_auth_dependency)) -> JSONResponse:
        runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
        monitoring_cfg = getattr(getattr(paths, "config", None), "monitoring", None)
        dashboard_refresh_ms = int(getattr(runtime_cfg, "dashboard_refresh_ms", 500) or 500)
        fast_interval = _runtime_interval("monitoring_fast_snapshot_interval_sec", 0.5, minimum=0.25, maximum=10.0)
        heavy_interval = _runtime_interval("monitoring_heavy_snapshot_interval_sec", 5.0, minimum=1.0, maximum=60.0)
        return JSONResponse(
            {
                "dashboard_refresh_ms": max(250, min(dashboard_refresh_ms, 10_000)),
                "fast_snapshot_interval_sec": fast_interval,
                "heavy_snapshot_interval_sec": heavy_interval,
                "standalone_process": True,
                "embedded_in_all_mode": bool(getattr(monitoring_cfg, "embedded_in_all_mode", False)),
                "snapshot_only": True,
            }
        )

    @app.get("/api/execution/fast")
    async def api_execution_fast(_: None = Depends(_auth_dependency)) -> JSONResponse:
        # Backward-friendly alias for clients that only need open positions/PnL/price freshness.
        return JSONResponse(_fast_runtime_payload())

    @app.get("/api/events")
    async def api_events(limit: int = 200, _: None = Depends(_auth_dependency)) -> JSONResponse:
        bounded_limit = max(1, min(int(limit), 500))
        return JSONResponse({"events": publisher.events(limit=bounded_limit)})

    @app.get("/api/control")
    async def api_control(_: None = Depends(_auth_dependency)) -> JSONResponse:
        payload = control_store.summary()
        publisher.update_control_plane(payload)
        return JSONResponse(payload)

    @app.post("/api/control/pause")
    async def api_control_pause(payload: PauseRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        state = control_store.set_pause(paused=payload.paused, pause_reason=payload.reason, updated_by=payload.updated_by)
        publisher.update_control_plane(state)
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        publisher.publish_event(
            level="WARNING" if state["paused"] else "INFO",
            component="control_plane",
            category="pause_state_changed",
            message="Trading paused" if state["paused"] else "Trading resumed",
            data=state,
            alert=bool(state["paused"]),
        )
        return JSONResponse({"ok": True, "control_plane": state})

    @app.post("/api/control/circuit/reset")
    async def api_control_circuit_reset(payload: CircuitResetRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        circuit = control_store.reset_circuit_breaker(updated_by=payload.updated_by, reason=payload.reason)
        publisher.update_control_plane(control_store.summary())
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        publisher.publish_event(
            level="WARNING",
            component="control_plane",
            category="circuit_breaker_reset",
            message="Circuit breaker reset",
            data={"reason": payload.reason, "updated_by": payload.updated_by},
            alert=False,
        )
        return JSONResponse({"ok": True, "circuit_breaker": circuit})

    @app.get("/api/decision-journal")
    async def api_decision_journal(limit: int = 100, symbol: str | None = None, _: None = Depends(_auth_dependency)) -> JSONResponse:
        bounded_limit = max(1, min(int(limit), 500))
        rows = _recent_decision_journal(paths, bounded_limit * 3)
        if symbol:
            rows = [row for row in rows if str(row.get("symbol", "")).upper() == str(symbol).upper()]
        return JSONResponse({"entries": rows[:bounded_limit]})

    @app.get("/api/decision-lineage")
    async def api_decision_lineage(limit: int = 100, symbol: str | None = None, _: None = Depends(_auth_dependency)) -> JSONResponse:
        bounded_limit = max(1, min(int(limit), 1000))
        rows = control_store.summary().get("decision_lineage_preview", [])
        if symbol:
            rows = [row for row in rows if str(row.get("symbol", "")).upper() == str(symbol).upper()]
        return JSONResponse({"entries": rows[:bounded_limit]})

    @app.get("/api/shadow-candidates")
    async def api_shadow_candidates(limit: int = 100, symbol: str | None = None, _: None = Depends(_auth_dependency)) -> JSONResponse:
        bounded_limit = max(1, min(int(limit), 1000))
        rows = control_store.summary().get("shadow_candidate_preview", [])
        if symbol:
            rows = [row for row in rows if str(row.get("symbol", "")).upper() == str(symbol).upper()]
        return JSONResponse({"entries": rows[:bounded_limit]})

    @app.get("/api/runtime-observability")
    async def api_runtime_observability(_: None = Depends(_auth_dependency)) -> JSONResponse:
        control = control_store.summary()
        payload = {
            "runtime_observability": control.get("runtime_observability", {}),
            "live_runtime": control.get("live_runtime", {}),
            "runtime_gate": control.get("runtime_gate", {}),
        }
        publisher.update_control_plane(control)
        return JSONResponse(payload)

    @app.get("/api/model-versions")
    async def api_model_versions(_: None = Depends(_auth_dependency)) -> JSONResponse:
        payload = {"model_versions": control_store.list_model_versions()}
        publisher.update_model_versions(payload["model_versions"])
        return JSONResponse(payload)

    @app.post("/api/model-versions/rollback")
    async def api_model_versions_rollback(payload: RollbackRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        result = control_store.rollback_model_version(
            model_type=payload.model_type,
            symbol=payload.symbol,
            interval=payload.interval,
            manifest_path=payload.manifest_path,
            model_role=payload.model_role,
            updated_by=payload.updated_by,
        )
        publisher.update_model_versions(control_store.list_model_versions())
        publisher.update_control_plane(control_store.summary())
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        publisher.publish_event(
            level="WARNING",
            component="control_plane",
            category="model_rollback",
            message=f"Model rollback applied for {payload.model_type} {payload.symbol.upper()} {payload.interval}",
            data=result,
            alert=False,
        )
        return JSONResponse(result)

    @app.get("/api/training-registry")
    async def api_training_registry(_: None = Depends(_auth_dependency)) -> JSONResponse:
        snapshot = publisher.snapshot().get("training_registry", {})
        return JSONResponse(snapshot)

    @app.post("/api/symbols/retry")
    async def retry_symbols(payload: RetryRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        actions = load_symbol_actions(paths)
        requested = {str(s).upper() for s in actions.get("symbols", []) if s}
        requested.update(str(s).upper() for s in payload.symbols if s)
        actions["symbols"] = sorted(requested)
        actions["retry_all_blocked"] = bool(actions.get("retry_all_blocked")) or bool(payload.retry_all_blocked)
        save_symbol_actions(paths, actions)
        control_store.record_action(
            action="symbols_retry",
            updated_by=payload.updated_by,
            message="Queued symbols for retry",
            data={"symbols": actions.get("symbols", []), "retry_all_blocked": actions.get("retry_all_blocked", False)},
        )
        publisher.update_control_plane(control_store.summary())
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        return JSONResponse({"ok": True, "queued_symbols": actions.get("symbols", []), "retry_all_blocked": actions.get("retry_all_blocked", False)})

    @app.post("/api/symbols/block")
    async def block_symbol(payload: SymbolControlRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        state = load_symbol_state(paths)
        entry = symbol_entry(state, payload.symbol)
        entry["status"] = "BLOCKED"
        entry["manually_blocked"] = True
        entry["retry_requested"] = False
        entry["blocked_since_ts"] = now_utc_ts()
        save_symbol_state(paths, state)
        control_store.record_action(
            action="symbol_blocked",
            updated_by=payload.updated_by,
            message=f"Blocked symbol {str(payload.symbol).upper()}",
            data={"symbol": str(payload.symbol).upper()},
        )
        publisher.update_control_plane(control_store.summary())
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        publisher.publish_event(
            level="WARNING",
            component="control_plane",
            category="symbol_blocked",
            message=f"Blocked symbol {str(payload.symbol).upper()}",
            data={"symbol": str(payload.symbol).upper(), "updated_by": payload.updated_by},
            alert=True,
        )
        return JSONResponse({"ok": True, "symbol": str(payload.symbol).upper(), "status": entry["status"]})

    @app.post("/api/symbols/unblock")
    async def unblock_symbol(payload: SymbolControlRequest, _: None = Depends(_auth_dependency)) -> JSONResponse:
        state = load_symbol_state(paths)
        entry = symbol_entry(state, payload.symbol)
        entry["status"] = "ACTIVE"
        entry["manually_blocked"] = False
        entry["retry_requested"] = True
        entry["retry_not_before_ts"] = None
        save_symbol_state(paths, state)
        actions = load_symbol_actions(paths)
        pending = {str(s).upper() for s in actions.get("symbols", []) if s}
        pending.add(str(payload.symbol).upper())
        actions["symbols"] = sorted(pending)
        save_symbol_actions(paths, actions)
        control_store.record_action(
            action="symbol_unblocked",
            updated_by=payload.updated_by,
            message=f"Unblocked symbol {str(payload.symbol).upper()}",
            data={"symbol": str(payload.symbol).upper()},
        )
        publisher.update_control_plane(control_store.summary())
        publisher.append_control_action(control_store.recent_actions(limit=1)[0])
        publisher.publish_event(
            level="INFO",
            component="control_plane",
            category="symbol_unblocked",
            message=f"Unblocked symbol {str(payload.symbol).upper()}",
            data={"symbol": str(payload.symbol).upper(), "updated_by": payload.updated_by},
            alert=False,
        )
        return JSONResponse({"ok": True, "symbol": str(payload.symbol).upper(), "status": entry["status"], "retry_requested": True})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            await websocket_token_required(websocket, paths)
            await hub.add(websocket)
            await websocket.send_json({"type": "snapshot", "payload": _normalize_snapshot(publisher.snapshot(), paths)})
            await websocket.send_json({"type": "events", "payload": publisher.events(limit=100)})
            while True:
                await websocket.receive_text()
        except Exception:
            pass
        finally:
            await hub.remove(websocket)
            with suppress(Exception):
                await websocket.close()

    return app


def run_web_monitor(paths, stop_event: threading.Event | None = None) -> None:
    """Run the monitoring web server until completion or external stop."""
    cfg = paths.config.monitoring
    config = uvicorn.Config(
        create_app(paths),
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.uvicorn_log_level,
        log_config=None,
    )
    server = uvicorn.Server(config)

    watcher: threading.Thread | None = None
    if stop_event is not None:

        def _watch_stop() -> None:
            stop_event.wait()
            server.should_exit = True

        watcher = threading.Thread(target=_watch_stop, name="monitoring-stop-watcher", daemon=True)
        watcher.start()

    server.run()

    if watcher is not None:
        watcher.join(timeout=0.1)


def run_monitoring_server(paths, stop_event: threading.Event | None = None) -> None:
    """Run the monitoring control plane server."""
    run_web_monitor(paths, stop_event=stop_event)
