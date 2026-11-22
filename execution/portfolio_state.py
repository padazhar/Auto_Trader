from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.shared_utils import atomic_write_bytes, load_json


class PositionStateInvariantError(RuntimeError):
    """Raised when an execution would overwrite an authoritative open position."""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and not value.strip():
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _closed_trade_pnl(row: dict[str, Any]) -> float:
    return _safe_float(
        row.get("pnl_usd", row.get("realized_pnl_usd", row.get("pnl", 0.0))),
        0.0,
    )


def _fill_cost_payload(payload: dict[str, Any] | None) -> dict[str, float]:
    """Normalize execution fill cost metadata for paper/live portfolio accounting.

    Fill price already reflects paper slippage/spread where the adapter applies it;
    this helper carries the explicit cost fields into portfolio and reports so net
    PnL is not silently reported as zero-cost PnL.
    """
    raw = dict(payload or {}) if isinstance(payload, dict) else {}
    meta = dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {}
    return {
        "fee_usd": _safe_float(raw.get("fee_usd", meta.get("fee_usd", 0.0)), 0.0),
        "estimated_slippage_usd": _safe_float(raw.get("slippage_usd", meta.get("estimated_slippage_usd", 0.0)), 0.0),
        "reference_price": _safe_float(raw.get("reference_price", meta.get("reference_price", 0.0)), 0.0),
        "effective_slippage_bps": _safe_float(raw.get("effective_slippage_bps", meta.get("effective_slippage_bps", 0.0)), 0.0),
        "latency_bps": _safe_float(raw.get("latency_bps", meta.get("latency_bps", 0.0)), 0.0),
        "spread_bps": _safe_float(raw.get("spread_bps", meta.get("spread_bps", 0.0)), 0.0),
    }



def build_trade_excursion_attribution(*, trade: dict[str, Any], candles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a read-only MAE/MFE payload for a closed trade.

    Step 7 is forensic/reporting-only. The helper consumes a closed trade and
    already persisted candles, then reports excursions without changing stops,
    risk sizing, execution, or PnL. Missing inputs are reported explicitly.
    """
    trade_payload = dict(trade or {})
    symbol = str(trade_payload.get("symbol") or "").upper()
    side = str(trade_payload.get("side") or "").upper()
    entry_price = _safe_float(trade_payload.get("entry_price"), 0.0)
    exit_price = _safe_float(trade_payload.get("exit_price"), 0.0)
    pnl = _safe_float(trade_payload.get("pnl_usd"), 0.0)
    risk_plan = trade_payload.get("risk_plan") if isinstance(trade_payload.get("risk_plan"), dict) else {}
    close_event = trade_payload.get("close_event") if isinstance(trade_payload.get("close_event"), dict) else {}
    stop_loss = _safe_float(trade_payload.get("stop_loss", risk_plan.get("stop_loss", close_event.get("stop_loss", 0.0))), 0.0)
    take_profit = _safe_float(trade_payload.get("take_profit", risk_plan.get("take_profit", close_event.get("take_profit", 0.0))), 0.0)
    rows = [dict(row) for row in (candles or []) if isinstance(row, dict)]
    result: dict[str, Any] = {
        "schema_version": "trade_excursion_attribution.v1",
        "symbol": symbol,
        "side": side,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "pnl_usd": float(pnl),
        "exit_reason": str(trade_payload.get("exit_reason") or ""),
        "candle_count": len(rows),
        "analysis_status": "ok",
        "analysis_reason": "",
        "max_favorable_excursion_usd": 0.0,
        "max_adverse_excursion_usd": 0.0,
        "max_favorable_excursion_bps": 0.0,
        "max_adverse_excursion_bps": 0.0,
        "max_high": 0.0,
        "min_low": 0.0,
        "take_profit_reachable": None,
        "stop_loss_reachable": None,
        "sl_too_tight_candidate": None,
        "tp_was_reachable_candidate": None,
        "directionally_wrong_candidate": None,
    }
    if entry_price <= 0.0:
        result["analysis_status"] = "missing_entry_price"
        result["analysis_reason"] = "entry_price_missing_or_non_positive"
        return result
    if side not in {"LONG", "SHORT"}:
        result["analysis_status"] = "unsupported_side"
        result["analysis_reason"] = "side_must_be_LONG_or_SHORT"
        return result
    if not rows:
        result["analysis_status"] = "missing_candles"
        result["analysis_reason"] = "no_market_candles_found_for_trade_window"
        result["directionally_wrong_candidate"] = bool(pnl < 0.0)
        return result

    highs = [_safe_float(row.get("high"), 0.0) for row in rows]
    lows = [_safe_float(row.get("low"), 0.0) for row in rows]
    highs = [value for value in highs if value > 0.0]
    lows = [value for value in lows if value > 0.0]
    if not highs or not lows:
        result["analysis_status"] = "invalid_candles"
        result["analysis_reason"] = "candle_high_low_missing_or_non_positive"
        return result

    max_high = max(highs)
    min_low = min(lows)
    if side == "LONG":
        mfe = max(0.0, max_high - entry_price)
        mae = max(0.0, entry_price - min_low)
        tp_reachable = bool(take_profit > 0.0 and max_high >= take_profit)
        sl_reachable = bool(stop_loss > 0.0 and min_low <= stop_loss)
    else:
        mfe = max(0.0, entry_price - min_low)
        mae = max(0.0, max_high - entry_price)
        tp_reachable = bool(take_profit > 0.0 and min_low <= take_profit)
        sl_reachable = bool(stop_loss > 0.0 and max_high >= stop_loss)

    exit_reason = str(trade_payload.get("exit_reason") or "").upper()
    result.update({
        "max_favorable_excursion_usd": float(mfe),
        "max_adverse_excursion_usd": float(mae),
        "max_favorable_excursion_bps": float((mfe / entry_price) * 10000.0),
        "max_adverse_excursion_bps": float((mae / entry_price) * 10000.0),
        "max_high": float(max_high),
        "min_low": float(min_low),
        "take_profit_reachable": tp_reachable if take_profit > 0.0 else None,
        "stop_loss_reachable": sl_reachable if stop_loss > 0.0 else None,
        "sl_too_tight_candidate": bool(exit_reason == "SL_HIT" and mfe > 0.0),
        "tp_was_reachable_candidate": bool(tp_reachable) if take_profit > 0.0 else None,
        "directionally_wrong_candidate": bool(pnl < 0.0 and mae >= mfe),
    })
    return result


def build_closed_trade_attribution(*, trade: dict[str, Any], lineage: dict[str, Any] | None = None, risk_plan: dict[str, Any] | None = None, close_event: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic closed-trade attribution payload from persisted fields.

    Step 5 is observability-only: this helper never changes execution decisions,
    order routing, sizing, or PnL. It normalizes the lineage already carried by
    the entry decision into a report-friendly contract so every closed trade can
    be linked back to its decision, model candidates, risk approval, and exit.
    Missing values are preserved as ``unknown`` instead of invented.
    """
    lineage_payload = dict(lineage or {})
    risk_payload = dict(risk_plan or {})
    close_payload = dict(close_event or {})
    trade_payload = dict(trade or {})
    model_candidates = lineage_payload.get("model_candidates") if isinstance(lineage_payload.get("model_candidates"), dict) else {}
    prediction_vector = lineage_payload.get("prediction_vector") if isinstance(lineage_payload.get("prediction_vector"), dict) else {}
    entry_action = str(lineage_payload.get("signal_action") or lineage_payload.get("entry_action") or trade_payload.get("side") or "unknown")
    exit_reason = str(trade_payload.get("exit_reason") or close_payload.get("exit_reason") or "unknown")
    expected_side = str(lineage_payload.get("expected_side") or ("LONG" if "LONG" in entry_action.upper() else ("SHORT" if "SHORT" in entry_action.upper() else trade_payload.get("side") or "unknown"))).upper()
    pnl = _safe_float(trade_payload.get("pnl_usd"), 0.0)
    if expected_side in {"LONG", "SHORT"}:
        matched = pnl > 0.0
    else:
        matched = None
    return {
        "schema_version": "closed_trade_attribution.v1",
        "decision_id": str(lineage_payload.get("decision_id") or trade_payload.get("decision_id") or "unknown"),
        "symbol": str(trade_payload.get("symbol") or lineage_payload.get("symbol") or "").upper(),
        "side": str(trade_payload.get("side") or expected_side or "unknown").upper(),
        "entry_action": entry_action,
        "entry_reason": str(lineage_payload.get("signal_reason") or lineage_payload.get("entry_reason") or "unknown"),
        "ev_score": _safe_float(lineage_payload.get("ev_score", lineage_payload.get("score", 0.0)), 0.0),
        "risk_approved": bool(lineage_payload.get("risk_approved", True if risk_payload else False)),
        "risk_reason": str(lineage_payload.get("risk_reason") or risk_payload.get("reason") or ""),
        "risk_plan_version": str(risk_payload.get("contract_version") or ""),
        "rf_candidate_id": str(model_candidates.get("rf") or lineage_payload.get("rf_candidate_id") or ""),
        "gbm_candidate_id": str(model_candidates.get("gbm") or lineage_payload.get("gbm_candidate_id") or ""),
        "policy_candidate_id": str(model_candidates.get("policy") or model_candidates.get("dqn") or lineage_payload.get("policy_candidate_id") or ""),
        "prediction_vector": prediction_vector,
        "entry_price": _safe_float(trade_payload.get("entry_price"), 0.0),
        "exit_price": _safe_float(trade_payload.get("exit_price"), 0.0),
        "stop_loss": _safe_float(risk_payload.get("stop_loss", close_payload.get("stop_loss", 0.0)), 0.0),
        "take_profit": _safe_float(risk_payload.get("take_profit", close_payload.get("take_profit", 0.0)), 0.0),
        "pnl_usd": pnl,
        "exit_reason": exit_reason,
        "close_trigger": str(lineage_payload.get("close_trigger") or close_payload.get("trigger") or "unknown"),
        "prediction_matched_outcome": matched,
    }


def summarize_closed_trade_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the canonical closed-trade aggregate used by Execution UI/reporting.

    The aggregate is intentionally small and source-neutral so CSV, runtime
    journals, and portfolio snapshots can be compared without changing trading
    behavior. Break-even trades count as completed trades, but not as wins or
    losses.
    """
    closed_rows = [row for row in rows if isinstance(row, dict)]
    completed_count = len(closed_rows)
    wins = sum(1 for row in closed_rows if _closed_trade_pnl(row) > 0.0)
    losses = sum(1 for row in closed_rows if _closed_trade_pnl(row) < 0.0)
    flat = max(0, completed_count - wins - losses)
    total_pnl = sum(_closed_trade_pnl(row) for row in closed_rows)
    gross_pnl = sum(_safe_float(row.get("gross_pnl_usd"), _closed_trade_pnl(row)) for row in closed_rows)
    total_fees = sum(_safe_float(row.get("fees_usd"), 0.0) for row in closed_rows)
    total_slippage = sum(_safe_float(row.get("slippage_usd"), 0.0) for row in closed_rows)
    return {
        "completed_count": completed_count,
        "trades": completed_count,
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "total_pnl": float(total_pnl),
        "realized_pnl_usd": float(total_pnl),
        "gross_pnl_usd": float(gross_pnl),
        "fees_usd": float(total_fees),
        "slippage_usd": float(total_slippage),
        "win_rate": (float(wins) / float(completed_count)) if completed_count > 0 else 0.0,
    }


def normalize_closed_trade_row(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").upper().strip()
    side = str(row.get("side") or row.get("action") or "").upper().strip()
    return {
        "symbol": symbol,
        "side": side,
        "mode": str(row.get("mode") or row.get("authority_mode") or "").strip(),
        "entry_time": row.get("entry_time") or row.get("opened_at") or row.get("entry_ts"),
        "exit_time": row.get("exit_time") or row.get("closed_at") or row.get("exit_ts") or row.get("ts"),
        "entry_price": _safe_float(row.get("entry_price"), 0.0),
        "exit_price": _safe_float(row.get("exit_price") or row.get("close_price"), 0.0),
        "pnl_usd": _closed_trade_pnl(row),
        "pnl_pct": _safe_float(row.get("pnl_pct"), 0.0),
        "notional_usd": _safe_float(row.get("notional_usd") or row.get("amount"), 0.0),
        "exit_reason": str(row.get("exit_reason") or row.get("reason") or "").strip(),
        "gross_pnl_usd": _safe_float(row.get("gross_pnl_usd"), _closed_trade_pnl(row)),
        "fees_usd": _safe_float(row.get("fees_usd"), 0.0),
        "slippage_usd": _safe_float(row.get("slippage_usd"), 0.0),
        "latency_bps": _safe_float(row.get("latency_bps"), 0.0),
        "attribution": dict(row.get("attribution") or {}) if isinstance(row.get("attribution"), dict) else {},
        "source": source,
    }


def load_closed_trade_csv(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the historical-backtest closed-trade CSV for separate reporting only.

    The CSV is not an authoritative paper-execution ledger and must never be
    projected into paper PnL, paper trade history, or paper-promotion evidence.
    """
    csv_path = Path(path)
    metadata: dict[str, Any] = {
        "path": str(csv_path),
        "exists": csv_path.exists(),
        "source": "historical_backtest_ledger",
        "authoritative_for_paper_execution": False,
        "row_count": 0,
        "usable": False,
        "reason": "missing",
    }
    if not csv_path.exists():
        return [], metadata
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            metadata["columns"] = fieldnames
            rows = [normalize_closed_trade_row(row, source="historical_backtest_ledger") for row in reader if isinstance(row, dict)]
    except Exception as exc:
        metadata["reason"] = f"read_error:{exc.__class__.__name__}"
        return [], metadata
    rows = [row for row in rows if str(row.get("symbol") or "").strip()]
    metadata["row_count"] = len(rows)
    metadata["usable"] = bool(rows)
    metadata["reason"] = "loaded" if rows else "header_only"
    return rows, metadata


def execution_trade_source_from_paths(paths: Any) -> dict[str, Any]:
    """Return the separately labelled historical-backtest ledger, when present.

    This helper remains available for backtest/report surfaces only. Paper
    execution callers must use :func:`paper_execution_trade_source_from_paths`.
    """
    project_root = Path(getattr(paths, "project_root", Path.cwd()))
    csv_path = Path(getattr(paths, "live_backtest_trades_path", project_root / "data" / "live_backtest_trades.csv"))
    csv_rows, csv_meta = load_closed_trade_csv(csv_path)
    return {
        "selected_source": "historical_backtest_ledger" if csv_rows else "historical_backtest_unavailable",
        "authoritative_for_paper_execution": False,
        "csv": csv_meta,
        "rows": csv_rows,
        "summary": summarize_closed_trade_rows(csv_rows) if csv_rows else {},
    }


def paper_execution_trade_source_from_paths(paths: Any) -> dict[str, Any]:
    """Return the authoritative persisted paper-execution outcome ledger.

    Paper PnL and paper-promotion evidence are derived only from the portfolio
    state written by the execution authority. Historical/backtest ledgers are
    intentionally excluded from this contract.
    """
    project_root = Path(getattr(paths, "project_root", Path.cwd()))
    portfolio_path = Path(getattr(paths, "portfolio_state_path", project_root / "data" / "runtime" / "portfolio_state.json"))
    metadata: dict[str, Any] = {
        "path": str(portfolio_path),
        "exists": portfolio_path.exists(),
        "source": "paper_execution_ledger",
        "authoritative_for_paper_execution": True,
        "row_count": 0,
        "usable": False,
        "reason": "missing",
    }
    if not portfolio_path.exists():
        return {"selected_source": "runtime_execution_payload", "authoritative_for_paper_execution": True, "portfolio": metadata, "rows": [], "summary": {}}
    payload = load_json(portfolio_path) or {}
    if not isinstance(payload, dict):
        metadata["reason"] = "unreadable"
        return {"selected_source": "runtime_execution_payload", "authoritative_for_paper_execution": True, "portfolio": metadata, "rows": [], "summary": {}}
    raw_rows = payload.get("closed_trades", [])
    if not isinstance(raw_rows, list):
        raw_rows = []
    rows = [normalize_closed_trade_row(row, source="paper_execution_ledger") for row in raw_rows if isinstance(row, dict)]
    rows = [row for row in rows if str(row.get("symbol") or "").strip()]
    summary = summarize_closed_trade_rows(rows)
    summary["source"] = "paper_execution_ledger"
    state_realized = payload.get("realized_pnl_usd")
    if state_realized not in (None, ""):
        state_realized_f = _safe_float(state_realized, summary.get("realized_pnl_usd", 0.0))
        metadata["state_realized_pnl_usd"] = state_realized_f
        metadata["closed_trade_pnl_usd"] = _safe_float(summary.get("realized_pnl_usd"), 0.0)
        metadata["pnl_matches_state"] = abs(state_realized_f - metadata["closed_trade_pnl_usd"]) < 1e-9
    metadata["row_count"] = len(rows)
    metadata["usable"] = True
    metadata["reason"] = "loaded"
    return {
        "selected_source": "paper_execution_ledger",
        "authoritative_for_paper_execution": True,
        "portfolio": metadata,
        "rows": rows,
        "summary": summary,
    }


def apply_execution_trade_source(runtime: dict[str, Any], *, paths: Any, history_limit: int = 120) -> dict[str, Any]:
    """Project authoritative paper-execution outcomes onto a paper runtime payload.

    Historical backtest CSV results remain visible only as explicitly
    non-authoritative reference metadata. They can never replace paper PnL or
    paper trade history in monitoring/API surfaces.
    """
    normalized = dict(runtime or {})
    paper_source = paper_execution_trade_source_from_paths(paths)
    backtest_source = execution_trade_source_from_paths(paths)
    paper_metadata = {k: v for k, v in paper_source.items() if k != "rows"}
    paper_metadata["backtest_reference"] = {k: v for k, v in backtest_source.items() if k not in {"rows", "summary"}}
    normalized["trade_result_source"] = paper_metadata

    portfolio_meta = paper_source.get("portfolio") if isinstance(paper_source.get("portfolio"), dict) else {}
    if bool(portfolio_meta.get("exists", False)) and str(portfolio_meta.get("reason")) == "loaded":
        rows = list(paper_source.get("rows") or [])
        existing_summary = dict(normalized.get("summary") or {})
        paper_summary = dict(paper_source.get("summary") or {})
        open_count = _safe_int(
            existing_summary.get("open_positions", existing_summary.get("open_count", normalized.get("open_position_count", len(normalized.get("open_positions") or [])))),
            len(normalized.get("open_positions") or []),
        )
        paper_summary["open_positions"] = open_count
        paper_summary["open_count"] = open_count
        normalized["summary"] = paper_summary
        normalized["trade_history"] = rows[-max(1, int(history_limit)):]
        normalized["trade_history_source"] = "paper_execution_ledger"
        return normalized

    existing_summary = dict(normalized.get("summary") or {})
    if str(existing_summary.get("source") or "") in {"csv_closed_trade_ledger", "historical_backtest_ledger"}:
        existing_summary["source"] = "runtime_execution_payload"
    else:
        existing_summary.setdefault("source", "runtime_execution_payload")
    normalized["summary"] = existing_summary
    normalized["trade_history_source"] = str(normalized.get("trade_history_source") or "runtime_execution_payload")
    return normalized


@dataclass(slots=True)
class PositionState:
    """Persisted open-position state used by the runtime risk layer."""

    symbol: str
    side: str
    quantity: float
    notional_usd: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: float
    updated_at: float
    unrealized_pnl_usd: float = 0.0
    entry_fee_usd: float = 0.0
    entry_slippage_usd: float = 0.0
    entry_reference_price: float = 0.0
    entry_effective_slippage_bps: float = 0.0
    latency_bps: float = 0.0
    spread_bps: float = 0.0
    lineage: dict[str, Any] = field(default_factory=dict)
    risk_plan: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClosedTradeState:
    """Persisted closed-trade state produced by the paper/live execution authority."""

    symbol: str
    side: str
    quantity: float
    notional_usd: float
    entry_price: float
    exit_price: float
    opened_at: float
    closed_at: float
    pnl_usd: float
    exit_reason: str = ""
    mode: str = "paper"
    gross_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
    latency_bps: float = 0.0
    spread_bps: float = 0.0
    entry_reference_price: float = 0.0
    exit_reference_price: float = 0.0
    lineage: dict[str, Any] = field(default_factory=dict)
    risk_plan: dict[str, Any] = field(default_factory=dict)
    close_event: dict[str, Any] = field(default_factory=dict)
    attribution: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PortfolioState:
    """Persistent portfolio snapshot for risk and restart recovery."""

    positions: dict[str, PositionState] = field(default_factory=dict)
    closed_trades: list[ClosedTradeState] = field(default_factory=list)
    realized_pnl_usd: float = 0.0
    daily_realized_pnl_usd: float = 0.0
    last_realized_reset_day: str = ""
    last_equity_usd: float = 0.0
    peak_equity_usd: float = 0.0
    updated_at: float = 0.0
    authority_mode: str = "paper_local"
    reconciliation_status: str = "paper_local"
    last_reconciled_at: float | None = None
    exchange_position_count: int = 0
    reconciliation_mismatch_count: int = 0
    last_reconciliation_error: str = ""
    exchange_snapshot_hash: str = ""
    execution_integrity_status: str = "ok"
    execution_integrity_error: str = ""
    execution_integrity_failure_at: float | None = None


class PortfolioStateStore:
    """Load, persist, and mutate portfolio state for the runtime execution layer."""

    def __init__(self, path, *, starting_equity_usd: float) -> None:
        self.path = path
        self.starting_equity_usd = float(starting_equity_usd)

    def close_blocked_positions(
        self,
        state: PortfolioState,
        *,
        blocked_symbols: list[str],
        price_by_symbol: dict[str, float],
        ts: float,
        execution_mode: str = "paper",
    ) -> dict[str, Any]:
        """Close blocked-symbol positions in paper; fail closed in live."""
        normalized = [str(symbol or "").upper() for symbol in blocked_symbols if str(symbol or "").strip()]
        targets = [symbol for symbol in normalized if symbol in state.positions]
        report: dict[str, Any] = {
            "requested": normalized,
            "targets": targets,
            "closed": [],
            "failed": [],
            "skipped": [],
            "execution_mode": str(execution_mode or "paper").lower(),
            "fail_closed": False,
        }
        if not targets:
            return report

        live_mode = str(execution_mode or "paper").lower() == "live" or not str(state.authority_mode or "paper_local").startswith("paper")
        if live_mode:
            report["fail_closed"] = True
            report["failed"] = [
                {"symbol": symbol, "reason": "live_blocked_symbol_close_requires_exchange_reconciliation"}
                for symbol in targets
            ]
            return report

        for symbol in targets:
            try:
                price = float(price_by_symbol.get(symbol) or 0.0)
            except Exception:
                price = 0.0
            if price <= 0.0:
                report["failed"].append({"symbol": symbol, "reason": "missing_mark_price"})
                continue
            pnl = self._close_position(
                state,
                symbol=symbol,
                close_price=price,
                ts=float(ts),
                exit_reason="blocked_symbol_forced_close",
            )
            report["closed"].append({"symbol": symbol, "price": price, "pnl_usd": float(pnl)})
        return report

    @staticmethod
    def _day_key(ts: float) -> str:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _remove_closed_position_conflicts(
        positions: dict[str, PositionState],
        closed_trades: list[ClosedTradeState],
    ) -> dict[str, PositionState]:
        """Remove stale open-position rows that already have a matching close record.

        This restart-recovery guard preserves the execution authority contract:
        if a symbol has a closed-trade row at or after the open timestamp, that
        symbol must not be resurrected as open when portfolio_state.json is loaded.
        """
        if not positions or not closed_trades:
            return positions

        latest_close_by_symbol: dict[str, float] = {}
        for trade in closed_trades:
            symbol = str(getattr(trade, "symbol", "") or "").upper()
            if not symbol:
                continue
            closed_at = float(getattr(trade, "closed_at", 0.0) or 0.0)
            latest_close_by_symbol[symbol] = max(latest_close_by_symbol.get(symbol, 0.0), closed_at)

        cleaned: dict[str, PositionState] = {}
        for symbol, position in positions.items():
            normalized_symbol = str(symbol or getattr(position, "symbol", "") or "").upper()
            opened_at = float(getattr(position, "opened_at", 0.0) or 0.0)
            latest_close = float(latest_close_by_symbol.get(normalized_symbol, 0.0) or 0.0)
            if latest_close > 0.0 and latest_close >= opened_at:
                continue
            cleaned[normalized_symbol] = position
        return cleaned


    def reconcile_closed_position_conflicts(self, state: PortfolioState) -> PortfolioState:
        """Enforce the canonical ledger invariant before publishing or saving.

        A symbol cannot be both open and closed when the latest closed-trade
        timestamp is at or after the position open timestamp. This keeps closed
        trades from being resurrected as open positions after restart, UI refresh,
        or legacy-state replay.
        """
        state.positions = self._remove_closed_position_conflicts(state.positions, state.closed_trades)
        return state

    def open_position_snapshot(self, state: PortfolioState) -> dict[str, dict[str, Any]]:
        """Return a legacy-compatible open-position snapshot from portfolio authority.

        Signal generation previously inspected backtest_open_trades.json to infer
        whether a symbol was already long or short. The portfolio state is now
        the paper/live execution authority, so this adapter provides the same
        minimal shape from canonical positions without letting stale backtest
        state influence runtime decisions.
        """
        self.reconcile_closed_position_conflicts(state)
        snapshot: dict[str, dict[str, Any]] = {}
        for symbol, position in state.positions.items():
            normalized = str(symbol or position.symbol).upper()
            if not normalized:
                continue
            snapshot[normalized] = {
                "symbol": normalized,
                "side": str(position.side or "").upper(),
                "entry_price": float(position.entry_price),
                "notional_usd": float(position.notional_usd),
                "opened_at": float(position.opened_at),
                "updated_at": float(position.updated_at),
                "source": "portfolio_state",
                "state_owner": "live_trading_controller",
                "execution_surface": "paper_forward_execution" if state.authority_mode == "paper_local" else state.authority_mode,
            }
        return snapshot

    def load(self) -> PortfolioState:
        payload = load_json(self.path, default={}) or {}
        if not isinstance(payload, dict):
            payload = {}
        positions_payload = payload.get("positions", {})
        positions: dict[str, PositionState] = {}
        if isinstance(positions_payload, dict):
            for symbol, raw in positions_payload.items():
                if not isinstance(raw, dict):
                    continue
                positions[str(symbol).upper()] = PositionState(
                    symbol=str(raw.get("symbol") or symbol).upper(),
                    side=str(raw.get("side") or "").upper(),
                    quantity=float(raw.get("quantity") or 0.0),
                    notional_usd=float(raw.get("notional_usd") or 0.0),
                    entry_price=float(raw.get("entry_price") or 0.0),
                    stop_loss=float(raw.get("stop_loss") or 0.0),
                    take_profit=float(raw.get("take_profit") or 0.0),
                    opened_at=float(raw.get("opened_at") or 0.0),
                    updated_at=float(raw.get("updated_at") or 0.0),
                    unrealized_pnl_usd=float(raw.get("unrealized_pnl_usd") or 0.0),
                    entry_fee_usd=float(raw.get("entry_fee_usd") or 0.0),
                    entry_slippage_usd=float(raw.get("entry_slippage_usd") or 0.0),
                    entry_reference_price=float(raw.get("entry_reference_price") or 0.0),
                    entry_effective_slippage_bps=float(raw.get("entry_effective_slippage_bps") or 0.0),
                    latency_bps=float(raw.get("latency_bps") or 0.0),
                    spread_bps=float(raw.get("spread_bps") or 0.0),
                    lineage=dict(raw.get("lineage") or {}),
                    risk_plan=dict(raw.get("risk_plan") or {}),
                )
        closed_trades: list[ClosedTradeState] = []
        closed_payload = payload.get("closed_trades", payload.get("trade_history", []))
        if isinstance(closed_payload, list):
            for raw in closed_payload:
                if not isinstance(raw, dict):
                    continue
                try:
                    closed_trades.append(
                        ClosedTradeState(
                            symbol=str(raw.get("symbol") or "").upper(),
                            side=str(raw.get("side") or raw.get("action") or "").upper(),
                            quantity=float(raw.get("quantity") or 0.0),
                            notional_usd=float(raw.get("notional_usd") or raw.get("amount") or 0.0),
                            entry_price=float(raw.get("entry_price") or 0.0),
                            exit_price=float(raw.get("exit_price") or raw.get("close_price") or 0.0),
                            opened_at=float(raw.get("opened_at") or raw.get("entry_ts") or 0.0),
                            closed_at=float(raw.get("closed_at") or raw.get("exit_ts") or raw.get("ts") or 0.0),
                            pnl_usd=float(raw.get("pnl_usd") or raw.get("realized_pnl_usd") or 0.0),
                            exit_reason=str(raw.get("exit_reason") or raw.get("reason") or ""),
                            mode=str(raw.get("mode") or "paper"),
                            gross_pnl_usd=float(raw.get("gross_pnl_usd") or raw.get("pnl_usd") or raw.get("realized_pnl_usd") or 0.0),
                            fees_usd=float(raw.get("fees_usd") or 0.0),
                            slippage_usd=float(raw.get("slippage_usd") or 0.0),
                            latency_bps=float(raw.get("latency_bps") or 0.0),
                            spread_bps=float(raw.get("spread_bps") or 0.0),
                            entry_reference_price=float(raw.get("entry_reference_price") or 0.0),
                            exit_reference_price=float(raw.get("exit_reference_price") or 0.0),
                            lineage=dict(raw.get("lineage") or {}),
                            risk_plan=dict(raw.get("risk_plan") or {}),
                            close_event=dict(raw.get("close_event") or {}),
                            attribution=dict(raw.get("attribution") or {}),
                        )
                    )
                except Exception:
                    continue

        positions = self._remove_closed_position_conflicts(positions, closed_trades)

        state = PortfolioState(
            positions=positions,
            closed_trades=closed_trades,
            realized_pnl_usd=float(payload.get("realized_pnl_usd") or 0.0),
            daily_realized_pnl_usd=float(payload.get("daily_realized_pnl_usd") or 0.0),
            last_realized_reset_day=str(payload.get("last_realized_reset_day") or ""),
            last_equity_usd=float(payload.get("last_equity_usd") or self.starting_equity_usd),
            peak_equity_usd=float(payload.get("peak_equity_usd") or self.starting_equity_usd),
            updated_at=float(payload.get("updated_at") or 0.0),
            authority_mode=str(payload.get("authority_mode") or "paper_local"),
            reconciliation_status=str(payload.get("reconciliation_status") or "paper_local"),
            last_reconciled_at=(float(payload["last_reconciled_at"]) if payload.get("last_reconciled_at") is not None else None),
            exchange_position_count=int(payload.get("exchange_position_count") or 0),
            reconciliation_mismatch_count=int(payload.get("reconciliation_mismatch_count") or 0),
            last_reconciliation_error=str(payload.get("last_reconciliation_error") or ""),
            exchange_snapshot_hash=str(payload.get("exchange_snapshot_hash") or ""),
            execution_integrity_status=str(payload.get("execution_integrity_status") or "ok"),
            execution_integrity_error=str(payload.get("execution_integrity_error") or ""),
            execution_integrity_failure_at=(float(payload["execution_integrity_failure_at"]) if payload.get("execution_integrity_failure_at") is not None else None),
        )
        if state.last_equity_usd <= 0:
            state.last_equity_usd = self.starting_equity_usd
        if state.peak_equity_usd <= 0:
            state.peak_equity_usd = max(state.last_equity_usd, self.starting_equity_usd)
        return state

    def save(self, state: PortfolioState) -> None:
        self.reconcile_closed_position_conflicts(state)
        payload = {
            "state_domain": "execution_portfolio",
            "state_owner": "live_trading_controller",
            "positions": {symbol: asdict(position) for symbol, position in state.positions.items()},
            "closed_trades": [asdict(trade) for trade in state.closed_trades],
            "realized_pnl_usd": float(state.realized_pnl_usd),
            "daily_realized_pnl_usd": float(state.daily_realized_pnl_usd),
            "last_realized_reset_day": state.last_realized_reset_day,
            "last_equity_usd": float(state.last_equity_usd),
            "peak_equity_usd": float(state.peak_equity_usd),
            "updated_at": float(state.updated_at),
            "authority_mode": str(state.authority_mode or "paper_local"),
            "reconciliation_status": str(state.reconciliation_status or "paper_local"),
            "last_reconciled_at": state.last_reconciled_at,
            "exchange_position_count": int(state.exchange_position_count),
            "reconciliation_mismatch_count": int(state.reconciliation_mismatch_count),
            "last_reconciliation_error": str(state.last_reconciliation_error or ""),
            "exchange_snapshot_hash": str(state.exchange_snapshot_hash or ""),
            "execution_integrity_status": str(state.execution_integrity_status or "ok"),
            "execution_integrity_error": str(state.execution_integrity_error or ""),
            "execution_integrity_failure_at": state.execution_integrity_failure_at,
        }
        atomic_write_bytes(self.path, __import__("json").dumps(payload, indent=2, sort_keys=True).encode("utf-8"))

    def record_execution_integrity_failure(self, state: PortfolioState, *, reason: str, ts: float) -> None:
        """Persist a fail-closed state when a filled execution cannot be represented safely."""
        state.execution_integrity_status = "blocked"
        state.execution_integrity_error = str(reason or "execution_state_integrity_failure")
        state.execution_integrity_failure_at = float(ts)
        state.updated_at = float(ts)

    def reset_daily_if_needed(self, state: PortfolioState, *, ts: float) -> None:
        day_key = self._day_key(ts)
        if state.last_realized_reset_day != day_key:
            state.last_realized_reset_day = day_key
            state.daily_realized_pnl_usd = 0.0

    def mark_authority_mode(self, state: PortfolioState, *, execution_mode: str, reconciled: bool) -> None:
        """Record whether the local portfolio is authoritative for the active mode."""
        mode = str(execution_mode or "paper").strip().lower()
        if mode == "paper":
            state.authority_mode = "paper_local"
            state.reconciliation_status = "paper_local"
            state.last_reconciled_at = None
            return
        state.authority_mode = "live_exchange_required"
        if reconciled:
            state.reconciliation_status = "reconciled"
            state.last_reconciled_at = float(datetime.now(timezone.utc).timestamp())
        else:
            if state.reconciliation_status == "paper_local":
                state.reconciliation_status = "unreconciled"
            state.last_reconciled_at = None

    def record_reconciliation(self, state: PortfolioState, result, *, ts: float) -> None:
        """Persist reconciliation output into portfolio state."""
        state.updated_at = float(ts)
        state.exchange_position_count = int(getattr(result, "exchange_position_count", 0) or 0)
        state.reconciliation_mismatch_count = int(getattr(result, "mismatch_count", 0) or 0)
        state.last_reconciliation_error = str(getattr(result, "error", "") or "")
        state.exchange_snapshot_hash = str(getattr(result, "snapshot_hash", "") or "")

        status = str(getattr(result, "status", "") or "")
        details = getattr(result, "details", None) or {}
        exchange_positions = details.get("exchange_positions", []) if isinstance(details, dict) else []

        state.authority_mode = "live_exchange_required"
        if bool(getattr(result, "success", False)) and bool(getattr(result, "reconciled", False)):
            normalized_positions: dict[str, PositionState] = {}
            for row in exchange_positions:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "").upper()
                side = str(row.get("side") or "").upper()
                quantity = float(row.get("quantity") or 0.0)
                if not symbol or side == "FLAT" or quantity <= 0.0:
                    continue
                existing = state.positions.get(symbol)
                normalized_positions[symbol] = PositionState(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    notional_usd=float(row.get("notional_usd") or 0.0),
                    entry_price=float(row.get("entry_price") or 0.0),
                    stop_loss=float(getattr(existing, "stop_loss", 0.0) or 0.0),
                    take_profit=float(getattr(existing, "take_profit", 0.0) or 0.0),
                    opened_at=float(getattr(existing, "opened_at", ts) or ts),
                    updated_at=float(ts),
                    unrealized_pnl_usd=float(row.get("unrealized_pnl_usd") or 0.0),
                    entry_fee_usd=float(getattr(existing, "entry_fee_usd", 0.0) or 0.0),
                    entry_slippage_usd=float(getattr(existing, "entry_slippage_usd", 0.0) or 0.0),
                    entry_reference_price=float(getattr(existing, "entry_reference_price", 0.0) or 0.0),
                    entry_effective_slippage_bps=float(getattr(existing, "entry_effective_slippage_bps", 0.0) or 0.0),
                    latency_bps=float(getattr(existing, "latency_bps", 0.0) or 0.0),
                    spread_bps=float(getattr(existing, "spread_bps", 0.0) or 0.0),
                    lineage=dict(getattr(existing, "lineage", {}) or {}),
                    risk_plan=dict(getattr(existing, "risk_plan", {}) or {}),
                )
            state.positions = normalized_positions
            state.authority_mode = "live_exchange_reconciled"
            state.reconciliation_status = "reconciled"
            state.last_reconciled_at = float(ts)
            state.last_reconciliation_error = ""
        elif status:
            state.reconciliation_status = status
            if status != "reconciled":
                state.last_reconciled_at = None
        else:
            state.reconciliation_status = "error"
            state.last_reconciled_at = None

    @staticmethod
    def _classify_close_trigger(exit_reason: str) -> str:
        reason = str(exit_reason or "").lower()
        if "take_profit" in reason or reason == "tp_hit" or reason.endswith("tp_hit"):
            return "take_profit"
        if "stop_loss" in reason or reason == "sl_hit" or reason.endswith("sl_hit"):
            return "stop_loss"
        if "signal" in reason or "reversal" in reason:
            return "signal"
        if "manual" in reason:
            return "manual"
        if "timeout" in reason:
            return "timeout"
        if "risk" in reason or "blocked_symbol" in reason or "circuit" in reason:
            return "risk_guard"
        return "execution_exit"

    @classmethod
    def _build_close_event(cls, *, position: PositionState, close_price: float, ts: float, exit_reason: str, lineage_update: dict[str, Any] | None = None) -> dict[str, Any]:
        trigger = cls._classify_close_trigger(exit_reason)
        lineage_update = dict(lineage_update or {})
        return {
            "contract_version": "risk_close_contract_v1",
            "trigger": trigger,
            "exit_reason": str(exit_reason or "execution_exit"),
            "trigger_source": str(lineage_update.get("close_trigger_source") or ("protective_exit" if trigger in {"stop_loss", "take_profit"} else "execution")),
            "symbol": str(position.symbol).upper(),
            "side": str(position.side).upper(),
            "entry_price": float(position.entry_price),
            "exit_price": float(close_price),
            "close_price": float(close_price),
            "stop_loss": float(position.stop_loss),
            "take_profit": float(position.take_profit),
            "opened_at": float(position.opened_at),
            "closed_at": float(ts),
            "is_protective": bool(trigger in {"stop_loss", "take_profit"}),
            "is_sl_tp": bool(trigger in {"stop_loss", "take_profit"}),
        }

    def _close_position(self, state: PortfolioState, *, symbol: str, close_price: float, ts: float, exit_reason: str = "", lineage_update: dict[str, Any] | None = None) -> float:
        existing = state.positions.pop(symbol.upper(), None)
        if existing is None:
            return 0.0
        direction = 1.0 if existing.side == "LONG" else -1.0
        gross_pnl = (float(close_price) - float(existing.entry_price)) * float(existing.quantity) * direction
        close_fill = _fill_cost_payload((lineage_update or {}).get("execution_fill") if isinstance(lineage_update, dict) else None)
        entry_fee = _safe_float(getattr(existing, "entry_fee_usd", 0.0), 0.0)
        exit_fee = _safe_float(close_fill.get("fee_usd"), 0.0)
        fees_usd = entry_fee + exit_fee
        slippage_usd = _safe_float(getattr(existing, "entry_slippage_usd", 0.0), 0.0) + _safe_float(close_fill.get("estimated_slippage_usd"), 0.0)
        pnl = gross_pnl - fees_usd
        close_event = self._build_close_event(position=existing, close_price=float(close_price), ts=float(ts), exit_reason=exit_reason, lineage_update=lineage_update)
        close_event["gross_pnl_usd"] = float(gross_pnl)
        close_event["fees_usd"] = float(fees_usd)
        close_event["slippage_usd"] = float(slippage_usd)
        close_event["net_pnl_usd"] = float(pnl)
        close_event["exit_reference_price"] = float(close_fill.get("reference_price") or close_price or 0.0)
        lineage = dict(getattr(existing, "lineage", {}) or {})
        lineage.update(dict(lineage_update or {}))
        lineage.setdefault("contract_version", "trade_lineage_v1")
        lineage["close_trigger"] = close_event["trigger"]
        lineage["close_trigger_source"] = close_event["trigger_source"]
        lineage["close_event"] = close_event
        state.realized_pnl_usd += pnl
        state.daily_realized_pnl_usd += pnl
        trade_payload = {
            "symbol": str(existing.symbol).upper(),
            "side": str(existing.side).upper(),
            "entry_price": float(existing.entry_price),
            "exit_price": float(close_price),
            "pnl_usd": float(pnl),
            "gross_pnl_usd": float(gross_pnl),
            "fees_usd": float(fees_usd),
            "slippage_usd": float(slippage_usd),
            "exit_reason": str(exit_reason or "execution_exit"),
        }
        risk_payload = dict(getattr(existing, "risk_plan", {}) or {})
        attribution = build_closed_trade_attribution(
            trade=trade_payload,
            lineage=lineage,
            risk_plan=risk_payload,
            close_event=close_event,
        )
        state.closed_trades.append(
            ClosedTradeState(
                symbol=str(existing.symbol).upper(),
                side=str(existing.side).upper(),
                quantity=float(existing.quantity),
                notional_usd=float(existing.notional_usd),
                entry_price=float(existing.entry_price),
                exit_price=float(close_price),
                opened_at=float(existing.opened_at),
                closed_at=float(ts),
                pnl_usd=float(pnl),
                exit_reason=str(exit_reason or "execution_exit"),
                mode=str(state.authority_mode or "paper_local"),
                gross_pnl_usd=float(gross_pnl),
                fees_usd=float(fees_usd),
                slippage_usd=float(slippage_usd),
                latency_bps=max(_safe_float(getattr(existing, "latency_bps", 0.0), 0.0), _safe_float(close_fill.get("latency_bps"), 0.0)),
                spread_bps=max(_safe_float(getattr(existing, "spread_bps", 0.0), 0.0), _safe_float(close_fill.get("spread_bps"), 0.0)),
                entry_reference_price=float(getattr(existing, "entry_reference_price", 0.0) or existing.entry_price),
                exit_reference_price=float(close_fill.get("reference_price") or close_price or 0.0),
                lineage=lineage,
                risk_plan=risk_payload,
                close_event=close_event,
                attribution=attribution,
            )
        )
        state.updated_at = float(ts)
        return pnl

    def apply_execution(
        self,
        state: PortfolioState,
        *,
        symbol: str,
        action: str,
        price: float,
        ts: float,
        position_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply an approved execution intent to the persisted portfolio state."""
        self.reset_daily_if_needed(state, ts=ts)
        action = str(action or "HOLD").upper()
        symbol = str(symbol).upper()
        lineage = dict((position_plan or {}).get("lineage") or {}) if isinstance(position_plan, dict) else {}
        execution_report: dict[str, Any] = {"symbol": symbol, "action": action, "price": float(price), "ts": float(ts), "realized_pnl_usd": 0.0, "lineage": lineage}

        exit_reason = str((position_plan or {}).get("exit_reason") or action.lower()) if isinstance(position_plan, dict) else action.lower()

        if action in {"ENTER_LONG", "ENTER_SHORT"} and symbol in state.positions:
            existing = state.positions[symbol]
            raise PositionStateInvariantError(
                f"direct_entry_rejected_open_position:{symbol}:{str(existing.side or '').upper()}"
            )

        if action in {"EXIT", "EXIT_LONG", "EXIT_SHORT"}:
            execution_report["realized_pnl_usd"] = self._close_position(state, symbol=symbol, close_price=price, ts=ts, exit_reason=exit_reason, lineage_update=lineage)
            return execution_report

        if action in {"FLIP_TO_LONG", "FLIP_TO_SHORT"}:
            execution_report["realized_pnl_usd"] = self._close_position(state, symbol=symbol, close_price=price, ts=ts, exit_reason=exit_reason, lineage_update=lineage)

        if action not in {"ENTER_LONG", "ENTER_SHORT", "FLIP_TO_LONG", "FLIP_TO_SHORT"}:
            state.updated_at = float(ts)
            return execution_report

        plan = dict(position_plan or {})
        entry_fill = _fill_cost_payload(plan.get("execution_fill") if isinstance(plan, dict) else None)
        risk_plan = {k: v for k, v in plan.items() if k not in {"lineage"}}
        risk_plan.setdefault("contract_version", "risk_plan_v1")
        risk_plan.setdefault("entry_fee_usd", float(entry_fill.get("fee_usd", 0.0)))
        risk_plan.setdefault("entry_slippage_usd", float(entry_fill.get("estimated_slippage_usd", 0.0)))
        risk_plan.setdefault("entry_reference_price", float(entry_fill.get("reference_price", 0.0)))
        risk_plan.setdefault("entry_effective_slippage_bps", float(entry_fill.get("effective_slippage_bps", 0.0)))
        risk_plan.setdefault("latency_bps", float(entry_fill.get("latency_bps", 0.0)))
        risk_plan.setdefault("spread_bps", float(entry_fill.get("spread_bps", 0.0)))
        side = "LONG" if "LONG" in action else "SHORT"
        state.positions[symbol] = PositionState(
            symbol=symbol,
            side=side,
            quantity=float(plan.get("quantity") or 0.0),
            notional_usd=float(plan.get("notional_usd") or 0.0),
            entry_price=float(price),
            stop_loss=float(plan.get("stop_loss") or 0.0),
            take_profit=float(plan.get("take_profit") or 0.0),
            opened_at=float(ts),
            updated_at=float(ts),
            unrealized_pnl_usd=0.0,
            entry_fee_usd=float(entry_fill.get("fee_usd", 0.0)),
            entry_slippage_usd=float(entry_fill.get("estimated_slippage_usd", 0.0)),
            entry_reference_price=float(entry_fill.get("reference_price", 0.0) or price or 0.0),
            entry_effective_slippage_bps=float(entry_fill.get("effective_slippage_bps", 0.0)),
            latency_bps=float(entry_fill.get("latency_bps", 0.0)),
            spread_bps=float(entry_fill.get("spread_bps", 0.0)),
            lineage=lineage,
            risk_plan=risk_plan,
        )
        state.updated_at = float(ts)
        return execution_report

    def mark_to_market(self, state: PortfolioState, price_by_symbol: dict[str, float], *, ts: float) -> dict[str, Any]:
        """Update unrealized PnL and equity using the latest available prices."""
        self.reset_daily_if_needed(state, ts=ts)
        total_unrealized = 0.0
        gross_exposure = 0.0
        long_exposure = 0.0
        short_exposure = 0.0
        for symbol, position in state.positions.items():
            mark = float(price_by_symbol.get(symbol) or position.entry_price or 0.0)
            direction = 1.0 if position.side == "LONG" else -1.0
            pnl = (mark - float(position.entry_price)) * float(position.quantity) * direction
            position.unrealized_pnl_usd = float(pnl)
            position.updated_at = float(ts)
            total_unrealized += pnl
            gross_exposure += abs(float(position.notional_usd))
            if position.side == "LONG":
                long_exposure += abs(float(position.notional_usd))
            else:
                short_exposure += abs(float(position.notional_usd))
        state.last_equity_usd = self.starting_equity_usd + float(state.realized_pnl_usd) + float(total_unrealized)
        state.peak_equity_usd = max(float(state.peak_equity_usd), float(state.last_equity_usd), self.starting_equity_usd)
        state.updated_at = float(ts)
        correlation_groups: dict[str, str] = {}
        cluster_exposure_usd: dict[str, float] = {}
        cluster_position_counts: dict[str, int] = {}
        for symbol, position in state.positions.items():
            group = correlation_groups.get(str(symbol).upper(), "")
            if group:
                cluster_exposure_usd[group] = cluster_exposure_usd.get(group, 0.0) + abs(float(position.notional_usd))
                cluster_position_counts[group] = cluster_position_counts.get(group, 0) + 1
        return {
            "realized_pnl_usd": float(state.realized_pnl_usd),
            "daily_realized_pnl_usd": float(state.daily_realized_pnl_usd),
            "unrealized_pnl_usd": float(total_unrealized),
            "equity_usd": float(state.last_equity_usd),
            "peak_equity_usd": float(state.peak_equity_usd),
            "gross_exposure_usd": float(gross_exposure),
            "long_exposure_usd": float(long_exposure),
            "short_exposure_usd": float(short_exposure),
            "net_exposure_usd": float(long_exposure - short_exposure),
            "cluster_exposure_usd": cluster_exposure_usd,
            "cluster_position_counts": cluster_position_counts,
            "open_positions": len(state.positions),
            "authority_mode": str(state.authority_mode or "paper_local"),
            "reconciliation_status": str(state.reconciliation_status or "paper_local"),
            "last_reconciled_at": state.last_reconciled_at,
            "exchange_position_count": int(state.exchange_position_count),
            "reconciliation_mismatch_count": int(state.reconciliation_mismatch_count),
            "last_reconciliation_error": str(state.last_reconciliation_error or ""),
            "exchange_snapshot_hash": str(state.exchange_snapshot_hash or ""),
            "execution_integrity_status": str(state.execution_integrity_status or "ok"),
            "execution_integrity_error": str(state.execution_integrity_error or ""),
            "execution_integrity_failure_at": state.execution_integrity_failure_at,
        }

    @staticmethod
    def execution_summary(state: PortfolioState, *, unrealized_pnl_usd: float = 0.0) -> dict[str, Any]:
        closed = list(getattr(state, "closed_trades", []) or [])
        wins = sum(1 for trade in closed if float(getattr(trade, "pnl_usd", 0.0) or 0.0) > 0.0)
        losses = sum(1 for trade in closed if float(getattr(trade, "pnl_usd", 0.0) or 0.0) < 0.0)
        completed_count = len(closed)
        realized = float(getattr(state, "realized_pnl_usd", 0.0) or 0.0)
        gross_pnl = sum(float(getattr(trade, "gross_pnl_usd", getattr(trade, "pnl_usd", 0.0)) or 0.0) for trade in closed)
        fees_usd = sum(float(getattr(trade, "fees_usd", 0.0) or 0.0) for trade in closed)
        slippage_usd = sum(float(getattr(trade, "slippage_usd", 0.0) or 0.0) for trade in closed)
        return {
            "open_positions": len(getattr(state, "positions", {}) or {}),
            "open_count": len(getattr(state, "positions", {}) or {}),
            "completed_count": completed_count,
            "trades": completed_count,
            "wins": wins,
            "losses": losses,
            "total_pnl": realized,
            "realized_pnl_usd": realized,
            "gross_pnl_usd": float(gross_pnl),
            "fees_usd": float(fees_usd),
            "slippage_usd": float(slippage_usd),
            "unrealized_pnl_usd": float(unrealized_pnl_usd or 0.0),
            "win_rate": (float(wins) / float(completed_count)) if completed_count else 0.0,
            "authority_mode": str(getattr(state, "authority_mode", "paper_local") or "paper_local"),
            "reconciliation_status": str(getattr(state, "reconciliation_status", "paper_local") or "paper_local"),
            "execution_integrity_status": str(getattr(state, "execution_integrity_status", "ok") or "ok"),
            "execution_integrity_error": str(getattr(state, "execution_integrity_error", "") or ""),
            "execution_integrity_failure_at": getattr(state, "execution_integrity_failure_at", None),
        }

    @staticmethod
    def closed_trade_rows(state: PortfolioState, *, limit: int = 120) -> list[dict[str, Any]]:
        rows = [asdict(trade) for trade in list(getattr(state, "closed_trades", []) or [])]
        rows.sort(key=lambda row: float(row.get("closed_at") or 0.0), reverse=True)
        return rows[: max(0, int(limit))]

    @staticmethod
    def signed_positions(state: PortfolioState) -> dict[str, int]:
        return {
            symbol: (1 if position.side == "LONG" else -1)
            for symbol, position in state.positions.items()
        }
