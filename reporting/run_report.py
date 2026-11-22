from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.shared_utils import atomic_write_bytes, load_json, now_utc_ts
from execution.portfolio_state import build_closed_trade_attribution, build_trade_excursion_attribution, paper_execution_trade_source_from_paths
from monitoring.publisher import MonitorPublisher
from features.market_loader import load_market_window
from inference.readiness import readiness_blocker_report
from core.symbol_management import load_symbol_state, warmup_readiness_summary


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


def _utc_stamp(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(float(ts if ts is not None else now_utc_ts()), tz=timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


def _iso(ts: Any) -> str | None:
    if ts in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return str(ts)


def _load_jsonl_tail(path: Path, *, limit: int = 500, max_scan_bytes: int = 2_000_000) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_scan_bytes:
                handle.seek(max(0, size - max_scan_bytes))
                handle.readline()
            for raw in handle:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return rows[-max(1, int(limit)):]


def _get(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def _execution_summary(snapshot: dict[str, Any], paths: Any) -> dict[str, Any]:
    runtime = snapshot.get("execution_runtime") or snapshot.get("paper_execution_runtime") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    summary = runtime.get("execution_summary") or runtime.get("summary") or {}
    if not isinstance(summary, dict):
        summary = {}

    trade_source = paper_execution_trade_source_from_paths(paths)
    paper_summary = trade_source.get("summary") if isinstance(trade_source, dict) else {}
    portfolio_meta = trade_source.get("portfolio") if isinstance(trade_source, dict) else {}
    if isinstance(paper_summary, dict) and isinstance(portfolio_meta, dict) and portfolio_meta.get("exists") and portfolio_meta.get("reason") == "loaded":
        selected = dict(paper_summary)
        selected["source"] = "paper_execution_ledger"
        selected["source_row_count"] = _safe_int(portfolio_meta.get("row_count", 0), 0)
    else:
        selected = dict(summary)
        selected["source"] = "runtime_execution_payload"

    return {
        "source": selected.get("source", "runtime_execution_payload"),
        "completed_trades": _safe_int(selected.get("completed_count", selected.get("trades", 0)), 0),
        "wins": _safe_int(selected.get("wins", 0), 0),
        "losses": _safe_int(selected.get("losses", 0), 0),
        "flat": _safe_int(selected.get("flat", 0), 0),
        "win_rate": _safe_float(selected.get("win_rate", 0.0), 0.0),
        "total_pnl": _safe_float(selected.get("total_pnl", selected.get("realized_pnl_usd", 0.0)), 0.0),
        "gross_pnl_usd": _safe_float(selected.get("gross_pnl_usd", 0.0), 0.0),
        "fees_usd": _safe_float(selected.get("fees_usd", 0.0), 0.0),
        "slippage_usd": _safe_float(selected.get("slippage_usd", 0.0), 0.0),
        "open_positions": _safe_int(selected.get("open_positions", runtime.get("open_position_count", 0)), 0),
        "trade_result_source": runtime.get("trade_result_source", trade_source),
    }



def _registry_readiness_diagnostics(readiness_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = readiness_payload if isinstance(readiness_payload, dict) else {}
    rows = payload.get("rows") or payload.get("execution_readiness_matrix") or []
    if not isinstance(rows, list):
        rows = []
    model_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        interval = str(row.get("interval") or "")
        models = row.get("models") if isinstance(row.get("models"), dict) else {}
        for model_type, model in models.items():
            if not isinstance(model, dict):
                continue
            ready = bool(model.get("ready", False))
            reason = str(model.get("reason") or "unknown")
            model_rows.append({
                "symbol": symbol,
                "interval": interval,
                "model_type": str(model_type).lower(),
                "execution_ready": ready,
                "blocker_reason": "ready" if ready else reason,
                "active_manifest_exists": bool(model.get("active_manifest_exists", bool(model.get("active_manifest_path") or model.get("manifest_path")))),
                "active_manifest_valid": bool(model.get("active_manifest_valid", False)),
                "registry_entry_exists": bool(model.get("registry_entry_exists", False)),
                "registry_truth_status": str(model.get("registry_truth_status") or "unknown"),
                "registry_execution_ready": bool(model.get("registry_execution_ready", False)),
                "registry_promotion_approved": bool(model.get("registry_promotion_approved", False)),
                "registry_active_manifest_valid": bool(model.get("registry_active_manifest_valid", False)),
                "feature_compatible": bool(model.get("feature_compatible", False)),
                "deployment_stage": str(model.get("deployment_stage") or "unknown"),
                "feature_contract_diagnostics": dict(model.get("feature_contract_diagnostics") or {}),
            })
    blocked = [row for row in model_rows if not row.get("execution_ready")]
    return {
        "total_model_rows": len(model_rows),
        "blocked_model_rows": len(blocked),
        "execution_ready_model_rows": len(model_rows) - len(blocked),
        "model_diagnostics": sorted(model_rows, key=lambda item: (item["symbol"], item["model_type"], item["interval"])),
        "blocker_diagnostics": sorted(blocked, key=lambda item: (item["symbol"], item["model_type"], item["interval"])),
    }

def _training_summary(paths: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    registry_path = Path(getattr(paths, "training_registry_path", getattr(paths, "registry_path", Path("data/runtime/training_registry.json"))))
    registry = load_json(registry_path, default={}) or {}
    entries = [entry for entry in registry.values() if isinstance(entry, dict)] if isinstance(registry, dict) else []

    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    by_symbol_tf_model: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for entry in entries:
        model = str(entry.get("model_type") or "unknown")
        status = str(entry.get("status") or "unknown")
        truth_status = str(entry.get("registry_truth_status") or "unknown")
        by_model[model][status] += 1
        by_model[model][truth_status] += 1
        row = {
            "model_type": model,
            "symbol": str(entry.get("symbol") or "").upper(),
            "interval": str(entry.get("interval") or ""),
            "status": status,
            "truth_status": truth_status,
            "execution_ready": bool(entry.get("execution_ready", False)),
            "active_manifest_valid": bool(entry.get("active_manifest_valid", False)),
            "candidate_saved": bool(entry.get("candidate_saved", False)),
            "promotion_attempted": bool(entry.get("promotion_attempted", False)),
            "promotion_approved": bool(entry.get("promotion_approved", False)),
            "rows_used": _safe_int(entry.get("rows_used", 0), 0),
            "last_success_at": entry.get("last_success_at"),
            "truth_reasons": list(entry.get("truth_reasons") or []),
        }
        by_symbol_tf_model.append(row)
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        quality_rows.append(_model_quality_row(row, metrics))

    registry_snapshot = snapshot.get("training_registry") if isinstance(snapshot.get("training_registry"), dict) else {}
    return {
        "registry_path": str(registry_path),
        "entry_count": len(entries),
        "counts_by_model": {model: dict(counter) for model, counter in by_model.items()},
        "dashboard_counts": registry_snapshot.get("counts", {}),
        "dashboard_diagnostic_counts": registry_snapshot.get("diagnostic_counts", {}),
        "by_symbol_timeframe_model": sorted(by_symbol_tf_model, key=lambda r: (r["symbol"], r["model_type"], r["interval"])),
        "model_quality_usability": sorted(quality_rows, key=lambda r: (r["symbol"], r["model_type"], r["interval"])),
        "ml_quality_report": sorted(quality_rows, key=lambda r: (r["symbol"], r["model_type"], r["interval"])),
    }



def _model_quality_row(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    quality = metrics.get("directional_quality") if isinstance(metrics.get("directional_quality"), dict) else {}
    promotion = metrics.get("promotion") if isinstance(metrics.get("promotion"), dict) else {}
    return {
        "symbol": row.get("symbol"),
        "interval": row.get("interval"),
        "model_type": row.get("model_type"),
        "model_role": metrics.get("model_role"),
        "rows_used": row.get("rows_used"),
        "rows_train": _safe_int(metrics.get("rows_train", 0), 0),
        "rows_valid": _safe_int(metrics.get("rows_valid", 0), 0),
        "fold_count": _safe_int(metrics.get("fold_count", 0), 0),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "roc_auc_ovr": metrics.get("roc_auc_ovr"),
        "long_precision": quality.get("long_precision"),
        "long_recall": quality.get("long_recall"),
        "short_precision": quality.get("short_precision"),
        "short_recall": quality.get("short_recall"),
        "hold_true_fraction": quality.get("hold_true_fraction"),
        "hold_prediction_fraction": quality.get("hold_prediction_fraction"),
        "directional_prediction_fraction": quality.get("directional_prediction_fraction"),
        "hold_dominance_warning": bool(quality.get("hold_dominance_warning", False)),
        "class_distribution_train": metrics.get("class_distribution_train"),
        "class_distribution_valid": quality.get("class_distribution_valid", metrics.get("class_distribution_valid")),
        "prediction_distribution_valid": quality.get("prediction_distribution_valid"),
        "confusion_matrix": quality.get("confusion_matrix"),
        "promotion_attempted": bool(promotion.get("attempted", row.get("promotion_attempted", False))),
        "promotion_approved": bool(promotion.get("approved", row.get("promotion_approved", False))),
        "promotion_promoted": bool(promotion.get("promoted", False)),
        "promotion_reasons": list(promotion.get("reasons") or row.get("truth_reasons") or []),
        "execution_ready": bool(row.get("execution_ready", False)),
        "truth_status": row.get("truth_status"),
    }

def _signal_summary(paths: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    journal_path = Path(getattr(paths, "decision_journal_path", Path("data/runtime/decision_journal.jsonl")))
    rows = _load_jsonl_tail(journal_path, limit=1000)
    by_symbol = Counter(str(row.get("symbol") or "").upper() for row in rows if row.get("symbol"))
    by_action = Counter(str(row.get("action") or row.get("effective_action") or "unknown") for row in rows)
    by_reason = Counter(str(row.get("reason") or row.get("rejection_reason") or "unknown") for row in rows)
    hmm_gate_results: Counter[str] = Counter()
    hmm_gate_reasons: Counter[str] = Counter()
    hmm_gate_applies = 0
    hmm_gate_used = 0
    for row in rows:
        hmm = _nested_dict_get(row, "components", "hmm")
        if not isinstance(hmm, dict):
            continue
        if bool(hmm.get("gate_applies", False)):
            hmm_gate_applies += 1
        if bool(hmm.get("gate_used", False)):
            hmm_gate_used += 1
        result = str(hmm.get("gate_result") or "unknown")
        reason = str(hmm.get("gate_reason") or "unknown")
        hmm_gate_results[result] += 1
        hmm_gate_reasons[reason] += 1
    runtime = snapshot.get("signal_runtime") if isinstance(snapshot.get("signal_runtime"), dict) else {}
    return {
        "decision_journal_path": str(journal_path),
        "recent_rows_scanned": len(rows),
        "by_symbol": dict(by_symbol),
        "by_action": dict(by_action),
        "top_reasons": dict(by_reason.most_common(20)),
        "hmm_gate_summary": {
            "rows_with_gate_applicable": int(hmm_gate_applies),
            "rows_with_gate_used": int(hmm_gate_used),
            "gate_result_counts": dict(hmm_gate_results),
            "gate_reason_counts": dict(hmm_gate_reasons.most_common(20)),
            "artifact_governance_role": "shadow_candidate",
            "runtime_gate_scope": "paper_entry_or_flip",
            "live_influence_allowed": False,
        },
        "runtime_summary": {
            "symbols": runtime.get("symbols"),
            "recent_signal_count": len(runtime.get("recent_signals") or []) if isinstance(runtime.get("recent_signals"), list) else 0,
            "recent_decision_count": len(runtime.get("recent_decisions") or []) if isinstance(runtime.get("recent_decisions"), list) else 0,
            "updated_at": runtime.get("updated_at"),
        },
    }


def _nested_dict_get(row: dict[str, Any], *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _risk_breakdown_from_row(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("risk_breakdown")
    if isinstance(direct, dict) and direct:
        return dict(direct)
    for path in (
        ("components", "risk_breakdown"),
        ("components", "risk_rejection"),
        ("components", "risk_decision", "components", "risk_breakdown"),
        ("risk_decision", "components", "risk_breakdown"),
    ):
        value = _nested_dict_get(row, *path)
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _closed_trade_attribution(paths: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Summarize closed-trade lineage without changing execution state."""
    portfolio_path = Path(getattr(paths, "portfolio_state_path", Path("data/runtime/portfolio_state.json")))
    payload = load_json(portfolio_path, default={}) or {}
    closed = payload.get("closed_trades", payload.get("trade_history", [])) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    missing_decision_id = 0
    if isinstance(closed, list):
        for raw in closed[-1000:]:
            if not isinstance(raw, dict):
                continue
            attribution = raw.get("attribution") if isinstance(raw.get("attribution"), dict) else None
            if not attribution:
                attribution = build_closed_trade_attribution(
                    trade=raw,
                    lineage=raw.get("lineage") if isinstance(raw.get("lineage"), dict) else {},
                    risk_plan=raw.get("risk_plan") if isinstance(raw.get("risk_plan"), dict) else {},
                    close_event=raw.get("close_event") if isinstance(raw.get("close_event"), dict) else {},
                )
            normalized = dict(attribution)
            normalized.setdefault("symbol", str(raw.get("symbol") or "").upper())
            normalized.setdefault("pnl_usd", _safe_float(raw.get("pnl_usd"), 0.0))
            normalized.setdefault("exit_reason", str(raw.get("exit_reason") or ""))
            if str(normalized.get("decision_id") or "unknown") == "unknown":
                missing_decision_id += 1
            reason_counts[str(normalized.get("exit_reason") or "unknown")] += 1
            matched = normalized.get("prediction_matched_outcome")
            matched_counts[str(matched if matched is not None else "unknown")] += 1
            rows.append(normalized)
    return {
        "schema_version": "closed_trade_attribution_report.v1",
        "portfolio_state_path": str(portfolio_path),
        "closed_trade_count": len(rows),
        "missing_decision_id_count": missing_decision_id,
        "exit_reason_counts": dict(reason_counts),
        "prediction_match_counts": dict(matched_counts),
        "recent": rows[-100:],
    }



def _coerce_trade_ts_ms(value: Any) -> int | None:
    """Normalize persisted trade timestamps to milliseconds for candle lookup."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0.0:
            return None
        return int(numeric if numeric > 10_000_000_000 else numeric * 1000.0)
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        if numeric > 0.0:
            return int(numeric if numeric > 10_000_000_000 else numeric * 1000.0)
    except Exception:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000.0)
    except Exception:
        return None


def _trade_excursion_report(paths: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compute read-only MAE/MFE diagnostics for closed trades.

    This report is intentionally downstream of execution. It reads closed trade
    rows plus stored 1m candles, and never changes SL/TP, risk, orders, or PnL.
    """
    portfolio_path = Path(getattr(paths, "portfolio_state_path", Path("data/runtime/portfolio_state.json")))
    payload = load_json(portfolio_path, default={}) or {}
    portfolio_rows = payload.get("closed_trades", payload.get("trade_history", [])) if isinstance(payload, dict) else []
    rows = [row for row in portfolio_rows if isinstance(row, dict)] if isinstance(portfolio_rows, list) else []
    source = "paper_execution_ledger"

    analyzed: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    exit_counts: Counter[str] = Counter()
    sl_too_tight_candidates = 0
    tp_reachable_candidates = 0
    directionally_wrong_candidates = 0
    total_mfe_bps = 0.0
    total_mae_bps = 0.0

    for raw in rows[-1000:]:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or "").upper()
        entry_ms = _coerce_trade_ts_ms(raw.get("entry_time", raw.get("opened_at")))
        exit_ms = _coerce_trade_ts_ms(raw.get("exit_time", raw.get("closed_at")))
        candles: list[dict[str, Any]] = []
        if symbol and entry_ms is not None and exit_ms is not None:
            start_ms = min(entry_ms, exit_ms) - 60_000
            end_ms = max(entry_ms, exit_ms) + 60_000
            try:
                frame = load_market_window(paths, symbol, "1m", start_ms, end_ms)
                candles = frame.to_dict(orient="records") if hasattr(frame, "to_dict") else []
            except Exception as exc:
                candles = []
                raw = {**raw, "excursion_load_error": f"{exc.__class__.__name__}"}
        payload_row = build_trade_excursion_attribution(trade=raw, candles=candles)
        if raw.get("excursion_load_error"):
            payload_row["analysis_status"] = "market_load_error"
            payload_row["analysis_reason"] = str(raw.get("excursion_load_error"))
        payload_row["source"] = source
        payload_row["entry_time"] = raw.get("entry_time", raw.get("opened_at"))
        payload_row["exit_time"] = raw.get("exit_time", raw.get("closed_at"))
        analyzed.append(payload_row)
        status_counts[str(payload_row.get("analysis_status") or "unknown")] += 1
        exit_counts[str(payload_row.get("exit_reason") or "unknown")] += 1
        if payload_row.get("sl_too_tight_candidate") is True:
            sl_too_tight_candidates += 1
        if payload_row.get("tp_was_reachable_candidate") is True:
            tp_reachable_candidates += 1
        if payload_row.get("directionally_wrong_candidate") is True:
            directionally_wrong_candidates += 1
        total_mfe_bps += _safe_float(payload_row.get("max_favorable_excursion_bps"), 0.0)
        total_mae_bps += _safe_float(payload_row.get("max_adverse_excursion_bps"), 0.0)

    count = len(analyzed)
    return {
        "schema_version": "trade_excursion_report.v1",
        "source": source,
        "portfolio_state_path": str(portfolio_path),
        "closed_trade_count": count,
        "status_counts": dict(status_counts),
        "exit_reason_counts": dict(exit_counts),
        "sl_too_tight_candidate_count": sl_too_tight_candidates,
        "tp_reachable_candidate_count": tp_reachable_candidates,
        "directionally_wrong_candidate_count": directionally_wrong_candidates,
        "avg_mfe_bps": (total_mfe_bps / count) if count else 0.0,
        "avg_mae_bps": (total_mae_bps / count) if count else 0.0,
        "recent": analyzed[-100:],
    }


def _risk_rejection_attribution(paths: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Summarize risk rejections without changing risk decisions.

    Step 4 is observability-only: the report reads persisted decision rows and
    surfaces requested-vs-approved sizing, clamp reason, distortion, and EV/score
    so operators can diagnose missed trades before changing thresholds.
    """
    journal_path = Path(getattr(paths, "decision_journal_path", Path("data/runtime/decision_journal.jsonl")))
    rows = _load_jsonl_tail(journal_path, limit=3000)
    runtime = snapshot.get("execution_runtime") or snapshot.get("paper_execution_runtime") or {}
    if isinstance(runtime, dict):
        for item in runtime.get("recent_decisions") or []:
            if isinstance(item, dict):
                rows.append(item)

    seen: set[str] = set()
    risk_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    clamp_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    total_requested = 0.0
    total_approved = 0.0

    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("execution_status") or _nested_dict_get(row, "execution_contract", "execution_status") or "")
        stage = str(row.get("execution_blocker_stage") or _nested_dict_get(row, "execution_contract", "execution_blocker_stage") or "")
        reason = str(row.get("rejection_reason") or row.get("execution_blocker") or row.get("reason") or "")
        breakdown = _risk_breakdown_from_row(row)
        is_risk_rejection = status == "rejected_risk" or stage == "risk" or bool(breakdown)
        if not is_risk_rejection:
            continue
        decision_id = str(row.get("decision_id") or _nested_dict_get(row, "components", "lineage", "decision_id") or f"row-{len(risk_rows)}")
        if decision_id in seen:
            continue
        seen.add(decision_id)
        requested = _safe_float(breakdown.get("requested_notional_usd", breakdown.get("intended_notional_usd", 0.0)), 0.0)
        approved = _safe_float(breakdown.get("approved_notional_usd", breakdown.get("notional_usd", 0.0)), 0.0)
        symbol = str(row.get("symbol") or breakdown.get("symbol") or "").upper()
        normalized = {
            "decision_id": decision_id,
            "ts": row.get("ts") or row.get("timestamp") or row.get("updated_at"),
            "symbol": symbol,
            "side": breakdown.get("side") or row.get("effective_action") or row.get("action"),
            "status": status or "rejected_risk",
            "reason": breakdown.get("reason") or reason or "risk_rejected",
            "requested_notional_usd": requested,
            "approved_notional_usd": approved,
            "clamp_reason": str(breakdown.get("clamp_reason") or "none"),
            "risk_distortion_fraction": _safe_float(breakdown.get("risk_distortion_fraction"), 0.0),
            "risk_limit": _safe_float(breakdown.get("risk_limit", breakdown.get("max_risk_distortion_fraction", 0.0)), 0.0),
            "ev_score": row.get("ev_score", row.get("ev", row.get("score"))),
        }
        risk_rows.append(normalized)
        reason_counts[str(normalized["reason"])] += 1
        clamp_counts[str(normalized["clamp_reason"])] += 1
        if symbol:
            symbol_counts[symbol] += 1
        total_requested += requested
        total_approved += approved

    return {
        "schema_version": "risk_rejection_attribution.v1",
        "decision_journal_path": str(journal_path),
        "recent_rows_scanned": len(rows),
        "risk_rejection_count": len(risk_rows),
        "reason_counts": dict(reason_counts),
        "clamp_counts": dict(clamp_counts),
        "symbol_counts": dict(symbol_counts),
        "requested_notional_usd_total": float(total_requested),
        "approved_notional_usd_total": float(total_approved),
        "recent": risk_rows[-100:],
    }


def _symbol_lifecycle(paths: Any) -> dict[str, Any]:
    symbol_state_path = Path(getattr(paths, "symbol_state_path", Path("data/runtime/symbol_state.json")))
    state = load_json(symbol_state_path, default={}) or {}
    active: list[str] = []
    blocked: list[str] = []
    timed_out: list[str] = []
    rows: list[dict[str, Any]] = []
    if isinstance(state, dict):
        raw_rows = state.get("symbols", state)
        if isinstance(raw_rows, dict):
            iterable = raw_rows.items()
        elif isinstance(raw_rows, list):
            iterable = [(item.get("symbol"), item) for item in raw_rows if isinstance(item, dict)]
        else:
            iterable = []
        for symbol, payload in iterable:
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or payload.get("state") or "").upper()
            normalized_symbol = str(symbol or payload.get("symbol") or "").upper()
            rows.append({"symbol": normalized_symbol, "status": status, **payload})
            if status == "ACTIVE":
                active.append(normalized_symbol)
            elif status == "BLOCKED":
                blocked.append(normalized_symbol)
            elif status == "TIMED_OUT":
                timed_out.append(normalized_symbol)
    return {
        "symbol_state_path": str(symbol_state_path),
        "active_symbols": sorted(set(active)),
        "blocked_symbols": sorted(set(blocked)),
        "timed_out_symbols": sorted(set(timed_out)),
        "rows": rows,
    }




def _fmt(value: Any) -> str:
    if value in (None, ""):
        return "--"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    safe_headers = [_fmt(header) for header in headers]
    lines = ["| " + " | ".join(safe_headers) + " |", "| " + " | ".join("---" for _ in safe_headers) + " |"]
    for row in rows:
        cells = [_fmt(cell).replace("\n", " ") for cell in row]
        if len(cells) < len(safe_headers):
            cells.extend(["--"] * (len(safe_headers) - len(cells)))
        lines.append("| " + " | ".join(cells[: len(safe_headers)]) + " |")
    return lines


def _problem_reason(row: dict[str, Any]) -> str:
    reasons = row.get("truth_reasons")
    if isinstance(reasons, list) and reasons:
        return ", ".join(str(reason) for reason in reasons[:4])
    if not row.get("active_manifest_valid"):
        return "active_manifest_missing_or_invalid"
    if not row.get("execution_ready"):
        return "not_execution_ready"
    return ""


def _latest_signal_rows(paths: Any, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    journal_path = Path(getattr(paths, "decision_journal_path", Path("data/runtime/decision_journal.jsonl")))
    rows = _load_jsonl_tail(journal_path, limit=2000)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        ts = _safe_float(row.get("timestamp", row.get("ts", row.get("updated_at", 0.0))), 0.0)
        previous = latest.get(symbol)
        previous_ts = _safe_float(previous.get("timestamp", previous.get("ts", previous.get("updated_at", 0.0))) if previous else 0.0, 0.0)
        if previous is None or ts >= previous_ts:
            latest[symbol] = row

    runtime = snapshot.get("signal_runtime") if isinstance(snapshot.get("signal_runtime"), dict) else {}
    for row in runtime.get("recent_signals") or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol and symbol not in latest:
            latest[symbol] = row
    return [latest[key] for key in sorted(latest)]


def _enhanced_structured_report(report: dict[str, Any]) -> dict[str, Any]:
    runtime = report.get("runtime", {}) if isinstance(report.get("runtime"), dict) else {}
    readiness = report.get("readiness", {}) if isinstance(report.get("readiness"), dict) else {}
    execution = report.get("execution", {}) if isinstance(report.get("execution"), dict) else {}
    price = report.get("execution_latency_price_freshness", {}) if isinstance(report.get("execution_latency_price_freshness"), dict) else {}
    training = report.get("training", {}) if isinstance(report.get("training"), dict) else {}
    warnings_errors = report.get("warnings_errors", {}) if isinstance(report.get("warnings_errors"), dict) else {}
    controllers = runtime.get("controllers") if isinstance(runtime.get("controllers"), dict) else {}

    system_state = [
        {"area": "Runtime stability", "current_system_state": f"duration_seconds={_fmt(runtime.get('duration_seconds'))}; controllers={len(controllers)}", "remarks": "Stable if controllers are running and no blocking errors are present."},
        {"area": "Market data freshness", "current_system_state": f"priced={_fmt(price.get('priced_symbol_count'))}/{_fmt(price.get('expected_symbol_count'))}; stale={_fmt(price.get('stale_symbol_count'))}", "remarks": "Uses runtime price snapshot fields; missing values are left blank."},
        {"area": "ML training freshness", "current_system_state": f"registry_entries={_fmt(training.get('entry_count'))}", "remarks": "Inspect per-symbol rows for missing active manifests or stale models."},
        {"area": "Model quality", "current_system_state": "metrics captured where registry provides them", "remarks": "No fabricated score is produced."},
        {"area": "Signal quality", "current_system_state": f"recent_rows_scanned={_fmt(report.get('signals', {}).get('recent_rows_scanned') if isinstance(report.get('signals'), dict) else None)}", "remarks": "Outcome quality requires lineage-linked trade attribution."},
        {"area": "Execution quality", "current_system_state": f"source={_fmt(execution.get('source'))}; completed={_fmt(execution.get('completed_trades'))}", "remarks": "CSV closed-trade ledger preferred when populated."},
        {"area": "Risk controls", "current_system_state": "close/risk fields reported when runtime provides them", "remarks": "Full interpretation depends on risk-plan persistence."},
        {"area": "PnL / win-loss", "current_system_state": f"wins={_fmt(execution.get('wins'))}; losses={_fmt(execution.get('losses'))}; pnl={_fmt(execution.get('total_pnl'))}", "remarks": "Negative expectancy should block live progression."},
        {"area": "Trade lifecycle traceability", "current_system_state": "lineage fields reported when present", "remarks": "Missing IDs are highlighted by blank report fields."},
        {"area": "Shutdown / reporting", "current_system_state": f"shutdown={_fmt(runtime.get('shutdown'))}", "remarks": "Report is generated from runtime snapshot and persisted ledgers."},
        {"area": "UI observability", "current_system_state": f"readiness={_fmt(readiness.get('status'))}", "remarks": "UI must consume corrected runtime/report contracts."},
        {"area": "Live safety", "current_system_state": f"execution_mode={_fmt(runtime.get('execution_mode'))}; readiness={_fmt(readiness.get('status'))}", "remarks": "Live must remain fail-closed unless explicit gates pass."},
    ]

    health = [
        {"area": "Main runtime", "running_status": "available" if runtime else "missing", "latest_update_or_error": f"duration_seconds={_fmt(runtime.get('duration_seconds'))}"},
        {"area": "Controllers", "running_status": f"{len(controllers)} reported", "latest_update_or_error": ", ".join(sorted(controllers.keys())[:8]) if controllers else "--"},
        {"area": "Trading readiness", "running_status": _fmt(readiness.get("status")), "latest_update_or_error": _fmt(readiness.get("reason"))},
        {"area": "Market DB", "running_status": "see snapshot", "latest_update_or_error": "not separately inferred by report"},
        {"area": "Price cache", "running_status": f"priced={_fmt(price.get('priced_symbol_count'))}/{_fmt(price.get('expected_symbol_count'))}", "latest_update_or_error": f"age={_fmt(price.get('age_seconds'))}; stale={_fmt(price.get('stale_symbol_count'))}"},
        {"area": "Training", "running_status": f"entries={_fmt(training.get('entry_count'))}", "latest_update_or_error": _fmt(training.get("dashboard_diagnostic_counts"))},
        {"area": "Execution", "running_status": f"completed={_fmt(execution.get('completed_trades'))}; open={_fmt(execution.get('open_positions'))}", "latest_update_or_error": f"source={_fmt(execution.get('source'))}"},
        {"area": "UI", "running_status": "snapshot-derived", "latest_update_or_error": f"warnings={_fmt(warnings_errors.get('warnings_count'))}; errors={_fmt(warnings_errors.get('errors_count'))}"},
    ]

    training_rows = []
    for row in training.get("by_symbol_timeframe_model") or []:
        if not isinstance(row, dict):
            continue
        training_rows.append({
            "symbol": row.get("symbol"),
            "timeframe": row.get("interval"),
            "model": row.get("model_type"),
            "active_manifest": "Yes" if row.get("active_manifest_valid") else "No",
            "execution_ready": "Yes" if row.get("execution_ready") else "No",
            "problem_reason": _problem_reason(row),
        })

    latest_signal_rows = []
    for row in report.get("latest_signal_rows", []):
        if not isinstance(row, dict):
            continue
        latest_signal_rows.append({
            "symbol": str(row.get("symbol") or "").upper(),
            "latest_signal": row.get("effective_action") or row.get("action") or row.get("signal") or "--",
            "mode": row.get("mode") or row.get("entry_mode") or "--",
            "reason": row.get("reason") or row.get("rejection_reason") or row.get("decision_reason") or "--",
            "ev_score": row.get("ev_score", row.get("score", row.get("confidence", "--"))),
            "rf": row.get("rf_score", row.get("rf", "--")),
            "gbm": row.get("gbm_score", row.get("gbm", "--")),
            "policy": row.get("policy_score", row.get("policy", row.get("policy_used", "--"))),
            "technical_bias": row.get("technical_bias", row.get("regime", "--")),
            "attempted_execution": row.get("attempted_execution", row.get("attempted", "--")),
            "rejected": row.get("rejected", "--"),
            "rejection_reason": row.get("rejection_reason", row.get("execution_rejection_reason", "--")),
        })

    return {
        "system_state_table": system_state,
        "system_health_table": health,
        "ml_training_forensic_table": training_rows,
        "signal_generator_table": latest_signal_rows,
        "data_quality_notes": {
            "missing_data": [
                "Fields absent from runtime artifacts are intentionally rendered as -- rather than inferred.",
                "Full signal-to-trade outcome scoring requires lineage-linked closed trades.",
            ],
            "inconsistencies": [],
            "system_reliability_concerns": [
                "Live progression should remain blocked until readiness, lineage, risk, and report contracts are validated together."
            ],
        },
    }

def _markdown(report: dict[str, Any]) -> str:
    execution = report.get("execution", {})
    runtime = report.get("runtime", {})
    readiness = report.get("readiness", {})
    price = report.get("execution_latency_price_freshness", {})
    lines = [
        "# Auto Trader Final Run Report",
        "",
        f"- Report ID: `{report.get('report_id')}`",
        f"- Generated at UTC: `{report.get('generated_at_utc')}`",
        f"- Run duration seconds: `{runtime.get('duration_seconds')}`",
        f"- Execution mode: `{runtime.get('execution_mode')}`",
        f"- Trading readiness: `{readiness.get('status')}`",
        f"- Readiness reason: `{readiness.get('reason') or readiness.get('details', {}).get('summary') or '--'}`",
        "",
        "## Readiness Blockers by Layer",
        "",
    ]
    blocker_report = readiness.get("blocker_report") if isinstance(readiness.get("blocker_report"), dict) else {}
    blocker_summary = blocker_report.get("summary") if isinstance(blocker_report.get("summary"), dict) else {}
    if blocker_summary:
        blocker_rows = [[key, value] for key, value in sorted(blocker_summary.items())]
        lines.extend(_md_table(["Layer", "Count"], blocker_rows))
    else:
        lines.append("No structured readiness blockers were reported.")
    lines.extend([
        "",
        "## Execution",
        "",
        f"- Source: `{execution.get('source')}`",
        f"- Completed trades: `{execution.get('completed_trades')}`",
        f"- Wins / Losses / Flat: `{execution.get('wins')}` / `{execution.get('losses')}` / `{execution.get('flat')}`",
        f"- Win rate: `{execution.get('win_rate')}`",
        f"- Total PnL: `{execution.get('total_pnl')}`",
        f"- Open positions: `{execution.get('open_positions')}`",
        "",
        "## Closed Trade Attribution",
        "",
    ])
    closed_attr = report.get("closed_trade_attribution") if isinstance(report.get("closed_trade_attribution"), dict) else {}
    lines.extend([
        f"- Closed trades attributed: `{closed_attr.get('closed_trade_count', 0)}`",
        f"- Missing decision IDs: `{closed_attr.get('missing_decision_id_count', 0)}`",
        f"- Exit reasons: `{closed_attr.get('exit_reason_counts', {})}`",
        f"- Prediction match counts: `{closed_attr.get('prediction_match_counts', {})}`",
        "",
        "## Risk Rejection Attribution",
        "",
    ])
    risk_rejections = report.get("risk_rejection_attribution") if isinstance(report.get("risk_rejection_attribution"), dict) else {}
    lines.extend([
        f"- Risk rejections scanned: `{risk_rejections.get('risk_rejection_count', 0)}`",
        f"- Rejection reasons: `{risk_rejections.get('reason_counts', {})}`",
        f"- Clamp reasons: `{risk_rejections.get('clamp_counts', {})}`",
        f"- Requested / approved notional: `{risk_rejections.get('requested_notional_usd_total', 0.0)}` / `{risk_rejections.get('approved_notional_usd_total', 0.0)}`",
        "",
        "## Price Freshness",
        "",
        f"- Priced symbols: `{price.get('priced_symbol_count')}/{price.get('expected_symbol_count')}`",
        f"- Age seconds: `{price.get('age_seconds')}`",
        f"- Max price age seconds: `{price.get('max_price_age_seconds')}`",
        f"- Sources: `{price.get('price_source_counts')}`",
        "",
        "## Training / Registry",
        "",
        f"- Registry entries: `{report.get('training', {}).get('entry_count')}`",
        f"- Counts by model: `{report.get('training', {}).get('counts_by_model')}`",
        "",
    ])
    enhanced = report.get("enhanced_structured_report") if isinstance(report.get("enhanced_structured_report"), dict) else {}
    lines.extend([
        "## Enhanced Structured Trading System Report",
        "",
        "### Executive Summary and Current System State",
        "",
        "This section groups runtime, data, ML, signal, execution, risk, reporting, UI, and live-safety state using only fields available in the generated report. Missing values are rendered as `--` instead of inferred.",
        "",
    ])
    lines.extend(_md_table(
        ["Area", "Current System State", "Remarks"],
        [[row.get("area"), row.get("current_system_state"), row.get("remarks")] for row in enhanced.get("system_state_table", []) if isinstance(row, dict)],
    ))
    lines.extend([
        "",
        "### Runtime / System Health Status",
        "",
    ])
    lines.extend(_md_table(
        ["Area", "Running Status", "Latest Update or Error"],
        [[row.get("area"), row.get("running_status"), row.get("latest_update_or_error")] for row in enhanced.get("system_health_table", []) if isinstance(row, dict)],
    ))
    lines.extend([
        "",
        "### ML Training Forensic Report",
        "",
    ])
    lines.extend(_md_table(
        ["Symbol", "Time Frame", "Model", "Active Manifest", "Execution Ready", "If Problem, Reason"],
        [[row.get("symbol"), row.get("timeframe"), row.get("model"), row.get("active_manifest"), row.get("execution_ready"), row.get("problem_reason")] for row in enhanced.get("ml_training_forensic_table", []) if isinstance(row, dict)],
    ))
    lines.extend([
        "",
        "### Signal Generator Report",
        "",
    ])
    lines.extend(_md_table(
        ["Symbol", "Latest Signal", "Mode", "Reason", "EV/Score", "RF", "GBM", "Policy", "Technical Bias", "Attempted Execution", "Rejected?", "Rejection Reason"],
        [[row.get("symbol"), row.get("latest_signal"), row.get("mode"), row.get("reason"), row.get("ev_score"), row.get("rf"), row.get("gbm"), row.get("policy"), row.get("technical_bias"), row.get("attempted_execution"), row.get("rejected"), row.get("rejection_reason")] for row in enhanced.get("signal_generator_table", []) if isinstance(row, dict)],
    ))
    quality = enhanced.get("data_quality_notes", {}) if isinstance(enhanced.get("data_quality_notes"), dict) else {}
    lines.extend([
        "",
        "### Overall System Assessment",
        "",
        "- Strengths: runtime/report contracts now consolidate available execution, training, signal, readiness, and freshness fields in one generated artifact.",
        "- Weaknesses: missing source fields remain visible as `--`; no synthetic metrics are invented.",
        "- Critical risks: live trading must remain gated until readiness, lineage, risk, signal, and report contracts are validated together.",
        "- Immediate fix recommendation: resolve any blank lineage or readiness fields surfaced by this report before tuning further.",
        "",
        "### Data Quality Notes",
        "",
        f"- Missing data: `{quality.get('missing_data', [])}`",
        f"- Inconsistencies: `{quality.get('inconsistencies', [])}`",
        f"- System reliability concerns: `{quality.get('system_reliability_concerns', [])}`",
        "",
        "## Warnings / Errors",
        "",
        f"- Warnings: `{report.get('warnings_errors', {}).get('warnings_count')}`",
        f"- Errors: `{report.get('warnings_errors', {}).get('errors_count')}`",
        "",
    ])
    return "\n".join(lines)


def build_run_report(paths: Any, *, shutdown_state: Any | None = None, monitor: Any | None = None) -> dict[str, Any]:
    monitor = monitor or MonitorPublisher(paths)
    snapshot = monitor.snapshot() if hasattr(monitor, "snapshot") else {}
    events = monitor.events(limit=500) if hasattr(monitor, "events") else []
    if not isinstance(snapshot, dict):
        snapshot = {}
    if not isinstance(events, list):
        events = []

    system = snapshot.get("system") if isinstance(snapshot.get("system"), dict) else {}
    generated_ts = now_utc_ts()
    started_at = _safe_float(system.get("started_at"), 0.0)
    duration = max(0.0, generated_ts - started_at) if started_at > 0 else None
    price = _get(snapshot, "market_data", "runtime_price_snapshot", default={})
    if not isinstance(price, dict):
        price = {}
    readiness = _get(snapshot, "trading_readiness", default={})
    if not isinstance(readiness, dict):
        readiness = {}
    signal_runtime_for_readiness = snapshot.get("signal_runtime") if isinstance(snapshot.get("signal_runtime"), dict) else {}
    signal_readiness_for_report = signal_runtime_for_readiness.get("readiness") if isinstance(signal_runtime_for_readiness.get("readiness"), dict) else readiness.get("signal_readiness")
    blocker_report = readiness.get("readiness_blocker_report") if isinstance(readiness.get("readiness_blocker_report"), dict) else readiness_blocker_report(
        signal_readiness_for_report if isinstance(signal_readiness_for_report, dict) else {},
        runtime_blockers=list(readiness.get("blockers") or []),
    )
    control = snapshot.get("control_plane") if isinstance(snapshot.get("control_plane"), dict) else {}

    warnings = [event for event in events if str(event.get("level") or "").upper() == "WARNING"]
    snapshot_errors = list(snapshot.get("errors") or []) if isinstance(snapshot.get("errors"), list) else []
    event_errors = [event for event in events if str(event.get("level") or "").upper() == "ERROR"]
    errors = snapshot_errors + event_errors
    event_counters = snapshot.get("runtime_event_counters") if isinstance(snapshot.get("runtime_event_counters"), dict) else {}
    error_state = snapshot.get("error_state") if isinstance(snapshot.get("error_state"), dict) else {}
    active_controller_errors = error_state.get("active_controller_errors") if isinstance(error_state.get("active_controller_errors"), dict) else {}
    recovered_controller_errors = error_state.get("recovered_controller_errors") if isinstance(error_state.get("recovered_controller_errors"), list) else []
    training_runtime = snapshot.get("training_runtime") if isinstance(snapshot.get("training_runtime"), dict) else {}
    training_failed_jobs = training_runtime.get("failed_jobs") if isinstance(training_runtime.get("failed_jobs"), list) else []
    registry = snapshot.get("training_registry") if isinstance(snapshot.get("training_registry"), dict) else {}
    registry_counts = registry.get("counts") if isinstance(registry.get("counts"), dict) else {}
    registry_failed_count = int(registry_counts.get("failed") or registry_counts.get("error") or 0)
    errors_since_start = max(
        int(event_counters.get("errors_since_start") or 0),
        len(snapshot_errors),
        len(event_errors),
        len(training_failed_jobs),
        registry_failed_count,
    )
    warnings_since_start = max(int(event_counters.get("warnings_since_start") or 0), len(warnings))

    shutdown_payload = shutdown_state.as_dict() if hasattr(shutdown_state, "as_dict") else dict(shutdown_state or {}) if isinstance(shutdown_state, dict) else {}

    report = {
        "schema_version": "run_report.v1",
        "report_id": f"run_report_{_utc_stamp(generated_ts)}",
        "generated_at": generated_ts,
        "generated_at_utc": _iso(generated_ts),
        "runtime": {
            "launcher_mode": system.get("launcher_mode"),
            "execution_mode": system.get("execution_mode"),
            "started_at": system.get("started_at"),
            "started_at_utc": _iso(system.get("started_at")),
            "duration_seconds": duration,
            "controllers": snapshot.get("controllers", {}),
            "shutdown": shutdown_payload,
        },
        "readiness": {
            "status": readiness.get("status", _get(snapshot, "health", "status", default="unknown")),
            "reason": readiness.get("reason", ""),
            "details": readiness,
            "blocker_report": blocker_report,
            "registry_readiness_diagnostics": _registry_readiness_diagnostics(signal_readiness_for_report if isinstance(signal_readiness_for_report, dict) else {}),
            "warmup_readiness_summary": warmup_readiness_summary(load_symbol_state(paths)),
        },
        "execution": _execution_summary(snapshot, paths),
        "execution_latency_price_freshness": {
            "path": price.get("path"),
            "exists": price.get("exists"),
            "expected_symbol_count": price.get("expected_symbol_count"),
            "priced_symbol_count": price.get("priced_symbol_count"),
            "missing_symbol_count": price.get("missing_symbol_count"),
            "stale_symbol_count": price.get("stale_symbol_count"),
            "age_seconds": price.get("age_seconds"),
            "max_price_age_seconds": price.get("max_price_age_seconds"),
            "price_source_counts": price.get("price_source_counts", {}),
            "updated_at": price.get("updated_at"),
        },
        "training": _training_summary(paths, snapshot),
        "signals": _signal_summary(paths, snapshot),
        "risk_rejection_attribution": _risk_rejection_attribution(paths, snapshot),
        "closed_trade_attribution": _closed_trade_attribution(paths, snapshot),
        "trade_excursion_attribution": _trade_excursion_report(paths, snapshot),
        "symbol_lifecycle": _symbol_lifecycle(paths),
        "adaptive_mode_changes": load_json(Path(getattr(paths, "runtime_dir", Path("data/runtime"))) / "adaptive_mode_state.json", default={}) or {},
        "warnings_errors": {
            "warnings_count": warnings_since_start,
            "errors_count": errors_since_start,
            "warnings_since_start": warnings_since_start,
            "errors_since_start": errors_since_start,
            "event_window_warnings_count": len(warnings),
            "event_window_errors_count": len(event_errors),
            "active_controller_errors_count": len(active_controller_errors),
            "active_controller_errors": active_controller_errors,
            "recovered_controller_errors_count": len(recovered_controller_errors),
            "recovered_controller_errors": recovered_controller_errors[-50:],
            "training_failure_evidence_count": max(len(training_failed_jobs), registry_failed_count),
            "training_failed_jobs": training_failed_jobs[-100:],
            "warnings": warnings[-100:],
            "errors": errors[-100:],
            "alerts": snapshot.get("alerts", []),
        },
        "control_plane": {
            "paused": control.get("paused"),
            "pause_reason": control.get("pause_reason"),
            "circuit_triggered": control.get("circuit_triggered"),
            "recent_actions": control.get("recent_actions", []),
        },
    }
    report["latest_signal_rows"] = _latest_signal_rows(paths, snapshot)
    report["enhanced_structured_report"] = _enhanced_structured_report(report)
    return report


def write_run_report(paths: Any, *, shutdown_state: Any | None = None, monitor: Any | None = None, timestamp: str | None = None) -> dict[str, Any]:
    reports_dir = Path(getattr(paths, "reports_dir", Path(getattr(paths, "runtime_dir", Path("data/runtime"))) / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = build_run_report(paths, shutdown_state=shutdown_state, monitor=monitor)
    stamp = timestamp or str(report.get("report_id", f"run_report_{_utc_stamp()}"))[len("run_report_"):]
    json_path = reports_dir / f"run_report_{stamp}.json"
    md_path = reports_dir / f"run_report_{stamp}.md"
    atomic_write_bytes(json_path, json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))
    atomic_write_bytes(md_path, _markdown(report).encode("utf-8"))
    return {"json_path": str(json_path), "md_path": str(md_path), "report": report}
