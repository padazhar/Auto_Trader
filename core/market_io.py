from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from core.shared_utils import load_yaml


def _normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _target_active_symbols(paths) -> int | None:
    try:
        value = int(getattr(getattr(paths, "config", None), "symbols", None).target_active_symbols)
    except Exception:
        value = None
    return value if value and value > 0 else None


def _unique_symbols(symbols: list[object]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols:
        symbol = _normalize_symbol(raw_symbol)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


def _read_symbol_state_active_symbols(paths) -> list[str]:
    state_path = getattr(paths, "symbol_state_path", None)
    if state_path is None:
        return []
    state_path = Path(state_path)
    if not state_path.exists():
        return []
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    raw_symbols = payload.get("active_symbols", [])
    if not isinstance(raw_symbols, list):
        return []
    return _unique_symbols(raw_symbols)


def _read_tradable_symbols(paths) -> list[str]:
    if not hasattr(paths, "tradable_symbols_path"):
        return []
    tradable = load_yaml(paths.tradable_symbols_path) or {}
    if not isinstance(tradable, dict):
        return []
    raw = tradable.get("symbols", [])
    if not isinstance(raw, list):
        return []
    return _unique_symbols(raw)


def _read_market_db_symbols(paths) -> list[str]:
    market_dir = Path(getattr(paths, "market_storage_dir", Path(".")))
    if not market_dir.exists():
        return []
    return _unique_symbols([db_path.stem for db_path in market_dir.glob("*.db")])


def load_active_universe_symbols(paths, *, limit: int | None = None, include_db_fallback: bool = True) -> list[str]:
    """Return the canonical runtime active universe.

    The active universe is resolved in the same order used by the runtime
    controllers: symbol-state first, then tradable_symbols.yaml, then market DBs
    only as a last-resort bootstrap fallback. This prevents execution from
    expanding to old database symbols that the market-data price snapshot is not
    pricing.
    """
    symbols = _read_symbol_state_active_symbols(paths)
    if not symbols:
        symbols = _read_tradable_symbols(paths)
    if not symbols and include_db_fallback:
        symbols = _read_market_db_symbols(paths)

    target = _target_active_symbols(paths)
    effective_limit = limit if limit is not None else target
    if effective_limit is not None:
        try:
            max_items = max(1, int(effective_limit))
            symbols = symbols[:max_items]
        except Exception:
            pass
    return symbols


def _candidate_symbols(paths) -> list[str]:
    return load_active_universe_symbols(paths, include_db_fallback=True)


def load_open_position_symbols(paths) -> list[str]:
    """Return symbols with currently open runtime positions.

    The market-data price snapshot is the execution layer's authoritative
    mark-price source. Active-trading symbols alone are not enough after symbol
    rotation: a position can remain open after its symbol leaves the active
    universe. This helper reads the lightweight portfolio_state.json contract
    without importing execution classes, avoiding circular runtime dependencies.
    """
    portfolio_path = getattr(paths, "portfolio_state_path", None)
    if portfolio_path is None:
        return []
    portfolio_path = Path(portfolio_path)
    if not portfolio_path.exists():
        return []
    try:
        payload = json.loads(portfolio_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    positions = payload.get("positions", {})
    raw_symbols: list[object] = []
    if isinstance(positions, dict):
        for symbol, row in positions.items():
            if isinstance(row, dict):
                raw_symbols.append(row.get("symbol") or symbol)
            else:
                raw_symbols.append(symbol)
    elif isinstance(positions, list):
        for row in positions:
            if isinstance(row, dict):
                raw_symbols.append(row.get("symbol"))
    return _unique_symbols(raw_symbols)


def load_runtime_price_symbols(paths, *, active_symbols: list[object] | None = None, limit: int | None = None) -> list[str]:
    """Return symbols that must be priced at runtime.

    Runtime prices must cover both the active universe and any open positions.
    This keeps stop-loss/take-profit monitoring and UI PnL working even when a
    symbol rotates out of the active universe while a position is still open.
    """
    if active_symbols is None:
        active = load_active_universe_symbols(paths, limit=limit, include_db_fallback=True)
    else:
        active = _unique_symbols(list(active_symbols))
    open_symbols = load_open_position_symbols(paths)
    return _unique_symbols(list(active) + list(open_symbols))


def _market_latest_path(paths) -> Path | None:
    candidate = getattr(paths, "market_latest_snapshot_path", None)
    if candidate is not None:
        return Path(candidate)
    runtime_dir = getattr(paths, "runtime_dir", None)
    if runtime_dir is None:
        return None
    return Path(runtime_dir) / "market" / "latest.json"


def load_market_latest_snapshot(paths) -> dict[str, Any]:
    snapshot_path = _market_latest_path(paths)
    if snapshot_path is None or not snapshot_path.exists():
        return {}
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_market_latest_snapshot(paths, snapshot: dict[str, Any]) -> None:
    """Atomically persist the runtime market price snapshot."""
    snapshot_path = _market_latest_path(paths)
    if snapshot_path is None:
        return
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = snapshot_path.with_name(f"{snapshot_path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")
    os.replace(tmp_path, snapshot_path)


def _coerce_positive_float(value: object) -> float | None:
    try:
        price = float(value)
    except Exception:
        return None
    if price <= 0:
        return None
    return price


def normalize_price_components(
    price: object,
    *,
    price_source: str = "unknown",
    price_ts_ms: object | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Return the canonical price payload used by inference and execution.

    Step 2 standardizes decision components so downstream code does not have
    to guess between ``price``, ``latest_price`` and ``reference_price``.
    A non-positive or non-numeric price is represented explicitly as
    unavailable instead of being silently converted to zero.
    """
    normalized_price = _coerce_positive_float(price)
    ts: int | None
    try:
        ts = int(price_ts_ms) if price_ts_ms is not None else None
    except Exception:
        ts = None

    if normalized_price is None:
        return {
            "price": None,
            "latest_price": None,
            "reference_price": None,
            "price_source": str(price_source or fallback_reason or "unavailable"),
            "price_ts_ms": ts,
            "price_available": False,
            "price_blocker": str(fallback_reason or price_source or "price_unavailable"),
        }

    return {
        "price": normalized_price,
        "latest_price": normalized_price,
        "reference_price": normalized_price,
        "price_source": str(price_source or "unknown"),
        "price_ts_ms": ts,
        "price_available": True,
        "price_blocker": "",
    }




def _runtime_price_max_age_seconds(paths, default: float = 180.0) -> float:
    runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
    try:
        value = float(getattr(runtime_cfg, "price_max_age_seconds", default))
    except Exception:
        value = default
    return max(1.0, value)


def _coerce_timestamp_ms(value: object) -> int | None:
    try:
        ts = int(float(value)) if value is not None else None
    except Exception:
        return None
    return ts if ts and ts > 0 else None


def _price_age_seconds(price_ts_ms: int | None, *, now_ms: int | None = None) -> float | None:
    if price_ts_ms is None:
        return None
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    ts_ms = int(price_ts_ms * 1000) if price_ts_ms < 10_000_000_000 else int(price_ts_ms)
    return max(0.0, (now - ts_ms) / 1000.0)


def latest_symbol_price_status(
    paths,
    symbol: str,
    *,
    max_age_seconds: float | None = None,
    now_ms: int | None = None,
    require_fresh: bool = True,
) -> dict[str, Any]:
    """Return the execution-safe runtime price status for one symbol."""
    normalized_symbol = _normalize_symbol(symbol)
    snapshot = load_market_latest_snapshot(paths)
    symbols = snapshot.get("symbols") if isinstance(snapshot.get("symbols"), dict) else {}
    row = symbols.get(normalized_symbol) if isinstance(symbols, dict) else None
    max_age = max(1.0, float(max_age_seconds if max_age_seconds is not None else _runtime_price_max_age_seconds(paths)))
    base: dict[str, Any] = {
        "symbol": normalized_symbol,
        "price": None,
        "latest_price": None,
        "reference_price": None,
        "price_source": "snapshot_missing",
        "price_ts_ms": None,
        "price_available": False,
        "price_fresh": False,
        "price_age_seconds": None,
        "max_age_seconds": max_age,
        "reason": "snapshot_missing",
    }
    if not isinstance(row, dict):
        return base

    # Open-trade monitoring must prefer exchange mark price over generic
    # display aliases. The market snapshot stores several aliases for UI and
    # backward compatibility, but execution freshness and SL/TP decisions should
    # use the most liquidation-relevant price available first.
    for field, source in (
        ("mark_price", "runtime_market_mark"),
        ("mark_price_average", "runtime_market_mark_average"),
        ("primary_price", "runtime_market_primary"),
        ("latest_price", "runtime_market_latest"),
        ("price", "runtime_market_price"),
        ("reference_price", "runtime_market_reference"),
        ("ticker_price", "runtime_market_ticker"),
        ("index_price", "runtime_market_index"),
        ("db_close", "runtime_market_db_close"),
        ("close", "market_latest_snapshot"),
    ):
        price = _coerce_positive_float(row.get(field))
        if price is None:
            continue
        ts = _coerce_timestamp_ms(row.get("price_ts_ms") or row.get("event_time_ms") or row.get("timestamp"))
        age = _price_age_seconds(ts, now_ms=now_ms)
        fresh = bool(age is not None and age <= max_age)
        reason = "price_ready"
        if require_fresh and ts is None:
            reason = "price_timestamp_missing"
        elif require_fresh and not fresh:
            reason = "price_stale"
        source_value = str(row.get("price_source") or row.get("source") or source)
        return {
            "symbol": normalized_symbol,
            "price": price,
            "latest_price": price,
            "reference_price": price,
            "price_source": source_value or source,
            "selected_price_field": field,
            "price_ts_ms": ts,
            "price_available": True,
            "price_fresh": fresh if require_fresh else True,
            "price_age_seconds": age,
            "max_age_seconds": max_age,
            "reason": reason,
        }

    reason = str(row.get("price_source") or row.get("source") or "snapshot_unpriced")
    base.update({"price_source": reason, "reason": reason})
    return base


def latest_symbol_price(paths, symbol: str) -> tuple[float | None, str, int | None]:
    status = latest_symbol_price_status(paths, symbol, require_fresh=False)
    price = _coerce_positive_float(status.get("price"))
    source = str(status.get("price_source") or status.get("reason") or "snapshot_missing")
    ts = _coerce_timestamp_ms(status.get("price_ts_ms"))
    return price, source, ts

def load_enriched_market_rows(paths, limit: int = 64) -> list[dict[str, Any]]:
    anchor_interval = str(
        getattr(getattr(paths, "config", None), "frames", None).anchor_interval
        if getattr(getattr(paths, "config", None), "frames", None)
        else "15m"
    )
    table_name = f"klines_{anchor_interval}"
    runtime_snapshot = load_market_latest_snapshot(paths)
    runtime_symbols = runtime_snapshot.get("symbols") if isinstance(runtime_snapshot.get("symbols"), dict) else {}
    rows: list[dict[str, Any]] = []
    for symbol in load_active_universe_symbols(paths, limit=max(1, int(limit)), include_db_fallback=True):
        row: dict[str, Any] = {"symbol": symbol, "state_domain": "market_data", "state_owner": "market_data_controller"}
        runtime_row = runtime_symbols.get(symbol) if isinstance(runtime_symbols, dict) else None
        if isinstance(runtime_row, dict):
            row.update(
                {
                    "price": runtime_row.get("price"),
                    "latest_price": runtime_row.get("latest_price"),
                    "reference_price": runtime_row.get("reference_price"),
                    "primary_price": runtime_row.get("primary_price"),
                    "mark_price": runtime_row.get("mark_price"),
                    "ticker_price": runtime_row.get("ticker_price"),
                    "price_source": runtime_row.get("price_source"),
                    "price_ts_ms": runtime_row.get("price_ts_ms"),
                }
            )
        db_path = Path(paths.market_db_path(symbol)) if hasattr(paths, "market_db_path") else None
        if db_path is None or not db_path.exists():
            rows.append(row)
            continue
        try:
            with sqlite3.connect(db_path) as conn:
                latest = conn.execute(
                    f'SELECT timestamp, close, volume FROM "{table_name}" ORDER BY timestamp DESC LIMIT 1'
                ).fetchone()
        except Exception:
            latest = None
        if latest:
            row.update(
                {
                    "timestamp": int(latest[0]) if latest[0] is not None else None,
                    "close": float(latest[1]) if latest[1] is not None else None,
                    "volume": float(latest[2]) if latest[2] is not None else None,
                }
            )
        rows.append(row)
    return rows[: max(1, int(limit))]

def market_latest_snapshot_diagnostics(paths, *, stale_after_seconds: float = 180.0) -> dict[str, Any]:
    """Return readiness-grade diagnostics for the runtime latest-price snapshot."""
    snapshot_path = _market_latest_path(paths)
    now_ms = int(time.time() * 1000)
    diagnostics: dict[str, Any] = {
        "exists": False,
        "path": str(snapshot_path) if snapshot_path is not None else "",
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
        "status": "missing",
    }
    if snapshot_path is None or not Path(snapshot_path).exists():
        return diagnostics

    diagnostics["exists"] = True
    payload = load_market_latest_snapshot(paths)
    if not isinstance(payload, dict) or not payload:
        diagnostics["status"] = "unreadable"
        return diagnostics

    symbols = payload.get("symbols") if isinstance(payload.get("symbols"), dict) else {}
    price_symbols = [str(symbol).strip().upper() for symbol in list(payload.get("price_symbols") or []) if symbol]
    active_symbols = [str(symbol).strip().upper() for symbol in list(payload.get("active_symbols") or []) if symbol]
    if not price_symbols:
        price_symbols = list(active_symbols)
    if not price_symbols:
        price_symbols = [str(symbol).strip().upper() for symbol in symbols.keys() if symbol]

    expected = int(payload.get("expected_symbol_count") or len(price_symbols) or len(symbols) or 0)
    missing_symbols = [str(symbol).strip().upper() for symbol in list(payload.get("missing_symbols") or []) if symbol]
    priced_symbols: list[str] = []
    stale_symbols: list[str] = []
    price_source_counts: dict[str, int] = {}
    max_price_age_seconds: float | None = None

    for symbol in price_symbols:
        row = symbols.get(symbol) if isinstance(symbols, dict) else None
        if not isinstance(row, dict):
            if symbol not in missing_symbols:
                missing_symbols.append(symbol)
            continue
        price = None
        for field in ("price", "latest_price", "reference_price", "primary_price", "mark_price", "ticker_price", "db_close", "close"):
            price = _coerce_positive_float(row.get(field))
            if price is not None:
                break
        if price is None:
            if symbol not in missing_symbols:
                missing_symbols.append(symbol)
            continue
        priced_symbols.append(symbol)
        source = str(row.get("price_source") or row.get("source") or "unknown")
        price_source_counts[source] = price_source_counts.get(source, 0) + 1
        ts_raw = row.get("price_ts_ms") or row.get("event_time_ms") or row.get("timestamp")
        try:
            ts_ms = int(ts_raw) if ts_raw is not None else None
        except Exception:
            ts_ms = None
        if ts_ms:
            age_seconds = max(0.0, (now_ms - ts_ms) / 1000.0)
            max_price_age_seconds = age_seconds if max_price_age_seconds is None else max(max_price_age_seconds, age_seconds)
            if age_seconds > float(stale_after_seconds):
                stale_symbols.append(symbol)

    created_at_raw = payload.get("created_at_ms") or payload.get("created_ms")
    try:
        created_at_ms = int(created_at_raw) if created_at_raw is not None else None
    except Exception:
        created_at_ms = None

    all_symbols_priced = bool(payload.get("all_symbols_priced")) if "all_symbols_priced" in payload else bool(expected > 0 and len(missing_symbols) == 0 and len(priced_symbols) >= expected)
    ready = bool(expected > 0 and all_symbols_priced and len(missing_symbols) == 0 and len(stale_symbols) == 0 and len(priced_symbols) >= expected)
    diagnostics.update({
        "source": str(payload.get("source") or ""),
        "reason": str(payload.get("reason") or ""),
        "expected_symbol_count": expected,
        "priced_symbol_count": int(payload.get("priced_symbol_count") or len(priced_symbols)),
        "missing_symbol_count": len(missing_symbols),
        "stale_symbol_count": len(stale_symbols),
        "all_symbols_priced": all_symbols_priced,
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
        "ready": ready,
        "status": "ready" if ready else "degraded",
    })
    return diagnostics

