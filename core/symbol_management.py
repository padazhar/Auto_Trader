from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml

from core.shared_utils import load_json, dump_json, atomic_write_bytes


_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,30}USDT$")




def required_market_history_rows(
    *,
    fusion_intervals: list[str],
    anchor_interval: str,
    rf_min_rows: int,
) -> dict[str, int]:
    """Return the shared lifecycle history contract per fusion interval.

    TechRev and MarketData must agree on the exact number of historical rows
    required before a symbol can leave data warm-up. Keeping the rule in one
    place prevents candidate-admission logic from drifting away from the
    readiness check that ultimately governs execution eligibility.
    """
    anchor = str(anchor_interval or "15m")
    anchor_required = max(80, int(rf_min_rows) + 80)
    out: dict[str, int] = {}
    for raw_interval in fusion_intervals:
        interval = str(raw_interval)
        if not interval:
            continue
        out[interval] = anchor_required if interval == anchor else (60 if interval == "1d" else 80)
    return out

def normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def is_valid_usdt_symbol(symbol: object) -> bool:
    candidate = normalize_symbol(symbol)
    return bool(_SYMBOL_RE.fullmatch(candidate))


LIFECYCLE_OWNER = "tech_rev_controller"
LIFECYCLE_VERSION = 5
ACTIVE = "ACTIVE"
TIMED_OUT = "TIMED_OUT"
BLOCKED = "BLOCKED"
EXECUTION_ACTIVE = "execution_active"
WARMING_DATA = "warming_data"
TRAINING_REQUIRED = "training_required"
MODEL_READY = "model_ready"


def _default_state() -> dict[str, Any]:
    return {
        "lifecycle_owner": LIFECYCLE_OWNER,
        "lifecycle_version": LIFECYCLE_VERSION,
        "symbols": {},
        "ranked_candidates": [],
        "active_symbols": [],
        "candidate_symbols": [],
        "blocked_symbols": [],
        "historical_symbols": [],
        "warming_symbols": [],
        "execution_eligible_symbols": [],
        "ranked_universe_size": 20,
        "target_active_symbols": None,
        "active_symbols_source": "uninitialized",
        "updated_at": time.time(),
    }


def default_symbol_lifecycle_entry() -> dict[str, Any]:
    return {
        "status": ACTIVE,
        "consecutive_failures": 0,
        "cooldown_level": 0,
        "retry_not_before_ts": None,
        "eligible_for_auto_retry": True,
        "manually_blocked": False,
        "retry_requested": False,
        "last_error": "--",
        "last_failed_model": "--",
        "last_failed_interval": "--",
        "blocked_since_ts": None,
        "readiness_failures": 0,
        "readiness_not_ready_since_ts": None,
        "readiness_last_checked_ts": None,
        "readiness_last_reason": "--",
        "activation_stage": EXECUTION_ACTIVE,
        "warmup_requested_ts": None,
        "data_ready": False,
        "data_ready_ts": None,
        "data_ready_reason": "unchecked",
        "execution_eligible": False,
        "model_ready_ts": None,
        "model_ready_reason": "unchecked",
        "blocked_position_close_required": False,
        "blocked_position_close_status": "not_required",
        "blocked_position_close_reason": "",
        "blocked_position_close_ts": None,
    }


def normalize_symbol_lifecycle_entry(entry: dict[str, Any] | None) -> dict[str, Any]:
    payload = default_symbol_lifecycle_entry()
    if isinstance(entry, dict):
        payload.update(entry)
    payload["status"] = str(payload.get("status") or ACTIVE).upper()
    if payload["status"] not in {ACTIVE, TIMED_OUT, BLOCKED}:
        payload["status"] = ACTIVE
    try:
        payload["consecutive_failures"] = max(0, int(payload.get("consecutive_failures") or 0))
    except (TypeError, ValueError):
        payload["consecutive_failures"] = 0
    try:
        payload["cooldown_level"] = max(0, int(payload.get("cooldown_level") or 0))
    except (TypeError, ValueError):
        payload["cooldown_level"] = 0
    retry_at = payload.get("retry_not_before_ts")
    try:
        payload["retry_not_before_ts"] = float(retry_at) if retry_at is not None else None
    except (TypeError, ValueError):
        payload["retry_not_before_ts"] = None
    blocked_since = payload.get("blocked_since_ts")
    try:
        payload["blocked_since_ts"] = float(blocked_since) if blocked_since is not None else None
    except (TypeError, ValueError):
        payload["blocked_since_ts"] = None
    for field in ("readiness_not_ready_since_ts", "readiness_last_checked_ts", "warmup_requested_ts", "data_ready_ts", "model_ready_ts", "blocked_position_close_ts"):
        value = payload.get(field)
        try:
            payload[field] = float(value) if value is not None else None
        except (TypeError, ValueError):
            payload[field] = None
    try:
        payload["readiness_failures"] = max(0, int(payload.get("readiness_failures") or 0))
    except (TypeError, ValueError):
        payload["readiness_failures"] = 0
    payload["readiness_last_reason"] = str(payload.get("readiness_last_reason") or "--")
    stage = str(payload.get("activation_stage") or EXECUTION_ACTIVE).strip().lower()
    if stage not in {EXECUTION_ACTIVE, WARMING_DATA, TRAINING_REQUIRED, MODEL_READY}:
        stage = EXECUTION_ACTIVE
    payload["activation_stage"] = stage
    payload["data_ready"] = bool(payload.get("data_ready"))
    payload["data_ready_reason"] = str(payload.get("data_ready_reason") or "unchecked")
    payload["execution_eligible"] = bool(payload.get("execution_eligible"))
    payload["model_ready_reason"] = str(payload.get("model_ready_reason") or "unchecked")
    payload["blocked_position_close_required"] = bool(payload.get("blocked_position_close_required"))
    payload["blocked_position_close_status"] = str(payload.get("blocked_position_close_status") or "not_required")
    payload["blocked_position_close_reason"] = str(payload.get("blocked_position_close_reason") or "")
    payload["eligible_for_auto_retry"] = bool(payload.get("eligible_for_auto_retry"))
    payload["manually_blocked"] = bool(payload.get("manually_blocked"))
    payload["retry_requested"] = bool(payload.get("retry_requested"))
    payload["last_error"] = str(payload.get("last_error") or "--")
    payload["last_failed_model"] = str(payload.get("last_failed_model") or "--")
    payload["last_failed_interval"] = str(payload.get("last_failed_interval") or "--")
    return payload


def load_symbol_state(paths) -> dict[str, Any]:
    state = load_json(paths.symbol_state_path, default={}) or {}
    if not isinstance(state, dict):
        state = {}
    merged = _default_state()
    merged.update(state)
    if not isinstance(merged.get("symbols"), dict):
        merged["symbols"] = {}
    if not isinstance(merged.get("ranked_candidates"), list):
        merged["ranked_candidates"] = []
    if not isinstance(merged.get("active_symbols"), list):
        merged["active_symbols"] = []
    if not isinstance(merged.get("candidate_symbols"), list):
        merged["candidate_symbols"] = []
    if not isinstance(merged.get("warming_symbols"), list):
        merged["warming_symbols"] = []
    if not isinstance(merged.get("execution_eligible_symbols"), list):
        merged["execution_eligible_symbols"] = []
    if not isinstance(merged.get("blocked_symbols"), list):
        merged["blocked_symbols"] = []
    if not isinstance(merged.get("historical_symbols"), list):
        merged["historical_symbols"] = []
    try:
        merged["ranked_universe_size"] = max(1, int(merged.get("ranked_universe_size") or 20))
    except (TypeError, ValueError):
        merged["ranked_universe_size"] = 20
    try:
        target = merged.get("target_active_symbols")
        merged["target_active_symbols"] = max(1, int(target)) if target is not None else None
    except (TypeError, ValueError):
        merged["target_active_symbols"] = None
    merged["lifecycle_owner"] = str(merged.get("lifecycle_owner") or LIFECYCLE_OWNER)
    merged["lifecycle_version"] = int(merged.get("lifecycle_version") or LIFECYCLE_VERSION)
    merged["active_symbols_source"] = str(merged.get("active_symbols_source") or "uninitialized")
    merged["symbols"] = {
        normalize_symbol(symbol): normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {})
        for symbol, entry in merged["symbols"].items()
        if is_valid_usdt_symbol(symbol)
    }
    return merged


def normalize_symbol_list(symbols: list[object] | tuple[object, ...] | set[object] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_symbol in symbols or []:
        symbol = normalize_symbol(raw_symbol)
        if not is_valid_usdt_symbol(symbol) or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def save_symbol_state(paths, state: dict[str, Any]) -> None:
    payload = _default_state()
    payload.update(state)
    payload["lifecycle_owner"] = LIFECYCLE_OWNER
    payload["lifecycle_version"] = LIFECYCLE_VERSION
    payload["updated_at"] = time.time()
    payload["symbols"] = {
        normalize_symbol(symbol): normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {})
        for symbol, entry in (payload.get("symbols") or {}).items()
        if is_valid_usdt_symbol(symbol)
    }
    dump_json(paths.symbol_state_path, payload)


def load_symbol_actions(paths) -> dict[str, Any]:
    actions = load_json(paths.symbol_actions_path, default={}) or {}
    return actions if isinstance(actions, dict) else {}


def save_symbol_actions(paths, actions: dict[str, Any]) -> None:
    dump_json(paths.symbol_actions_path, actions)


def clear_symbol_actions(paths) -> None:
    save_symbol_actions(paths, {})


def symbol_entry(state: dict[str, Any], symbol: str) -> dict[str, Any]:
    key = normalize_symbol(symbol)
    symbols = state.setdefault("symbols", {})
    entry = normalize_symbol_lifecycle_entry(symbols.get(key) if isinstance(symbols.get(key), dict) else {})
    symbols[key] = entry
    return entry


def blocked_symbols(state: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for symbol, entry in state.get("symbols", {}).items():
        if normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {}).get("status") == BLOCKED:
            out.add(str(symbol).upper())
    return out


def warming_symbols(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for symbol, entry in state.get("symbols", {}).items():
        payload = normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {})
        if payload.get("status") == BLOCKED:
            continue
        if payload.get("activation_stage") in {WARMING_DATA, TRAINING_REQUIRED}:
            out.append(str(symbol).upper())
    return normalize_symbol_list(out)


def execution_eligible_symbols(state: dict[str, Any]) -> list[str]:
    active = set(normalize_symbol_list(state.get("active_symbols", [])))
    blocked = blocked_symbols(state)
    out: list[str] = []
    for symbol in normalize_symbol_list(state.get("active_symbols", [])):
        entry = symbol_entry(state, symbol)
        if symbol in blocked or entry.get("status") != ACTIVE:
            continue
        # Execution eligibility must be an explicit readiness result.  Do not
        # infer eligibility from a default activation_stage alone; that can
        # accidentally route newly-promoted or legacy symbols to execution
        # before data and anchor models are both ready.
        if bool(entry.get("execution_eligible")) and entry.get("activation_stage") == EXECUTION_ACTIVE:
            out.append(symbol)
    return [symbol for symbol in out if symbol in active]


def mark_symbol_warming_entry(entry: dict[str, Any], *, now_ts: float, reason: str = "promoted_symbol_data_warmup_required") -> dict[str, Any]:
    payload = normalize_symbol_lifecycle_entry(entry)
    if payload.get("activation_stage") == EXECUTION_ACTIVE and bool(payload.get("execution_eligible")):
        return payload
    payload["activation_stage"] = WARMING_DATA
    payload["warmup_requested_ts"] = float(now_ts)
    payload["data_ready"] = False
    payload["data_ready_ts"] = None
    payload["data_ready_reason"] = str(reason or "promoted_symbol_data_warmup_required")
    payload["execution_eligible"] = False
    payload["model_ready_ts"] = None
    payload["model_ready_reason"] = "waiting_for_data"
    return payload


def record_symbol_data_readiness_entry(entry: dict[str, Any], *, ready: bool, reason: str, now_ts: float) -> tuple[dict[str, Any], str]:
    payload = normalize_symbol_lifecycle_entry(entry)
    payload["data_ready"] = bool(ready)
    payload["data_ready_ts"] = float(now_ts)
    payload["data_ready_reason"] = str(reason or ("data_ready" if ready else "data_not_ready"))
    if ready and payload.get("activation_stage") == WARMING_DATA:
        payload["activation_stage"] = TRAINING_REQUIRED
        return payload, "training_required"
    if not ready and payload.get("activation_stage") not in {WARMING_DATA, TRAINING_REQUIRED}:
        payload["activation_stage"] = WARMING_DATA
        payload["execution_eligible"] = False
        return payload, "warming"
    return payload, "ready" if ready else "waiting"


def record_symbol_execution_readiness_entry(entry: dict[str, Any], *, ready: bool, reason: str, now_ts: float) -> tuple[dict[str, Any], str]:
    payload = normalize_symbol_lifecycle_entry(entry)
    payload["model_ready_ts"] = float(now_ts)
    payload["model_ready_reason"] = str(reason or ("ready" if ready else "model_not_ready"))
    payload["execution_eligible"] = bool(ready and payload.get("data_ready"))
    if payload["execution_eligible"]:
        payload["activation_stage"] = EXECUTION_ACTIVE
        return payload, "execution_active"
    if payload.get("data_ready"):
        payload["activation_stage"] = TRAINING_REQUIRED
        return payload, "training_required"
    payload["activation_stage"] = WARMING_DATA
    return payload, "warming"


def requested_retry_symbols(state: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for symbol, entry in state.get("symbols", {}).items():
        payload = normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {})
        if bool(payload.get("retry_requested")) and payload.get("status") != BLOCKED:
            out.append(str(symbol).upper())
    return out


def symbols_requiring_blocked_position_close(state: dict[str, Any]) -> list[str]:
    """Return BLOCKED symbols whose open positions must be force-closed."""
    out: list[str] = []
    for symbol, entry in (state.get("symbols") or {}).items():
        payload = normalize_symbol_lifecycle_entry(entry if isinstance(entry, dict) else {})
        if payload.get("status") != BLOCKED:
            continue
        if bool(payload.get("blocked_position_close_required")) and payload.get("blocked_position_close_status") in {"pending", "failed"}:
            out.append(str(symbol).upper())
    return normalize_symbol_list(out)


def record_blocked_position_close_result_entry(entry: dict[str, Any], *, closed: bool, reason: str, now_ts: float) -> dict[str, Any]:
    payload = normalize_symbol_lifecycle_entry(entry)
    if closed:
        payload["blocked_position_close_required"] = False
        payload["blocked_position_close_status"] = "closed"
    else:
        payload["blocked_position_close_required"] = True
        payload["blocked_position_close_status"] = "failed"
    payload["blocked_position_close_reason"] = str(reason or ("closed" if closed else "close_failed"))
    payload["blocked_position_close_ts"] = float(now_ts)
    return payload


def is_symbol_backed_off_entry(entry: dict[str, Any], now_ts: float) -> bool:
    payload = normalize_symbol_lifecycle_entry(entry)
    if payload["status"] == BLOCKED:
        return True
    retry_at = payload.get("retry_not_before_ts")
    return retry_at is not None and float(retry_at) > now_ts


def clear_expired_symbol_backoff_entry(entry: dict[str, Any], now_ts: float) -> tuple[dict[str, Any], bool]:
    payload = normalize_symbol_lifecycle_entry(entry)
    retry_at = payload.get("retry_not_before_ts")
    if payload["status"] == TIMED_OUT and retry_at is not None and float(retry_at) <= now_ts:
        payload["retry_not_before_ts"] = None
        payload["status"] = ACTIVE
        payload["eligible_for_auto_retry"] = True
        return payload, True
    return payload, False



def record_symbol_readiness_result_entry(
    entry: dict[str, Any],
    *,
    ready: bool,
    reason: str,
    max_readiness_failures: int,
    min_not_ready_age_sec: float,
    check_interval_sec: float,
    now_ts: float,
) -> tuple[dict[str, Any], str]:
    """Update lifecycle state from repeated PART/not-ready readiness checks.

    Readiness failures are intentionally separate from hard trainer exceptions.
    A symbol is not removed from the active universe on the first PART result;
    it is only blocked after it has remained not-ready long enough and the
    not-ready condition has been observed repeatedly.
    """
    payload = normalize_symbol_lifecycle_entry(entry)
    if ready:
        changed = bool(
            payload.get("readiness_failures")
            or payload.get("readiness_not_ready_since_ts") is not None
            or payload.get("readiness_last_checked_ts") is not None
            or str(payload.get("readiness_last_reason") or "--") != "--"
        )
        payload["readiness_failures"] = 0
        payload["readiness_not_ready_since_ts"] = None
        payload["readiness_last_checked_ts"] = None
        payload["readiness_last_reason"] = "--"
        return payload, "reset" if changed else "ready"

    if payload.get("status") == BLOCKED:
        return payload, "already_blocked"

    last_checked = payload.get("readiness_last_checked_ts")
    if last_checked is not None and (now_ts - float(last_checked)) < max(1.0, float(check_interval_sec)):
        return payload, "skipped"

    first_seen = payload.get("readiness_not_ready_since_ts")
    if first_seen is None:
        first_seen = now_ts
        payload["readiness_not_ready_since_ts"] = now_ts
    payload["readiness_last_checked_ts"] = now_ts
    payload["readiness_last_reason"] = str(reason or "readiness_not_ready")

    age_sec = max(0.0, now_ts - float(first_seen))
    if age_sec < max(0.0, float(min_not_ready_age_sec)):
        return payload, "observing"

    payload["readiness_failures"] = int(payload.get("readiness_failures") or 0) + 1
    if payload["readiness_failures"] >= max(1, int(max_readiness_failures)):
        payload["status"] = BLOCKED
        payload["consecutive_failures"] = max(
            int(payload.get("consecutive_failures") or 0),
            int(payload["readiness_failures"]),
        )
        payload["retry_not_before_ts"] = None
        payload["eligible_for_auto_retry"] = False
        payload["manually_blocked"] = True
        payload["retry_requested"] = False
        payload["blocked_since_ts"] = now_ts
        payload["last_error"] = str(reason or "readiness_not_ready")
        payload["last_failed_model"] = "readiness"
        payload["last_failed_interval"] = "all"
        payload["blocked_position_close_required"] = True
        payload["blocked_position_close_status"] = "pending"
        payload["blocked_position_close_reason"] = str(reason or "readiness_not_ready")
        payload["blocked_position_close_ts"] = now_ts
        return payload, "blocked"

    return payload, "counted"



def record_symbol_warmup_timeout_entry(
    entry: dict[str, Any],
    *,
    reason: str,
    now_ts: float,
) -> dict[str, Any]:
    """Block a symbol only after its explicit warmup timeout expires.

    This is intentionally separate from normal PART/readiness failures.  A
    warming symbol is allowed to occupy an active slot while it downloads data;
    it is replaced only when the configured warmup SLA expires.
    """
    payload = normalize_symbol_lifecycle_entry(entry)
    payload["status"] = BLOCKED
    payload["consecutive_failures"] = max(1, int(payload.get("consecutive_failures") or 0))
    payload["retry_not_before_ts"] = None
    payload["eligible_for_auto_retry"] = False
    payload["manually_blocked"] = True
    payload["retry_requested"] = False
    payload["blocked_since_ts"] = float(now_ts)
    payload["last_error"] = str(reason or "warmup_timeout")
    payload["last_failed_model"] = "warmup"
    payload["last_failed_interval"] = "all"
    payload["readiness_last_reason"] = str(reason or "warmup_timeout")
    payload["activation_stage"] = WARMING_DATA
    payload["execution_eligible"] = False
    payload["blocked_position_close_required"] = True
    payload["blocked_position_close_status"] = "pending"
    payload["blocked_position_close_reason"] = str(reason or "warmup_timeout")
    payload["blocked_position_close_ts"] = float(now_ts)
    return payload

def record_symbol_failure_entry(
    entry: dict[str, Any],
    *,
    model_type: str,
    interval: str,
    reason: str,
    max_failures_before_block: int,
    backoff_first_sec: float,
    backoff_repeat_sec: float,
    now_ts: float,
) -> tuple[dict[str, Any], str]:
    payload = normalize_symbol_lifecycle_entry(entry)
    payload["consecutive_failures"] = int(payload.get("consecutive_failures") or 0) + 1
    payload["last_error"] = str(reason or "unknown")
    payload["last_failed_model"] = str(model_type or "--")
    payload["last_failed_interval"] = str(interval or "--")
    immediate_reblock = bool(payload.get("retry_requested"))
    if immediate_reblock or payload["consecutive_failures"] >= max(1, int(max_failures_before_block)):
        payload["status"] = BLOCKED
        payload["retry_not_before_ts"] = None
        payload["eligible_for_auto_retry"] = False
        payload["manually_blocked"] = True
        payload["retry_requested"] = False
        payload["blocked_since_ts"] = now_ts
        payload["blocked_position_close_required"] = True
        payload["blocked_position_close_status"] = "pending"
        payload["blocked_position_close_reason"] = str(reason or "training_failure")
        payload["blocked_position_close_ts"] = now_ts
        return payload, "blocked"

    payload["cooldown_level"] = 1 if int(payload.get("cooldown_level") or 0) == 0 else min(int(payload.get("cooldown_level") or 0) + 1, 2)
    delay_sec = float(backoff_first_sec if payload["cooldown_level"] == 1 else backoff_repeat_sec)
    payload["retry_not_before_ts"] = now_ts + max(0.0, delay_sec)
    payload["status"] = TIMED_OUT
    payload["eligible_for_auto_retry"] = True
    return payload, "timed_out"


def record_symbol_success_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_symbol_lifecycle_entry(entry)
    payload.update(default_symbol_lifecycle_entry())
    return payload


def request_symbol_retry_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = normalize_symbol_lifecycle_entry(entry)
    payload["status"] = ACTIVE
    payload["consecutive_failures"] = 0
    payload["cooldown_level"] = 0
    payload["retry_not_before_ts"] = None
    payload["last_error"] = "manual_retry_requested"
    payload["last_failed_model"] = "--"
    payload["last_failed_interval"] = "--"
    payload["eligible_for_auto_retry"] = True
    payload["manually_blocked"] = False
    payload["retry_requested"] = True
    payload["blocked_since_ts"] = None
    payload["blocked_position_close_required"] = False
    payload["blocked_position_close_status"] = "not_required"
    payload["blocked_position_close_reason"] = ""
    payload["blocked_position_close_ts"] = None
    return payload


def timed_out_symbols_payload(symbol_entries: dict[str, dict[str, Any]], due_counts: dict[str, int], now_ts: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol, raw_entry in sorted(symbol_entries.items()):
        entry = normalize_symbol_lifecycle_entry(raw_entry)
        if entry["status"] == ACTIVE:
            continue
        retry_at = entry.get("retry_not_before_ts")
        seconds_left = max(0, int((float(retry_at) if retry_at else 0) - now_ts)) if retry_at else 0
        rows.append({
            "symbol": symbol,
            "status": entry["status"],
            "retry_at": retry_at,
            "seconds_left": seconds_left,
            "time_left": f"{seconds_left // 3600}h {(seconds_left % 3600) // 60:02d}m" if seconds_left >= 3600 else f"{seconds_left // 60}m",
            "blocked": due_counts.get(symbol, 0),
            "last_error": entry["last_error"],
            "last_failed_model": entry["last_failed_model"],
            "last_failed_interval": entry["last_failed_interval"],
            "consecutive_failures": entry["consecutive_failures"],
            "cooldown_level": entry["cooldown_level"],
            "eligible_for_auto_retry": entry["eligible_for_auto_retry"],
            "manually_blocked": entry["manually_blocked"],
            "readiness_failures": entry["readiness_failures"],
            "readiness_last_reason": entry["readiness_last_reason"],
        })
    return rows


def compute_active_symbols(
    *,
    ranked_candidates: list[str],
    target_active: int,
    pinned_symbols: list[str],
    skip_symbols: set[str],
    state: dict[str, Any],
    current_active_symbols: list[str] | None = None,
    ranked_universe_size: int = 20,
) -> list[str]:
    """Return the strict active execution universe.

    When ranked candidates exist, the allowed universe is the first
    ``ranked_universe_size`` candidates only.  Active symbols are then chosen
    from that bounded pool and capped at ``target_active``.  BLOCKED and skipped
    symbols are always excluded, and replacements are pulled from the next
    ranked symbols without ever exceeding the active limit.

    If no ranked universe has been published yet, the previous active/legacy
    bootstrap list is allowed so a fresh repository can boot before the first
    exchange universe refresh.
    """
    limit = max(1, int(target_active))
    universe_limit = max(limit, int(ranked_universe_size or 20))
    ranked_pool = normalize_symbol_list(ranked_candidates)[:universe_limit]
    ranked_pool_set = set(ranked_pool)
    strict_ranked_mode = bool(ranked_pool)
    blocked = blocked_symbols(state)
    retry_requested = requested_retry_symbols(state)
    active: list[str] = []

    def allowed(symbol: str) -> bool:
        if not is_valid_usdt_symbol(symbol):
            return False
        if symbol in skip_symbols or symbol in blocked or symbol in active:
            return False
        if strict_ranked_mode and symbol not in ranked_pool_set:
            return False
        return True

    def add(symbol: str) -> None:
        symbol = normalize_symbol(symbol)
        if allowed(symbol):
            active.append(symbol)

    if strict_ranked_mode:
        sources = (retry_requested, pinned_symbols, current_active_symbols or [], ranked_pool)
    else:
        sources = (retry_requested, pinned_symbols, current_active_symbols or [], ranked_candidates)

    for source in sources:
        for symbol in source:
            add(symbol)
            if len(active) >= limit:
                return active[:limit]
    return active[:limit]



def warmup_readiness_summary(state: dict[str, Any]) -> dict[str, Any]:
    """Return diagnostic-only warmup readiness details for active symbols.

    This does not change symbol lifecycle state. It exists so operators can
    distinguish a safe market-data warmup block from registry/model blockers
    when the UI shows less than the target execution-ready universe.
    """
    active = normalize_symbol_list(state.get("active_symbols") or [])
    symbols_state = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    warming_rows: list[dict[str, Any]] = []
    for symbol in active:
        entry = normalize_symbol_lifecycle_entry(symbols_state.get(symbol) if isinstance(symbols_state, dict) else None)
        if entry.get("activation_stage") != WARMING_DATA:
            continue
        warming_rows.append({
            "symbol": symbol,
            "activation_stage": entry.get("activation_stage"),
            "data_ready": bool(entry.get("data_ready", False)),
            "data_ready_reason": str(entry.get("data_ready_reason") or ""),
            "warmup_requested_ts": entry.get("warmup_requested_ts"),
            "execution_eligible": bool(entry.get("execution_eligible", False)),
            "readiness_last_reason": str(entry.get("readiness_last_reason") or ""),
        })
    return {
        "schema_version": "symbol_warmup_readiness.v1",
        "active_symbol_count": len(active),
        "warming_symbol_count": len(warming_rows),
        "warming_symbols": [row["symbol"] for row in warming_rows],
        "rows": warming_rows,
    }

def classify_symbol_universe(
    *,
    ranked_candidates: list[str],
    active_symbols: list[str],
    state: dict[str, Any],
    ranked_universe_size: int = 20,
) -> dict[str, list[str]]:
    """Return active/candidate/blocked/historical buckets for operators."""
    universe_limit = max(1, int(ranked_universe_size or 20))
    ranked_pool = normalize_symbol_list(ranked_candidates)[:universe_limit]
    active = normalize_symbol_list(active_symbols)
    active_set = set(active)
    blocked = sorted(blocked_symbols(state))
    blocked_set = set(blocked)
    symbols_state = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    warming = [
        symbol
        for symbol in active
        if symbol in symbols_state and normalize_symbol_lifecycle_entry(symbols_state.get(symbol) or {}).get("activation_stage") == WARMING_DATA
    ]
    eligible = [
        symbol
        for symbol in active
        if symbol not in blocked_set
        and bool(normalize_symbol_lifecycle_entry(symbols_state.get(symbol) or {}).get("execution_eligible"))
        and normalize_symbol_lifecycle_entry(symbols_state.get(symbol) or {}).get("activation_stage") == EXECUTION_ACTIVE
    ]
    candidates = [symbol for symbol in ranked_pool if symbol not in active_set and symbol not in blocked_set]
    known = set(ranked_pool) | active_set | blocked_set
    historical = sorted(
        symbol
        for symbol in normalize_symbol_list(list((state.get("symbols") or {}).keys()))
        if symbol not in known
    )
    return {
        "active_symbols": active,
        "candidate_symbols": candidates,
        "warming_symbols": warming,
        "execution_eligible_symbols": eligible,
        "blocked_symbols": blocked,
        "historical_symbols": historical,
    }


def _readiness_rows_by_symbol(readiness_snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return readiness rows keyed by symbol from a signal readiness payload.

    The signal readiness snapshot is the same RF/GBM anchor authority used by
    the signal loop.  Symbol lifecycle exports must not claim execution
    eligibility that contradicts this payload when it is available.
    """
    payload = readiness_snapshot if isinstance(readiness_snapshot, dict) else {}
    rows = payload.get("execution_readiness_matrix") or payload.get("rows") or []
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = normalize_symbol(row.get("symbol"))
        if not is_valid_usdt_symbol(symbol):
            continue
        out[symbol] = row
    return out


def reconcile_symbol_lifecycle_with_readiness_snapshot(
    state: dict[str, Any],
    readiness_snapshot: dict[str, Any] | None,
    *,
    now_ts: float | None = None,
) -> tuple[dict[str, Any], bool]:
    """Project authoritative signal readiness into lifecycle eligibility.

    This function does not create readiness.  It only prevents an already
    published symbol lifecycle view from claiming execution eligibility when
    the canonical RF/GBM anchor readiness snapshot says the symbol is blocked,
    and it marks paper-ready symbols eligible only when data is also ready.
    """
    rows_by_symbol = _readiness_rows_by_symbol(readiness_snapshot)
    if not rows_by_symbol:
        return state, False

    changed = False
    active = normalize_symbol_list(state.get("active_symbols") or [])
    symbols_state = state.setdefault("symbols", {})
    timestamp = float(now_ts if now_ts is not None else time.time())
    eligible: list[str] = []

    for symbol in active:
        row = rows_by_symbol.get(symbol)
        if row is None:
            continue
        entry = normalize_symbol_lifecycle_entry(symbols_state.get(symbol) if isinstance(symbols_state.get(symbol), dict) else {})
        before = dict(entry)
        row_ready = bool(row.get("ready") or row.get("execution_ready"))
        reason = str(row.get("reason") or ("anchor_ready" if row_ready else "anchor_readiness_not_ready"))
        entry["model_ready_ts"] = timestamp
        entry["model_ready_reason"] = reason

        if entry.get("status") != ACTIVE:
            entry["execution_eligible"] = False
        elif row_ready and bool(entry.get("data_ready")):
            entry["execution_eligible"] = True
            entry["activation_stage"] = EXECUTION_ACTIVE
            eligible.append(symbol)
        else:
            entry["execution_eligible"] = False
            if bool(entry.get("data_ready")):
                entry["activation_stage"] = TRAINING_REQUIRED
            else:
                entry["activation_stage"] = WARMING_DATA

        symbols_state[symbol] = entry
        if entry != before:
            changed = True

    previous_eligible = normalize_symbol_list(state.get("execution_eligible_symbols") or [])
    state["execution_eligible_symbols"] = eligible
    if previous_eligible != eligible:
        changed = True
    return state, changed


def write_tradable_symbols_yaml(paths, *, symbols: list[str], pinned_symbols: list[str], skip_symbols: set[str], source: str) -> None:
    unique_symbols: list[str] = []
    for symbol in symbols:
        symbol = normalize_symbol(symbol)
        if not is_valid_usdt_symbol(symbol) or symbol in skip_symbols or symbol in unique_symbols:
            continue
        unique_symbols.append(symbol)
    payload = {
        "symbols": unique_symbols,
        "pinned_symbols": [str(s).upper() for s in pinned_symbols],
        "skip_symbols": sorted(str(s).upper() for s in skip_symbols),
        "universe_source": source,
        "last_refresh_ts": int(time.time()),
    }
    rendered = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    atomic_write_bytes(Path(paths.tradable_symbols_path), rendered)
