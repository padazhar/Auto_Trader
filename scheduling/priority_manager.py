from __future__ import annotations

import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.artifact_contract import load_json, resolve_active_manifest_path, resolve_artifact_path, validate_artifact_bundle
from core.job_definition import TrainingJob
from core.symbol_management import blocked_symbols, is_valid_usdt_symbol, load_symbol_state, normalize_symbol


DATA_READY_REASON = "market_data_ready"
_DEFAULT_RETRAIN_CYCLES = {"15m": 1, "5m": 2, "3m": 4, "1m": 7}
_DEFAULT_RETRAIN_MIN_NEW_BARS = {
    "rf": dict(_DEFAULT_RETRAIN_CYCLES),
    "gbm": dict(_DEFAULT_RETRAIN_CYCLES),
    "dqn": {"15m": 1, "5m": 2, "3m": 4, "1m": 7},
    "hmm": {"15m": 1, "5m": 2, "3m": 4, "1m": 7},
}
_INTERVAL_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900}


def _now_ts() -> float:
    return time.time()


def _is_enabled_for_job(paths, model_type: str, interval: str) -> bool:
    training_cfg = paths.config.training
    if model_type == "rf":
        return bool(training_cfg.rf.enabled)
    if model_type == "gbm":
        return bool(training_cfg.gbm.enabled)
    if model_type == "dqn":
        if not bool(training_cfg.dqn.enabled):
            return False
        allowed = list(training_cfg.dqn.allowed_intervals or [])
        return interval in allowed if allowed else True
    if model_type == "hmm":
        hmm_cfg = getattr(training_cfg, "hmm", None)
        return bool(getattr(hmm_cfg, "enabled", False) and getattr(hmm_cfg, "candidate_training_enabled", False))
    return False


def _model_cfg(paths, model_type: str):
    training_cfg = paths.config.training
    if model_type == "rf":
        return training_cfg.rf
    if model_type == "gbm":
        return training_cfg.gbm
    if model_type == "dqn":
        return training_cfg.dqn
    if model_type == "hmm":
        return training_cfg.hmm
    raise ValueError(f"unsupported model_type: {model_type}")


def _max_age_minutes(paths, model_type: str) -> int:
    return int(_model_cfg(paths, model_type).max_age_minutes)


def _legacy_manifest_path(paths, model_type: str, symbol: str, interval: str) -> Path:
    if model_type == "rf":
        return paths.rf_model_dir(symbol, interval) / "manifest.json"
    if model_type == "gbm":
        return paths.gbm_model_dir(symbol, interval) / "manifest.json"
    if model_type == "dqn":
        return paths.dqn_model_dir(symbol, interval) / "manifest.json"
    if model_type == "hmm":
        return paths.hmm_model_dir(symbol, interval) / "manifest.json"
    raise ValueError(f"unsupported model_type: {model_type}")


def _load_symbols(paths) -> list[str]:
    """Load scheduler symbols from the canonical lifecycle state.

    In production, ``symbol_state.json`` remains the sole scheduler authority so
    stale or blocked compatibility exports cannot revive removed symbols. For
    lightweight scripts/tests that do not expose ``symbol_state_path``, fall back
    to ``tradable_symbols.yaml`` only as a bootstrap compatibility source.
    """
    def _load_tradable_export() -> list[str]:
        raw_cfg = {}
        try:
            raw_cfg = paths.load_yaml(paths.tradable_symbols_path) or {}
        except Exception:
            raw_cfg = {}
        raw_symbols = raw_cfg.get("symbols", []) if isinstance(raw_cfg, dict) else []
        seen: set[str] = set()
        out: list[str] = []
        for raw_symbol in raw_symbols if isinstance(raw_symbols, list) else []:
            symbol = normalize_symbol(raw_symbol)
            if not is_valid_usdt_symbol(symbol) or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    if not hasattr(paths, "symbol_state_path"):
        return _load_tradable_export()

    symbol_state_path = Path(getattr(paths, "symbol_state_path"))
    if not symbol_state_path.exists():
        return _load_tradable_export()

    state = load_symbol_state(paths)
    blocked = blocked_symbols(state)
    active_from_state = state.get("active_symbols", [])
    if not isinstance(active_from_state, list) or not active_from_state:
        return []

    seen: set[str] = set()
    out: list[str] = []
    for raw_symbol in active_from_state:
        symbol = normalize_symbol(raw_symbol)
        if not is_valid_usdt_symbol(symbol) or symbol in blocked or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _interval_priority(interval: str, interval_order: list[str]) -> int:
    try:
        return interval_order.index(interval)
    except ValueError:
        return len(interval_order) + 100


def _model_priority(model_type: str) -> int:
    ordering = {"rf": 0, "gbm": 1, "dqn": 2, "hmm": 3}
    return ordering.get(model_type, 99)


def _scheduler_list(paths, name: str, default: list[str]) -> list[str]:
    scheduling_cfg = getattr(getattr(paths, "config", None), "scheduling", None)
    raw = getattr(scheduling_cfg, name, None) if scheduling_cfg is not None else None
    if not isinstance(raw, list) or not raw:
        return list(default)
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out or list(default)


def _scheduler_bool(paths, name: str, default: bool) -> bool:
    scheduling_cfg = getattr(getattr(paths, "config", None), "scheduling", None)
    raw = getattr(scheduling_cfg, name, default) if scheduling_cfg is not None else default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


def _has_active_manifest(paths, model_type: str, symbol: str, interval: str) -> bool:
    try:
        model_role = "direction" if str(model_type).lower() == "gbm" else None
        manifest = resolve_active_manifest_path(paths, model_type, symbol, interval, model_role=model_role)
        return bool(manifest is not None and manifest.exists())
    except Exception:
        return False


def _manifest_gap_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    return (
        "manifest_missing" in text
        or "active_manifest_missing" in text
        or "manifest_unreadable" in text
        or "artifact_missing" in text
        or "deployment_stage" in text
        or "feature_incompatible" in text
    )


def _parse_anchor_recovery_blockers(reason: str, *, default_interval: str) -> list[dict[str, str]]:
    """Extract RF/GBM anchor blockers that should force retraining.

    Step 7 must repair symbols that are blocked because the current active or
    candidate anchor models are not execution-valid.  The signal/readiness
    layers may publish either the older interval-explicit format::

        anchor_readiness_not_ready:rf:15m:feature_incompatible:feature_set_missing

    or the compact dashboard format::

        anchor_training_incomplete:rf:deployment_stage:candidate,gbm:feature_incompatible:feature_set_missing

    The compact format intentionally omits the interval, so the configured
    anchor interval is used.  This helper only schedules retraining; it never
    marks a model approved or execution-ready.
    """
    text = str(reason or "")
    if not text:
        return []
    normalized = text
    for prefix in ("anchor_readiness_not_ready:", "anchor_training_incomplete:"):
        normalized = normalized.replace(prefix, "")
    blockers: list[dict[str, str]] = []
    fallback_interval = str(default_interval or "15m")
    for fragment in normalized.split(","):
        raw = fragment.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) < 2:
            continue
        model_type = parts[0].strip().lower()
        if model_type not in {"rf", "gbm"}:
            continue
        if len(parts) >= 3 and re.fullmatch(r"\d+[mhdw]", parts[1].strip().lower()):
            interval = parts[1].strip()
            detail = ":".join(parts[2:]).strip() or "unknown"
        else:
            interval = fallback_interval
            detail = ":".join(parts[1:]).strip() or "unknown"
        if not _manifest_gap_reason(detail):
            continue
        blockers.append({"model_type": model_type, "interval": interval, "reason": detail})
    return blockers


def _affected_anchor_recovery_jobs(
    paths,
    registry,
    *,
    symbols: list[str],
    market_state_cache: dict[str, dict[str, dict[str, int]]],
    now_ts: float,
    interval_order: list[str],
) -> list[TrainingJob]:
    """Build highest-priority retraining jobs for active non-ready anchors.

    Normal scheduler freshness checks can treat a freshly trained candidate or
    feature-incompatible manifest as not due.  That is correct for ordinary
    cadence scheduling but wrong for recovery: execution remains fail-closed
    forever unless the affected RF/GBM anchors are retrained and requalified.
    This function closes that Step 7 gap by translating readiness blockers into
    retraining jobs while preserving every registry/readiness gate.
    """
    if not hasattr(paths, "symbol_state_path"):
        return []
    state = load_symbol_state(paths)
    state_symbols = state.get("symbols") if isinstance(state.get("symbols"), dict) else {}
    eligible = {normalize_symbol(sym) for sym in state.get("execution_eligible_symbols", []) if is_valid_usdt_symbol(sym)}
    anchor_interval = str(getattr(getattr(paths.config, "frames", None), "anchor_interval", "15m") or "15m")
    allowed_symbols = {normalize_symbol(sym) for sym in symbols}
    jobs: list[TrainingJob] = []
    seen: set[tuple[str, str, str]] = set()

    for symbol_idx, symbol in enumerate(symbols):
        symbol = normalize_symbol(symbol)
        if not symbol or symbol not in allowed_symbols or symbol in eligible:
            continue
        entry_state = state_symbols.get(symbol) if isinstance(state_symbols.get(symbol), dict) else {}
        reason_text = str(entry_state.get("model_ready_reason") or entry_state.get("readiness_last_reason") or "")
        blockers = _parse_anchor_recovery_blockers(reason_text, default_interval=anchor_interval)
        if not blockers:
            continue
        market_state = market_state_cache.setdefault(symbol, _load_market_state(paths, symbol))
        for blocker in blockers:
            model_type = blocker["model_type"]
            interval = blocker["interval"]
            if not _is_enabled_for_job(paths, model_type, interval):
                continue
            ready, readiness_reason, readiness_meta = _market_data_ready(
                paths,
                symbol=symbol,
                interval=interval,
                model_type=model_type,
                market_state=market_state,
            )
            if not ready:
                continue
            registry_entry = registry.get_entry(model_type, symbol, interval)
            if registry_entry.status in {"queued", "running"}:
                continue
            if _failure_cooldown_active(paths, registry_entry, now_ts):
                continue
            if (
                registry_entry.status == "success"
                and bool(getattr(registry_entry, "promotion_attempted", False))
                and not bool(getattr(registry_entry, "promotion_approved", False))
            ):
                due, due_reason = _job_due_reason(
                    paths,
                    registry_entry,
                    _max_age_minutes(paths, model_type),
                    now_ts,
                    model_type=model_type,
                    symbol=symbol,
                    interval=interval,
                    market_state=market_state,
                )
                if not due:
                    continue
            key = (model_type, symbol, interval)
            if key in seen:
                continue
            seen.add(key)
            priority = -1_000_000 + _model_priority(model_type) * 10_000 + _interval_priority(interval, interval_order) * 100 + symbol_idx
            jobs.append(
                TrainingJob(
                    model_type=model_type,
                    symbol=symbol,
                    interval=interval,
                    priority=priority,
                    requested_at=now_ts,
                    reason=f"anchor_recovery:{blocker['reason']}",
                    metadata={
                        "market_readiness": readiness_meta,
                        "recovery_scope": "affected_anchor",
                        "source_reason": reason_text,
                        "market_readiness_reason": readiness_reason,
                    },
                )
            )
    return jobs


def _fairness_priority_tier(paths, *, model_type: str, symbol: str, interval: str, reason: str) -> int:
    """Return a starvation-resistant scheduler tier.

    Anchor RF/GBM gaps remain first because they determine basic execution
    eligibility. Once an anchor has an active manifest, repeated 15m refreshes
    are pushed behind missing/stale fast-timeframe RF/GBM work so 1m/3m jobs do
    not starve behind heavier anchor/5m retraining. DQN remains optional and is
    intentionally behind RF/GBM readiness work.
    """
    model_type = str(model_type).lower()
    interval = str(interval)
    fast_tfs = set(_scheduler_list(paths, "fairness_fast_timeframes", ["1m", "3m"]))
    anchor_tfs = set(_scheduler_list(paths, "fairness_anchor_timeframes", ["15m"]))
    secondary_tfs = set(_scheduler_list(paths, "fairness_secondary_timeframes", ["5m"]))
    deprioritize_anchor = _scheduler_bool(paths, "deprioritize_anchor_after_ready", True)
    is_manifest_gap = _manifest_gap_reason(reason)
    is_core_model = model_type in {"rf", "gbm"}
    active_manifest_exists = _has_active_manifest(paths, model_type, symbol, interval)

    if is_core_model and interval in anchor_tfs and is_manifest_gap and not active_manifest_exists:
        return 0
    if is_core_model and interval in fast_tfs and is_manifest_gap:
        return 1
    if is_core_model and interval in secondary_tfs and is_manifest_gap:
        return 2
    if is_core_model and interval in fast_tfs:
        return 3
    if is_core_model and interval in secondary_tfs:
        return 4
    if is_core_model and interval in anchor_tfs:
        if deprioritize_anchor and active_manifest_exists:
            return 6
        return 5
    if model_type == "dqn":
        return 8
    if model_type == "hmm":
        return 9
    return 10


def _job_priority(paths, *, model_type: str, symbol_idx: int, symbol: str, interval: str, interval_order: list[str], reason: str) -> int:
    tier = _fairness_priority_tier(paths, model_type=model_type, symbol=symbol, interval=interval, reason=reason)
    return tier * 1_000_000 + _model_priority(model_type) * 10_000 + _interval_priority(interval, interval_order) * 100 + symbol_idx


def _failure_cooldown_active(paths, entry, now_ts: float) -> bool:
    if entry.status != "failed":
        return False
    last_failed_ts = float(entry.completed_at or 0.0)
    if last_failed_ts <= 0:
        return False
    cooldown_sec = float(paths.config.scheduling.failure_cooldown_sec or 0.0)
    return cooldown_sec > 0 and (now_ts - last_failed_ts) < cooldown_sec


def _table_name(interval: str) -> str:
    return f"klines_{interval}"


def _load_market_state(paths, symbol: str) -> dict[str, dict[str, int]]:
    db_path = paths.market_db_path(symbol)
    state: dict[str, dict[str, int]] = {
        interval: {"rows": 0, "last_ts": 0}
        for interval in paths.config.frames.fusion_intervals
    }
    if not db_path.exists():
        return state

    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            for interval in paths.config.frames.fusion_intervals:
                table = _table_name(interval)
                try:
                    row = conn.execute(f'SELECT COUNT(1), COALESCE(MAX(timestamp), 0) FROM "{table}"').fetchone()
                except sqlite3.Error:
                    row = (0, 0)
                rows = int(row[0] or 0) if row else 0
                last_ts = int(row[1] or 0) if row else 0
                state[interval] = {"rows": rows, "last_ts": last_ts}
    except sqlite3.Error:
        return state

    return state


def _required_anchor_rows(paths, model_type: str) -> int:
    model_cfg = _model_cfg(paths, model_type)
    return int(model_cfg.min_rows) + 80


def _required_support_rows(interval: str) -> int:
    if interval == "1d":
        return 60
    return 80


def _market_data_ready(
    paths,
    *,
    symbol: str,
    interval: str,
    model_type: str,
    market_state: dict[str, dict[str, int]],
) -> tuple[bool, str, dict[str, Any]]:
    anchor_required = _required_anchor_rows(paths, model_type)
    anchor_state = market_state.get(interval, {"rows": 0, "last_ts": 0})
    if anchor_state["rows"] < anchor_required or anchor_state["last_ts"] <= 0:
        return False, "anchor_rows_missing", {
            "interval": interval,
            "have_rows": int(anchor_state["rows"]),
            "need_rows": anchor_required,
            "last_ts": int(anchor_state["last_ts"]),
        }

    missing_support: list[dict[str, int | str]] = []
    for support_interval in paths.config.frames.fusion_intervals:
        support_state = market_state.get(support_interval, {"rows": 0, "last_ts": 0})
        support_required = _required_support_rows(support_interval)
        if support_state["rows"] < support_required or support_state["last_ts"] <= 0:
            missing_support.append(
                {
                    "interval": support_interval,
                    "have_rows": int(support_state["rows"]),
                    "need_rows": support_required,
                    "last_ts": int(support_state["last_ts"]),
                }
            )

    if missing_support:
        return False, "support_rows_missing", {"missing_support": missing_support}

    return True, DATA_READY_REASON, {
        "anchor_interval": interval,
        "anchor_rows": int(anchor_state["rows"]),
        "anchor_last_ts": int(anchor_state["last_ts"]),
    }


def _manifest_mtime(manifest_path: Path) -> float:
    try:
        return float(manifest_path.stat().st_mtime)
    except Exception:
        return 0.0


def _entry_reference_manifest(paths, entry, model_type: str, symbol: str, interval: str) -> Path | None:
    """Resolve the best available manifest path for scheduling freshness checks."""
    manifest_raw = str(getattr(entry, "manifest_path", "") or "").strip()
    if manifest_raw:
        resolved = resolve_artifact_path(paths, manifest_raw)
        if resolved.exists():
            return resolved

    active_manifest = resolve_active_manifest_path(paths, model_type, symbol, interval)
    if active_manifest is not None and active_manifest.exists():
        return active_manifest

    legacy = _legacy_manifest_path(paths, model_type, symbol, interval)
    if legacy.exists():
        return legacy
    return None



def _parse_insufficient_fold_reason(candidate_only_reasons: list[str]) -> int | None:
    """Extract observed fold count from policy governance reasons."""
    for reason in candidate_only_reasons:
        text = str(reason)
        if text.startswith("insufficient_fold_count:"):
            try:
                return int(text.split(":", 1)[1])
            except Exception:
                return None
    return None


def _configured_retrain_threshold(paths, model_type: str, interval: str) -> int:
    defaults = _DEFAULT_RETRAIN_MIN_NEW_BARS.get(model_type, {})
    default_value = int(defaults.get(interval, _DEFAULT_RETRAIN_CYCLES.get(interval, 1)))

    training_cfg = getattr(getattr(paths, "config", None), "training", None)
    retrain_cycles = getattr(training_cfg, "retrain_cycles", {}) if training_cfg is not None else {}
    if isinstance(retrain_cycles, dict) and retrain_cycles:
        try:
            return max(1, int(retrain_cycles.get(interval, default_value)))
        except Exception:
            return max(1, default_value)

    scheduling_cfg = getattr(getattr(paths, "config", None), "scheduling", None)
    configured = getattr(scheduling_cfg, "retrain_min_new_bars", {}) if scheduling_cfg is not None else {}
    if not isinstance(configured, dict):
        return max(1, default_value)
    model_map = configured.get(model_type, {})
    if not isinstance(model_map, dict):
        return max(1, default_value)
    try:
        return max(1, int(model_map.get(interval, default_value)))
    except Exception:
        return max(1, default_value)


def _effective_retrain_threshold(paths, entry, model_type: str, interval: str) -> int:
    base = _configured_retrain_threshold(paths, model_type, interval)
    metrics = getattr(entry, "metrics", {}) if isinstance(getattr(entry, "metrics", {}), dict) else {}
    hinted = metrics.get("retrain_min_new_bars")
    try:
        hinted_int = int(hinted)
    except Exception:
        hinted_int = 0
    if hinted_int > 0:
        base = max(base, hinted_int)

    candidate_only_reasons = metrics.get("candidate_only_reasons")
    if not isinstance(candidate_only_reasons, list):
        return base
    observed_fold_count = _parse_insufficient_fold_reason(candidate_only_reasons)
    if observed_fold_count is None:
        return base
    if hinted_int > 0:
        return base

    promotion_cfg = getattr(getattr(paths.config, "training", None), "promotion", None)
    min_fold_count = int(getattr(promotion_cfg, "min_fold_count", 1) or 1) if promotion_cfg is not None else 1
    deficit = max(0, min_fold_count - observed_fold_count)
    return max(base, base * (deficit + 1))

def _entry_reference_ts(entry, manifest_path: Path | None) -> float:
    """Return the best available training reference timestamp in seconds."""
    metrics = getattr(entry, "metrics", {}) if isinstance(getattr(entry, "metrics", {}), dict) else {}
    anchor_ts_ms = metrics.get("training_reference_anchor_ts_ms")
    try:
        anchor_sec = float(anchor_ts_ms or 0) / 1000.0
    except Exception:
        anchor_sec = 0.0
    if anchor_sec > 0:
        return anchor_sec

    if manifest_path is not None and manifest_path.exists():
        manifest = load_json(manifest_path) or {}
        training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
        extra_anchor_ts_ms = training.get("anchor_last_ts_ms") or manifest.get("training_reference_anchor_ts_ms")
        try:
            extra_anchor_sec = float(extra_anchor_ts_ms or 0) / 1000.0
        except Exception:
            extra_anchor_sec = 0.0
        if extra_anchor_sec > 0:
            return extra_anchor_sec

        timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
        for key in ("updated_at", "created_at"):
            try:
                value = timestamps.get(key)
                if value:
                    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
            except Exception:
                pass

        mtime = _manifest_mtime(manifest_path)
        if mtime > 0:
            return mtime

    for attr in ("last_success_at", "completed_at", "started_at"):
        value = getattr(entry, attr, None)
        try:
            ts = float(value or 0.0)
        except Exception:
            ts = 0.0
        if ts > 0:
            return ts
    return 0.0


def _statistical_ineligible_due_reason(paths, entry, *, interval: str, market_state: dict[str, dict[str, int]]) -> tuple[bool, str] | None:
    """Return a retry decision for statistically ineligible RF/GBM tuples.

    The tuple remains ineligible until enough new anchor bars have arrived to
    make a materially different temporal validation split possible. Existing
    active incumbents are intentionally ignored for this retry reference; the
    reference is the market timestamp at the failed statistical-support check.
    """
    if str(getattr(entry, "status", "") or "").lower() != "ineligible":
        return None
    try:
        reference_ms = int(getattr(entry, "ineligible_reference_anchor_ts_ms", 0) or 0)
    except Exception:
        reference_ms = 0
    if reference_ms <= 0:
        return False, "statistical_ineligible_wait:no_reference_anchor"
    anchor_last_ts = int((market_state.get(interval) or {}).get("last_ts") or 0)
    if anchor_last_ts <= reference_ms:
        return False, "statistical_ineligible_wait:no_new_bars"
    iv_sec = int(_INTERVAL_SECONDS.get(interval, 60))
    threshold = _configured_retrain_threshold(paths, str(getattr(entry, "model_type", "") or "rf"), interval)
    new_bars = int(max(0, anchor_last_ts - reference_ms) // max(1, iv_sec * 1000))
    if new_bars >= threshold:
        return True, f"statistical_ineligible_new_bars:{new_bars}/{threshold}"
    return False, f"statistical_ineligible_wait:{new_bars}/{threshold}"


def _job_due_reason(
    paths,
    entry,
    max_age_minutes: int,
    now_ts: float,
    *,
    model_type: str,
    symbol: str,
    interval: str,
    market_state: dict[str, dict[str, int]],
) -> tuple[bool, str]:
    ineligible_due = _statistical_ineligible_due_reason(paths, entry, interval=interval, market_state=market_state)
    if ineligible_due is not None:
        return ineligible_due

    manifest_path = _entry_reference_manifest(paths, entry, model_type, symbol, interval)
    if manifest_path is not None:
        valid, reason = validate_artifact_bundle(paths, manifest_path)
        if not valid:
            return True, reason

    reference_ts = _entry_reference_ts(entry, manifest_path)
    if reference_ts <= 0:
        return True, "manifest_missing"

    anchor_last_ts = int((market_state.get(interval) or {}).get("last_ts") or 0)
    anchor_last_sec = anchor_last_ts / 1000.0 if anchor_last_ts > 0 else 0.0

    if anchor_last_sec > 0 and reference_ts > 0:
        iv_sec = int(_INTERVAL_SECONDS.get(interval, 60))
        threshold = _effective_retrain_threshold(paths, entry, model_type, interval)
        new_bars = int(max(0.0, anchor_last_sec - reference_ts) // max(1, iv_sec))
        if new_bars >= threshold:
            return True, f"new_bars:{new_bars}/{threshold}"
        if new_bars > 0:
            return False, f"new_bars_wait:{new_bars}/{threshold}"
        return False, "market_synced"

    age_minutes = (now_ts - reference_ts) / 60.0
    if age_minutes >= max_age_minutes:
        return True, "artifact_stale"

    return False, "fresh"


def build_training_backlog_diagnostics(paths, registry, *, limit: int = 500) -> dict[str, Any]:
    """Build observability-only scheduler/registry backlog diagnostics.

    This function does not queue jobs, mutate registry rows, or change symbol
    lifecycle state. It explains why enabled active symbol/model/interval tuples
    are currently promoted, waiting, blocked by market data, or due.
    """
    interval_order = list(paths.config.scheduling.priority_intervals)
    symbols = _load_symbols(paths)
    now_ts = _now_ts()
    snapshot = registry.snapshot() if hasattr(registry, "snapshot") else {"entries": {}}
    entry_map = snapshot.get("entries", {}) if isinstance(snapshot, dict) else {}
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    market_state_cache: dict[str, dict[str, dict[str, int]]] = {}
    for symbol_idx, symbol in enumerate(symbols):
        market_state = market_state_cache.setdefault(symbol, _load_market_state(paths, symbol))
        for interval in interval_order:
            for model_type in ("rf", "gbm", "dqn", "hmm"):
                if not _is_enabled_for_job(paths, model_type, interval):
                    continue
                key = f"{model_type}:{symbol}:{interval}"
                entry_payload = entry_map.get(key)
                if isinstance(entry_payload, dict):
                    entry = type("EntryView", (), {})()
                    for k, v in entry_payload.items():
                        setattr(entry, k, v)
                    registry_status = str(entry_payload.get("status") or "idle")
                else:
                    entry = None
                    registry_status = "missing"

                ready, readiness_reason, readiness_meta = _market_data_ready(
                    paths,
                    symbol=symbol,
                    interval=interval,
                    model_type=model_type,
                    market_state=market_state,
                )

                due = False
                due_reason = "market_data_not_ready"
                if not ready:
                    category = "market_data_not_ready"
                    detail = readiness_reason
                elif entry is None:
                    category = "registry_missing"
                    detail = "enabled_tuple_missing_registry_entry"
                    due = True
                    due_reason = "manifest_missing"
                elif registry_status in {"queued", "running"}:
                    category = registry_status
                    detail = f"scheduler_status:{registry_status}"
                    due_reason = registry_status
                elif _failure_cooldown_active(paths, entry, now_ts):
                    category = "failure_cooldown"
                    detail = "failed_entry_in_cooldown"
                    due_reason = "failure_cooldown"
                else:
                    max_age_minutes = _max_age_minutes(paths, model_type)
                    due, due_reason = _job_due_reason(
                        paths,
                        entry,
                        max_age_minutes,
                        now_ts,
                        model_type=model_type,
                        symbol=symbol,
                        interval=interval,
                        market_state=market_state,
                    )
                    if registry_status == "ineligible" and not due:
                        category = "statistical_ineligible"
                        detail = due_reason
                    elif due:
                        category = "due"
                        detail = due_reason
                    else:
                        category = "waiting"
                        detail = due_reason

                counts[category] = counts.get(category, 0) + 1
                if len(rows) < max(1, int(limit)):
                    rows.append(
                        {
                            "symbol": symbol,
                            "symbol_rank": symbol_idx,
                            "model_type": model_type,
                            "interval": interval,
                            "registry_status": registry_status,
                            "category": category,
                            "detail": detail,
                            "market_data_ready": bool(ready),
                            "market_readiness_reason": readiness_reason,
                            "market_readiness": readiness_meta,
                            "due": bool(due),
                            "due_reason": due_reason,
                        }
                    )

    return {
        "counts": counts,
        "rows": rows,
        "total_rows": sum(counts.values()),
        "truncated": sum(counts.values()) > len(rows),
    }


def build_due_training_jobs(paths, registry) -> list[TrainingJob]:
    interval_order = list(paths.config.scheduling.priority_intervals)
    symbols = _load_symbols(paths)
    now_ts = _now_ts()
    jobs: list[TrainingJob] = []

    market_state_cache: dict[str, dict[str, dict[str, int]]] = {}
    seen_keys: set[tuple[str, str, str]] = set()

    recovery_jobs = _affected_anchor_recovery_jobs(
        paths,
        registry,
        symbols=symbols,
        market_state_cache=market_state_cache,
        now_ts=now_ts,
        interval_order=interval_order,
    )
    jobs.extend(recovery_jobs)
    seen_keys.update(job.key for job in recovery_jobs)

    for symbol_idx, symbol in enumerate(symbols):
        market_state = market_state_cache.setdefault(symbol, _load_market_state(paths, symbol))

        for interval in interval_order:
            for model_type in ("rf", "gbm", "dqn", "hmm"):
                if not _is_enabled_for_job(paths, model_type, interval):
                    continue

                ready, _, readiness_meta = _market_data_ready(
                    paths,
                    symbol=symbol,
                    interval=interval,
                    model_type=model_type,
                    market_state=market_state,
                )
                if not ready:
                    continue

                entry = registry.get_entry(model_type, symbol, interval)
                if entry.status in {"queued", "running"}:
                    continue
                if _failure_cooldown_active(paths, entry, now_ts):
                    continue

                max_age_minutes = _max_age_minutes(paths, model_type)
                due, reason = _job_due_reason(
                    paths,
                    entry,
                    max_age_minutes,
                    now_ts,
                    model_type=model_type,
                    symbol=symbol,
                    interval=interval,
                    market_state=market_state,
                )
                if not due:
                    continue

                if (model_type, symbol, interval) in seen_keys:
                    continue

                priority = _job_priority(
                    paths,
                    model_type=model_type,
                    symbol_idx=symbol_idx,
                    symbol=symbol,
                    interval=interval,
                    interval_order=interval_order,
                    reason=reason,
                )
                seen_keys.add((model_type, symbol, interval))
                jobs.append(
                    TrainingJob(
                        model_type=model_type,
                        symbol=symbol,
                        interval=interval,
                        priority=priority,
                        requested_at=now_ts,
                        reason=reason,
                        metadata={"market_readiness": readiness_meta},
                    )
                )

    jobs.sort(key=lambda job: (job.priority, job.symbol, job.interval, job.model_type))
    return jobs
