from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.shared_utils import interval_to_ms

FEATURE_STORE_SCHEMA_VERSION = 1
DEFAULT_FEATURE_RECIPE_VERSION = "feature_store_v2"


@dataclass(frozen=True, slots=True)
class FeatureStoreMetadata:
    """Stable feature-store metadata attached to training artifacts.

    The recipe version is intentionally part of the signature so a logic-only
    feature engineering change cannot silently reuse an old model merely because
    the column names stayed the same.
    """

    schema_version: int
    recipe_version: str
    signature: str
    column_count: int
    columns: list[str]
    anchor_interval: str | None = None
    optional_context_fields: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata payload."""
        return {
            "schema_version": int(self.schema_version),
            "version": str(self.recipe_version),
            "recipe_version": str(self.recipe_version),
            "signature": str(self.signature),
            "column_count": int(self.column_count),
            "columns": list(self.columns),
            "anchor_interval": self.anchor_interval,
            "optional_context_fields": list(self.optional_context_fields or []),
        }


@dataclass(frozen=True, slots=True)
class FreshnessCheck:
    ok: bool
    reason: str
    latest_timestamp_ms: int | None
    age_ms: int | None
    max_age_ms: int | None
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "latest_timestamp_ms": self.latest_timestamp_ms,
            "age_ms": self.age_ms,
            "max_age_ms": self.max_age_ms,
            "details": dict(self.details or {}),
        }


def resolve_feature_recipe_version(source: Any | None = None) -> str:
    """Resolve the configured feature recipe version with a safe default."""
    if source is None:
        return DEFAULT_FEATURE_RECIPE_VERSION
    feature_store = getattr(source, "feature_store", None)
    if feature_store is not None:
        value = getattr(feature_store, "recipe_version", "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    value = getattr(source, "recipe_version", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return DEFAULT_FEATURE_RECIPE_VERSION


def feature_store_signature(
    feature_columns: Sequence[str],
    *,
    recipe_version: str,
    anchor_interval: str | None = None,
    optional_context_fields: Sequence[str] | None = None,
) -> str:
    """Build a stable signature for the feature-store contract."""
    payload = {
        "schema_version": FEATURE_STORE_SCHEMA_VERSION,
        "recipe_version": str(recipe_version),
        "anchor_interval": anchor_interval,
        "columns": [str(col) for col in feature_columns],
        "optional_context_fields": [str(col) for col in (optional_context_fields or [])],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_feature_store_metadata(
    feature_columns: Sequence[str],
    *,
    recipe_version: str,
    anchor_interval: str | None = None,
    optional_context_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build canonical feature-store metadata for manifests and audits."""
    ordered = [str(col) for col in feature_columns]
    optional_fields = [str(col) for col in (optional_context_fields or [])]
    signature = feature_store_signature(
        ordered,
        recipe_version=recipe_version,
        anchor_interval=anchor_interval,
        optional_context_fields=optional_fields,
    )
    metadata = FeatureStoreMetadata(
        schema_version=FEATURE_STORE_SCHEMA_VERSION,
        recipe_version=str(recipe_version),
        signature=signature,
        column_count=len(ordered),
        columns=ordered,
        anchor_interval=anchor_interval,
        optional_context_fields=optional_fields,
    )
    return metadata.as_dict()




def validate_feature_store_metadata(
    feature_set: Mapping[str, Any] | None,
    *,
    feature_columns: Sequence[str],
    recipe_version: str,
    anchor_interval: str | None = None,
    optional_context_fields: Sequence[str] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate explicit feature-store metadata against the canonical contract."""
    expected = build_feature_store_metadata(
        feature_columns,
        recipe_version=recipe_version,
        anchor_interval=anchor_interval,
        optional_context_fields=optional_context_fields,
    )
    payload = dict(feature_set or {})
    reasons: list[str] = []
    if not payload:
        return False, ["feature_set_missing"], expected

    columns_raw = payload.get("columns")
    if not isinstance(columns_raw, list) or not columns_raw:
        reasons.append("feature_columns_missing")
        columns = []
    else:
        columns = [str(col) for col in columns_raw]

    declared_schema = payload.get("schema_version")
    if declared_schema is not None:
        try:
            if int(declared_schema) != int(FEATURE_STORE_SCHEMA_VERSION):
                reasons.append(f"feature_schema_version_mismatch:{declared_schema}->{FEATURE_STORE_SCHEMA_VERSION}")
        except Exception:
            reasons.append("feature_schema_version_invalid")

    declared_count = payload.get("column_count")
    if declared_count is not None:
        try:
            if int(declared_count) != len(columns):
                reasons.append("feature_column_count_mismatch")
        except Exception:
            reasons.append("feature_column_count_invalid")

    manifest_recipe_version = str(payload.get("recipe_version") or payload.get("version") or "").strip()
    if not manifest_recipe_version:
        reasons.append("feature_recipe_version_missing")
    elif manifest_recipe_version != str(recipe_version):
        reasons.append(f"feature_recipe_version_mismatch:{manifest_recipe_version}->{recipe_version}")

    manifest_anchor_interval = payload.get("anchor_interval")
    if anchor_interval is not None and str(manifest_anchor_interval or "") != str(anchor_interval):
        reasons.append("feature_anchor_interval_mismatch")

    manifest_optional = [str(col) for col in (payload.get("optional_context_fields") or [])]
    expected_optional = [str(col) for col in (optional_context_fields or [])]
    if manifest_optional != expected_optional:
        reasons.append("feature_optional_context_mismatch")

    if columns and columns != [str(col) for col in feature_columns]:
        reasons.append("feature_columns_mismatch")

    expected_signature = str(expected.get("signature") or "")
    manifest_signature = str(payload.get("signature") or "").strip()
    if not manifest_signature:
        reasons.append("feature_signature_missing")
    elif manifest_signature != expected_signature:
        reasons.append("feature_signature_mismatch")

    return len(reasons) == 0, reasons, expected

def latest_timestamp_ms(df: Any, timestamp_column: str = "timestamp") -> int | None:
    if df is None:
        return None
    series = getattr(df, "get", lambda *_args, **_kwargs: None)(timestamp_column)
    if series is None:
        return None
    try:
        cleaned = series.dropna()
    except Exception:
        return None
    if len(cleaned) <= 0:
        return None
    try:
        return int(cleaned.iloc[-1])
    except Exception:
        return None


def _freshness_cfg(paths_or_cfg: Any) -> Any:
    cfg = getattr(paths_or_cfg, "config", paths_or_cfg)
    return getattr(cfg, "feature_store", None)


def _anchor_max_age_ms(paths_or_cfg: Any, anchor_interval: str) -> int:
    cfg = _freshness_cfg(paths_or_cfg)
    bars = int(getattr(cfg, "max_anchor_staleness_intervals", 3) or 3)
    return max(1, bars) * int(interval_to_ms(anchor_interval))


def _source_max_age_ms(paths_or_cfg: Any, source_interval: str) -> int:
    cfg = _freshness_cfg(paths_or_cfg)
    bars = int(getattr(cfg, "max_source_staleness_intervals", 3) or 3)
    return max(1, bars) * int(interval_to_ms(source_interval))


def _context_max_age_ms(paths_or_cfg: Any, context_name: str, anchor_interval: str) -> int:
    cfg = _freshness_cfg(paths_or_cfg)
    default_map = {
        "funding_rate": 12 * 60 * 60 * 1000,
        "open_interest": 4 * int(interval_to_ms(anchor_interval)),
    }
    attr_name = f"max_{context_name}_staleness_ms"
    value = int(getattr(cfg, attr_name, default_map.get(context_name, 6 * int(interval_to_ms(anchor_interval)))) or default_map.get(context_name, 6 * int(interval_to_ms(anchor_interval))))
    return max(1, value)


def validate_market_data_freshness(
    paths_or_cfg: Any,
    *,
    symbol: str,
    interval: str,
    latest_timestamp_ms: int | None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    if latest_timestamp_ms is None:
        return FreshnessCheck(False, "market_data_unavailable", None, None, _anchor_max_age_ms(paths_or_cfg, interval), {"symbol": str(symbol).upper(), "interval": str(interval)}).as_dict()
    latest = int(latest_timestamp_ms)
    max_age_ms = _anchor_max_age_ms(paths_or_cfg, interval)
    if latest < 100_000_000_000:
        return FreshnessCheck(True, "synthetic_market_timestamp", latest, 0, max_age_ms, {"symbol": str(symbol).upper(), "interval": str(interval)}).as_dict()
    age_ms = max(0, now_ms - latest)
    if age_ms > max_age_ms:
        return FreshnessCheck(False, "stale_market_data", latest, age_ms, max_age_ms, {"symbol": str(symbol).upper(), "interval": str(interval)}).as_dict()
    return FreshnessCheck(True, "market_data_fresh", latest, age_ms, max_age_ms, {"symbol": str(symbol).upper(), "interval": str(interval)}).as_dict()


def validate_fused_frame_freshness(
    paths_or_cfg: Any,
    fused_frame: Any,
    *,
    symbol: str,
    anchor_interval: str,
    now_ms: int | None = None,
) -> dict[str, Any]:
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    if fused_frame is None or getattr(fused_frame, "empty", True):
        return FreshnessCheck(False, "no_features", None, None, _anchor_max_age_ms(paths_or_cfg, anchor_interval), {"symbol": str(symbol).upper(), "anchor_interval": str(anchor_interval)}).as_dict()

    freshness_meta = dict(getattr(fused_frame, "attrs", {}).get("freshness", {}) or {})
    anchor_ts = freshness_meta.get("anchor_timestamp_ms")
    if anchor_ts is None:
        anchor_ts = latest_timestamp_ms(fused_frame, "timestamp")
    if anchor_ts is None:
        return FreshnessCheck(False, "feature_timestamp_missing", None, None, _anchor_max_age_ms(paths_or_cfg, anchor_interval), {"symbol": str(symbol).upper(), "anchor_interval": str(anchor_interval)}).as_dict()

    anchor_ts = int(anchor_ts)
    anchor_max_age_ms = _anchor_max_age_ms(paths_or_cfg, anchor_interval)
    if anchor_ts < 100_000_000_000:
        return FreshnessCheck(True, "synthetic_feature_timestamp", anchor_ts, 0, anchor_max_age_ms, {
            "symbol": str(symbol).upper(),
            "anchor_interval": str(anchor_interval),
            "sources": {},
            "contexts": {},
        }).as_dict()
    anchor_age_ms = max(0, now_ms - anchor_ts)
    source_timestamps = dict(freshness_meta.get("source_timestamps_ms", {}) or {})
    if anchor_interval not in source_timestamps:
        source_timestamps[str(anchor_interval)] = anchor_ts
    context_timestamps = dict(freshness_meta.get("context_timestamps_ms", {}) or {})

    source_details: dict[str, Any] = {}
    for source_interval, source_ts in source_timestamps.items():
        if source_ts is None:
            source_details[str(source_interval)] = {"ok": False, "reason": "missing_source_timestamp"}
            return FreshnessCheck(False, "stale_source_features", anchor_ts, anchor_age_ms, anchor_max_age_ms, {
                "symbol": str(symbol).upper(),
                "anchor_interval": str(anchor_interval),
                "sources": source_details,
                "contexts": context_timestamps,
            }).as_dict()
        latest = int(source_ts)
        age_ms = max(0, now_ms - latest)
        max_age_ms = _source_max_age_ms(paths_or_cfg, str(source_interval))
        ok = age_ms <= max_age_ms
        source_details[str(source_interval)] = {
            "ok": ok,
            "latest_timestamp_ms": latest,
            "age_ms": age_ms,
            "max_age_ms": max_age_ms,
        }
        if not ok:
            return FreshnessCheck(False, "stale_source_features", anchor_ts, anchor_age_ms, anchor_max_age_ms, {
                "symbol": str(symbol).upper(),
                "anchor_interval": str(anchor_interval),
                "sources": source_details,
                "contexts": context_timestamps,
            }).as_dict()

    context_details: dict[str, Any] = {}
    for context_name, context_ts in context_timestamps.items():
        if context_ts is None:
            continue
        latest = int(context_ts)
        age_ms = max(0, now_ms - latest)
        max_age_ms = _context_max_age_ms(paths_or_cfg, str(context_name), anchor_interval)
        ok = age_ms <= max_age_ms
        context_details[str(context_name)] = {
            "ok": ok,
            "latest_timestamp_ms": latest,
            "age_ms": age_ms,
            "max_age_ms": max_age_ms,
        }
        if not ok:
            return FreshnessCheck(False, "stale_context_features", anchor_ts, anchor_age_ms, anchor_max_age_ms, {
                "symbol": str(symbol).upper(),
                "anchor_interval": str(anchor_interval),
                "sources": source_details,
                "contexts": context_details,
            }).as_dict()

    if anchor_age_ms > anchor_max_age_ms:
        return FreshnessCheck(False, "stale_anchor_features", anchor_ts, anchor_age_ms, anchor_max_age_ms, {
            "symbol": str(symbol).upper(),
            "anchor_interval": str(anchor_interval),
            "sources": source_details,
            "contexts": context_details,
        }).as_dict()

    return FreshnessCheck(True, "features_fresh", anchor_ts, anchor_age_ms, anchor_max_age_ms, {
        "symbol": str(symbol).upper(),
        "anchor_interval": str(anchor_interval),
        "sources": source_details,
        "contexts": context_details,
    }).as_dict()
