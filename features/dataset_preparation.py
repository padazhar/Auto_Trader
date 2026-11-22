from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Set

from features.feature_store_contract import build_feature_store_metadata, resolve_feature_recipe_version

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TrainValidSplit:
    """Simple time-ordered train/validation split."""

    train: pd.DataFrame
    valid: pd.DataFrame


@dataclass(slots=True)
class WalkForwardFold:
    """Single walk-forward fold with deterministic row bounds."""

    fold_index: int
    train: pd.DataFrame
    valid: pd.DataFrame
    train_start_ts: float | None
    train_end_ts: float | None
    valid_start_ts: float | None
    valid_end_ts: float | None


def coerce_numeric_frame(
    df: pd.DataFrame,
    skip_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert all non-skipped columns to numeric."""
    skip = set(skip_columns or [])
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def select_feature_columns(
    df: pd.DataFrame,
    exclude_columns: Set[str] | None = None,
) -> List[str]:
    """Select numeric feature columns suitable for supervised training."""
    exclude = set(exclude_columns or set())
    cols: List[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        if col == "timestamp":
            continue
        if col.endswith("_timestamp"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)

    return cols


def fill_feature_nans(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Forward/back-fill numeric feature NaNs deterministically."""
    out = df.copy()
    if not feature_columns:
        return out

    out[list(feature_columns)] = out[list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    out[list(feature_columns)] = out[list(feature_columns)].ffill().bfill().fillna(0.0)
    return out


def drop_constant_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, List[str]]:
    """Drop feature columns that are constant across the frame."""
    keep: List[str] = []
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col]
        if series.nunique(dropna=False) > 1:
            keep.append(col)
    return df.copy(), keep


def build_train_valid_split(
    df: pd.DataFrame,
    valid_fraction: float,
    target_column: str,
    timestamp_column: str = "timestamp",
    purge_rows: int = 0,
) -> TrainValidSplit:
    """Build a trailing time-based train/validation split with a purge gap."""
    out = df.copy()
    out = out.sort_values(timestamp_column).reset_index(drop=True)
    out = out.dropna(subset=[target_column]).copy()

    n_total = len(out)
    if n_total <= 1:
        return TrainValidSplit(train=out.copy(), valid=out.copy())

    n_valid = max(1, int(round(n_total * valid_fraction)))
    n_valid = min(n_valid, n_total - 1)
    split_idx = n_total - n_valid
    train_end = max(1, split_idx - int(max(0, purge_rows)))

    train = out.iloc[:train_end].copy()
    valid = out.iloc[split_idx:].copy()

    return TrainValidSplit(train=train, valid=valid)


def build_walk_forward_splits(
    df: pd.DataFrame,
    *,
    valid_fraction: float,
    target_column: str,
    timestamp_column: str = "timestamp",
    fold_count: int = 3,
    min_train_rows: int = 100,
    purge_rows: int = 0,
    embargo_rows: int = 0,
) -> list[WalkForwardFold]:
    """Build deterministic expanding-window walk-forward folds.

    Purge rows are excluded immediately before each validation window. Embargo rows
    are skipped between successive validation windows so adjacent folds do not use
    nearly identical samples.
    """
    out = df.copy()
    out = out.sort_values(timestamp_column).reset_index(drop=True)
    out = out.dropna(subset=[target_column]).copy()
    n_total = len(out)
    if n_total < 2:
        return []

    valid_rows = max(1, int(round(n_total * float(valid_fraction))))
    block_rows = max(1, valid_rows + int(max(0, embargo_rows)))
    max_possible_folds = max(1, (n_total - int(min_train_rows)) // block_rows)
    actual_folds = max(1, min(int(fold_count), max_possible_folds))
    folds: list[WalkForwardFold] = []
    start_valid = n_total - (actual_folds * block_rows)

    for fold_index in range(actual_folds):
        valid_start = start_valid + (fold_index * block_rows)
        valid_end = min(n_total, valid_start + valid_rows)
        train_end = max(0, valid_start - int(max(0, purge_rows)))
        if train_end < int(min_train_rows) or valid_end <= valid_start:
            continue
        train = out.iloc[:train_end].copy()
        valid = out.iloc[valid_start:valid_end].copy()
        if train.empty or valid.empty:
            continue
        folds.append(
            WalkForwardFold(
                fold_index=len(folds),
                train=train,
                valid=valid,
                train_start_ts=_series_ts_or_none(train, timestamp_column, first=True),
                train_end_ts=_series_ts_or_none(train, timestamp_column, first=False),
                valid_start_ts=_series_ts_or_none(valid, timestamp_column, first=True),
                valid_end_ts=_series_ts_or_none(valid, timestamp_column, first=False),
            )
        )
    return folds


def feature_set_signature(feature_columns: Sequence[str], *, recipe_version: str) -> str:
    """Build a stable hash signature for a feature set and its recipe version."""
    metadata = build_feature_store_metadata(feature_columns, recipe_version=recipe_version)
    return str(metadata["signature"])


def build_feature_set_metadata(
    feature_columns: Sequence[str],
    *,
    recipe_version: str,
    anchor_interval: str | None = None,
    optional_context_fields: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build canonical feature-set metadata for manifests and audits."""
    return build_feature_store_metadata(
        feature_columns,
        recipe_version=recipe_version,
        anchor_interval=anchor_interval,
        optional_context_fields=optional_context_fields,
    )


def _anchor_prefix_from_columns(df: pd.DataFrame) -> str | None:
    close_cols = [col[:-6] for col in df.columns if col.endswith("_close")]
    if not close_cols:
        return None
    ranked = sorted(close_cols, key=lambda prefix: (0 if prefix.endswith("15m") else 1, prefix))
    return ranked[0]


def _timestamp_series(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_numeric(df.get("timestamp"), errors="coerce")
    ts = ts.ffill().bfill()
    return ts.astype("float64")


def _series_ts_or_none(df: pd.DataFrame, timestamp_column: str, *, first: bool) -> float | None:
    if timestamp_column not in df.columns or df.empty:
        return None
    series = pd.to_numeric(df[timestamp_column], errors="coerce").dropna()
    if series.empty:
        return None
    value = series.iloc[0] if first else series.iloc[-1]
    return float(value)


def _inject_minimal_fallback_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = _timestamp_series(out)
    if not ts.empty:
        dt = pd.to_datetime(ts, unit="ms", utc=True, errors="coerce")
        index_norm = np.linspace(0.0, 1.0, num=len(out), endpoint=True) if len(out) else np.array([])
        out["__fallback_index_norm"] = index_norm
        out["__fallback_hour_sin"] = np.sin(2.0 * np.pi * (dt.dt.hour.fillna(0) / 24.0))
        out["__fallback_hour_cos"] = np.cos(2.0 * np.pi * (dt.dt.hour.fillna(0) / 24.0))
        out["__fallback_dow_sin"] = np.sin(2.0 * np.pi * (dt.dt.dayofweek.fillna(0) / 7.0))
        out["__fallback_dow_cos"] = np.cos(2.0 * np.pi * (dt.dt.dayofweek.fillna(0) / 7.0))
        out["__fallback_minutes_from_open"] = (dt.dt.hour.fillna(0) * 60.0) + dt.dt.minute.fillna(0)

    anchor_prefix = _anchor_prefix_from_columns(out)
    if not anchor_prefix:
        return out

    close_col = f"{anchor_prefix}_close"
    volume_col = f"{anchor_prefix}_volume"
    high_col = f"{anchor_prefix}_high" if f"{anchor_prefix}_high" in out.columns else None
    low_col = f"{anchor_prefix}_low" if f"{anchor_prefix}_low" in out.columns else None
    open_col = f"{anchor_prefix}_open" if f"{anchor_prefix}_open" in out.columns else None

    close = pd.to_numeric(out.get(close_col), errors="coerce")
    if close is None:
        return out

    ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan)
    out["__fallback_ret1"] = ret1
    out["__fallback_ret3"] = close.pct_change(3).replace([np.inf, -np.inf], np.nan)
    out["__fallback_ret5"] = close.pct_change(5).replace([np.inf, -np.inf], np.nan)
    out["__fallback_ma_gap_5"] = (close - close.rolling(5).mean()) / close.rolling(5).mean().replace(0, np.nan)
    out["__fallback_ma_gap_10"] = (close - close.rolling(10).mean()) / close.rolling(10).mean().replace(0, np.nan)
    out["__fallback_vol_5"] = ret1.rolling(5).std()
    out["__fallback_vol_10"] = ret1.rolling(10).std()
    out["__fallback_abs_ret1"] = ret1.abs()

    if open_col in out.columns:
        open_px = pd.to_numeric(out[open_col], errors="coerce")
        out["__fallback_body_pct"] = (close - open_px) / open_px.replace(0, np.nan)

    if volume_col in out.columns:
        volume = pd.to_numeric(out[volume_col], errors="coerce")
        out["__fallback_volume_ret1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        out["__fallback_volume_ratio_5"] = volume / volume.rolling(5).mean().replace(0, np.nan)
        out["__fallback_volume_z5"] = (volume - volume.rolling(5).mean()) / volume.rolling(5).std().replace(0, np.nan)

    if high_col and low_col:
        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")
        out["__fallback_range_pct"] = (high - low) / close.replace(0, np.nan)

    return out


def prepare_supervised_dataset(
    df: pd.DataFrame,
    *,
    target_column: str,
    exclude_columns: Set[str] | None = None,
    timestamp_column: str = "timestamp",
    valid_fraction: float = 0.2,
    min_rows_required: int = 100,
    passthrough_columns: Sequence[str] | None = None,
    walk_forward_folds: int = 3,
    walk_forward_purge_rows: int = 0,
    walk_forward_embargo_rows: int = 0,
    feature_recipe_version: str | None = None,
    anchor_interval: str | None = None,
    optional_context_fields: Sequence[str] | None = None,
) -> dict:
    """Prepare a deterministic supervised-learning dataset and walk-forward folds."""
    if df.empty:
        raise ValueError("input dataframe is empty")

    recipe_version = resolve_feature_recipe_version(feature_recipe_version)
    out = df.copy()
    skip_numeric = [timestamp_column]
    if passthrough_columns:
        skip_numeric.extend([str(col) for col in passthrough_columns])
    out = coerce_numeric_frame(out, skip_columns=skip_numeric)
    out = out.dropna(subset=[target_column]).copy()

    exclude = set(exclude_columns or set())
    exclude.add(target_column)
    exclude.add(timestamp_column)
    passthrough = [col for col in (passthrough_columns or []) if col in out.columns and col not in {target_column, timestamp_column}]

    def _prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        feature_columns = select_feature_columns(frame, exclude_columns=exclude)
        if not feature_columns:
            raise ValueError("no usable feature columns after filtering")
        frame = fill_feature_nans(frame, feature_columns)
        frame, feature_columns = drop_constant_columns(frame, feature_columns)
        if not feature_columns:
            raise ValueError("all feature columns became constant after filtering")
        return frame, feature_columns

    fallback_variant = "none"
    try:
        out, feature_columns = _prepare_frame(out)
    except ValueError as exc:
        if "constant" not in str(exc).lower() and "no usable feature columns" not in str(exc).lower():
            raise
        out = _inject_minimal_fallback_features(out)
        fallback_variant = "minimal"
        try:
            out, feature_columns = _prepare_frame(out)
        except ValueError:
            ts = _timestamp_series(out)
            out["__fallback_monotonic_index"] = np.arange(len(out), dtype="float64")
            out["__fallback_timestamp_delta"] = ts.diff().fillna(0.0)
            out, feature_columns = _prepare_frame(out)
            fallback_variant = "time_index_only"

    if len(out) < min_rows_required:
        raise ValueError(f"not enough rows after preparation: {len(out)} < {min_rows_required}")

    split_columns = [timestamp_column, target_column] + passthrough + feature_columns
    working = out.loc[:, [col for col in split_columns if col in out.columns]].copy()
    split = build_train_valid_split(
        working,
        valid_fraction=valid_fraction,
        target_column=target_column,
        timestamp_column=timestamp_column,
        purge_rows=walk_forward_purge_rows,
    )
    walk_forward = build_walk_forward_splits(
        working,
        valid_fraction=valid_fraction,
        target_column=target_column,
        timestamp_column=timestamp_column,
        fold_count=walk_forward_folds,
        min_train_rows=min_rows_required,
        purge_rows=walk_forward_purge_rows,
        embargo_rows=walk_forward_embargo_rows,
    )

    return {
        "frame": out,
        "feature_columns": feature_columns,
        "feature_set": build_feature_set_metadata(
            feature_columns,
            recipe_version=recipe_version,
            anchor_interval=anchor_interval,
            optional_context_fields=optional_context_fields,
        ),
        "passthrough_columns": passthrough,
        "split": split,
        "walk_forward_splits": walk_forward,
        "rows_total": int(len(out)),
        "used_fallback_features": fallback_variant != "none",
        "fallback_variant": fallback_variant,
    }


def target_class_summary(series: pd.Series | Sequence[Any]) -> dict[str, Any]:
    """Return deterministic target-label support metadata for governance."""
    clean = pd.to_numeric(pd.Series(series), errors="coerce").dropna().astype(int)
    counts = clean.value_counts().sort_index().to_dict()
    return {
        "rows": int(len(clean)),
        "class_count": int(len(counts)),
        "classes": [int(k) for k in counts.keys()],
        "class_distribution": {str(int(k)): int(v) for k, v in counts.items()},
    }


def validate_target_class_support(
    series: pd.Series | Sequence[Any],
    *,
    min_classes: int,
    required_classes: Sequence[int] | None = None,
    context: str = "target",
) -> dict[str, Any]:
    """Validate label support and return governance-friendly metadata."""
    summary = target_class_summary(series)
    reasons: list[str] = []
    if int(summary["class_count"]) < int(min_classes):
        reasons.append(f"insufficient_class_count:{summary['class_count']}<{int(min_classes)}")
    if required_classes is not None:
        observed = {int(v) for v in summary["classes"]}
        missing = [int(v) for v in required_classes if int(v) not in observed]
        if missing:
            reasons.append("missing_required_classes:" + ",".join(str(v) for v in missing))
    summary["valid"] = not reasons
    summary["reasons"] = reasons
    summary["context"] = str(context)
    if reasons:
        raise ValueError(f"{context}_label_support_invalid:" + ";".join(reasons))
    return summary



def latest_timestamp_ms(df: pd.DataFrame, timestamp_column: str = "timestamp") -> int | None:
    """Return the latest non-null timestamp from a frame."""
    if timestamp_column not in df.columns or df.empty:
        return None
    series = pd.to_numeric(df[timestamp_column], errors="coerce").dropna()
    if series.empty:
        return None
    return int(series.iloc[-1])
