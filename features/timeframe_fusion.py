from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from core.config_contract import FramesConfig
from features.market_loader import load_funding_rate_frame, load_open_interest_frame, resolve_open_interest_period


OPTIONAL_CONTEXT_SPECS = {
    "funding_rate": {
        "tables": ("context_funding_rate", "funding_rates", "funding_rate", "symbol_funding_rates"),
        "timestamp": ("timestamp", "funding_time", "time"),
        "value": ("funding_rate", "rate", "funding"),
    },
    "open_interest": {
        "tables": ("open_interest", "openInterest", "symbol_open_interest"),
        "timestamp": ("timestamp", "open_time", "time"),
        "value": ("open_interest", "sum_open_interest", "openInterest", "value"),
    },
}


def _db_path(paths, symbol: str) -> Path:
    return paths.market_db_path(symbol)


def _table_name(interval: str) -> str:
    return f"klines_{interval}"


def _safe_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_raw_timeframe(paths, symbol: str, interval: str, tail_rows: int) -> pd.DataFrame:
    db_path = _db_path(paths, symbol)
    if not db_path.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    query = f'''
        SELECT timestamp, open, high, low, close, volume
        FROM "{_table_name(interval)}"
        ORDER BY timestamp DESC
        LIMIT {int(tail_rows)}
    '''

    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = df.iloc[::-1].reset_index(drop=True)
    df = _safe_numeric(df, ["timestamp", "open", "high", "low", "close", "volume"])
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    return df


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).clip(0.0, 100.0)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    numerator = (price * volume).rolling(window).sum()
    denominator = volume.rolling(window).sum().replace(0.0, np.nan)
    return numerator / denominator


def _bollinger(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + (2.0 * std)
    lower = mid - (2.0 * std)
    width = (upper - lower) / mid.replace(0.0, np.nan)
    percent_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return mid, upper, lower, width, percent_b


def _read_sql_with_columns(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(str(db_path)) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table}" ORDER BY 1 DESC LIMIT 5000', conn)


def _load_legacy_optional_context_frame(paths, symbol: str, *, tail_rows: int = 5000) -> pd.DataFrame:
    db_candidates = [paths.market_db_path(symbol), paths.market_storage_dir / "market.db"]
    db_path = next((candidate for candidate in db_candidates if candidate.exists()), None)
    if db_path is None:
        return pd.DataFrame(columns=["timestamp"])

    out = pd.DataFrame(columns=["timestamp"])
    for feature_name, spec in OPTIONAL_CONTEXT_SPECS.items():
        raw = None
        for table_name in spec["tables"]:
            try:
                raw = _read_sql_with_columns(db_path, table_name)
            except Exception:
                raw = None
            if raw is not None and not raw.empty:
                break
        if raw is None or raw.empty:
            continue
        raw.columns = [str(col) for col in raw.columns]
        timestamp_col = next((col for col in spec["timestamp"] if col in raw.columns), None)
        value_col = next((col for col in spec["value"] if col in raw.columns), None)
        if timestamp_col is None or value_col is None:
            continue
        frame = raw[[timestamp_col, value_col]].copy()
        frame.columns = ["timestamp", feature_name]
        frame = _safe_numeric(frame, ["timestamp", feature_name]).dropna()
        if frame.empty:
            continue
        frame = frame.iloc[::-1].drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").tail(int(tail_rows)).reset_index(drop=True)
        if out.empty:
            out = frame
        else:
            out = pd.merge_asof(out.sort_values("timestamp"), frame.sort_values("timestamp"), on="timestamp", direction="backward", allow_exact_matches=True)
    return out if not out.empty else pd.DataFrame(columns=["timestamp"])


def _load_optional_context_frame(paths, symbol: str, *, tail_rows: int = 5000, anchor_interval: str | None = None) -> pd.DataFrame:
    cfg = getattr(paths.config, "feature_store", None)
    if cfg is not None and not bool(getattr(cfg, "include_optional_futures_context", True)):
        return pd.DataFrame(columns=["timestamp"])

    interval = str(anchor_interval or getattr(paths.config.frames, "anchor_interval", "15m"))
    oi_period = resolve_open_interest_period(interval)
    funding = load_funding_rate_frame(paths, symbol, limit=tail_rows)
    open_interest = load_open_interest_frame(paths, symbol, oi_period, limit=tail_rows)

    out = pd.DataFrame(columns=["timestamp"])
    context_timestamps: dict[str, int] = {}
    if not funding.empty:
        funding = funding[["timestamp", "funding_rate"]].copy()
        out = funding
        context_timestamps["funding_rate"] = int(funding["timestamp"].iloc[-1])
    if not open_interest.empty:
        open_interest = open_interest[["timestamp", "open_interest"]].copy()
        if out.empty:
            out = open_interest
        else:
            out = pd.merge_asof(
                out.sort_values("timestamp"),
                open_interest.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                allow_exact_matches=True,
            )
        context_timestamps["open_interest"] = int(open_interest["timestamp"].iloc[-1])

    if out.empty:
        legacy = _load_legacy_optional_context_frame(paths, symbol, tail_rows=tail_rows)
        legacy.attrs["context_timestamps_ms"] = {}
        return legacy

    out = out.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").tail(int(tail_rows)).reset_index(drop=True)
    out.attrs["context_timestamps_ms"] = context_timestamps
    return out


def _engineer_optional_context(context_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if context_df.empty:
        return pd.DataFrame(columns=["timestamp"])
    out = pd.DataFrame({"timestamp": context_df["timestamp"].astype("int64")})
    prefix = f"{interval}_"
    if "funding_rate" in context_df.columns:
        funding = pd.to_numeric(context_df["funding_rate"], errors="coerce")
        out[f"{prefix}funding_rate"] = funding
        out[f"{prefix}funding_rate_ema_10"] = funding.ewm(span=10, adjust=False).mean()
        out[f"{prefix}funding_rate_abs_z"] = (funding.abs() - funding.abs().rolling(50).mean()) / funding.abs().rolling(50).std().replace(0.0, np.nan)
    if "open_interest" in context_df.columns:
        oi = pd.to_numeric(context_df["open_interest"], errors="coerce")
        out[f"{prefix}open_interest"] = oi
        out[f"{prefix}open_interest_ret_1"] = oi.pct_change()
        out[f"{prefix}open_interest_abs_z"] = (oi - oi.rolling(50).mean()).abs() / oi.rolling(50).std().replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def engineer_interval_features(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["timestamp"])

    df = raw.copy()
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]
    typical_price = (high + low + close) / 3.0

    out = pd.DataFrame({"timestamp": df["timestamp"].astype("int64")})
    prefix = f"{interval}_"

    out[f"{prefix}open"] = open_
    out[f"{prefix}high"] = high
    out[f"{prefix}low"] = low
    out[f"{prefix}close"] = close
    out[f"{prefix}volume"] = volume
    out[f"{prefix}ret_1"] = close.pct_change(1)
    out[f"{prefix}ret_3"] = close.pct_change(3)
    out[f"{prefix}ret_5"] = close.pct_change(5)
    out[f"{prefix}ret_10"] = close.pct_change(10)
    out[f"{prefix}log_ret_1"] = np.log(close / close.shift(1))
    out[f"{prefix}range_pct"] = (high - low) / close.replace(0, np.nan)
    out[f"{prefix}body_pct"] = (close - open_) / open_.replace(0, np.nan)
    out[f"{prefix}upper_wick_pct"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    out[f"{prefix}lower_wick_pct"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)

    for win in (5, 10, 20, 50):
        ma = close.rolling(win).mean()
        std = close.rolling(win).std()
        vol_ma = volume.rolling(win).mean()
        out[f"{prefix}sma_{win}"] = ma
        out[f"{prefix}dist_sma_{win}"] = (close - ma) / ma.replace(0, np.nan)
        out[f"{prefix}std_{win}"] = std
        out[f"{prefix}vol_ma_{win}"] = vol_ma
        out[f"{prefix}vol_ratio_{win}"] = volume / vol_ma.replace(0, np.nan)

    out[f"{prefix}atr_14"] = _atr(df, 14)
    out[f"{prefix}atr_pct_14"] = out[f"{prefix}atr_14"] / close.replace(0, np.nan)
    out[f"{prefix}rsi_14"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out[f"{prefix}ema_12"] = ema12
    out[f"{prefix}ema_26"] = ema26
    out[f"{prefix}macd"] = macd
    out[f"{prefix}macd_signal"] = macd_signal
    out[f"{prefix}macd_hist"] = macd - macd_signal

    bb_mid, bb_upper, bb_lower, bb_width, bb_percent_b = _bollinger(close, 20)
    out[f"{prefix}bb_mid_20"] = bb_mid
    out[f"{prefix}bb_upper_20"] = bb_upper
    out[f"{prefix}bb_lower_20"] = bb_lower
    out[f"{prefix}bb_width_20"] = bb_width
    out[f"{prefix}bb_percent_b_20"] = bb_percent_b
    out[f"{prefix}bb_width_z_20"] = (bb_width - bb_width.rolling(50).mean()) / bb_width.rolling(50).std().replace(0.0, np.nan)

    out[f"{prefix}realized_vol_20"] = out[f"{prefix}log_ret_1"].rolling(20).std() * np.sqrt(20)
    out[f"{prefix}realized_vol_z_20"] = (
        out[f"{prefix}realized_vol_20"] - out[f"{prefix}realized_vol_20"].rolling(50).mean()
    ) / out[f"{prefix}realized_vol_20"].rolling(50).std().replace(0.0, np.nan)

    vwap_20 = _rolling_vwap(typical_price, volume, 20)
    out[f"{prefix}vwap_20"] = vwap_20
    out[f"{prefix}vwap_dev_20"] = (close - vwap_20) / vwap_20.replace(0.0, np.nan)
    out[f"{prefix}trend_strength_20"] = (close - close.shift(20)).abs() / out[f"{prefix}atr_14"].replace(0.0, np.nan)
    out[f"{prefix}volume_z_20"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std().replace(0.0, np.nan)

    return out.replace([np.inf, -np.inf], np.nan)


def _prepare_merge_source(features: pd.DataFrame, source_interval: str, anchor_interval: str) -> pd.DataFrame:
    source = features.copy()
    right_key = f"{source_interval}_timestamp"
    source = source.rename(columns={"timestamp": right_key})
    source[f"{source_interval}_age_bars_vs_{anchor_interval}"] = 0
    return source


def build_fused_frame(paths, symbol: str, anchor_interval: str, tail_rows: int = 2500) -> pd.DataFrame:
    frames_cfg: FramesConfig = paths.config.frames
    if anchor_interval not in frames_cfg.fusion_intervals:
        raise ValueError(f"anchor interval '{anchor_interval}' is not in the configured fusion contract")

    anchor_raw = _load_raw_timeframe(paths, symbol, anchor_interval, tail_rows=tail_rows)
    if anchor_raw.empty:
        return pd.DataFrame()

    anchor_features = engineer_interval_features(anchor_raw, anchor_interval)
    base = anchor_features.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)
    source_timestamps_ms: dict[str, int] = {str(anchor_interval): int(anchor_raw["timestamp"].iloc[-1])}
    context_timestamps_ms: dict[str, int] = {}

    optional_context = _load_optional_context_frame(paths, symbol, tail_rows=max(int(tail_rows), 500), anchor_interval=anchor_interval)
    if not optional_context.empty:
        context_timestamps_ms.update(dict(optional_context.attrs.get("context_timestamps_ms", {}) or {}))
        optional_features = _engineer_optional_context(optional_context, anchor_interval)
        if not optional_features.empty:
            base = pd.merge_asof(
                base.sort_values("timestamp"),
                optional_features.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                allow_exact_matches=True,
            )

    for interval in frames_cfg.fusion_intervals:
        if interval == anchor_interval:
            continue
        raw = _load_raw_timeframe(paths, symbol, interval, tail_rows=tail_rows)
        if raw.empty:
            continue
        source_timestamps_ms[str(interval)] = int(raw["timestamp"].iloc[-1])
        feat = engineer_interval_features(raw, interval)
        if feat.empty:
            continue

        src = _prepare_merge_source(feat, interval, anchor_interval)
        right_key = f"{interval}_timestamp"
        src = src.drop_duplicates(subset=[right_key], keep="last").sort_values(right_key).reset_index(drop=True)

        base = pd.merge_asof(
            base.sort_values("timestamp"),
            src.sort_values(right_key),
            left_on="timestamp",
            right_on=right_key,
            direction="backward",
            allow_exact_matches=True,
        )

    base = base.loc[:, ~base.columns.duplicated()]
    base = base.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    base = base.replace([np.inf, -np.inf], np.nan)
    base.attrs["freshness"] = {
        "anchor_interval": str(anchor_interval),
        "anchor_timestamp_ms": int(anchor_raw["timestamp"].iloc[-1]),
        "source_timestamps_ms": source_timestamps_ms,
        "context_timestamps_ms": context_timestamps_ms,
    }
    return base
