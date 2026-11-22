from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from core.path_manager import PathManager
from core.shared_utils import interval_to_ms, round_timestamp_to_interval

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
FUNDING_RATE_COLUMNS = ["timestamp", "funding_rate"]
OPEN_INTEREST_COLUMNS = ["timestamp", "open_interest"]
FUNDING_RATE_TABLE = "context_funding_rate"
OPEN_INTEREST_TABLE_PREFIX = "context_open_interest_"
SUPPORTED_OPEN_INTEREST_PERIODS = ("5m", "15m", "30m", "1h", "4h", "1d")
_OPEN_INTEREST_MS = {
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


@dataclass(frozen=True)
class MarketFrame:
    symbol: str
    interval: str
    frame: pd.DataFrame


def _table_name(interval: str) -> str:
    return f"klines_{interval}"


def funding_rate_table_name() -> str:
    """Return the canonical funding-rate table name."""
    return FUNDING_RATE_TABLE


def open_interest_table_name(period: str) -> str:
    """Return the canonical open-interest table name for a period."""
    return f"{OPEN_INTEREST_TABLE_PREFIX}{str(period)}"


def resolve_open_interest_period(interval: str) -> str:
    """Map an arbitrary fusion interval to the nearest supported open-interest period."""
    normalized = str(interval or "").strip()
    if normalized in SUPPORTED_OPEN_INTEREST_PERIODS:
        return normalized
    try:
        target_ms = interval_to_ms(normalized)
    except Exception:
        return "15m"
    for period in SUPPORTED_OPEN_INTEREST_PERIODS:
        if _OPEN_INTEREST_MS[period] >= target_ms:
            return period
    return SUPPORTED_OPEN_INTEREST_PERIODS[-1]


def _request_with_retry(url: str, *, params: dict[str, Any], timeout: float, max_retries: int, backoff_base_sec: float) -> Any:
    delay = float(backoff_base_sec)
    last_error: Exception | None = None
    for attempt in range(int(max_retries)):
        try:
            response = requests.get(url, params=params, timeout=float(timeout), headers={"User-Agent": "AutoTrader/MarketLoader"})
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - network failure exercised by monkeypatching requests in tests
            last_error = exc
            if attempt >= int(max_retries) - 1:
                break
            time.sleep(delay)
            delay *= 1.75
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable retry loop")


def ensure_symbol_database(paths: PathManager, symbol: str) -> None:
    """Ensure canonical market and futures-context tables exist for a symbol."""
    db_path = paths.market_db_path(symbol)
    fusion_intervals = paths.config.frames.fusion_intervals
    with sqlite3.connect(db_path) as conn:
        for interval in fusion_intervals:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_table_name(interval)} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                )
                """
            )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {funding_rate_table_name()} (
                timestamp INTEGER PRIMARY KEY,
                funding_rate REAL NOT NULL
            )
            """
        )
        for period in SUPPORTED_OPEN_INTEREST_PERIODS:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {open_interest_table_name(period)} (
                    timestamp INTEGER PRIMARY KEY,
                    open_interest REAL NOT NULL
                )
                """
            )
        conn.commit()


def _save_context_frame(paths: PathManager, symbol: str, table_name: str, frame: pd.DataFrame, columns: list[str]) -> None:
    ensure_symbol_database(paths, symbol)
    if frame.empty:
        return
    frame = frame.copy()[columns].drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    rows = [tuple(row[col] for col in columns) for row in frame.to_dict(orient="records")]
    placeholders = ", ".join(["?"] * len(columns))
    quoted_columns = ", ".join(f'"{col}"' for col in columns)
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        conn.executemany(
            f'INSERT OR REPLACE INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
            rows,
        )
        conn.commit()


def save_market_frame(paths: PathManager, symbol: str, interval: str, frame: pd.DataFrame) -> None:
    ensure_symbol_database(paths, symbol)
    frame = frame.copy()[OHLCV_COLUMNS].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        frame.to_sql(_table_name(interval), conn, if_exists="append", index=False)
        conn.execute(
            f'DELETE FROM "{_table_name(interval)}" WHERE rowid NOT IN (SELECT MIN(rowid) FROM "{_table_name(interval)}" GROUP BY timestamp)'
        )
        conn.commit()


def save_funding_rate_frame(paths: PathManager, symbol: str, frame: pd.DataFrame) -> None:
    """Persist canonical funding-rate history for a symbol."""
    _save_context_frame(paths, symbol, funding_rate_table_name(), frame, FUNDING_RATE_COLUMNS)


def save_open_interest_frame(paths: PathManager, symbol: str, period: str, frame: pd.DataFrame) -> None:
    """Persist canonical open-interest history for a symbol and period."""
    _save_context_frame(paths, symbol, open_interest_table_name(resolve_open_interest_period(period)), frame, OPEN_INTEREST_COLUMNS)


def _load_context_frame(paths: PathManager, symbol: str, table_name: str, columns: list[str], limit: int | None = None) -> pd.DataFrame:
    ensure_symbol_database(paths, symbol)
    sql = f'SELECT {", ".join(columns)} FROM "{table_name}" ORDER BY timestamp DESC'
    if limit:
        sql += f" LIMIT {int(limit)}"
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        frame = pd.read_sql_query(sql, conn)
    if frame.empty:
        return frame
    for col in columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna().sort_values("timestamp").reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].astype("int64")
    return frame


def load_market_frame(paths: PathManager, symbol: str, interval: str, limit: int | None = None) -> pd.DataFrame:
    ensure_symbol_database(paths, symbol)
    sql = f'SELECT timestamp, open, high, low, close, volume FROM "{_table_name(interval)}" ORDER BY timestamp DESC'
    if limit:
        sql += f" LIMIT {int(limit)}"
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        frame = pd.read_sql_query(sql, conn)
    if frame.empty:
        return frame
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_funding_rate_frame(paths: PathManager, symbol: str, limit: int | None = None) -> pd.DataFrame:
    """Load canonical funding-rate history for a symbol."""
    return _load_context_frame(paths, symbol, funding_rate_table_name(), FUNDING_RATE_COLUMNS, limit=limit)


def load_open_interest_frame(paths: PathManager, symbol: str, period: str, limit: int | None = None) -> pd.DataFrame:
    """Load canonical open-interest history for a symbol and period."""
    return _load_context_frame(paths, symbol, open_interest_table_name(resolve_open_interest_period(period)), OPEN_INTEREST_COLUMNS, limit=limit)




def load_market_window(paths: PathManager, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Load OHLCV candles for a bounded trade-analysis window.

    This is read-only reporting support used by post-trade MAE/MFE analysis.
    It does not fetch missing data, mutate the database, or affect execution.
    """
    ensure_symbol_database(paths, symbol)
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        frame = pd.read_sql_query(
            f'SELECT timestamp, open, high, low, close, volume FROM "{_table_name(interval)}" '
            'WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC',
            conn,
            params=(start_ms, end_ms),
        )
    if frame.empty:
        return frame
    for col in OHLCV_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna().sort_values("timestamp").reset_index(drop=True)


def latest_market_timestamp(paths: PathManager, symbol: str, interval: str) -> int | None:
    ensure_symbol_database(paths, symbol)
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        row = conn.execute(f'SELECT MAX(timestamp) FROM "{_table_name(interval)}"').fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else None


def _latest_context_timestamp(paths: PathManager, symbol: str, table_name: str) -> int | None:
    ensure_symbol_database(paths, symbol)
    with sqlite3.connect(paths.market_db_path(symbol)) as conn:
        row = conn.execute(f'SELECT MAX(timestamp) FROM "{table_name}"').fetchone()
    value = row[0] if row else None
    return int(value) if value is not None else None


def fetch_binance_klines(paths: PathManager, symbol: str, interval: str, start_ms: int, end_ms: int | None = None, limit: int | None = None) -> pd.DataFrame:
    settings = paths.config.market
    url = settings.futures_base_url.rstrip("/") + "/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_ms),
        "limit": int(limit or settings.klines_limit),
    }
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    payload = _request_with_retry(
        url,
        params=params,
        timeout=float(settings.fetch_timeout_sec),
        max_retries=int(settings.max_retries),
        backoff_base_sec=float(settings.backoff_base_sec),
    )
    rows = [
        {
            "timestamp": int(item[0]),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5]),
        }
        for item in payload
    ]
    return pd.DataFrame(rows, columns=OHLCV_COLUMNS)


def fetch_binance_funding_rates(paths: PathManager, symbol: str, start_ms: int, end_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    """Fetch Binance funding-rate history for a symbol."""
    settings = paths.config.market
    url = settings.futures_base_url.rstrip("/") + "/fapi/v1/fundingRate"
    params: dict[str, Any] = {
        "symbol": str(symbol).upper(),
        "startTime": int(start_ms),
        "limit": int(limit),
    }
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    payload = _request_with_retry(
        url,
        params=params,
        timeout=float(settings.fetch_timeout_sec),
        max_retries=int(settings.max_retries),
        backoff_base_sec=float(settings.backoff_base_sec),
    )
    rows = [
        {
            "timestamp": int(item.get("fundingTime") or item.get("time") or 0),
            "funding_rate": float(item.get("fundingRate") or item.get("funding_rate") or 0.0),
        }
        for item in (payload or [])
        if item is not None
    ]
    frame = pd.DataFrame(rows, columns=FUNDING_RATE_COLUMNS)
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def fetch_binance_open_interest(paths: PathManager, symbol: str, period: str, start_ms: int, end_ms: int | None = None, limit: int = 500) -> pd.DataFrame:
    """Fetch Binance historical open interest for a symbol and period."""
    settings = paths.config.market
    resolved_period = resolve_open_interest_period(period)
    url = settings.futures_base_url.rstrip("/") + "/futures/data/openInterestHist"
    params: dict[str, Any] = {
        "symbol": str(symbol).upper(),
        "period": resolved_period,
        "startTime": int(start_ms),
        "limit": int(limit),
    }
    if end_ms is not None:
        params["endTime"] = int(end_ms)
    payload = _request_with_retry(
        url,
        params=params,
        timeout=float(settings.fetch_timeout_sec),
        max_retries=int(settings.max_retries),
        backoff_base_sec=float(settings.backoff_base_sec),
    )
    rows = [
        {
            "timestamp": int(item.get("timestamp") or item.get("time") or 0),
            "open_interest": float(item.get("sumOpenInterest") or item.get("openInterest") or item.get("open_interest") or 0.0),
        }
        for item in (payload or [])
        if item is not None
    ]
    frame = pd.DataFrame(rows, columns=OPEN_INTEREST_COLUMNS)
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)


def refresh_market_data(paths: PathManager, symbol: str, interval: str) -> int:
    latest = latest_market_timestamp(paths, symbol, interval)
    step = interval_to_ms(interval)
    if latest is None:
        lookback_rows = int(paths.config.frames.lookback_rows[interval])
        start_ms = round_timestamp_to_interval(int(time.time() * 1000) - lookback_rows * step, interval)
    else:
        start_ms = latest + step
    frame = fetch_binance_klines(paths, symbol, interval, start_ms=start_ms)
    if frame.empty:
        return 0
    save_market_frame(paths, symbol, interval, frame)
    return len(frame)


def refresh_funding_rate_data(paths: PathManager, symbol: str, *, lookback_ms: int | None = None, end_ms: int | None = None) -> int:
    """Refresh canonical funding-rate history for a symbol."""
    latest = _latest_context_timestamp(paths, symbol, funding_rate_table_name())
    start_ms = int(latest + 1) if latest is not None else int(time.time() * 1000) - int(lookback_ms or 30 * 24 * 60 * 60 * 1000)
    frame = fetch_binance_funding_rates(paths, symbol, start_ms=start_ms, end_ms=end_ms)
    if frame.empty:
        return 0
    save_funding_rate_frame(paths, symbol, frame)
    return int(len(frame))


def refresh_open_interest_data(paths: PathManager, symbol: str, period: str, *, lookback_ms: int | None = None, end_ms: int | None = None) -> int:
    """Refresh canonical open-interest history for a symbol and period."""
    resolved_period = resolve_open_interest_period(period)
    latest = _latest_context_timestamp(paths, symbol, open_interest_table_name(resolved_period))
    period_ms = _OPEN_INTEREST_MS[resolved_period]
    default_lookback_ms = max(period_ms * 500, 24 * 60 * 60 * 1000)
    start_ms = int(latest + 1) if latest is not None else int(time.time() * 1000) - int(lookback_ms or default_lookback_ms)
    frame = fetch_binance_open_interest(paths, symbol, resolved_period, start_ms=start_ms, end_ms=end_ms)
    if frame.empty:
        return 0
    save_open_interest_frame(paths, symbol, resolved_period, frame)
    return int(len(frame))
