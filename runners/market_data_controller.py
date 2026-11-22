from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml

from core.log_manager import get_logger
from monitoring.publisher import MonitorPublisher
from core.symbol_management import (
    is_valid_usdt_symbol,
    load_symbol_state,
    normalize_symbol,
    required_market_history_rows,
    save_symbol_state,
)
from features.market_loader import refresh_funding_rate_data, refresh_open_interest_data, resolve_open_interest_period
from core.market_io import load_active_universe_symbols, load_open_position_symbols, load_runtime_price_symbols, load_market_latest_snapshot, write_market_latest_snapshot

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUTURES_TIME_URL = "https://fapi.binance.com/fapi/v1/time"
BINANCE_FUTURES_24H_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"
BINANCE_FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FUTURES_PRICE_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FUTURES_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_FUTURES_MARK_PRICE_WS_URL = "wss://fstream.binance.com/market/ws/!markPrice@arr@1s"

TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


@dataclass(frozen=True)
class UpdateResult:
    symbol: str
    timeframe: str
    rows_added: int
    last_ts: int


class MarketDataController:
    def __init__(self, paths, training_tick_callback: Optional[Callable[[str, str, int], None]] = None) -> None:
        self.paths = paths
        self.logger = get_logger("market_data_controller", paths)
        self.training_tick_callback = training_tick_callback
        self.monitor = MonitorPublisher(paths)
        self.settings = self._load_yaml(paths.config_dir / "system_settings.yaml")
        self.symbols_config = self._load_yaml(paths.config_dir / "tradable_symbols.yaml")
        cfg = paths.config.market

        self.enabled = cfg.use_binance_http_updater
        self.refresh_universe_enabled = cfg.refresh_universe_enabled
        self.refresh_universe_sec = cfg.refresh_universe_sec
        self.top_n = cfg.top_n
        self.target_active_symbols = int(paths.config.symbols.target_active_symbols)
        self.ranked_universe_size = int(getattr(paths.config.symbols, "ranked_universe_size", 20) or 20)
        self.fast_frames = list(cfg.fast_frames)
        self.slow_frames = list(cfg.slow_frames)
        self.sem_fast = cfg.sem_fast
        self.sem_slow = cfg.sem_slow
        self.request_timeout_sec = cfg.request_timeout_sec
        self.request_retries = cfg.request_retries
        self._runtime_stop_event: threading.Event | None = None
        self.bar_close_delay_ms = cfg.bar_close_delay_ms
        self.server_time_cache_ms = cfg.server_time_cache_ms
        self.lookback_days = dict(cfg.lookback_days)
        self.include_optional_futures_context = bool(getattr(paths.config.feature_store, "include_optional_futures_context", True))
        self.anchor_interval = str(paths.config.frames.anchor_interval)
        self.fusion_intervals = [
            str(interval)
            for interval in list(getattr(paths.config.frames, "fusion_intervals", [self.anchor_interval]) or [self.anchor_interval])
        ]
        training_cfg = getattr(paths.config, "training", None)
        rf_cfg = getattr(training_cfg, "rf", None)
        self.rf_min_rows = int(getattr(rf_cfg, "min_rows", 300) or 300)
        self.history_row_requirements = required_market_history_rows(
            fusion_intervals=self.fusion_intervals,
            anchor_interval=self.anchor_interval,
            rf_min_rows=self.rf_min_rows,
        )
        self.minimum_history_age_ms = self._minimum_history_age_ms(self.history_row_requirements)
        self._exchange_onboard_date_ms: dict[str, int] = {}
        self._history_bootstrap_complete: set[str] = set()
        self._history_bootstrap_last_attempt_ms: dict[str, int] = {}
        self._history_bootstrap_retry_ms = 60_000

        self.symbol_file_path = paths.config_dir / "tradable_symbols.yaml"
        self.market_storage_dir = paths.market_storage_dir
        self.market_storage_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = getattr(paths, "market_runtime_dir", None)
        if runtime_dir is None:
            base_runtime_dir = getattr(paths, "runtime_dir", paths.project_root / "data" / "runtime")
            runtime_dir = base_runtime_dir / "market"
        self.market_runtime_dir = runtime_dir
        self.market_runtime_dir.mkdir(parents=True, exist_ok=True)

        self.pinned_symbols: List[str] = [str(s).upper() for s in self.symbols_config.get("pinned_symbols", [])]
        self.skip_symbols = {str(s).upper() for s in self.symbols_config.get("skip_symbols", [])}

        self._server_offset_ms = 0
        self._last_server_poll_ms = 0
        self._db_locks: Dict[Path, asyncio.Lock] = {}
        self._frame_failures: Dict[str, int] = {}
        self.runtime_snapshot_interval_sec = float(getattr(cfg, "runtime_snapshot_interval_sec", 5.0) or 5.0)
        self.runtime_price_websocket_enabled = bool(getattr(cfg, "runtime_price_websocket_enabled", False))
        self.runtime_price_websocket_url = self._normalize_mark_price_websocket_url(
            str(getattr(cfg, "runtime_price_websocket_url", BINANCE_FUTURES_MARK_PRICE_WS_URL) or BINANCE_FUTURES_MARK_PRICE_WS_URL)
        )
        self.runtime_price_websocket_reconnect_sec = float(
            getattr(cfg, "runtime_price_websocket_reconnect_sec", 5.0) or 5.0
        )
        self._ws_mark_price_cache: dict[str, dict[str, Any]] = {}
        self._ws_mark_price_cache_ttl_sec = max(10.0, float(self.runtime_snapshot_interval_sec) * 3.0)
        # Step 4: all-market mark-price payloads are often partial.  Do not
        # repeatedly overwrite latest.json through HTTP fallback every time a
        # single websocket message omits a symbol; retain a fresh persisted
        # runtime price where possible and throttle unavoidable HTTP fallback.
        # Step 3 remains the execution safety boundary: stale prices still block
        # entries/flips even if retained for UI/PnL observability.
        self._ws_http_fallback_cooldown_sec = max(30.0, float(self.runtime_snapshot_interval_sec) * 6.0)
        self._last_ws_http_fallback_ms = 0
        self._ws_partial_snapshot_events = 0
        self._ws_http_fallback_events = 0
        self._ws_retained_snapshot_events = 0

    @staticmethod
    def _minimum_history_age_ms(requirements: dict[str, int]) -> int:
        """Return the minimum listing age needed by the shared history contract."""
        required_age_ms = 0
        for interval, required_rows in requirements.items():
            interval_ms = TIMEFRAME_TO_MS.get(str(interval))
            if interval_ms is None:
                continue
            required_age_ms = max(required_age_ms, int(required_rows) * int(interval_ms))
        return int(required_age_ms)

    def _history_age_eligible(self, symbol: str, *, now_ms: int | None = None) -> tuple[bool, str]:
        """Validate that exchange listing age can satisfy the history contract.

        When exchange metadata has been loaded, missing/invalid ``onboardDate``
        fails closed for ranked replacement admission.  If callers replace the
        exchange-symbol lookup in tests or a compatibility integration and no
        metadata cache exists at all, retain the historical behaviour.
        """
        if not self._exchange_onboard_date_ms:
            return True, "exchange_onboard_metadata_unavailable"
        onboard_ms = int(self._exchange_onboard_date_ms.get(str(symbol).upper(), 0) or 0)
        if onboard_ms <= 0:
            return False, "exchange_onboard_date_missing"
        reference_ms = int(now_ms if now_ms is not None else self.now_ms_corrected())
        age_ms = max(0, reference_ms - onboard_ms)
        if age_ms < self.minimum_history_age_ms:
            return False, f"insufficient_listing_history:{age_ms}/{self.minimum_history_age_ms}"
        return True, "history_age_eligible"

    @staticmethod
    def _normalize_mark_price_websocket_url(url: str) -> str:
        """Return a routed Binance Futures market websocket URL.

        Binance USDⓈ-M Futures mark-price streams belong to the /market routed
        endpoint. The legacy unrouted /ws endpoint can connect but may produce no
        market messages, which would starve latest.json if the HTTP watchdog were
        not present. Keep this normalizer local to the controller so existing
        configs using the older URL fail safe without requiring manual cleanup.
        """
        normalized = str(url or BINANCE_FUTURES_MARK_PRICE_WS_URL).strip()
        legacy_prefix = "wss://fstream.binance.com/ws/"
        routed_prefix = "wss://fstream.binance.com/market/ws/"
        if normalized.startswith(legacy_prefix):
            return routed_prefix + normalized[len(legacy_prefix):]
        return normalized

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}

    def _now_ms_local(self) -> int:
        return int(time.time() * 1000)

    def now_ms_corrected(self) -> int:
        return self._now_ms_local() + self._server_offset_ms

    def _publish_heartbeat(self, timeframe: str, mode: str) -> None:
        self.monitor.heartbeat("market_data", status="running", details={"timeframe": timeframe, "mode": mode})

    def _publish_connectivity(self) -> None:
        self.monitor.update_market_connectivity(status="connected")

    def _publish_universe(self, symbols: List[str], source: str) -> None:
        self.monitor.update_market_universe(symbols=symbols, source=source, last_refresh_ts=int(time.time()))

    def _publish_update(self, result: UpdateResult) -> None:
        self.monitor.update_market_result(symbol=result.symbol, timeframe=result.timeframe, rows_added=result.rows_added, last_ts=result.last_ts, status="updated" if result.rows_added > 0 else "no_change")

    async def _request_json(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None, retries: Optional[int] = None) -> Optional[Any]:
        timeout = timeout or self.request_timeout_sec
        retries = retries or self.request_retries

        def _shutdown_requested() -> bool:
            stop_event = self._runtime_stop_event
            return bool(stop_event is not None and stop_event.is_set())

        def _run() -> Optional[Any]:
            wait_s = 0.5
            for _ in range(retries):
                # Do not begin new blocking network work after coordinated
                # shutdown has been requested.  A request already in flight is
                # still allowed to complete under its existing bounded timeout.
                if _shutdown_requested():
                    return None
                try:
                    response = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "AutoTrader/MarketDataController"})
                    response.raise_for_status()
                    return response.json()
                except Exception:
                    if _shutdown_requested():
                        return None
                    stop_event = self._runtime_stop_event
                    if stop_event is not None:
                        if stop_event.wait(wait_s):
                            return None
                    else:
                        time.sleep(wait_s)
                    wait_s = min(wait_s * 1.7, 5.0)
            return None

        return await asyncio.to_thread(_run)


    async def refresh_runtime_price_snapshot(self, symbols: Optional[List[str]] = None, *, reason: str = "scheduled") -> dict[str, Any]:
        active_symbols = self._runtime_active_symbols(symbols)
        price_symbols = self._runtime_price_symbols(active_symbols)
        expected_count = len(price_symbols)
        if not price_symbols:
            snapshot = {
                "source": "binance_futures_http",
                "created_at_ms": int(self.now_ms_corrected()),
                "symbols": {},
                "reason": reason,
                "active_symbols": [],
                "open_position_symbols": [],
                "price_symbols": [],
                "expected_symbol_count": 0,
                "priced_symbol_count": 0,
                "missing_symbols": [],
            }
            write_market_latest_snapshot(self.paths, snapshot)
            self.logger.info("Market snapshot written: 0 symbols reason=%s path=%s", reason, self.paths.market_latest_snapshot_path)
            return snapshot
        try:
            snapshot = await self.fetch_runtime_price_snapshot(price_symbols)
            snapshot["reason"] = reason
            snapshot_symbols = snapshot.get("symbols") if isinstance(snapshot.get("symbols"), dict) else {}
            priced_symbols = [
                symbol
                for symbol in price_symbols
                if isinstance(snapshot_symbols.get(symbol), dict) and snapshot_symbols[symbol].get("primary_price")
            ]
            missing_symbols = [symbol for symbol in price_symbols if symbol not in priced_symbols]
            snapshot["active_symbols"] = active_symbols
            snapshot["open_position_symbols"] = self._runtime_open_position_symbols()
            snapshot["price_symbols"] = price_symbols
            snapshot["expected_symbol_count"] = expected_count
            snapshot["priced_symbol_count"] = len(priced_symbols)
            snapshot["missing_symbols"] = missing_symbols
            snapshot["all_symbols_priced"] = len(missing_symbols) == 0
            snapshot["snapshot_health"] = "ready" if snapshot["all_symbols_priced"] else "partial"
            write_market_latest_snapshot(self.paths, snapshot)
            self.logger.info(
                "Market snapshot written: %d/%d priced symbols missing=%s reason=%s path=%s",
                len(priced_symbols),
                expected_count,
                missing_symbols,
                reason,
                self.paths.market_latest_snapshot_path,
            )
            return snapshot
        except Exception:
            self.logger.exception("Market snapshot FAILED reason=%s path=%s", reason, self.paths.market_latest_snapshot_path)
            raise

    async def runtime_snapshot_loop(self, stop_event: threading.Event | None = None) -> None:
        if self.runtime_price_websocket_enabled:
            await self.websocket_mark_price_snapshot_loop(stop_event=stop_event)
            return
        await self.http_runtime_snapshot_loop(stop_event=stop_event)

    async def http_runtime_snapshot_loop(self, stop_event: threading.Event | None = None) -> None:
        while not (stop_event and stop_event.is_set()):
            try:
                await self.refresh_runtime_price_snapshot(reason="loop")
            except Exception as exc:
                self.monitor.publish_event(
                    level="WARNING",
                    component="market_data_controller",
                    category="runtime_snapshot_failed",
                    message="Runtime market snapshot refresh failed",
                    data={"error": str(exc)},
                    alert=False,
                )
            if await self._sleep_or_stop(stop_event, self.runtime_snapshot_interval_sec):
                return

    def _runtime_active_symbols(self, symbols: Optional[List[str]] = None) -> list[str]:
        active_symbols = [normalize_symbol(symbol) for symbol in (symbols or self._read_symbols())]
        return [symbol for symbol in active_symbols if symbol]

    def _runtime_open_position_symbols(self) -> list[str]:
        return [normalize_symbol(symbol) for symbol in load_open_position_symbols(self.paths) if normalize_symbol(symbol)]

    def _runtime_price_symbols(self, active_symbols: Optional[List[str]] = None) -> list[str]:
        active = self._runtime_active_symbols(active_symbols)
        return [normalize_symbol(symbol) for symbol in load_runtime_price_symbols(self.paths, active_symbols=active) if normalize_symbol(symbol)]

    def _fresh_persisted_runtime_price_rows(self, symbols: list[str], *, created_at_ms: int) -> dict[str, dict[str, Any]]:
        """Return fresh rows from the last persisted runtime snapshot.

        Binance's all-market websocket stream sends partial payloads.  A missing
        symbol in a single payload should not force an HTTP rewrite when the
        previous runtime snapshot already has a fresh exchange-derived price for
        that symbol.  DB-close rows are intentionally excluded because they are
        not a live execution-price source and Step 3 will fail closed on stale or
        unproven order prices.
        """
        payload = load_market_latest_snapshot(self.paths)
        rows = payload.get("symbols") if isinstance(payload.get("symbols"), dict) else {}
        if not isinstance(rows, dict):
            return {}
        max_age_ms = int(max(self._ws_mark_price_cache_ttl_sec, float(self.runtime_snapshot_interval_sec) * 2.0) * 1000)
        out: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            row = rows.get(symbol)
            if not isinstance(row, dict):
                continue
            source = str(row.get("price_source") or row.get("source") or "")
            if source in {"", "db_close", "unavailable", "snapshot_missing", "snapshot_unpriced"}:
                continue
            try:
                ts_ms = int(row.get("price_ts_ms") or row.get("event_time_ms") or row.get("timestamp") or 0)
            except Exception:
                continue
            if ts_ms <= 0 or created_at_ms - ts_ms > max_age_ms:
                continue
            price = row.get("primary_price") or row.get("mark_price") or row.get("latest_price") or row.get("price")
            try:
                if float(price) <= 0.0:
                    continue
            except Exception:
                continue
            retained = dict(row)
            retained["source"] = "runtime_price_snapshot_cached"
            retained["price_source"] = "runtime_price_snapshot_cached"
            retained["cache_origin_price_source"] = source
            retained["cache_origin_created_at_ms"] = payload.get("created_at_ms") or payload.get("created_ms")
            out[symbol] = retained
        return out

    def _should_write_websocket_http_fallback(self, *, now_ms: int) -> bool:
        latest_path = getattr(self.paths, "market_latest_snapshot_path", None)
        if latest_path is None or not Path(latest_path).exists():
            self._last_ws_http_fallback_ms = int(now_ms)
            self._ws_http_fallback_events += 1
            return True
        elapsed_sec = (int(now_ms) - int(self._last_ws_http_fallback_ms or 0)) / 1000.0
        if elapsed_sec >= self._ws_http_fallback_cooldown_sec:
            self._last_ws_http_fallback_ms = int(now_ms)
            self._ws_http_fallback_events += 1
            return True
        self._ws_retained_snapshot_events += 1
        return False

    def _build_mark_price_snapshot_from_ws(
        self,
        payload: Any,
        active_symbols: list[str],
        *,
        reason: str = "websocket",
        created_at_ms: int | None = None,
    ) -> dict[str, Any]:
        rows: Any = payload
        if isinstance(payload, dict) and "data" in payload:
            rows = payload.get("data")
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            rows = []

        active = [normalize_symbol(symbol) for symbol in active_symbols if normalize_symbol(symbol)]
        active_set = set(active)
        open_position_symbols = [symbol for symbol in self._runtime_open_position_symbols() if symbol in active_set]
        created_ms = int(created_at_ms or self.now_ms_corrected())
        persisted_runtime_rows = self._fresh_persisted_runtime_price_rows(active, created_at_ms=created_ms)

        def _as_float(value: Any) -> float | None:
            try:
                parsed = float(value)
            except Exception:
                return None
            return parsed if parsed > 0.0 else None

        ws_rows: dict[str, dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("s"))
            if not symbol or symbol not in active_set:
                continue
            mark_price = _as_float(item.get("p"))
            index_price = _as_float(item.get("i"))
            average_mark_price = _as_float(item.get("ap"))
            try:
                event_time_ms = int(item.get("E") or created_ms)
            except Exception:
                event_time_ms = created_ms
            funding_rate = None
            try:
                funding_rate = float(item.get("r")) if item.get("r") is not None else None
            except Exception:
                funding_rate = None
            next_funding_time_ms = None
            try:
                next_funding_time_ms = int(item.get("T")) if item.get("T") is not None else None
            except Exception:
                next_funding_time_ms = None
            db_close = self._latest_db_close(symbol, self.anchor_interval)
            primary_price = mark_price or average_mark_price or index_price or db_close
            ws_rows[symbol] = {
                "price": primary_price,
                "latest_price": primary_price,
                "reference_price": primary_price,
                "primary_price": primary_price,
                "mark_price": mark_price,
                "mark_price_average": average_mark_price,
                "index_price": index_price,
                "ticker_price": None,
                "db_close": db_close,
                "funding_rate": funding_rate,
                "next_funding_time_ms": next_funding_time_ms,
                "source": "mark_price_websocket" if mark_price else ("db_close" if db_close else "unavailable"),
                "price_source": "mark_price_websocket" if mark_price else ("db_close" if db_close else "unavailable"),
                "timestamp": event_time_ms,
                "price_ts_ms": event_time_ms,
                "event_time_ms": event_time_ms,
            }

        for symbol, row in ws_rows.items():
            # The all-market stream is not guaranteed to include every active symbol
            # in every payload. Keep the latest websocket mark price per symbol so a
            # partial payload does not overwrite live prices with DB closes.
            self._ws_mark_price_cache[symbol] = dict(row)

        snapshot_symbols: dict[str, dict[str, Any]] = {}
        cache_ttl_ms = int(self._ws_mark_price_cache_ttl_sec * 1000)
        for symbol in active:
            if symbol in ws_rows:
                snapshot_symbols[symbol] = ws_rows[symbol]
                continue

            cached = self._ws_mark_price_cache.get(symbol)
            cached_ts = None
            if isinstance(cached, dict):
                try:
                    cached_ts = int(cached.get("price_ts_ms") or cached.get("event_time_ms") or cached.get("timestamp") or 0)
                except Exception:
                    cached_ts = None
            if cached and cached_ts and created_ms - cached_ts <= cache_ttl_ms and cached.get("primary_price"):
                cached_row = dict(cached)
                cached_row["source"] = "mark_price_websocket_cached"
                cached_row["price_source"] = "mark_price_websocket_cached"
                snapshot_symbols[symbol] = cached_row
                continue

            persisted_row = persisted_runtime_rows.get(symbol)
            if isinstance(persisted_row, dict) and persisted_row.get("primary_price"):
                snapshot_symbols[symbol] = dict(persisted_row)
                continue

            db_close = self._latest_db_close(symbol, self.anchor_interval)
            snapshot_symbols[symbol] = {
                "price": db_close,
                "latest_price": db_close,
                "reference_price": db_close,
                "primary_price": db_close,
                "mark_price": None,
                "mark_price_average": None,
                "index_price": None,
                "ticker_price": None,
                "db_close": db_close,
                "source": "db_close" if db_close else "unavailable",
                "price_source": "db_close" if db_close else "unavailable",
                "timestamp": created_ms if db_close else None,
                "price_ts_ms": created_ms if db_close else None,
                "event_time_ms": created_ms if db_close else None,
            }

        priced_symbols = [symbol for symbol, row in snapshot_symbols.items() if row.get("primary_price")]
        missing_symbols = [symbol for symbol in active if symbol not in priced_symbols]
        snapshot = {
            "source": "binance_futures_ws_mark_price",
            "stream": "!markPrice@arr@1s",
            "created_at_ms": created_ms,
            "symbols": snapshot_symbols,
            "reason": reason,
            "active_symbols": self._runtime_active_symbols(),
            "open_position_symbols": open_position_symbols,
            "price_symbols": active,
            "expected_symbol_count": len(active),
            "priced_symbol_count": len(priced_symbols),
            "missing_symbols": missing_symbols,
            "all_symbols_priced": len(missing_symbols) == 0,
            "snapshot_health": "ready" if len(missing_symbols) == 0 else "partial",
            "websocket_cache_status": "ready" if len(missing_symbols) == 0 else "partial",
            "websocket_partial_snapshot_events": self._ws_partial_snapshot_events,
            "websocket_http_fallback_events": self._ws_http_fallback_events,
            "websocket_retained_snapshot_events": self._ws_retained_snapshot_events,
        }
        return snapshot

    async def websocket_mark_price_snapshot_loop(self, stop_event: threading.Event | None = None) -> None:
        try:
            import websockets  # type: ignore
        except Exception as exc:
            self.logger.warning("websockets package unavailable; falling back to HTTP runtime price snapshots: %s", exc)
            self.monitor.publish_event(
                level="WARNING",
                component="market_data_controller",
                category="websocket_price_unavailable",
                message="Websocket mark-price stream unavailable; using HTTP runtime snapshot fallback",
                data={"error": str(exc)},
                alert=False,
            )
            await self.http_runtime_snapshot_loop(stop_event=stop_event)
            return

        while not (stop_event and stop_event.is_set()):
            active_symbols = self._runtime_price_symbols()
            if not active_symbols:
                await self.refresh_runtime_price_snapshot(symbols=[], reason="websocket_no_symbols")
                if await self._sleep_or_stop(stop_event, self.runtime_price_websocket_reconnect_sec):
                    return
                continue
            try:
                self.logger.info("Connecting Binance Futures mark-price websocket: %s", self.runtime_price_websocket_url)
                self.monitor.publish_event(
                    level="INFO",
                    component="market_data_controller",
                    category="websocket_price_connecting",
                    message="Connecting Binance Futures mark-price websocket",
                    data={"url": self.runtime_price_websocket_url, "symbols": len(active_symbols)},
                    alert=False,
                )
                async with websockets.connect(self.runtime_price_websocket_url, ping_interval=20, ping_timeout=20, close_timeout=5) as websocket:
                    # Binance documents !markPrice@arr@1s as a 1000ms stream. Do not allow a
                    # connected-but-silent websocket to starve latest.json; execution depends on
                    # this file for the no-price/no-trade guard. If no message arrives inside the
                    # normal snapshot window, write one HTTP fallback snapshot and keep listening.
                    websocket_read_timeout_sec = max(3.0, float(self.runtime_snapshot_interval_sec))
                    while not (stop_event and stop_event.is_set()):
                        try:
                            raw_message = await asyncio.wait_for(websocket.recv(), timeout=websocket_read_timeout_sec)
                        except asyncio.TimeoutError:
                            now_ms = self.now_ms_corrected()
                            use_http_fallback = self._should_write_websocket_http_fallback(now_ms=now_ms)
                            self.logger.warning(
                                "Mark-price websocket produced no message for %.1fs; %s",
                                websocket_read_timeout_sec,
                                "writing HTTP fallback snapshot" if use_http_fallback else "retaining previous snapshot until fallback cooldown elapses",
                            )
                            self.monitor.publish_event(
                                level="WARNING",
                                component="market_data_controller",
                                category="websocket_price_silent_fallback" if use_http_fallback else "websocket_price_silent_retained",
                                message=(
                                    "Mark-price websocket silent; wrote HTTP fallback snapshot"
                                    if use_http_fallback
                                    else "Mark-price websocket silent; retained previous snapshot until fallback cooldown elapses"
                                ),
                                data={
                                    "timeout_sec": websocket_read_timeout_sec,
                                    "fallback_cooldown_sec": self._ws_http_fallback_cooldown_sec,
                                    "http_fallback_written": use_http_fallback,
                                },
                                alert=False,
                            )
                            if use_http_fallback:
                                try:
                                    await self.refresh_runtime_price_snapshot(reason="websocket_silent_http_fallback")
                                except Exception:
                                    self.logger.exception("HTTP fallback snapshot failed while websocket was silent")
                            continue
                        try:
                            payload = json.loads(raw_message)
                        except Exception as exc:
                            self.logger.warning("Invalid websocket price payload: %s", exc)
                            continue
                        active_symbols = self._runtime_price_symbols()
                        snapshot = self._build_mark_price_snapshot_from_ws(
                            payload,
                            active_symbols,
                            reason="websocket",
                            created_at_ms=self.now_ms_corrected(),
                        )
                        # Do not let a partial or still-warming websocket snapshot overwrite
                        # a healthy HTTP/runtime snapshot. The execution layer treats
                        # latest.json as authoritative; writing 0/10 or DB-close-backed rows
                        # here can temporarily create false price_unavailable or stale-price
                        # states. Until every active symbol has a websocket mark price, use
                        # one HTTP fallback snapshot and keep listening. Once the per-symbol
                        # websocket cache is warm, partial Binance payloads are completed from
                        # that cache and are safe to publish.
                        ws_symbol_sources = {
                            symbol: str((snapshot.get("symbols") or {}).get(symbol, {}).get("price_source") or "")
                            for symbol in active_symbols
                        }
                        incomplete_symbols = [
                            symbol
                            for symbol in active_symbols
                            if ws_symbol_sources.get(symbol) not in {"mark_price_websocket", "mark_price_websocket_cached", "runtime_price_snapshot_cached"}
                        ]
                        if snapshot.get("missing_symbols") or incomplete_symbols:
                            self._ws_partial_snapshot_events += 1
                            now_ms = self.now_ms_corrected()
                            use_http_fallback = self._should_write_websocket_http_fallback(now_ms=now_ms)
                            fallback_phrase = "writing HTTP fallback" if use_http_fallback else "retaining previous snapshot"
                            self.logger.warning(
                                "Incomplete websocket mark-price snapshot; %s: "
                                "priced=%d/%d missing=%s incomplete=%s",
                                fallback_phrase,
                                snapshot.get("priced_symbol_count", 0),
                                snapshot.get("expected_symbol_count", 0),
                                snapshot.get("missing_symbols", []),
                                incomplete_symbols,
                            )
                            self.monitor.publish_event(
                                level="WARNING",
                                component="market_data_controller",
                                category="websocket_price_incomplete_fallback" if use_http_fallback else "websocket_price_incomplete_retained",
                                message=(
                                    "Incomplete websocket mark-price snapshot; wrote HTTP fallback snapshot"
                                    if use_http_fallback
                                    else "Incomplete websocket mark-price snapshot; retained previous snapshot until cache warms"
                                ),
                                data={
                                    "priced_symbol_count": snapshot.get("priced_symbol_count", 0),
                                    "expected_symbol_count": snapshot.get("expected_symbol_count", 0),
                                    "missing_symbols": snapshot.get("missing_symbols", []),
                                    "incomplete_symbols": incomplete_symbols,
                                    "fallback_cooldown_sec": self._ws_http_fallback_cooldown_sec,
                                    "http_fallback_written": use_http_fallback,
                                },
                                alert=False,
                            )
                            if use_http_fallback:
                                try:
                                    await self.refresh_runtime_price_snapshot(reason="websocket_incomplete_http_fallback")
                                except Exception:
                                    self.logger.exception("HTTP fallback snapshot failed while websocket cache was incomplete")
                            continue

                        write_market_latest_snapshot(self.paths, snapshot)
                        self.logger.info(
                            "Market websocket snapshot written: %d/%d priced symbols missing=%s path=%s",
                            snapshot.get("priced_symbol_count", 0),
                            snapshot.get("expected_symbol_count", 0),
                            snapshot.get("missing_symbols", []),
                            self.paths.market_latest_snapshot_path,
                        )
            except Exception as exc:
                self.logger.warning("Mark-price websocket disconnected; falling back to one HTTP snapshot before reconnect: %s", exc)
                self.monitor.publish_event(
                    level="WARNING",
                    component="market_data_controller",
                    category="websocket_price_reconnect",
                    message="Mark-price websocket disconnected; reconnecting",
                    data={"error": str(exc), "reconnect_sec": self.runtime_price_websocket_reconnect_sec},
                    alert=False,
                )
                if self._should_write_websocket_http_fallback(now_ms=self.now_ms_corrected()):
                    try:
                        await self.refresh_runtime_price_snapshot(reason="websocket_reconnect_http_fallback")
                    except Exception:
                        self.logger.exception("HTTP fallback snapshot failed during websocket reconnect")
                if await self._sleep_or_stop(stop_event, self.runtime_price_websocket_reconnect_sec):
                    return

    async def fetch_runtime_price_snapshot(self, symbols: List[str]) -> dict[str, Any]:
        normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)]
        ticker_payload, premium_payload, server_ms = await asyncio.gather(
            self._request_json(BINANCE_FUTURES_PRICE_TICKER_URL, timeout=self.request_timeout_sec, retries=self.request_retries),
            self._request_json(BINANCE_FUTURES_PREMIUM_INDEX_URL, timeout=self.request_timeout_sec, retries=self.request_retries),
            self.sync_server_time(),
        )
        ticker_rows = ticker_payload if isinstance(ticker_payload, list) else []
        premium_rows = premium_payload if isinstance(premium_payload, list) else []
        ticker_map: Dict[str, dict[str, Any]] = {}
        premium_map: Dict[str, dict[str, Any]] = {}
        for item in ticker_rows:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            if not symbol:
                continue
            ticker_map[symbol] = item
        for item in premium_rows:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            if not symbol:
                continue
            premium_map[symbol] = item

        snapshot_symbols: Dict[str, dict[str, Any]] = {}
        created_at_ms = int(server_ms or self.now_ms_corrected())
        for symbol in normalized_symbols:
            ticker_item = ticker_map.get(symbol, {})
            premium_item = premium_map.get(symbol, {})
            def _as_float(value: Any) -> float | None:
                try:
                    parsed = float(value)
                except Exception:
                    return None
                return parsed if parsed > 0.0 else None
            ticker_price = _as_float(ticker_item.get("price"))
            mark_price = _as_float(premium_item.get("markPrice"))
            db_close = self._latest_db_close(symbol, self.anchor_interval)
            primary_price = mark_price or ticker_price or db_close
            event_time_ms = premium_item.get("time") or created_at_ms
            try:
                event_time_ms = int(event_time_ms) if event_time_ms is not None else created_at_ms
            except Exception:
                event_time_ms = created_at_ms
            snapshot_symbols[symbol] = {
                "price": primary_price,
                "ticker_price": ticker_price,
                "mark_price": mark_price,
                "db_close": db_close,
                "primary_price": primary_price,
                "source": "mark_price" if mark_price else ("ticker_price" if ticker_price else ("db_close" if db_close else "unavailable")),
                "price_source": "mark_price" if mark_price else ("ticker_price" if ticker_price else ("db_close" if db_close else "unavailable")),
                "timestamp": event_time_ms,
                "price_ts_ms": event_time_ms,
                "event_time_ms": event_time_ms,
            }
        snapshot = {
            "source": "binance_futures_http",
            "created_at_ms": created_at_ms,
            "symbols": snapshot_symbols,
            "expected_symbol_count": len(normalized_symbols),
            "priced_symbol_count": sum(1 for row in snapshot_symbols.values() if row.get("primary_price")),
            "missing_symbols": [symbol for symbol, row in snapshot_symbols.items() if not row.get("primary_price")],
        }
        snapshot["all_symbols_priced"] = not snapshot["missing_symbols"]
        snapshot["snapshot_health"] = "ready" if snapshot["all_symbols_priced"] else "partial"
        return snapshot

    def _latest_db_close(self, symbol: str, timeframe: str) -> float | None:
        db_path = self.paths.market_db_path(symbol)
        if not db_path.exists():
            return None
        table_name = f'klines_{timeframe}'
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    f'SELECT close FROM "{table_name}" WHERE close IS NOT NULL ORDER BY timestamp DESC LIMIT 1'
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        try:
            value = float(row[0])
        except Exception:
            return None
        return value if value > 0.0 else None

    async def sync_server_time(self) -> int:
        now = self._now_ms_local()
        if self._server_offset_ms != 0 and now - self._last_server_poll_ms < self.server_time_cache_ms:
            return self.now_ms_corrected()

        payload = await self._request_json(BINANCE_FUTURES_TIME_URL, timeout=self.request_timeout_sec, retries=self.request_retries)
        if isinstance(payload, dict) and "serverTime" in payload:
            server_ms = int(payload["serverTime"])
            local_ms = self._now_ms_local()
            self._server_offset_ms = server_ms - local_ms
            self._last_server_poll_ms = local_ms
        return self.now_ms_corrected()

    async def fetch_exchange_tradable_symbols(self) -> set[str]:
        payload = await self._request_json(BINANCE_FUTURES_EXCHANGE_INFO_URL, timeout=self.request_timeout_sec, retries=self.request_retries)
        if not isinstance(payload, dict):
            self._exchange_onboard_date_ms = {}
            return set()
        symbols_payload = payload.get("symbols")
        if not isinstance(symbols_payload, list):
            self._exchange_onboard_date_ms = {}
            return set()
        tradable: set[str] = set()
        onboard_dates: dict[str, int] = {}
        for item in symbols_payload:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            if not is_valid_usdt_symbol(symbol):
                continue
            if symbol in self.skip_symbols:
                continue
            if str(item.get("status") or "").upper() != "TRADING":
                continue
            if str(item.get("contractType") or "").upper() not in {"PERPETUAL"}:
                continue
            if str(item.get("quoteAsset") or "").upper() != "USDT":
                continue
            tradable.add(symbol)
            try:
                onboard_dates[symbol] = int(item.get("onboardDate") or 0)
            except Exception:
                onboard_dates[symbol] = 0
        self._exchange_onboard_date_ms = onboard_dates
        return tradable

    async def fetch_volume_ranked_symbols(self) -> List[str]:
        exchange_tradable = await self.fetch_exchange_tradable_symbols()
        payload = await self._request_json(BINANCE_FUTURES_24H_URL, timeout=self.request_timeout_sec, retries=self.request_retries)
        rows = payload if isinstance(payload, list) else []
        filtered: List[Tuple[str, float]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = normalize_symbol(item.get("symbol"))
            if not is_valid_usdt_symbol(symbol) or symbol in self.skip_symbols:
                continue
            if exchange_tradable and symbol not in exchange_tradable:
                continue
            try:
                quote_volume = float(item.get("quoteVolume") or item.get("quote_volume") or 0.0)
            except Exception:
                continue
            if quote_volume <= 0.0:
                continue
            filtered.append((symbol, quote_volume))
        filtered.sort(key=lambda x: x[1], reverse=True)
        now_ms = self.now_ms_corrected()
        ranked: list[str] = []
        rejected_for_history: list[tuple[str, str]] = []
        for symbol, _ in filtered:
            eligible, reason = self._history_age_eligible(symbol, now_ms=now_ms)
            if eligible:
                ranked.append(symbol)
            else:
                rejected_for_history.append((symbol, reason))
        if rejected_for_history:
            self.logger.info(
                "Excluded %d ranked replacement candidates that cannot satisfy historical-data readiness: %s",
                len(rejected_for_history),
                rejected_for_history[:10],
            )
        return ranked

    def _active_symbols_for_reporting(self) -> list[str]:
        state = load_symbol_state(self.paths)
        active = state.get("active_symbols", [])
        if not isinstance(active, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for raw_symbol in active:
            symbol = normalize_symbol(raw_symbol)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    def _read_symbols(self) -> List[str]:
        return load_active_universe_symbols(self.paths, limit=self.target_active_symbols, include_db_fallback=False)

    def market_freshness_snapshot(self, symbol: str, interval: str) -> dict[str, Any]:
        latest_ts = latest_market_timestamp(self.paths, symbol, interval)
        return validate_market_data_freshness(self.paths, symbol=symbol, interval=interval, latest_timestamp_ms=latest_ts)

    async def refresh_universe_once(self) -> List[str]:
        """Refresh exchange-ranked replacement candidates only.

        Symbol lifecycle ownership belongs to TechRevController.  MarketDataController
        must not mutate active_symbols, blocked symbols, retry state, or the exported
        tradable_symbols.yaml file during a universe refresh; doing so can revive
        blocked symbols and overwrite replacement decisions made by the training
        lifecycle controller.
        """
        ranked = [normalize_symbol(symbol) for symbol in await self.fetch_volume_ranked_symbols()]
        ranked = [symbol for symbol in ranked if symbol]
        ranked = ranked[: max(self.target_active_symbols, self.ranked_universe_size)]
        state = load_symbol_state(self.paths)
        state["ranked_candidates"] = [str(symbol).upper() for symbol in ranked]
        state["ranked_universe_size"] = self.ranked_universe_size
        state["target_active_symbols"] = self.target_active_symbols
        save_symbol_state(self.paths, state)

        active_for_reporting = self._active_symbols_for_reporting()
        self._publish_universe(
            active_for_reporting or state["ranked_candidates"][: self.target_active_symbols],
            source="volume_rank_candidates",
        )
        return state["ranked_candidates"]

    async def universe_refresh_loop(self, stop_event: threading.Event | None = None) -> None:
        while not (stop_event and stop_event.is_set()):
            try:
                await self.refresh_universe_once()
            except Exception as exc:
                self.logger.exception("Universe refresh failed")
                self.monitor.publish_event(level="ERROR", component="market_data_controller", category="universe_refresh_failed", message="Universe refresh failed", data={"error": str(exc)}, alert=True)
            if stop_event is None:
                await asyncio.sleep(self.refresh_universe_sec)
            else:
                if await asyncio.to_thread(stop_event.wait, self.refresh_universe_sec):
                    return

    async def fetch_klines(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        limit = 1500
        interval_ms = TIMEFRAME_TO_MS[timeframe]
        current = start_ms
        out: List[pd.DataFrame] = []
        while current <= end_ms:
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "startTime": current,
                "endTime": end_ms,
                "limit": limit,
            }
            payload = await self._request_json(BINANCE_FUTURES_KLINES_URL, params=params)
            if not isinstance(payload, list) or not payload:
                break
            rows = []
            for item in payload:
                if not isinstance(item, list) or len(item) < 6:
                    continue
                rows.append(
                    {
                        "timestamp": int(item[0]),
                        "open": item[1],
                        "high": item[2],
                        "low": item[3],
                        "close": item[4],
                        "volume": item[5],
                    }
                )
            df = pd.DataFrame(rows)
            if df.empty:
                break
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("int64")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna().reset_index(drop=True)
            if df.empty:
                break
            out.append(df)
            if len(df) < limit:
                break
            current = int(df["timestamp"].iloc[-1]) + interval_ms
            await asyncio.sleep(0.2)
        if not out:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        merged = pd.concat(out, ignore_index=True)
        return merged.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    def _db_path_for_symbol(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{symbol.upper()}.db"

    def _table_name_for_timeframe(self, timeframe: str) -> str:
        return f"klines_{timeframe}"

    def _update_symbol_timeframe_sync(self, db_path: Path, table_name: str, frame: pd.DataFrame) -> Tuple[int, int]:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        last_exc: sqlite3.OperationalError | None = None
        for attempt in range(4):
            try:
                with sqlite3.connect(str(db_path), timeout=60) as conn:
                    conn.execute("PRAGMA busy_timeout=60000;")
                    try:
                        conn.execute("PRAGMA journal_mode=WAL;")
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc).lower():
                            raise
                    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" (timestamp INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)')
                    conn.execute(f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_timestamp" ON "{table_name}" (timestamp)')
                    row = conn.execute(f'SELECT MAX(timestamp) FROM "{table_name}"').fetchone()
                    prev_max = int(row[0]) if row and row[0] is not None else 0
                    rows = [(int(r.timestamp), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)) for r in frame.itertuples(index=False)]
                    conn.executemany(f'INSERT OR IGNORE INTO "{table_name}" (timestamp, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)', rows)
                    conn.commit()
                    row2 = conn.execute(f'SELECT MAX(timestamp) FROM "{table_name}"').fetchone()
                    new_max = int(row2[0]) if row2 and row2[0] is not None else 0
                rows_added = int((frame["timestamp"] > prev_max).sum()) if prev_max > 0 else int(len(frame))
                return rows_added, new_max
            except sqlite3.OperationalError as exc:
                last_exc = exc
                if "locked" not in str(exc).lower() or attempt >= 3:
                    raise
                time.sleep(0.25 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        raise sqlite3.OperationalError("sqlite update failed without an exception")

    async def update_symbol_timeframe(self, symbol: str, timeframe: str) -> UpdateResult:
        symbol = symbol.upper()
        if symbol in self.skip_symbols or timeframe not in TIMEFRAME_TO_MS:
            return UpdateResult(symbol, timeframe, 0, 0)

        interval_ms = TIMEFRAME_TO_MS[timeframe]
        now_ms = await self.sync_server_time()
        current_open = (now_ms // interval_ms) * interval_ms
        last_closed_open = current_open - interval_ms
        if last_closed_open <= 0:
            return UpdateResult(symbol, timeframe, 0, 0)

        db_path = self._db_path_for_symbol(symbol)
        table_name = self._table_name_for_timeframe(timeframe)
        db_lock = self._db_locks.setdefault(db_path, asyncio.Lock())

        async with db_lock:
            def _get_last_ts() -> int:
                if not db_path.exists():
                    return 0
                try:
                    with sqlite3.connect(str(db_path), timeout=10) as conn:
                        row = conn.execute(f'SELECT MAX(timestamp) FROM "{table_name}"').fetchone()
                        return int(row[0]) if row and row[0] is not None else 0
                except Exception:
                    return 0

            last_ts = await asyncio.to_thread(_get_last_ts)
            start_ms = last_closed_open - self.lookback_days.get(timeframe, 7) * 24 * 60 * 60 * 1000 if last_ts <= 0 else last_ts + interval_ms
            if start_ms > last_closed_open:
                return UpdateResult(symbol, timeframe, 0, last_ts)

            frame = await self.fetch_klines(symbol=symbol, timeframe=timeframe, start_ms=start_ms, end_ms=last_closed_open)
            if frame.empty:
                return UpdateResult(symbol, timeframe, 0, last_ts)

            rows_added, new_last_ts = await asyncio.to_thread(self._update_symbol_timeframe_sync, db_path, table_name, frame)
            return UpdateResult(symbol, timeframe, rows_added, new_last_ts)

    def _symbol_history_ready(self, symbol: str) -> bool:
        db_path = self._db_path_for_symbol(symbol)
        if not db_path.exists():
            return False
        try:
            with sqlite3.connect(str(db_path), timeout=5) as conn:
                for interval, required_rows in self.history_row_requirements.items():
                    table_name = self._table_name_for_timeframe(interval)
                    try:
                        row = conn.execute(
                            f'SELECT COUNT(1), COALESCE(MAX(timestamp), 0) FROM "{table_name}"'
                        ).fetchone()
                    except sqlite3.Error:
                        return False
                    if not row or int(row[0] or 0) < int(required_rows) or int(row[1] or 0) <= 0:
                        return False
        except sqlite3.Error:
            return False
        return True

    async def bootstrap_new_active_symbols_once(self) -> list[str]:
        """Immediately backfill newly active symbols across all fusion intervals.

        Slow-frame loops run on their candle cadence. A symbol promoted between
        those runs can otherwise wait four hours (or a day) before its first
        historical backfill, which is longer than the lifecycle warm-up timeout.
        This bounded bootstrap runs only for newly active/unready symbols and
        stops once the same history contract used by TechRev is satisfied.
        """
        active_symbols = self._read_symbols()
        if not active_symbols:
            return []
        now_ms = self.now_ms_corrected()
        bootstrapped: list[str] = []
        for symbol in active_symbols:
            normalized = normalize_symbol(symbol)
            if not normalized or normalized in self._history_bootstrap_complete:
                continue
            if self._symbol_history_ready(normalized):
                self._history_bootstrap_complete.add(normalized)
                bootstrapped.append(normalized)
                continue
            last_attempt = int(self._history_bootstrap_last_attempt_ms.get(normalized, 0) or 0)
            if last_attempt and now_ms - last_attempt < self._history_bootstrap_retry_ms:
                continue
            self._history_bootstrap_last_attempt_ms[normalized] = now_ms
            for interval in self.history_row_requirements:
                try:
                    result = await self.update_symbol_timeframe(normalized, interval)
                    self._publish_update(result)
                except Exception as exc:
                    self.logger.warning(
                        "Active-symbol history bootstrap failed symbol=%s interval=%s error=%s",
                        normalized,
                        interval,
                        exc,
                    )
                    break
            if self._symbol_history_ready(normalized):
                self._history_bootstrap_complete.add(normalized)
                bootstrapped.append(normalized)
                self.logger.info("Active-symbol history bootstrap complete symbol=%s", normalized)
        active_set = set(active_symbols)
        self._history_bootstrap_complete.intersection_update(active_set)
        self._history_bootstrap_last_attempt_ms = {
            symbol: ts for symbol, ts in self._history_bootstrap_last_attempt_ms.items() if symbol in active_set
        }
        return bootstrapped

    async def active_history_bootstrap_loop(self, stop_event: threading.Event | None = None) -> None:
        while not (stop_event and stop_event.is_set()):
            try:
                await self.bootstrap_new_active_symbols_once()
            except Exception as exc:
                self.logger.exception("Active-symbol history bootstrap loop failed")
                self.monitor.publish_event(
                    level="ERROR",
                    component="market_data_controller",
                    category="active_history_bootstrap_failed",
                    message="Active-symbol history bootstrap failed",
                    data={"error": str(exc)},
                    alert=True,
                )
            if await self._sleep_or_stop(stop_event, max(5.0, float(self.runtime_snapshot_interval_sec))):
                return

    async def update_optional_context_for_symbol(self, symbol: str, timeframe: str) -> dict[str, int]:
        """Refresh canonical futures-context tables for the active anchor interval.

        Funding rate and open interest are now owned by the ingestion layer and
        persisted canonically into each symbol database. These refreshes are
        intentionally tied to the anchor interval so train/infer consumers have
        one stable source of truth rather than opportunistic side reads.
        """
        if not self.include_optional_futures_context:
            return {"funding_rate_rows": 0, "open_interest_rows": 0}
        if str(timeframe) != self.anchor_interval:
            return {"funding_rate_rows": 0, "open_interest_rows": 0}

        open_interest_period = resolve_open_interest_period(timeframe)
        funding_rows = 0
        open_interest_rows = 0
        try:
            funding_rows = await asyncio.to_thread(refresh_funding_rate_data, self.paths, symbol)
        except Exception as exc:
            self.logger.exception("Funding-rate refresh failed for %s", symbol)
            self.monitor.publish_event(
                level="WARNING",
                component="market_data_controller",
                category="funding_rate_refresh_failed",
                message=f"Funding-rate refresh failed for {symbol}",
                data={"symbol": symbol, "error": str(exc)},
                alert=False,
            )
        try:
            open_interest_rows = await asyncio.to_thread(refresh_open_interest_data, self.paths, symbol, open_interest_period)
        except Exception as exc:
            self.logger.exception("Open-interest refresh failed for %s period=%s", symbol, open_interest_period)
            self.monitor.publish_event(
                level="WARNING",
                component="market_data_controller",
                category="open_interest_refresh_failed",
                message=f"Open-interest refresh failed for {symbol}",
                data={"symbol": symbol, "period": open_interest_period, "error": str(exc)},
                alert=False,
            )
        return {"funding_rate_rows": int(funding_rows), "open_interest_rows": int(open_interest_rows)}

    async def _run_frame_once(self, timeframe: str, concurrency: int) -> None:
        symbols = self._read_symbols()
        semaphore = asyncio.Semaphore(concurrency)
        results: Dict[str, UpdateResult] = {}

        async def worker(symbol: str) -> None:
            async with semaphore:
                try:
                    result = await self.update_symbol_timeframe(symbol, timeframe)
                    results[symbol] = result
                    self._publish_update(result)
                    if result.last_ts > 0:
                        context_rows = await self.update_optional_context_for_symbol(result.symbol, result.timeframe)
                        if any(context_rows.values()):
                            self.monitor.publish_event(
                                level="INFO",
                                component="market_data_controller",
                                category="optional_context_updated",
                                message=f"Updated futures context for {result.symbol}",
                                data={"symbol": result.symbol, "timeframe": result.timeframe, **context_rows},
                                alert=False,
                            )
                    self._frame_failures[timeframe] = 0
                    if result.rows_added > 0 and result.last_ts > 0 and self.training_tick_callback is not None:
                        try:
                            maybe = self.training_tick_callback(result.symbol, result.timeframe, result.last_ts)
                            if asyncio.iscoroutine(maybe):
                                await maybe
                        except Exception:
                            self.logger.exception("Training tick callback failed for %s %s", result.symbol, result.timeframe)
                except Exception as exc:
                    self.logger.exception("Update failed for %s %s", symbol, timeframe)
                    self._frame_failures[timeframe] = self._frame_failures.get(timeframe, 0) + 1
                    self.monitor.publish_event(level="ERROR", component="market_data_controller", category="frame_update_failed", message=f"Update failed for {symbol} {timeframe}", data={"error": str(exc), "symbol": symbol, "timeframe": timeframe}, alert=True)
                    if self._frame_failures[timeframe] >= 3:
                        self.monitor.controller_error("market_data", f"Repeated update failures on {timeframe}: {self._frame_failures[timeframe]}")
                    results[symbol] = UpdateResult(symbol, timeframe, 0, 0)

        await asyncio.gather(*(worker(symbol) for symbol in symbols), return_exceptions=True)
        tick_ts = self.now_ms_corrected()
        for symbol in symbols:
            result = results.get(symbol, UpdateResult(symbol, timeframe, 0, 0))
            self.logger.info("[UPD] tick=%s tf=%s symbol=%s rows_added=%s last_ts=%s", tick_ts, result.timeframe, result.symbol, result.rows_added, result.last_ts)

    async def run_fast_frame_loop(self, timeframe: str, stop_event: threading.Event | None = None) -> None:
        interval_ms = TIMEFRAME_TO_MS[timeframe]
        first = True
        while not (stop_event and stop_event.is_set()):
            if not first:
                now_ms = self.now_ms_corrected()
                next_open = ((now_ms // interval_ms) + 1) * interval_ms
                wake_ms = next_open + self.bar_close_delay_ms
                if await self._sleep_or_stop(stop_event, max(0.0, (wake_ms - now_ms) / 1000.0)):
                    return
            first = False
            self._publish_heartbeat(timeframe, "fast")
            await self._run_frame_once(timeframe, self.sem_fast)

    async def run_slow_frame_loop(self, timeframe: str, stop_event: threading.Event | None = None) -> None:
        interval_ms = TIMEFRAME_TO_MS[timeframe]
        while not (stop_event and stop_event.is_set()):
            start_ms = self.now_ms_corrected()
            self._publish_heartbeat(timeframe, "slow")
            await self._run_frame_once(timeframe, self.sem_slow)
            elapsed_ms = self.now_ms_corrected() - start_ms
            if await self._sleep_or_stop(stop_event, max(0.0, (interval_ms - elapsed_ms) / 1000.0)):
                return

    async def _sleep_or_stop(self, stop_event: threading.Event | None, seconds: float) -> bool:
        if stop_event is None:
            await asyncio.sleep(seconds)
            return False
        return await asyncio.to_thread(stop_event.wait, max(0.0, seconds))

    async def run_async(self, stop_event: threading.Event | None = None) -> None:
        self._runtime_stop_event = stop_event
        if not self.enabled:
            self.logger.warning("Market updater is disabled in config")
            self.monitor.publish_event(level="WARNING", component="market_data_controller", category="updater_disabled", message="Market updater is disabled in config", data={}, alert=True)
            while not (stop_event and stop_event.is_set()):
                if await self._sleep_or_stop(stop_event, 60):
                    return

        await self.sync_server_time()
        self._publish_connectivity()
        tasks: List[asyncio.Task[Any]] = []

        if self.refresh_universe_enabled:
            try:
                await self.refresh_universe_once()
            except Exception as exc:
                self.logger.exception("Startup universe refresh failed")
                self.monitor.publish_event(
                    level="ERROR",
                    component="market_data_controller",
                    category="universe_refresh_failed",
                    message="Startup universe refresh failed",
                    data={"error": str(exc)},
                    alert=True,
                )
            tasks.append(asyncio.create_task(self.universe_refresh_loop(stop_event=stop_event), name="universe-refresh"))

        await self.refresh_runtime_price_snapshot(reason="startup")
        tasks.append(asyncio.create_task(self.runtime_snapshot_loop(stop_event=stop_event), name="runtime-snapshot"))
        tasks.append(asyncio.create_task(self.active_history_bootstrap_loop(stop_event=stop_event), name="active-history-bootstrap"))

        for timeframe in self.fast_frames:
            tasks.append(asyncio.create_task(self.run_fast_frame_loop(timeframe, stop_event=stop_event), name=f"fast-{timeframe}"))
        for timeframe in self.slow_frames:
            tasks.append(asyncio.create_task(self.run_slow_frame_loop(timeframe, stop_event=stop_event), name=f"slow-{timeframe}"))

        self.logger.info("MarketDataController started: fast=%s slow=%s", self.fast_frames, self.slow_frames)

        if stop_event is None:
            await asyncio.gather(*tasks)
            return

        async def _wait_for_stop() -> None:
            # Poll the shared threading.Event from the event loop instead of
            # delegating an indefinite wait to the default executor.  This
            # keeps shutdown cancellation local to the controller and avoids
            # leaving a stop-wait worker behind while asyncio.run() is closing.
            while not stop_event.is_set():
                await asyncio.sleep(0.05)

        worker_group = asyncio.gather(*tasks)
        stop_waiter = asyncio.create_task(_wait_for_stop(), name="market-data-stop-waiter")
        try:
            done, _ = await asyncio.wait(
                {worker_group, stop_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker_group in done:
                # Preserve the historical crash/failure contract: if any
                # worker exits with an exception before shutdown is requested,
                # propagate it to run_forever() and the launcher supervisor.
                await worker_group
                return

            # Shutdown was requested.  Do not wait for websocket receive or
            # long frame sleeps to expire naturally; cancel the controller's
            # async workers and drain them before returning to asyncio.run().
            worker_group.cancel()
            try:
                await worker_group
            except asyncio.CancelledError:
                pass
        finally:
            if not stop_waiter.done():
                stop_waiter.cancel()
                try:
                    await stop_waiter
                except asyncio.CancelledError:
                    pass

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        self.logger.info("MarketDataController started")
        try:
            self.monitor.heartbeat("market_data", status="running", details={"phase": "startup"})
            asyncio.run(self.run_async(stop_event=stop_event))
        except KeyboardInterrupt:
            self.logger.warning("MarketDataController interrupted by user")
        except Exception as exc:
            self.logger.exception("MarketDataController crashed")
            self.monitor.controller_error("market_data", str(exc))
            self.monitor.publish_event(level="ERROR", component="market_data_controller", category="controller_crashed", message="Market data controller crashed", data={"error": str(exc)}, alert=True)
            raise
        finally:
            # During coordinated shutdown the monitoring publisher may be
            # contended by the standalone trainer/dashboard processes.  The
            # runtime coordinator already owns final shutdown reporting; do not
            # keep the MarketData thread alive waiting on cross-process
            # observability persistence after its workers have stopped.
            if not (stop_event and stop_event.is_set()):
                self.monitor.flush()
