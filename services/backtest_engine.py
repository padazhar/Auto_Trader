from __future__ import annotations

import csv
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from core.market_io import latest_symbol_price_status
from core.shared_utils import atomic_write_bytes


PRETTY_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<symbol>[A-Z0-9_]+)\s+\|\s*Mode:\s*(?P<mode>[^|]+)\|\s*"
    r"(?P<action>ENTER LONG|ENTER SHORT|EXIT|HOLD|SKIP)"
)
RF_P_RE = re.compile(r"RF\s*p\s*=\s*(?P<value>[0-9]*\.?[0-9]+)")
PLAN_RE = re.compile(r"Plan\s+qty=(?P<qty>[0-9]*\.?[0-9]+)\s+notional=(?P<notional>[0-9]*\.?[0-9]+)\s+sl=(?P<sl>[0-9]*\.?[0-9]+)\s+tp=(?P<tp>[0-9]*\.?[0-9]+)")


@dataclass(slots=True)
class ParsedSignal:
    ts_epoch: int
    symbol: str
    side: str
    confidence: float
    mode: str
    raw_line: str
    reason: str = ""
    notional_usd: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None


@dataclass(slots=True)
class OpenTrade:
    symbol: str
    side: str
    mode: str
    entry_time: str
    entry_price: float
    notional_usd: float
    tp_usd: float
    sl_usd: float
    breakeven_min_usd: float
    opened_at: str
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    source_line: str = ""
    max_fav_pnl: float = 0.0
    max_adv_pnl: float = 0.0
    confidence: float = 0.0
    entry_reference_price: float = 0.0
    entry_fee_usd: float = 0.0
    estimated_entry_slippage_usd: float = 0.0
    entry_effective_slippage_bps: float = 0.0
    latency_bps: float = 0.0
    spread_bps: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "mode": self.mode,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "notional_usd": self.notional_usd,
            "tp_usd": self.tp_usd,
            "sl_usd": self.sl_usd,
            "breakeven_min_usd": self.breakeven_min_usd,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "opened_at": self.opened_at,
            "max_fav_pnl": self.max_fav_pnl,
            "max_adv_pnl": self.max_adv_pnl,
            "confidence": self.confidence,
            "entry_reference_price": self.entry_reference_price,
            "entry_fee_usd": self.entry_fee_usd,
            "estimated_entry_slippage_usd": self.estimated_entry_slippage_usd,
            "entry_effective_slippage_bps": self.entry_effective_slippage_bps,
            "latency_bps": self.latency_bps,
            "spread_bps": self.spread_bps,
            "source_line": self.source_line,
        }

    @classmethod
    def from_json(cls, symbol: str, payload: dict[str, Any]) -> "OpenTrade":
        return cls(
            symbol=symbol,
            side=str(payload.get("side", "")).upper() or "LONG",
            mode=str(payload.get("mode", "Medium")),
            entry_time=str(payload.get("entry_time", "")),
            entry_price=float(payload.get("entry_price", 0.0) or 0.0),
            notional_usd=float(payload.get("notional_usd", 1000.0) or 1000.0),
            tp_usd=float(payload.get("tp_usd", 9.0) or 9.0),
            sl_usd=float(payload.get("sl_usd", 6.0) or 6.0),
            breakeven_min_usd=float(payload.get("breakeven_min_usd", 1.0) or 1.0),
            stop_loss_price=float(payload.get("stop_loss_price", 0.0) or 0.0) or None,
            take_profit_price=float(payload.get("take_profit_price", 0.0) or 0.0) or None,
            opened_at=str(payload.get("opened_at", utc_now_iso())),
            source_line=str(payload.get("source_line", "")),
            max_fav_pnl=float(payload.get("max_fav_pnl", 0.0) or 0.0),
            max_adv_pnl=float(payload.get("max_adv_pnl", 0.0) or 0.0),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            entry_reference_price=float(payload.get("entry_reference_price", 0.0) or 0.0),
            entry_fee_usd=float(payload.get("entry_fee_usd", 0.0) or 0.0),
            estimated_entry_slippage_usd=float(payload.get("estimated_entry_slippage_usd", 0.0) or 0.0),
            entry_effective_slippage_bps=float(payload.get("entry_effective_slippage_bps", 0.0) or 0.0),
            latency_bps=float(payload.get("latency_bps", 0.0) or 0.0),
            spread_bps=float(payload.get("spread_bps", 0.0) or 0.0),
        )


@dataclass(slots=True)
class ClosedTrade:
    symbol: str
    side: str
    mode: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    notional_usd: float
    exit_reason: str
    gross_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    slippage_usd: float = 0.0
    latency_bps: float = 0.0
    spread_bps: float = 0.0
    entry_reference_price: float = 0.0
    exit_reference_price: float = 0.0

    @property
    def trade_id(self) -> str:
        return f"{self.symbol}|{self.side}|{self.entry_time}|{self.exit_time}|{self.entry_price:.8f}|{self.exit_price:.8f}"

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "mode": self.mode,
            "entry_time": self.entry_time,
            "entry_price": round(self.entry_price, 8),
            "exit_time": self.exit_time,
            "exit_price": round(self.exit_price, 8),
            "pnl_usd": round(self.pnl_usd, 6),
            "pnl_pct": round(self.pnl_pct, 6),
            "notional_usd": round(self.notional_usd, 6),
            "exit_reason": self.exit_reason,
            "gross_pnl_usd": round(self.gross_pnl_usd, 6),
            "fees_usd": round(self.fees_usd, 6),
            "slippage_usd": round(self.slippage_usd, 6),
            "latency_bps": round(self.latency_bps, 6),
            "spread_bps": round(self.spread_bps, 6),
            "entry_reference_price": round(self.entry_reference_price, 8),
            "exit_reference_price": round(self.exit_reference_price, 8),
        }


@dataclass(slots=True)
class BacktestSummary:
    open_count: int
    completed_count: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    latency_bps: float = 0.0


class SignalLogTail:
    def __init__(self, path: Path, start_from_end: bool = True) -> None:
        self.path = path
        self._offset = 0
        self._started = False
        self._start_from_end = start_from_end

    def read_new_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            if not self._started:
                if self._start_from_end:
                    handle.seek(0, 2)
                    self._offset = handle.tell()
                self._started = True
            handle.seek(self._offset)
            lines = handle.readlines()
            self._offset = handle.tell()
            return [line.rstrip("\n") for line in lines if line.strip()]


class MarkPriceClient:
    def __init__(self, base_url: str, paths: Any | None = None) -> None:
        self.url = base_url.rstrip("/") + "/fapi/v1/premiumIndex"
        self.paths = paths
        self.cache: dict[str, tuple[float, float]] = {}
        self.min_refresh_sec = 2.0
        self.stale_after_sec = 180.0

    def _execution_price_max_age_seconds(self) -> float:
        runtime_cfg = getattr(getattr(getattr(self.paths, "config", None), "runtime", None), "execution_price_max_age_seconds", None)
        try:
            value = float(runtime_cfg) if runtime_cfg is not None else self.stale_after_sec
        except Exception:
            value = self.stale_after_sec
        return max(0.5, value)

    def _runtime_mark_price(self, symbol: str) -> tuple[float | None, float | None, bool]:
        if self.paths is None:
            return None, None, True
        status = latest_symbol_price_status(
            self.paths,
            symbol,
            max_age_seconds=self._execution_price_max_age_seconds(),
        )
        if not status.get("price_available"):
            return None, None, True
        try:
            price = float(status.get("price") or 0.0)
        except Exception:
            return None, None, True
        if price <= 0.0:
            return None, None, True
        ts_ms = status.get("price_ts_ms")
        try:
            parsed_ts = float(ts_ms) if ts_ms is not None else time.time()
            ts_seconds = parsed_ts / 1000.0 if parsed_ts > 10_000_000_000 else parsed_ts
        except Exception:
            ts_seconds = time.time()
        stale = not bool(status.get("price_fresh"))
        if not stale:
            self.cache[str(symbol).upper()] = (price, float(ts_seconds or time.time()))
        return price, ts_seconds, stale

    def get_mark_price(self, symbol: str, *, force: bool = False) -> tuple[float | None, float | None, bool]:
        normalized_symbol = str(symbol or "").upper()
        # Prefer the shared runtime market snapshot. This keeps paper/open-position
        # UI and protective checks aligned with the low-latency mark-price stream
        # instead of waiting on a separate HTTP premiumIndex call.
        runtime_price, runtime_ts, runtime_stale = self._runtime_mark_price(normalized_symbol)
        if runtime_price is not None and not runtime_stale:
            return runtime_price, runtime_ts, False

        now = time.time()
        cached = self.cache.get(normalized_symbol)
        if cached and not force and (now - cached[1]) < self.min_refresh_sec:
            return cached[0], cached[1], (now - cached[1]) >= self.stale_after_sec
        try:
            response = requests.get(self.url, params={"symbol": normalized_symbol}, timeout=5)
            response.raise_for_status()
            payload = response.json()
            price = float(payload["markPrice"])
            self.cache[normalized_symbol] = (price, now)
            return price, now, False
        except Exception:
            if cached:
                return cached[0], cached[1], (now - cached[1]) >= self.stale_after_sec
            if runtime_price is not None:
                return runtime_price, runtime_ts, runtime_stale
            return None, None, True


class BacktestEngine:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.signal_log_path = paths.signal_events_log_path
        self.open_trades_path = paths.backtest_open_trades_path
        self.completed_csv_path = paths.live_backtest_trades_path
        self.trade_history_db_path = paths.trade_history_db_path
        self.tail = SignalLogTail(self.signal_log_path)
        self.price_client = MarkPriceClient(paths.config.market.futures_base_url, paths=paths)
        inf = paths.config.inference
        trading_cfg = getattr(paths.config, "trading", None)
        runtime_cfg = getattr(paths.config, "runtime", None)
        self.default_notional = 1000.0
        self.default_tp_usd = 9.0
        self.default_sl_usd = 6.0
        self.default_breakeven = 1.0
        self.enter_threshold = float(inf.consensus_enter_threshold)
        self.flip_threshold = float(inf.consensus_flip_threshold)
        self.fees_bps = float(getattr(trading_cfg, "fees_bps", 3.0) or 0.0)
        self.slippage_bps = float(getattr(trading_cfg, "slippage_bps", 2.0) or 0.0)
        self.spread_bps = float(getattr(trading_cfg, "spread_bps", 0.0) or 0.0)
        self.latency_bps = min(self.slippage_bps, max(0.0, float(getattr(runtime_cfg, "market_poll_seconds", 0.0) or 0.0)) * 0.1)
        self._ensure_trade_history_db()
        self._bootstrap_trade_db_from_csv()
        self.export_completed_trades_csv()

    def _ensure_trade_history_db(self) -> None:
        self.trade_history_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.trade_history_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS closed_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_time TEXT NOT NULL,
                    exit_price REAL NOT NULL,
                    pnl_usd REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    notional_usd REAL NOT NULL,
                    exit_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_closed_trades_exit_time ON closed_trades(exit_time DESC)")
            conn.commit()

    def _bootstrap_trade_db_from_csv(self) -> None:
        with sqlite3.connect(self.trade_history_db_path) as conn:
            existing = conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0]
        if existing:
            return
        rows = self._restore_completed_trades_from_csv()
        if rows:
            self._append_completed_trades_to_db(rows)

    def parse_signal_line(self, line: str) -> ParsedSignal | None:
        csv_parts = [part.strip() for part in line.split(",")]
        if len(csv_parts) == 4 and csv_parts[0].isdigit():
            try:
                ts_epoch = int(csv_parts[0])
                symbol = csv_parts[1].upper()
                side = csv_parts[2].upper()
                confidence = float(csv_parts[3])
                return ParsedSignal(ts_epoch=ts_epoch, symbol=symbol, side=side, confidence=confidence, mode="Medium", raw_line=line)
            except Exception:
                pass

        match = PRETTY_LINE_RE.search(line)
        if not match:
            return None
        ts_epoch = int(datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        symbol = match.group("symbol").upper()
        action = match.group("action").upper()
        mode = match.group("mode").strip().title()
        reason = line.split("Reason:", 1)[1].strip() if "Reason:" in line else ""
        confidence = 0.5
        rf_match = RF_P_RE.search(line)
        if rf_match:
            value = float(rf_match.group("value"))
            confidence = value if "LONG" in action else (1.0 - value if "SHORT" in action else 0.5)
        if action == "ENTER LONG":
            side = "LONG"
        elif action == "ENTER SHORT":
            side = "SHORT"
        else:
            side = "EXIT"
        plan_match = PLAN_RE.search(line)
        notional_usd = None
        stop_loss_price = None
        take_profit_price = None
        if plan_match:
            try:
                notional_usd = float(plan_match.group("notional"))
                stop_loss_price = float(plan_match.group("sl"))
                take_profit_price = float(plan_match.group("tp"))
            except Exception:
                notional_usd = None
                stop_loss_price = None
                take_profit_price = None
        return ParsedSignal(
            ts_epoch=ts_epoch,
            symbol=symbol,
            side=side,
            confidence=confidence,
            mode=mode,
            raw_line=line,
            reason=reason,
            notional_usd=notional_usd,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
        )

    def drain_signal_lines(self) -> list[ParsedSignal]:
        parsed: list[ParsedSignal] = []
        for line in self.tail.read_new_lines():
            hit = self.parse_signal_line(line)
            if hit is not None:
                parsed.append(hit)
        return parsed

    def restore_open_trades(self) -> dict[str, OpenTrade]:
        if not self.open_trades_path.exists():
            return {}
        try:
            payload = json.loads(self.open_trades_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, OpenTrade] = {}
        for symbol, trade_payload in payload.items():
            if isinstance(trade_payload, dict):
                out[str(symbol).upper()] = OpenTrade.from_json(str(symbol).upper(), trade_payload)
        return self._remove_completed_open_trade_conflicts(out)

    def _remove_completed_open_trade_conflicts(
        self,
        open_trades: dict[str, OpenTrade],
        completed: Iterable[ClosedTrade] | None = None,
    ) -> dict[str, OpenTrade]:
        """Remove legacy historical-backtest rows that already have a close.

        backtest_open_trades.json is not the runtime execution authority, but it
        still exists for historical backtest display and legacy tests. This guard
        prevents stale rows from making closed trades appear open on restart.
        """
        completed_rows = list(completed) if completed is not None else self.restore_completed_trades()
        if not open_trades or not completed_rows:
            return dict(open_trades)

        latest_close_by_symbol: dict[str, datetime] = {}
        for trade in completed_rows:
            symbol = str(trade.symbol or "").upper()
            closed_dt = self._parse_wall_time(str(trade.exit_time or ""))
            if not symbol or closed_dt is None:
                continue
            previous = latest_close_by_symbol.get(symbol)
            if previous is None or closed_dt >= previous:
                latest_close_by_symbol[symbol] = closed_dt

        cleaned: dict[str, OpenTrade] = {}
        for symbol, trade in open_trades.items():
            normalized = str(symbol or trade.symbol).upper()
            opened_dt = self._parse_wall_time(str(trade.entry_time or "")) or self._parse_wall_time(str(trade.opened_at or ""))
            latest_close = latest_close_by_symbol.get(normalized)
            if latest_close is not None and opened_dt is not None and latest_close >= opened_dt:
                continue
            cleaned[normalized] = trade
        return cleaned

    def persist_open_trades(self, open_trades: dict[str, OpenTrade]) -> None:
        self.open_trades_path.parent.mkdir(parents=True, exist_ok=True)
        sanitized = self._remove_completed_open_trade_conflicts(open_trades)
        open_trades.clear()
        open_trades.update(sanitized)
        payload = {symbol: trade.to_json() for symbol, trade in sanitized.items()}
        atomic_write_bytes(self.open_trades_path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))

    def _restore_completed_trades_from_csv(self) -> list[ClosedTrade]:
        if not self.completed_csv_path.exists():
            return []
        out: list[ClosedTrade] = []
        try:
            with self.completed_csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        out.append(ClosedTrade(
                            symbol=str(row.get("symbol", "")).upper(),
                            side=str(row.get("side", "")).upper() or "LONG",
                            mode=str(row.get("mode", "Medium")),
                            entry_time=str(row.get("entry_time", "")),
                            entry_price=float(row.get("entry_price", 0.0) or 0.0),
                            exit_time=str(row.get("exit_time", "")),
                            exit_price=float(row.get("exit_price", 0.0) or 0.0),
                            pnl_usd=float(row.get("pnl_usd", 0.0) or 0.0),
                            pnl_pct=float(row.get("pnl_pct", 0.0) or 0.0),
                            notional_usd=float(row.get("notional_usd", 0.0) or 0.0),
                            exit_reason=str(row.get("exit_reason", "")),
                            gross_pnl_usd=float(row.get("gross_pnl_usd", row.get("pnl_usd", 0.0)) or 0.0),
                            fees_usd=float(row.get("fees_usd", 0.0) or 0.0),
                            slippage_usd=float(row.get("slippage_usd", 0.0) or 0.0),
                            latency_bps=float(row.get("latency_bps", 0.0) or 0.0),
                            entry_reference_price=float(row.get("entry_reference_price", row.get("entry_price", 0.0)) or 0.0),
                            exit_reference_price=float(row.get("exit_reference_price", row.get("exit_price", 0.0)) or 0.0),
                        ))
                    except Exception:
                        continue
        except Exception:
            return []
        return out

    def restore_completed_trades(self) -> list[ClosedTrade]:
        out: list[ClosedTrade] = []
        try:
            with sqlite3.connect(self.trade_history_db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT symbol, side, mode, entry_time, entry_price, exit_time, exit_price,
                           pnl_usd, pnl_pct, notional_usd, exit_reason
                    FROM closed_trades
                    ORDER BY exit_time DESC
                    """
                ).fetchall()
            for row in rows:
                out.append(ClosedTrade(*row))
        except Exception:
            out = []
        if out:
            return out
        return self._restore_completed_trades_from_csv()

    def export_completed_trades_csv(self, completed: Iterable[ClosedTrade] | None = None) -> list[ClosedTrade]:
        canonical_rows = list(completed) if completed is not None else self.restore_completed_trades()
        self.completed_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.completed_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "symbol", "side", "mode", "entry_time", "entry_price", "exit_time",
                    "exit_price", "pnl_usd", "pnl_pct", "notional_usd", "exit_reason",
                    "gross_pnl_usd", "fees_usd", "slippage_usd", "latency_bps",
                    "spread_bps", "entry_reference_price", "exit_reference_price",
                ],
            )
            writer.writeheader()
            for row in reversed(canonical_rows):
                writer.writerow(row.to_csv_row())
        return canonical_rows

    def _append_completed_trades_to_db(self, rows: Iterable[ClosedTrade]) -> None:
        rows = list(rows)
        if not rows:
            return
        with sqlite3.connect(self.trade_history_db_path) as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO closed_trades (
                    trade_id, symbol, side, mode, entry_time, entry_price, exit_time, exit_price,
                    pnl_usd, pnl_pct, notional_usd, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.trade_id,
                        row.symbol,
                        row.side,
                        row.mode,
                        row.entry_time,
                        row.entry_price,
                        row.exit_time,
                        row.exit_price,
                        row.pnl_usd,
                        row.pnl_pct,
                        row.notional_usd,
                        row.exit_reason,
                    )
                    for row in rows
                ],
            )
            conn.commit()

    def _fee_usd(self, notional_usd: float) -> float:
        return max(0.0, float(notional_usd) * (self.fees_bps / 10000.0))

    def _effective_slippage_bps(self) -> float:
        return max(0.0, self.spread_bps + self.slippage_bps + self.latency_bps)

    def _estimated_slippage_usd(self, notional_usd: float) -> float:
        return max(0.0, float(notional_usd) * (self._effective_slippage_bps() / 10000.0))

    def _apply_adverse_fill_price(self, side: str, reference_price: float, *, phase: str) -> float:
        if reference_price <= 0.0:
            return 0.0
        bps = self._effective_slippage_bps() / 10000.0
        direction = str(side).upper()
        is_entry = phase == "entry"
        if direction == "LONG":
            return reference_price * (1.0 + bps) if is_entry else reference_price * (1.0 - bps)
        return reference_price * (1.0 - bps) if is_entry else reference_price * (1.0 + bps)

    def on_signal(self, signal: ParsedSignal, open_trades: dict[str, OpenTrade], messages: list[str]) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        ts_str = datetime.fromtimestamp(signal.ts_epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if signal.side in {"LONG", "SHORT"}:
            if signal.symbol in open_trades:
                messages.append(f"{ts_str} DUP {signal.symbol} already open")
                return closed
            price, _, _ = self.price_client.get_mark_price(signal.symbol, force=True)
            if price is None:
                messages.append(f"{ts_str} WARN no mark price for entry {signal.symbol}")
                return closed
            notional_usd = float(signal.notional_usd or self.default_notional)
            fill_price = self._apply_adverse_fill_price(signal.side, float(price), phase="entry")
            open_trades[signal.symbol] = OpenTrade(
                symbol=signal.symbol,
                side=signal.side,
                mode=signal.mode,
                entry_time=ts_str,
                entry_price=float(fill_price),
                notional_usd=notional_usd,
                tp_usd=self.default_tp_usd,
                sl_usd=self.default_sl_usd,
                breakeven_min_usd=self.default_breakeven,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                opened_at=utc_now_iso(),
                source_line=signal.raw_line,
                confidence=signal.confidence,
                entry_reference_price=float(price),
                entry_fee_usd=self._fee_usd(notional_usd),
                estimated_entry_slippage_usd=self._estimated_slippage_usd(notional_usd),
                entry_effective_slippage_bps=self._effective_slippage_bps(),
                latency_bps=self.latency_bps,
                spread_bps=self.spread_bps,
            )
            messages.append(f"{ts_str} ENTRY {signal.symbol} {signal.side} ({signal.mode}) @ {fill_price:.4f} ref={price:.4f}")
            return closed

        trade = open_trades.get(signal.symbol)
        if trade is None:
            return closed
        price, _, _ = self.price_client.get_mark_price(signal.symbol, force=True)
        if price is None:
            messages.append(f"{ts_str} WARN no mark price for exit {signal.symbol}")
            return closed
        gross = self.pnl_usd(trade.side, trade.entry_price, float(price), trade.notional_usd)
        if gross < trade.breakeven_min_usd and gross > -trade.sl_usd:
            messages.append(f"{ts_str} EXIT deferred {signal.symbol} pnl={gross:+.2f}")
            return closed
        open_trades.pop(signal.symbol, None)
        closed_trade = self._close_trade(trade, float(price), ts_str, exit_reason="LOG_EXIT")
        closed.append(closed_trade)
        messages.append(f"{ts_str} CLOSED {signal.symbol} {trade.side} pnl={closed_trade.pnl_usd:+.2f}")
        return closed

    def risk_checks(self, open_trades: dict[str, OpenTrade], messages: list[str]) -> list[ClosedTrade]:
        closed: list[ClosedTrade] = []
        now_dt = datetime.now(timezone.utc)
        for symbol, trade in list(open_trades.items()):
            price, _, _ = self.price_client.get_mark_price(symbol, force=True)
            if price is None:
                continue
            pnl = self.pnl_usd(trade.side, trade.entry_price, float(price), trade.notional_usd)
            trade.max_fav_pnl = max(trade.max_fav_pnl, pnl)
            trade.max_adv_pnl = min(trade.max_adv_pnl, pnl)
            hit_tp = False
            hit_sl = False
            if trade.take_profit_price:
                hit_tp = (trade.side == "LONG" and float(price) >= float(trade.take_profit_price)) or (trade.side == "SHORT" and float(price) <= float(trade.take_profit_price))
            else:
                hit_tp = pnl >= trade.tp_usd
            if trade.stop_loss_price:
                hit_sl = (trade.side == "LONG" and float(price) <= float(trade.stop_loss_price)) or (trade.side == "SHORT" and float(price) >= float(trade.stop_loss_price))
            else:
                hit_sl = pnl <= -trade.sl_usd
            if hit_tp:
                open_trades.pop(symbol, None)
                closed_trade = self._close_trade(trade, float(price), now_dt.strftime("%Y-%m-%d %H:%M:%S"), exit_reason="TP_HIT")
                closed.append(closed_trade)
                messages.append(f"{symbol} TP hit pnl={closed_trade.pnl_usd:+.2f}")
            elif hit_sl:
                open_trades.pop(symbol, None)
                closed_trade = self._close_trade(trade, float(price), now_dt.strftime("%Y-%m-%d %H:%M:%S"), exit_reason="SL_HIT")
                closed.append(closed_trade)
                messages.append(f"{symbol} SL hit pnl={closed_trade.pnl_usd:+.2f}")
        return closed

    def pnl_usd(self, side: str, entry: float, current: float, notional: float) -> float:
        if entry <= 0.0:
            return 0.0
        if side == "LONG":
            return ((current - entry) / entry) * notional
        return ((entry - current) / entry) * notional

    def estimate_open_trade_row(self, trade: OpenTrade) -> dict[str, Any]:
        price, ts, stale = self.price_client.get_mark_price(trade.symbol, force=False)
        pnl = self.pnl_usd(trade.side, trade.entry_price, float(price or trade.entry_price), trade.notional_usd)
        expected_exit_fee = self._fee_usd(trade.notional_usd)
        estimated_net_pnl = pnl - trade.entry_fee_usd - expected_exit_fee
        return {
            "symbol": trade.symbol,
            "side": trade.side,
            "mode": trade.mode,
            "entry_price": trade.entry_price,
            "last_price": price,
            "last_price_ts": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S") if ts else "--:--:--",
            "stale": stale,
            "estimated_pnl": pnl,
            "estimated_net_pnl": estimated_net_pnl,
            "entry_fee_usd": trade.entry_fee_usd,
            "effective_slippage_bps": trade.entry_effective_slippage_bps,
            "latency_bps": trade.latency_bps,
            "spread_bps": trade.spread_bps,
        }

    def _parse_wall_time(self, value: str) -> datetime | None:
        if not value:
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S",):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    def trade_duration_text(self, start_value: str, end_value: str | None = None) -> str:
        start_dt = self._parse_wall_time(start_value)
        end_dt = self._parse_wall_time(end_value) if end_value else datetime.now(timezone.utc)
        if start_dt is None or end_dt is None:
            return "--"
        total = max(0, int((end_dt - start_dt).total_seconds()))
        hours = total // 3600
        minutes = (total % 3600) // 60
        if hours:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def trade_history_row_from_closed(self, trade: ClosedTrade) -> dict[str, Any]:
        return {
            "symbol": trade.symbol,
            "action": trade.side.title(),
            "mode": trade.mode,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "duration": self.trade_duration_text(trade.entry_time, trade.exit_time),
            "notional_usd": trade.notional_usd,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "current_mark_price": None,
            "current_mark_ts": "--",
            "pnl_usd": trade.pnl_usd,
            "pnl_pct": trade.pnl_pct,
            "gross_pnl_usd": trade.gross_pnl_usd,
            "fees_usd": trade.fees_usd,
            "slippage_usd": trade.slippage_usd,
            "exit_reason": trade.exit_reason,
            "status": "Closed",
            "win_loss": "Win" if trade.pnl_usd > 0 else ("Loss" if trade.pnl_usd < 0 else "Flat"),
            "stale": False,
        }

    def summarize(self, open_trades: dict[str, OpenTrade], completed: list[ClosedTrade]) -> BacktestSummary:
        total_pnl = sum(item.pnl_usd for item in completed)
        gross_pnl = sum(item.gross_pnl_usd for item in completed)
        total_fees = sum(item.fees_usd for item in completed)
        total_slippage = sum(item.slippage_usd for item in completed)
        wins = sum(1 for item in completed if item.pnl_usd > 0)
        losses = sum(1 for item in completed if item.pnl_usd < 0)
        count = len(completed)
        return BacktestSummary(
            open_count=len(open_trades),
            completed_count=count,
            wins=wins,
            losses=losses,
            total_pnl=total_pnl,
            win_rate=(wins / count * 100.0) if count else 0.0,
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            total_slippage=total_slippage,
            latency_bps=self.latency_bps,
        )

    def append_completed_trades(self, rows: Iterable[ClosedTrade]) -> None:
        rows = list(rows)
        if not rows:
            return
        self._append_completed_trades_to_db(rows)
        self.export_completed_trades_csv()

    def _close_trade(self, trade: OpenTrade, exit_price: float, exit_time: str, exit_reason: str) -> ClosedTrade:
        exit_reference_price = float(exit_price)
        filled_exit_price = self._apply_adverse_fill_price(trade.side, exit_reference_price, phase="exit")
        gross = self.pnl_usd(trade.side, trade.entry_price, filled_exit_price, trade.notional_usd)
        exit_fee_usd = self._fee_usd(trade.notional_usd)
        fees_usd = trade.entry_fee_usd + exit_fee_usd
        slippage_usd = trade.estimated_entry_slippage_usd + self._estimated_slippage_usd(trade.notional_usd)
        net = gross - fees_usd
        pct = (net / trade.notional_usd) * 100.0 if trade.notional_usd else 0.0
        return ClosedTrade(
            symbol=trade.symbol,
            side=trade.side,
            mode=trade.mode,
            entry_time=trade.entry_time,
            entry_price=trade.entry_price,
            exit_time=exit_time,
            exit_price=filled_exit_price,
            pnl_usd=net,
            pnl_pct=pct,
            notional_usd=trade.notional_usd,
            exit_reason=exit_reason,
            gross_pnl_usd=gross,
            fees_usd=fees_usd,
            slippage_usd=slippage_usd,
            latency_bps=trade.latency_bps,
            spread_bps=trade.spread_bps,
            entry_reference_price=trade.entry_reference_price,
            exit_reference_price=exit_reference_price,
        )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
