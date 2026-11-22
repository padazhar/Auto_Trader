from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable


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


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


@dataclass(slots=True, frozen=True)
class SymbolExpectancyGuardConfig:
    enabled: bool = True
    lookback_trades: int = 10
    min_trades: int = 5
    min_win_rate: float = 0.30
    min_expectancy_usd: float = -2.0
    max_recent_loss_usd: float = -50.0
    cooldown_minutes: float = 240.0

    @classmethod
    def from_config(cls, cfg: Any) -> "SymbolExpectancyGuardConfig":
        raw = getattr(cfg, "symbol_expectancy_guard", None)
        if isinstance(raw, dict):
            source = raw
            getter = lambda key, default: source.get(key, default)
        else:
            getter = lambda key, default: getattr(cfg, f"symbol_expectancy_guard_{key}", default)
        return cls(
            enabled=bool(getter("enabled", True)),
            lookback_trades=max(1, _safe_int(getter("lookback_trades", 10), 10)),
            min_trades=max(1, _safe_int(getter("min_trades", 5), 5)),
            min_win_rate=max(0.0, min(1.0, _safe_float(getter("min_win_rate", 0.30), 0.30))),
            min_expectancy_usd=_safe_float(getter("min_expectancy_usd", -2.0), -2.0),
            max_recent_loss_usd=_safe_float(getter("max_recent_loss_usd", -50.0), -50.0),
            cooldown_minutes=max(0.0, _safe_float(getter("cooldown_minutes", 240.0), 240.0)),
        )


@dataclass(slots=True, frozen=True)
class SymbolExpectancySnapshot:
    symbol: str
    side: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    realized_pnl_usd: float
    expectancy_usd: float
    latest_closed_at: float
    enabled: bool
    quarantined: bool
    reason: str
    thresholds: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _closed_trade_rows_for_symbol(rows: Iterable[Any], symbol: str, *, side: str = "") -> list[Any]:
    target = str(symbol or "").upper()
    target_side = str(side or "").upper()
    result: list[Any] = []
    for row in rows or []:
        row_symbol = str(_row_value(row, "symbol", "") or "").upper()
        if row_symbol != target:
            continue
        row_side = str(_row_value(row, "side", "") or "").upper()
        if target_side in {"LONG", "SHORT"} and row_side != target_side:
            continue
        result.append(row)
    result.sort(key=lambda item: _safe_float(_row_value(item, "closed_at", _row_value(item, "updated_at", 0.0)), 0.0))
    return result


def build_symbol_expectancy_snapshot(
    rows: Iterable[Any],
    *,
    symbol: str,
    side: str = "",
    ts: float | None = None,
    cfg: SymbolExpectancyGuardConfig | None = None,
) -> SymbolExpectancySnapshot:
    guard_cfg = cfg or SymbolExpectancyGuardConfig()
    normalized_symbol = str(symbol or "").upper()
    normalized_side = str(side or "").upper()
    symbol_rows = _closed_trade_rows_for_symbol(rows, normalized_symbol, side=normalized_side)
    recent = symbol_rows[-guard_cfg.lookback_trades :]
    pnls = [_safe_float(_row_value(row, "pnl_usd", _row_value(row, "realized_pnl_usd", 0.0)), 0.0) for row in recent]
    trades = len(pnls)
    wins = sum(1 for pnl in pnls if pnl > 0.0)
    losses = sum(1 for pnl in pnls if pnl < 0.0)
    total_pnl = float(sum(pnls))
    win_rate = float(wins) / float(trades) if trades else 0.0
    expectancy = total_pnl / float(trades) if trades else 0.0
    latest_closed_at = _safe_float(_row_value(recent[-1], "closed_at", _row_value(recent[-1], "updated_at", 0.0)), 0.0) if recent else 0.0
    thresholds = {
        "lookback_trades": guard_cfg.lookback_trades,
        "min_trades": guard_cfg.min_trades,
        "min_win_rate": guard_cfg.min_win_rate,
        "min_expectancy_usd": guard_cfg.min_expectancy_usd,
        "max_recent_loss_usd": guard_cfg.max_recent_loss_usd,
        "cooldown_minutes": guard_cfg.cooldown_minutes,
    }
    reason = "insufficient_history"
    quarantined = False
    if not guard_cfg.enabled:
        reason = "disabled"
    elif trades >= guard_cfg.min_trades:
        failures: list[str] = []
        if win_rate < guard_cfg.min_win_rate:
            failures.append("win_rate_below_threshold")
        if expectancy < guard_cfg.min_expectancy_usd:
            failures.append("expectancy_below_threshold")
        if total_pnl <= guard_cfg.max_recent_loss_usd:
            failures.append("recent_loss_limit")
        if failures:
            in_cooldown = True
            if guard_cfg.cooldown_minutes > 0 and ts is not None and latest_closed_at > 0:
                in_cooldown = float(ts) <= latest_closed_at + guard_cfg.cooldown_minutes * 60.0
            if in_cooldown:
                quarantined = True
                reason = ";".join(failures)
            else:
                reason = "cooldown_expired_after_" + ";".join(failures)
        else:
            reason = "healthy"
    return SymbolExpectancySnapshot(
        symbol=normalized_symbol,
        side=normalized_side,
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        realized_pnl_usd=total_pnl,
        expectancy_usd=expectancy,
        latest_closed_at=latest_closed_at,
        enabled=guard_cfg.enabled,
        quarantined=quarantined,
        reason=reason,
        thresholds=thresholds,
    )


def summarize_symbol_expectancy(rows: Iterable[Any], *, cfg: SymbolExpectancyGuardConfig | None = None, limit: int = 50) -> dict[str, Any]:
    symbols = sorted({str(_row_value(row, "symbol", "") or "").upper() for row in rows or [] if str(_row_value(row, "symbol", "") or "").strip()})
    snapshots = [build_symbol_expectancy_snapshot(rows, symbol=symbol, cfg=cfg).to_dict() for symbol in symbols]
    quarantined = [row for row in snapshots if row.get("quarantined")]
    return {
        "schema_version": "symbol_expectancy_guard.v1",
        "symbols": snapshots[: max(1, int(limit))],
        "quarantined_symbols": [row.get("symbol") for row in quarantined],
        "quarantined_count": len(quarantined),
    }
