from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class PositionPlan:
    """Single-position sizing plan with explicit risk distortion and stop-geometry metadata."""

    quantity: float
    notional_usd: float
    stop_loss: float
    take_profit: float
    risk_amount_usd: float
    stop_distance: float
    intended_notional_usd: float
    actual_risk_amount_usd: float
    risk_distortion_fraction: float
    clamp_applied: bool
    clamp_reason: str
    sl_atr_mult: float = 0.0
    tp_atr_mult: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _cfg_get(source: Any, name: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


@dataclass(frozen=True)
class AdaptiveStopConfig:
    """Policy for MAE/MFE-guided stop geometry.

    The policy is deliberately fail-safe: if insufficient excursion evidence is
    present, the base ATR multipliers are returned unchanged.
    """

    enabled: bool = False
    lookback_trades: int = 20
    min_trades: int = 8
    min_sl_atr_mult: float = 0.8
    max_sl_atr_mult: float = 2.2
    min_tp_atr_mult: float = 1.0
    max_tp_atr_mult: float = 3.2
    mae_safety_mult: float = 1.15
    mfe_capture_mult: float = 0.85
    min_reward_risk_ratio: float = 1.10

    @classmethod
    def from_config(cls, cfg: Any) -> "AdaptiveStopConfig":
        raw = getattr(cfg, "adaptive_stops", None)
        return cls(
            enabled=bool(_cfg_get(raw, "enabled", False)),
            lookback_trades=max(1, int(_safe_float(_cfg_get(raw, "lookback_trades", 20), 20))),
            min_trades=max(1, int(_safe_float(_cfg_get(raw, "min_trades", 8), 8))),
            min_sl_atr_mult=max(0.01, _safe_float(_cfg_get(raw, "min_sl_atr_mult", 0.8), 0.8)),
            max_sl_atr_mult=max(0.01, _safe_float(_cfg_get(raw, "max_sl_atr_mult", 2.2), 2.2)),
            min_tp_atr_mult=max(0.01, _safe_float(_cfg_get(raw, "min_tp_atr_mult", 1.0), 1.0)),
            max_tp_atr_mult=max(0.01, _safe_float(_cfg_get(raw, "max_tp_atr_mult", 3.2), 3.2)),
            mae_safety_mult=max(0.01, _safe_float(_cfg_get(raw, "mae_safety_mult", 1.15), 1.15)),
            mfe_capture_mult=max(0.01, _safe_float(_cfg_get(raw, "mfe_capture_mult", 0.85), 0.85)),
            min_reward_risk_ratio=max(0.01, _safe_float(_cfg_get(raw, "min_reward_risk_ratio", 1.10), 1.10)),
        )


@dataclass(frozen=True)
class AdaptiveStopAdjustment:
    enabled: bool
    applied: bool
    reason: str
    symbol: str
    side: str
    trades_used: int
    base_sl_atr_mult: float
    base_tp_atr_mult: float
    sl_atr_mult: float
    tp_atr_mult: float
    avg_mae_atr_mult: float = 0.0
    avg_mfe_atr_mult: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _excursion_bps(row: Any, names: tuple[str, ...]) -> float:
    attribution = _row_value(row, "attribution", {})
    for source in (row, attribution if isinstance(attribution, dict) else {}):
        for name in names:
            value = _row_value(source, name, None)
            if value is not None:
                return _safe_float(value, 0.0)
    return 0.0


def compute_adaptive_stop_multipliers(
    rows: Iterable[Any],
    *,
    symbol: str,
    side: str,
    atr_fraction: float,
    base_sl_atr_mult: float,
    base_tp_atr_mult: float,
    cfg: AdaptiveStopConfig | None = None,
) -> AdaptiveStopAdjustment:
    stop_cfg = cfg or AdaptiveStopConfig()
    base_sl = float(base_sl_atr_mult)
    base_tp = float(base_tp_atr_mult)
    normalized_symbol = str(symbol or "").upper()
    normalized_side = str(side or "").upper()
    if not stop_cfg.enabled:
        return AdaptiveStopAdjustment(False, False, "disabled", normalized_symbol, normalized_side, 0, base_sl, base_tp, base_sl, base_tp)
    atr = max(_safe_float(atr_fraction, 0.0), 1e-9)
    matches = []
    for row in rows or []:
        if str(_row_value(row, "symbol", "") or "").upper() != normalized_symbol:
            continue
        row_side = str(_row_value(row, "side", "") or "").upper()
        if normalized_side in {"LONG", "SHORT"} and row_side != normalized_side:
            continue
        mae_bps = _excursion_bps(row, ("max_adverse_excursion_bps", "mae_bps"))
        mfe_bps = _excursion_bps(row, ("max_favorable_excursion_bps", "mfe_bps"))
        if mae_bps > 0.0 or mfe_bps > 0.0:
            matches.append(row)
    matches = matches[-stop_cfg.lookback_trades :]
    if len(matches) < stop_cfg.min_trades:
        return AdaptiveStopAdjustment(True, False, "insufficient_mae_mfe_history", normalized_symbol, normalized_side, len(matches), base_sl, base_tp, base_sl, base_tp)

    mae_atr = [(_excursion_bps(row, ("max_adverse_excursion_bps", "mae_bps")) / 10000.0) / atr for row in matches]
    mfe_atr = [(_excursion_bps(row, ("max_favorable_excursion_bps", "mfe_bps")) / 10000.0) / atr for row in matches]
    avg_mae = max(0.0, mean([value for value in mae_atr if value > 0.0] or [0.0]))
    avg_mfe = max(0.0, mean([value for value in mfe_atr if value > 0.0] or [0.0]))
    target_sl = min(max(avg_mae * stop_cfg.mae_safety_mult, stop_cfg.min_sl_atr_mult), stop_cfg.max_sl_atr_mult)
    target_tp = min(max(avg_mfe * stop_cfg.mfe_capture_mult, stop_cfg.min_tp_atr_mult), stop_cfg.max_tp_atr_mult)
    target_tp = min(max(target_tp, target_sl * stop_cfg.min_reward_risk_ratio), stop_cfg.max_tp_atr_mult)
    applied = abs(target_sl - base_sl) > 1e-12 or abs(target_tp - base_tp) > 1e-12
    return AdaptiveStopAdjustment(True, applied, "mae_mfe_adjusted" if applied else "base_geometry_already_aligned", normalized_symbol, normalized_side, len(matches), base_sl, base_tp, target_sl, target_tp, avg_mae, avg_mfe)


def compute_position_plan(
    *,
    balance: float,
    risk_per_trade: float,
    price: float,
    atr_fraction: float,
    sl_atr_mult: float,
    tp_atr_mult: float,
    max_notional_usd: float,
    min_notional_usd: float,
    direction: str,
) -> PositionPlan:
    """Compute a position plan and surface any clamp-driven risk distortion.

    The returned plan explicitly reports the intended risk, actual risk after
    notional clamps, and the fractional distortion between them. Callers should
    reject plans whose distortion exceeds policy tolerance rather than silently
    accepting inflated risk.
    """
    price = max(float(price), 1e-9)
    atr_fraction = max(float(atr_fraction), 1e-9)
    atr_abs = price * atr_fraction
    stop_distance = max(atr_abs * float(sl_atr_mult), price * 1e-6)
    intended_risk_amount = max(float(balance) * float(risk_per_trade), 0.0)

    intended_quantity = intended_risk_amount / stop_distance if stop_distance > 0 else 0.0
    intended_notional = intended_quantity * price
    requested_notional = intended_notional

    clamp_applied = False
    clamp_reason = "none"
    if requested_notional < float(min_notional_usd):
        requested_notional = float(min_notional_usd)
        clamp_applied = True
        clamp_reason = "min_notional"
    if requested_notional > float(max_notional_usd):
        requested_notional = float(max_notional_usd)
        clamp_applied = True
        clamp_reason = "max_notional"

    quantity = requested_notional / price if price > 0 else 0.0
    actual_risk_amount = quantity * stop_distance
    distortion = 0.0
    if intended_risk_amount > 0:
        distortion = abs(actual_risk_amount - intended_risk_amount) / intended_risk_amount

    tp_distance = atr_abs * float(tp_atr_mult)
    side = str(direction).upper()
    if side == "LONG":
        stop_loss = price - stop_distance
        take_profit = price + tp_distance
    else:
        stop_loss = price + stop_distance
        take_profit = price - tp_distance

    return PositionPlan(
        quantity=float(quantity),
        notional_usd=float(requested_notional),
        stop_loss=float(stop_loss),
        take_profit=float(take_profit),
        risk_amount_usd=float(intended_risk_amount),
        stop_distance=float(stop_distance),
        intended_notional_usd=float(intended_notional),
        actual_risk_amount_usd=float(actual_risk_amount),
        risk_distortion_fraction=float(distortion),
        clamp_applied=bool(clamp_applied),
        clamp_reason=clamp_reason,
        sl_atr_mult=float(sl_atr_mult),
        tp_atr_mult=float(tp_atr_mult),
    )


def scale_position_plan(plan: PositionPlan, *, target_notional_usd: float) -> PositionPlan:
    """Return a resized plan that preserves stop geometry while updating notional and quantity."""
    target = max(0.0, float(target_notional_usd))
    original = max(float(plan.notional_usd), 1e-9)
    scale = target / original
    return PositionPlan(
        quantity=float(plan.quantity) * scale,
        notional_usd=float(target),
        stop_loss=float(plan.stop_loss),
        take_profit=float(plan.take_profit),
        risk_amount_usd=float(plan.risk_amount_usd),
        stop_distance=float(plan.stop_distance),
        intended_notional_usd=float(plan.intended_notional_usd),
        actual_risk_amount_usd=float(plan.actual_risk_amount_usd) * scale,
        risk_distortion_fraction=float(plan.risk_distortion_fraction),
        clamp_applied=bool(plan.clamp_applied or abs(scale - 1.0) > 1e-12),
        clamp_reason=('portfolio_allocation' if abs(scale - 1.0) > 1e-12 else plan.clamp_reason),
        sl_atr_mult=float(plan.sl_atr_mult),
        tp_atr_mult=float(plan.tp_atr_mult),
    )
