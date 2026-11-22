
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SHORT_CLASS = 0
HOLD_CLASS = 1
LONG_CLASS = 2


@dataclass(frozen=True)
class DirectionalClassEncoding:
    short: int = SHORT_CLASS
    hold: int = HOLD_CLASS
    long: int = LONG_CLASS


DEFAULT_ENCODING = DirectionalClassEncoding()


class LabelBuilderError(ValueError):
    pass


def _anchor_close_column(anchor_interval: str) -> str:
    return f"{anchor_interval}_close"


def _anchor_atr_column(anchor_interval: str) -> str:
    return f"{anchor_interval}_atr_pct_14"


def _future_return_series(df: pd.DataFrame, *, anchor_interval: str, horizon: int) -> pd.Series:
    close_col = _anchor_close_column(anchor_interval)
    if close_col not in df.columns:
        raise LabelBuilderError(f"missing anchor close column: {close_col}")
    close = pd.to_numeric(df[close_col], errors="coerce")
    future_close = close.shift(-int(horizon))
    return (future_close - close) / close.replace(0.0, np.nan)


def _dynamic_threshold_series(
    df: pd.DataFrame,
    *,
    anchor_interval: str,
    atr_multiplier: float,
    atr_floor: float,
    override_atr_column: str | None = None,
) -> pd.Series:
    atr_col = override_atr_column or _anchor_atr_column(anchor_interval)
    atr_pct = None
    if atr_col in df.columns:
        atr_pct = pd.to_numeric(df[atr_col], errors="coerce").abs()
    else:
        candidates = [c for c in df.columns if c.endswith("atr_pct_14") or c.endswith("ATR_N") or c.endswith("atr_pct")]
        if candidates:
            atr_pct = pd.to_numeric(df[candidates[0]], errors="coerce").abs()
    if atr_pct is None:
        atr_pct = pd.Series(index=df.index, dtype="float64")
    thr = atr_pct * float(atr_multiplier)
    thr = thr.fillna(float(atr_floor))
    return thr.clip(lower=float(atr_floor))


def directional_labels_from_returns(
    future_return: pd.Series,
    thr_dyn: pd.Series,
    *,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
) -> pd.Series:
    future_return = pd.to_numeric(future_return, errors="coerce")
    thr_dyn = pd.to_numeric(thr_dyn, errors="coerce")
    label = pd.Series(np.nan, index=future_return.index, dtype="float64")
    label.loc[future_return < -thr_dyn] = float(encoding.short)
    label.loc[future_return.between(-thr_dyn, thr_dyn, inclusive="both")] = float(encoding.hold)
    label.loc[future_return > thr_dyn] = float(encoding.long)
    label.loc[future_return.isna() | thr_dyn.isna()] = np.nan
    return label


def compute_cost_rate(
    cfg: Any | None = None,
    *,
    fees_bps: float | None = None,
    slippage_bps: float | None = None,
    spread_bps: float | None = None,
    funding_bps: float | None = None,
) -> float:
    """Return the one-trade cost rate used by labels and EV.

    The rate is intentionally expressed as a return fraction so the same value can
    be subtracted from future-return labels and expected-value payoffs. Existing
    callers that only provide fees/slippage remain compatible; Step 8 extends the
    contract to include spread and optional funding without changing live routing.
    """
    if cfg is not None:
        fees_bps = float(getattr(cfg, "fees_bps", fees_bps or 0.0) or 0.0)
        slippage_bps = float(getattr(cfg, "slippage_bps", slippage_bps or 0.0) or 0.0)
        spread_bps = float(getattr(cfg, "spread_bps", spread_bps or 0.0) or 0.0)
        funding_bps = float(getattr(cfg, "funding_bps", funding_bps or 0.0) or 0.0)
    total_bps = float(fees_bps or 0.0) + float(slippage_bps or 0.0) + float(spread_bps or 0.0) + float(funding_bps or 0.0)
    return max(total_bps, 0.0) / 10000.0


def compute_net_return(
    future_return: pd.Series,
    *,
    direction_pred: pd.Series | None = None,
    cfg: Any | None = None,
    fees_bps: float | None = None,
    slippage_bps: float | None = None,
    spread_bps: float | None = None,
    funding_bps: float | None = None,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
) -> pd.Series:
    ret = pd.to_numeric(future_return, errors="coerce")
    cost_rate = compute_cost_rate(cfg, fees_bps=fees_bps, slippage_bps=slippage_bps, spread_bps=spread_bps, funding_bps=funding_bps)
    if direction_pred is None:
        return ret - cost_rate
    direction = pd.to_numeric(direction_pred, errors="coerce")
    signed = pd.Series(np.nan, index=ret.index, dtype="float64")
    signed.loc[direction == float(encoding.long)] = ret.loc[direction == float(encoding.long)]
    signed.loc[direction == float(encoding.short)] = -ret.loc[direction == float(encoding.short)]
    signed.loc[direction == float(encoding.hold)] = 0.0
    trade_mask = direction.isin([float(encoding.short), float(encoding.long)])
    signed.loc[trade_mask] = signed.loc[trade_mask] - cost_rate
    return signed


def build_directional_labels(
    df: pd.DataFrame,
    *,
    anchor_interval: str,
    horizon: int,
    atr_multiplier: float,
    atr_floor: float,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
    label_column: str = "direction_label",
    trading_cfg: Any | None = None,
) -> pd.DataFrame:
    out = df.copy()
    future_return = _future_return_series(out, anchor_interval=anchor_interval, horizon=int(horizon))
    thr_dyn = _dynamic_threshold_series(
        out,
        anchor_interval=anchor_interval,
        atr_multiplier=float(atr_multiplier),
        atr_floor=float(atr_floor),
    )
    cost_rate = compute_cost_rate(trading_cfg)
    cost_aware_thr = (thr_dyn + cost_rate).astype("float64")
    labels = directional_labels_from_returns(future_return, cost_aware_thr, encoding=encoding)
    out["future_return"] = future_return
    out["thr_dyn"] = thr_dyn
    out["cost_rate"] = cost_rate
    out["cost_aware_thr"] = cost_aware_thr
    out[label_column] = labels
    return out


def build_rf_labels(df: pd.DataFrame, *, anchor_interval: str, horizon: int, atr_multiplier: float, atr_floor: float, encoding: DirectionalClassEncoding = DEFAULT_ENCODING, trading_cfg: Any | None = None) -> pd.DataFrame:
    return build_directional_labels(df, anchor_interval=anchor_interval, horizon=horizon, atr_multiplier=atr_multiplier, atr_floor=atr_floor, encoding=encoding, label_column="rf_label", trading_cfg=trading_cfg)


def build_direction_labels(df: pd.DataFrame, *, anchor_interval: str, horizon: int, atr_multiplier: float, atr_floor: float, encoding: DirectionalClassEncoding = DEFAULT_ENCODING, trading_cfg: Any | None = None) -> pd.DataFrame:
    return build_directional_labels(df, anchor_interval=anchor_interval, horizon=horizon, atr_multiplier=atr_multiplier, atr_floor=atr_floor, encoding=encoding, label_column="direction_label", trading_cfg=trading_cfg)


def build_meta_label_from_direction(
    df: pd.DataFrame,
    *,
    direction_pred_col: str = "direction_pred",
    ret_col: str = "future_return",
    thr_col: str = "thr_dyn",
    trading_cfg: Any | None = None,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
) -> pd.Series:
    if direction_pred_col not in df.columns:
        raise LabelBuilderError(f"missing direction prediction column: {direction_pred_col}")
    if ret_col not in df.columns:
        raise LabelBuilderError(f"missing return column: {ret_col}")
    if thr_col not in df.columns:
        raise LabelBuilderError(f"missing threshold column: {thr_col}")

    direction = pd.to_numeric(df[direction_pred_col], errors="coerce")
    ret = pd.to_numeric(df[ret_col], errors="coerce")
    thr = pd.to_numeric(df[thr_col], errors="coerce").abs()
    min_profit_mult = float(getattr(trading_cfg, "min_profit_atr_mult", 0.0) or 0.0)
    min_profit_thr = thr * min_profit_mult
    net_ret = compute_net_return(ret, direction_pred=direction, cfg=trading_cfg, encoding=encoding)

    meta = pd.Series(np.nan, index=df.index, dtype="float64")
    valid = ~(ret.isna() | thr.isna() | direction.isna())
    meta.loc[valid] = 0.0
    profitable_trade = direction.isin([float(encoding.long), float(encoding.short)]) & (net_ret > min_profit_thr)
    actionable_hold = direction == float(encoding.hold)
    meta.loc[profitable_trade | actionable_hold] = 1.0
    return meta


def build_meta_label(
    df: pd.DataFrame,
    *,
    anchor_interval: str,
    horizon: int,
    atr_multiplier: float,
    atr_floor: float,
    trading_cfg: Any | None = None,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
    direction_pred_col: str = "direction_pred",
) -> pd.DataFrame:
    out = build_direction_labels(df, anchor_interval=anchor_interval, horizon=horizon, atr_multiplier=atr_multiplier, atr_floor=atr_floor, encoding=encoding, trading_cfg=trading_cfg)
    if direction_pred_col not in out.columns:
        out[direction_pred_col] = out["direction_label"]
    out["net_return"] = compute_net_return(out["future_return"], direction_pred=out[direction_pred_col], cfg=trading_cfg, encoding=encoding)
    out["meta_label"] = build_meta_label_from_direction(out, direction_pred_col=direction_pred_col, ret_col="future_return", thr_col="thr_dyn", trading_cfg=trading_cfg, encoding=encoding)
    return out


def add_dqn_labels(
    df: pd.DataFrame,
    *,
    anchor_interval: str,
    horizon: int,
    hold_threshold: float | None = None,
    atr_multiplier: float | None = None,
    atr_floor: float = 0.0,
    encoding: DirectionalClassEncoding = DEFAULT_ENCODING,
) -> pd.DataFrame:
    out = df.copy()
    if atr_multiplier is not None:
        labeled = build_direction_labels(out, anchor_interval=anchor_interval, horizon=horizon, atr_multiplier=atr_multiplier, atr_floor=atr_floor, encoding=encoding)
    else:
        future_return = _future_return_series(out, anchor_interval=anchor_interval, horizon=horizon)
        thr = pd.Series(float(hold_threshold or 0.0), index=out.index, dtype="float64")
        labeled = out.copy()
        labeled["future_return"] = future_return
        labeled["thr_dyn"] = thr
        labeled["direction_label"] = directional_labels_from_returns(future_return, thr, encoding=encoding)

    dqn_action = pd.Series(np.nan, index=labeled.index, dtype="float64")
    dqn_action.loc[labeled["direction_label"] == float(encoding.hold)] = 0.0
    dqn_action.loc[labeled["direction_label"] == float(encoding.long)] = 1.0
    dqn_action.loc[labeled["direction_label"] == float(encoding.short)] = 2.0
    labeled["dqn_action"] = dqn_action
    return labeled


def get_class_distribution(series: pd.Series) -> dict[str, int]:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    counts = clean.value_counts().to_dict()
    return {str(SHORT_CLASS): int(counts.get(SHORT_CLASS, 0)), str(HOLD_CLASS): int(counts.get(HOLD_CLASS, 0)), str(LONG_CLASS): int(counts.get(LONG_CLASS, 0))}


def labels_config_value(labels_cfg: Any, key: str, interval: str | None = None, default: Any = None) -> Any:
    if interval and hasattr(labels_cfg, "interval_overrides"):
        overrides = getattr(labels_cfg, "interval_overrides") or {}
        if isinstance(overrides, dict):
            interval_payload = overrides.get(interval, {})
            if isinstance(interval_payload, dict) and key in interval_payload:
                return interval_payload[key]
    return getattr(labels_cfg, key, default)
