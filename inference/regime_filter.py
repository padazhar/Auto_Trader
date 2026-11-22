from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class RegimeState:
    """Latest inferred market regime exposed under the shadow-only governance boundary."""

    name: str
    severity: str
    allow_entries: bool
    score_scale: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float | None] = field(default_factory=dict)
    decision_influence_allowed: bool = False
    shadow_candidate_allow_entries: bool = True
    shadow_candidate_score_scale: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable telemetry view of the regime state."""
        return {
            "name": self.name,
            "severity": self.severity,
            "mode": "shadow_only",
            "decision_influence_allowed": bool(self.decision_influence_allowed),
            # Runtime-safe values: these fields remain neutral under shadow-only governance.
            "allow_entries": bool(self.allow_entries),
            "score_scale": float(self.score_scale),
            # Candidate telemetry: observable but never applied to active decisions.
            "shadow_candidate_allow_entries": bool(self.shadow_candidate_allow_entries),
            "shadow_candidate_score_scale": float(self.shadow_candidate_score_scale),
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def _float(row: pd.Series, key: str) -> float | None:
    if key not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _shadow_state(
    *,
    name: str,
    severity: str,
    candidate_allow_entries: bool,
    candidate_score_scale: float,
    reasons: list[str],
    signals: dict[str, float | None],
) -> RegimeState:
    """Wrap candidate regime output in a runtime-neutral shadow telemetry contract."""
    return RegimeState(
        name=name,
        severity=severity,
        allow_entries=True,
        score_scale=1.0,
        reasons=list(reasons),
        signals=dict(signals),
        decision_influence_allowed=False,
        shadow_candidate_allow_entries=bool(candidate_allow_entries),
        shadow_candidate_score_scale=max(0.0, min(float(candidate_score_scale), 1.0)),
    )


def evaluate_regime(row: pd.Series, *, interval: str, regime_cfg: Any | None) -> RegimeState:
    """Calculate deterministic regime telemetry without changing active decisions.

    Under the approved governance scope, regime calculations are observable shadow
    telemetry only. Candidate blocking/scaling values are reported separately but
    runtime-safe ``allow_entries`` and ``score_scale`` stay neutral. Missing regime
    inputs also retain a neutral/passive telemetry state for backward compatibility.
    """
    if regime_cfg is None or not bool(getattr(regime_cfg, "enabled", True)):
        return _shadow_state(
            name="disabled",
            severity="none",
            candidate_allow_entries=True,
            candidate_score_scale=1.0,
            reasons=[],
            signals={},
        )

    prefix = f"{interval}_"
    realized_vol_z = _float(row, f"{prefix}realized_vol_z_20")
    bb_width = _float(row, f"{prefix}bb_width_20")
    trend_strength = _float(row, f"{prefix}trend_strength_20")
    funding_abs_z = _float(row, f"{prefix}funding_rate_abs_z")
    oi_abs_z = _float(row, f"{prefix}open_interest_abs_z")
    vwap_dev = abs(_float(row, f"{prefix}vwap_dev_20") or 0.0)
    signals = {
        "realized_vol_z_20": realized_vol_z,
        "bb_width_20": bb_width,
        "trend_strength_20": trend_strength,
        "funding_rate_abs_z": funding_abs_z,
        "open_interest_abs_z": oi_abs_z,
        "vwap_dev_20": vwap_dev,
    }

    reasons: list[str] = []
    candidate_allow_entries = True
    candidate_score_scale = 1.0
    severity = "none"
    name = "neutral"

    if realized_vol_z is not None and realized_vol_z >= float(getattr(regime_cfg, "crisis_realized_vol_z", 3.0)):
        return _shadow_state(
            name="crisis_volatility",
            severity="critical",
            candidate_allow_entries=False,
            candidate_score_scale=0.0,
            reasons=["realized_volatility_crisis"],
            signals=signals,
        )

    if realized_vol_z is not None and realized_vol_z >= float(getattr(regime_cfg, "high_realized_vol_z", 1.75)):
        severity = "elevated"
        name = "high_volatility"
        candidate_score_scale *= float(getattr(regime_cfg, "high_vol_score_scale", 0.7))
        reasons.append("realized_volatility_elevated")

    if (
        funding_abs_z is not None and oi_abs_z is not None
        and funding_abs_z >= float(getattr(regime_cfg, "funding_abs_z_block", 3.0))
        and oi_abs_z >= float(getattr(regime_cfg, "open_interest_abs_z_block", 3.0))
    ):
        severity = "elevated" if severity == "none" else severity
        name = "futures_flow_stress"
        candidate_score_scale *= float(getattr(regime_cfg, "stress_score_scale", 0.6))
        reasons.append("funding_and_oi_stress")

    if (
        trend_strength is not None and bb_width is not None
        and trend_strength <= float(getattr(regime_cfg, "weak_trend_strength", 0.35))
        and bb_width <= float(getattr(regime_cfg, "squeeze_width_threshold", 0.025))
    ):
        if name == "neutral":
            name = "rangebound_squeeze"
        candidate_score_scale *= float(getattr(regime_cfg, "rangebound_score_scale", 0.75))
        severity = "elevated" if severity == "none" else severity
        reasons.append("weak_trend_squeeze")

    return _shadow_state(
        name=name,
        severity=severity,
        candidate_allow_entries=candidate_allow_entries,
        candidate_score_scale=candidate_score_scale,
        reasons=reasons,
        signals=signals,
    )
