from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from execution.position_sizing import PositionPlan, scale_position_plan


@dataclass(slots=True)
class AllocationDecision:
    approved: bool
    reason: str
    allocation_fraction: float
    target_notional_usd: float
    portfolio_budget_usd: float
    side_budget_usd: float
    correlation_penalty: float
    components: dict[str, Any]


def allocate_position_plan(
    *,
    decision,
    plan: PositionPlan,
    exposure: dict[str, Any],
    cfg,
    existing_notional_usd: float,
    existing_same_side_notional_usd: float,
    correlation_group: str,
) -> AllocationDecision:
    """Resize a single-position plan to fit portfolio-level budgets."""
    score = abs(float(getattr(decision, 'score', 1.0) or 1.0))
    confidence_fraction = min(max(score, 0.0), 1.0)
    confidence_multiplier = 0.40 + (0.60 * confidence_fraction)

    gross_headroom = max(0.0, float(cfg.max_total_exposure_usd) - max(0.0, float(exposure.get('gross_exposure_usd', 0.0)) - float(existing_notional_usd)))
    side_key = 'long_exposure_usd' if 'LONG' in str(getattr(decision, 'action', '')).upper() else 'short_exposure_usd'
    side_headroom = max(0.0, float(cfg.max_same_direction_exposure_usd) - max(0.0, float(exposure.get(side_key, 0.0)) - float(existing_same_side_notional_usd)))
    symbol_budget = max(0.0, float(cfg.max_symbol_notional_usd))

    correlation_penalty = 1.0
    if correlation_group:
        existing_group_count = int((exposure.get('correlated_counts') or {}).get(correlation_group, 0))
        existing_group_exposure = float((exposure.get('correlation_exposure_usd') or {}).get(correlation_group, 0.0) or 0.0)
        if existing_group_count > 0:
            correlation_penalty *= 0.75
        if existing_group_exposure > 0.0:
            correlation_penalty *= 0.90

    target_notional = float(plan.notional_usd) * confidence_multiplier * correlation_penalty
    target_notional = min(target_notional, gross_headroom, side_headroom, symbol_budget)

    if target_notional <= 0.0:
        return AllocationDecision(
            approved=False,
            reason='portfolio_budget_exhausted',
            allocation_fraction=0.0,
            target_notional_usd=0.0,
            portfolio_budget_usd=float(gross_headroom),
            side_budget_usd=float(side_headroom),
            correlation_penalty=float(correlation_penalty),
            components={'confidence_multiplier': confidence_multiplier},
        )

    if target_notional < float(cfg.min_notional_usd):
        return AllocationDecision(
            approved=False,
            reason='portfolio_headroom_below_min_notional',
            allocation_fraction=0.0,
            target_notional_usd=float(target_notional),
            portfolio_budget_usd=float(gross_headroom),
            side_budget_usd=float(side_headroom),
            correlation_penalty=float(correlation_penalty),
            components={'confidence_multiplier': confidence_multiplier},
        )

    allocation_fraction = min(1.0, max(0.0, float(target_notional) / max(float(plan.notional_usd), 1e-9)))
    scaled_plan = scale_position_plan(plan, target_notional_usd=float(target_notional))
    return AllocationDecision(
        approved=True,
        reason='portfolio_allocated',
        allocation_fraction=float(allocation_fraction),
        target_notional_usd=float(target_notional),
        portfolio_budget_usd=float(gross_headroom),
        side_budget_usd=float(side_headroom),
        correlation_penalty=float(correlation_penalty),
        components={
            'confidence_multiplier': float(confidence_multiplier),
            'original_position_plan': asdict(plan),
            'allocated_position_plan': asdict(scaled_plan),
        },
    )
