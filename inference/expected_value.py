from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def compute_cost_rate(cfg: Any | None = None, *, fees_bps: float | None = None, slippage_bps: float | None = None, spread_bps: float | None = None, funding_bps: float | None = None) -> float:
    """Return cost rate as a return fraction for inference EV.

    Kept local to inference to avoid importing training/feature code into the
    decision path. Values are bps in config and converted to decimal returns.
    """
    if cfg is not None:
        fees_bps = float(getattr(cfg, "fees_bps", fees_bps or 0.0) or 0.0)
        slippage_bps = float(getattr(cfg, "slippage_bps", slippage_bps or 0.0) or 0.0)
        spread_bps = float(getattr(cfg, "spread_bps", spread_bps or 0.0) or 0.0)
        funding_bps = float(getattr(cfg, "funding_bps", funding_bps or 0.0) or 0.0)
    total_bps = float(fees_bps or 0.0) + float(slippage_bps or 0.0) + float(spread_bps or 0.0) + float(funding_bps or 0.0)
    return max(total_bps, 0.0) / 10000.0


@dataclass(frozen=True)
class EVEstimate:
    """Expected-value estimate for a single action."""

    action: str
    value: float
    success_prob: float
    failure_prob: float
    abstain_prob: float
    avg_win: float
    avg_loss: float


@dataclass(frozen=True)
class ActionEVSummary:
    """Directional EV summary across long, hold, and short actions."""

    long: EVEstimate
    hold: EVEstimate
    short: EVEstimate

    @property
    def best_directional(self) -> EVEstimate:
        """Return the better of the two directional actions."""
        return self.long if self.long.value >= self.short.value else self.short

    @property
    def best_overall(self) -> EVEstimate:
        """Return the overall best action including hold.

        A directional action must beat hold on EV and also have enough edge over
        the abstain state to be considered actionable. This prevents a mostly-hold
        distribution from being treated as a trade merely because the directional
        EV is slightly above zero.
        """
        best = self.best_directional
        if best.value <= self.hold.value:
            return self.hold
        minimum_actionable_edge = max(best.avg_loss * 0.10, 1e-9)
        if best.success_prob <= best.abstain_prob and best.value <= minimum_actionable_edge:
            return self.hold
        return best


def clamp_probability(value: float) -> float:
    """Clamp a probability-like value into the inclusive [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))


def estimate_payoff_metrics(metrics: dict[str, Any] | None, *, atr_fraction: float, cost_rate: float = 0.0) -> tuple[float, float]:
    """Estimate cost-aware average win/loss magnitudes.

    Average wins are reduced by configured trading costs and average losses are
    increased by those same costs. This makes inference EV use the same economic
    hurdle as labels without loosening thresholds or risk limits.
    """
    metrics = dict(metrics or {})
    avg_win = metrics.get("avg_win_return")
    avg_loss = metrics.get("avg_loss_return")
    if avg_win is None or float(avg_win) <= 0:
        avg_win = max(abs(float(atr_fraction or 0.0)) * 1.5, 1e-6)
    if avg_loss is None or float(avg_loss) <= 0:
        avg_loss = max(abs(float(atr_fraction or 0.0)), 1e-6)
    cost = max(float(cost_rate or 0.0), 0.0)
    return max(float(avg_win) - cost, 1e-9), max(float(avg_loss) + cost, 1e-9)


def compute_action_ev(
    *,
    action: str,
    p_success: float,
    p_failure: float,
    p_abstain: float,
    avg_win: float,
    avg_loss: float,
) -> EVEstimate:
    """Compute EV for a specific action without treating hold probability as a loss.

    For a directional action, only the opposite directional class is treated as a
    loss. The hold probability is treated as abstention or no-trade probability and
    does not directly contribute to payoff.
    """
    success_prob = clamp_probability(p_success)
    failure_prob = clamp_probability(p_failure)
    abstain_prob = clamp_probability(p_abstain)
    total = success_prob + failure_prob + abstain_prob
    if total > 0:
        success_prob /= total
        failure_prob /= total
        abstain_prob /= total
    avg_win = max(float(avg_win), 1e-9)
    avg_loss = max(float(avg_loss), 1e-9)
    ev = (success_prob * avg_win) - (failure_prob * avg_loss)
    return EVEstimate(
        action=str(action),
        value=float(ev),
        success_prob=success_prob,
        failure_prob=failure_prob,
        abstain_prob=abstain_prob,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def compute_action_evs(*, p_long: float, p_hold: float, p_short: float, avg_win: float, avg_loss: float) -> ActionEVSummary:
    """Compute action-specific EVs for a three-class long/hold/short model."""
    long_ev = compute_action_ev(
        action="LONG",
        p_success=p_long,
        p_failure=p_short,
        p_abstain=p_hold,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
    short_ev = compute_action_ev(
        action="SHORT",
        p_success=p_short,
        p_failure=p_long,
        p_abstain=p_hold,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
    hold_probability = clamp_probability(p_hold)
    hold_ev = EVEstimate(
        action="HOLD",
        value=0.0,
        success_prob=hold_probability,
        failure_prob=0.0,
        abstain_prob=1.0 - hold_probability,
        avg_win=max(float(avg_win), 1e-9),
        avg_loss=max(float(avg_loss), 1e-9),
    )
    return ActionEVSummary(long=long_ev, hold=hold_ev, short=short_ev)


def compute_ev(*, p_long: float, p_short: float, avg_win: float, avg_loss: float, p_hold: float = 0.0) -> EVEstimate:
    """Backward-compatible directional EV helper.

    This keeps existing call sites working while routing them through the explicit
    three-class EV contract. It returns the best directional EV estimate only.
    """
    summary = compute_action_evs(
        p_long=p_long,
        p_hold=p_hold,
        p_short=p_short,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )
    return summary.best_directional


def adjust_ev_with_dqn(ev: float, *, dqn_score: float, strength: float) -> float:
    """Tilt directional EV using the optional confirmatory-policy score.

    The legacy function name is preserved for compatibility. A zero strength
    makes the confirmatory policy observability-only and guarantees no runtime
    influence on trade scoring.
    """
    if float(strength) == 0.0:
        return float(ev)
    return float(ev) * (1.0 + float(dqn_score) * float(strength))
