from __future__ import annotations

from typing import Any

from execution.portfolio_state import PortfolioState


def summarize_portfolio_exposure(portfolio_state: PortfolioState, correlation_groups: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a portfolio exposure summary suitable for allocation and monitoring."""
    groups = {str(k).upper(): str(v) for k, v in dict(correlation_groups or {}).items()}
    gross = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    correlated_counts: dict[str, int] = {}
    correlation_exposure_usd: dict[str, float] = {}
    symbol_exposures: dict[str, float] = {}
    for symbol, position in getattr(portfolio_state, 'positions', {}).items():
        key = str(symbol).upper()
        notional = abs(float(getattr(position, 'notional_usd', 0.0) or 0.0))
        gross += notional
        symbol_exposures[key] = notional
        side = str(getattr(position, 'side', '') or '').upper()
        if side == 'LONG':
            long_exposure += notional
        elif side == 'SHORT':
            short_exposure += notional
        group = groups.get(key, '')
        if group:
            correlated_counts[group] = correlated_counts.get(group, 0) + 1
            correlation_exposure_usd[group] = correlation_exposure_usd.get(group, 0.0) + notional
    net_exposure = long_exposure - short_exposure
    return {
        'gross_exposure_usd': float(gross),
        'long_exposure_usd': float(long_exposure),
        'short_exposure_usd': float(short_exposure),
        'net_exposure_usd': float(net_exposure),
        'open_positions': int(len(getattr(portfolio_state, 'positions', {}))),
        'correlated_counts': correlated_counts,
        'correlation_exposure_usd': correlation_exposure_usd,
        'symbol_exposures_usd': symbol_exposures,
    }
