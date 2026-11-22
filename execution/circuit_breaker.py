from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from core.shared_utils import atomic_write_bytes, load_json


@dataclass(slots=True)
class CircuitBreakerState:
    """Persistent circuit-breaker state that survives process restarts."""

    triggered: bool = False
    reason: str = ""
    triggered_at: float | None = None
    last_checked_at: float | None = None
    peak_equity_usd: float = 0.0
    current_equity_usd: float = 0.0
    drawdown_fraction: float = 0.0
    daily_realized_pnl_usd: float = 0.0
    day_key: str = ""


class CircuitBreakerStore:
    """Persistence helper for the runtime circuit-breaker state."""

    def __init__(self, path) -> None:
        self.path = path

    def load(self) -> CircuitBreakerState:
        payload = load_json(self.path, default={}) or {}
        if not isinstance(payload, dict):
            payload = {}
        return CircuitBreakerState(
            triggered=bool(payload.get("triggered", False)),
            reason=str(payload.get("reason") or ""),
            triggered_at=payload.get("triggered_at"),
            last_checked_at=payload.get("last_checked_at"),
            peak_equity_usd=float(payload.get("peak_equity_usd") or 0.0),
            current_equity_usd=float(payload.get("current_equity_usd") or 0.0),
            drawdown_fraction=float(payload.get("drawdown_fraction") or 0.0),
            daily_realized_pnl_usd=float(payload.get("daily_realized_pnl_usd") or 0.0),
            day_key=str(payload.get("day_key") or ""),
        )

    def save(self, state: CircuitBreakerState) -> None:
        atomic_write_bytes(self.path, json.dumps(asdict(state), indent=2, sort_keys=True).encode("utf-8"))
