from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from core.model_naming import canonical_model_type

ModelType = Literal["rf", "gbm", "dqn", "hmm"]


@dataclass(frozen=True)
class TrainingJob:
    """Canonical training job definition used by schedulers and controllers."""

    model_type: ModelType
    symbol: str
    interval: str
    priority: int
    requested_at: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize model aliases while keeping the frozen dataclass contract."""
        object.__setattr__(self, "model_type", canonical_model_type(self.model_type))

    @property
    def key(self) -> tuple[str, str, str]:
        """Return a stable job identity tuple."""
        return (self.model_type, self.symbol, self.interval)
