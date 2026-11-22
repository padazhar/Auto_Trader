from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OrderIntent:
    """Strategy-approved intent before exchange routing."""

    symbol: str
    action: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    tif: str = "GTC"
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    client_order_id: str = ""
    signal_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderRequest:
    """Normalized broker/exchange order request."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    tif: str = "GTC"
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FillEvent:
    """Normalized execution fill event."""

    order_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fill_id: str = ""
    fee_usd: float = 0.0
    liquidity: str = ""
    ts: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderState:
    """Persisted order lifecycle state."""

    order_id: str
    client_order_id: str
    symbol: str
    action: str
    side: str
    order_type: str
    tif: str
    quantity: float
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    status: str = "created"
    requested_at: float = 0.0
    updated_at: float = 0.0
    submitted_at: float | None = None
    canceled_at: float | None = None
    rejected_at: float | None = None
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: float = 0.0
    exchange_order_id: str = ""
    exchange_status: str = ""
    signal_reason: str = ""
    last_error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def terminal(self) -> bool:
        return self.status in {"filled", "canceled", "rejected", "expired"}
