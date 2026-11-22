from __future__ import annotations

from typing import Any

from execution.order_models import FillEvent, OrderState
from execution.order_store import OrderStore


class FillHandler:
    """Apply fill events to the persistent order lifecycle store."""

    def __init__(self, order_store: OrderStore) -> None:
        self.order_store = order_store

    def apply_exchange_result(self, order_state: OrderState, exchange_result, *, ts: float) -> tuple[OrderState, FillEvent | None, dict[str, Any]]:
        if getattr(exchange_result, "accepted", False):
            updated = self.order_store.update_submission(
                order_state.order_id,
                status=str(getattr(exchange_result, "status", "submitted") or "submitted"),
                ts=float(ts),
                exchange_order_id=str(getattr(exchange_result, "exchange_order_id", "") or ""),
                exchange_status=str(getattr(exchange_result, "exchange_status", "") or ""),
                last_error=str(getattr(exchange_result, "error", "") or ""),
            )
        else:
            updated = self.order_store.reject(order_state.order_id, ts=float(ts), reason=str(getattr(exchange_result, "error", "order_rejected") or "order_rejected"))

        fill = getattr(exchange_result, "fill", None)
        if fill is not None:
            updated = self.order_store.apply_fill(fill)

        report = {
            "order_id": updated.order_id,
            "client_order_id": updated.client_order_id,
            "status": updated.status,
            "exchange_status": updated.exchange_status,
            "exchange_order_id": updated.exchange_order_id,
            "last_error": updated.last_error,
            "filled_quantity": updated.filled_quantity,
            "remaining_quantity": updated.remaining_quantity,
            "avg_fill_price": updated.avg_fill_price,
        }
        return updated, fill, report
