from __future__ import annotations

from typing import Any

from core.shared_utils import now_utc_ts
from execution.exchange_adapter import ExchangeAdapter
from execution.fill_handler import FillHandler
from execution.order_models import OrderIntent, OrderState
from execution.order_store import OrderStore


class OrderRouter:
    """Translate approved intents into persisted exchange-routing attempts."""

    def __init__(self, order_store: OrderStore, exchange_adapter: ExchangeAdapter, fill_handler: FillHandler) -> None:
        self.order_store = order_store
        self.exchange_adapter = exchange_adapter
        self.fill_handler = fill_handler

    @staticmethod
    def _report_from_state(
        state: OrderState,
        *,
        accepted: bool,
        duplicate: bool = False,
        recovery_source: str = "",
        recovery_authoritative: bool | None = None,
    ) -> dict[str, Any]:
        fill = None
        if float(state.filled_quantity) > 0.0 and float(state.avg_fill_price) > 0.0:
            fill = {
                "order_id": state.order_id,
                "client_order_id": state.client_order_id,
                "symbol": state.symbol,
                "side": state.side,
                "quantity": state.filled_quantity,
                "price": state.avg_fill_price,
                "fill_id": state.exchange_order_id or state.order_id,
                "fee_usd": 0.0,
                "liquidity": "recovered" if duplicate else "unknown",
                "ts": state.updated_at,
                "metadata": {},
            }
        return {
            "order_id": state.order_id,
            "client_order_id": state.client_order_id,
            "status": state.status,
            "exchange_status": state.exchange_status,
            "exchange_order_id": state.exchange_order_id,
            "last_error": state.last_error,
            "filled_quantity": state.filled_quantity,
            "remaining_quantity": state.remaining_quantity,
            "avg_fill_price": state.avg_fill_price,
            "accepted": accepted,
            "fill": fill,
            "duplicate": duplicate,
            "recovery_source": str(recovery_source or state.metadata.get("last_recovery_source", "") if isinstance(state.metadata, dict) else ""),
            "recovery_authoritative": bool(recovery_authoritative) if recovery_authoritative is not None else bool((state.metadata or {}).get("last_recovery_authoritative", False)),
        }

    def route_intent(self, intent: OrderIntent, *, ts: float | None = None, previous_position: int = 0) -> dict[str, Any]:
        resolved_ts = float(ts or now_utc_ts())
        client_order_id = self.order_store.build_client_order_id(intent, previous_position=previous_position)
        existing = self.order_store.get_by_client_order_id(client_order_id)
        if existing is not None:
            return self._report_from_state(existing, accepted=(existing.status != "rejected"), duplicate=True)
        order_state = self.order_store.create_from_intent(intent, ts=resolved_ts, previous_position=previous_position)
        request = self.order_store.to_request(order_state)
        exchange_result = self.exchange_adapter.place_order(request, ts=resolved_ts)
        updated_state, fill, report = self.fill_handler.apply_exchange_result(order_state, exchange_result, ts=resolved_ts)
        report["accepted"] = bool(getattr(exchange_result, "accepted", False))
        report["fill"] = None if fill is None else {
            "order_id": fill.order_id,
            "client_order_id": fill.client_order_id,
            "symbol": fill.symbol,
            "side": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "fill_id": fill.fill_id,
            "fee_usd": fill.fee_usd,
            "liquidity": fill.liquidity,
            "ts": fill.ts,
            "metadata": dict(fill.metadata or {}),
        }
        report["duplicate"] = False
        report["order_state"] = updated_state.to_dict()
        return report

    def recover_open_orders(self, *, ts: float | None = None) -> dict[str, Any]:
        resolved_ts = float(ts or now_utc_ts())
        open_exchange_orders = self.exchange_adapter.get_open_orders()
        by_client_order_id = {
            str(row.get("client_order_id") or row.get("clientOrderId") or "").strip(): row
            for row in open_exchange_orders
            if isinstance(row, dict)
        }
        recovered = []
        for state in self.order_store.list_recoverable_orders():
            match = by_client_order_id.get(state.client_order_id)
            if match is None:
                if not bool(getattr(self.exchange_adapter, "recovery_requires_exchange_authority", False)):
                    recovery_source = "paper_local_ledger"
                    recovery_authoritative = True
                    synced = self.order_store.sync_exchange_state(
                        state.order_id,
                        ts=resolved_ts,
                        exchange_status="NEW" if state.status in {"created", "submitted"} else "PARTIALLY_FILLED",
                        exchange_order_id=state.exchange_order_id or state.order_id,
                        last_error=state.last_error,
                        recovery_source=recovery_source,
                        recovery_authoritative=recovery_authoritative,
                    )
                else:
                    recovery_source = "exchange_absent_terminalized"
                    recovery_authoritative = False
                    synced = self.order_store.sync_exchange_state(
                        state.order_id,
                        ts=resolved_ts,
                        exchange_status="EXPIRED" if state.status == "created" else "CANCELED",
                        exchange_order_id=state.exchange_order_id,
                        last_error=state.last_error,
                        recovery_source=recovery_source,
                        recovery_authoritative=recovery_authoritative,
                    )
            else:
                recovery_source = "exchange_open_orders"
                recovery_authoritative = True
                synced = self.order_store.sync_exchange_state(
                    state.order_id,
                    ts=resolved_ts,
                    exchange_status=str(match.get("status") or match.get("exchange_status") or state.exchange_status or "NEW"),
                    exchange_order_id=str(match.get("exchange_order_id") or match.get("orderId") or state.exchange_order_id or ""),
                    last_error=str(match.get("error") or state.last_error or ""),
                    recovery_source=recovery_source,
                    recovery_authoritative=recovery_authoritative,
                )
            recovered.append(self._report_from_state(synced, accepted=(synced.status != "rejected"), duplicate=True, recovery_source=recovery_source, recovery_authoritative=recovery_authoritative))
        return {
            "ts": resolved_ts,
            "recovered_orders": recovered,
            "recovery_sources": {
                "exchange_open_orders": sum(1 for row in recovered if row.get("recovery_source") == "exchange_open_orders"),
                "paper_local_ledger": sum(1 for row in recovered if row.get("recovery_source") == "paper_local_ledger"),
                "exchange_absent_terminalized": sum(1 for row in recovered if row.get("recovery_source") == "exchange_absent_terminalized"),
            },
            "open_order_count": len(self.order_store.list_open_orders()),
        }
