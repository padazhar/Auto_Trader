from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.shared_utils import atomic_write_bytes, load_json, now_utc_ts, stable_hash_parts
from execution.order_models import FillEvent, OrderIntent, OrderRequest, OrderState


class OrderStore:
    """Persist and mutate order lifecycle state and journal events."""

    def __init__(self, state_path: Path, journal_path: Path) -> None:
        self.state_path = Path(state_path)
        self.journal_path = Path(journal_path)

    def _load_payload(self) -> dict[str, Any]:
        payload = load_json(self.state_path, default={}) or {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("state_domain", "execution_orders")
        payload.setdefault("state_owner", "live_trading_controller")
        payload.setdefault("orders", {})
        payload.setdefault("client_order_index", {})
        payload.setdefault("updated_at", 0.0)
        return payload

    def _save_payload(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload["state_domain"] = "execution_orders"
        payload["state_owner"] = "live_trading_controller"
        payload["updated_at"] = float(now_utc_ts())
        atomic_write_bytes(self.state_path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))

    def _append_journal(self, row: dict[str, Any]) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            enriched = {"state_domain": "execution_orders", "state_owner": "live_trading_controller", **dict(row)}
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")

    @staticmethod
    def build_order_id(*, symbol: str, action: str, ts: float, quantity: float, previous_position: int = 0) -> str:
        digest = stable_hash_parts([
            str(symbol).upper(),
            str(action).upper(),
            f"{float(ts):.6f}",
            f"{float(quantity):.8f}",
            int(previous_position),
        ])
        return f"ord_{digest[:20]}"

    @staticmethod
    def build_client_order_id(intent: OrderIntent, *, previous_position: int = 0) -> str:
        if str(intent.client_order_id or "").strip():
            return str(intent.client_order_id).strip()
        metadata = dict(intent.metadata or {})
        idempotency_key = str(metadata.get("idempotency_key") or "").strip()
        if idempotency_key:
            digest = stable_hash_parts([
                str(intent.symbol).upper(),
                str(intent.action).upper(),
                str(intent.side).upper(),
                f"{float(intent.quantity):.8f}",
                idempotency_key,
                int(previous_position),
            ])
        else:
            digest = stable_hash_parts([
                str(intent.symbol).upper(),
                str(intent.action).upper(),
                str(intent.side).upper(),
                f"{float(intent.quantity):.8f}",
                str(intent.order_type or "MARKET").upper(),
                f"{float(intent.limit_price):.8f}" if intent.limit_price is not None else "",
                f"{float(intent.stop_price):.8f}" if intent.stop_price is not None else "",
                bool(intent.reduce_only),
                int(previous_position),
            ])
        return f"cli_{digest[:28]}"

    def get_by_client_order_id(self, client_order_id: str) -> OrderState | None:
        lookup = str(client_order_id or "").strip()
        if not lookup:
            return None
        payload = self._load_payload()
        order_id = payload.get("client_order_index", {}).get(lookup)
        if not isinstance(order_id, str) or not order_id:
            return None
        return self.get(order_id)

    def create_from_intent(self, intent: OrderIntent, *, ts: float, previous_position: int = 0) -> OrderState:
        client_order_id = self.build_client_order_id(intent, previous_position=previous_position)
        existing = self.get_by_client_order_id(client_order_id)
        if existing is not None:
            return existing
        order_id = intent.client_order_id or self.build_order_id(
            symbol=intent.symbol,
            action=intent.action,
            ts=ts,
            quantity=intent.quantity,
            previous_position=previous_position,
        )
        lineage = dict(intent.lineage or {})
        if not lineage and isinstance(intent.metadata, dict):
            lineage = dict(intent.metadata.get("lineage") or {})
        state = OrderState(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=str(intent.symbol).upper(),
            action=str(intent.action).upper(),
            side=str(intent.side).upper(),
            order_type=str(intent.order_type or "MARKET").upper(),
            tif=str(intent.tif or "GTC").upper(),
            quantity=float(intent.quantity),
            limit_price=(float(intent.limit_price) if intent.limit_price is not None else None),
            stop_price=(float(intent.stop_price) if intent.stop_price is not None else None),
            reduce_only=bool(intent.reduce_only),
            status="created",
            requested_at=float(ts),
            updated_at=float(ts),
            remaining_quantity=float(intent.quantity),
            signal_reason=str(intent.signal_reason or ""),
            metadata=dict(intent.metadata or {}),
            lineage=lineage,
        )
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        self._append_journal({"event": "created", "ts": float(ts), "order_id": state.order_id, "client_order_id": state.client_order_id, "lineage": dict(state.lineage or {}), "order": state.to_dict()})
        return state

    def get(self, order_id: str) -> OrderState | None:
        raw = self._load_payload().get("orders", {}).get(str(order_id))
        if not isinstance(raw, dict):
            return None
        return OrderState(**raw)

    def list_orders(self) -> list[OrderState]:
        rows = self._load_payload().get("orders", {})
        if not isinstance(rows, dict):
            return []
        states: list[OrderState] = []
        for raw in rows.values():
            if isinstance(raw, dict):
                states.append(OrderState(**raw))
        states.sort(key=lambda row: (row.requested_at, row.order_id))
        return states

    def list_open_orders(self) -> list[OrderState]:
        return [row for row in self.list_orders() if not row.terminal]

    def list_recoverable_orders(self) -> list[OrderState]:
        return [row for row in self.list_orders() if row.status in {"created", "submitted", "partially_filled"}]

    def update_submission(
        self,
        order_id: str,
        *,
        status: str,
        ts: float,
        exchange_order_id: str = "",
        exchange_status: str = "",
        last_error: str = "",
    ) -> OrderState:
        state = self.get(order_id)
        if state is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        state.status = str(status)
        state.updated_at = float(ts)
        state.submitted_at = float(ts)
        state.exchange_order_id = str(exchange_order_id or state.exchange_order_id or "")
        state.exchange_status = str(exchange_status or state.exchange_status or "")
        state.last_error = str(last_error or "")
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        self._append_journal({
            "event": "submitted",
            "ts": float(ts),
            "order_id": state.order_id,
            "status": state.status,
            "exchange_order_id": state.exchange_order_id,
            "exchange_status": state.exchange_status,
            "last_error": state.last_error,
            "lineage": dict(state.lineage or {}),
        })
        return state

    def sync_exchange_state(
        self,
        order_id: str,
        *,
        ts: float,
        exchange_status: str = "",
        exchange_order_id: str = "",
        last_error: str = "",
        recovery_source: str = "",
        recovery_authoritative: bool | None = None,
    ) -> OrderState:
        state = self.get(order_id)
        if state is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        normalized_exchange = str(exchange_status or state.exchange_status or "").upper()
        normalized_status = {
            "NEW": "submitted",
            "PARTIALLY_FILLED": "partially_filled",
            "FILLED": "filled",
            "CANCELED": "canceled",
            "CANCELLED": "canceled",
            "REJECTED": "rejected",
            "EXPIRED": "expired",
        }.get(normalized_exchange, state.status)
        state.status = normalized_status
        state.updated_at = float(ts)
        state.exchange_status = normalized_exchange or state.exchange_status
        state.exchange_order_id = str(exchange_order_id or state.exchange_order_id or "")
        if last_error:
            state.last_error = str(last_error)
        if recovery_source:
            state.metadata = dict(state.metadata or {})
            state.metadata["last_recovery_source"] = str(recovery_source)
            state.metadata["last_recovery_ts"] = float(ts)
            if recovery_authoritative is not None:
                state.metadata["last_recovery_authoritative"] = bool(recovery_authoritative)
        if normalized_status in {"canceled", "rejected", "expired"}:
            state.remaining_quantity = max(0.0, float(state.quantity) - float(state.filled_quantity))
        if normalized_status == "filled":
            state.remaining_quantity = 0.0
            state.filled_quantity = float(state.quantity)
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        self._append_journal({
            "event": "exchange_sync",
            "ts": float(ts),
            "order_id": state.order_id,
            "status": state.status,
            "exchange_status": state.exchange_status,
            "exchange_order_id": state.exchange_order_id,
            "last_error": state.last_error,
            "recovery_source": str(recovery_source or ""),
            "recovery_authoritative": None if recovery_authoritative is None else bool(recovery_authoritative),
            "lineage": dict(state.lineage or {}),
        })
        return state

    def apply_fill(self, fill: FillEvent) -> OrderState:
        state = self.get(fill.order_id)
        if state is None:
            raise KeyError(f"Unknown order_id: {fill.order_id}")
        executed_qty = max(0.0, float(fill.quantity))
        prior_qty = float(state.filled_quantity)
        next_qty = min(float(state.quantity), prior_qty + executed_qty)
        weighted_notional = (prior_qty * float(state.avg_fill_price)) + (executed_qty * float(fill.price))
        state.filled_quantity = next_qty
        state.remaining_quantity = max(0.0, float(state.quantity) - next_qty)
        if next_qty > 0:
            state.avg_fill_price = weighted_notional / next_qty
        state.updated_at = float(fill.ts)
        state.exchange_status = "FILLED" if state.remaining_quantity <= 1e-12 else "PARTIALLY_FILLED"
        state.status = "filled" if state.remaining_quantity <= 1e-12 else "partially_filled"
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        if not getattr(fill, "lineage", None):
            fill.lineage = dict(state.lineage or {})
        self._append_journal({"event": "fill", "ts": float(fill.ts), "order_id": state.order_id, "client_order_id": state.client_order_id, "lineage": dict(state.lineage or {}), "fill": asdict(fill), "order": state.to_dict()})
        return state

    def cancel(self, order_id: str, *, ts: float, reason: str = "") -> OrderState:
        state = self.get(order_id)
        if state is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        state.status = "canceled"
        state.canceled_at = float(ts)
        state.updated_at = float(ts)
        state.exchange_status = state.exchange_status or "CANCELED"
        if reason:
            state.last_error = str(reason)
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        self._append_journal({"event": "canceled", "ts": float(ts), "order_id": state.order_id, "client_order_id": state.client_order_id, "lineage": dict(state.lineage or {}), "reason": str(reason or "")})
        return state

    def reject(self, order_id: str, *, ts: float, reason: str) -> OrderState:
        state = self.get(order_id)
        if state is None:
            raise KeyError(f"Unknown order_id: {order_id}")
        state.status = "rejected"
        state.rejected_at = float(ts)
        state.updated_at = float(ts)
        state.exchange_status = "REJECTED"
        state.last_error = str(reason or "order_rejected")
        payload = self._load_payload()
        payload["orders"][state.order_id] = state.to_dict()
        payload["client_order_index"][state.client_order_id] = state.order_id
        self._save_payload(payload)
        self._append_journal({"event": "rejected", "ts": float(ts), "order_id": state.order_id, "client_order_id": state.client_order_id, "lineage": dict(state.lineage or {}), "reason": state.last_error})
        return state

    @staticmethod
    def to_request(state: OrderState) -> OrderRequest:
        return OrderRequest(
            order_id=state.order_id,
            client_order_id=state.client_order_id,
            symbol=state.symbol,
            side=state.side,
            order_type=state.order_type,
            quantity=state.quantity,
            tif=state.tif,
            limit_price=state.limit_price,
            stop_price=state.stop_price,
            reduce_only=state.reduce_only,
            metadata=dict(state.metadata or {}),
            lineage=dict(state.lineage or {}),
        )
