from __future__ import annotations

import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import requests

from execution.order_models import FillEvent, OrderRequest


@dataclass(slots=True)
class ExchangeOrderResult:
    accepted: bool
    status: str
    exchange_order_id: str = ""
    exchange_status: str = ""
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    fill: FillEvent | None = None


class ExchangeAdapter(ABC):
    recovery_requires_exchange_authority = False
    authority_mode = "paper_local"
    requires_exchange_authority = False
    authority_reason = "paper_local_authority"

    @abstractmethod
    def place_order(self, request: OrderRequest, *, ts: float) -> ExchangeOrderResult:
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None, ts: float | None = None) -> ExchangeOrderResult:
        return ExchangeOrderResult(accepted=False, status="unsupported", error="cancel_not_supported")

    def get_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        return {"order_id": order_id, "symbol": symbol, "status": "unknown"}

    def get_open_orders(self) -> list[dict[str, Any]]:
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        return []

    def get_balances(self) -> list[dict[str, Any]]:
        return []


class PaperExchangeAdapter(ExchangeAdapter):
    """Immediate-fill paper adapter used by the controller path."""

    authority_mode = "paper_local"
    requires_exchange_authority = False
    authority_reason = "paper_local_authority"

    def __init__(self, paths=None) -> None:
        cfg = getattr(getattr(paths, "config", None), "trading", None)
        runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
        self.fees_bps = float(getattr(cfg, "fees_bps", 0.0) or 0.0)
        self.slippage_bps = float(getattr(cfg, "slippage_bps", 0.0) or 0.0)
        self.spread_bps = float(getattr(cfg, "spread_bps", 0.0) or 0.0)
        poll_seconds = float(getattr(runtime_cfg, "market_poll_seconds", 0.0) or 0.0)
        self.latency_bps = min(self.slippage_bps, max(0.0, poll_seconds) * 0.1)

    def _effective_bps(self) -> float:
        return max(0.0, self.spread_bps + self.slippage_bps + self.latency_bps)

    def _apply_adverse_fill(self, side: str, price: float) -> float:
        if price <= 0.0:
            return 0.0
        bps = self._effective_bps() / 10000.0
        if str(side).upper() == "LONG":
            return price * (1.0 + bps)
        return price * (1.0 - bps)

    def place_order(self, request: OrderRequest, *, ts: float) -> ExchangeOrderResult:
        reference_price = float(request.limit_price or request.stop_price or request.metadata.get("reference_price") or 0.0)
        fill_price = float(self._apply_adverse_fill(request.side, reference_price) or reference_price)
        notional = abs(float(request.quantity) * float(fill_price))
        fee_usd = notional * (self.fees_bps / 10000.0) if notional > 0.0 else 0.0
        estimated_slippage_usd = abs(float(request.quantity)) * abs(float(fill_price) - float(reference_price)) if reference_price > 0.0 else 0.0
        fill = FillEvent(
            order_id=request.order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=float(request.quantity),
            price=float(fill_price),
            fill_id=f"paper-{request.order_id}",
            fee_usd=float(fee_usd),
            liquidity="paper",
            ts=float(ts),
            metadata={
                "mode": "paper",
                "reference_price": reference_price,
                "fee_bps": self.fees_bps,
                "spread_bps": self.spread_bps,
                "slippage_bps": self.slippage_bps,
                "latency_bps": self.latency_bps,
                "effective_slippage_bps": self._effective_bps(),
                "estimated_slippage_usd": estimated_slippage_usd,
                "notional_usd": notional,
            },
        )
        return ExchangeOrderResult(
            accepted=True,
            status="filled",
            exchange_order_id=request.order_id,
            exchange_status="FILLED",
            details={"mode": "paper", "effective_slippage_bps": self._effective_bps(), "fee_usd": fee_usd, "estimated_slippage_usd": estimated_slippage_usd},
            fill=fill,
        )


class BinanceFuturesExchangeAdapter(ExchangeAdapter):
    """Thin Binance Futures adapter for execution routing."""

    recovery_requires_exchange_authority = True
    authority_mode = "live_exchange_required"
    requires_exchange_authority = True
    authority_reason = "live_exchange_authority_required"

    def __init__(self, paths) -> None:
        self.paths = paths
        exchange_cfg = getattr(getattr(paths, "config", None), "exchange", None)
        self.cfg = exchange_cfg or SimpleNamespace(
            futures_rest_base_url="https://fapi.binance.com",
            api_key="",
            api_secret="",
            recv_window_ms=5000,
            use_testnet=False,
        )
        self.timeout_sec = 10.0

    @property
    def base_url(self) -> str:
        if bool(getattr(self.cfg, "use_testnet", False)):
            return "https://testnet.binancefuture.com"
        return str(getattr(self.cfg, "futures_rest_base_url", "https://fapi.binance.com") or "https://fapi.binance.com").rstrip("/")

    @property
    def credentials_available(self) -> bool:
        return bool(str(getattr(self.cfg, "api_key", "") or "").strip()) and bool(str(getattr(self.cfg, "api_secret", "") or "").strip())

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        secret = str(getattr(self.cfg, "api_secret", "") or "")
        query = "&".join(f"{key}={params[key]}" for key in sorted(params.keys()))
        signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        signed = dict(params)
        signed["signature"] = signature
        return signed

    def _signed_request(self, method: str, path: str, params: dict[str, Any]) -> Any:
        headers = {"X-MBX-APIKEY": str(getattr(self.cfg, "api_key", "") or ""), "User-Agent": "AutoTrader/Execution"}
        signed_params = self._sign_params(params)
        response = requests.request(method.upper(), f"{self.base_url}{path}", params=signed_params, headers=headers, timeout=self.timeout_sec)
        response.raise_for_status()
        return response.json()

    def place_order(self, request: OrderRequest, *, ts: float) -> ExchangeOrderResult:
        if not self.credentials_available:
            return ExchangeOrderResult(accepted=False, status="rejected", error="missing_exchange_credentials")
        side = "BUY" if str(request.side).upper() == "LONG" else "SELL"
        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": side,
            "type": str(request.order_type or "MARKET").upper(),
            "quantity": request.quantity,
            "newClientOrderId": request.client_order_id,
            "recvWindow": int(getattr(self.cfg, "recv_window_ms", 5000) or 5000),
            "timestamp": int(max(float(ts), time.time()) * 1000),
        }
        if request.limit_price is not None:
            params["price"] = request.limit_price
        if request.stop_price is not None:
            params["stopPrice"] = request.stop_price
        if request.tif and params["type"] != "MARKET":
            params["timeInForce"] = request.tif
        if request.reduce_only:
            params["reduceOnly"] = "true"
        try:
            payload = self._signed_request("POST", "/fapi/v1/order", params)
        except Exception as exc:
            return ExchangeOrderResult(accepted=False, status="rejected", error=str(exc))
        exchange_order_id = str(payload.get("orderId") or "")
        exchange_status = str(payload.get("status") or "NEW")
        result = ExchangeOrderResult(
            accepted=True,
            status="submitted",
            exchange_order_id=exchange_order_id,
            exchange_status=exchange_status,
            details=payload if isinstance(payload, dict) else {"payload": payload},
        )
        executed_qty = float(payload.get("executedQty") or 0.0) if isinstance(payload, dict) else 0.0
        avg_price = float(payload.get("avgPrice") or payload.get("price") or 0.0) if isinstance(payload, dict) else 0.0
        if executed_qty > 0.0 and avg_price > 0.0:
            result.status = "filled" if abs(executed_qty - float(request.quantity)) <= 1e-12 else "partially_filled"
            result.fill = FillEvent(
                order_id=request.order_id,
                client_order_id=request.client_order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=executed_qty,
                price=avg_price,
                fill_id=exchange_order_id,
                fee_usd=0.0,
                liquidity="exchange",
                ts=float(ts),
                metadata={"exchange_status": exchange_status},
            )
        return result

    def get_order(self, order_id: str, *, symbol: str | None = None) -> dict[str, Any]:
        if not self.credentials_available:
            return {"order_id": order_id, "symbol": symbol, "status": "credentials_missing"}
        params = {
            "origClientOrderId": order_id,
            "recvWindow": int(getattr(self.cfg, "recv_window_ms", 5000) or 5000),
            "timestamp": int(time.time() * 1000),
        }
        if symbol:
            params["symbol"] = symbol
        try:
            payload = self._signed_request("GET", "/fapi/v1/order", params)
        except Exception as exc:
            return {"order_id": order_id, "symbol": symbol, "status": "error", "error": str(exc)}
        if isinstance(payload, dict):
            return {
                "order_id": str(payload.get("origClientOrderId") or order_id),
                "exchange_order_id": str(payload.get("orderId") or ""),
                "symbol": str(payload.get("symbol") or symbol or ""),
                "status": str(payload.get("status") or "unknown"),
            }
        return {"order_id": order_id, "symbol": symbol, "status": "unknown"}

    def get_open_orders(self) -> list[dict[str, Any]]:
        if not self.credentials_available:
            return []
        params = {
            "recvWindow": int(getattr(self.cfg, "recv_window_ms", 5000) or 5000),
            "timestamp": int(time.time() * 1000),
        }
        try:
            payload = self._signed_request("GET", "/fapi/v1/openOrders", params)
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            rows.append({
                "client_order_id": str(item.get("clientOrderId") or ""),
                "exchange_order_id": str(item.get("orderId") or ""),
                "symbol": str(item.get("symbol") or ""),
                "status": str(item.get("status") or "NEW"),
            })
        return rows
