from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests


@dataclass(slots=True)
class ExchangePosition:
    """Normalized exchange position used for reconciliation."""

    symbol: str
    side: str
    quantity: float
    notional_usd: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usd: float


@dataclass(slots=True)
class ReconciliationResult:
    """Outcome of exchange-vs-ledger reconciliation."""

    success: bool
    reconciled: bool
    status: str
    exchange_position_count: int
    local_position_count: int
    mismatch_count: int
    snapshot_hash: str
    error: str = ""
    details: dict[str, Any] | None = None


class BinanceFuturesReconciler:
    """Fetch and reconcile Binance Futures positions against local runtime state."""

    def __init__(self, paths) -> None:
        self.paths = paths
        exchange_cfg = getattr(getattr(paths, "config", None), "exchange", None)
        self.cfg = exchange_cfg or SimpleNamespace(
            futures_rest_base_url="https://fapi.binance.com",
            api_key="",
            api_secret="",
            recv_window_ms=5000,
            use_testnet=False,
            position_reconciliation_enabled=False,
            require_reconciliation_for_live=True,
            max_reconciliation_age_sec=30,
        )
        self.timeout_sec = 10.0

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.cfg, "position_reconciliation_enabled", False))

    @property
    def credentials_available(self) -> bool:
        """Return whether the configured exchange credentials are present."""
        return bool(str(getattr(self.cfg, "api_key", "") or "").strip()) and bool(str(getattr(self.cfg, "api_secret", "") or "").strip())

    @property
    def base_url(self) -> str:
        if bool(getattr(self.cfg, "use_testnet", False)):
            return "https://testnet.binancefuture.com"
        return str(getattr(self.cfg, "futures_rest_base_url", "https://fapi.binance.com") or "https://fapi.binance.com").rstrip("/")

    def _sign_params(self, params: dict[str, Any]) -> dict[str, Any]:
        secret = str(getattr(self.cfg, "api_secret", "") or "")
        query = "&".join(f"{key}={params[key]}" for key in sorted(params.keys()))
        signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        signed = dict(params)
        signed["signature"] = signature
        return signed

    def _signed_get_json(self, path: str, params: dict[str, Any]) -> Any:
        headers = {"X-MBX-APIKEY": str(getattr(self.cfg, "api_key", "") or ""), "User-Agent": "AutoTrader/Reconciler"}
        signed_params = self._sign_params(params)
        response = requests.get(f"{self.base_url}{path}", params=signed_params, headers=headers, timeout=self.timeout_sec)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _snapshot_hash(payload: list[dict[str, Any]]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fetch_open_positions(self) -> list[ExchangePosition]:
        """Fetch currently open Binance Futures positions."""
        now_ms = int(time.time() * 1000)
        payload = self._signed_get_json(
            "/fapi/v2/positionRisk",
            {
                "timestamp": now_ms,
                "recvWindow": int(getattr(self.cfg, "recv_window_ms", 5000) or 5000),
            },
        )
        rows = payload if isinstance(payload, list) else []
        positions: list[ExchangePosition] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                quantity = float(item.get("positionAmt") or 0.0)
            except Exception:
                continue
            if abs(quantity) <= 0.0:
                continue
            symbol = str(item.get("symbol") or "").upper()
            side = "LONG" if quantity > 0 else "SHORT"
            mark_price = float(item.get("markPrice") or 0.0)
            entry_price = float(item.get("entryPrice") or 0.0)
            unrealized_pnl_usd = float(item.get("unRealizedProfit") or 0.0)
            notional_usd = abs(float(item.get("notional") or (mark_price * quantity)))
            positions.append(
                ExchangePosition(
                    symbol=symbol,
                    side=side,
                    quantity=abs(quantity),
                    notional_usd=abs(notional_usd),
                    entry_price=entry_price,
                    mark_price=mark_price,
                    unrealized_pnl_usd=unrealized_pnl_usd,
                )
            )
        positions.sort(key=lambda row: row.symbol)
        return positions

    def reconcile(self, portfolio_state, *, ts: float) -> ReconciliationResult:
        """Reconcile local runtime state against exchange positions."""
        if not self.enabled:
            return ReconciliationResult(
                success=False,
                reconciled=False,
                status="disabled",
                exchange_position_count=0,
                local_position_count=len(getattr(portfolio_state, "positions", {})),
                mismatch_count=0,
                snapshot_hash="",
                error="position_reconciliation_disabled",
                details={"reason": "position_reconciliation_disabled"},
            )

        if not self.credentials_available:
            return ReconciliationResult(
                success=False,
                reconciled=False,
                status="config_error",
                exchange_position_count=0,
                local_position_count=len(getattr(portfolio_state, "positions", {})),
                mismatch_count=0,
                snapshot_hash="",
                error="missing_exchange_credentials",
                details={"reason": "missing_exchange_credentials"},
            )

        try:
            exchange_positions = self.fetch_open_positions()
        except Exception as exc:
            return ReconciliationResult(
                success=False,
                reconciled=False,
                status="error",
                exchange_position_count=0,
                local_position_count=len(getattr(portfolio_state, "positions", {})),
                mismatch_count=0,
                snapshot_hash="",
                error=str(exc),
                details={"reason": "exchange_fetch_failed"},
            )

        exchange_map = {row.symbol: row for row in exchange_positions}
        local_map = {str(symbol).upper(): pos for symbol, pos in getattr(portfolio_state, "positions", {}).items()}
        mismatches: list[dict[str, Any]] = []

        for symbol in sorted(set(exchange_map.keys()) | set(local_map.keys())):
            exchange_pos = exchange_map.get(symbol)
            local_pos = local_map.get(symbol)
            if exchange_pos is None and local_pos is not None:
                mismatches.append({"symbol": symbol, "reason": "local_only"})
                continue
            if exchange_pos is not None and local_pos is None:
                mismatches.append({"symbol": symbol, "reason": "exchange_only"})
                continue
            if exchange_pos is None or local_pos is None:
                continue
            local_side = str(getattr(local_pos, "side", "") or "").upper()
            if local_side != exchange_pos.side:
                mismatches.append({"symbol": symbol, "reason": "side_mismatch", "local": local_side, "exchange": exchange_pos.side})
                continue
            local_qty = float(getattr(local_pos, "quantity", 0.0) or 0.0)
            if abs(local_qty - exchange_pos.quantity) > 1e-8:
                mismatches.append({"symbol": symbol, "reason": "quantity_mismatch", "local": local_qty, "exchange": exchange_pos.quantity})

        snapshot_rows = [
            {
                "symbol": row.symbol,
                "side": row.side,
                "quantity": row.quantity,
                "notional_usd": row.notional_usd,
                "entry_price": row.entry_price,
                "mark_price": row.mark_price,
                "unrealized_pnl_usd": row.unrealized_pnl_usd,
            }
            for row in exchange_positions
        ]
        snapshot_hash = self._snapshot_hash(snapshot_rows)
        reconciled = len(mismatches) == 0
        return ReconciliationResult(
            success=True,
            reconciled=reconciled,
            status="reconciled" if reconciled else "mismatch",
            exchange_position_count=len(exchange_positions),
            local_position_count=len(local_map),
            mismatch_count=len(mismatches),
            snapshot_hash=snapshot_hash,
            error="",
            details={"exchange_positions": snapshot_rows, "mismatches": mismatches, "checked_at": float(ts)},
        )
