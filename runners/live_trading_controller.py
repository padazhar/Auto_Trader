from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.log_manager import get_logger
from core.symbol_management import load_symbol_state, record_blocked_position_close_result_entry, symbols_requiring_blocked_position_close
from core.market_io import latest_symbol_price, latest_symbol_price_status, load_active_universe_symbols, load_enriched_market_rows, normalize_price_components
from core.model_approval import resolve_live_incumbent_requirements
from core.shared_utils import now_utc_ts
from execution.exchange_reconciliation import BinanceFuturesReconciler
from execution.exchange_adapter import BinanceFuturesExchangeAdapter, PaperExchangeAdapter
from execution.fill_handler import FillHandler
from execution.order_models import OrderIntent
from execution.order_router import OrderRouter
from execution.order_store import OrderStore
from execution.portfolio_state import PortfolioStateStore
from execution.risk_engine import RiskEngine
from features.feature_store_contract import validate_market_data_freshness
from inference.trade_decision_logic import TradeDecisionLogic
from monitoring.control_plane_store import ControlPlaneStore
from monitoring.publisher import MonitorPublisher
from services.signal_engine import SignalEngine, build_decision_lineage_id, normalize_lineage_payload


class LiveTradingController:
    def __init__(self, paths, registry=None, *, signal_engine: SignalEngine | None = None) -> None:
        self.paths = paths
        self.logger = get_logger("runners.live_trading_controller", paths)

        resolved_signal_engine = signal_engine
        if resolved_signal_engine is None:
            try:
                resolved_signal_engine = SignalEngine(paths)
            except Exception as exc:
                self.logger.warning("SignalEngine unavailable during controller init; falling back to direct logic path: %s", exc)
                resolved_signal_engine = None
        self.signal_engine = resolved_signal_engine

        self.monitor = MonitorPublisher(paths)
        self.control_store = ControlPlaneStore(paths)
        self.portfolio_store = PortfolioStateStore(
            paths.portfolio_state_path,
            starting_equity_usd=float(paths.config.risk.account_balance_usd),
        )
        self.execution_mode = str(getattr(paths.config.runtime, "execution_mode", "paper") or "paper").lower()
        self.exchange_cfg = getattr(paths.config, "exchange", None)
        self.reconciler = BinanceFuturesReconciler(paths)
        self.risk_engine = RiskEngine(paths)
        runtime_dir = getattr(paths, "runtime_dir", None)
        if runtime_dir is None:
            portfolio_state_path = getattr(paths, "portfolio_state_path", None)
            if portfolio_state_path is not None:
                runtime_dir = getattr(portfolio_state_path, "parent", None)
            if runtime_dir is None:
                runtime_dir = __import__("pathlib").Path(".")
        order_state_path = getattr(paths, "order_state_path", None) or (runtime_dir / "order_state.json")
        execution_journal_path = getattr(paths, "execution_journal_path", None) or (runtime_dir / "execution_journal.jsonl")
        self.order_store = OrderStore(order_state_path, execution_journal_path)
        self.fill_handler = FillHandler(self.order_store)
        if self.execution_mode != "live":
            try:
                self.exchange_adapter = PaperExchangeAdapter(paths)
            except TypeError:
                self.exchange_adapter = PaperExchangeAdapter()
        else:
            self.exchange_adapter = BinanceFuturesExchangeAdapter(paths)
        self.order_router = OrderRouter(self.order_store, self.exchange_adapter, self.fill_handler)
        self._recover_execution_state()
        self.portfolio_store.save(self.portfolio_store.load())
        self.logic = getattr(self.signal_engine, "logic", None) if self.signal_engine is not None else None
        if self.logic is None:
            self.logic = TradeDecisionLogic(paths)
        self.registry = registry
        self._last_full_signal_cycle_ts = 0.0
        self._last_fast_mark_publish_ts = 0.0
        self._startup_authority_verified = False
        self._current_authority_reason: str | None = None

    def _live_canary_limits(self) -> dict[str, object]:
        runtime_cfg = getattr(getattr(self.paths, "config", None), "runtime", None)
        exchange_cfg = getattr(getattr(self.paths, "config", None), "exchange", None)
        return {
            "enabled": bool(getattr(runtime_cfg, "live_canary_enabled", True)),
            "max_symbols": int(getattr(runtime_cfg, "live_canary_max_symbols", 1) or 1),
            "max_notional_usd": float(getattr(runtime_cfg, "live_canary_max_notional_usd", 50.0) or 0.0),
            "max_concurrent_positions": int(getattr(runtime_cfg, "live_canary_max_concurrent_positions", 1) or 1),
            "require_testnet": bool(getattr(runtime_cfg, "live_canary_require_testnet", True)),
            "use_testnet": bool(getattr(exchange_cfg, "use_testnet", False)),
        }


    def _recover_execution_state(self) -> None:
        try:
            recovery = self.order_router.recover_open_orders(ts=now_utc_ts())
        except Exception as exc:
            self.logger.warning("Execution-state recovery failed during controller init: %s", exc)
            self._call_monitor("publish_event", "execution_recovery_failed", {"error": str(exc)})
            return
        recovered_orders = list(recovery.get("recovered_orders") or [])
        if recovered_orders:
            self._call_monitor("publish_event", "execution_state_recovered", recovery)

    def _safe_asdict(self, value: object) -> dict[str, object]:
        if value is None:
            return {}
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        return dict(getattr(value, "__dict__", {}) or {})

    @staticmethod
    def _candidate_id_from_artifact(payload: dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("candidate_id") or payload.get("id") or "")

    def _extend_trade_entry_lineage(
        self,
        *,
        base_lineage: dict[str, Any],
        decision: Any,
        decision_components: dict[str, Any],
        risk_decision_payload: dict[str, Any],
        approved: bool,
    ) -> dict[str, Any]:
        """Attach Step 5 closed-trade attribution inputs to entry lineage only.

        This is intentionally additive. It does not change the action, risk
        approval, order intent, or paper/live authority.
        """
        lineage = dict(base_lineage or {})
        rf = decision_components.get("rf") if isinstance(decision_components.get("rf"), dict) else {}
        gbm = decision_components.get("gbm") if isinstance(decision_components.get("gbm"), dict) else {}
        dqn = decision_components.get("dqn") if isinstance(decision_components.get("dqn"), dict) else {}
        rf_artifact = rf.get("artifact") if isinstance(rf.get("artifact"), dict) else {}
        gbm_artifact = gbm.get("artifact") if isinstance(gbm.get("artifact"), dict) else {}
        gbm_direction = gbm_artifact.get("direction") if isinstance(gbm_artifact.get("direction"), dict) else {}
        dqn_artifact = dqn.get("artifact") if isinstance(dqn.get("artifact"), dict) else {}
        consensus = decision_components.get("consensus") if isinstance(decision_components.get("consensus"), dict) else {}
        lineage.update({
            "signal_action": str(getattr(decision, "action", "") or ""),
            "signal_reason": str(getattr(decision, "reason", "") or ""),
            "entry_action": str(getattr(decision, "action", "") or ""),
            "entry_reason": str(getattr(decision, "reason", "") or ""),
            "score": float(getattr(decision, "score", 0.0) or 0.0),
            "ev_score": float(consensus.get("ev_score", getattr(decision, "score", 0.0)) or 0.0),
            "risk_approved": bool(approved),
            "risk_reason": str(risk_decision_payload.get("reason") or ""),
            "model_candidates": {
                "rf": self._candidate_id_from_artifact(rf_artifact),
                "gbm": self._candidate_id_from_artifact(gbm_direction),
                "policy": self._candidate_id_from_artifact(dqn_artifact),
            },
            "prediction_vector": {
                "rf": {"short": rf.get("p_short"), "hold": rf.get("p_hold"), "long": rf.get("p_long")},
                "gbm": {"short": gbm.get("p_short"), "hold": gbm.get("p_hold"), "long": gbm.get("p_long"), "meta_probability": gbm.get("meta_probability")},
                "policy": {"action": dqn.get("action"), "confidence": dqn.get("confidence"), "score": dqn.get("score")},
            },
        })
        return lineage

    def _call_monitor(self, method_name: str, *args, **kwargs) -> None:
        method = getattr(self.monitor, method_name, None)
        if callable(method):
            method(*args, **kwargs)

    def _log_debug(self, message: str, *args) -> None:
        method = getattr(self.logger, "debug", None)
        if callable(method):
            method(message, *args)


    def _normalize_decision_price(self, symbol: str, components: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
        # Prefer mark_price for execution/open-trade decisions. price/latest_price
        # are display aliases and can lag or point at ticker/db-close fallbacks.
        candidate_price = (
            components.get("mark_price")
            or components.get("primary_price")
            or components.get("latest_price")
            or components.get("price")
            or components.get("reference_price")
        )
        price_payload = normalize_price_components(
            candidate_price,
            price_source=str(components.get("price_source") or components.get("source") or "decision_components"),
            price_ts_ms=components.get("price_ts_ms") or components.get("event_time_ms") or components.get("timestamp"),
        )
        if not price_payload.get("price_available"):
            fallback_status = latest_symbol_price_status(self.paths, symbol)
            fallback_price = fallback_status.get("price") if fallback_status.get("price_fresh") else None
            price_payload = normalize_price_components(
                fallback_price,
                price_source=str(fallback_status.get("price_source") or fallback_status.get("reason") or "runtime_market_price"),
                price_ts_ms=fallback_status.get("price_ts_ms"),
                fallback_reason=str(fallback_status.get("reason") or "price_unavailable") if fallback_price is None else None,
            )
            price_payload.update({
                "price_fresh": bool(fallback_status.get("price_fresh")),
                "price_age_seconds": fallback_status.get("price_age_seconds"),
                "price_max_age_seconds": fallback_status.get("max_age_seconds"),
                "price_status_reason": fallback_status.get("reason"),
            })
        components.update(price_payload)
        latest_price = price_payload.get("latest_price")
        try:
            return (float(latest_price) if latest_price is not None else None), components
        except Exception:
            return None, components


    def _runtime_price_snapshot_status(self, symbols: list[str]) -> dict[str, Any]:
        """Return whether the runtime market-price snapshot is ready for execution.

        The market-data controller and live controller start concurrently in
        ``--mode all``. Without this guard, the live controller can evaluate
        one full symbol loop before ``data/runtime/market/latest.json`` is
        written, which creates a noisy global ``price_unavailable`` decision
        burst even though the price pipeline becomes healthy seconds later.
        Step 2 requires missing runtime prices to be explicit, but startup
        ordering should be reported as a controller-level wait state rather
        than as per-symbol execution decisions.
        """
        checked_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
        # Minimal unit-test and diagnostic harnesses may construct a PathManager-like
        # object without the runtime market snapshot path contract. In that case, do
        # not block the execution loop waiting for a snapshot that cannot be written.
        # Real runtime PathManager instances provide either runtime_dir or
        # market_latest_snapshot_path, so production startup still waits for market
        # prices before evaluating new entries.
        if not hasattr(self.paths, "market_latest_snapshot_path"):
            return {
                "ready": True,
                "total_symbols": len(checked_symbols),
                "priced_symbols": 0,
                "missing_symbols": [],
                "stale_symbols": [],
                "prices": {},
                "sources": {},
                "price_ts_ms": {},
                "price_age_seconds": {},
                "max_age_seconds": float(getattr(getattr(self.paths.config, "runtime", None), "price_max_age_seconds", 180.0) or 180.0),
                "reason": "market_price_snapshot_not_configured",
            }
        priced: dict[str, float] = {}
        missing: list[str] = []
        stale: list[str] = []
        sources: dict[str, str] = {}
        price_ts_ms: dict[str, int | None] = {}
        ages: dict[str, float | None] = {}
        max_age_seconds = float(getattr(getattr(self.paths.config, "runtime", None), "price_max_age_seconds", 180.0) or 180.0)
        for symbol in checked_symbols:
            status = latest_symbol_price_status(self.paths, symbol, max_age_seconds=max_age_seconds)
            if not status.get("price_available"):
                missing.append(symbol)
                continue
            if not status.get("price_fresh"):
                stale.append(symbol)
                continue
            price = self._coerce_positive_price(status.get("price"))
            if price is None:
                missing.append(symbol)
                continue
            priced[symbol] = float(price)
            sources[symbol] = str(status.get("price_source") or "unknown")
            price_ts_ms[symbol] = status.get("price_ts_ms")
            ages[symbol] = status.get("price_age_seconds")
        ready = bool(checked_symbols) and not missing and not stale
        reason = "market_prices_ready" if ready else ("market_prices_stale" if stale else "market_price_snapshot_not_ready")
        return {
            "ready": ready,
            "total_symbols": len(checked_symbols),
            "priced_symbols": len(priced),
            "missing_symbols": missing,
            "stale_symbols": stale,
            "prices": priced,
            "sources": sources,
            "price_ts_ms": price_ts_ms,
            "price_age_seconds": ages,
            "max_age_seconds": max_age_seconds,
            "reason": reason,
        }

    @staticmethod
    def _coerce_positive_price(value: object) -> float | None:
        try:
            price = float(value)
        except Exception:
            return None
        return price if price > 0.0 else None

    def _runtime_float(self, name: str, default: float, *, minimum: float = 0.0) -> float:
        runtime_cfg = getattr(getattr(self.paths, "config", None), "runtime", None)
        try:
            value = float(getattr(runtime_cfg, name, default))
        except Exception:
            value = float(default)
        return max(float(minimum), float(value))

    def _execution_price_refresh_seconds(self) -> float:
        return self._runtime_float("execution_price_refresh_seconds", 1.0, minimum=0.1)

    def _execution_price_max_age_seconds(self) -> float:
        fallback = self._runtime_float("price_max_age_seconds", 180.0, minimum=1.0)
        return self._runtime_float("execution_price_max_age_seconds", min(3.0, fallback), minimum=0.5)

    @staticmethod
    def _requires_fresh_execution_entry_price(action: str) -> bool:
        """Return whether an action can create or reverse market exposure."""
        return str(action or "").upper() in {"ENTER_LONG", "ENTER_SHORT", "FLIP_TO_LONG", "FLIP_TO_SHORT"}

    def _execution_order_price_gate(self, symbol: str, action: str, *, now_ms: int | None = None) -> dict[str, Any]:
        """Resolve the final routing-time price proof for new exposure.

        Signal generation may retain an older reference price for diagnostic
        traceability, but an entry or flip is not authorised unless the runtime
        market-price snapshot proves a price within the strict execution-age
        contract immediately before risk/order routing. Exits and protective
        handling retain their separate reduction/protection paths.
        """
        action_u = str(action or "").upper()
        required = self._requires_fresh_execution_entry_price(action_u)
        if not required:
            return {
                "required": False,
                "ok": True,
                "blocked": False,
                "symbol": str(symbol or "").upper(),
                "action": action_u,
                "reason": "execution_price_gate_not_required",
                "max_age_seconds": self._execution_price_max_age_seconds(),
            }

        status = latest_symbol_price_status(
            self.paths,
            str(symbol or "").upper(),
            max_age_seconds=self._execution_price_max_age_seconds(),
            now_ms=now_ms,
            require_fresh=True,
        )
        available = bool(status.get("price_available", False))
        fresh = bool(status.get("price_fresh", False))
        status_reason = str(status.get("reason") or "")
        if available and fresh:
            reason = "execution_price_ready"
        elif not available:
            reason = "execution_price_unavailable"
        elif status_reason == "price_timestamp_missing":
            reason = "execution_price_timestamp_missing"
        else:
            reason = "execution_price_stale"
        return {
            "required": True,
            "ok": bool(available and fresh),
            "blocked": not bool(available and fresh),
            "symbol": str(symbol or "").upper(),
            "action": action_u,
            "price": status.get("price") if available and fresh else None,
            "observed_price": status.get("price"),
            "price_source": str(status.get("price_source") or status_reason or "unavailable"),
            "price_ts_ms": status.get("price_ts_ms"),
            "price_age_seconds": status.get("price_age_seconds"),
            "max_age_seconds": status.get("max_age_seconds", self._execution_price_max_age_seconds()),
            "status_reason": status_reason,
            "reason": reason,
        }

    @staticmethod
    def _has_open_positions(portfolio_state: Any) -> bool:
        return bool(getattr(portfolio_state, "positions", {}) or {})

    def _open_position_count(self) -> int:
        try:
            state = self.portfolio_store.load()
            return len(getattr(state, "positions", {}) or {})
        except Exception as exc:
            self._log_debug("Failed to resolve open-position count for execution refresh cadence: %s", exc)
            return 0

    def _next_loop_wait_seconds(self) -> float:
        base_seconds = self._runtime_float("market_poll_seconds", 10.0, minimum=0.1)
        if self._open_position_count() <= 0:
            return base_seconds
        return min(base_seconds, self._execution_price_refresh_seconds())

    def _should_wait_for_market_prices(
        self,
        symbols: list[str],
        price_snapshot_status: dict[str, Any],
        portfolio_state: Any,
    ) -> bool:
        """Return True only when it is safe to pause before active prices are ready.

        If any open position exists, the controller must keep marking from the
        runtime market-price cache and checking protective SL/TP. Active-symbol
        readiness can be temporarily incomplete without disabling protection for
        already-open exposure.
        """
        if not symbols:
            return False
        if bool(price_snapshot_status.get("ready", False)):
            return False
        return not self._has_open_positions(portfolio_state)

    def _signal_cycle_seconds(self) -> float:
        return self._runtime_float("market_poll_seconds", 10.0, minimum=0.1)

    def _should_run_full_signal_cycle(self, *, ts: float, portfolio_state: Any) -> bool:
        if not self._has_open_positions(portfolio_state):
            return True
        last_full = float(getattr(self, "_last_full_signal_cycle_ts", 0.0) or 0.0)
        if last_full <= 0.0:
            return True
        return (float(ts) - last_full) >= self._signal_cycle_seconds()

    def _build_open_position_row(
        self,
        symbol: str,
        pos: Any,
        latest_prices: dict[str, float] | None = None,
        price_ts_ms: dict[str, int | None] | None = None,
        price_sources: dict[str, str] | None = None,
        *,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Return the canonical execution/UI position payload."""
        normalized_symbol = str(symbol or getattr(pos, "symbol", "") or "").upper()
        latest_prices = latest_prices or {}
        price_ts_ms = price_ts_ms or {}
        price_sources = price_sources or {}

        last_price = self._coerce_positive_price(latest_prices.get(normalized_symbol))
        source = str(price_sources.get(normalized_symbol) or "runtime_latest_prices")
        ts_ms = price_ts_ms.get(normalized_symbol)
        price_status_reason = ""
        if last_price is None:
            fallback_status = latest_symbol_price_status(self.paths, normalized_symbol)
            fallback_price = fallback_status.get("price")
            last_price = self._coerce_positive_price(fallback_price) if fallback_status.get("price_fresh") else None
            source = str(fallback_status.get("price_source") or source or "unavailable")
            ts_ms = fallback_status.get("price_ts_ms") if fallback_status.get("price_ts_ms") is not None else ts_ms
            price_status_reason = str(fallback_status.get("reason") or "")

        entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
        quantity = float(getattr(pos, "quantity", 0.0) or 0.0)
        side = str(getattr(pos, "side", "") or "").upper()
        direction = 1.0 if side == "LONG" else -1.0
        estimated_pnl = float(getattr(pos, "unrealized_pnl_usd", 0.0) or 0.0)
        if last_price is not None and entry_price > 0.0 and quantity > 0.0:
            estimated_pnl = (float(last_price) - entry_price) * quantity * direction

        last_price_ts = getattr(pos, "updated_at", None) or getattr(pos, "opened_at", None)
        if ts_ms is not None:
            try:
                last_price_ts = float(ts_ms) / 1000.0 if float(ts_ms) > 10_000_000_000 else float(ts_ms)
            except Exception:
                pass

        return {
            "symbol": normalized_symbol,
            "side": side,
            "quantity": quantity,
            "notional_usd": float(getattr(pos, "notional_usd", 0.0) or 0.0),
            "entry_price": entry_price,
            "stop_loss": float(getattr(pos, "stop_loss", 0.0) or 0.0),
            "take_profit": float(getattr(pos, "take_profit", 0.0) or 0.0),
            "opened_at": float(getattr(pos, "opened_at", 0.0) or 0.0),
            "updated_at": float(getattr(pos, "updated_at", 0.0) or 0.0),
            "mode": str(mode or self.execution_mode),
            "last_price": last_price,
            "latest_price": last_price,
            "reference_price": last_price,
            "last_price_ts": last_price_ts,
            "price_ts_ms": ts_ms,
            "price_source": source if last_price is not None else "unavailable",
            "estimated_pnl": float(estimated_pnl),
            "pnl_usd": float(estimated_pnl),
            "unrealized_pnl_usd": float(estimated_pnl),
            "stale": last_price is None or price_status_reason == "price_stale",
            "price_status_reason": price_status_reason,
        }

    def _refresh_open_position_marks(
        self,
        portfolio_state: Any,
        latest_prices: dict[str, float],
        price_ts_ms: dict[str, int | None],
        price_sources: dict[str, str],
        *,
        ts: float,
    ) -> dict[str, Any]:
        """Refresh open-position marks from the runtime price cache every loop."""
        refreshed: list[str] = []
        missing: list[str] = []
        for symbol in list(getattr(portfolio_state, "positions", {}) or {}):
            normalized = str(symbol or "").upper()
            if not normalized:
                continue
            status = latest_symbol_price_status(
                self.paths,
                normalized,
                max_age_seconds=self._execution_price_max_age_seconds(),
            )
            coerced = self._coerce_positive_price(status.get("price")) if status.get("price_fresh") else None
            if coerced is None:
                missing.append(normalized)
                continue
            latest_prices[normalized] = float(coerced)
            price_sources[normalized] = str(status.get("price_source") or "runtime_market_price")
            price_ts_ms[normalized] = status.get("price_ts_ms")
            refreshed.append(normalized)
        pnl_summary = self.portfolio_store.mark_to_market(portfolio_state, latest_prices, ts=ts)
        return {
            "refreshed_symbols": refreshed,
            "missing_symbols": missing,
            "pnl_summary": dict(pnl_summary or {}),
            "max_age_seconds": self._execution_price_max_age_seconds(),
        }

    @staticmethod
    def _protective_exit_action(pos: Any, mark_price: float | None) -> tuple[str | None, str]:
        """Return an EXIT action when an open position hits stop-loss or take-profit."""
        try:
            price = float(mark_price) if mark_price is not None else 0.0
        except Exception:
            price = 0.0
        if price <= 0.0:
            return None, ""
        side = str(getattr(pos, "side", "") or "").upper()
        try:
            stop_loss = float(getattr(pos, "stop_loss", 0.0) or 0.0)
        except Exception:
            stop_loss = 0.0
        try:
            take_profit = float(getattr(pos, "take_profit", 0.0) or 0.0)
        except Exception:
            take_profit = 0.0
        if side == "LONG":
            if stop_loss > 0.0 and price <= stop_loss:
                return "EXIT_LONG", "protective_stop_loss"
            if take_profit > 0.0 and price >= take_profit:
                return "EXIT_LONG", "protective_take_profit"
        elif side == "SHORT":
            if stop_loss > 0.0 and price >= stop_loss:
                return "EXIT_SHORT", "protective_stop_loss"
            if take_profit > 0.0 and price <= take_profit:
                return "EXIT_SHORT", "protective_take_profit"
        return None, ""

    @staticmethod
    def _order_side_for_action(action: str, current_side: str = "") -> str:
        """Return exchange side direction implied by the execution action."""
        action_u = str(action or "").upper()
        current_side_u = str(current_side or "").upper()
        if action_u in {"ENTER_LONG", "FLIP_TO_LONG", "EXIT_SHORT"}:
            return "LONG"
        if action_u in {"ENTER_SHORT", "FLIP_TO_SHORT", "EXIT_LONG"}:
            return "SHORT"
        if action_u == "EXIT":
            return "SHORT" if current_side_u == "LONG" else "LONG"
        return "LONG" if "LONG" in action_u else "SHORT"

    def _route_protective_exit(
        self,
        *,
        symbol: str,
        pos: Any,
        action: str,
        reason: str,
        mark_price: float,
        ts: float,
        previous_position: int,
        portfolio_state: Any,
    ) -> dict[str, Any]:
        """Route a reduce-only protective exit and update portfolio state on fill."""
        current_side = str(getattr(pos, "side", "") or "").upper()
        quantity = float(getattr(pos, "quantity", 0.0) or 0.0)
        intent = OrderIntent(
            symbol=str(symbol).upper(),
            action=str(action).upper(),
            side=self._order_side_for_action(action, current_side),
            quantity=quantity,
            order_type="MARKET",
            reduce_only=True,
            signal_reason=reason,
            metadata={
                "reference_price": float(mark_price),
                "mode": self.execution_mode,
                "protective_exit": True,
                "position_side": current_side,
                "idempotency_key": f"protective:{str(symbol).upper()}:{str(action).upper()}:{reason}:{float(ts):.0f}",
                "lineage": {**dict(getattr(pos, "lineage", {}) or {}), "close_trigger": str(reason), "close_trigger_source": "protective_exit", "contract_version": "trade_lineage_v1"},
            },
            lineage={**dict(getattr(pos, "lineage", {}) or {}), "close_trigger": str(reason), "close_trigger_source": "protective_exit", "contract_version": "trade_lineage_v1"},
        )
        execution_result = self.order_router.route_intent(intent, ts=ts, previous_position=previous_position)
        fill_payload = dict(execution_result.get("fill") or {})
        fill_price = float(fill_payload.get("price") or mark_price or 0.0)
        if execution_result.get("status") in {"filled", "partially_filled"} and fill_price > 0.0:
            self.portfolio_store.apply_execution(
                portfolio_state,
                symbol=symbol,
                action=action,
                price=fill_price,
                ts=ts,
                position_plan={"quantity": quantity, "exit_reason": reason, "lineage": {**dict(intent.lineage or {}), "execution_fill": fill_payload}, "execution_fill": fill_payload},
            )
            self.portfolio_store.save(portfolio_state)
        return dict(execution_result or {})

    def _control_plane_summary(self) -> dict[str, object]:
        method = getattr(self.control_store, "summary", None)
        if callable(method):
            try:
                result = method()
                if isinstance(result, dict):
                    return result
            except Exception as exc:
                self._log_debug("Failed to fetch control-plane summary: %s", exc)
        pause_state = self.control_store.pause_state()
        return {
            "paused": bool(pause_state.get("paused", False)),
            "pause_reason": str(pause_state.get("pause_reason") or ""),
            "updated_at": pause_state.get("updated_at"),
            "updated_by": pause_state.get("updated_by"),
        }

    def _requires_exchange_authority(self) -> bool:
        return bool(getattr(self.exchange_adapter, "requires_exchange_authority", self.execution_mode == "live"))

    def _mark_authority_mode(self, portfolio_state, *, reconciled: bool) -> None:
        method = getattr(self.portfolio_store, "mark_authority_mode", None)
        effective_mode = "live" if self._requires_exchange_authority() else "paper"
        if callable(method):
            method(portfolio_state, execution_mode=effective_mode, reconciled=reconciled)
            return
        if self._requires_exchange_authority():
            portfolio_state.authority_mode = "live_exchange_reconciled" if reconciled else "live_exchange_required"
            portfolio_state.reconciliation_status = "reconciled" if reconciled else str(getattr(portfolio_state, "reconciliation_status", "") or "live_reconciliation_required")
        else:
            portfolio_state.authority_mode = "paper_local"
            portfolio_state.reconciliation_status = "paper_local"

    def _runtime_gate(self, portfolio_state, circuit_state, authority: dict[str, object]) -> dict[str, object]:
        method = getattr(self.control_store, "runtime_gate", None)
        if callable(method):
            gate = dict(method(portfolio_state=portfolio_state, circuit_state=self._safe_asdict(circuit_state), authority=authority))
        else:
            pause_state = self.control_store.pause_state()
            paused = bool(pause_state.get("paused", False))
            pause_reason = str(pause_state.get("pause_reason") or "")
            circuit_triggered = bool(getattr(circuit_state, "triggered", False) or self._safe_asdict(circuit_state).get("triggered", False))
            circuit_reason = str(getattr(circuit_state, "reason", "") or self._safe_asdict(circuit_state).get("reason", "") or "")
            authoritative = bool(authority.get("authoritative", False))
            live_reason = str(authority.get("reason") or "")
            blocked = paused or circuit_triggered or (self._requires_exchange_authority() and not authoritative)
            reason = "control_plane_paused" if paused else ("circuit_breaker_triggered" if circuit_triggered else live_reason)
            status = "paused" if paused else ("blocked" if circuit_triggered else ("guarded" if blocked else "running"))
            gate = {
                "blocked": blocked,
                "reason": reason,
                "status": status,
                "trading_enabled": not blocked,
                "paused": paused,
                "pause_reason": pause_reason,
                "circuit_triggered": circuit_triggered,
                "circuit_reason": circuit_reason,
                "live_authoritative": authoritative,
                "live_authority_reason": live_reason,
            }

        integrity_status = str(getattr(portfolio_state, "execution_integrity_status", "ok") or "ok").lower()
        if integrity_status != "ok":
            integrity_reason = str(getattr(portfolio_state, "execution_integrity_error", "") or "execution_state_integrity_failure")
            gate.update({
                "blocked": True,
                "reason": integrity_reason,
                "status": "blocked",
                "trading_enabled": False,
                "execution_integrity_status": integrity_status,
                "execution_integrity_error": integrity_reason,
                "execution_integrity_failure_at": getattr(portfolio_state, "execution_integrity_failure_at", None),
            })
        return gate

    def _block_on_execution_integrity_failure(
        self,
        portfolio_state: Any,
        *,
        symbol: str,
        exc: RuntimeError,
        ts: float,
        circuit_state: Any,
        authority: dict[str, object],
    ) -> dict[str, object]:
        """Persistently halt new order submission after an execution/position mismatch."""
        reason = f"execution_state_integrity_failure:{str(symbol).upper()}:{exc}"
        self.portfolio_store.record_execution_integrity_failure(portfolio_state, reason=reason, ts=ts)
        self.portfolio_store.save(portfolio_state)
        try:
            self.control_store.set_pause(paused=True, pause_reason=reason, updated_by="live_trading_controller")
        except Exception as pause_exc:
            self.logger.error("Unable to persist control-plane pause after execution integrity failure: %s", pause_exc)
        self._call_monitor(
            "publish_event",
            "execution_state_integrity_failure",
            {"symbol": str(symbol).upper(), "reason": reason, "ts": float(ts), "fail_closed": True},
        )
        self.logger.error("%s; order submission is now fail-closed", reason)
        return self._runtime_gate(portfolio_state, circuit_state, authority)

    def _record_authority_failure(self, portfolio_state, reason: str) -> None:
        normalized = str(reason or "live_reconciliation_required")
        portfolio_state.authority_mode = "live_exchange_required"
        portfolio_state.reconciliation_status = normalized
        portfolio_state.last_reconciliation_error = normalized
        self.portfolio_store.save(portfolio_state)
        self._current_authority_reason = normalized

    def _verify_startup_live_authority(self) -> None:
        if not self._requires_exchange_authority() or self._startup_authority_verified:
            return
        portfolio_state = self.portfolio_store.load()
        if self.exchange_cfg is None:
            self._startup_authority_verified = True
            return
        if not getattr(self.exchange_cfg, "position_reconciliation_enabled", False):
            self._record_authority_failure(portfolio_state, "live_reconciliation_disabled")
            raise RuntimeError("live_reconciliation_disabled")
        result = self.reconciler.reconcile(portfolio_state, ts=now_utc_ts())
        if not bool(getattr(result, "success", False)):
            reason = str(getattr(result, "error", "") or getattr(result, "status", "") or "live_reconciliation_required")
            self._record_authority_failure(portfolio_state, reason)
            raise RuntimeError(reason)
        self._apply_reconciliation_result(portfolio_state, result, execution_mode=self.execution_mode)
        if not bool(getattr(result, "reconciled", False)):
            reason = str(getattr(result, "status", "") or "live_reconciliation_mismatch")
            self._record_authority_failure(portfolio_state, reason)
            raise RuntimeError(reason)
        portfolio_state.last_reconciliation_error = ""
        self.portfolio_store.save(portfolio_state)
        self._startup_authority_verified = True
        self._current_authority_reason = None

    def _apply_reconciliation_result(self, portfolio_state, result: Any, *, execution_mode: str) -> None:
        if execution_mode != "live":
            self._mark_authority_mode(portfolio_state, reconciled=False)
            portfolio_state.reconciliation_status = "paper_local"
            portfolio_state.last_reconciled_at = None
            self.portfolio_store.save(portfolio_state)
            return

        self.portfolio_store.record_reconciliation(portfolio_state, result, ts=now_utc_ts())
        self.portfolio_store.save(portfolio_state)

    def _publish_model_versions(self) -> None:
        try:
            versions = self.control_store.list_model_versions()
        except Exception as exc:
            self._log_debug("Failed to refresh model versions: %s", exc)
            return
        self._call_monitor("update_model_versions", versions)

    def load_symbols(self) -> list[str]:
        symbols: list[str] = []
        active_universe = load_active_universe_symbols(self.paths, include_db_fallback=False)

        if self.signal_engine is not None:
            configured = self.signal_engine.load_symbols()
            if configured:
                symbols = configured

        if not symbols and self.execution_mode != "live":
            symbols = list(active_universe)

        if not symbols:
            approved = resolve_live_incumbent_requirements(self.paths)
            if approved.get("ready"):
                approved_symbols = sorted({str(item.get("symbol") or "").upper() for item in approved.get("active_incumbents", []) if item.get("symbol")})
                if active_universe:
                    active_set = set(active_universe)
                    approved_symbols = [symbol for symbol in approved_symbols if symbol in active_set]
                symbols = approved_symbols

        if not symbols:
            rows = load_enriched_market_rows(self.paths, limit=64)
            symbols = sorted({str(row.get("symbol") or "").upper() for row in rows if row.get("symbol")})

        if self.execution_mode == "live":
            canary = self._live_canary_limits()
            max_symbols = int(canary.get("max_symbols", 1) or 1)
            symbols = list(symbols)[:max(1, max_symbols)]
        return symbols

    @staticmethod
    def _position_int(portfolio_state, symbol: str) -> int:
        position = getattr(portfolio_state, "positions", {}).get(str(symbol).upper())
        if position is None:
            return 0
        side = str(getattr(position, "side", "") or "").upper()
        if side == "LONG":
            return 1
        if side == "SHORT":
            return -1
        return 0

    def _build_hold_decision_row(
        self,
        *,
        symbol: str,
        interval: str,
        previous_position: int,
        reason: str,
        pause_reason: str = "",
        authority: dict[str, object] | None = None,
        runtime_gate: dict[str, object] | None = None,
        ts: float | None = None,
        blocker_stage: str = "signal",
    ) -> dict[str, object]:
        components: dict[str, object] = {
            "control_plane": {
                "paused": bool(pause_reason) or (runtime_gate or {}).get("paused", False),
                "pause_reason": pause_reason,
            }
        }
        if authority is not None:
            components["authority"] = dict(authority)
        if runtime_gate is not None:
            components["runtime_gate"] = dict(runtime_gate)
            components["control_plane"] = {
                "paused": bool(runtime_gate.get("paused", False)),
                "pause_reason": str(runtime_gate.get("pause_reason") or pause_reason or ""),
            }
        row = {
            "ts": ts or now_utc_ts(),
            "symbol": str(symbol).upper(),
            "interval": str(interval),
            "action": "HOLD",
            "effective_action": "HOLD",
            "approved": False,
            "score": 0.0,
            "reason": str(reason),
            "mode": self.execution_mode,
            "previous_position": int(previous_position),
            "components": components,
        }
        return self._attach_execution_contract(
            row,
            approved=False,
            action="HOLD",
            effective_action="HOLD",
            reason=str(reason),
            stage=blocker_stage,
            execution_result={"status": "not_attempted", "reason": str(reason)},
        )


    @staticmethod
    def _execution_contract(
        *,
        approved: bool,
        action: str,
        effective_action: str,
        reason: str,
        stage: str,
        execution_result: dict[str, Any] | None = None,
        risk_decision: Any | None = None,
    ) -> dict[str, Any]:
        """Return the canonical signal-to-execution blocker contract."""
        action_u = str(action or "HOLD").upper()
        effective_u = str(effective_action or action_u or "HOLD").upper()
        reason_s = str(reason or "")
        result = dict(execution_result or {})
        order_status = str(result.get("status") or "")
        risk_reason = str(getattr(risk_decision, "reason", "") or "") if risk_decision is not None else ""
        attempted = bool(result) and order_status not in {"", "skipped", "not_attempted"}

        if approved and order_status in {"filled", "partially_filled", "submitted"}:
            final_status = "executed" if order_status in {"filled", "partially_filled"} else "submitted"
            blocker = ""
            blocker_stage = "none"
        elif stage == "runtime_gate":
            final_status = "rejected_runtime_gate"
            blocker = reason_s or "runtime_gate_blocked"
            blocker_stage = "runtime_gate"
        elif stage == "freshness":
            final_status = "rejected_data_freshness"
            blocker = reason_s or "stale_market_data"
            blocker_stage = "data"
        elif stage == "price":
            final_status = "rejected_price"
            blocker = reason_s or "price_unavailable"
            blocker_stage = "price"
        elif stage == "risk":
            final_status = "rejected_risk"
            blocker = risk_reason or reason_s or "risk_rejected"
            blocker_stage = "risk"
        elif action_u == "HOLD" and effective_u == "HOLD":
            final_status = "rejected_signal_hold"
            blocker = reason_s or "signal_hold"
            blocker_stage = "signal"
        elif result:
            final_status = "rejected_order_router"
            blocker = str(result.get("reason") or reason_s or order_status or "order_router_rejected")
            blocker_stage = "order_router"
        else:
            final_status = "not_attempted"
            blocker = reason_s or "execution_not_attempted"
            blocker_stage = str(stage or "unknown")

        return {
            "execution_status": final_status,
            "execution_attempted": attempted,
            "execution_blocker_stage": blocker_stage,
            "execution_blocker": blocker,
            "rejection_reason": blocker,
            "order_status": order_status or "not_attempted",
            "approved_for_order": bool(approved),
        }

    @classmethod
    def _attach_execution_contract(
        cls,
        row: dict[str, object],
        *,
        approved: bool = False,
        action: str = "HOLD",
        effective_action: str = "HOLD",
        reason: str = "",
        stage: str = "unknown",
        execution_result: dict[str, Any] | None = None,
        risk_decision: Any | None = None,
    ) -> dict[str, object]:
        contract = cls._execution_contract(
            approved=approved,
            action=action,
            effective_action=effective_action,
            reason=reason,
            stage=stage,
            execution_result=execution_result,
            risk_decision=risk_decision,
        )
        row.update(contract)
        ts = float(row.get("ts") or now_utc_ts())
        decision_id = str(row.get("decision_id") or "")
        if not decision_id:
            decision_id = build_decision_lineage_id(
                symbol=str(row.get("symbol") or ""),
                interval=str(row.get("interval") or ""),
                action=str(action or row.get("action") or "HOLD"),
                reason=str(reason or row.get("reason") or ""),
                score=float(row.get("score") or 0.0),
                previous_position=int(row.get("previous_position") or 0),
                ts=ts,
            )
        row["decision_id"] = decision_id
        components = row.setdefault("components", {})
        if isinstance(components, dict):
            lineage = normalize_lineage_payload(
                components.get("lineage") if isinstance(components.get("lineage"), dict) else {},
                decision_id=decision_id,
                symbol=str(row.get("symbol") or ""),
                interval=str(row.get("interval") or ""),
                ts=ts,
                source="live_trading_controller",
            )
            if execution_result is not None:
                order_id = execution_result.get("order_id")
                client_order_id = execution_result.get("client_order_id")
                if order_id:
                    lineage.setdefault("order_id", str(order_id))
                if client_order_id:
                    lineage.setdefault("client_order_id", str(client_order_id))
            components["lineage"] = lineage
            components["execution_contract"] = dict(contract)
            if execution_result is not None:
                components.setdefault("execution_result", dict(execution_result))
            row["lineage"] = lineage
        return row

    @staticmethod
    def _execution_rejection_distribution(decisions: list[dict[str, object]]) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        blockers: dict[str, int] = {}
        stages: dict[str, int] = {}
        for row in decisions:
            status = str(row.get("execution_status") or "unknown")
            stage = str(row.get("execution_blocker_stage") or "unknown")
            blocker = str(row.get("rejection_reason") or row.get("execution_blocker") or "")
            statuses[status] = statuses.get(status, 0) + 1
            if stage:
                stages[stage] = stages.get(stage, 0) + 1
            if blocker:
                blockers[blocker] = blockers.get(blocker, 0) + 1
        return {"statuses": statuses, "stages": stages, "reasons": blockers}

    @staticmethod
    def _execution_messages(decisions: list[dict[str, object]], *, limit: int = 25) -> list[str]:
        messages: list[str] = []
        for row in decisions[: max(1, int(limit))]:
            symbol = str(row.get("symbol") or "--")
            action = str(row.get("effective_action") or row.get("action") or "--")
            status = str(row.get("execution_status") or "unknown")
            reason = str(row.get("rejection_reason") or row.get("reason") or "")
            risk_breakdown = row.get("risk_breakdown")
            if not isinstance(risk_breakdown, dict):
                components = row.get("components") if isinstance(row.get("components"), dict) else {}
                risk_breakdown = components.get("risk_breakdown") if isinstance(components, dict) else None
            suffix = f": {reason}" if reason else ""
            if isinstance(risk_breakdown, dict) and risk_breakdown:
                distortion = risk_breakdown.get("risk_distortion_fraction")
                limit_value = risk_breakdown.get("max_risk_distortion_fraction")
                clamp = risk_breakdown.get("clamp_reason") or "none"
                try:
                    suffix += f" | distortion={float(distortion):.4f} limit={float(limit_value):.4f} clamp={clamp}"
                except Exception:
                    suffix += f" | distortion={distortion} limit={limit_value} clamp={clamp}"
            messages.append(f"{symbol} {action} {status}" + suffix)
        return messages

    def _force_close_blocked_symbol_positions(
        self,
        portfolio_state: Any,
        latest_prices: dict[str, float],
        *,
        ts: float,
    ) -> dict[str, Any]:
        """Apply Step 6 blocked-symbol close contract.

        Paper/local positions are closed immediately with the current markPrice.
        Live mode remains fail-closed: this method records the failed close
        requirement and surfaces it to monitoring rather than pretending a local
        portfolio mutation closed exchange risk.
        """
        symbols_cfg = getattr(getattr(self.paths, "config", None), "symbols", None)
        if not bool(getattr(symbols_cfg, "close_position_on_block", True)):
            return {"enabled": False, "closed": [], "failed": [], "fail_closed": False}
        if not hasattr(self.paths, "symbol_state_path"):
            return {"enabled": True, "closed": [], "failed": [], "fail_closed": False, "reason": "symbol_state_not_configured"}
        symbol_state = load_symbol_state(self.paths)
        blocked_targets = symbols_requiring_blocked_position_close(symbol_state)
        if not blocked_targets:
            return {"enabled": True, "closed": [], "failed": [], "fail_closed": False}
        report = self.portfolio_store.close_blocked_positions(
            portfolio_state,
            blocked_symbols=blocked_targets,
            price_by_symbol=latest_prices,
            ts=float(ts),
            execution_mode=self.execution_mode,
        )
        symbols_state = symbol_state.setdefault("symbols", {})
        for row in list(report.get("closed") or []):
            symbol = str(row.get("symbol") or "").upper()
            if symbol and isinstance(symbols_state.get(symbol), dict):
                symbols_state[symbol] = record_blocked_position_close_result_entry(
                    symbols_state[symbol],
                    closed=True,
                    reason="paper_forced_close_completed",
                    now_ts=float(ts),
                )
        for row in list(report.get("failed") or []):
            symbol = str(row.get("symbol") or "").upper()
            if symbol and isinstance(symbols_state.get(symbol), dict):
                symbols_state[symbol] = record_blocked_position_close_result_entry(
                    symbols_state[symbol],
                    closed=False,
                    reason=str(row.get("reason") or "blocked_symbol_close_failed"),
                    now_ts=float(ts),
                )
        try:
            from core.symbol_management import save_symbol_state
            save_symbol_state(self.paths, symbol_state)
        except Exception as exc:
            self.logger.warning("Failed to persist blocked-symbol close status: %s", exc)
        if report.get("closed") or report.get("failed"):
            self._call_monitor("publish_event", "blocked_symbol_close", report)
        return report

    def _fast_mark_publish_due(self, *, now_ts: float) -> bool:
        last_publish = float(getattr(self, "_last_fast_mark_publish_ts", 0.0) or 0.0)
        if last_publish <= 0.0:
            return True
        return (float(now_ts) - last_publish) >= self._execution_price_refresh_seconds()

    def _portfolio_execution_summary(self, portfolio_state: Any, *, unrealized_pnl_usd: float = 0.0) -> dict[str, Any]:
        summary_fn = getattr(self.portfolio_store, "execution_summary", None)
        if callable(summary_fn):
            return dict(summary_fn(portfolio_state, unrealized_pnl_usd=unrealized_pnl_usd) or {})
        positions = getattr(portfolio_state, "positions", {}) or {}
        return {
            "open_trades": len(positions),
            "completed_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": float(unrealized_pnl_usd or 0.0),
            "win_rate": 0.0,
        }

    def _portfolio_closed_trade_rows(self, portfolio_state: Any, *, limit: int = 120) -> list[dict[str, Any]]:
        rows_fn = getattr(self.portfolio_store, "closed_trade_rows", None)
        if callable(rows_fn):
            return list(rows_fn(portfolio_state, limit=limit) or [])
        return []

    def _publish_fast_execution_mark_to_market(
        self,
        *,
        portfolio_state: Any,
        latest_prices: dict[str, float],
        latest_price_ts_ms: dict[str, int | None],
        latest_price_sources: dict[str, str],
        decisions: list[dict[str, object]],
        authority: dict[str, object],
        runtime_gate: dict[str, object],
        circuit_state: Any,
        paused: bool,
        pause_reason: str,
        mark_refresh: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Publish fresh open-position marks without waiting for signal evaluation.

        MarketDataController remains the only price producer. This method only
        consumes the runtime price cache, marks open positions, and republishes
        the execution runtime so the dashboard and protective state stay fresh
        while slower RF/GBM/signal work continues.
        """
        if not self._has_open_positions(portfolio_state):
            return dict(mark_refresh or {})

        mark_ts = now_utc_ts()
        if not force and not self._fast_mark_publish_due(now_ts=mark_ts):
            return dict(mark_refresh or {})

        refreshed = self._refresh_open_position_marks(
            portfolio_state,
            latest_prices,
            latest_price_ts_ms,
            latest_price_sources,
            ts=mark_ts,
        )
        merged_refresh = {**dict(mark_refresh or {}), **dict(refreshed or {})}
        pnl_summary = self.portfolio_store.mark_to_market(portfolio_state, latest_prices, ts=mark_ts)
        execution_summary = self._portfolio_execution_summary(
            portfolio_state,
            unrealized_pnl_usd=float((pnl_summary or {}).get("unrealized_pnl_usd", 0.0) or 0.0),
        )
        trade_history_rows = self._portfolio_closed_trade_rows(portfolio_state, limit=120)
        self.portfolio_store.save(portfolio_state)

        open_position_rows = [
            self._build_open_position_row(
                symbol,
                pos,
                latest_prices,
                latest_price_ts_ms,
                latest_price_sources,
                mode=self.execution_mode,
            )
            for symbol, pos in portfolio_state.positions.items()
        ]
        inference_open_positions = {row["symbol"]: row for row in open_position_rows}
        rejection_distribution = self._execution_rejection_distribution(decisions)
        execution_messages = self._execution_messages(decisions)
        pnl_payload = {
            **dict(pnl_summary or {}),
            **dict(execution_summary or {}),
            "paused": paused,
            "pause_reason": pause_reason,
            "execution_mode": self.execution_mode,
            "authority": dict(authority or {}),
            "runtime_gate": dict(runtime_gate or {}),
            "reconciliation_status": getattr(portfolio_state, "reconciliation_status", ""),
            "last_reconciled_at": getattr(portfolio_state, "last_reconciled_at", None),
            "exchange_position_count": getattr(portfolio_state, "exchange_position_count", 0),
            "reconciliation_mismatch_count": getattr(portfolio_state, "reconciliation_mismatch_count", 0),
            "mark_refresh": dict(merged_refresh or {}),
            "execution_rejection_distribution": dict(rejection_distribution),
            "fast_mark_cycle": True,
        }

        if self.execution_mode != "live":
            self._call_monitor(
                "update_paper_execution_runtime",
                execution_mode=self.execution_mode,
                source_name="live_trading_controller",
                open_positions=open_position_rows,
                recent_decisions=decisions[:100],
                recent_messages=execution_messages,
                summary=dict(execution_summary or pnl_payload),
                trade_history=trade_history_rows,
                authority=dict(authority or {}),
                runtime_gate=dict(runtime_gate or {}),
                mark_refresh=dict(merged_refresh or {}),
                fast_mark_cycle=True,
            )
        self._call_monitor(
            "update_inference",
            decisions=decisions[:100],
            open_positions=inference_open_positions,
            pnl_summary=pnl_payload,
            last_error=None,
            risk_summary={
                "circuit_breaker": self._safe_asdict(circuit_state),
                "paused": paused,
                "pause_reason": pause_reason,
                "execution_mode": self.execution_mode,
                "authority": dict(authority or {}),
                "runtime_gate": dict(runtime_gate or {}),
            },
        )
        self._call_monitor(
            "heartbeat",
            "live_trading",
            status="mark_to_market",
            details={
                "positions": len(open_position_rows),
                "execution_mode": self.execution_mode,
                "authority": dict(authority or {}),
                "runtime_gate": dict(runtime_gate or {}),
                "mark_refresh": dict(merged_refresh or {}),
                "fast_mark_cycle": True,
            },
        )
        self._last_fast_mark_publish_ts = float(mark_ts)
        return dict(merged_refresh or {})

    def _live_authority_summary(self, portfolio_state) -> dict[str, object]:
        method = getattr(self.risk_engine, "live_authority_summary", None)
        if callable(method):
            summary = method(portfolio_state)
            if isinstance(summary, dict):
                return summary
            return self._safe_asdict(summary)
        if self._requires_exchange_authority():
            reconciliation_status = str(getattr(portfolio_state, "reconciliation_status", "") or "live_reconciliation_required")
            authoritative = str(getattr(portfolio_state, "authority_mode", "") or "") == "live_exchange_reconciled"
            return {
                "execution_mode": "live",
                "exchange_authority_required": True,
                "authoritative": authoritative,
                "guard_required": not authoritative,
                "reason": "live_reconciled_authority" if authoritative else reconciliation_status,
                "reconciled": authoritative,
            }
        return {
            "execution_mode": "paper",
            "exchange_authority_required": False,
            "authoritative": True,
            "guard_required": False,
            "reason": "paper_local_authority",
            "reconciled": False,
        }

    def _run_reconciliation(self, portfolio_state) -> None:
        if not self._requires_exchange_authority():
            self._mark_authority_mode(portfolio_state, reconciled=False)
            self.portfolio_store.save(portfolio_state)
            return
        if not getattr(self.exchange_cfg, "position_reconciliation_enabled", False):
            portfolio_state.authority_mode = "live_exchange_required"
            portfolio_state.reconciliation_status = "live_reconciliation_disabled"
            portfolio_state.last_reconciled_at = None
            portfolio_state.last_reconciliation_error = "live_reconciliation_disabled"
            self.portfolio_store.save(portfolio_state)
            return
        try:
            result = self.reconciler.reconcile(portfolio_state, ts=now_utc_ts())
        except Exception as exc:
            portfolio_state.authority_mode = "live_exchange_required"
            portfolio_state.reconciliation_status = "error"
            portfolio_state.last_reconciliation_error = str(exc)
            portfolio_state.last_reconciled_at = None
            self.portfolio_store.save(portfolio_state)
            return
        self._apply_reconciliation_result(portfolio_state, result, execution_mode=self.execution_mode)

    def _maybe_reconcile(self, portfolio_state) -> None:
        self._run_reconciliation(portfolio_state)

    def _execution_open_position_snapshot(self, portfolio_state: Any) -> dict[str, dict[str, Any]]:
        """Return current positions from the execution authority for signal evaluation."""
        snapshot_builder = getattr(self.portfolio_store, "open_position_snapshot", None)
        if callable(snapshot_builder):
            snapshot = snapshot_builder(portfolio_state)
            return snapshot if isinstance(snapshot, dict) else {}
        return {
            str(symbol).upper(): {
                "symbol": str(symbol).upper(),
                "side": str(getattr(position, "side", "") or "").upper(),
                "source": "portfolio_state",
                "state_owner": "live_trading_controller",
            }
            for symbol, position in dict(getattr(portfolio_state, "positions", {}) or {}).items()
        }

    def _compute_decision(
        self,
        symbol: str,
        interval: str,
        previous_position: int,
        open_position_snapshot: dict[str, Any],
    ):
        """Compute a decision using the portfolio authority for existing-position context."""
        if self.signal_engine is not None and hasattr(self.signal_engine, "compute_decision"):
            try:
                return self.signal_engine.compute_decision(
                    symbol,
                    open_position_snapshot,
                    paused=False,
                    pause_reason="",
                )
            except Exception as exc:
                self._log_debug("SignalEngine decision path failed for %s; falling back to direct logic path: %s", symbol, exc)
        return self.logic.decide(symbol, interval, previous_position=previous_position)

    def _decision_freshness_gate(self, symbol: str, interval: str, decision: Any) -> dict[str, object]:
        components = dict(getattr(decision, "components", {}) or {})
        freshness = components.get("freshness")
        if isinstance(freshness, dict):
            return dict(freshness)
        latest_ts = None
        try:
            details = self.order_router.recover_open_orders  # noqa: B018 - ensure router exists for init ordering
            del details
        except Exception:
            pass
        try:
            from features.market_loader import latest_market_timestamp
            latest_ts = latest_market_timestamp(self.paths, str(symbol).upper(), str(interval))
        except Exception as exc:
            self._log_debug("Failed to resolve market freshness for %s %s: %s", symbol, interval, exc)
        if latest_ts is None:
            return {"ok": True, "reason": "freshness_unavailable_local"}
        return validate_market_data_freshness(self.paths, symbol=str(symbol).upper(), interval=str(interval), latest_timestamp_ms=latest_ts)




    def _append_decision_journal(self, row: dict[str, object]) -> None:
        path = getattr(self.paths, "decision_journal_path", None)
        if path is None:
            runtime_dir = getattr(self.paths, "runtime_dir", None) or Path(".")
            path = Path(runtime_dir) / "decision_journal.jsonl"
        journal_path = Path(path)
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = dict(row)
        normalized.setdefault("decision_stage", "execution")
        normalized.setdefault("journal_writer", "live_trading_controller")
        normalized.setdefault("state_owner", "live_trading_controller")
        normalized.setdefault("execution_surface", "live_forward_execution" if self.execution_mode == "live" else "paper_forward_execution")
        import json
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized, sort_keys=True) + "\n")

    def _compute_shadow_candidate_decision(self, symbol: str, interval: str, previous_position: int, incumbent_decision: Any | None = None) -> dict[str, object]:
        try:
            shadow = self.logic.decide(symbol, interval, previous_position=previous_position, use_candidate_artifacts=True)
        except TypeError:
            return {"available": False, "reason": "shadow_contract_unavailable"}
        except Exception as exc:
            return {"available": False, "reason": f"shadow_error:{exc}"}

        available = str(getattr(shadow, "reason", "") or "") != "candidate_artifact_missing"
        action = str(getattr(shadow, "action", "HOLD") or "HOLD")
        score = float(getattr(shadow, "score", 0.0) or 0.0)
        reason = str(getattr(shadow, "reason", "") or "")
        components = dict(getattr(shadow, "components", {}) or {})
        incumbent_action = str(getattr(incumbent_decision, "action", "") or "") if incumbent_decision is not None else ""
        different_action = bool(available and incumbent_action and action != incumbent_action)
        return {
            "available": available,
            "action": action,
            "score": score,
            "reason": reason,
            "different_action": different_action,
            "lineage": dict((components.get("lineage") if isinstance(components.get("lineage"), dict) else {})),
            "components": components,
        }

    def loop_once(self) -> dict[str, Any]:
        self._verify_startup_live_authority()
        portfolio_state = self.portfolio_store.load()
        self._maybe_reconcile(portfolio_state)
        portfolio_state = self.portfolio_store.load()
        authority = self._live_authority_summary(portfolio_state)
        circuit_state = self.risk_engine.load_circuit_breaker()
        runtime_gate = self._runtime_gate(portfolio_state, circuit_state, authority)
        paused = bool(runtime_gate.get("paused", False))
        pause_reason = str(runtime_gate.get("pause_reason") or "")
        blocked = bool(runtime_gate.get("blocked", False))
        block_reason = str(runtime_gate.get("reason") or "")
        status = str(runtime_gate.get("status") or ("paused" if paused else "running"))

        symbols = self.load_symbols()
        decisions: list[dict[str, object]] = []
        signal_rows: list[dict[str, object]] = []
        ts = now_utc_ts()
        latest_prices: dict[str, float] = {}

        price_snapshot_status = self._runtime_price_snapshot_status(symbols)
        if self._should_wait_for_market_prices(symbols, price_snapshot_status, portfolio_state):
            wait_reason = str(price_snapshot_status.get("reason") or "market_price_snapshot_not_ready")
            self._call_monitor(
                "heartbeat",
                "live_trading",
                status="waiting_for_market_prices",
                details={
                    "symbols": len(symbols),
                    "priced_symbols": int(price_snapshot_status.get("priced_symbols") or 0),
                    "missing_symbols": list(price_snapshot_status.get("missing_symbols") or []),
                    "stale_symbols": list(price_snapshot_status.get("stale_symbols") or []),
                    "max_age_seconds": price_snapshot_status.get("max_age_seconds"),
                    "reason": wait_reason,
                    "execution_mode": self.execution_mode,
                    "authority": authority,
                    "runtime_gate": dict(runtime_gate),
                },
            )
            self._call_monitor("update_control_plane", self._control_plane_summary())
            return {
                "status": "waiting_for_market_prices",
                "symbols": len(symbols),
                "decisions": 0,
                "paused": paused,
                "pause_reason": pause_reason,
                "authority": authority,
                "runtime_gate": dict(runtime_gate),
                "price_snapshot": price_snapshot_status,
            }

        latest_prices.update(dict(price_snapshot_status.get("prices") or {}))
        latest_price_ts_ms = dict(price_snapshot_status.get("price_ts_ms") or {})
        latest_price_sources = dict(price_snapshot_status.get("sources") or {})
        mark_refresh = self._refresh_open_position_marks(
            portfolio_state,
            latest_prices,
            latest_price_ts_ms,
            latest_price_sources,
            ts=ts,
        )
        blocked_close_report = self._force_close_blocked_symbol_positions(portfolio_state, latest_prices, ts=ts)
        if blocked_close_report.get("closed"):
            self.portfolio_store.save(portfolio_state)
        open_positions_summary = {
            symbol: self._build_open_position_row(
                symbol,
                pos,
                latest_prices,
                latest_price_ts_ms,
                latest_price_sources,
                mode=self.execution_mode,
            )
            for symbol, pos in portfolio_state.positions.items()
        }

        for open_symbol, open_pos in list(portfolio_state.positions.items()):
            normalized_open_symbol = str(open_symbol or "").upper()
            mark_price = self._coerce_positive_price(latest_prices.get(normalized_open_symbol))
            protective_action, protective_reason = self._protective_exit_action(open_pos, mark_price)
            if not protective_action or mark_price is None:
                continue
            previous_position = self._position_int(portfolio_state, normalized_open_symbol)
            try:
                execution_result = self._route_protective_exit(
                    symbol=normalized_open_symbol,
                    pos=open_pos,
                    action=protective_action,
                    reason=protective_reason,
                    mark_price=mark_price,
                    ts=ts,
                    previous_position=previous_position,
                    portfolio_state=portfolio_state,
                )
                decision_row = {
                    "ts": ts,
                    "symbol": normalized_open_symbol,
                    "interval": str(self.paths.config.frames.anchor_interval),
                    "action": protective_action,
                    "effective_action": protective_action,
                    "approved": bool(execution_result.get("status") in {"filled", "partially_filled", "submitted"}),
                    "score": 1.0,
                    "reason": protective_reason,
                    "mode": self.execution_mode,
                    "previous_position": int(previous_position),
                    "components": {
                        "latest_price": mark_price,
                        "price": mark_price,
                        "reference_price": mark_price,
                        "price_source": latest_price_sources.get(normalized_open_symbol, "runtime_market_price"),
                        "price_ts_ms": latest_price_ts_ms.get(normalized_open_symbol),
                        "protective_exit": True,
                        "execution_result": execution_result,
                        "authority": authority,
                        "runtime_gate": dict(runtime_gate),
                    },
                }
                decision_row = self._attach_execution_contract(
                    decision_row,
                    approved=bool(execution_result.get("status") in {"filled", "partially_filled", "submitted"}),
                    action=protective_action,
                    effective_action=protective_action,
                    reason=protective_reason,
                    stage="protective_exit",
                    execution_result=execution_result,
                )
                decisions.append(decision_row)
                self._append_decision_journal(decision_row)
                signal_rows.append({
                    "symbol": normalized_open_symbol,
                    "action": protective_action,
                    "mode": self.execution_mode,
                    "reason": protective_reason,
                    "ts": ts,
                    "raw": f"{normalized_open_symbol} | {protective_action} | {protective_reason}",
                })
            except Exception as exc:
                self.logger.exception("Protective exit failed for %s", normalized_open_symbol)
                error_row = {
                    "ts": ts,
                    "symbol": normalized_open_symbol,
                    "interval": str(self.paths.config.frames.anchor_interval),
                    "action": "HOLD",
                    "effective_action": "HOLD",
                    "approved": False,
                    "score": 0.0,
                    "reason": f"protective_exit_error:{exc}",
                    "mode": self.execution_mode,
                    "previous_position": int(previous_position),
                    "components": {"protective_exit": True, "error": str(exc)},
                }
                decisions.append(self._attach_execution_contract(
                    error_row,
                    approved=False,
                    action="HOLD",
                    effective_action="HOLD",
                    reason=f"protective_exit_error:{exc}",
                    stage="protective_exit",
                    execution_result={"status": "skipped", "reason": f"protective_exit_error:{exc}"},
                ))

        if self._has_open_positions(portfolio_state):
            mark_refresh = self._publish_fast_execution_mark_to_market(
                portfolio_state=portfolio_state,
                latest_prices=latest_prices,
                latest_price_ts_ms=latest_price_ts_ms,
                latest_price_sources=latest_price_sources,
                decisions=decisions,
                authority=authority,
                runtime_gate=runtime_gate,
                circuit_state=circuit_state,
                paused=paused,
                pause_reason=pause_reason,
                mark_refresh=mark_refresh,
                force=True,
            )

        if not self._should_run_full_signal_cycle(ts=ts, portfolio_state=portfolio_state):
            pnl_summary = self.portfolio_store.mark_to_market(portfolio_state, latest_prices, ts=ts)
            execution_summary = self._portfolio_execution_summary(
                portfolio_state,
                unrealized_pnl_usd=float((pnl_summary or {}).get("unrealized_pnl_usd", 0.0) or 0.0),
            )
            trade_history_rows = self._portfolio_closed_trade_rows(portfolio_state, limit=120)
            self.portfolio_store.save(portfolio_state)
            rejection_distribution = self._execution_rejection_distribution(decisions)
            execution_messages = self._execution_messages(decisions)
            inference_open_positions = {
                symbol: self._build_open_position_row(
                    symbol,
                    pos,
                    latest_prices,
                    latest_price_ts_ms,
                    latest_price_sources,
                    mode=self.execution_mode,
                )
                for symbol, pos in portfolio_state.positions.items()
            }
            inference_payload = {
                "decisions": decisions[:100],
                "open_positions": inference_open_positions,
                "pnl_summary": {
                    **dict(pnl_summary or {}),
                    **dict(execution_summary or {}),
                    "paused": paused,
                    "pause_reason": pause_reason,
                    "execution_mode": self.execution_mode,
                    "authority": authority,
                    "runtime_gate": dict(runtime_gate),
                    "reconciliation_status": getattr(portfolio_state, "reconciliation_status", ""),
                    "last_reconciled_at": getattr(portfolio_state, "last_reconciled_at", None),
                    "exchange_position_count": getattr(portfolio_state, "exchange_position_count", 0),
                    "reconciliation_mismatch_count": getattr(portfolio_state, "reconciliation_mismatch_count", 0),
                    "mark_refresh": dict(mark_refresh or {}),
                    "execution_rejection_distribution": dict(rejection_distribution),
                    "mark_only_cycle": True,
                },
                "last_error": None,
                "risk_summary": {
                    "circuit_breaker": self._safe_asdict(circuit_state),
                    "paused": paused,
                    "pause_reason": pause_reason,
                    "execution_mode": self.execution_mode,
                    "authority": authority,
                    "runtime_gate": dict(runtime_gate),
                },
            }
            if self.execution_mode != "live":
                self._call_monitor(
                    "update_paper_execution_runtime",
                    execution_mode=self.execution_mode,
                    source_name="live_trading_controller",
                    open_positions=list(inference_open_positions.values()),
                    recent_decisions=decisions[:100],
                    recent_messages=execution_messages,
                    summary=dict(execution_summary or inference_payload["pnl_summary"]),
                    trade_history=trade_history_rows,
                    authority=dict(authority or {}),
                    runtime_gate=dict(runtime_gate or {}),
                )
            self._call_monitor("update_inference", **inference_payload)
            self._call_monitor("update_control_plane", self._control_plane_summary())
            self._call_monitor(
                "heartbeat",
                "live_trading",
                status="mark_to_market",
                details={
                    "symbols": len(symbols),
                    "paused": paused,
                    "pause_reason": pause_reason,
                    "execution_mode": self.execution_mode,
                    "authority": authority,
                    "runtime_gate": dict(runtime_gate),
                    "positions": len(portfolio_state.positions),
                    "mark_refresh": dict(mark_refresh or {}),
                    "execution_rejection_distribution": dict(rejection_distribution),
                },
            )
            return {
                "status": "mark_to_market",
                "symbols": len(symbols),
                "decisions": len(decisions),
                "paused": paused,
                "pause_reason": pause_reason,
                "authority": authority,
                "runtime_gate": dict(runtime_gate),
                "execution_rejection_distribution": dict(rejection_distribution),
            }

        self._last_full_signal_cycle_ts = float(ts)

        for symbol in symbols:
            interval = str(self.paths.config.frames.anchor_interval)
            previous_position = self._position_int(portfolio_state, symbol)

            if self._has_open_positions(portfolio_state) and self._fast_mark_publish_due(now_ts=now_utc_ts()):
                mark_refresh = self._publish_fast_execution_mark_to_market(
                    portfolio_state=portfolio_state,
                    latest_prices=latest_prices,
                    latest_price_ts_ms=latest_price_ts_ms,
                    latest_price_sources=latest_price_sources,
                    decisions=decisions,
                    authority=authority,
                    runtime_gate=runtime_gate,
                    circuit_state=circuit_state,
                    paused=paused,
                    pause_reason=pause_reason,
                    mark_refresh=mark_refresh,
                )

            try:
                if blocked:
                    decision_row = self._build_hold_decision_row(
                        symbol=symbol,
                        interval=interval,
                        previous_position=previous_position,
                        reason=block_reason or ("control_plane_paused" if paused else "runtime_gate_blocked"),
                        pause_reason=pause_reason,
                        authority=authority,
                        runtime_gate=runtime_gate,
                        ts=ts,
                        blocker_stage="runtime_gate",
                    )
                    decisions.append(decision_row)
                    self._append_decision_journal(decision_row)
                    signal_rows.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": "HOLD",
                            "mode": self.execution_mode,
                            "reason": decision_row["reason"],
                            "ts": ts,
                            "raw": f"{str(symbol).upper()} | HOLD | {decision_row['reason']}",
                        }
                    )
                    continue

                open_position_snapshot = self._execution_open_position_snapshot(portfolio_state)
                decision = self._compute_decision(symbol, interval, previous_position, open_position_snapshot)
                decision_components = dict(getattr(decision, "components", {}) or {})
                decision_id = build_decision_lineage_id(
                    symbol=str(symbol).upper(),
                    interval=str(getattr(decision, "interval", interval)),
                    action=str(getattr(decision, "action", "HOLD")),
                    reason=str(getattr(decision, "reason", "")),
                    score=float(getattr(decision, "score", 0.0) or 0.0),
                    previous_position=int(previous_position),
                    ts=ts,
                )
                decision_components["lineage"] = normalize_lineage_payload(
                    decision_components.get("lineage") if isinstance(decision_components.get("lineage"), dict) else {},
                    decision_id=decision_id,
                    symbol=str(symbol).upper(),
                    interval=str(getattr(decision, "interval", interval)),
                    ts=ts,
                    source="live_trading_controller",
                )
                freshness_gate = self._decision_freshness_gate(symbol, interval, decision)
                if not bool(freshness_gate.get("ok", False)):
                    decision_row = self._build_hold_decision_row(
                        symbol=symbol,
                        interval=getattr(decision, "interval", interval),
                        previous_position=previous_position,
                        reason=str(freshness_gate.get("reason") or "stale_market_data"),
                        authority=authority,
                        runtime_gate=runtime_gate,
                        ts=ts,
                        blocker_stage="freshness",
                    )
                    decision_row.setdefault("components", {})["freshness"] = dict(freshness_gate)
                    decisions.append(decision_row)
                    self._append_decision_journal(decision_row)
                    signal_rows.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": "HOLD",
                            "mode": self.execution_mode,
                            "reason": decision_row["reason"],
                            "ts": ts,
                            "raw": f"{str(symbol).upper()} | HOLD | {decision_row['reason']}",
                        }
                    )
                    continue
                decision_components["freshness"] = dict(freshness_gate)
                shadow_candidate = self._compute_shadow_candidate_decision(symbol, interval, previous_position, decision)
                decision_components["shadow_candidate"] = shadow_candidate
                latest_price, decision_components = self._normalize_decision_price(symbol, decision_components)
                if latest_price is None or latest_price <= 0.0:
                    decision_row = self._build_hold_decision_row(
                        symbol=symbol,
                        interval=decision.interval,
                        previous_position=previous_position,
                        reason=str(decision_components.get("price_blocker") or "price_unavailable"),
                        authority=authority,
                        runtime_gate=runtime_gate,
                        ts=ts,
                        blocker_stage="price",
                    )
                    decision_row.setdefault("components", {}).update(decision_components)
                    decisions.append(decision_row)
                    self._append_decision_journal(decision_row)
                    signal_rows.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": "HOLD",
                            "mode": self.execution_mode,
                            "reason": str(decision_components.get("price_blocker") or "price_unavailable"),
                            "ts": ts,
                            "raw": f"{str(symbol).upper()} | HOLD | {str(decision_components.get('price_blocker') or 'price_unavailable')}",
                        }
                    )
                    continue

                execution_price_gate = self._execution_order_price_gate(
                    symbol,
                    str(getattr(decision, "action", "HOLD") or "HOLD"),
                    now_ms=int(float(now_utc_ts()) * 1000),
                )
                decision_components["execution_price_gate"] = dict(execution_price_gate)
                if bool(execution_price_gate.get("required", False)):
                    if not bool(execution_price_gate.get("ok", False)):
                        gate_reason = str(execution_price_gate.get("reason") or "execution_price_stale")
                        decision_row = self._build_hold_decision_row(
                            symbol=symbol,
                            interval=decision.interval,
                            previous_position=previous_position,
                            reason=gate_reason,
                            authority=authority,
                            runtime_gate=runtime_gate,
                            ts=ts,
                            blocker_stage="price",
                        )
                        decision_row.setdefault("components", {}).update(decision_components)
                        decisions.append(decision_row)
                        self._append_decision_journal(decision_row)
                        signal_rows.append(
                            {
                                "symbol": str(symbol).upper(),
                                "action": "HOLD",
                                "mode": self.execution_mode,
                                "reason": gate_reason,
                                "ts": ts,
                                "raw": f"{str(symbol).upper()} | HOLD | {gate_reason}",
                            }
                        )
                        continue
                    latest_price = float(execution_price_gate["price"])
                    decision_components.update(
                        {
                            "price": latest_price,
                            "latest_price": latest_price,
                            "reference_price": latest_price,
                            "price_source": execution_price_gate.get("price_source"),
                            "price_ts_ms": execution_price_gate.get("price_ts_ms"),
                            "price_fresh": True,
                            "price_age_seconds": execution_price_gate.get("price_age_seconds"),
                            "price_max_age_seconds": execution_price_gate.get("max_age_seconds"),
                        }
                    )
                    try:
                        decision.components = dict(decision_components)
                    except Exception:
                        pass

                latest_prices[str(symbol).upper()] = latest_price
                risk_decision = self.risk_engine.evaluate_trade(decision=decision, portfolio_state=portfolio_state, ts=ts)
                approved = bool(getattr(risk_decision, "approved", False))
                effective_action = str(getattr(risk_decision, "final_action", "HOLD") or "HOLD").upper()
                risk_decision_payload = self._safe_asdict(risk_decision)
                risk_components = dict(risk_decision_payload.get("components") or {}) if isinstance(risk_decision_payload, dict) else {}
                risk_breakdown = dict(risk_components.get("risk_breakdown") or risk_components.get("risk_rejection") or {})
                execution_payload = {
                    **decision_components,
                    "signal_action": decision.action,
                    "signal_reason": decision.reason,
                    "risk_decision": risk_decision_payload,
                    "risk_breakdown": risk_breakdown,
                    "authority": authority,
                    "runtime_gate": dict(runtime_gate),
                    "control_plane": {
                        "paused": False,
                        "pause_reason": "",
                    },
                }

                if approved:
                    entry_lineage = self._extend_trade_entry_lineage(
                        base_lineage=dict(decision_components.get("lineage") or {}),
                        decision=decision,
                        decision_components=decision_components,
                        risk_decision_payload=risk_decision_payload,
                        approved=approved,
                    )
                    decision_components["lineage"] = entry_lineage
                    intent = OrderIntent(
                        symbol=str(symbol).upper(),
                        action=effective_action,
                        side=self._order_side_for_action(effective_action, getattr(portfolio_state.positions.get(str(symbol).upper()), "side", "")),
                        quantity=float(getattr(getattr(risk_decision, "position_plan", None), "quantity", 0.0) or 0.0),
                        order_type="MARKET",
                        signal_reason=str(decision.reason or ""),
                        metadata={
                            "reference_price": latest_price,
                            "mode": self.execution_mode,
                            "position_plan": self._safe_asdict(getattr(risk_decision, "position_plan", None)),
                            "lineage": entry_lineage,
                        },
                        lineage=entry_lineage,
                    )
                    execution_result = self.order_router.route_intent(intent, ts=ts, previous_position=previous_position)
                    fill_payload = dict(execution_result.get("fill") or {})
                    fill_price = float(fill_payload.get("price") or latest_price or 0.0)
                    if execution_result.get("status") in {"filled", "partially_filled"} and fill_price > 0.0:
                        self.portfolio_store.apply_execution(
                            portfolio_state,
                            symbol=symbol,
                            action=effective_action,
                            price=fill_price,
                            ts=ts,
                            position_plan={**self._safe_asdict(getattr(risk_decision, "position_plan", None)), "lineage": dict(decision_components.get("lineage") or {}), "execution_fill": fill_payload},
                        )
                        self.portfolio_store.save(portfolio_state)
                else:
                    execution_result = {
                        "status": "skipped",
                        "reason": str(getattr(risk_decision, "reason", "") or decision.reason),
                    }
                execution_payload["execution_result"] = execution_result

                decision_reason = str(decision.reason or getattr(risk_decision, "reason", "") or "")
                decision_row = {
                    "ts": ts,
                    "decision_id": decision_id,
                    "lineage": dict(decision_components.get("lineage") or {}),
                    "symbol": str(symbol).upper(),
                    "interval": decision.interval,
                    "action": decision.action,
                    "effective_action": effective_action,
                    "approved": approved,
                    "score": float(decision.score),
                    "reason": decision_reason,
                    "mode": self.execution_mode,
                    "previous_position": int(previous_position),
                    "components": execution_payload,
                }
                if risk_breakdown:
                    decision_row["risk_breakdown"] = dict(risk_breakdown)
                execution_stage = "risk" if not approved else "order_router"
                decision_row = self._attach_execution_contract(
                    decision_row,
                    approved=approved,
                    action=str(decision.action or "HOLD"),
                    effective_action=effective_action,
                    reason=decision_reason,
                    stage=execution_stage,
                    execution_result=execution_result,
                    risk_decision=risk_decision,
                )
                decisions.append(decision_row)
                self._append_decision_journal(decision_row)
                signal_rows.append(
                    {
                        "symbol": str(symbol).upper(),
                        "action": effective_action,
                        "mode": self.execution_mode,
                        "reason": decision_reason,
                        "ts": ts,
                        "raw": f"{str(symbol).upper()} | {effective_action} | {decision_reason}",
                    }
                )
            except Exception as exc:
                if exc.__class__.__name__ == "PositionStateInvariantError":
                    runtime_gate = self._block_on_execution_integrity_failure(
                        portfolio_state,
                        symbol=symbol,
                        exc=exc,
                        ts=ts,
                        circuit_state=circuit_state,
                        authority=authority,
                    )
                    paused = bool(runtime_gate.get("paused", True))
                    pause_reason = str(runtime_gate.get("reason") or "execution_state_integrity_failure")
                    blocked = True
                    block_reason = pause_reason
                    error_row = self._build_hold_decision_row(
                            symbol=symbol,
                            interval=interval,
                            previous_position=previous_position,
                            reason=pause_reason,
                            authority=authority,
                            runtime_gate=runtime_gate,
                            ts=ts,
                            blocker_stage="runtime_gate",
                        )
                    decisions.append(error_row)
                    self._append_decision_journal(error_row)
                    signal_rows.append(
                        {
                            "symbol": str(symbol).upper(),
                            "action": "HOLD",
                            "mode": self.execution_mode,
                            "reason": pause_reason,
                            "ts": ts,
                            "raw": f"{str(symbol).upper()} | HOLD | {pause_reason}",
                        }
                    )
                    continue
                self.logger.exception("Live loop failed for %s", symbol)
                error_row = self._build_hold_decision_row(
                        symbol=symbol,
                        interval=interval,
                        previous_position=previous_position,
                        reason=f"live_loop_error:{exc}",
                        authority=authority,
                        runtime_gate=runtime_gate,
                        ts=ts,
                        blocker_stage="controller_error",
                    )
                decisions.append(error_row)
                self._append_decision_journal(error_row)
                signal_rows.append(
                    {
                        "symbol": str(symbol).upper(),
                        "action": "HOLD",
                        "mode": self.execution_mode,
                        "reason": f"live_loop_error:{exc}",
                        "ts": ts,
                        "raw": f"{str(symbol).upper()} | HOLD | live_loop_error:{exc}",
                    }
                )

        pnl_summary = self.portfolio_store.mark_to_market(portfolio_state, latest_prices, ts=ts)
        execution_summary = self._portfolio_execution_summary(
            portfolio_state,
            unrealized_pnl_usd=float((pnl_summary or {}).get("unrealized_pnl_usd", 0.0) or 0.0),
        )
        trade_history_rows = self._portfolio_closed_trade_rows(portfolio_state, limit=120)
        self.portfolio_store.save(portfolio_state)
        rejection_distribution = self._execution_rejection_distribution(decisions)
        execution_messages = self._execution_messages(decisions)

        inference_payload = {
            "decisions": decisions[:100],
            "open_positions": {
                symbol: self._build_open_position_row(
                    symbol,
                    pos,
                    latest_prices,
                    latest_price_ts_ms,
                    latest_price_sources,
                    mode=self.execution_mode,
                )
                for symbol, pos in portfolio_state.positions.items()
            },
            "pnl_summary": {
                **dict(pnl_summary or {}),
                **dict(execution_summary or {}),
                "paused": paused,
                "pause_reason": pause_reason,
                "execution_mode": self.execution_mode,
                "authority": authority,
                "runtime_gate": dict(runtime_gate),
                "reconciliation_status": getattr(portfolio_state, "reconciliation_status", ""),
                "last_reconciled_at": getattr(portfolio_state, "last_reconciled_at", None),
                "exchange_position_count": getattr(portfolio_state, "exchange_position_count", 0),
                "reconciliation_mismatch_count": getattr(portfolio_state, "reconciliation_mismatch_count", 0),
                "mark_refresh": dict(mark_refresh or {}),
                "execution_rejection_distribution": dict(rejection_distribution),
            },
            "last_error": None,
            "risk_summary": {
                "circuit_breaker": self._safe_asdict(circuit_state),
                "paused": paused,
                "pause_reason": pause_reason,
                "execution_mode": self.execution_mode,
                "authority": authority,
                "runtime_gate": dict(runtime_gate),
            },
        }
        if self.execution_mode != "live":
            self._call_monitor(
                "update_paper_execution_runtime",
                execution_mode=self.execution_mode,
                source_name="live_trading_controller",
                open_positions=[
                    self._build_open_position_row(
                        symbol,
                        pos,
                        latest_prices,
                        latest_price_ts_ms,
                        latest_price_sources,
                        mode=self.execution_mode,
                    )
                    for symbol, pos in portfolio_state.positions.items()
                ],
                recent_decisions=decisions[:100],
                recent_messages=execution_messages,
                summary=dict(execution_summary or inference_payload["pnl_summary"]),
                trade_history=trade_history_rows,
                authority=dict(authority or {}),
                runtime_gate=dict(runtime_gate or {}),
            )

        self._call_monitor("update_inference", **inference_payload)
        self._call_monitor("update_control_plane", self._control_plane_summary())
        self._call_monitor(
            "heartbeat",
            "live_trading",
            status=status,
            details={
                "symbols": len(symbols),
                "paused": paused,
                "pause_reason": pause_reason,
                "execution_mode": self.execution_mode,
                "authority": authority,
                "runtime_gate": dict(runtime_gate),
                "positions": len(open_positions_summary),
                "execution_rejection_distribution": dict(rejection_distribution),
            },
        )
        self._publish_model_versions()

        return {
            "status": status,
            "symbols": len(symbols),
            "decisions": len(decisions),
            "paused": paused,
            "pause_reason": pause_reason,
            "authority": authority,
            "runtime_gate": dict(runtime_gate),
            "execution_rejection_distribution": dict(rejection_distribution),
        }

    def run_forever(self, stop_event=None) -> None:
        import threading
        import time

        stop_event = stop_event or threading.Event()
        self.logger.info(
            "LiveTradingController started",
        )
        try:
            self._call_monitor("heartbeat", "live_trading", status="running", details={"phase": "startup", "execution_mode": self.execution_mode})
            while not stop_event.is_set():
                summary = self.loop_once()
                status = str((summary or {}).get("status") or "running")
                details = dict(summary or {})
                self._call_monitor("heartbeat", "live_trading", status=status, details=details)
                wait_seconds = self._next_loop_wait_seconds()
                if stop_event.wait(wait_seconds):
                    break
        except KeyboardInterrupt:
            self.logger.warning("LiveTradingController interrupted by user")
        except Exception as exc:
            self.logger.exception("LiveTradingController crashed")
            self._call_monitor("controller_error", "live_trading", str(exc))
            self._call_monitor(
                "publish_event",
                level="ERROR",
                component="live_trading",
                category="controller_crashed",
                message="Live trading controller crashed",
                data={"error": str(exc)},
                alert=True,
            )
            raise
        finally:
            self._call_monitor("flush")
