from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from execution.circuit_breaker import CircuitBreakerState, CircuitBreakerStore
from execution.exposure_model import summarize_portfolio_exposure
from execution.portfolio_state import PortfolioState, PortfolioStateStore
from execution.portfolio_allocator import allocate_position_plan
from execution.position_sizing import (
    AdaptiveStopConfig,
    PositionPlan,
    compute_adaptive_stop_multipliers,
    compute_position_plan,
)
from execution.symbol_expectancy_guard import SymbolExpectancyGuardConfig, build_symbol_expectancy_snapshot


@dataclass(slots=True)
class RiskDecision:
    """Outcome of runtime portfolio and circuit-breaker checks."""

    approved: bool
    final_action: str
    reason: str
    position_plan: PositionPlan | None = None
    components: dict[str, Any] = field(default_factory=dict)


class RiskEngine:
    """Evaluate execution decisions against portfolio limits and mode authority."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.cfg = paths.config.risk
        self.runtime_cfg = getattr(paths.config, "runtime", None)
        self.exchange_cfg = getattr(paths.config, "exchange", None)
        self.portfolio_store = PortfolioStateStore(paths.portfolio_state_path, starting_equity_usd=self.cfg.account_balance_usd)
        self.circuit_store = CircuitBreakerStore(paths.circuit_breaker_state_path)

    @staticmethod
    def _day_key(ts: float) -> str:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _direction(action: str) -> str:
        return "LONG" if "LONG" in str(action).upper() else "SHORT"

    @property
    def execution_mode(self) -> str:
        return str(getattr(self.runtime_cfg, "execution_mode", "paper") or "paper").strip().lower()

    @property
    def allow_unreconciled_live_mode(self) -> bool:
        return bool(getattr(self.runtime_cfg, "allow_unreconciled_live_mode", False))

    @property
    def requires_exchange_authority(self) -> bool:
        return self.execution_mode == "live"

    def _live_canary_limits(self) -> dict[str, Any]:
        return {
            "enabled": bool(getattr(self.runtime_cfg, "live_canary_enabled", True)),
            "max_symbols": int(getattr(self.runtime_cfg, "live_canary_max_symbols", 1) or 1),
            "max_notional_usd": float(getattr(self.runtime_cfg, "live_canary_max_notional_usd", 50.0) or 0.0),
            "max_concurrent_positions": int(getattr(self.runtime_cfg, "live_canary_max_concurrent_positions", 1) or 1),
            "require_testnet": bool(getattr(self.runtime_cfg, "live_canary_require_testnet", True)),
            "use_testnet": bool(getattr(self.exchange_cfg, "use_testnet", False)),
        }

    def live_authority_summary(self, portfolio_state: PortfolioState) -> dict[str, Any]:
        """Describe whether the current runtime has sufficient authority for execution."""
        mode = self.execution_mode
        if not self.requires_exchange_authority:
            return {
                "execution_mode": mode,
                "exchange_authority_required": False,
                "authoritative": True,
                "guard_required": False,
                "reason": "paper_local_authority",
                "reconciled": False,
            }

        exchange_cfg = self.exchange_cfg
        reconciliation_required = bool(getattr(exchange_cfg, "require_reconciliation_for_live", True))
        reconciliation_enabled = bool(getattr(exchange_cfg, "position_reconciliation_enabled", False))
        max_age_sec = int(getattr(exchange_cfg, "max_reconciliation_age_sec", 30) or 30)

        if reconciliation_required and not reconciliation_enabled:
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_reconciliation_required",
                "reconciled": False,
            }

        status = str(getattr(portfolio_state, "reconciliation_status", "") or "")
        last_reconciled_at = getattr(portfolio_state, "last_reconciled_at", None)
        now_ts = datetime.now(timezone.utc).timestamp()

        if status == "reconciled" and last_reconciled_at is not None:
            age_sec = max(0.0, float(now_ts) - float(last_reconciled_at))
            if age_sec > float(max_age_sec):
                return {
                    "execution_mode": mode,
                    "exchange_authority_required": True,
                    "authoritative": False,
                    "guard_required": True,
                    "reason": "live_reconciliation_stale",
                    "reconciled": False,
                    "age_sec": age_sec,
                }
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": True,
                "guard_required": False,
                "reason": "live_reconciled_authority",
                "reconciled": True,
                "age_sec": age_sec,
            }

        if status == "disabled":
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_reconciliation_disabled",
                "reconciled": False,
            }
        if status == "config_error":
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_reconciliation_config_error",
                "reconciled": False,
            }
        if status == "mismatch":
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_reconciliation_mismatch",
                "reconciled": False,
            }
        if status == "error":
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_reconciliation_error",
                "reconciled": False,
            }
        if self.allow_unreconciled_live_mode:
            return {
                "execution_mode": mode,
                "exchange_authority_required": True,
                "authoritative": False,
                "guard_required": True,
                "reason": "live_unreconciled_override",
                "reconciled": False,
            }
        return {
            "execution_mode": mode,
            "exchange_authority_required": True,
            "authoritative": False,
            "guard_required": True,
            "reason": "live_reconciliation_required",
            "reconciled": False,
        }

    def load_portfolio(self) -> PortfolioState:
        return self.portfolio_store.load()

    def save_portfolio(self, state: PortfolioState) -> None:
        self.portfolio_store.save(state)

    def load_circuit_breaker(self) -> CircuitBreakerState:
        state = self.circuit_store.load()
        if state.peak_equity_usd <= 0:
            state.peak_equity_usd = float(self.cfg.account_balance_usd)
        if state.current_equity_usd <= 0:
            state.current_equity_usd = float(self.cfg.account_balance_usd)
        return state

    def refresh_circuit_breaker(self, portfolio_state: PortfolioState, *, pnl_summary: dict[str, Any], ts: float) -> CircuitBreakerState:
        """Update circuit-breaker state using the latest portfolio equity snapshot."""
        state = self.load_circuit_breaker()
        current_equity = float(pnl_summary.get("equity_usd") or portfolio_state.last_equity_usd or self.cfg.account_balance_usd)
        peak_equity = max(float(state.peak_equity_usd or 0.0), float(portfolio_state.peak_equity_usd or 0.0), current_equity, float(self.cfg.account_balance_usd))
        drawdown_fraction = 0.0
        if peak_equity > 0:
            drawdown_fraction = max(0.0, (peak_equity - current_equity) / peak_equity)
        state.last_checked_at = float(ts)
        state.peak_equity_usd = float(peak_equity)
        state.current_equity_usd = float(current_equity)
        state.drawdown_fraction = float(drawdown_fraction)
        state.daily_realized_pnl_usd = float(pnl_summary.get("daily_realized_pnl_usd") or portfolio_state.daily_realized_pnl_usd)
        state.day_key = self._day_key(ts)

        trigger_reason = ""
        if self.cfg.daily_loss_limit_usd > 0 and state.daily_realized_pnl_usd <= -abs(float(self.cfg.daily_loss_limit_usd)):
            trigger_reason = "daily_loss_limit"
        elif self.cfg.max_drawdown_fraction > 0 and state.drawdown_fraction >= float(self.cfg.max_drawdown_fraction):
            trigger_reason = "max_drawdown"

        if trigger_reason:
            if not state.triggered:
                state.triggered = True
                state.triggered_at = float(ts)
            state.reason = trigger_reason
        elif state.day_key == self._day_key(ts):
            state.triggered = False
            state.reason = ""
            state.triggered_at = None

        self.circuit_store.save(state)
        return state

    def _correlation_group(self, symbol: str) -> str:
        groups = dict(getattr(self.cfg, "correlation_groups", {}) or {})
        return str(groups.get(str(symbol).upper()) or "")

    def _current_exposure(self, portfolio_state: PortfolioState) -> dict[str, Any]:
        return summarize_portfolio_exposure(portfolio_state, getattr(self.cfg, "correlation_groups", {}))


    def _risk_breakdown(self, *, decision: Any, plan: PositionPlan, action: str, stage: str) -> dict[str, Any]:
        """Return an operator-facing explanation for risk geometry decisions.

        This payload is intentionally numeric and compact so monitoring/UI and
        reports can explain a rejection without recomputing the risk model.
        """
        components = dict(getattr(decision, "components", {}) or {})
        price = float(components.get("price") or 0.0)
        atr_fraction = abs(float(components.get("atr_fraction") or 0.0))
        requested_notional = float(plan.intended_notional_usd)
        approved_notional = float(plan.notional_usd)
        risk_limit = float(self.cfg.max_risk_distortion_fraction)
        return {
            "schema_version": "risk_rejection.v1",
            "reason": "risk_distortion_exceeded" if stage == "distortion" else stage,
            "symbol": str(getattr(decision, "symbol", "") or "").upper(),
            "action": str(action or "").upper(),
            "side": self._direction(action),
            "entry_price": price,
            "atr_fraction": atr_fraction,
            "stop_loss": float(plan.stop_loss),
            "take_profit": float(plan.take_profit),
            "stop_distance": float(plan.stop_distance),
            "quantity": float(plan.quantity),
            "requested_notional_usd": requested_notional,
            "approved_notional_usd": approved_notional,
            "notional_usd": approved_notional,
            "intended_notional_usd": requested_notional,
            "risk_amount_usd": float(plan.risk_amount_usd),
            "actual_risk_amount_usd": float(plan.actual_risk_amount_usd),
            "risk_distortion_fraction": float(plan.risk_distortion_fraction),
            "risk_limit": risk_limit,
            "max_risk_distortion_fraction": risk_limit,
            "clamp_applied": bool(plan.clamp_applied),
            "clamp_reason": str(plan.clamp_reason),
            "stage": "risk",
            "approved": False,
        }

    def evaluate_trade(self, *, decision, portfolio_state: PortfolioState, ts: float) -> RiskDecision:
        """Evaluate a trade decision against circuit, portfolio, and authority constraints."""
        action = str(decision.action or "HOLD").upper()
        price = float((decision.components or {}).get("price") or 0.0)
        atr_fraction = abs(float((decision.components or {}).get("atr_fraction") or 0.0))
        circuit = self.load_circuit_breaker()
        authority = self.live_authority_summary(portfolio_state)

        if action == "HOLD":
            return RiskDecision(False, "HOLD", decision.reason, None, {"circuit_breaker": asdict(circuit), "authority": authority})

        authority_reason = str(authority.get("reason") or "live_reconciliation_required")
        if authority.get("guard_required"):
            return RiskDecision(False, "HOLD", authority_reason, None, {"circuit_breaker": asdict(circuit), "authority": authority})
        if self.execution_mode == "live" and not bool(authority.get("authoritative", False)):
            return RiskDecision(False, "HOLD", authority_reason, None, {"circuit_breaker": asdict(circuit), "authority": authority})
        if circuit.triggered:
            return RiskDecision(False, "HOLD", f"circuit_breaker_{circuit.reason}", None, {"circuit_breaker": asdict(circuit), "authority": authority})
        if action.startswith("EXIT"):
            return RiskDecision(True, action, "risk_exit_allowed", None, {"circuit_breaker": asdict(circuit), "authority": authority})

        symbol = str(decision.symbol).upper()
        side = self._direction(action)
        existing = portfolio_state.positions.get(symbol)
        if action in {"ENTER_LONG", "ENTER_SHORT"} and existing is not None:
            existing_side = str(getattr(existing, "side", "") or "").upper()
            reason = "same_side_position_already_open" if existing_side == side else "position_open_requires_flip"
            return RiskDecision(
                False,
                "HOLD",
                reason,
                None,
                {
                    "circuit_breaker": asdict(circuit),
                    "authority": authority,
                    "existing_position": {
                        "symbol": symbol,
                        "side": existing_side,
                        "notional_usd": abs(float(getattr(existing, "notional_usd", 0.0) or 0.0)),
                    },
                    "position_invariant": {
                        "contract_version": "single_position_per_symbol_v1",
                        "direct_entry_allowed_with_open_position": False,
                    },
                },
            )
        guard_cfg = SymbolExpectancyGuardConfig.from_config(self.cfg)
        expectancy_snapshot = build_symbol_expectancy_snapshot(
            getattr(portfolio_state, "closed_trades", []) or [],
            symbol=symbol,
            side=side,
            ts=ts,
            cfg=guard_cfg,
        )
        if expectancy_snapshot.quarantined:
            return RiskDecision(
                False,
                "HOLD",
                "symbol_expectancy_quarantine",
                None,
                {
                    "circuit_breaker": asdict(circuit),
                    "authority": authority,
                    "symbol_expectancy_guard": expectancy_snapshot.to_dict(),
                },
            )

        adaptive_stop = compute_adaptive_stop_multipliers(
            getattr(portfolio_state, "closed_trades", []) or [],
            symbol=symbol,
            side=side,
            atr_fraction=max(atr_fraction, 1e-6),
            base_sl_atr_mult=self.cfg.sl_atr_mult,
            base_tp_atr_mult=self.cfg.tp_atr_mult,
            cfg=AdaptiveStopConfig.from_config(self.cfg),
        )
        plan = compute_position_plan(
            balance=self.cfg.account_balance_usd,
            risk_per_trade=self.cfg.risk_per_trade,
            price=price,
            atr_fraction=max(atr_fraction, 1e-6),
            sl_atr_mult=adaptive_stop.sl_atr_mult,
            tp_atr_mult=adaptive_stop.tp_atr_mult,
            max_notional_usd=min(self.cfg.max_notional_usd, self.cfg.max_symbol_notional_usd),
            min_notional_usd=self.cfg.min_notional_usd,
            direction=side,
        )
        components = {
            "position_plan": asdict(plan),
            "circuit_breaker": asdict(circuit),
            "authority": authority,
            "symbol_expectancy_guard": expectancy_snapshot.to_dict(),
            "adaptive_stops": adaptive_stop.to_dict(),
        }
        if plan.risk_distortion_fraction > float(self.cfg.max_risk_distortion_fraction):
            risk_breakdown = self._risk_breakdown(decision=decision, plan=plan, action=action, stage="distortion")
            components["risk_breakdown"] = risk_breakdown
            components["risk_rejection"] = risk_breakdown
            return RiskDecision(False, "HOLD", "risk_distortion_exceeded", plan, components)

        exposure = self._current_exposure(portfolio_state)
        replacing_notional = abs(float(existing.notional_usd)) if existing is not None else 0.0

        group = self._correlation_group(symbol)
        if group:
            correlated_count = int(exposure["correlated_counts"].get(group, 0))
            if existing is None or self._correlation_group(symbol) != group:
                correlated_count += 1
            if correlated_count > int(self.cfg.max_correlated_positions):
                return RiskDecision(False, "HOLD", "max_correlated_positions", plan, {**components, "correlation_group": group, "exposure": exposure})

        existing_same_side_notional = abs(float(existing.notional_usd)) if existing is not None and existing.side == side else 0.0
        allocation = allocate_position_plan(
            decision=decision,
            plan=plan,
            exposure=exposure,
            cfg=self.cfg,
            existing_notional_usd=replacing_notional,
            existing_same_side_notional_usd=existing_same_side_notional,
            correlation_group=group,
        )
        components["allocation"] = asdict(allocation)
        components["exposure"] = exposure
        if not allocation.approved:
            return RiskDecision(False, "HOLD", allocation.reason, plan, components)

        canary = self._live_canary_limits()
        if self.execution_mode == "live":
            if not bool(canary.get("enabled", True)):
                return RiskDecision(False, "HOLD", "live_canary_disabled", plan, {**components, "live_canary": canary})
            if bool(canary.get("require_testnet", True)) and not bool(canary.get("use_testnet", False)):
                return RiskDecision(False, "HOLD", "live_canary_requires_testnet", plan, {**components, "live_canary": canary})

        target_notional = float(allocation.target_notional_usd)
        if self.execution_mode == "live":
            target_notional = min(target_notional, float(canary.get("max_notional_usd", target_notional) or target_notional))
            if target_notional < float(self.cfg.min_notional_usd):
                return RiskDecision(False, "HOLD", "live_canary_below_min_notional", plan, {**components, "live_canary": canary})

        plan = compute_position_plan(
            balance=self.cfg.account_balance_usd,
            risk_per_trade=self.cfg.risk_per_trade,
            price=price,
            atr_fraction=max(atr_fraction, 1e-6),
            sl_atr_mult=adaptive_stop.sl_atr_mult,
            tp_atr_mult=adaptive_stop.tp_atr_mult,
            max_notional_usd=min(target_notional, self.cfg.max_symbol_notional_usd),
            min_notional_usd=min(self.cfg.min_notional_usd, target_notional),
            direction=side,
        )
        components["position_plan"] = asdict(plan)
        components["live_canary"] = canary

        # Portfolio allocation and live-canary sizing can shrink the executable
        # notional after the initial risk-distortion check. Revalidate the
        # actual position plan that would be sent to execution so no downstream
        # sizing transform can bypass the configured risk tolerance.
        if plan.risk_distortion_fraction > float(self.cfg.max_risk_distortion_fraction):
            risk_breakdown = self._risk_breakdown(decision=decision, plan=plan, action=action, stage="distortion")
            components["risk_breakdown"] = risk_breakdown
            components["risk_rejection"] = risk_breakdown
            return RiskDecision(False, "HOLD", "risk_distortion_exceeded", plan, components)

        gross_after = float(exposure["gross_exposure_usd"]) - replacing_notional + abs(float(plan.notional_usd))
        open_positions_after = int(exposure["open_positions"]) if existing is not None else int(exposure["open_positions"]) + 1
        if self.execution_mode == "live" and open_positions_after > int(canary.get("max_concurrent_positions", 1) or 1):
            return RiskDecision(False, "HOLD", "live_canary_max_concurrent_positions", plan, {**components, "gross_after": gross_after, "open_positions_after": open_positions_after})
        if open_positions_after > int(self.cfg.max_concurrent_positions):
            return RiskDecision(False, "HOLD", "max_concurrent_positions", plan, {**components, "gross_after": gross_after})
        if gross_after > float(self.cfg.max_total_exposure_usd):
            return RiskDecision(False, "HOLD", "max_total_exposure", plan, {**components, "gross_after": gross_after})

        side_after = float(exposure["long_exposure_usd"] if side == "LONG" else exposure["short_exposure_usd"])
        if existing is not None and existing.side == side:
            side_after -= abs(float(existing.notional_usd))
        side_after += abs(float(plan.notional_usd))
        if side_after > float(self.cfg.max_same_direction_exposure_usd):
            return RiskDecision(False, "HOLD", "max_same_direction_exposure", plan, {**components, "side_after": side_after})
        if abs(float(plan.notional_usd)) > float(self.cfg.max_symbol_notional_usd):
            return RiskDecision(False, "HOLD", "max_symbol_notional", plan, components)

        return RiskDecision(True, action, "risk_approved", plan, {**components, "gross_after": gross_after, "open_positions_after": open_positions_after})
