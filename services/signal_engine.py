from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.market_io import load_active_universe_symbols
from core.shared_utils import dump_json, load_json, load_yaml, now_utc_ts, stable_hash_parts
from execution.portfolio_state import PortfolioStateStore
from inference.readiness import signal_eligible_symbols
from inference.trade_decision_logic import TradeDecisionLogic
from monitoring.control_plane_store import ControlPlaneStore


def build_decision_lineage_id(*, symbol: str, interval: str, action: str, reason: str, score: float, previous_position: int, ts: float) -> str:
    digest = stable_hash_parts([
        str(symbol).upper(),
        str(interval),
        str(action).upper(),
        str(reason),
        f"{float(score):.10f}",
        int(previous_position),
        f"{float(ts):.6f}",
    ])
    return f"dec_{digest[:20]}"


def normalize_lineage_payload(lineage: dict[str, Any] | None, *, decision_id: str, symbol: str, interval: str, ts: float, source: str) -> dict[str, Any]:
    payload = dict(lineage or {})
    payload.setdefault("decision_id", decision_id)
    payload.setdefault("symbol", str(symbol).upper())
    payload.setdefault("interval", str(interval))
    payload.setdefault("decision_ts", float(ts))
    payload.setdefault("lineage_source", str(source))
    payload.setdefault("contract_version", "trade_lineage_v1")
    return payload


class AdaptiveModeController:
    """Controlled runtime mode adjuster for prolonged HOLD and bad-trade periods.

    The controller is intentionally conservative:
    - it only uses paper/runtime state already produced by the system;
    - it applies a cooldown between transitions;
    - it writes its decision to a small runtime JSON file for UI/debugging;
    - live safety is not weakened because this only changes inference thresholds,
      while execution and readiness gates still apply.
    """

    VALID_MODES = {"Light", "Medium", "Heavy"}

    def __init__(self, paths) -> None:
        self.paths = paths
        self.cfg = getattr(paths.config, "adaptive_mode", None)
        self.base_mode = str(getattr(paths.config.inference, "active_mode", "Medium") or "Medium").title()
        if self.base_mode not in self.VALID_MODES:
            self.base_mode = "Medium"
        runtime_dir = getattr(paths, "runtime_dir", None) or (getattr(paths, "project_root", Path(".")) / "data" / "runtime")
        self.state_path = Path(runtime_dir) / "adaptive_mode_state.json"

    def _now(self) -> float:
        return now_utc_ts()

    def load_state(self) -> dict[str, Any]:
        payload = load_json(self.state_path, default={})
        state = payload if isinstance(payload, dict) else {}
        mode = str(state.get("current_mode") or self.base_mode).title()
        if mode not in self.VALID_MODES:
            mode = self.base_mode
        state.setdefault("current_mode", mode)
        state.setdefault("base_mode", self.base_mode)
        state.setdefault("hold_started_ts", None)
        state.setdefault("last_non_hold_ts", None)
        state.setdefault("last_transition_ts", None)
        state.setdefault("cooldown_until_ts", 0.0)
        state.setdefault("transition_reason", "initial")
        state.setdefault("enabled", bool(getattr(self.cfg, "enabled", True)))
        return state

    def save_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = dict(state or {})
        payload["updated_at"] = self._now()
        payload["base_mode"] = self.base_mode
        payload["enabled"] = bool(getattr(self.cfg, "enabled", True))
        dump_json(self.state_path, payload)
        return payload

    def current_mode(self) -> str:
        if not bool(getattr(self.cfg, "enabled", True)):
            return self.base_mode
        return str(self.load_state().get("current_mode") or self.base_mode).title()

    def _closed_trade_rows(self) -> list[dict[str, Any]]:
        portfolio = load_json(self.paths.portfolio_state_path, default={})
        rows = portfolio.get("closed_trades", []) if isinstance(portfolio, dict) else []
        return [dict(row) for row in rows if isinstance(row, dict)]

    @staticmethod
    def _trade_pnl(row: dict[str, Any]) -> float:
        for key in ("pnl_usd", "realized_pnl_usd", "pnl", "net_pnl_usd"):
            try:
                return float(row.get(key))
            except Exception:
                continue
        return 0.0

    def _loss_streak(self, rows: list[dict[str, Any]]) -> int:
        streak = 0
        for row in reversed(rows):
            pnl = self._trade_pnl(row)
            if pnl < 0.0:
                streak += 1
            elif pnl > 0.0:
                break
        return streak

    def _win_rate_low(self, rows: list[dict[str, Any]]) -> bool:
        lookback = max(1, int(getattr(self.cfg, "heavy_win_rate_lookback", 8) or 8))
        sample = rows[-lookback:]
        if len(sample) < lookback:
            return False
        wins = sum(1 for row in sample if self._trade_pnl(row) > 0.0)
        threshold = float(getattr(self.cfg, "heavy_on_win_rate_below", 0.25) or 0.25)
        return (wins / float(len(sample))) < threshold

    def update_after_decisions(self, decisions: list[dict[str, Any]], *, open_snapshot: dict[str, Any]) -> dict[str, Any]:
        state = self.load_state()
        now = self._now()
        if not bool(getattr(self.cfg, "enabled", True)):
            state["current_mode"] = self.base_mode
            state["transition_reason"] = "adaptive_mode_disabled"
            return self.save_state(state)

        actions = [str(row.get("action") or "HOLD").upper() for row in decisions if isinstance(row, dict)]
        non_hold_seen = any(action != "HOLD" for action in actions)
        if non_hold_seen:
            state["last_non_hold_ts"] = now
            state["hold_started_ts"] = None
        elif actions:
            state["hold_started_ts"] = state.get("hold_started_ts") or now

        hold_started = state.get("hold_started_ts")
        try:
            hold_age_sec = max(0.0, now - float(hold_started)) if hold_started else 0.0
        except Exception:
            hold_age_sec = 0.0
        state["hold_age_seconds"] = hold_age_sec

        closed = self._closed_trade_rows()
        loss_streak = self._loss_streak(closed)
        state["loss_streak"] = loss_streak
        state["closed_trade_count"] = len(closed)
        cooldown_until = float(state.get("cooldown_until_ts") or 0.0)
        in_cooldown = now < cooldown_until
        state["cooldown_active"] = in_cooldown
        state["cooldown_seconds_left"] = max(0.0, cooldown_until - now)

        current_mode = str(state.get("current_mode") or self.base_mode).title()
        if current_mode not in self.VALID_MODES:
            current_mode = self.base_mode

        target_mode = current_mode
        reason = str(state.get("transition_reason") or "unchanged")
        heavy_threshold = max(1, int(getattr(self.cfg, "heavy_on_loss_streak", 3) or 3))
        if loss_streak >= heavy_threshold or self._win_rate_low(closed):
            target_mode = "Heavy"
            reason = "loss_quality_guard"
        elif (
            bool(getattr(self.cfg, "medium_to_light_when_no_trades", True))
            and current_mode == "Medium"
            and not open_snapshot
            and actions
            and not non_hold_seen
            and hold_age_sec >= float(getattr(self.cfg, "hold_minutes", 30.0) or 30.0) * 60.0
        ):
            target_mode = "Light"
            reason = "prolonged_hold_no_trades"
        elif current_mode in {"Light", "Heavy"} and not in_cooldown and non_hold_seen:
            target_mode = self.base_mode
            reason = "activity_restored"

        if target_mode != current_mode and not in_cooldown:
            state["previous_mode"] = current_mode
            state["current_mode"] = target_mode
            state["last_transition_ts"] = now
            state["transition_reason"] = reason
            state["cooldown_until_ts"] = now + float(getattr(self.cfg, "cooldown_minutes", 30.0) or 30.0) * 60.0
        else:
            state["current_mode"] = current_mode

        state["recent_actions"] = actions[-50:]
        state["evaluated_decision_count"] = len(actions)
        return self.save_state(state)


@dataclass(slots=True)
class EngineDecision:
    symbol: str
    interval: str
    action: str
    score: float
    reason: str
    mode: str
    components: dict[str, Any]

    @staticmethod
    def _policy_component(components: dict[str, Any] | None) -> dict[str, Any]:
        """Return the operator-facing policy component from legacy or new keys."""
        if not isinstance(components, dict):
            return {}
        candidate = components.get("policy")
        if isinstance(candidate, dict):
            return candidate
        candidate = components.get("dqn")
        return candidate if isinstance(candidate, dict) else {}

    def pretty_line(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        action_text = self._pretty_action()
        rf = self.components.get("rf", {}) if isinstance(self.components, dict) else {}
        gbm = self.components.get("gbm", {}) if isinstance(self.components, dict) else {}
        policy = self._policy_component(self.components)
        parts: list[str] = []
        if rf:
            parts.append(
                "RF p_short={:.3f} p_hold={:.3f} p_long={:.3f}".format(
                    float(rf.get("p_short", 0.0) or 0.0),
                    float(rf.get("p_hold", 0.0) or 0.0),
                    float(rf.get("p_long", 0.0) or 0.0),
                )
            )
        if gbm:
            parts.append(
                "GBM p_short={:.3f} p_hold={:.3f} p_long={:.3f} meta={:.3f}".format(
                    float(gbm.get("p_short", 0.0) or 0.0),
                    float(gbm.get("p_hold", 0.0) or 0.0),
                    float(gbm.get("p_long", 0.0) or 0.0),
                    float(gbm.get("meta_probability", 0.0) or 0.0),
                )
            )
        if policy and policy.get("used"):
            parts.append(
                "Policy action={} conf={:.3f} score={:.3f}".format(
                    policy.get("action", "hold"),
                    float(policy.get("confidence", 0.0) or 0.0),
                    float(policy.get("score", 0.0) or 0.0),
                )
            )
        plan = self.components.get("risk_plan", {}) if isinstance(self.components, dict) else {}
        if plan:
            parts.append(
                "Plan qty={:.6f} notional={:.2f} sl={:.5f} tp={:.5f}".format(
                    float(plan.get("quantity", 0.0) or 0.0),
                    float(plan.get("notional_usd", 0.0) or 0.0),
                    float(plan.get("stop_loss", 0.0) or 0.0),
                    float(plan.get("take_profit", 0.0) or 0.0),
                )
            )
        parts.append("EV={:.6f}".format(float(self.score or 0.0)))
        detail = " | ".join(parts)
        suffix = f" | {detail}" if detail else ""
        return f"{ts} {self.symbol} | Mode: {self.mode} | {action_text} | Reason: {self.reason}{suffix}"

    def _pretty_action(self) -> str:
        mapping = {
            "ENTER_LONG": "ENTER LONG",
            "ENTER_SHORT": "ENTER SHORT",
            "FLIP_TO_LONG": "ENTER LONG",
            "FLIP_TO_SHORT": "ENTER SHORT",
            "EXIT": "EXIT",
            "EXIT_LONG": "EXIT",
            "EXIT_SHORT": "EXIT",
            "HOLD": "HOLD",
            "SKIP": "SKIP",
        }
        return mapping.get(self.action, self.action.replace("_", " ").upper())


class SignalEngine:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.cfg = paths.config
        self.logic = TradeDecisionLogic(paths)
        self.default_mode = str(getattr(self.cfg.inference, "active_mode", "Medium") or "Medium").title()
        self.adaptive_mode = AdaptiveModeController(paths)
        self.control_store = ControlPlaneStore(paths)

    def load_configured_symbols(self) -> list[str]:
        symbols = load_active_universe_symbols(self.paths, include_db_fallback=False)
        if symbols:
            return symbols
        cfg = load_yaml(self.paths.tradable_symbols_path)
        raw_symbols = cfg.get("symbols", []) if isinstance(cfg, dict) else []
        return [str(x).upper() for x in raw_symbols if str(x).strip()]

    def load_symbols(self) -> list[str]:
        raw_symbols = self.load_configured_symbols()
        interval = self.paths.config.frames.anchor_interval
        ready_symbols, _ = signal_eligible_symbols(self.paths, symbols=raw_symbols, interval=interval)
        return ready_symbols

    def load_open_trade_snapshot(self) -> dict[str, Any]:
        """Load open-position state from the canonical portfolio authority.

        backtest_open_trades.json is retained only as a legacy fallback for old
        tests and isolated historical-backtest runs. Runtime signal decisions
        must not treat stale historical backtest rows as live/paper positions.
        """
        try:
            store = PortfolioStateStore(
                self.paths.portfolio_state_path,
                starting_equity_usd=float(getattr(self.paths.config.risk, "account_balance_usd", 0.0) or 0.0),
            )
            state = store.load()
            snapshot = store.open_position_snapshot(state)
            if snapshot:
                return snapshot
        except Exception:
            pass

        payload = load_json(self.paths.backtest_open_trades_path, default={})
        return payload if isinstance(payload, dict) else {}

    def load_control_state(self) -> dict[str, Any]:
        return self.control_store.summary()

    def load_adaptive_mode_state(self) -> dict[str, Any]:
        state = self.adaptive_mode.load_state()
        state["enabled"] = False
        state["current_mode"] = self.default_mode
        state["transition_reason"] = "disabled_step11_scope_guard"
        return state

    def update_adaptive_mode_state(self, decisions: list[dict[str, Any]], *, open_snapshot: dict[str, Any]) -> dict[str, Any]:
        state = self.adaptive_mode.load_state()
        state["enabled"] = False
        state["current_mode"] = self.default_mode
        state["transition_reason"] = "disabled_step11_scope_guard"
        saved = self.adaptive_mode.save_state(state)
        saved["enabled"] = False
        saved["current_mode"] = self.default_mode
        saved["transition_reason"] = "disabled_step11_scope_guard"
        dump_json(self.adaptive_mode.state_path, saved)
        return saved

    def save_control_state(self, *, paused: bool, pause_reason: str = "", updated_by: str = "system") -> dict[str, Any]:
        return self.control_store.set_pause(paused=paused, pause_reason=pause_reason, updated_by=updated_by)

    @staticmethod
    def _signal_execution_contract(decision: EngineDecision, *, paused: bool = False, pause_reason: str = "") -> dict[str, Any]:
        """Return the canonical signal-stage attempt/reject contract.

        The signal generator is not the execution authority. Actionable ENTER/EXIT/FLIP
        decisions are therefore marked as valid signal intents but still
        ``not_attempted`` at this surface. Non-actionable HOLD decisions carry the
        decision blocker from the decision logic instead of being mislabeled as
        execution rejections.
        """
        components = decision.components if isinstance(decision.components, dict) else {}
        decision_state = components.get("decision_state") if isinstance(components.get("decision_state"), dict) else {}
        action = str(decision_state.get("action") or decision.action or "HOLD").upper()
        reason = str(decision_state.get("reason") or decision.reason or "unspecified")
        blocker = str(decision_state.get("blocker") or components.get("decision_blocker") or reason)
        signal_actionable = bool(decision_state.get("execution_eligible", components.get("execution_eligible", False)))
        if paused:
            return {
                "execution_status": "not_attempted",
                "execution_attempted": False,
                "execution_blocker_stage": "runtime_gate",
                "execution_blocker": str(pause_reason or "runtime_gate_blocked"),
                "rejection_reason": str(pause_reason or "runtime_gate_blocked"),
                "order_status": "not_attempted",
                "approved_for_order": False,
                "signal_actionable": False,
                "signal_intent": action,
                "signal_contract_version": "signal_attempt_contract_v1",
            }
        if signal_actionable:
            return {
                "execution_status": "not_attempted",
                "execution_attempted": False,
                "execution_blocker_stage": "signal_generation",
                "execution_blocker": "signal_generation_only",
                "rejection_reason": "signal_generation_only",
                "order_status": "not_attempted",
                "approved_for_order": False,
                "signal_actionable": True,
                "signal_intent": action,
                "signal_contract_version": "signal_attempt_contract_v1",
            }
        return {
            "execution_status": "not_attempted",
            "execution_attempted": False,
            "execution_blocker_stage": "decision_logic",
            "execution_blocker": blocker,
            "rejection_reason": blocker,
            "order_status": "not_attempted",
            "approved_for_order": False,
            "signal_actionable": False,
            "signal_intent": action,
            "signal_contract_version": "signal_attempt_contract_v1",
        }

    def _decision_journal_record(self, decision: EngineDecision, previous_position: int, *, paused: bool = False, pause_reason: str = "") -> dict[str, Any]:
        ts = now_utc_ts()
        decision_id = build_decision_lineage_id(
            symbol=decision.symbol,
            interval=decision.interval,
            action=decision.action,
            reason=decision.reason,
            score=float(decision.score),
            previous_position=int(previous_position),
            ts=ts,
        )
        components = dict(decision.components or {})
        control_plane = components.get("control_plane") if isinstance(components.get("control_plane"), dict) else {}
        components["control_plane"] = {
            "paused": bool(paused or control_plane.get("paused", False)),
            "pause_reason": str(pause_reason or control_plane.get("pause_reason") or ""),
        }
        contract = self._signal_execution_contract(decision, paused=paused, pause_reason=pause_reason)
        execution_result = {"status": contract["execution_status"], "reason": contract["execution_blocker"]}
        components.setdefault("execution_result", execution_result)
        components.setdefault("execution_contract", dict(contract))
        lineage = normalize_lineage_payload(
            components.get("lineage") if isinstance(components.get("lineage"), dict) else {},
            decision_id=decision_id,
            symbol=decision.symbol,
            interval=decision.interval,
            ts=ts,
            source="signal_engine",
        )
        components["lineage"] = lineage
        return {
            "ts": ts,
            "decision_id": decision_id,
            "lineage": lineage,
            "symbol": decision.symbol,
            "interval": decision.interval,
            "action": decision.action,
            "effective_action": decision.action,
            "approved": False,
            "score": float(decision.score),
            "reason": decision.reason,
            "mode": decision.mode,
            "previous_position": int(previous_position),
            "paused": bool(paused),
            "pause_reason": str(pause_reason or ""),
            "decision_stage": "signal_generation",
            "journal_writer": "signal_engine",
            "state_owner": "signal_generator_controller",
            "execution_surface": "signal_generation_only",
            **contract,
            "components": components,
        }

    def append_decision_journal(self, decision: EngineDecision, previous_position: int, *, paused: bool = False, pause_reason: str = "") -> dict[str, Any]:
        record = self._decision_journal_record(decision, previous_position, paused=paused, pause_reason=pause_reason)
        self.paths.decision_journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.decision_journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def recent_decision_journal(self, limit: int = 100) -> list[dict[str, Any]]:
        path = self.paths.decision_journal_path
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        rows.append(payload)
        except Exception:
            return []
        rows.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
        return rows[: max(1, int(limit))]

    def _previous_position_from_snapshot(self, symbol: str, snapshot: dict[str, Any]) -> int:
        payload = snapshot.get(symbol, {}) if isinstance(snapshot, dict) else {}
        side = str(payload.get("side", "")).upper()
        if side == "LONG":
            return 1
        if side == "SHORT":
            return -1
        return 0

    def compute_decision(self, symbol: str, open_snapshot: dict[str, Any], *, paused: bool = False, pause_reason: str = "") -> EngineDecision:
        interval = self.paths.config.frames.anchor_interval
        prev = self._previous_position_from_snapshot(symbol, open_snapshot)
        effective_mode = self.default_mode
        if paused:
            return EngineDecision(
                symbol=symbol,
                interval=interval,
                action="HOLD",
                score=0.0,
                reason="control_plane_paused",
                mode=effective_mode,
                components={"control_plane": {"paused": True, "pause_reason": pause_reason}, "previous_position": prev, "adaptive_mode": self.load_adaptive_mode_state()},
            )
        td = self.logic.decide(symbol, interval, previous_position=prev, mode=effective_mode)
        components = dict(td.components)
        components.setdefault("control_plane", {"paused": False, "pause_reason": ""})
        components["previous_position"] = prev
        components["adaptive_mode"] = self.load_adaptive_mode_state()
        return EngineDecision(
            symbol=td.symbol,
            interval=td.interval,
            action=td.action,
            score=float(td.score),
            reason=str(td.reason),
            mode=effective_mode,
            components=components,
        )

    def append_signal_line(self, line: str) -> None:
        path = self.paths.signal_events_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
