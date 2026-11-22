# monitoring/control_plane_store.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core.artifact_contract import (
    incumbent_pointer_path,
    load_json,
    load_manifest_metrics,
    resolve_active_manifest_path,
    resolve_artifact_path,
    save_json,
)
from core.shared_utils import atomic_write_bytes, load_json as runtime_load_json, now_utc_ts
from core.symbol_management import load_symbol_state
from execution.circuit_breaker import CircuitBreakerState, CircuitBreakerStore
from execution.order_store import OrderStore
from execution.portfolio_state import PortfolioStateStore


@dataclass(slots=True)
class ControlPlaneState:
    """Persisted control-plane state used by monitoring and runtime services."""

    paused: bool = False
    pause_reason: str = ""
    updated_at: float | None = None
    updated_by: str = "system"


class ControlPlaneStore:
    """Centralized persistence for pause, control actions, and model rollback operations."""

    def __init__(self, paths) -> None:
        self.paths = paths
        circuit_path = getattr(paths, "circuit_breaker_state_path", None)
        if circuit_path is None:
            runtime_dir = getattr(paths, "runtime_dir", getattr(paths, "project_root", Path(".")) / "data" / "runtime")
            circuit_path = Path(runtime_dir) / "circuit_breaker_state.json"
        self.circuit_store = CircuitBreakerStore(circuit_path)
        self.portfolio_store = self._build_portfolio_store_if_available()
        self.order_store = self._build_order_store_if_available()

    def _build_portfolio_store_if_available(self) -> PortfolioStateStore | None:
        """Create a portfolio store only when the current fixture/runtime supports it.

        Monitoring and control-plane tests often use lightweight path fixtures that
        intentionally omit execution-state paths. The control plane should degrade
        gracefully in those contexts instead of crashing during construction.
        """
        portfolio_state_path = getattr(self.paths, "portfolio_state_path", None)
        if portfolio_state_path is None:
            return None
        config = getattr(self.paths, "config", None)
        risk_cfg = getattr(config, "risk", None) if config is not None else None
        if risk_cfg is None:
            return None
        starting_equity_usd = float(getattr(risk_cfg, "account_balance_usd", 0.0) or 0.0)
        return PortfolioStateStore(portfolio_state_path, starting_equity_usd=starting_equity_usd)


    def _build_order_store_if_available(self) -> OrderStore | None:
        order_state_path = getattr(self.paths, "order_state_path", None)
        execution_journal_path = getattr(self.paths, "execution_journal_path", None)
        if order_state_path is None or execution_journal_path is None:
            return None
        return OrderStore(order_state_path, execution_journal_path)

    def _control_plane_state_path(self) -> Path:
        explicit = getattr(self.paths, "control_plane_state_path", None)
        if explicit is not None:
            return Path(explicit)
        runtime_dir = getattr(self.paths, "runtime_dir", getattr(self.paths, "project_root", Path(".")) / "data" / "runtime")
        return Path(runtime_dir) / "control_plane_state.json"

    def _decision_journal_path(self) -> Path:
        explicit = getattr(self.paths, "decision_journal_path", None)
        if explicit is not None:
            return Path(explicit)
        runtime_dir = getattr(self.paths, "runtime_dir", getattr(self.paths, "project_root", Path(".")) / "data" / "runtime")
        return Path(runtime_dir) / "decision_journal.jsonl"

    def _default_state(self) -> ControlPlaneState:
        return ControlPlaneState(paused=False, pause_reason="", updated_at=None, updated_by="system")

    def load_state(self) -> ControlPlaneState:
        payload = runtime_load_json(self._control_plane_state_path(), default={}) or {}
        if not isinstance(payload, dict):
            payload = {}
        return ControlPlaneState(
            paused=bool(payload.get("paused", False)),
            pause_reason=str(payload.get("pause_reason") or ""),
            updated_at=payload.get("updated_at"),
            updated_by=str(payload.get("updated_by") or "system"),
        )

    def pause_state(self) -> dict[str, Any]:
        """Return the current pause state in a compact runtime-friendly shape."""
        state = self.load_state()
        return {
            "paused": bool(state.paused),
            "pause_reason": str(state.pause_reason or ""),
            "updated_at": state.updated_at,
            "updated_by": str(state.updated_by or "system"),
        }

    def is_paused(self) -> tuple[bool, str]:
        """Return whether trading is paused and the current pause reason."""
        state = self.pause_state()
        return bool(state["paused"]), str(state["pause_reason"] or "")

    def save_state(self, state: ControlPlaneState) -> ControlPlaneState:
        payload = asdict(state)
        atomic_write_bytes(self._control_plane_state_path(), json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        return state

    def set_pause(self, *, paused: bool, pause_reason: str = "", updated_by: str = "system") -> dict[str, Any]:
        state = ControlPlaneState(
            paused=bool(paused),
            pause_reason=str(pause_reason or ""),
            updated_at=now_utc_ts(),
            updated_by=str(updated_by or "system"),
        )
        self.save_state(state)
        self.record_action(
            action="pause" if state.paused else "resume",
            updated_by=state.updated_by,
            message="Trading paused" if state.paused else "Trading resumed",
            data={"pause_reason": state.pause_reason},
        )
        return self.summary()

    def _actions_path(self) -> Path:
        explicit = getattr(self.paths, "control_plane_actions_path", None)
        if explicit is not None:
            return Path(explicit)
        runtime_dir = getattr(self.paths, "runtime_dir", getattr(self.paths, "project_root", Path(".")) / "data" / "runtime")
        return Path(runtime_dir) / "control_plane_actions.jsonl"

    def _read_actions(self) -> list[dict[str, Any]]:
        path = self._actions_path()
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
        return rows

    def recent_actions(self, limit: int = 25) -> list[dict[str, Any]]:
        return self._read_actions()[: max(1, int(limit))]

    def record_action(self, *, action: str, updated_by: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "ts": now_utc_ts(),
            "action": str(action or "unknown"),
            "updated_by": str(updated_by or "system"),
            "message": str(message or ""),
            "data": dict(data or {}),
        }
        actions_path = self._actions_path()
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        with actions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def reset_circuit_breaker(self, *, updated_by: str = "ui", reason: str = "manual_reset") -> dict[str, Any]:
        current = self.circuit_store.load()
        reset = CircuitBreakerState(
            triggered=False,
            reason="",
            triggered_at=None,
            last_checked_at=now_utc_ts(),
            peak_equity_usd=float(current.peak_equity_usd or 0.0),
            current_equity_usd=float(current.current_equity_usd or 0.0),
            drawdown_fraction=0.0,
            daily_realized_pnl_usd=float(current.daily_realized_pnl_usd or 0.0),
            day_key=str(current.day_key or ""),
        )
        self.circuit_store.save(reset)
        self.record_action(
            action="circuit_breaker_reset",
            updated_by=updated_by,
            message="Circuit breaker reset",
            data={"reason": reason},
        )
        return asdict(reset)

    def _symbol_summary(self) -> dict[str, Any]:
        if not hasattr(self.paths, "symbol_state_path"):
            return {"counts": {"ACTIVE": 0, "TIMED_OUT": 0, "BLOCKED": 0}, "rows": []}
        state = load_symbol_state(self.paths)
        symbols_payload = state.get("symbols", {}) if isinstance(state, dict) else {}
        rows: list[dict[str, Any]] = []
        counts = {"ACTIVE": 0, "TIMED_OUT": 0, "BLOCKED": 0}
        if isinstance(symbols_payload, dict):
            for symbol, raw in symbols_payload.items():
                if not isinstance(raw, dict):
                    continue
                status = str(raw.get("status") or "ACTIVE").upper()
                counts[status] = counts.get(status, 0) + 1
                rows.append(
                    {
                        "symbol": str(symbol).upper(),
                        "status": status,
                        "retry_not_before_ts": raw.get("retry_not_before_ts"),
                        "last_error": raw.get("last_error"),
                        "consecutive_failures": int(raw.get("consecutive_failures") or 0),
                        "manually_blocked": bool(raw.get("manually_blocked", False)),
                    }
                )
        rows.sort(key=lambda row: (row["status"], row["symbol"]))
        return {"counts": counts, "rows": rows[:200]}

    def _decision_lineage_preview(self, limit: int = 20) -> list[dict[str, Any]]:
        path = self._decision_journal_path()
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
                    if not isinstance(payload, dict):
                        continue
                    components = payload.get("components") if isinstance(payload.get("components"), dict) else {}
                    lineage = payload.get("lineage") if isinstance(payload.get("lineage"), dict) else {}
                    if not lineage and isinstance(components, dict):
                        lineage = components.get("lineage") if isinstance(components.get("lineage"), dict) else {}
                    row = {
                        "ts": payload.get("ts"),
                        "symbol": payload.get("symbol"),
                        "interval": payload.get("interval"),
                        "action": payload.get("action"),
                        "reason": payload.get("reason"),
                        "score": payload.get("score"),
                        "previous_position": payload.get("previous_position"),
                        "paused": bool(payload.get("paused", False)),
                        "pause_reason": payload.get("pause_reason"),
                        "components": components,
                        "lineage": lineage,
                        "decision_id": lineage.get("decision_id") if isinstance(lineage, dict) else None,
                        "order_id": lineage.get("order_id") if isinstance(lineage, dict) else None,
                        "close_trigger": lineage.get("close_trigger") if isinstance(lineage, dict) else None,
                    }
                    rows.append(row)
        except Exception:
            return []
        rows.sort(key=lambda row: float(row.get("ts") or 0.0), reverse=True)
        return rows[: max(1, int(limit))]



    def _shadow_candidate_preview(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._decision_lineage_preview(limit=max(20, int(limit) * 3))
        preview: list[dict[str, Any]] = []
        for row in rows:
            components = row.get("components") if isinstance(row.get("components"), dict) else {}
            shadow = components.get("shadow_candidate") if isinstance(components, dict) else {}
            if not isinstance(shadow, dict) or not shadow:
                continue
            preview.append({
                "ts": row.get("ts"),
                "symbol": row.get("symbol"),
                "interval": row.get("interval"),
                "incumbent_action": row.get("action"),
                "incumbent_reason": row.get("reason"),
                "candidate_available": bool(shadow.get("available", False)),
                "candidate_action": shadow.get("action"),
                "candidate_reason": shadow.get("reason"),
                "candidate_score": shadow.get("score"),
                "different_action": bool(shadow.get("different_action", False)),
                "candidate_lineage": shadow.get("lineage") if isinstance(shadow.get("lineage"), dict) else {},
            })
            if len(preview) >= max(1, int(limit)):
                break
        return preview

    def _runtime_observability_summary(self, limit: int = 50) -> dict[str, Any]:
        rows = self._decision_lineage_preview(limit=max(20, int(limit) * 2))
        recent_rows = rows[: max(1, int(limit))]
        stale_count = 0
        freshness_ok = 0
        freshness_ages_s: list[float] = []
        score_values: list[float] = []
        recent_action_counts: dict[str, int] = {}
        baseline_action_counts: dict[str, int] = {}
        divergent_shadow = 0
        shadow_available = 0
        for idx, row in enumerate(rows):
            action = str(row.get("action") or "HOLD").upper()
            if idx < max(1, int(limit)):
                recent_action_counts[action] = recent_action_counts.get(action, 0) + 1
                score = row.get("score")
                try:
                    score_values.append(float(score))
                except Exception:
                    pass
                components = row.get("components") if isinstance(row.get("components"), dict) else {}
                freshness = components.get("freshness") if isinstance(components, dict) else {}
                if isinstance(freshness, dict) and freshness:
                    if bool(freshness.get("ok", False)):
                        freshness_ok += 1
                    else:
                        stale_count += 1
                    age_seconds = freshness.get("age_seconds")
                    try:
                        if age_seconds is not None:
                            freshness_ages_s.append(float(age_seconds))
                    except Exception:
                        pass
                shadow = components.get("shadow_candidate") if isinstance(components, dict) else {}
                if isinstance(shadow, dict) and shadow:
                    if bool(shadow.get("available", False)):
                        shadow_available += 1
                    if bool(shadow.get("different_action", False)):
                        divergent_shadow += 1
            else:
                baseline_action_counts[action] = baseline_action_counts.get(action, 0) + 1

        recent_non_hold = sum(count for action, count in recent_action_counts.items() if action != "HOLD")
        baseline_non_hold = sum(count for action, count in baseline_action_counts.items() if action != "HOLD")
        recent_total = max(1, len(recent_rows))
        baseline_total = max(1, sum(baseline_action_counts.values()))

        latest_lineage = recent_rows[0].get("lineage") if recent_rows and isinstance(recent_rows[0].get("lineage"), dict) else {}
        try:
            active_versions = self.list_model_versions()
        except Exception:
            active_versions = {}
        active_model_count = 0
        candidate_model_count = 0
        for model_payload in active_versions.values():
            if not isinstance(model_payload, dict):
                continue
            for symbol_payload in model_payload.values():
                if not isinstance(symbol_payload, dict):
                    continue
                for interval_payload in symbol_payload.values():
                    if not isinstance(interval_payload, dict):
                        continue
                    active_model_count += len(interval_payload.get("active", []) or [])
                    candidate_model_count += len(interval_payload.get("candidates", []) or [])

        live_runtime = self._live_runtime_summary()
        portfolio_runtime: dict[str, Any] = {}
        if self.portfolio_store is not None:
            try:
                portfolio = self.portfolio_store.load()
                portfolio_runtime = {
                    "realized_pnl_usd": float(getattr(portfolio, "realized_pnl_usd", 0.0) or 0.0),
                    "daily_realized_pnl_usd": float(getattr(portfolio, "daily_realized_pnl_usd", 0.0) or 0.0),
                    "equity_usd": float(getattr(portfolio, "last_equity_usd", 0.0) or 0.0),
                    "open_positions": len(getattr(portfolio, "positions", {}) or {}),
                    "authority_mode": str(getattr(portfolio, "authority_mode", "") or live_runtime.get("authority_mode", "paper_local")),
                    "reconciliation_status": str(getattr(portfolio, "reconciliation_status", "") or live_runtime.get("reconciliation_status", "paper_local")),
                }
            except Exception as exc:
                portfolio_runtime = {"error": str(exc)}

        return {
            "freshness": {
                "recent_entries": len(recent_rows),
                "fresh_entries": int(freshness_ok),
                "stale_entries": int(stale_count),
                "max_age_seconds": max(freshness_ages_s) if freshness_ages_s else None,
                "avg_age_seconds": (sum(freshness_ages_s) / len(freshness_ages_s)) if freshness_ages_s else None,
                "latest_reason": str(recent_rows[0].get("reason") or "") if recent_rows else "",
            },
            "prediction_drift": {
                "recent_action_counts": recent_action_counts,
                "baseline_action_counts": baseline_action_counts,
                "recent_non_hold_rate": float(recent_non_hold / recent_total),
                "baseline_non_hold_rate": float(baseline_non_hold / baseline_total),
                "avg_score_recent": (sum(score_values) / len(score_values)) if score_values else None,
                "shadow_available": int(shadow_available),
                "shadow_divergence_rate": float(divergent_shadow / max(1, shadow_available)),
            },
            "model_state": {
                "active_model_count": int(active_model_count),
                "candidate_model_count": int(candidate_model_count),
                "latest_lineage": latest_lineage,
            },
            "runtime_performance": {
                "recent_expected_direction_balance": int(recent_action_counts.get("ENTER_LONG", 0) + recent_action_counts.get("FLIP_TO_LONG", 0) - recent_action_counts.get("ENTER_SHORT", 0) - recent_action_counts.get("FLIP_TO_SHORT", 0)),
                "portfolio_runtime": portfolio_runtime,
                "live_runtime": live_runtime,
            },
        }

    def _parse_model_path(self, path: Path) -> tuple[str, str, str, str | None] | None:
        parts = list(path.parts)
        try:
            idx = parts.index("model_storage")
        except ValueError:
            return None
        tail = list(parts[idx + 1 :])
        if len(tail) < 4:
            return None
        directory = tail[0]
        mapping = {
            "random_forest_models": "rf",
            "gradient_boost_models": "gbm",
            "reinforcement_agents": "dqn",
            "policy_models": "dqn",
        }
        model_type = mapping.get(directory)
        if model_type is None:
            return None
        symbol = tail[1]
        interval = tail[2]
        role: str | None = None
        if model_type == "gbm" and len(tail) >= 5 and tail[3] in {"direction", "meta"}:
            role = tail[3]
        return model_type, symbol, interval, role

    def list_model_versions(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        active_refs: set[tuple[str, str, str, str | None, str]] = set()

        model_storage_dir = getattr(self.paths, "model_storage_dir", None)
        if model_storage_dir is None:
            return result
        model_storage_dir = Path(model_storage_dir)
        if not model_storage_dir.exists():
            return result

        for pointer_path in model_storage_dir.rglob("incumbent.json"):
            pointer = load_json(pointer_path) or {}
            manifest_raw = pointer.get("manifest_path")
            parsed = self._parse_model_path(pointer_path)
            if parsed is None or not manifest_raw:
                continue
            model_type, symbol, interval, role = parsed
            manifest_path = resolve_artifact_path(self.paths, str(manifest_raw))
            manifest = load_json(manifest_path) or {}
            metrics = load_manifest_metrics(self.paths, manifest_path)
            lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
            feature_set = manifest.get("feature_set") if isinstance(manifest.get("feature_set"), dict) else {}
            timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
            bucket = result.setdefault(model_type, {}).setdefault(symbol.upper(), {}).setdefault(interval, {"active": [], "candidates": []})
            entry = {
                "role": role,
                "candidate_id": lifecycle.get("candidate_id") or pointer.get("candidate_id"),
                "manifest_path": str(manifest_path),
                "promoted_at": pointer.get("promoted_at") or lifecycle.get("promoted_at") or timestamps.get("updated_at"),
                "feature_set_version": feature_set.get("version"),
                "feature_set_signature": feature_set.get("signature"),
                "metrics": metrics,
                "stage": "active",
            }
            bucket["active"].append(entry)
            active_refs.add((model_type, symbol.upper(), interval, role, str(manifest_path)))

        for manifest_path in model_storage_dir.rglob("candidates/*/manifest.json"):
            parsed = self._parse_model_path(manifest_path)
            if parsed is None:
                continue
            model_type, symbol, interval, role = parsed
            manifest = load_json(manifest_path) or {}
            if not manifest:
                continue
            lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
            feature_set = manifest.get("feature_set") if isinstance(manifest.get("feature_set"), dict) else {}
            timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
            metrics = load_manifest_metrics(self.paths, manifest_path)
            bucket = result.setdefault(model_type, {}).setdefault(symbol.upper(), {}).setdefault(interval, {"active": [], "candidates": []})
            key = (model_type, symbol.upper(), interval, role, str(manifest_path))
            entry = {
                "role": role,
                "candidate_id": lifecycle.get("candidate_id"),
                "manifest_path": str(manifest_path),
                "created_at": timestamps.get("created_at"),
                "updated_at": timestamps.get("updated_at"),
                "feature_set_version": feature_set.get("version"),
                "feature_set_signature": feature_set.get("signature"),
                "metrics": metrics,
                "stage": "active" if key in active_refs else "candidate",
            }
            if key in active_refs:
                continue
            bucket["candidates"].append(entry)

        return result

    def rollback_model_version(
        self,
        *,
        model_type: str,
        symbol: str,
        interval: str,
        manifest_path: str,
        updated_by: str = "ui",
        model_role: str | None = None,
    ) -> dict[str, Any]:
        manifest = load_json(resolve_artifact_path(self.paths, manifest_path))
        if not manifest:
            raise ValueError("manifest_not_found")
        manifest_abs = resolve_artifact_path(self.paths, manifest_path)
        previous = resolve_active_manifest_path(self.paths, model_type, symbol, interval, model_role=model_role)
        lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
        timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
        metrics = load_manifest_metrics(self.paths, manifest_abs)
        pointer_payload = {
            "schema_version": 1,
            "model_type": str(model_type),
            "model_role": model_role,
            "symbol": str(symbol).upper(),
            "interval": str(interval),
            "candidate_id": lifecycle.get("candidate_id"),
            "manifest_path": str(manifest_abs.relative_to(self.paths.project_root)),
            "promoted_at": now_utc_ts(),
            "metrics_summary": metrics,
        }
        save_json(incumbent_pointer_path(self.paths, model_type, symbol, interval, model_role=model_role), pointer_payload)
        lifecycle.update({"stage": "incumbent", "promoted_at": pointer_payload["promoted_at"]})
        manifest["lifecycle"] = lifecycle
        timestamps["updated_at"] = pointer_payload["promoted_at"]
        manifest["timestamps"] = timestamps
        save_json(manifest_abs, manifest)
        self.record_action(
            action="model_rollback",
            updated_by=updated_by,
            message=f"Rolled back {model_type} {str(symbol).upper()} {interval}",
            data={
                "model_type": model_type,
                "symbol": str(symbol).upper(),
                "interval": str(interval),
                "model_role": model_role,
                "manifest_path": str(manifest_abs),
                "previous_manifest_path": str(previous) if previous else None,
                "candidate_id": lifecycle.get("candidate_id"),
            },
        )
        return {
            "ok": True,
            "pointer": pointer_payload,
            "previous_manifest_path": str(previous) if previous else None,
        }

    def _live_runtime_summary(self) -> dict[str, Any]:
        """Return a compact live runtime authority snapshot for monitoring and diagnostics."""
        config = getattr(self.paths, "config", None)
        runtime_cfg = getattr(config, "runtime", None) if config is not None else None
        exchange_cfg = getattr(config, "exchange", None) if config is not None else None
        execution_mode = str(getattr(runtime_cfg, "execution_mode", "paper") or "paper")
        reconciliation_enabled = bool(getattr(exchange_cfg, "position_reconciliation_enabled", False))
        reconciliation_required = bool(getattr(exchange_cfg, "require_reconciliation_for_live", True))

        base = {
            "execution_mode": execution_mode,
            "reconciliation_enabled": reconciliation_enabled,
            "reconciliation_required": reconciliation_required,
            "authority_mode": "unavailable",
            "reconciliation_status": "unavailable",
            "last_reconciled_at": None,
            "exchange_position_count": 0,
            "reconciliation_mismatch_count": 0,
            "last_reconciliation_error": "",
            "exchange_snapshot_hash": "",
            "exchange_authority_required": execution_mode == "live",
            "authoritative": execution_mode != "live",
            "authority_reason": "paper_local_authority" if execution_mode != "live" else "live_reconciliation_required",
        }

        open_orders = []
        if self.order_store is not None:
            try:
                open_orders = self.order_store.list_open_orders()
            except Exception:
                open_orders = []
        base["open_order_count"] = len(open_orders)
        base["last_recovery_sources"] = sorted({
            str((getattr(order, "metadata", {}) or {}).get("last_recovery_source") or "")
            for order in open_orders
            if str((getattr(order, "metadata", {}) or {}).get("last_recovery_source") or "")
        })

        if self.portfolio_store is None:
            if execution_mode == "live":
                base["authoritative"] = False
                base["authority_reason"] = "live_runtime_state_unavailable"
            return base
        try:
            portfolio = self.portfolio_store.load()
        except Exception as exc:
            base.update(
                {
                    "authority_mode": "unknown",
                    "reconciliation_status": "error",
                    "last_reconciliation_error": str(exc),
                    "exchange_authority_required": execution_mode == "live",
                    "authoritative": False if execution_mode == "live" else True,
                    "authority_reason": "live_runtime_state_error" if execution_mode == "live" else "paper_local_authority",
                }
            )
            return base

        authority_mode = str(getattr(portfolio, "authority_mode", "") or ("paper_local" if execution_mode != "live" else "unknown"))
        reconciliation_status = str(getattr(portfolio, "reconciliation_status", "") or ("paper_local" if execution_mode != "live" else "unknown"))
        exchange_authority_required = execution_mode == "live"
        authoritative = not exchange_authority_required or reconciliation_status == "reconciled"
        authority_reason = "paper_local_authority" if not exchange_authority_required else str(reconciliation_status or "live_reconciliation_required")
        return {
            "execution_mode": execution_mode,
            "open_position_count": len(getattr(portfolio, "positions", {}) or {}),
            "open_order_count": len(open_orders),
            "last_recovery_sources": sorted({
                str((getattr(order, "metadata", {}) or {}).get("last_recovery_source") or "")
                for order in open_orders
                if str((getattr(order, "metadata", {}) or {}).get("last_recovery_source") or "")
            }),
            "reconciliation_enabled": reconciliation_enabled,
            "reconciliation_required": reconciliation_required,
            "exchange_authority_required": exchange_authority_required,
            "authority_mode": authority_mode,
            "reconciliation_status": reconciliation_status,
            "last_reconciled_at": getattr(portfolio, "last_reconciled_at", None),
            "exchange_position_count": int(getattr(portfolio, "exchange_position_count", 0) or 0),
            "reconciliation_mismatch_count": int(getattr(portfolio, "reconciliation_mismatch_count", 0) or 0),
            "last_reconciliation_error": str(getattr(portfolio, "last_reconciliation_error", "") or ""),
            "exchange_snapshot_hash": str(getattr(portfolio, "exchange_snapshot_hash", "") or ""),
            "authoritative": bool(authoritative),
            "authority_reason": authority_reason,
        }

    def runtime_gate(
        self,
        *,
        portfolio_state: Any | None = None,
        circuit_state: Any | None = None,
        authority: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the authoritative runtime trading gate for operator and controller use.

        The gate centralizes control-plane pause state, breaker state, and live
        authority so signal and execution controllers can make consistent,
        fail-closed decisions.
        """
        pause_state = self.pause_state()
        paused = bool(pause_state.get("paused", False))
        pause_reason = str(pause_state.get("pause_reason") or "")

        if circuit_state is None:
            circuit_payload = asdict(self.circuit_store.load())
        elif isinstance(circuit_state, dict):
            circuit_payload = dict(circuit_state)
        else:
            circuit_payload = dict(getattr(circuit_state, "__dict__", {}))
        circuit_triggered = bool(circuit_payload.get("triggered", False))
        circuit_reason = str(circuit_payload.get("reason") or "")

        if authority is None:
            live_runtime = self._live_runtime_summary()
            authoritative = bool(live_runtime.get("authoritative", False))
            authority_reason = str(live_runtime.get("authority_reason") or "")
        else:
            live_runtime = self._live_runtime_summary() if portfolio_state is None else {
                **self._live_runtime_summary(),
                "authority_mode": str(getattr(portfolio_state, "authority_mode", "") or self._live_runtime_summary().get("authority_mode", "unknown")),
                "reconciliation_status": str(getattr(portfolio_state, "reconciliation_status", "") or self._live_runtime_summary().get("reconciliation_status", "unknown")),
                "last_reconciled_at": getattr(portfolio_state, "last_reconciled_at", None),
                "exchange_position_count": int(getattr(portfolio_state, "exchange_position_count", 0) or 0),
                "reconciliation_mismatch_count": int(getattr(portfolio_state, "reconciliation_mismatch_count", 0) or 0),
                "last_reconciliation_error": str(getattr(portfolio_state, "last_reconciliation_error", "") or ""),
                "exchange_snapshot_hash": str(getattr(portfolio_state, "exchange_snapshot_hash", "") or ""),
            }
            authoritative = bool(authority.get("authoritative", False))
            authority_reason = str(authority.get("reason") or authority.get("authority_reason") or live_runtime.get("authority_reason") or "")
            live_runtime["authoritative"] = authoritative
            live_runtime["authority_reason"] = authority_reason

        blocked = False
        reason = ""
        status = "running"
        if paused:
            blocked = True
            reason = "control_plane_paused"
            status = "paused"
        elif circuit_triggered:
            blocked = True
            reason = "circuit_breaker_triggered"
            status = "blocked"
        elif str(live_runtime.get("execution_mode") or "paper").lower() == "live" and not authoritative:
            blocked = True
            reason = authority_reason or "live_reconciliation_required"
            status = "guarded"

        return {
            "blocked": blocked,
            "reason": reason,
            "status": status,
            "trading_enabled": not blocked,
            "paused": paused,
            "pause_reason": pause_reason,
            "circuit_triggered": circuit_triggered,
            "circuit_reason": circuit_reason,
            "live_authoritative": authoritative,
            "live_authority_reason": authority_reason,
            "live_runtime": live_runtime,
        }

    def summary(self) -> dict[str, Any]:
        pause_state = self.pause_state()
        circuit_breaker = asdict(self.circuit_store.load())
        live_runtime = self._live_runtime_summary()
        runtime_gate = self.runtime_gate(circuit_state=circuit_breaker, authority={
            "authoritative": live_runtime.get("authoritative", False),
            "reason": live_runtime.get("authority_reason", ""),
        })
        return {
            "paused": pause_state["paused"],
            "pause_reason": pause_state["pause_reason"],
            "updated_at": pause_state["updated_at"],
            "updated_by": pause_state["updated_by"],
            "circuit_breaker": circuit_breaker,
            "symbol_summary": self._symbol_summary(),
            "recent_actions": self.recent_actions(limit=25),
            "runtime_state_owners": {
                "market_data": "market_data_controller",
                "signal_runtime": "signal_generator_controller",
                "backtest_runtime": "backtest_controller",
                "paper_execution_runtime": "live_trading_controller",
                "execution_orders": "live_trading_controller",
                "execution_portfolio": "live_trading_controller",
                "control_plane": "control_plane_store",
            },
            "decision_lineage_preview": self._decision_lineage_preview(limit=20),
            "shadow_candidate_preview": self._shadow_candidate_preview(limit=20),
            "runtime_observability": self._runtime_observability_summary(limit=50),
            "live_runtime": live_runtime,
            "runtime_gate": runtime_gate,
        }
