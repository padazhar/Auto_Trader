from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any

from core.artifact_contract import (
    active_manifest_truth,
    incumbent_pointer_path,
    paper_bootstrap_active_authority_truth,
    load_json,
    resolve_active_manifest_path,
    resolve_artifact_path,
    validate_artifact_bundle,
)
from core.log_manager import get_logger
from core.path_manager import PathManager
from core.shared_utils import atomic_write_bytes, load_json as load_runtime_json, now_utc_ts
from monitoring.publisher import MonitorPublisher

VALID_STATUSES = {"idle", "queued", "running", "success", "failed", "ineligible"}
DIAGNOSTIC_CATEGORIES = {
    "promoted",
    "candidate_only",
    "missing_active_manifest",
    "active_manifest_invalid",
    "active_without_registry_approval",
    "feature_incompatible",
    "queued",
    "running",
    "failed",
    "ineligible",
    "idle",
    "unknown",
}


@dataclass
class RegistryEntry:
    model_type: str
    symbol: str
    interval: str
    status: str = "idle"
    created_at: float = field(default_factory=now_utc_ts)
    queued_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    last_success_at: float | None = None
    rows_used: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    attempt_count: int = 0
    candidate_id: str | None = None
    promoted_at: str | None = None
    manifest_path: str | None = None
    feature_set_version: str | None = None
    feature_set_signature: str | None = None
    promotion_attempted: bool = False
    promotion_approved: bool = False
    promotion_reasons: list[str] = field(default_factory=list)
    latest_candidate_promotion_approved: bool = False
    latest_candidate_promoted: bool = False
    latest_candidate_paper_evidence_status: str | None = None
    latest_candidate_paper_bootstrap_only: bool = False
    latest_candidate_paper_evidence_sufficient: bool | None = None
    pointer_path: str | None = None
    candidate_saved: bool = False
    candidate_manifest_valid: bool = False
    active_manifest_path: str | None = None
    active_manifest_valid: bool = False
    active_incumbent_candidate_id: str | None = None
    active_incumbent_promoted_at: str | None = None
    active_incumbent_approval_source: str | None = None
    active_incumbent_deployment_stage: str | None = None
    active_incumbent_paper_execution_only: bool = False
    execution_ready: bool = False
    registry_truth_status: str = "uninitialized"
    truth_reasons: list[str] = field(default_factory=list)
    ineligible_reason: str | None = None
    ineligible_at: float | None = None
    ineligible_reference_anchor_ts_ms: int | None = None


class TrainingRegistry:
    def __init__(self, paths: PathManager) -> None:
        self.paths = paths
        self.monitor = MonitorPublisher(paths)
        self.logger = get_logger("core.training_registry", paths)
        self._lock = RLock()
        self._entries: dict[str, RegistryEntry] = {}
        self._load()
        self._publish_snapshot()

    def _warn_invalid_entry(self, key: str, item: Any, error: Exception | None = None) -> None:
        message = f"Skipped invalid training registry entry: {key}"
        self.logger.warning("%s payload=%r error=%s", message, item, error)
        try:
            self.monitor.publish_event(
                level="WARNING",
                component="training_registry",
                category="invalid_registry_entry",
                message=message,
                data={"key": key, "error": str(error) if error else None},
                alert=False,
            )
        except Exception:
            self.logger.exception("Failed to publish invalid registry entry warning")

    def _load(self) -> None:
        raw = load_runtime_json(self.paths.registry_path, default={}) or {}
        if not isinstance(raw, dict):
            self.logger.warning("Training registry payload is not a mapping; ignoring file")
            return
        with self._lock:
            for key, item in raw.items():
                if not isinstance(item, dict):
                    self._warn_invalid_entry(str(key), item)
                    continue
                try:
                    self._entries[key] = RegistryEntry(**item)
                except Exception as exc:
                    self._warn_invalid_entry(str(key), item, exc)
                    continue
            # Migrate persisted registry truth from the active incumbent authority.
            # This makes a valid incumbent immediately available after restart even
            # when the latest candidate was rejected before this contract existed.
            changed = False
            for entry in self._entries.values():
                before = asdict(entry)
                self._refresh_entry_truth_locked(entry)
                changed = changed or before != asdict(entry)
            if changed:
                self._save_locked()

    def _save_locked(self) -> None:
        payload = {k: asdict(v) for k, v in self._entries.items()}
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        atomic_write_bytes(self.paths.registry_path, encoded)

    def _key(self, model_type: str, symbol: str, interval: str) -> str:
        return f"{model_type}:{symbol}:{interval}"

    def _get_or_create_locked(self, model_type: str, symbol: str, interval: str) -> RegistryEntry:
        key = self._key(model_type, symbol, interval)
        if key not in self._entries:
            self._entries[key] = RegistryEntry(model_type=model_type, symbol=symbol, interval=interval)
        return self._entries[key]

    def _extract_manifest_info(self, artifacts: dict[str, Any]) -> dict[str, Any]:
        normalized = {str(k): str(v) for k, v in (artifacts or {}).items() if v is not None}
        manifest_candidate = normalized.get("manifest")
        if not manifest_candidate:
            for key in ("direction_manifest", "meta_manifest"):
                if normalized.get(key):
                    manifest_candidate = normalized[key]
                    break
        if not manifest_candidate:
            return {}
        manifest_path = resolve_artifact_path(self.paths, manifest_candidate)
        manifest = load_json(manifest_path) or {}
        if not manifest:
            return {"manifest_path": str(manifest_path)}
        lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
        feature_set = manifest.get("feature_set") if isinstance(manifest.get("feature_set"), dict) else {}
        timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
        return {
            "manifest_path": str(manifest_path),
            "candidate_id": lifecycle.get("candidate_id") or normalized.get("candidate_id"),
            "promoted_at": lifecycle.get("promoted_at"),
            "feature_set_version": feature_set.get("version"),
            "feature_set_signature": feature_set.get("signature"),
            "manifest_created_at": timestamps.get("created_at"),
            "manifest_updated_at": timestamps.get("updated_at"),
        }

    @staticmethod
    def _extract_promotion_info(metrics: dict[str, Any]) -> dict[str, Any]:
        promotion = (metrics or {}).get("promotion") if isinstance((metrics or {}).get("promotion"), dict) else {}
        pointer = promotion.get("pointer") if isinstance(promotion.get("pointer"), dict) else {}
        paper_governance = promotion.get("paper_evidence_governance") if isinstance(promotion.get("paper_evidence_governance"), dict) else {}
        return {
            "attempted": bool(promotion.get("attempted", False)),
            "approved": bool(promotion.get("approved", False)),
            "promoted": bool(promotion.get("promoted", False)),
            "reasons": [str(reason) for reason in promotion.get("reasons", [])] if isinstance(promotion.get("reasons", []), list) else [],
            "candidate_id": promotion.get("candidate_id"),
            "manifest_path": promotion.get("manifest_path"),
            "pointer_path": promotion.get("pointer_path"),
            "promoted_at": pointer.get("promoted_at"),
            "paper_evidence_status": promotion.get("paper_evidence_status") or paper_governance.get("status"),
            "paper_bootstrap_only": bool(promotion.get("paper_bootstrap_only", paper_governance.get("paper_bootstrap_only", False))),
            "paper_evidence_sufficient": promotion.get("paper_evidence_sufficient", paper_governance.get("evidence_sufficient")),
        }

    def _active_truth_locked(self, entry: RegistryEntry) -> dict[str, Any]:
        # GBM execution consumes the role-specific direction incumbent.  The
        # bundle pointer is useful for diagnostics, but it must not make a valid
        # direction incumbent look missing in the execution-readiness matrix.
        model_role = "direction" if str(entry.model_type).lower() == "gbm" else None
        try:
            return active_manifest_truth(self.paths, entry.model_type, entry.symbol, entry.interval, model_role=model_role)
        except Exception as exc:
            return {
                "execution_ready": False,
                "reason": f"active_manifest_truth_error:{exc}",
                "active_manifest_path": None,
            }

    def _candidate_manifest_valid_locked(self, entry: RegistryEntry) -> tuple[bool, str]:
        if not entry.manifest_path:
            return False, "candidate_manifest_missing"
        try:
            manifest_path = resolve_artifact_path(self.paths, entry.manifest_path)
        except Exception as exc:
            return False, f"candidate_manifest_path_error:{exc}"
        if not manifest_path.exists():
            return False, "candidate_manifest_missing"
        try:
            valid, reason = validate_artifact_bundle(self.paths, manifest_path)
        except Exception as exc:
            return False, f"candidate_manifest_validation_error:{exc}"
        return bool(valid), str(reason)

    @staticmethod
    def _truth_status_from_parts(
        *,
        status: str,
        candidate_saved: bool,
        candidate_manifest_valid: bool,
        promotion_attempted: bool,
        promotion_approved: bool,
        active_manifest_valid: bool,
    ) -> str:
        status = str(status or "idle").lower()
        # Active incumbent authority is independent from the lifecycle of the
        # latest candidate. A queued, running, rejected, or failed retrain must
        # not withdraw a still-valid approved incumbent from execution readiness.
        if active_manifest_valid and promotion_approved:
            return "execution_ready"
        if status in {"queued", "running", "failed", "ineligible", "idle"}:
            return status
        if active_manifest_valid and not promotion_approved:
            return "active_without_registry_approval"
        if promotion_attempted and not promotion_approved:
            return "promotion_rejected"
        if candidate_saved and candidate_manifest_valid:
            return "candidate_only"
        if candidate_saved and not candidate_manifest_valid:
            return "candidate_invalid"
        if status == "success":
            return "success_without_candidate"
        return "unknown"

    def _active_incumbent_approval_truth_locked(self, entry: RegistryEntry, active_truth: dict[str, Any]) -> dict[str, Any]:
        """Return approval truth for the active incumbent independently of the latest candidate.

        A newly rejected or running candidate must not revoke an already promoted
        incumbent.  Recovery is therefore grounded in the active pointer and its
        incumbent manifest lifecycle, not in equality with ``entry.candidate_id``
        (which represents the latest candidate).  A narrow legacy fallback keeps
        previously supported matching-entry incumbents backward compatible.
        """
        result: dict[str, Any] = {
            "approved": False,
            "candidate_id": None,
            "promoted_at": None,
            "source": None,
            "deployment_stage": None,
            "paper_execution_only": False,
        }
        # Paper-bootstrap authority is proven by the active pointer and manifest
        # lifecycle, independently of the latest candidate registry outcome.
        # This repairs a persisted stale-registry condition observed at runtime:
        # valid paper-bootstrap manifests were execution-ready for paper mode but
        # later rejected candidates left the registry row non-ready.
        model_role = "direction" if str(entry.model_type).lower() == "gbm" else None
        bootstrap_authority = paper_bootstrap_active_authority_truth(
            self.paths, entry.model_type, entry.symbol, entry.interval, model_role=model_role
        )
        if bool(bootstrap_authority.get("authority_approved", False)):
            result.update(
                {
                    "approved": True,
                    "candidate_id": bootstrap_authority.get("candidate_id"),
                    "promoted_at": bootstrap_authority.get("promoted_at"),
                    "source": bootstrap_authority.get("authority_source"),
                    "deployment_stage": "paper_bootstrap",
                    "paper_execution_only": True,
                    "manifest_path": bootstrap_authority.get("active_manifest_path"),
                    "reason": bootstrap_authority.get("authority_reason"),
                }
            )
            return result

        if not bool(active_truth.get("execution_ready", False)):
            return result
        active_path_raw = active_truth.get("active_manifest_path")
        if not active_path_raw:
            return result
        try:
            active_path = Path(str(active_path_raw)).resolve()
            active_manifest = load_json(active_path) or {}
        except Exception:
            return result
        if not isinstance(active_manifest, dict):
            return result
        lifecycle = active_manifest.get("lifecycle") if isinstance(active_manifest.get("lifecycle"), dict) else {}
        deployment_stage = str(lifecycle.get("stage") or "").strip().lower()
        if deployment_stage not in {"incumbent", "paper_bootstrap"}:
            return result
        paper_execution_only = bool(lifecycle.get("paper_execution_only", False))
        if deployment_stage == "paper_bootstrap":
            runtime_cfg = getattr(getattr(self.paths, "config", None), "runtime", None)
            execution_mode = str(getattr(runtime_cfg, "execution_mode", "paper") or "paper").strip().lower()
            if execution_mode != "paper" or not paper_execution_only or bool(lifecycle.get("live_eligible", True)):
                return result
        active_candidate_id = str(lifecycle.get("candidate_id") or "").strip()
        promoted_at = str(lifecycle.get("promoted_at") or "").strip()
        if not active_candidate_id or not promoted_at:
            return result
        result.update(
            {
                "candidate_id": active_candidate_id,
                "promoted_at": promoted_at,
                "deployment_stage": deployment_stage,
                "paper_execution_only": paper_execution_only,
            }
        )

        try:
            pointer = load_json(incumbent_pointer_path(self.paths, entry.model_type, entry.symbol, entry.interval, model_role=model_role)) or {}
        except Exception:
            pointer = {}
        if pointer.get("manifest_path"):
            try:
                pointer_manifest = resolve_artifact_path(self.paths, str(pointer["manifest_path"])).resolve()
            except Exception:
                return result
            pointer_candidate_id = str(pointer.get("candidate_id") or "").strip()
            pointer_promoted_at = str(pointer.get("promoted_at") or "").strip()
            if (
                pointer_manifest == active_path
                and pointer_candidate_id == active_candidate_id
                and pointer_promoted_at == promoted_at
            ):
                result.update(
                    {
                        "approved": True,
                        "source": "active_paper_bootstrap_pointer" if deployment_stage == "paper_bootstrap" else "active_incumbent_pointer",
                    }
                )
            return result

        # Backward-compatible recovery for legacy incumbent records that predate
        # pointer persistence: retain the old matching-candidate safeguard only.
        registry_candidate_id = str(entry.candidate_id or "").strip()
        if deployment_stage == "incumbent" and registry_candidate_id and registry_candidate_id == active_candidate_id:
            result.update({"approved": True, "source": "legacy_matching_incumbent"})
        return result

    def _refresh_entry_truth_locked(self, entry: RegistryEntry, *, promotion_info: dict[str, Any] | None = None) -> dict[str, Any]:
        promotion_info = promotion_info if promotion_info is not None else self._extract_promotion_info(entry.metrics)
        candidate_valid, candidate_reason = self._candidate_manifest_valid_locked(entry)
        active_truth = self._active_truth_locked(entry)
        raw_approved = bool(promotion_info.get("approved", False))
        raw_promoted = bool(promotion_info.get("promoted", False))
        active_approval = self._active_incumbent_approval_truth_locked(entry, active_truth)
        recovered_approved = bool(active_approval.get("approved", False))
        active_valid = bool(active_truth.get("execution_ready", False) or recovered_approved)

        reasons = [str(reason) for reason in (entry.promotion_reasons or [])]
        active_reason = str(
            active_approval.get("reason")
            if recovered_approved
            else (active_truth.get("reason") or "active_manifest_missing")
        )
        # Execution readiness is authorised only by the active incumbent's own
        # validated promotion provenance.  Latest-candidate approval remains
        # candidate telemetry and cannot make an unrelated incumbent executable.
        effective_approved = bool(recovered_approved)
        if raw_approved and not active_valid and active_reason not in reasons:
            reasons.append(active_reason)
        if raw_promoted and not active_valid and "promoted_without_valid_active_manifest" not in reasons:
            reasons.append("promoted_without_valid_active_manifest")

        entry.candidate_saved = bool(entry.manifest_path)
        entry.candidate_manifest_valid = bool(candidate_valid)
        entry.latest_candidate_promotion_approved = bool(raw_approved)
        entry.latest_candidate_promoted = bool(raw_promoted)
        entry.latest_candidate_paper_evidence_status = str(promotion_info.get("paper_evidence_status") or "") or None
        entry.latest_candidate_paper_bootstrap_only = bool(promotion_info.get("paper_bootstrap_only", False))
        paper_evidence_sufficient = promotion_info.get("paper_evidence_sufficient")
        entry.latest_candidate_paper_evidence_sufficient = (
            bool(paper_evidence_sufficient) if paper_evidence_sufficient is not None else None
        )
        entry.active_manifest_path = str(active_approval.get("manifest_path") or active_truth.get("active_manifest_path") or "") or None
        entry.active_manifest_valid = bool(active_valid)
        entry.active_incumbent_candidate_id = str(active_approval.get("candidate_id") or "") or None
        entry.active_incumbent_promoted_at = str(active_approval.get("promoted_at") or "") or None
        entry.active_incumbent_approval_source = str(active_approval.get("source") or "") or None
        entry.active_incumbent_deployment_stage = str(active_approval.get("deployment_stage") or "") or None
        entry.active_incumbent_paper_execution_only = bool(active_approval.get("paper_execution_only", False))
        entry.execution_ready = bool(active_valid and effective_approved)
        # ``promotion_approved`` remains the effective active-authority gate for
        # backward-compatible readiness consumers. Latest-candidate approval is
        # exposed separately so candidate rejection remains observable.
        entry.promotion_approved = bool(effective_approved and active_valid)
        entry.promotion_reasons = reasons

        truth_reasons: list[str] = []
        if entry.candidate_saved and not candidate_valid:
            truth_reasons.append(candidate_reason)
        if not active_valid:
            truth_reasons.append(active_reason)
        if raw_approved and not active_valid:
            truth_reasons.append("approval_downgraded_until_active_manifest_valid")
        if recovered_approved and not raw_approved:
            truth_reasons.append("active_incumbent_retained_while_latest_candidate_not_approved")
        if entry.active_incumbent_paper_execution_only:
            truth_reasons.append("paper_bootstrap_execution_only_non_live")
        entry.truth_reasons = [str(reason) for reason in truth_reasons if reason]
        entry.registry_truth_status = self._truth_status_from_parts(
            status=entry.status,
            candidate_saved=entry.candidate_saved,
            candidate_manifest_valid=entry.candidate_manifest_valid,
            promotion_attempted=entry.promotion_attempted,
            promotion_approved=entry.promotion_approved,
            active_manifest_valid=entry.active_manifest_valid,
        )
        return {
            "raw_approved": raw_approved,
            "raw_promoted": raw_promoted,
            "recovered_approved": recovered_approved,
            "active_reason": active_reason,
            "candidate_reason": candidate_reason,
            "active_incumbent_candidate_id": entry.active_incumbent_candidate_id,
            "active_incumbent_approval_source": entry.active_incumbent_approval_source,
            "active_incumbent_deployment_stage": entry.active_incumbent_deployment_stage,
            "active_incumbent_paper_execution_only": entry.active_incumbent_paper_execution_only,
        }

    def _latest_market_timestamp_ms(self, symbol: str, interval: str) -> int | None:
        """Return the latest anchor timestamp persisted for the current symbol/interval.

        This gives the scheduler a stable trained-through market reference that is
        independent of manifest file mtimes or wall-clock registry timestamps.
        """
        market_db_path = getattr(self.paths, "market_db_path", None)
        if not callable(market_db_path):
            return None
        try:
            db_path = market_db_path(symbol)
        except Exception:
            return None
        if db_path is None:
            return None
        try:
            db_exists = Path(db_path).exists()
        except Exception:
            db_exists = False
        if not db_exists:
            return None

        try:
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    f'SELECT MAX(timestamp) FROM "klines_{str(interval)}"'
                ).fetchone()
            value = int((row or [0])[0] or 0)
            return value if value > 0 else None
        except Exception:
            return None


    @staticmethod
    def _model_state_contract_from_entry(entry: RegistryEntry, diagnostic: dict[str, Any]) -> dict[str, Any]:
        """Return explicit model-state buckets for dashboard/report consumers.

        Step 2 contract: candidate, promotion, active-manifest, and execution
        eligibility states must be separate.  A candidate can be valid without
        being promoted; a promotion can be approved in metrics but still be
        downgraded until a valid active manifest exists; and execution readiness
        is true only when the active manifest validates.
        """
        if entry.candidate_saved:
            candidate_state = "saved_valid" if entry.candidate_manifest_valid else "saved_invalid"
        else:
            candidate_state = "not_saved"

        raw_approved = bool(entry.latest_candidate_promotion_approved)
        raw_promoted = bool(entry.latest_candidate_promoted)
        if raw_approved and raw_promoted and entry.execution_ready:
            promotion_state = "approved_active"
        elif raw_approved or raw_promoted:
            promotion_state = "approved_but_active_manifest_invalid"
        elif entry.promotion_attempted:
            promotion_state = "rejected"
        else:
            promotion_state = "not_attempted"

        active_path = diagnostic.get("active_manifest_path") or entry.active_manifest_path
        active_valid = bool(diagnostic.get("active_manifest_valid", entry.active_manifest_valid))
        if active_valid:
            active_manifest_state = "valid"
        elif active_path:
            active_manifest_state = "invalid"
        else:
            active_manifest_state = "missing"

        execution_eligibility_state = "eligible" if entry.execution_ready else "not_eligible"
        lifecycle_state = str(entry.registry_truth_status or diagnostic.get("diagnostic_category") or "unknown")
        if entry.execution_ready and entry.active_incumbent_approval_source:
            active_incumbent_state = "approved_active"
        elif active_valid:
            active_incumbent_state = "valid_without_approval"
        elif active_path:
            active_incumbent_state = "invalid"
        else:
            active_incumbent_state = "missing"
        return {
            "candidate_state": candidate_state,
            "promotion_state": promotion_state,
            "active_manifest_state": active_manifest_state,
            "execution_eligibility_state": execution_eligibility_state,
            "lifecycle_state": lifecycle_state,
            "active_incumbent_state": active_incumbent_state,
            "active_incumbent_candidate_id": entry.active_incumbent_candidate_id,
            "latest_candidate_promotion_approved": bool(entry.latest_candidate_promotion_approved),
            "latest_candidate_promoted": bool(entry.latest_candidate_promoted),
            "state_contract_version": "model_state_v1",
        }

    def _entry_diagnostic_locked(self, entry: RegistryEntry) -> dict[str, Any]:
        """Classify why a registry row is or is not execution-ready.

        This is observability-only: it never promotes, deletes, retries, or changes
        scheduler state. The dashboard uses this to distinguish idle work from
        candidate-only artifacts, missing active manifests, and invalid manifests.
        """
        status = str(entry.status or "idle").lower()
        active_manifest_path: Path | None = None
        active_manifest_valid = False
        active_manifest_reason = "active_manifest_missing"
        try:
            model_role = "direction" if str(entry.model_type).lower() == "gbm" else None
            active_manifest_path = resolve_active_manifest_path(self.paths, entry.model_type, entry.symbol, entry.interval, model_role=model_role)
        except Exception:
            active_manifest_path = None

        if active_manifest_path is not None and active_manifest_path.exists():
            try:
                active_manifest_valid, active_manifest_reason = validate_artifact_bundle(self.paths, active_manifest_path)
            except Exception as exc:
                active_manifest_valid = False
                active_manifest_reason = f"active_manifest_validation_error:{exc}"

        candidate_manifest_exists = False
        if entry.manifest_path:
            try:
                candidate_manifest_exists = resolve_artifact_path(self.paths, entry.manifest_path).exists()
            except Exception:
                candidate_manifest_exists = False

        if status in {"queued", "running"}:
            category = status
            detail = f"scheduler_status:{status}"
        elif bool(entry.execution_ready):
            category = "promoted"
            detail = "active_manifest_valid"
        elif active_manifest_valid and not bool(entry.promotion_approved):
            category = "active_without_registry_approval"
            detail = "active_manifest_valid_without_registry_approval"
        elif active_manifest_path is not None and active_manifest_path.exists():
            lower_reason = str(active_manifest_reason or "").lower()
            category = "feature_incompatible" if "feature" in lower_reason or "signature" in lower_reason else "active_manifest_invalid"
            detail = str(active_manifest_reason or "active_manifest_invalid")
        elif status == "success" and str(entry.registry_truth_status) == "candidate_invalid":
            category = "active_manifest_invalid"
            detail = ",".join(entry.truth_reasons or []) or "candidate_manifest_invalid"
        elif status == "success" and not bool(entry.promotion_approved):
            category = "candidate_only" if candidate_manifest_exists or entry.promotion_attempted or entry.promotion_reasons else "missing_active_manifest"
            detail = ",".join(str(r) for r in (entry.promotion_reasons or entry.truth_reasons)) or "success_without_active_manifest"
        elif status == "failed":
            category = "failed"
            detail = str(entry.last_error or "failed")
        elif status == "ineligible":
            category = "ineligible"
            detail = str(entry.ineligible_reason or entry.last_error or "statistical_ineligible")
        elif status == "idle":
            category = "missing_active_manifest"
            detail = "idle_without_active_manifest"
        else:
            category = status if status in DIAGNOSTIC_CATEGORIES else "unknown"
            detail = f"registry_status:{status}"

        return {
            "diagnostic_category": category,
            "diagnostic_detail": detail,
            "active_manifest_path": str(active_manifest_path) if active_manifest_path is not None else None,
            "active_manifest_valid": bool(active_manifest_valid),
            "active_manifest_reason": str(active_manifest_reason),
            "candidate_manifest_exists": bool(candidate_manifest_exists),
            "has_candidate_manifest": bool(entry.manifest_path),
            "candidate_only_is_execution_ready": False,
            "execution_ready": bool(entry.execution_ready),
            "readiness_matrix_stage": category,
        }

    def _dashboard_entry_payload_locked(self, entry: RegistryEntry) -> dict[str, Any]:
        try:
            self._refresh_entry_truth_locked(entry)
        except Exception:
            self.logger.exception("failed to refresh registry truth model=%s symbol=%s interval=%s", entry.model_type, entry.symbol, entry.interval)
        payload = asdict(entry)
        diagnostic = self._entry_diagnostic_locked(entry)
        payload.update(diagnostic)
        payload.update(self._model_state_contract_from_entry(entry, diagnostic))
        return payload

    def _model_versions_payload_locked(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for entry in self._entries.values():
            if not entry.manifest_path:
                continue
            diagnostic = self._entry_diagnostic_locked(entry)
            model_bucket = payload.setdefault(entry.model_type, {})
            symbol_bucket = model_bucket.setdefault(entry.symbol.upper(), {})
            symbol_bucket[entry.interval] = {
                "status": entry.status,
                "manifest_path": entry.manifest_path,
                "candidate_id": entry.candidate_id,
                "promoted_at": entry.promoted_at,
                "feature_set_version": entry.feature_set_version,
                "feature_set_signature": entry.feature_set_signature,
                "rows_used": entry.rows_used,
                "last_success_at": entry.last_success_at,
                "promotion_attempted": entry.promotion_attempted,
                "promotion_approved": entry.promotion_approved,
                "promotion_reasons": list(entry.promotion_reasons),
                "pointer_path": entry.pointer_path,
                "candidate_saved": entry.candidate_saved,
                "candidate_manifest_valid": entry.candidate_manifest_valid,
                "active_manifest_path": entry.active_manifest_path,
                "active_manifest_valid": entry.active_manifest_valid,
                "active_incumbent_candidate_id": entry.active_incumbent_candidate_id,
                "active_incumbent_promoted_at": entry.active_incumbent_promoted_at,
                "active_incumbent_approval_source": entry.active_incumbent_approval_source,
                "active_incumbent_deployment_stage": entry.active_incumbent_deployment_stage,
                "active_incumbent_paper_execution_only": entry.active_incumbent_paper_execution_only,
                "latest_candidate_promotion_approved": entry.latest_candidate_promotion_approved,
                "latest_candidate_promoted": entry.latest_candidate_promoted,
                "latest_candidate_paper_evidence_status": entry.latest_candidate_paper_evidence_status,
                "latest_candidate_paper_bootstrap_only": entry.latest_candidate_paper_bootstrap_only,
                "latest_candidate_paper_evidence_sufficient": entry.latest_candidate_paper_evidence_sufficient,
                "execution_ready": entry.execution_ready,
                "registry_truth_status": entry.registry_truth_status,
                "truth_reasons": list(entry.truth_reasons),
                "ineligible_reason": entry.ineligible_reason,
                "ineligible_at": entry.ineligible_at,
                "ineligible_reference_anchor_ts_ms": entry.ineligible_reference_anchor_ts_ms,
                "diagnostic_category": diagnostic.get("diagnostic_category"),
                "diagnostic_detail": diagnostic.get("diagnostic_detail"),
                **self._model_state_contract_from_entry(entry, diagnostic),
            }
        return payload

    def _snapshot_payload_locked(self) -> dict[str, Any]:
        entries = [self._dashboard_entry_payload_locked(entry) for entry in self._entries.values()]
        counts = {status: 0 for status in VALID_STATUSES}
        counts_by_model: dict[str, dict[str, int]] = {}
        diagnostic_counts: dict[str, int] = {category: 0 for category in DIAGNOSTIC_CATEGORIES}
        diagnostic_counts_by_model: dict[str, dict[str, int]] = {}
        for entry, payload in zip(self._entries.values(), entries):
            counts[entry.status] = counts.get(entry.status, 0) + 1
            bucket = counts_by_model.setdefault(entry.model_type, {status: 0 for status in VALID_STATUSES})
            bucket[entry.status] = bucket.get(entry.status, 0) + 1
            category = str(payload.get("diagnostic_category") or "unknown")
            diagnostic_counts[category] = diagnostic_counts.get(category, 0) + 1
            diag_bucket = diagnostic_counts_by_model.setdefault(entry.model_type, {category: 0 for category in DIAGNOSTIC_CATEGORIES})
            diag_bucket[category] = diag_bucket.get(category, 0) + 1
        return {
            "counts": counts,
            "counts_by_model": counts_by_model,
            "diagnostic_counts": diagnostic_counts,
            "diagnostic_counts_by_model": diagnostic_counts_by_model,
            "entries": entries,
            "model_versions": self._model_versions_payload_locked(),
        }

    def _publish_snapshot(self) -> None:
        with self._lock:
            payload = self._snapshot_payload_locked()
        self.monitor.update_registry(
            counts=payload["counts"],
            counts_by_model=payload["counts_by_model"],
            diagnostic_counts=payload.get("diagnostic_counts"),
            diagnostic_counts_by_model=payload.get("diagnostic_counts_by_model"),
            entries=payload["entries"],
        )
        self.monitor.update_model_versions(payload.get("model_versions", {}))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"entries": {k: asdict(v) for k, v in self._entries.items()}}

    def mark_idle(self, model_type: str, symbol: str, interval: str) -> None:
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            entry.status = "idle"
            self._refresh_entry_truth_locked(entry)
            self._save_locked()
        self._publish_snapshot()

    def mark_queued(self, model_type: str, symbol: str, interval: str) -> None:
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            entry.status = "queued"
            entry.queued_at = now_utc_ts()
            entry.last_error = None
            self._refresh_entry_truth_locked(entry)
            self._save_locked()
        self._publish_snapshot()

    def mark_running(self, model_type: str, symbol: str, interval: str) -> None:
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            entry.status = "running"
            entry.started_at = now_utc_ts()
            entry.attempt_count += 1
            self._refresh_entry_truth_locked(entry)
            self._save_locked()
        self._publish_snapshot()

    def mark_failed(self, model_type: str, symbol: str, interval: str, error: str) -> None:
        result_payload: dict[str, Any]
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            entry.status = "failed"
            entry.completed_at = now_utc_ts()
            entry.last_error = str(error)
            entry.ineligible_reason = None
            entry.ineligible_at = None
            entry.ineligible_reference_anchor_ts_ms = None
            self._refresh_entry_truth_locked(entry)
            self._save_locked()
            result_payload = {
                "ok": False,
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "rows_used": entry.rows_used,
                "error": str(error),
                "status": "failed",
                "updated_at": now_utc_ts(),
            }
        self._publish_snapshot()
        self.monitor.append_training_result(result_payload)

    def mark_ineligible(self, model_type: str, symbol: str, interval: str, reason: str) -> None:
        """Record a statistically ineligible training tuple without treating it as a runtime failure.

        This state is reserved for evidence-backed RF/GBM target-support failures.
        It preserves any active incumbent and gives the scheduler a market-data
        reference point so the same tuple is not immediately retried until new
        bars arrive.
        """
        result_payload: dict[str, Any]
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            now = now_utc_ts()
            reference_ts_ms = self._latest_market_timestamp_ms(symbol, interval)
            entry.status = "ineligible"
            entry.completed_at = now
            entry.last_error = str(reason)
            entry.ineligible_reason = str(reason)
            entry.ineligible_at = now
            entry.ineligible_reference_anchor_ts_ms = int(reference_ts_ms) if reference_ts_ms else None
            self._refresh_entry_truth_locked(entry)
            self._save_locked()
            result_payload = {
                "ok": False,
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "rows_used": entry.rows_used,
                "error": str(reason),
                "status": "ineligible",
                "failure_class": "statistical_ineligible",
                "reference_anchor_ts_ms": entry.ineligible_reference_anchor_ts_ms,
                "updated_at": now,
            }
        self._publish_snapshot()
        self.monitor.append_training_result(result_payload)

    def mark_success(
        self,
        *,
        model_type: str,
        symbol: str,
        interval: str,
        metrics: dict[str, Any],
        artifacts: dict[str, Any],
        rows_used: int,
    ) -> None:
        result_payload: dict[str, Any]
        promotion_event_payload: dict[str, Any] | None = None
        candidate_event_payload: dict[str, Any] | None = None
        rejected_event_payload: dict[str, Any] | None = None
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            now = now_utc_ts()
            entry.status = "success"
            entry.completed_at = now
            entry.ineligible_reason = None
            entry.ineligible_at = None
            entry.ineligible_reference_anchor_ts_ms = None
            entry.last_success_at = now
            entry.rows_used = int(rows_used)
            entry.metrics = dict(metrics or {})
            entry.artifacts = dict(artifacts or {})
            entry.last_error = None

            trained_anchor_ts_ms = self._latest_market_timestamp_ms(symbol, interval)
            if trained_anchor_ts_ms is not None and int(trained_anchor_ts_ms) > 0:
                entry.metrics.setdefault("training_reference_anchor_ts_ms", int(trained_anchor_ts_ms))

            manifest_info = self._extract_manifest_info(entry.artifacts)
            promotion_info = self._extract_promotion_info(entry.metrics)

            entry.candidate_id = str(promotion_info.get("candidate_id") or manifest_info.get("candidate_id") or "") or None
            entry.manifest_path = str(promotion_info.get("manifest_path") or manifest_info.get("manifest_path") or "") or None
            entry.feature_set_version = manifest_info.get("feature_set_version")
            entry.feature_set_signature = manifest_info.get("feature_set_signature")
            entry.promotion_attempted = bool(promotion_info.get("attempted", False))
            entry.promotion_approved = bool(promotion_info.get("approved", False))
            entry.promotion_reasons = list(promotion_info.get("reasons", []))
            entry.pointer_path = str(promotion_info.get("pointer_path") or "") or None
            entry.promoted_at = str(promotion_info.get("promoted_at") or "") or None
            truth_context = self._refresh_entry_truth_locked(entry, promotion_info=promotion_info)
            self._save_locked()

            result_payload = {
                "ok": True,
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "rows_used": int(rows_used),
                "metrics": dict(metrics or {}),
                "status": "success",
                "updated_at": now,
                "candidate_id": entry.candidate_id,
                "manifest_path": entry.manifest_path,
                "feature_set_version": entry.feature_set_version,
                "promotion": {
                    "attempted": entry.promotion_attempted,
                    "approved": entry.latest_candidate_promotion_approved,
                    "promoted": entry.latest_candidate_promoted,
                    "reasons": list(entry.promotion_reasons),
                    "pointer_path": entry.pointer_path,
                    "promoted_at": entry.promoted_at,
                    "paper_evidence_status": entry.latest_candidate_paper_evidence_status,
                    "paper_bootstrap_only": entry.latest_candidate_paper_bootstrap_only,
                    "paper_evidence_sufficient": entry.latest_candidate_paper_evidence_sufficient,
                },
                "training_reference_anchor_ts_ms": entry.metrics.get("training_reference_anchor_ts_ms"),
                "candidate_saved": entry.candidate_saved,
                "candidate_manifest_valid": entry.candidate_manifest_valid,
                "active_manifest_path": entry.active_manifest_path,
                "active_manifest_valid": entry.active_manifest_valid,
                "active_incumbent_candidate_id": entry.active_incumbent_candidate_id,
                "active_incumbent_promoted_at": entry.active_incumbent_promoted_at,
                "active_incumbent_approval_source": entry.active_incumbent_approval_source,
                "active_incumbent_deployment_stage": entry.active_incumbent_deployment_stage,
                "active_incumbent_paper_execution_only": entry.active_incumbent_paper_execution_only,
                "latest_candidate_promotion_approved": entry.latest_candidate_promotion_approved,
                "latest_candidate_promoted": entry.latest_candidate_promoted,
                "latest_candidate_paper_evidence_status": entry.latest_candidate_paper_evidence_status,
                "latest_candidate_paper_bootstrap_only": entry.latest_candidate_paper_bootstrap_only,
                "latest_candidate_paper_evidence_sufficient": entry.latest_candidate_paper_evidence_sufficient,
                "execution_ready": entry.execution_ready,
                "registry_truth_status": entry.registry_truth_status,
                "truth_reasons": list(entry.truth_reasons),
                "model_state_contract": self._model_state_contract_from_entry(entry, self._entry_diagnostic_locked(entry)),
            }

            base_event_data = {
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "candidate_id": entry.candidate_id,
                "manifest_path": entry.manifest_path,
                "feature_set_version": entry.feature_set_version,
                "promotion_attempted": entry.promotion_attempted,
                "promotion_approved": entry.promotion_approved,
                "promotion_reasons": list(entry.promotion_reasons),
                "candidate_saved": entry.candidate_saved,
                "candidate_manifest_valid": entry.candidate_manifest_valid,
                "active_manifest_path": entry.active_manifest_path,
                "active_manifest_valid": entry.active_manifest_valid,
                "active_incumbent_candidate_id": entry.active_incumbent_candidate_id,
                "active_incumbent_promoted_at": entry.active_incumbent_promoted_at,
                "active_incumbent_approval_source": entry.active_incumbent_approval_source,
                "active_incumbent_deployment_stage": entry.active_incumbent_deployment_stage,
                "active_incumbent_paper_execution_only": entry.active_incumbent_paper_execution_only,
                "latest_candidate_promotion_approved": entry.latest_candidate_promotion_approved,
                "latest_candidate_promoted": entry.latest_candidate_promoted,
                "latest_candidate_paper_evidence_status": entry.latest_candidate_paper_evidence_status,
                "latest_candidate_paper_bootstrap_only": entry.latest_candidate_paper_bootstrap_only,
                "latest_candidate_paper_evidence_sufficient": entry.latest_candidate_paper_evidence_sufficient,
                "execution_ready": entry.execution_ready,
                "registry_truth_status": entry.registry_truth_status,
                "truth_reasons": list(entry.truth_reasons),
                "model_state_contract": self._model_state_contract_from_entry(entry, self._entry_diagnostic_locked(entry)),
            }
            candidate_event_payload = base_event_data
            latest_candidate_is_active = bool(
                entry.candidate_id
                and entry.active_incumbent_candidate_id
                and entry.candidate_id == entry.active_incumbent_candidate_id
            )
            if entry.latest_candidate_promotion_approved and latest_candidate_is_active and entry.execution_ready:
                promotion_event_payload = dict(base_event_data, promoted_at=entry.promoted_at, pointer_path=entry.pointer_path, execution_ready=True)
            elif entry.promotion_attempted or bool(truth_context.get("raw_approved")):
                rejected_event_payload = dict(base_event_data, registry_truth_status=entry.registry_truth_status, truth_reasons=list(entry.truth_reasons))

        self._publish_snapshot()
        self.monitor.append_training_result(result_payload)
        if candidate_event_payload is not None:
            self.monitor.publish_event(
                level="INFO",
                component="training_registry",
                category="candidate_written",
                message=f"Recorded {model_type} candidate for {symbol} {interval}",
                data=candidate_event_payload,
                alert=False,
            )
        if rejected_event_payload is not None:
            self.monitor.publish_event(
                level="INFO",
                component="training_registry",
                category="promotion_rejected",
                message=f"Candidate {model_type} artifact not promoted for {symbol} {interval}",
                data=rejected_event_payload,
                alert=False,
            )
        if promotion_event_payload is not None:
            self.monitor.publish_event(
                level="INFO",
                component="training_registry",
                category="artifact_promoted",
                message=f"Promoted {model_type} artifact for {symbol} {interval}",
                data=promotion_event_payload,
                alert=False,
            )

    def get_entry(self, model_type: str, symbol: str, interval: str) -> RegistryEntry:
        with self._lock:
            return self._get_or_create_locked(model_type, symbol, interval)

    def ensure_entry(self, model_type: str, symbol: str, interval: str) -> RegistryEntry:
        """Ensure a registry entry exists without changing its effective status."""
        with self._lock:
            entry = self._get_or_create_locked(model_type, symbol, interval)
            self._save_locked()
        self._publish_snapshot()
        return entry

    def ensure_entries(self, jobs: list[tuple[str, str, str]]) -> None:
        """Pre-seed registry rows for symbols and intervals that entered the active universe.

        This prevents the signal path from reporting `registry_status:missing`
        for symbols that are already part of the managed training universe but
        have not yet been queued by the scheduler.
        """
        changed = False
        with self._lock:
            for model_type, symbol, interval in jobs:
                key = self._key(model_type, symbol, interval)
                if key not in self._entries:
                    self._entries[key] = RegistryEntry(model_type=model_type, symbol=symbol, interval=interval)
                    changed = True
            if changed:
                self._save_locked()
        if changed:
            self._publish_snapshot()

    def all_entries(self) -> list[RegistryEntry]:
        with self._lock:
            return list(self._entries.values())

    def as_dashboard_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._dashboard_entry_payload_locked(entry) for entry in self._entries.values()]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._snapshot_payload_locked()["counts"])

    def counts_by_model(self) -> dict[str, dict[str, int]]:
        with self._lock:
            payload = self._snapshot_payload_locked()["counts_by_model"]
            return {model: dict(counts) for model, counts in payload.items()}
