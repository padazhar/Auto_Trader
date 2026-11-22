from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.artifact_contract import (
    load_json,
    paper_bootstrap_active_authority_truth,
    resolve_active_manifest_path,
    resolve_artifact_path,
    validate_execution_manifest,
    validate_manifest_feature_compatibility,
)
from features.feature_store_contract import resolve_feature_recipe_version


@dataclass(slots=True)
class ModelReadiness:
    model_type: str
    ready: bool
    status: str
    reason: str
    manifest_path: str | None = None
    deployment_stage: str = "unknown"
    registry_entry_exists: bool = False
    registry_truth_status: str = "unknown"
    registry_execution_ready: bool = False
    registry_promotion_approved: bool = False
    registry_active_manifest_valid: bool = False
    active_manifest_exists: bool = False
    active_manifest_valid: bool = False
    feature_compatible: bool = False
    paper_execution_only: bool = False
    active_authority_source: str | None = None
    registry_reconciled_from_active_authority: bool = False
    feature_contract_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnchorReadiness:
    ready: bool
    reason: str
    models: dict[str, ModelReadiness]


def _load_registry(paths) -> dict[str, Any]:
    path = getattr(paths, "registry_path", None)
    if path is None:
        return {}
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _entry_key(model_type: str, symbol: str, interval: str) -> str:
    return f"{model_type}:{symbol}:{interval}"




def _registry_execution_gate_reason(registry: dict[str, Any], entry: dict[str, Any], *, model_type: str) -> str | None:
    """Return None only when registry truth allows execution for an active manifest.

    Step 2 contract: a resolvable incumbent manifest is necessary but not
    sufficient.  Execution readiness must also be approved by the registry
    truth fields so a stale pointer or manually copied incumbent cannot create
    a false-ready inference path.
    """
    if not registry:
        return "registry_unavailable"
    if not entry:
        return "registry_entry_missing"
    truth = str(entry.get("registry_truth_status") or "").strip().lower()
    status = str(entry.get("status") or "").strip().lower()
    execution_ready = bool(entry.get("execution_ready", False))
    promotion_approved = bool(entry.get("promotion_approved", False))
    active_valid = bool(entry.get("active_manifest_valid", False))
    if truth == "execution_ready" and execution_ready and promotion_approved and active_valid:
        return None
    if not execution_ready:
        return f"registry_not_execution_ready:{truth or status or 'execution_ready_false'}"
    if not promotion_approved:
        return f"registry_not_execution_ready:{truth or status or 'promotion_not_approved'}"
    if not active_valid:
        return f"registry_not_execution_ready:{truth or status or 'active_manifest_invalid'}"
    if truth:
        return f"registry_not_execution_ready:{truth}"
    if status:
        return f"registry_not_execution_ready:{status}"
    return "registry_not_execution_ready:unknown"

def _manifest_feature_compatibility_reason(paths, manifest_path) -> str | None:
    """Return a feature-compatibility failure reason when the manifest is incompatible."""
    manifest = load_json(manifest_path) or {}
    if not manifest:
        return "manifest_unreadable"
    compatible, reasons = validate_manifest_feature_compatibility(paths, manifest)
    if compatible:
        return None
    return "feature_incompatible:" + ",".join(str(reason) for reason in reasons)


def _active_manifest_path_for_model(paths, *, model_type: str, symbol: str, interval: str):
    """Resolve the incumbent manifest that may be used for execution readiness.

    RF and policy/confirmatory artifacts have one active manifest. GBM has a
    bundle pointer plus role-specific direction/meta pointers. The execution
    scorer loads the role manifests, so readiness must primarily prove the
    active direction role exists and is execution-ready. The optional bundle
    manifest is validated separately for richer diagnostics when present.
    """
    normalized = str(model_type).lower()
    if normalized == "gbm":
        return resolve_active_manifest_path(paths, "gbm", symbol, interval, model_role="direction")
    return resolve_active_manifest_path(paths, normalized, symbol, interval)


def _manifest_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path)
    except Exception:
        return None


def _registry_status_from_entry(registry: dict[str, Any], entry: dict[str, Any]) -> str:
    return str(entry.get("status") or ("unknown" if registry else "registry_unavailable"))


def _registry_diagnostic_fields(registry: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    entry = entry if isinstance(entry, dict) else {}
    return {
        "registry_entry_exists": bool(entry),
        "registry_truth_status": str(entry.get("registry_truth_status") or ("unknown" if registry else "registry_unavailable")),
        "registry_execution_ready": bool(entry.get("execution_ready", False)),
        "registry_promotion_approved": bool(entry.get("promotion_approved", False)),
        "registry_active_manifest_valid": bool(entry.get("active_manifest_valid", False)),
    }




def _list_from_payload(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _feature_columns_from_manifest(paths, manifest: dict[str, Any]) -> list[str]:
    """Best-effort resolution of artifact feature columns for diagnostics only."""
    if not isinstance(manifest, dict):
        return []
    raw_path = manifest.get("feature_columns_path")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    raw_path = raw_path or artifacts.get("feature_columns")
    if not raw_path:
        return []
    feature_path = resolve_artifact_path(paths, str(raw_path))
    payload = load_json(feature_path) or {}
    return _list_from_payload(payload.get("feature_columns"))


def _feature_contract_diagnostics(paths, manifest_path: Path | None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return operator-facing feature contract diff details without changing readiness.

    This is diagnostic-only. It explains feature-set failures such as
    ``feature_set_missing`` by comparing the manifest feature metadata with the
    persisted artifact feature_columns.json when available. It does not alter,
    repair, or bypass the compatibility gate.
    """
    if manifest_path is None:
        return {
            "schema_version": "feature_contract_diagnostics.v1",
            "manifest_path": None,
            "manifest_readable": False,
            "reason": "manifest_path_missing",
        }
    loaded = manifest if isinstance(manifest, dict) else (load_json(manifest_path) or {})
    if not isinstance(loaded, dict) or not loaded:
        return {
            "schema_version": "feature_contract_diagnostics.v1",
            "manifest_path": _manifest_path_text(manifest_path),
            "manifest_readable": False,
            "reason": "manifest_unreadable",
        }
    feature_set = loaded.get("feature_set") if isinstance(loaded.get("feature_set"), dict) else {}
    manifest_columns = _list_from_payload(feature_set.get("columns"))
    artifact_columns = _feature_columns_from_manifest(paths, loaded)
    compatible, reasons = validate_manifest_feature_compatibility(paths, loaded)
    missing_from_manifest_metadata = [col for col in artifact_columns if col not in manifest_columns]
    extra_in_manifest_metadata = [col for col in manifest_columns if col not in artifact_columns]
    order_mismatch = bool(manifest_columns and artifact_columns and manifest_columns != artifact_columns and set(manifest_columns) == set(artifact_columns))
    return {
        "schema_version": "feature_contract_diagnostics.v1",
        "manifest_path": _manifest_path_text(manifest_path),
        "manifest_readable": True,
        "compatible": bool(compatible),
        "compatibility_reasons": [str(reason) for reason in reasons],
        "feature_set_present": bool(feature_set),
        "manifest_feature_count": len(manifest_columns),
        "artifact_feature_count": len(artifact_columns),
        "manifest_recipe_version": str(feature_set.get("recipe_version") or feature_set.get("version") or ""),
        "runtime_recipe_version": str(resolve_feature_recipe_version(getattr(paths, 'config', None))),
        "manifest_anchor_interval": feature_set.get("anchor_interval"),
        "feature_signature_present": bool(str(feature_set.get("signature") or "").strip()),
        "feature_signature": str(feature_set.get("signature") or ""),
        "missing_from_manifest_metadata": missing_from_manifest_metadata[:50],
        "extra_in_manifest_metadata": extra_in_manifest_metadata[:50],
        "feature_order_mismatch": order_mismatch,
        "manifest_feature_preview": manifest_columns[:20],
        "artifact_feature_preview": artifact_columns[:20],
    }


def _readiness_result(
    *,
    model_type: str,
    ready: bool,
    status: str,
    reason: str,
    registry_fields: dict[str, Any],
    manifest_path: Path | None = None,
    deployment_stage: str = "unknown",
    active_manifest_exists: bool = False,
    active_manifest_valid: bool = False,
    feature_compatible: bool = False,
    paper_execution_only: bool = False,
    active_authority_source: str | None = None,
    registry_reconciled_from_active_authority: bool = False,
    feature_contract_diagnostics: dict[str, Any] | None = None,
) -> ModelReadiness:
    return ModelReadiness(
        model_type=model_type,
        ready=ready,
        status=status,
        reason=reason,
        manifest_path=_manifest_path_text(manifest_path),
        deployment_stage=str(deployment_stage),
        registry_entry_exists=bool(registry_fields.get("registry_entry_exists", False)),
        registry_truth_status=str(registry_fields.get("registry_truth_status", "unknown")),
        registry_execution_ready=bool(registry_fields.get("registry_execution_ready", False)),
        registry_promotion_approved=bool(registry_fields.get("registry_promotion_approved", False)),
        registry_active_manifest_valid=bool(registry_fields.get("registry_active_manifest_valid", False)),
        active_manifest_exists=bool(active_manifest_exists),
        active_manifest_valid=bool(active_manifest_valid),
        feature_compatible=bool(feature_compatible),
        paper_execution_only=bool(paper_execution_only),
        active_authority_source=str(active_authority_source) if active_authority_source else None,
        registry_reconciled_from_active_authority=bool(registry_reconciled_from_active_authority),
        feature_contract_diagnostics=dict(feature_contract_diagnostics or {}),
    )


def _gbm_bundle_diagnostic(paths, *, symbol: str, interval: str) -> str | None:
    """Return a GBM bundle failure reason without replacing role readiness.

    Legacy systems may have only role-level GBM incumbents. That remains usable
    if the direction role validates. When a bundle pointer exists, validate it so
    a broken combined manifest cannot produce a false-ready dashboard state.
    """
    bundle_path = resolve_active_manifest_path(paths, "gbm", symbol, interval, model_role=None)
    if bundle_path is None:
        return None
    valid, reason, _manifest = validate_execution_manifest(paths, bundle_path)
    if not valid:
        return f"gbm_bundle:{reason}"
    feature_reason = _manifest_feature_compatibility_reason(paths, bundle_path)
    if feature_reason is not None:
        return f"gbm_bundle:{feature_reason}"
    return None


def model_registry_readiness(paths, *, model_type: str, symbol: str, interval: str) -> ModelReadiness:
    """Return execution readiness for the active incumbent artifact.

    The runtime gate is intentionally based on promoted incumbent manifests,
    not on latest candidate outputs or raw training-registry success entries.
    This keeps PART/candidate artifacts observable while preventing them from
    entering the execution candidate set.
    """
    normalized_model = str(model_type).lower()
    registry = _load_registry(paths)
    entry = registry.get(_entry_key(normalized_model, symbol, interval), {}) if isinstance(registry, dict) else {}
    entry = entry if isinstance(entry, dict) else {}
    status = _registry_status_from_entry(registry, entry)
    registry_fields = _registry_diagnostic_fields(registry, entry)

    manifest_path = _active_manifest_path_for_model(paths, model_type=normalized_model, symbol=symbol, interval=interval)
    if manifest_path is None:
        artifacts = entry.get("artifacts") if isinstance(entry, dict) else {}
        raw_manifest = None
        if isinstance(artifacts, dict):
            raw_manifest = artifacts.get("manifest") or artifacts.get("direction_manifest")
        if raw_manifest:
            diagnostic_path = resolve_artifact_path(paths, str(raw_manifest))
            if diagnostic_path.exists():
                feature_reason = _manifest_feature_compatibility_reason(paths, diagnostic_path)
                if feature_reason is not None:
                    return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=feature_reason, registry_fields=registry_fields, manifest_path=diagnostic_path, deployment_stage="candidate_or_diagnostic", active_manifest_exists=False, active_manifest_valid=False, feature_compatible=False, feature_contract_diagnostics=_feature_contract_diagnostics(paths, diagnostic_path))
                valid, reason, _manifest = validate_execution_manifest(paths, diagnostic_path)
                return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=reason if not valid else "active_manifest_missing", registry_fields=registry_fields, manifest_path=diagnostic_path, deployment_stage="candidate_or_diagnostic", active_manifest_exists=False, active_manifest_valid=False, feature_compatible=True, feature_contract_diagnostics=_feature_contract_diagnostics(paths, diagnostic_path))
        if status not in {"unknown", "registry_unavailable", "success"}:
            return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=f"registry_status:{status}", registry_fields=registry_fields)
        missing_reason = "gbm_direction_active_manifest_missing" if normalized_model == "gbm" else "active_manifest_missing"
        return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=missing_reason, registry_fields=registry_fields)

    valid, reason, manifest = validate_execution_manifest(paths, manifest_path)
    if not valid:
        return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=reason, registry_fields=registry_fields, manifest_path=manifest_path, deployment_stage=str((manifest.get("lifecycle") or {}).get("stage") or "unknown"), active_manifest_exists=True, active_manifest_valid=False, feature_compatible=False, feature_contract_diagnostics=_feature_contract_diagnostics(paths, manifest_path, manifest))

    feature_reason = _manifest_feature_compatibility_reason(paths, manifest_path)
    if feature_reason is not None:
        return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=feature_reason, registry_fields=registry_fields, manifest_path=manifest_path, deployment_stage=str((manifest.get("lifecycle") or {}).get("stage") or "unknown"), active_manifest_exists=True, active_manifest_valid=True, feature_compatible=False, feature_contract_diagnostics=_feature_contract_diagnostics(paths, manifest_path, manifest))

    lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
    deployment_stage = str(lifecycle.get("stage") or "incumbent")
    paper_execution_only = bool(deployment_stage == "paper_bootstrap" and lifecycle.get("paper_execution_only", False))

    if normalized_model == "gbm":
        bundle_reason = _gbm_bundle_diagnostic(paths, symbol=symbol, interval=interval)
        if bundle_reason is not None:
            return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=bundle_reason, registry_fields=registry_fields, manifest_path=manifest_path, deployment_stage=deployment_stage, active_manifest_exists=True, active_manifest_valid=True, feature_compatible=True, paper_execution_only=paper_execution_only)

    registry_gate_reason = _registry_execution_gate_reason(registry, entry, model_type=normalized_model)
    authority_source: str | None = None
    registry_reconciled = False
    if registry_gate_reason is not None and paper_execution_only:
        model_role = "direction" if normalized_model == "gbm" else None
        bootstrap_authority = paper_bootstrap_active_authority_truth(
            paths, normalized_model, symbol, interval, model_role=model_role
        )
        authority_path = bootstrap_authority.get("active_manifest_path")
        same_manifest = bool(authority_path and Path(str(authority_path)).resolve() == Path(manifest_path).resolve())
        if bool(bootstrap_authority.get("authority_approved", False)) and same_manifest:
            # The active pointer/manifest lifecycle is the deployment authority;
            # a later rejected candidate remains visible in registry diagnostics
            # but must not withdraw a valid paper-only bootstrap incumbent.
            authority_source = str(bootstrap_authority.get("authority_source") or "active_paper_bootstrap_pointer")
            registry_reconciled = True
            registry_gate_reason = None
    if registry_gate_reason is not None:
        return _readiness_result(model_type=normalized_model, ready=False, status=status, reason=registry_gate_reason, registry_fields=registry_fields, manifest_path=manifest_path, deployment_stage=deployment_stage, active_manifest_exists=True, active_manifest_valid=True, feature_compatible=True, paper_execution_only=paper_execution_only)

    readiness_reason = "paper_bootstrap_ready" if paper_execution_only else "ready"
    return _readiness_result(model_type=normalized_model, ready=True, status=status, reason=readiness_reason, registry_fields=registry_fields, manifest_path=manifest_path, deployment_stage=deployment_stage, active_manifest_exists=True, active_manifest_valid=True, feature_compatible=True, paper_execution_only=paper_execution_only, active_authority_source=authority_source, registry_reconciled_from_active_authority=registry_reconciled)

def anchor_model_readiness(paths, *, symbol: str, interval: str, required_models: tuple[str, ...] = ("rf", "gbm")) -> AnchorReadiness:
    models = {model: model_registry_readiness(paths, model_type=model, symbol=symbol, interval=interval) for model in required_models}
    blocking = [f"{model}:{state.reason}" for model, state in models.items() if not state.ready]
    if blocking:
        return AnchorReadiness(ready=False, reason="anchor_training_incomplete:" + ",".join(blocking), models=models)
    return AnchorReadiness(ready=True, reason="anchor_ready", models=models)


def signal_eligible_symbols(
    paths,
    *,
    symbols: list[str],
    interval: str,
    required_models: tuple[str, ...] = ("rf", "gbm"),
) -> tuple[list[str], dict[str, str]]:
    """Return signal-eligible symbols plus per-symbol anchor readiness reasons.

    The signal engine should evaluate only symbols whose anchor artifacts are
    actually ready. Symbols that only recently entered the active universe
    remain visible through readiness reasons, but are not emitted into the live
    signal loop until the anchor gate is satisfied.
    """
    ready_symbols: list[str] = []
    reasons: dict[str, str] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        readiness = anchor_model_readiness(paths, symbol=symbol, interval=interval, required_models=required_models)
        reasons[symbol] = readiness.reason
        if readiness.ready:
            ready_symbols.append(symbol)
    return ready_symbols, reasons


def _model_readiness_to_payload(state: ModelReadiness) -> dict[str, Any]:
    ready = bool(state.ready)
    reason = str(state.reason)
    manifest_path = getattr(state, "manifest_path", None)
    return {
        "model_type": state.model_type,
        "ready": ready,
        "execution_ready": ready,
        "status": str(state.status),
        "reason": reason,
        "manifest_path": manifest_path,
        "active_manifest_path": manifest_path if getattr(state, "active_manifest_exists", False) else None,
        "active_manifest_exists": bool(getattr(state, "active_manifest_exists", False)),
        "active_manifest_valid": bool(getattr(state, "active_manifest_valid", ready)),
        "feature_compatible": bool(getattr(state, "feature_compatible", ready or not reason.startswith("feature_incompatible:"))),
        "paper_execution_only": bool(getattr(state, "paper_execution_only", False)),
        "live_eligible": not bool(getattr(state, "paper_execution_only", False)),
        "active_authority_source": getattr(state, "active_authority_source", None),
        "registry_reconciled_from_active_authority": bool(getattr(state, "registry_reconciled_from_active_authority", False)),
        "registry_entry_exists": bool(getattr(state, "registry_entry_exists", False)),
        "registry_truth_status": str(getattr(state, "registry_truth_status", "unknown")),
        "registry_execution_ready": bool(getattr(state, "registry_execution_ready", False)),
        "registry_promotion_approved": bool(getattr(state, "registry_promotion_approved", False)),
        "registry_active_manifest_valid": bool(getattr(state, "registry_active_manifest_valid", False)),
        "deployment_stage": str(getattr(state, "deployment_stage", "unknown")),
        "feature_contract_diagnostics": dict(getattr(state, "feature_contract_diagnostics", {}) or {}),
    }


def anchor_readiness_to_payload(symbol: str, interval: str, readiness: AnchorReadiness) -> dict[str, Any]:
    """Return an operator-facing, JSON-safe readiness row for one symbol.

    This payload is diagnostic only. It does not loosen or bypass the runtime
    execution gate; it explains why the existing anchor gate is allowing or
    blocking signal evaluation.
    """
    model_payload = {model: _model_readiness_to_payload(state) for model, state in readiness.models.items()}
    blocking_models = [str(model) for model, state in readiness.models.items() if not state.ready]
    return {
        "symbol": str(symbol).upper(),
        "interval": str(interval),
        "ready": bool(readiness.ready),
        "execution_ready": bool(readiness.ready),
        "reason": str(readiness.reason),
        "required_active_models": [str(model) for model in readiness.models.keys()],
        "blocking_models": blocking_models,
        "candidate_only_is_execution_ready": False,
        "models": model_payload,
    }


def signal_readiness_snapshot(
    paths,
    *,
    symbols: list[str],
    interval: str,
    required_models: tuple[str, ...] = ("rf", "gbm"),
) -> dict[str, Any]:
    """Build a complete anchor-readiness diagnostic for configured symbols.

    The signal loop may evaluate only ready symbols, but the dashboard still
    needs to show non-ready configured symbols and the exact RF/GBM blocker.
    This function intentionally publishes diagnostics for all configured
    symbols without changing eligibility or trading behavior.
    """
    rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    model_blockers: dict[str, int] = {str(model): 0 for model in required_models}
    ready_symbols: list[str] = []
    blocked_symbols: list[str] = []

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        readiness = anchor_model_readiness(paths, symbol=symbol, interval=interval, required_models=required_models)
        row = anchor_readiness_to_payload(symbol, interval, readiness)
        rows.append(row)
        if readiness.ready:
            ready_symbols.append(symbol)
            continue
        blocked_symbols.append(symbol)
        reason = str(readiness.reason or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for model, state in readiness.models.items():
            if not state.ready:
                key = str(model)
                model_blockers[key] = model_blockers.get(key, 0) + 1

    return {
        "interval": str(interval),
        "required_models": [str(model) for model in required_models],
        "total_symbols": len(rows),
        "ready_count": len(ready_symbols),
        "blocked_count": len(blocked_symbols),
        "execution_eligible_count": len(ready_symbols),
        "ready_symbols": ready_symbols,
        "execution_eligible_symbols": ready_symbols,
        "blocked_symbols": blocked_symbols,
        "reason_counts": reason_counts,
        "model_blockers": model_blockers,
        "rows": rows,
        "execution_readiness_matrix": rows,
    }


def readiness_blocker_report(readiness: dict[str, Any] | None, *, runtime_blockers: list[str] | None = None) -> dict[str, Any]:
    """Return a stable operator report of readiness blockers by layer.

    Step 3 contract: this is diagnostic-only. It never makes a symbol
    execution eligible; it only explains why the existing fail-closed
    readiness gate is blocking evaluation/execution.
    """
    payload = readiness if isinstance(readiness, dict) else {}
    rows = payload.get("rows") or payload.get("execution_readiness_matrix") or []
    if not isinstance(rows, list):
        rows = []

    buckets: dict[str, list[dict[str, Any]]] = {
        "symbol": [],
        "data": [],
        "rf": [],
        "gbm": [],
        "policy": [],
        "registry": [],
        "feature_compatibility": [],
        "execution": [],
        "control_plane": [],
        "unknown": [],
    }
    reason_counts: dict[str, int] = {}

    def _bucket_for(model: str, reason: str) -> str:
        text = f"{model} {reason}".lower()
        if "feature_incompatible" in text or "feature_" in text and "mismatch" in text:
            return "feature_compatibility"
        if "registry" in text or "promotion" in text or "approval" in text:
            return "registry"
        if "price" in text or "market_data" in text or "warming" in text or "candle" in text:
            return "data"
        if model == "rf" or text.startswith("rf"):
            return "rf"
        if model == "gbm" or text.startswith("gbm"):
            return "gbm"
        if model in {"policy", "dqn", "confirmatory_policy"}:
            return "policy"
        if "execution" in text or "manifest" in text:
            return "execution"
        if "symbol" in text or "active" in text or "eligible" in text:
            return "symbol"
        return "unknown"

    runtime_blocker_list = [str(item).strip() for item in list(runtime_blockers or []) if str(item).strip()]

    def _runtime_bucket(reason: str) -> str:
        text = str(reason).lower()
        if text.startswith("active_universe_count_mismatch") or text.startswith("blocked_symbols") or text.startswith("execution_eligible_symbols") or text.startswith("signal_readiness_blocked"):
            return "symbol"
        if "price_cache" in text or "market_data" in text:
            return "data"
        if text.startswith("registry_") or "anchor_model_readiness" in text:
            return "registry"
        if text.startswith("control_plane_paused") or text.startswith("circuit_breaker"):
            return "control_plane"
        if "execution_integrity" in text:
            return "execution"
        return "unknown"

    for reason in runtime_blocker_list:
        bucket = _runtime_bucket(reason)
        buckets.setdefault(bucket, []).append({
            "symbol": None,
            "interval": str(payload.get("interval") or ""),
            "model_type": None,
            "reason": reason,
            "status": "runtime_blocker",
            "deployment_stage": None,
            "manifest_path": None,
            "source": "runtime",
        })
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for row in rows:
        if not isinstance(row, dict) or bool(row.get("ready")):
            continue
        symbol = str(row.get("symbol") or "").upper()
        interval = str(row.get("interval") or payload.get("interval") or "")
        top_reason = str(row.get("reason") or "unknown")
        reason_counts[top_reason] = reason_counts.get(top_reason, 0) + 1
        models = row.get("models") if isinstance(row.get("models"), dict) else {}
        emitted = False
        for model, state in models.items():
            if not isinstance(state, dict) or bool(state.get("ready")):
                continue
            model_name = str(model).lower()
            reason = str(state.get("reason") or top_reason or "unknown")
            bucket = _bucket_for(model_name, reason)
            buckets.setdefault(bucket, []).append({
                "symbol": symbol,
                "interval": interval,
                "model_type": model_name,
                "reason": reason,
                "status": state.get("status"),
                "deployment_stage": state.get("deployment_stage"),
                "manifest_path": state.get("manifest_path") or state.get("active_manifest_path"),
            })
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            emitted = True
        if not emitted:
            bucket = _bucket_for("", top_reason)
            buckets.setdefault(bucket, []).append({
                "symbol": symbol,
                "interval": interval,
                "model_type": None,
                "reason": top_reason,
                "status": row.get("status"),
                "deployment_stage": row.get("deployment_stage"),
                "manifest_path": row.get("manifest_path"),
            })

    summary = {name: len(items) for name, items in buckets.items()}
    total = sum(summary.values())
    return {
        "schema_version": "readiness_blocker_report.v1",
        "total_blockers": total,
        "runtime_blocker_count": len(runtime_blocker_list),
        "runtime_blockers": runtime_blocker_list,
        "ready_count": int(payload.get("ready_count") or payload.get("execution_eligible_count") or 0),
        "blocked_count": int(payload.get("blocked_count") or 0),
        "summary": summary,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "buckets": buckets,
    }
