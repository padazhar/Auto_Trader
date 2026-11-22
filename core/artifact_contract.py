from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from core.model_naming import canonical_model_type, model_storage_directory
from features.feature_store_contract import build_feature_store_metadata, resolve_feature_recipe_version, validate_feature_store_metadata

MODEL_DIR_MAP = {model_type: model_storage_directory(model_type) for model_type in ("rf", "gbm", "dqn", "hmm")}

REQUIRED_KEYS = {
    "rf": ("model", "feature_columns", "metrics"),
    "dqn": ("model", "feature_columns", "metrics"),
    "hmm": ("model", "feature_columns", "metrics"),
}

ROLE_REQUIRED_KEYS = {
    ("gbm", "direction"): ("model", "feature_columns", "metrics"),
    ("gbm", "meta"): ("model", "feature_columns", "metrics"),
}


class ArtifactContractError(ValueError):
    """Raised when an artifact contract operation is invalid."""


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically write bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(str(temp_path), str(path))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a JSON payload atomically."""
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, encoded)


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON payload from disk when present."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def model_base_dir(
    paths,
    model_type: str,
    symbol: str,
    interval: str,
    *,
    model_role: str | None = None,
    stage: str = "incumbent",
    candidate_id: str | None = None,
) -> Path:
    """Return the base directory for an artifact bundle."""
    model_type = canonical_model_type(model_type)
    base = paths.model_storage_dir / MODEL_DIR_MAP[model_type] / symbol.upper() / interval
    if model_type == "gbm" and model_role:
        base = base / str(model_role).lower()
    if stage == "candidate":
        if not candidate_id:
            raise ArtifactContractError("candidate_id is required when stage='candidate'")
        return base / "candidates" / str(candidate_id)
    if stage != "incumbent":
        raise ArtifactContractError(f"unsupported stage: {stage}")
    return base


def artifact_paths(
    paths,
    model_type: str,
    symbol: str,
    interval: str,
    *,
    model_role: str | None = None,
    stage: str = "incumbent",
    candidate_id: str | None = None,
) -> dict[str, Path]:
    """Return the canonical artifact file paths for a bundle."""
    base = model_base_dir(paths, model_type, symbol, interval, model_role=model_role, stage=stage, candidate_id=candidate_id)
    return {
        "model": base / "model.joblib",
        "feature_columns": base / "feature_columns.json",
        "metrics": base / "metrics.json",
        "manifest": base / "manifest.json",
    }


def gbm_role_artifact_paths(paths, symbol: str, interval: str, model_role: str, *, stage: str = "incumbent", candidate_id: str | None = None) -> dict[str, Path]:
    """Return GBM role-specific artifact paths."""
    return artifact_paths(paths, "gbm", symbol, interval, model_role=model_role, stage=stage, candidate_id=candidate_id)


def incumbent_pointer_path(paths, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> Path:
    """Return the path of the active-incumbent pointer file."""
    return model_base_dir(paths, model_type, symbol, interval, model_role=model_role) / "incumbent.json"


def delete_incumbent_pointer(paths, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> None:
    """Remove an incumbent pointer when a role should no longer be active."""
    pointer = incumbent_pointer_path(paths, model_type, symbol, interval, model_role=model_role)
    if pointer.exists():
        pointer.unlink()


def gbm_combined_manifest_path(paths, symbol: str, interval: str, *, stage: str = "incumbent", candidate_id: str | None = None) -> Path:
    """Return the path for the GBM bundle manifest."""
    return model_base_dir(paths, "gbm", symbol, interval, stage=stage, candidate_id=candidate_id) / "manifest.json"


def relative_to_project(paths, path: Path) -> str:
    """Return a project-relative path when possible."""
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(paths.project_root).as_posix()
    except Exception:
        return candidate.name


def resolve_artifact_path(paths, raw_path: str | Path) -> Path:
    """Resolve an artifact path relative to the project root."""
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (paths.project_root / path).resolve()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()




def _parse_utc_datetime(value: Any) -> datetime | None:
    """Best-effort parser for artifact timestamps."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return max(int(minimum), out)


def _safe_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return max(float(minimum), out)


def artifact_lifecycle_settings(paths) -> dict[str, Any]:
    """Return bounded artifact lifecycle settings with safe defaults.

    The defaults preserve every promoted incumbent and keep a small recent
    candidate audit trail, while preventing unbounded overnight growth.
    """
    training = getattr(getattr(paths, "config", None), "training", None)
    cfg = getattr(training, "artifact_lifecycle", None)
    return {
        "enabled": bool(getattr(cfg, "enabled", True)),
        "prune_after_training": bool(getattr(cfg, "prune_after_training", True)),
        "keep_last_candidates": _safe_int(getattr(cfg, "keep_last_candidates", 3), 3, minimum=0),
        "max_candidate_age_hours": _safe_float(getattr(cfg, "max_candidate_age_hours", 24.0), 24.0, minimum=0.0),
        "max_deleted_per_run": _safe_int(getattr(cfg, "max_deleted_per_run", 200), 200, minimum=1),
        "protect_promoted": bool(getattr(cfg, "protect_promoted", True)),
    }


def candidate_parent_dir(
    paths,
    model_type: str,
    symbol: str,
    interval: str,
    *,
    model_role: str | None = None,
) -> Path:
    """Return the canonical candidates parent directory for a model slot."""
    return model_base_dir(paths, model_type, symbol, interval, model_role=model_role) / "candidates"


def _manifest_created_at(candidate_dir: Path) -> datetime:
    manifest = load_json(candidate_dir / "manifest.json") or {}
    timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
    parsed = _parse_utc_datetime(timestamps.get("created_at")) or _parse_utc_datetime(timestamps.get("updated_at"))
    if parsed is not None:
        return parsed
    try:
        return datetime.fromtimestamp(candidate_dir.stat().st_mtime, tz=timezone.utc)
    except Exception:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _candidate_manifest_path(candidate_dir: Path) -> Path:
    return candidate_dir / "manifest.json"


def _active_manifest_paths(paths, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> set[Path]:
    protected: set[Path] = set()
    pointer = load_json(incumbent_pointer_path(paths, model_type, symbol, interval, model_role=model_role)) or {}
    raw = pointer.get("manifest_path")
    if raw:
        try:
            protected.add(resolve_artifact_path(paths, raw).resolve())
        except Exception:
            pass
    active = resolve_active_manifest_path(paths, model_type, symbol, interval, model_role=model_role)
    if active is not None:
        try:
            protected.add(active.resolve())
        except Exception:
            pass
    return protected


def prune_candidate_artifacts(
    paths,
    *,
    model_type: str,
    symbol: str,
    interval: str,
    model_role: str | None = None,
    dry_run: bool = False,
    keep_last_candidates: int | None = None,
    max_candidate_age_hours: float | None = None,
    max_deleted: int | None = None,
) -> dict[str, Any]:
    """Prune stale candidate artifact directories for one model slot.

    Promoted incumbent manifests are protected. The policy deletes candidates
    that are older than the configured age or exceed the configured recent
    candidate count. This keeps storage bounded without deleting the active
    model bundle.
    """
    model_type = canonical_model_type(model_type)
    settings = artifact_lifecycle_settings(paths)
    if not settings["enabled"]:
        return {"enabled": False, "deleted": [], "kept": [], "protected": [], "reason": "artifact_lifecycle_disabled"}

    keep_last = settings["keep_last_candidates"] if keep_last_candidates is None else _safe_int(keep_last_candidates, settings["keep_last_candidates"], minimum=0)
    max_age_hours = settings["max_candidate_age_hours"] if max_candidate_age_hours is None else _safe_float(max_candidate_age_hours, settings["max_candidate_age_hours"], minimum=0.0)
    max_delete_count = settings["max_deleted_per_run"] if max_deleted is None else _safe_int(max_deleted, settings["max_deleted_per_run"], minimum=1)

    candidates_dir = candidate_parent_dir(paths, model_type, symbol, interval, model_role=model_role)
    if not candidates_dir.exists():
        return {"enabled": True, "deleted": [], "kept": [], "protected": [], "reason": "no_candidates_dir"}

    candidate_dirs = [entry for entry in candidates_dir.iterdir() if entry.is_dir()]
    candidate_dirs.sort(key=_manifest_created_at, reverse=True)
    protected_manifests = _active_manifest_paths(paths, model_type, symbol, interval, model_role=model_role) if settings["protect_promoted"] else set()
    newest_keep = {candidate.resolve() for candidate in candidate_dirs[:keep_last]} if keep_last > 0 else set()
    cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600.0)

    deleted: list[str] = []
    kept: list[str] = []
    protected: list[str] = []
    for candidate in candidate_dirs:
        candidate_resolved = candidate.resolve()
        manifest_path = _candidate_manifest_path(candidate).resolve()
        if manifest_path in protected_manifests:
            protected.append(relative_to_project(paths, candidate))
            continue
        created_at = _manifest_created_at(candidate)
        is_old = created_at.timestamp() < cutoff if max_age_hours > 0 else False
        exceeds_keep = candidate_resolved not in newest_keep
        if (is_old or exceeds_keep) and len(deleted) < max_delete_count:
            deleted.append(relative_to_project(paths, candidate))
            if not dry_run:
                shutil.rmtree(candidate, ignore_errors=True)
        else:
            kept.append(relative_to_project(paths, candidate))

    return {
        "enabled": True,
        "dry_run": bool(dry_run),
        "model_type": model_type,
        "model_role": model_role,
        "symbol": symbol.upper(),
        "interval": interval,
        "keep_last_candidates": keep_last,
        "max_candidate_age_hours": max_age_hours,
        "deleted": deleted,
        "kept": kept,
        "protected": protected,
    }


def prune_related_candidate_artifacts(
    paths,
    *,
    model_type: str,
    symbol: str,
    interval: str,
    model_role: str | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Prune candidate artifacts affected by a just-finished training run."""
    model_type = canonical_model_type(model_type)
    settings = artifact_lifecycle_settings(paths)
    if not settings["enabled"] or not settings["prune_after_training"]:
        return []
    roles: list[str | None]
    if model_type == "gbm" and model_role is None:
        roles = [None, "direction", "meta"]
    else:
        roles = [model_role]
    return [
        prune_candidate_artifacts(
            paths,
            model_type=model_type,
            symbol=symbol,
            interval=interval,
            model_role=role,
            dry_run=dry_run,
        )
        for role in roles
    ]


def _feature_set_payload(
    feature_columns: list[str],
    explicit_payload: dict[str, Any] | None = None,
    *,
    recipe_version: str | None = None,
    anchor_interval: str | None = None,
    optional_context_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build canonical feature-set metadata for a manifest.

    Older manifests used a column-only ``fset-*`` version/signature.  New
    training artifacts must persist the same canonical feature-store contract
    that inference validates: recipe version, anchor interval, optional context
    fields, ordered columns, and derived signature.  This keeps training and
    inference synchronized without weakening the fail-closed compatibility gate
    for existing stale artifacts.
    """
    payload = dict(explicit_payload or {})
    ordered = [str(col) for col in feature_columns]
    raw_recipe = str(payload.get("recipe_version") or payload.get("version") or recipe_version or "").strip()
    if not raw_recipe or raw_recipe.startswith("fset-"):
        raw_recipe = str(recipe_version or resolve_feature_recipe_version(None))
    resolved_anchor = payload.get("anchor_interval", anchor_interval)
    raw_optional = payload.get("optional_context_fields", optional_context_fields or [])
    optional = [str(col) for col in (raw_optional or [])]
    return build_feature_store_metadata(
        ordered,
        recipe_version=raw_recipe,
        anchor_interval=str(resolved_anchor) if resolved_anchor is not None else None,
        optional_context_fields=optional,
    )


def _manifest_feature_set(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a manifest feature-set payload when present."""
    feature_set = payload.get("feature_set")
    if not isinstance(feature_set, dict):
        return {}
    return dict(feature_set)


def validate_manifest_feature_compatibility(
    paths,
    payload: dict[str, Any],
    *,
    expected_recipe_version: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate manifest feature metadata against the active runtime feature recipe.

    This check is intentionally stricter than basic manifest validation. It ensures
    the active runtime does not silently load an artifact whose feature recipe no
    longer matches the configured feature-store contract.
    """
    feature_set = _manifest_feature_set(payload)
    runtime_recipe_version = str(expected_recipe_version or resolve_feature_recipe_version(getattr(paths, "config", None))).strip()
    columns_raw = feature_set.get("columns") if isinstance(feature_set, dict) else None
    columns = [str(col) for col in columns_raw] if isinstance(columns_raw, list) else []
    anchor_interval = feature_set.get("anchor_interval") if isinstance(feature_set, dict) else None
    optional_context_fields = feature_set.get("optional_context_fields") if isinstance(feature_set, dict) else []
    compatible, reasons, _expected = validate_feature_store_metadata(
        feature_set,
        feature_columns=columns,
        recipe_version=runtime_recipe_version or str((feature_set or {}).get("recipe_version") or ""),
        anchor_interval=anchor_interval,
        optional_context_fields=optional_context_fields or [],
    )

    raw_feature_path = payload.get("feature_columns_path")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    raw_feature_path = raw_feature_path or artifacts.get("feature_columns")
    if raw_feature_path:
        artifact_payload = load_json(resolve_artifact_path(paths, str(raw_feature_path))) or {}
        artifact_columns_raw = artifact_payload.get("feature_columns")
        artifact_columns = [str(col) for col in artifact_columns_raw] if isinstance(artifact_columns_raw, list) else []
        if not artifact_columns:
            reasons.append("artifact_feature_columns_missing")
        elif columns and artifact_columns != columns:
            if set(artifact_columns) == set(columns):
                reasons.append("artifact_feature_columns_order_mismatch")
            else:
                reasons.append("artifact_feature_columns_mismatch")
    return len(reasons) == 0, reasons


def write_manifest(
    paths,
    *,
    model_type: str,
    symbol: str,
    interval: str,
    rows_used: int,
    feature_count: int,
    artifacts: dict[str, Path],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a versioned manifest for a model artifact bundle."""
    extra = dict(extra or {})
    model_role = extra.get("model_role")
    manifest_path = artifacts["manifest"]
    stamp = _utc_stamp()
    feature_columns_payload = load_json(artifacts["feature_columns"]) or {}
    feature_columns = [str(c) for c in feature_columns_payload.get("feature_columns", [])]
    explicit_feature_set = extra.pop("feature_set", None)
    feature_set = _feature_set_payload(
        feature_columns,
        explicit_payload=explicit_feature_set,
        recipe_version=resolve_feature_recipe_version(getattr(paths, "config", None)),
        anchor_interval=interval,
    )
    validation = dict(extra.pop("validation", {}))
    training = dict(extra.pop("training", {}))
    lifecycle = dict(extra.pop("lifecycle", {}))
    payload = {
        "schema_version": 4,
        "model_type": model_type,
        "model_role": model_role,
        "symbol": symbol.upper(),
        "interval": interval,
        "rows_used": int(rows_used),
        "feature_count": int(feature_count),
        "feature_set": feature_set,
        "artifacts": {k: relative_to_project(paths, v) for k, v in artifacts.items() if k != "manifest"},
        "metrics_path": relative_to_project(paths, artifacts["metrics"]),
        "feature_columns_path": relative_to_project(paths, artifacts["feature_columns"]),
        "validation": validation,
        "training": training,
        "lifecycle": {
            "stage": lifecycle.get("stage", "incumbent"),
            "candidate_id": lifecycle.get("candidate_id"),
            "promoted_at": lifecycle.get("promoted_at"),
            "promotion_note": lifecycle.get("promotion_note"),
        },
        "timestamps": {
            "created_at": stamp,
            "updated_at": stamp,
        },
    }
    payload.update(extra)
    save_json(manifest_path, payload)
    save_json(artifacts["metrics"], metrics)
    return payload


def _gbm_role_feature_contract(paths, manifest_path: Path, *, expected_role: str) -> dict[str, Any]:
    """Return a validated GBM role feature contract for bundle provenance.

    The combined bundle is not an independent prediction model: direction is
    the execution-authoritative GBM role, while meta is an optional secondary
    role.  A bundle may therefore carry the direction contract as its primary
    ``feature_set`` only when the linked role manifest itself is readable,
    correctly typed, and feature-compatible with the configured runtime
    recipe.  This prevents a bundle from becoming ready through unrelated or
    incomplete feature metadata.
    """
    payload = load_json(Path(manifest_path)) or {}
    if not payload:
        raise ArtifactContractError(f"gbm_{expected_role}_manifest_unreadable")
    if str(payload.get("model_type") or "") != "gbm" or str(payload.get("model_role") or "") != expected_role:
        raise ArtifactContractError(f"gbm_{expected_role}_manifest_role_mismatch")
    feature_set = _manifest_feature_set(payload)
    if not str(feature_set.get("signature") or "").strip():
        raise ArtifactContractError(f"gbm_{expected_role}_feature_set_missing")
    compatible, reasons = validate_manifest_feature_compatibility(paths, payload)
    if not compatible:
        joined = ",".join(str(reason) for reason in reasons) or "unknown"
        raise ArtifactContractError(f"gbm_{expected_role}_feature_incompatible:{joined}")
    return feature_set




def _gbm_manifest_estimator_family(payload: dict[str, Any]) -> str:
    """Return the estimator family recorded by a GBM manifest.

    HistGBM is the backward-compatible default for legacy artefacts.  New Step 8
    GBM manifests record the selected estimator explicitly so LightGBM candidates
    remain auditable and rollback-safe without changing the canonical GBM model
    family used by inference.
    """
    raw = payload.get("estimator_family")
    if not raw and isinstance(payload.get("estimator_contract"), dict):
        raw = payload["estimator_contract"].get("estimator_family")
    if not raw and isinstance(payload.get("training"), dict):
        raw = payload["training"].get("estimator_family")
    family = str(raw or "histgbm").strip().lower().replace("-", "_")
    aliases = {"histgbm": "histgbm", "hist_gradient_boosting": "histgbm", "histgradientboosting": "histgbm", "lightgbm": "lightgbm", "lgbm": "lightgbm"}
    return aliases.get(family, family)

def write_gbm_combined_manifest(
    paths,
    *,
    symbol: str,
    interval: str,
    direction_manifest: Path,
    meta_manifest: Path | None,
    rows_used: int,
    meta_skipped: bool,
    meta_skip_reason: str | None = None,
    stage: str = "incumbent",
    candidate_id: str | None = None,
    validation: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a GBM bundle with a verifiable execution feature contract.

    Direction is the execution-authoritative GBM role used by readiness and
    scoring.  The bundle therefore publishes the already validated direction
    feature contract as its own primary contract and records the linked role
    signatures for auditability.  Meta remains optional, but when present its
    linked manifest must also carry a valid role-specific contract.
    """
    direction_manifest = Path(direction_manifest)
    meta_manifest = Path(meta_manifest) if meta_manifest is not None else None
    direction_feature_set = _gbm_role_feature_contract(paths, direction_manifest, expected_role="direction")
    meta_feature_set = (
        _gbm_role_feature_contract(paths, meta_manifest, expected_role="meta")
        if meta_manifest is not None
        else None
    )
    direction_payload = load_json(direction_manifest) or {}
    meta_payload = load_json(meta_manifest) or {} if meta_manifest is not None else {}
    direction_estimator_family = _gbm_manifest_estimator_family(direction_payload)
    meta_estimator_family = _gbm_manifest_estimator_family(meta_payload) if meta_payload else None
    if meta_estimator_family and meta_estimator_family != direction_estimator_family:
        raise ArtifactContractError("gbm_meta_estimator_family_mismatch")
    training_payload = dict(training or {})
    training_payload.setdefault("estimator_family", direction_estimator_family)
    training_payload.setdefault("estimator_selector", "training.gbm.estimator")
    training_payload.setdefault("rollback_family", "histgbm")
    estimator_contract = {
        "family": direction_estimator_family,
        "direction_family": direction_estimator_family,
        "meta_family": meta_estimator_family,
        "selector_key": "training.gbm.estimator",
        "rollback_family": "histgbm",
    }
    stamp = _utc_stamp()
    payload = {
        "schema_version": 4,
        "model_type": "gbm",
        "model_role": "bundle",
        "symbol": symbol.upper(),
        "interval": interval,
        "rows_used": int(rows_used),
        "feature_set": dict(direction_feature_set),
        "feature_contract": {
            "execution_authority": "direction",
            "direction_signature": str(direction_feature_set.get("signature") or ""),
            "meta_signature": str((meta_feature_set or {}).get("signature") or "") or None,
        },
        "artifacts": {
            "direction_manifest": relative_to_project(paths, direction_manifest),
        },
        "meta": {
            "manifest": relative_to_project(paths, meta_manifest) if meta_manifest else None,
            "skipped": bool(meta_skipped),
            "skip_reason": meta_skip_reason,
        },
        "validation": dict(validation or {}),
        "training": training_payload,
        "estimator_family": direction_estimator_family,
        "estimator_contract": estimator_contract,
        "lifecycle": {
            "stage": stage,
            "candidate_id": candidate_id,
            "promoted_at": None,
        },
        "timestamps": {
            "created_at": stamp,
            "updated_at": stamp,
        },
    }
    save_json(gbm_combined_manifest_path(paths, symbol, interval, stage=stage, candidate_id=candidate_id), payload)
    return payload


def load_manifest_metrics(paths, manifest_path: Path) -> dict[str, Any]:
    """Load the metrics payload referenced by a standard manifest."""
    manifest = load_json(manifest_path) or {}
    if not manifest:
        return {}
    metrics_path = manifest.get("metrics_path")
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    if not metrics_path:
        metrics_path = artifacts.get("metrics")
    if not metrics_path:
        return {}
    payload = load_json(resolve_artifact_path(paths, str(metrics_path)))
    return payload if isinstance(payload, dict) else {}


def promote_candidate_manifest(
    paths,
    manifest_path: Path | None = None,
    *,
    model_type: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
    candidate_id: str | None = None,
    model_role: str | None = None,
    metrics: dict[str, Any] | None = None,
    deployment_stage: str = "incumbent",
    paper_execution_only: bool = False,
    paper_evidence_status: str | None = None,
) -> dict[str, Any]:
    """Atomically point the incumbent bundle at a candidate manifest.

    The keyword form is the canonical Step 8 contract.  The optional positional
    ``manifest_path`` preserves older test/helper callers and derives missing
    metadata from the manifest payload before writing the incumbent pointer.
    """
    if manifest_path is None:
        raise TypeError("manifest_path is required")
    manifest_path = Path(manifest_path)
    manifest_payload = load_json(manifest_path) or {}
    model_type = canonical_model_type(model_type or manifest_payload.get("model_type") or "")
    symbol = str(symbol or manifest_payload.get("symbol") or "").upper()
    interval = str(interval or manifest_payload.get("interval") or "")
    model_role = model_role if model_role is not None else manifest_payload.get("model_role")
    lifecycle = manifest_payload.get("lifecycle") if isinstance(manifest_payload.get("lifecycle"), dict) else {}
    candidate_id = str(candidate_id or lifecycle.get("candidate_id") or manifest_path.parent.name)
    if not model_type or not symbol or not interval:
        raise ValueError("model_type, symbol, and interval are required to promote a candidate manifest")
    deployment_stage = str(deployment_stage or "incumbent").strip().lower()
    if deployment_stage not in {"incumbent", "paper_bootstrap"}:
        raise ValueError(f"unsupported deployment stage for active pointer: {deployment_stage}")
    if deployment_stage == "paper_bootstrap" and not bool(paper_execution_only):
        raise ValueError("paper_bootstrap stage requires paper_execution_only=true")
    live_eligible = bool(deployment_stage == "incumbent" and not paper_execution_only)
    stamp = _utc_stamp()
    pointer_payload = {
        "schema_version": 1,
        "model_type": model_type,
        "model_role": model_role,
        "symbol": symbol.upper(),
        "interval": interval,
        "candidate_id": str(candidate_id),
        "manifest_path": relative_to_project(paths, manifest_path),
        "promoted_at": stamp,
        "deployment_stage": deployment_stage,
        "paper_execution_only": bool(paper_execution_only),
        "paper_evidence_status": str(paper_evidence_status or "not_evaluated"),
        "live_eligible": live_eligible,
        "metrics_summary": dict(metrics or {}),
    }
    save_json(incumbent_pointer_path(paths, model_type, symbol, interval, model_role=model_role), pointer_payload)
    manifest = load_json(manifest_path)
    if manifest:
        lifecycle = dict(manifest.get("lifecycle") or {})
        lifecycle["stage"] = deployment_stage
        lifecycle["candidate_id"] = str(candidate_id)
        lifecycle["promoted_at"] = stamp
        lifecycle["paper_execution_only"] = bool(paper_execution_only)
        lifecycle["paper_evidence_status"] = str(paper_evidence_status or "not_evaluated")
        lifecycle["live_eligible"] = live_eligible
        manifest["lifecycle"] = lifecycle
        timestamps = dict(manifest.get("timestamps") or {})
        timestamps["updated_at"] = stamp
        manifest["timestamps"] = timestamps
        save_json(manifest_path, manifest)
    return pointer_payload



def manifest_deployment_stage(manifest_path: Path, payload: dict[str, Any] | None = None) -> str:
    """Return the execution deployment stage encoded by a manifest path/payload.

    Candidate bundles are never execution-ready. Evidence-backed promoted
    candidate manifests are execution-ready when lifecycle.stage is
    ``incumbent``. An explicitly governed initial ``paper_bootstrap`` stage is
    usable only while runtime.execution_mode is paper and is never live
    eligible. Legacy non-candidate incumbent-directory manifests are treated as
    incumbent for backwards compatibility.
    """
    manifest = dict(payload or {})
    lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
    raw_stage = str(lifecycle.get("stage") or "").strip().lower()
    path_parts = {str(part).lower() for part in Path(manifest_path).parts}
    if raw_stage:
        return raw_stage
    if "candidates" in path_parts:
        return "candidate"
    return "incumbent"


def validate_execution_manifest(paths, manifest_path: Path) -> tuple[bool, str, dict[str, Any]]:
    """Validate that a manifest is authorised for the current execution mode.

    ``paper_bootstrap`` artifacts exist only to generate controlled paper-trade
    evidence when no incumbent exists yet. They must fail closed in live mode,
    even when an operator later changes runtime configuration.
    """
    payload = load_json(manifest_path) or {}
    if not payload:
        return False, "manifest_unreadable", {}
    stage = manifest_deployment_stage(manifest_path, payload)
    lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
    if stage == "paper_bootstrap":
        runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
        execution_mode = str(getattr(runtime_cfg, "execution_mode", "paper") or "paper").strip().lower()
        if execution_mode != "paper":
            return False, "paper_bootstrap_forbidden_outside_paper_mode", payload
        if not bool(lifecycle.get("paper_execution_only", False)) or bool(lifecycle.get("live_eligible", True)):
            return False, "paper_bootstrap_governance_invalid", payload
    elif stage != "incumbent":
        return False, f"deployment_stage:{stage}", payload
    valid, reason = validate_artifact_bundle(paths, manifest_path)
    if not valid:
        return False, reason, payload
    return True, "paper_bootstrap_ready" if stage == "paper_bootstrap" else "ready", payload

def resolve_active_manifest_path(paths, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> Path | None:
    """Resolve the currently promoted manifest for a model bundle."""
    model_type = canonical_model_type(model_type)
    pointer = load_json(incumbent_pointer_path(paths, model_type, symbol, interval, model_role=model_role))
    if pointer and pointer.get("manifest_path"):
        manifest_path = resolve_artifact_path(paths, str(pointer["manifest_path"]))
        if manifest_path.exists():
            return manifest_path
    legacy = artifact_paths(paths, model_type, symbol, interval, model_role=model_role)["manifest"]
    if legacy.exists():
        return legacy
    return None


def active_manifest_truth(paths, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> dict[str, Any]:
    """Return canonical execution-readiness truth for one promoted artifact slot.

    Trainer metrics can say that promotion was attempted or approved, but the
    execution path may only trust a currently resolvable incumbent manifest that
    validates through the execution-manifest contract. This helper keeps that
    final proof in one place for registry and readiness diagnostics.
    """
    normalized_model = canonical_model_type(model_type)
    manifest_path = resolve_active_manifest_path(paths, normalized_model, symbol, interval, model_role=model_role)
    if manifest_path is None:
        return {
            "model_type": normalized_model,
            "model_role": model_role,
            "symbol": str(symbol).upper(),
            "interval": str(interval),
            "active_manifest_path": None,
            "active_manifest_exists": False,
            "active_manifest_valid": False,
            "execution_ready": False,
            "reason": "active_manifest_missing",
            "deployment_stage": "missing",
        }
    if not manifest_path.exists():
        return {
            "model_type": normalized_model,
            "model_role": model_role,
            "symbol": str(symbol).upper(),
            "interval": str(interval),
            "active_manifest_path": str(manifest_path),
            "active_manifest_exists": False,
            "active_manifest_valid": False,
            "execution_ready": False,
            "reason": "active_manifest_missing",
            "deployment_stage": "missing",
        }
    execution_ready, reason, manifest = validate_execution_manifest(paths, manifest_path)
    stage = manifest_deployment_stage(manifest_path, manifest) if manifest else "unknown"
    return {
        "model_type": normalized_model,
        "model_role": model_role,
        "symbol": str(symbol).upper(),
        "interval": str(interval),
        "active_manifest_path": str(manifest_path),
        "active_manifest_exists": True,
        "active_manifest_valid": bool(execution_ready),
        "execution_ready": bool(execution_ready),
        "reason": str(reason),
        "deployment_stage": str(stage),
    }


def paper_bootstrap_active_authority_truth(
    paths,
    model_type: str,
    symbol: str,
    interval: str,
    *,
    model_role: str | None = None,
) -> dict[str, Any]:
    """Return pointer-proven paper-bootstrap execution authority.

    A paper-bootstrap artifact is allowed to generate initial paper evidence only
    when the active pointer and its manifest agree on the exact candidate,
    promotion timestamp, deployment stage, and non-live policy.  This contract is
    intentionally narrower than ordinary incumbent validation and exists so that
    later rejected candidate telemetry cannot withdraw a valid paper-only active
    bootstrap artifact from the execution path.
    """
    try:
        truth = active_manifest_truth(paths, model_type, symbol, interval, model_role=model_role)
    except Exception as exc:
        return {
            "authority_approved": False,
            "authority_reason": f"active_pointer_authority_error:{exc}",
            "authority_source": None,
            "paper_execution_only": False,
            "live_eligible": False,
            "candidate_id": None,
            "promoted_at": None,
            "active_manifest_path": None,
            "execution_ready": False,
            "deployment_stage": "unknown",
        }
    result: dict[str, Any] = {
        **dict(truth),
        "authority_approved": False,
        "authority_reason": str(truth.get("reason") or "active_manifest_missing"),
        "authority_source": None,
        "paper_execution_only": False,
        "live_eligible": False,
        "candidate_id": None,
        "promoted_at": None,
    }
    if not bool(truth.get("execution_ready", False)):
        return result
    if str(truth.get("deployment_stage") or "").strip().lower() != "paper_bootstrap":
        result["authority_reason"] = "active_manifest_not_paper_bootstrap"
        return result

    active_path_raw = truth.get("active_manifest_path")
    if not active_path_raw:
        result["authority_reason"] = "active_manifest_missing"
        return result
    active_path = Path(str(active_path_raw)).resolve()
    manifest = load_json(active_path) or {}
    lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
    candidate_id = str(lifecycle.get("candidate_id") or "").strip()
    promoted_at = str(lifecycle.get("promoted_at") or "").strip()
    if not candidate_id or not promoted_at:
        result["authority_reason"] = "paper_bootstrap_lifecycle_provenance_missing"
        return result
    if not bool(lifecycle.get("paper_execution_only", False)) or bool(lifecycle.get("live_eligible", True)):
        result["authority_reason"] = "paper_bootstrap_governance_invalid"
        return result

    pointer_path = incumbent_pointer_path(paths, model_type, symbol, interval, model_role=model_role)
    pointer = load_json(pointer_path) or {}
    if not isinstance(pointer, dict) or not pointer.get("manifest_path"):
        result["authority_reason"] = "paper_bootstrap_pointer_missing"
        return result
    try:
        pointer_manifest = resolve_artifact_path(paths, str(pointer["manifest_path"])).resolve()
    except Exception:
        result["authority_reason"] = "paper_bootstrap_pointer_manifest_invalid"
        return result
    if pointer_manifest != active_path:
        result["authority_reason"] = "paper_bootstrap_pointer_manifest_mismatch"
        return result
    if str(pointer.get("candidate_id") or "").strip() != candidate_id:
        result["authority_reason"] = "paper_bootstrap_pointer_candidate_mismatch"
        return result
    if str(pointer.get("promoted_at") or "").strip() != promoted_at:
        result["authority_reason"] = "paper_bootstrap_pointer_promotion_timestamp_mismatch"
        return result
    if str(pointer.get("deployment_stage") or "").strip().lower() != "paper_bootstrap":
        result["authority_reason"] = "paper_bootstrap_pointer_stage_mismatch"
        return result
    if not bool(pointer.get("paper_execution_only", False)) or bool(pointer.get("live_eligible", True)):
        result["authority_reason"] = "paper_bootstrap_pointer_governance_invalid"
        return result

    result.update(
        {
            "authority_approved": True,
            "authority_reason": "paper_bootstrap_pointer_authority_ready",
            "authority_source": "active_paper_bootstrap_pointer",
            "paper_execution_only": True,
            "live_eligible": False,
            "candidate_id": candidate_id,
            "promoted_at": promoted_at,
            "pointer_path": str(pointer_path),
        }
    )
    return result


def _validate_standard_manifest(paths, payload: dict[str, Any]) -> tuple[bool, str]:
    model_type = str(payload.get("model_type", ""))
    model_role = payload.get("model_role")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "artifact_map_missing"
    required = ROLE_REQUIRED_KEYS.get((model_type, str(model_role or "")), REQUIRED_KEYS.get(model_type, ()))
    for key in required:
        raw = artifacts.get(key)
        if not raw:
            return False, f"artifact_missing:{key}"
        if not resolve_artifact_path(paths, raw).exists():
            return False, f"artifact_missing:{key}"
    metrics_path = payload.get("metrics_path") or artifacts.get("metrics")
    if not metrics_path or not resolve_artifact_path(paths, metrics_path).exists():
        return False, "metrics_missing"
    feature_columns_path = payload.get("feature_columns_path") or artifacts.get("feature_columns")
    if feature_columns_path and not resolve_artifact_path(paths, feature_columns_path).exists():
        return False, "feature_columns_missing"
    feature_set = payload.get("feature_set")
    if not isinstance(feature_set, dict) or not feature_set.get("signature"):
        return False, "feature_set_missing"
    return True, "fresh"


def _validate_gbm_bundle_manifest(paths, payload: dict[str, Any]) -> tuple[bool, str]:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, "artifact_map_missing"
    direction_manifest = artifacts.get("direction_manifest")
    if not direction_manifest:
        return False, "direction_manifest_missing"
    direction_path = resolve_artifact_path(paths, direction_manifest)
    if not direction_path.exists():
        return False, "direction_manifest_missing"
    valid, reason = validate_artifact_bundle(paths, direction_path)
    if not valid:
        return False, f"direction_{reason}"

    direction_payload = load_json(direction_path) or {}
    bundle_feature_set = _manifest_feature_set(payload)
    direction_feature_set = _manifest_feature_set(direction_payload)
    bundle_signature = str(bundle_feature_set.get("signature") or "").strip()
    direction_signature = str(direction_feature_set.get("signature") or "").strip()
    if not bundle_signature:
        return False, "feature_set_missing"
    if not direction_signature or bundle_signature != direction_signature:
        return False, "direction_feature_set_mismatch"

    contract = payload.get("feature_contract") if isinstance(payload.get("feature_contract"), dict) else {}
    contract_direction_signature = str(contract.get("direction_signature") or "").strip()
    if contract_direction_signature and contract_direction_signature != direction_signature:
        return False, "direction_feature_contract_mismatch"

    estimator_contract = payload.get("estimator_contract") if isinstance(payload.get("estimator_contract"), dict) else {}
    bundle_family = _gbm_manifest_estimator_family(payload)
    direction_family = _gbm_manifest_estimator_family(direction_payload)
    contract_family = str(estimator_contract.get("family") or "").strip().lower()
    contract_direction_family = str(estimator_contract.get("direction_family") or "").strip().lower()
    if bundle_family != direction_family:
        return False, "direction_estimator_family_mismatch"
    if contract_family and contract_family != direction_family:
        return False, "estimator_contract_family_mismatch"
    if contract_direction_family and contract_direction_family != direction_family:
        return False, "direction_estimator_contract_mismatch"

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    meta_manifest = meta.get("manifest")
    if meta_manifest:
        meta_path = resolve_artifact_path(paths, meta_manifest)
        if not meta_path.exists():
            return False, "meta_manifest_missing"
        valid, reason = validate_artifact_bundle(paths, meta_path)
        if not valid:
            return False, f"meta_{reason}"
        contract_meta_signature = str(contract.get("meta_signature") or "").strip()
        if contract_meta_signature:
            meta_payload = load_json(meta_path) or {}
            meta_signature = str(_manifest_feature_set(meta_payload).get("signature") or "").strip()
            if not meta_signature or contract_meta_signature != meta_signature:
                return False, "meta_feature_contract_mismatch"
            meta_family = _gbm_manifest_estimator_family(meta_payload)
            contract_meta_family = str(estimator_contract.get("meta_family") or "").strip().lower()
            if meta_family != direction_family:
                return False, "meta_estimator_family_mismatch"
            if contract_meta_family and contract_meta_family != meta_family:
                return False, "meta_estimator_contract_mismatch"
    elif not bool(meta.get("skipped", False)):
        return False, "meta_state_missing"
    return True, "fresh"


def validate_artifact_bundle(paths, manifest_path: Path) -> tuple[bool, str]:
    """Validate an artifact manifest and all required files beneath it."""
    payload = load_json(manifest_path)
    if not payload:
        return False, "manifest_unreadable"
    timestamps = payload.get("timestamps")
    if not isinstance(timestamps, dict) or not timestamps.get("created_at"):
        return False, "timestamps_missing"
    if str(payload.get("model_type", "")) == "gbm" and str(payload.get("model_role", "")) == "bundle":
        return _validate_gbm_bundle_manifest(paths, payload)
    return _validate_standard_manifest(paths, payload)
