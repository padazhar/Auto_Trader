from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from core.artifact_contract import (
    load_json,
    paper_bootstrap_active_authority_truth,
    resolve_active_manifest_path,
    resolve_artifact_path,
    validate_execution_manifest,
    validate_manifest_feature_compatibility,
)
from core.log_manager import get_logger


@dataclass(slots=True)
class LoadedRFArtifact:
    model: Any
    feature_columns: list[str]
    metrics: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(slots=True)
class LoadedGBMRoleArtifact:
    model: Any
    feature_columns: list[str]
    metrics: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(slots=True)
class LoadedGBMArtifact:
    direction: LoadedGBMRoleArtifact | None
    meta: LoadedGBMRoleArtifact | None


@dataclass(slots=True)
class LoadedPolicyArtifact:
    model: Any
    feature_columns: list[str]
    metrics: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(slots=True)
class LoadedHMMArtifact:
    model: Any
    feature_columns: list[str]
    metrics: dict[str, Any]
    manifest: dict[str, Any]


# Backward-compatible import alias. New code should use LoadedPolicyArtifact.
LoadedDQNArtifact = LoadedPolicyArtifact


class ModelArtifactLoader:
    """Load the currently promoted incumbent artifact bundles for inference.

    Active artifacts are validated against the current runtime feature-store
    contract before loading. Incompatible artifacts are rejected fail-closed.
    """

    def __init__(self, paths) -> None:
        self.paths = paths
        self.logger = get_logger("inference.model_artifact_loader", paths)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _load_model(path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    def _ensure_manifest_compatible(
        self,
        *,
        manifest_path: Path,
        manifest: dict[str, Any],
        kind: str,
        symbol: str | None = None,
        interval: str | None = None,
        role: str | None = None,
    ) -> bool:
        """Verify manifest feature compatibility against the active runtime config."""
        compatible, reasons = validate_manifest_feature_compatibility(self.paths, manifest)
        if compatible:
            return True

        self.logger.warning(
            "Rejected incompatible %s artifact symbol=%s interval=%s role=%s manifest=%s reasons=%s",
            kind,
            symbol,
            interval,
            role,
            manifest_path,
            reasons,
        )
        return False


    def _registry_allows_execution(
        self,
        model_type: str,
        symbol: str,
        interval: str,
        *,
        manifest_path: Path | None = None,
        model_role: str | None = None,
    ) -> tuple[bool, str]:
        """Fail closed unless active authority is proven for the artifact slot.

        The durable registry is the normal gate. For an explicitly governed
        paper-bootstrap active pointer, a later rejected candidate may leave a
        stale registry row while the active paper-only incumbent remains valid.
        In that narrow paper-only case the pointer/manifest authority contract is
        sufficient and remains non-live.
        """
        registry_path = getattr(self.paths, "registry_path", None)
        if registry_path is None or not Path(registry_path).exists():
            return False, "registry_unavailable"
        try:
            payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        except Exception:
            return False, "registry_unreadable"
        if not isinstance(payload, dict):
            return False, "registry_unreadable"
        key = f"{str(model_type).lower()}:{str(symbol).upper()}:{str(interval)}"
        entry = payload.get(key)
        if not isinstance(entry, dict):
            return False, "registry_entry_missing"
        truth = str(entry.get("registry_truth_status") or "").strip().lower()
        status = str(entry.get("status") or "").strip().lower()
        if (
            truth == "execution_ready"
            and bool(entry.get("execution_ready", False))
            and bool(entry.get("promotion_approved", False))
            and bool(entry.get("active_manifest_valid", False))
        ):
            return True, "ready"
        if (
            status == "success"
            and bool(entry.get("execution_ready", False))
            and bool(entry.get("promotion_approved", False))
            and bool(entry.get("active_manifest_valid", False))
        ):
            return True, "ready"
        registry_reason = f"registry_not_execution_ready:{truth or status or 'unknown'}"
        if manifest_path is not None:
            bootstrap_authority = paper_bootstrap_active_authority_truth(
                self.paths, model_type, symbol, interval, model_role=model_role
            )
            authority_path = bootstrap_authority.get("active_manifest_path")
            same_manifest = bool(
                authority_path and Path(str(authority_path)).resolve() == Path(manifest_path).resolve()
            )
            if bool(bootstrap_authority.get("authority_approved", False)) and same_manifest:
                return True, "paper_bootstrap_ready_registry_reconciled_from_active_pointer"
        return False, registry_reason

    def _paths_from_manifest(self, manifest: dict[str, Any]) -> tuple[Path | None, Path | None, Path | None]:
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
        model_path = resolve_artifact_path(self.paths, artifacts.get("model")) if artifacts.get("model") else None
        feature_columns_path = resolve_artifact_path(self.paths, manifest.get("feature_columns_path") or artifacts.get("feature_columns")) if (manifest.get("feature_columns_path") or artifacts.get("feature_columns")) else None
        metrics_path = resolve_artifact_path(self.paths, manifest.get("metrics_path") or artifacts.get("metrics")) if (manifest.get("metrics_path") or artifacts.get("metrics")) else None
        return model_path, feature_columns_path, metrics_path

    def _load_standard_from_manifest(self, manifest_path: Path, kind: str, *, symbol: str, interval: str, require_execution_ready: bool = True):
        if require_execution_ready:
            execution_ready, execution_reason, manifest = validate_execution_manifest(self.paths, manifest_path)
            if not execution_ready:
                self.logger.warning(
                    "Rejected non-ready %s artifact symbol=%s interval=%s manifest=%s reason=%s",
                    kind,
                    symbol,
                    interval,
                    manifest_path,
                    execution_reason,
                )
                return None
        else:
            manifest = load_json(manifest_path) or {}
            if not manifest:
                return None
        if not self._ensure_manifest_compatible(
            manifest_path=manifest_path,
            manifest=manifest,
            kind=kind,
            symbol=symbol,
            interval=interval,
        ):
            return None
        if require_execution_ready:
            allowed, registry_reason = self._registry_allows_execution(kind, symbol, interval, manifest_path=manifest_path)
            if not allowed:
                self.logger.warning(
                    "Rejected %s artifact without registry execution approval symbol=%s interval=%s manifest=%s reason=%s",
                    kind,
                    symbol,
                    interval,
                    manifest_path,
                    registry_reason,
                )
                return None

        model_path, feature_columns_path, metrics_path = self._paths_from_manifest(manifest)
        if model_path is None or feature_columns_path is None or metrics_path is None:
            return None
        model = self._load_model(model_path)
        if model is None:
            return None
        feature_payload = self._load_json(feature_columns_path)
        metrics_payload = self._load_json(metrics_path)
        feature_columns = [str(c) for c in feature_payload.get("feature_columns", [])]
        if not feature_columns:
            return None
        if kind == "rf":
            cls = LoadedRFArtifact
        elif kind == "hmm":
            cls = LoadedHMMArtifact
        else:
            cls = LoadedPolicyArtifact
        return cls(model=model, feature_columns=feature_columns, metrics=metrics_payload, manifest=manifest)

    def _load_gbm_role(self, symbol: str, interval: str, role: str) -> LoadedGBMRoleArtifact | None:
        manifest_path = resolve_active_manifest_path(self.paths, "gbm", symbol, interval, model_role=role)
        if manifest_path is None:
            return None
        execution_ready, execution_reason, manifest = validate_execution_manifest(self.paths, manifest_path)
        if not execution_ready:
            self.logger.warning(
                "Rejected non-ready gbm artifact symbol=%s interval=%s role=%s manifest=%s reason=%s",
                symbol,
                interval,
                role,
                manifest_path,
                execution_reason,
            )
            return None
        if not self._ensure_manifest_compatible(
            manifest_path=manifest_path,
            manifest=manifest,
            kind="gbm",
            symbol=symbol,
            interval=interval,
            role=role,
        ):
            return None
        allowed, registry_reason = self._registry_allows_execution("gbm", symbol, interval, manifest_path=manifest_path, model_role=role)
        if not allowed:
            self.logger.warning(
                "Rejected gbm artifact without registry execution approval symbol=%s interval=%s role=%s manifest=%s reason=%s",
                symbol,
                interval,
                role,
                manifest_path,
                registry_reason,
            )
            return None

        model_path, feature_columns_path, metrics_path = self._paths_from_manifest(manifest)
        if model_path is None or feature_columns_path is None or metrics_path is None:
            return None
        model = self._load_model(model_path)
        if model is None:
            return None
        feature_payload = self._load_json(feature_columns_path)
        metrics_payload = self._load_json(metrics_path)
        feature_columns = [str(c) for c in feature_payload.get("feature_columns", [])]
        if not feature_columns:
            return None
        return LoadedGBMRoleArtifact(model=model, feature_columns=feature_columns, metrics=metrics_payload, manifest=manifest)



    def _candidate_manifest_paths(self, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> list[Path]:
        base = self.paths.model_storage_dir
        if model_type == "rf":
            root = base / "random_forest_models" / str(symbol).upper() / str(interval)
        elif model_type == "gbm":
            root = base / "gradient_boost_models" / str(symbol).upper() / str(interval)
            if model_role:
                root = root / str(model_role).lower()
        elif model_type == "dqn":
            primary = base / "reinforcement_agents" / str(symbol).upper() / str(interval)
            secondary = base / "policy_models" / str(symbol).upper() / str(interval)
            roots = [primary, secondary]
            manifests: list[Path] = []
            for candidate_root in roots:
                manifests.extend(sorted(candidate_root.glob("candidates/*/manifest.json")))
            return manifests
        elif model_type == "hmm":
            root = base / "hmm_regime_models" / str(symbol).upper() / str(interval)
        else:
            return []
        return sorted(root.glob("candidates/*/manifest.json"))

    @staticmethod
    def _manifest_sort_key(manifest_path: Path) -> tuple[float, str]:
        manifest = load_json(manifest_path) or {}
        timestamps = manifest.get("timestamps") if isinstance(manifest.get("timestamps"), dict) else {}
        lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
        raw_updated = timestamps.get("updated_at") or timestamps.get("created_at") or lifecycle.get("promoted_at")
        if isinstance(raw_updated, str):
            try:
                from datetime import datetime
                stamp = datetime.fromisoformat(raw_updated.replace("Z", "+00:00")).timestamp()
            except Exception:
                stamp = manifest_path.stat().st_mtime
        else:
            stamp = manifest_path.stat().st_mtime
        return (float(stamp), str(manifest_path))

    def _latest_candidate_manifest_path(self, model_type: str, symbol: str, interval: str, *, model_role: str | None = None) -> Path | None:
        manifests = self._candidate_manifest_paths(model_type, symbol, interval, model_role=model_role)
        if not manifests:
            return None
        manifests.sort(key=self._manifest_sort_key, reverse=True)
        return manifests[0]

    def load_rf(self, symbol: str, interval: str) -> LoadedRFArtifact | None:
        manifest_path = resolve_active_manifest_path(self.paths, "rf", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "rf", symbol=symbol, interval=interval)

    def load_gbm(self, symbol: str, interval: str) -> LoadedGBMArtifact | None:
        direction = self._load_gbm_role(symbol, interval, "direction")
        meta = self._load_gbm_role(symbol, interval, "meta")
        if direction is None and meta is None:
            return None
        return LoadedGBMArtifact(direction=direction, meta=meta)

    def load_candidate_rf(self, symbol: str, interval: str) -> LoadedRFArtifact | None:
        manifest_path = self._latest_candidate_manifest_path("rf", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "rf", symbol=symbol, interval=interval, require_execution_ready=False)

    def load_candidate_gbm(self, symbol: str, interval: str) -> LoadedGBMArtifact | None:
        direction_manifest = self._latest_candidate_manifest_path("gbm", symbol, interval, model_role="direction")
        meta_manifest = self._latest_candidate_manifest_path("gbm", symbol, interval, model_role="meta")
        direction = self._load_gbm_role(symbol, interval, "direction") if False else None
        if direction_manifest is not None:
            direction_manifest_payload = load_json(direction_manifest) or {}
            if direction_manifest_payload and self._ensure_manifest_compatible(manifest_path=direction_manifest, manifest=direction_manifest_payload, kind="gbm", symbol=symbol, interval=interval, role="direction"):
                model_path, feature_columns_path, metrics_path = self._paths_from_manifest(direction_manifest_payload)
                if model_path is not None and feature_columns_path is not None and metrics_path is not None:
                    model = self._load_model(model_path)
                    feature_payload = self._load_json(feature_columns_path)
                    metrics_payload = self._load_json(metrics_path)
                    feature_columns = [str(c) for c in feature_payload.get("feature_columns", [])]
                    if model is not None and feature_columns:
                        direction = LoadedGBMRoleArtifact(model=model, feature_columns=feature_columns, metrics=metrics_payload, manifest=direction_manifest_payload)
        meta = None
        if meta_manifest is not None:
            meta_manifest_payload = load_json(meta_manifest) or {}
            if meta_manifest_payload and self._ensure_manifest_compatible(manifest_path=meta_manifest, manifest=meta_manifest_payload, kind="gbm", symbol=symbol, interval=interval, role="meta"):
                model_path, feature_columns_path, metrics_path = self._paths_from_manifest(meta_manifest_payload)
                if model_path is not None and feature_columns_path is not None and metrics_path is not None:
                    model = self._load_model(model_path)
                    feature_payload = self._load_json(feature_columns_path)
                    metrics_payload = self._load_json(metrics_path)
                    feature_columns = [str(c) for c in feature_payload.get("feature_columns", [])]
                    if model is not None and feature_columns:
                        meta = LoadedGBMRoleArtifact(model=model, feature_columns=feature_columns, metrics=metrics_payload, manifest=meta_manifest_payload)
        if direction is None and meta is None:
            return None
        return LoadedGBMArtifact(direction=direction, meta=meta)

    def load_candidate_hmm(self, symbol: str, interval: str) -> LoadedHMMArtifact | None:
        """Load the latest candidate HMM regime confirmer for shadow telemetry only."""
        manifest_path = self._latest_candidate_manifest_path("hmm", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "hmm", symbol=symbol, interval=interval, require_execution_ready=False)

    def load_hmm(self, symbol: str, interval: str) -> LoadedHMMArtifact | None:
        """Load an active HMM regime confirmer if a future paper gate promotes one."""
        manifest_path = resolve_active_manifest_path(self.paths, "hmm", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "hmm", symbol=symbol, interval=interval)

    def load_candidate_policy(self, symbol: str, interval: str) -> LoadedPolicyArtifact | None:
        """Load the latest candidate confirmatory policy artifact.

        Artifacts remain stored in the legacy ``dqn`` compatibility slot so
        historical manifests and paths continue to resolve.
        """
        manifest_path = self._latest_candidate_manifest_path("dqn", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "dqn", symbol=symbol, interval=interval, require_execution_ready=False)

    def load_policy(self, symbol: str, interval: str) -> LoadedPolicyArtifact | None:
        """Load the active confirmatory policy artifact from the legacy slot."""
        manifest_path = resolve_active_manifest_path(self.paths, "dqn", symbol, interval)
        if manifest_path is None:
            return None
        return self._load_standard_from_manifest(manifest_path, "dqn", symbol=symbol, interval=interval)

    def load_candidate_dqn(self, symbol: str, interval: str) -> LoadedPolicyArtifact | None:
        """Backward-compatible alias for the legacy dqn compatibility slot."""
        return self.load_candidate_policy(symbol, interval)

    def load_dqn(self, symbol: str, interval: str) -> LoadedPolicyArtifact | None:
        """Backward-compatible alias for the legacy dqn compatibility slot."""
        return self.load_policy(symbol, interval)
