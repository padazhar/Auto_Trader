from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib

from core.artifact_contract import artifact_paths, load_json, promote_candidate_manifest, save_json, write_manifest
from inference.model_artifact_loader import ModelArtifactLoader
from features.feature_store_contract import build_feature_store_metadata


class _DummyModel:
    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.1, 0.2, 0.7] for _ in range(len(X))], dtype=float)


def _paths(tmp_path: Path, *, recipe_version: str = "feature_store_v2") -> SimpleNamespace:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(feature_store=SimpleNamespace(recipe_version=recipe_version))
    return SimpleNamespace(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        registry_path=runtime_dir / "training_registry.json",
        config=config,
        log_file_path=lambda name: logs_dir / f"{name}.log",
    )


def _write_rf_candidate(paths, *, candidate_id: str, feature_set: dict[str, object]) -> Path:
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(_DummyModel(), artifacts["model"])
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a", "feat_b"]})
    save_json(artifacts["metrics"], {"precision_macro": 0.25, "roc_auc_ovr": 0.62, "balanced_accuracy": 0.48})
    write_manifest(
        paths,
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=10,
        feature_count=2,
        artifacts=artifacts,
        metrics={"precision_macro": 0.25, "roc_auc_ovr": 0.62, "balanced_accuracy": 0.48},
        extra={"feature_set": feature_set, "model_role": "direction"},
    )
    promote_candidate_manifest(
        paths,
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        manifest_path=artifacts["manifest"],
        candidate_id=candidate_id,
        model_role=None,
        metrics={"precision_macro": 0.25, "roc_auc_ovr": 0.62, "balanced_accuracy": 0.48},
    )
    save_json(
        paths.registry_path,
        {
            "rf:BTCUSDT:15m": {
                "model_type": "rf",
                "symbol": "BTCUSDT",
                "interval": "15m",
                "status": "success",
                "registry_truth_status": "execution_ready",
                "execution_ready": True,
                "promotion_approved": True,
                "active_manifest_valid": True,
            }
        },
    )
    return artifacts["manifest"]


def test_model_loader_accepts_compatible_feature_set(tmp_path: Path) -> None:
    paths = _paths(tmp_path, recipe_version="feature_store_v2")
    _write_rf_candidate(
        paths,
        candidate_id="cand-ok",
        feature_set=build_feature_store_metadata(
            ["feat_a", "feat_b"],
            recipe_version="feature_store_v2",
            anchor_interval="15m",
            optional_context_fields=[],
        ),
    )

    # Recompute the correct signature using the same contract helper by writing the manifest first,
    # then reading it back and re-saving with the matching value for this test.
    loader = ModelArtifactLoader(paths)
    artifact = loader.load_rf("BTCUSDT", "15m")
    assert artifact is not None
    assert artifact.feature_columns == ["feat_a", "feat_b"]


def test_model_loader_rejects_recipe_version_mismatch(tmp_path: Path) -> None:
    paths = _paths(tmp_path, recipe_version="feature_store_v2")
    manifest_path = _write_rf_candidate(
        paths,
        candidate_id="cand-bad-version",
        feature_set={
            "version": "feature_store_v1",
            "recipe_version": "feature_store_v1",
            "signature": "7b7f5182c5407d0b5a474ea4f88c02bc45a441fc57d6a445221189c4994177a7",
            "columns": ["feat_a", "feat_b"],
            "column_count": 2,
            "optional_context_fields": [],
        },
    )

    payload = json_load(manifest_path)
    assert payload is not None
    loader = ModelArtifactLoader(paths)
    artifact = loader.load_rf("BTCUSDT", "15m")
    assert artifact is None


def test_model_loader_rejects_signature_mismatch(tmp_path: Path) -> None:
    paths = _paths(tmp_path, recipe_version="feature_store_v2")
    manifest_path = _write_rf_candidate(
        paths,
        candidate_id="cand-bad-signature",
        feature_set={
            "version": "feature_store_v2",
            "recipe_version": "feature_store_v2",
            "signature": "not-a-real-signature",
            "columns": ["feat_a", "feat_b"],
            "column_count": 2,
            "optional_context_fields": [],
        },
    )

    payload = load_json(manifest_path)
    assert payload is not None
    payload["feature_set"]["signature"] = "not-a-real-signature"
    save_json(manifest_path, payload)

    loader = ModelArtifactLoader(paths)
    artifact = loader.load_rf("BTCUSDT", "15m")
    assert artifact is None


def json_load(path: Path) -> dict | None:
    import json

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
