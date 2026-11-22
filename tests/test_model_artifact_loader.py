from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib

from core.artifact_contract import artifact_paths, incumbent_pointer_path, load_json, save_json, write_manifest
from features.feature_store_contract import build_feature_store_metadata
from inference.model_artifact_loader import ModelArtifactLoader
from inference.readiness import model_registry_readiness


class _DummyLogger:
    def warning(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def _paths(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    for path in (model_storage_dir, runtime_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    registry_path = runtime_dir / "training_registry.json"
    registry_path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        registry_path=registry_path,
        config=SimpleNamespace(
            runtime=SimpleNamespace(execution_mode="paper"),
            feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
            frames=SimpleNamespace(anchor_interval="15m"),
        ),
        log_file_path=lambda name: logs_dir / f"{name}.log",
    )


def _write_rf_candidate(paths: SimpleNamespace, *, candidate_id: str, stage: str = "candidate") -> Path:
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": "dummy-model"}, artifacts["model"])
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a"]})
    metrics = {
        "rows_total": 100,
        "precision_macro": 0.25,
        "roc_auc_ovr": 0.60,
        "balanced_accuracy": 0.45,
        "fold_count": 3,
    }
    feature_set = build_feature_store_metadata(
        ["feat_a"],
        recipe_version="feature_store_v2",
        anchor_interval="15m",
        optional_context_fields=[],
    )
    write_manifest(
        paths,
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        feature_count=1,
        artifacts=artifacts,
        metrics=metrics,
        extra={
            "feature_set": feature_set,
            "lifecycle": {
                "stage": stage,
                "candidate_id": candidate_id,
                "promoted_at": "2026-04-24T00:00:00+00:00" if stage != "candidate" else None,
            },
        },
    )
    if stage == "paper_bootstrap":
        payload = load_json(artifacts["manifest"])
        lifecycle = dict(payload.get("lifecycle") or {})
        lifecycle.update({"paper_execution_only": True, "live_eligible": False})
        payload["lifecycle"] = lifecycle
        save_json(artifacts["manifest"], payload)
    return artifacts["manifest"]


def _point_incumbent_at(paths: SimpleNamespace, manifest_path: Path, *, candidate_id: str) -> None:
    pointer = incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m")
    pointer.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        pointer,
        {
            "schema_version": 1,
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": candidate_id,
            "manifest_path": manifest_path.relative_to(paths.project_root).as_posix(),
            "promoted_at": "2026-04-24T00:00:00+00:00",
        },
    )



def _write_registry_ready(paths: SimpleNamespace, *, model_type: str = "rf", symbol: str = "BTCUSDT", interval: str = "15m") -> None:
    paths.registry_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        paths.registry_path,
        {
            f"{model_type}:{symbol}:{interval}": {
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "status": "success",
                "promotion_attempted": True,
                "promotion_approved": True,
                "candidate_saved": True,
                "candidate_manifest_valid": True,
                "active_manifest_valid": True,
                "execution_ready": True,
                "registry_truth_status": "execution_ready",
            }
        },
    )

def test_loader_rejects_candidate_manifest_even_when_pointer_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("inference.model_artifact_loader.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    manifest_path = _write_rf_candidate(paths, candidate_id="cand-1", stage="candidate")
    _point_incumbent_at(paths, manifest_path, candidate_id="cand-1")

    assert ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m") is None

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert readiness.reason == "deployment_stage:candidate"


def test_loader_accepts_promoted_incumbent_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("inference.model_artifact_loader.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    manifest_path = _write_rf_candidate(paths, candidate_id="cand-2", stage="incumbent")
    _point_incumbent_at(paths, manifest_path, candidate_id="cand-2")

    _write_registry_ready(paths)

    artifact = ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m")
    assert artifact is not None
    assert artifact.manifest["lifecycle"]["stage"] == "incumbent"

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is True
    assert readiness.reason == "ready"


def test_loader_rejects_incumbent_manifest_without_registry_execution_approval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("inference.model_artifact_loader.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    manifest_path = _write_rf_candidate(paths, candidate_id="cand-3", stage="incumbent")
    _point_incumbent_at(paths, manifest_path, candidate_id="cand-3")

    assert ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m") is None

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert readiness.reason == "registry_unavailable"


def test_readiness_rejects_manifest_when_artifact_feature_columns_diverge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("inference.model_artifact_loader.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    manifest_path = _write_rf_candidate(paths, candidate_id="cand-feature-mismatch", stage="incumbent")
    feature_columns_path = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id="cand-feature-mismatch")["feature_columns"]
    save_json(feature_columns_path, {"feature_columns": ["unexpected_feat"]})
    _point_incumbent_at(paths, manifest_path, candidate_id="cand-feature-mismatch")
    _write_registry_ready(paths)

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert "feature_incompatible" in readiness.reason
    assert "artifact_feature_columns_mismatch" in readiness.reason
    assert ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m") is None


def test_loader_accepts_pointer_proven_paper_bootstrap_when_latest_registry_candidate_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("inference.model_artifact_loader.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    manifest_path = _write_rf_candidate(paths, candidate_id="paper-bootstrap-active", stage="paper_bootstrap")
    _point_incumbent_at(paths, manifest_path, candidate_id="paper-bootstrap-active")
    pointer = load_json(incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"))
    pointer.update({
        "deployment_stage": "paper_bootstrap",
        "paper_execution_only": True,
        "paper_evidence_status": "paper_bootstrap_execution_ready",
        "live_eligible": False,
    })
    save_json(incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"), pointer)
    save_json(
        paths.registry_path,
        {
            "rf:BTCUSDT:15m": {
                "model_type": "rf",
                "symbol": "BTCUSDT",
                "interval": "15m",
                "status": "success",
                "promotion_attempted": True,
                "promotion_approved": False,
                "active_manifest_valid": False,
                "execution_ready": False,
                "registry_truth_status": "promotion_rejected",
            }
        },
    )

    artifact = ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m")
    assert artifact is not None
    assert artifact.manifest["lifecycle"]["stage"] == "paper_bootstrap"

    paths.config.runtime.execution_mode = "live"
    assert ModelArtifactLoader(paths).load_rf("BTCUSDT", "15m") is None
