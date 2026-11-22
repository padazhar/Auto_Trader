from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import artifact_paths, load_json, save_json
from training.base_trainer import BaseTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _DummyTrainer(BaseTrainer):
    model_type = "rf"

    def train(self, symbol: str, interval: str):  # pragma: no cover - not used in this test
        raise NotImplementedError


class _PromotionCfg(SimpleNamespace):
    enabled: bool = True
    allow_initial_promotion: bool = True
    require_outperform_incumbent: bool = True
    min_fold_count: int = 3
    min_score_improvement: float = 0.002
    min_metric_improvement: float = 0.001
    max_metric_regression: float = 0.01
    min_precision_macro: float = 0.16
    min_roc_auc_ovr: float = 0.56
    min_balanced_accuracy: float = 0.38
    min_precision: float = 0.52
    min_roc_auc: float = 0.56
    require_directional_quality: bool = True
    min_long_precision: float = 0.20
    min_short_precision: float = 0.20
    min_long_recall: float = 0.05
    min_short_recall: float = 0.05
    max_hold_prediction_fraction: float = 0.85
    min_directional_prediction_fraction: float = 0.05


def _paths(tmp_path: Path, *, allow_initial_promotion: bool = True) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    promotion_cfg = _PromotionCfg(allow_initial_promotion=allow_initial_promotion)
    config = SimpleNamespace(training=SimpleNamespace(promotion=promotion_cfg))
    return SimpleNamespace(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        config=config,
        log_file_path=lambda name: logs_dir / f"{name}.log",
    )




def _directional_metrics() -> dict[str, object]:
    return {
        "directional_quality": {
            "available": True,
            "long_precision": 0.75,
            "short_precision": 0.70,
            "long_recall": 0.35,
            "short_recall": 0.30,
            "hold_prediction_fraction": 0.40,
            "directional_prediction_fraction": 0.60,
        }
    }

def test_save_training_bundle_returns_consistent_promotion_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)
    candidate_id = "cand123"
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["a", "b"]})

    bundle = trainer.save_training_bundle(
        symbol="BTCUSDT",
        interval="15m",
        rows_used=10,
        feature_columns=["a", "b"],
        artifacts=artifacts,
        metrics={"precision_macro": 0.2, "roc_auc_ovr": 0.6, "balanced_accuracy": 0.5, "fold_count": 4, **_directional_metrics()},
        candidate_id=candidate_id,
        model_role="direction",
        promote=True,
        extra_manifest={"feature_set": {"version": "feature_store_v2", "signature": "sig123", "columns": ["a", "b"], "column_count": 2}},
    )

    assert "manifest" in bundle
    assert "promoted" in bundle
    assert "promotion" in bundle
    assert bundle["promotion"]["attempted"] is True
    assert bundle["promotion"]["approved"] is True
    assert bundle["promotion"]["promoted"] is True
    assert bundle["promotion"]["candidate_id"] == candidate_id
    assert bundle["promotion"]["reasons"] == []

    pointer = load_json(paths.model_storage_dir / "random_forest_models" / "BTCUSDT" / "15m" / "incumbent.json")
    assert pointer is not None
    assert pointer["candidate_id"] == candidate_id


def test_save_training_bundle_returns_non_promoting_state_when_not_requested(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)
    candidate_id = "cand999"
    artifacts = artifact_paths(paths, "rf", "ETHUSDT", "5m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["x"]})

    bundle = trainer.save_training_bundle(
        symbol="ETHUSDT",
        interval="5m",
        rows_used=8,
        feature_columns=["x"],
        artifacts=artifacts,
        metrics={"precision_macro": 0.25, "roc_auc_ovr": 0.62, "balanced_accuracy": 0.48, "fold_count": 4, **_directional_metrics()},
        candidate_id=candidate_id,
        model_role="direction",
        promote=False,
        extra_manifest={"feature_set": {"version": "feature_store_v2", "signature": "sig999", "columns": ["x"], "column_count": 1}},
    )

    assert bundle["promotion"]["attempted"] is False
    assert bundle["promotion"]["approved"] is False
    assert bundle["promotion"]["promoted"] is False
    assert bundle["promotion"]["candidate_id"] == candidate_id
    assert bundle["promotion"]["reasons"] == ["promotion_not_requested"]
    assert bundle["promoted"] is None


def test_save_training_bundle_respects_initial_promotion_policy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path, allow_initial_promotion=False)
    trainer = _DummyTrainer(paths)
    candidate_id = "cand777"
    artifacts = artifact_paths(paths, "rf", "SOLUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["a"]})

    bundle = trainer.save_training_bundle(
        symbol="SOLUSDT",
        interval="15m",
        rows_used=12,
        feature_columns=["a"],
        artifacts=artifacts,
        metrics={"precision_macro": 0.25, "roc_auc_ovr": 0.62, "balanced_accuracy": 0.48, "fold_count": 4, **_directional_metrics()},
        candidate_id=candidate_id,
        model_role="direction",
        promote=True,
        extra_manifest={"feature_set": {"version": "feature_store_v2", "signature": "sig777", "columns": ["a"], "column_count": 1}},
    )

    assert bundle["promotion"]["attempted"] is True
    assert bundle["promotion"]["approved"] is False
    assert bundle["promotion"]["promoted"] is False
    assert "initial_promotion_disabled" in bundle["promotion"]["reasons"]
    assert bundle["promoted"] is None


def test_directional_quality_report_exposes_long_short_and_hold_dominance() -> None:
    report = BaseTrainer.directional_quality_report(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 1, 2, 0],
        class_encoding={0: "short", 1: "hold", 2: "long"},
    )["directional_quality"]

    assert report["available"] is True
    assert report["rows_valid"] == 6
    assert report["short_precision"] == 0.5
    assert report["short_recall"] == 0.5
    assert report["long_precision"] == 1.0
    assert report["long_recall"] == 0.5
    assert report["confusion_matrix"]["short"]["hold"] == 1
    assert report["class_distribution_valid"] == {"short": 2, "hold": 2, "long": 2}
    assert report["prediction_distribution_valid"] == {"short": 2, "hold": 3, "long": 1}


def test_promotion_rejects_missing_directional_quality_for_direction_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)

    decision = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"precision_macro": 0.30, "roc_auc_ovr": 0.70, "balanced_accuracy": 0.55, "fold_count": 4},
        manifest_path=tmp_path / "candidate.json",
        model_role="direction",
        promote_requested=True,
    )

    assert decision["approved"] is False
    assert "directional_quality_missing" in decision["reasons"]


def test_promotion_rejects_weak_directional_quality(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)

    decision = trainer.evaluate_candidate_promotion(
        symbol="ETHUSDT",
        interval="15m",
        metrics={
            "precision_macro": 0.30,
            "roc_auc_ovr": 0.70,
            "balanced_accuracy": 0.55,
            "fold_count": 4,
            "directional_quality": {
                "available": True,
                "long_precision": 0.10,
                "short_precision": 0.70,
                "long_recall": 0.04,
                "short_recall": 0.30,
                "hold_prediction_fraction": 0.90,
                "directional_prediction_fraction": 0.03,
            },
        },
        manifest_path=tmp_path / "candidate.json",
        model_role="direction",
        promote_requested=True,
    )

    assert decision["approved"] is False
    assert "below_min_long_precision" in decision["reasons"]
    assert "below_min_long_recall" in decision["reasons"]
    assert "hold_prediction_dominates" in decision["reasons"]
    assert "directional_prediction_too_low" in decision["reasons"]


def test_save_training_bundle_writes_canonical_feature_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)
    candidate_id = "cand-canonical"
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["a", "b"]})

    bundle = trainer.save_training_bundle(
        symbol="BTCUSDT",
        interval="15m",
        rows_used=20,
        feature_columns=["a", "b"],
        artifacts=artifacts,
        metrics={"precision_macro": 0.3, "roc_auc_ovr": 0.7, "balanced_accuracy": 0.55, "fold_count": 4, **_directional_metrics()},
        candidate_id=candidate_id,
        promote=False,
        extra_manifest={"feature_set": {"version": "feature_store_v2", "signature": "legacy", "columns": ["a", "b"], "column_count": 2}},
    )

    manifest = bundle["manifest"]
    assert manifest is not None
    feature_set = manifest["feature_set"]
    assert feature_set["recipe_version"] == "feature_store_v2"
    assert feature_set["anchor_interval"] == "15m"
    assert feature_set["columns"] == ["a", "b"]
    assert feature_set["signature"] != "legacy"
