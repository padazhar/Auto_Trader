from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.artifact_contract import load_json, save_json, validate_artifact_bundle
from training.gbm_training_engine import GBMTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _DummyPromotionCfg(SimpleNamespace):
    enabled: bool = True
    allow_initial_promotion: bool = True
    require_outperform_incumbent: bool = True
    min_fold_count: int = 1
    min_score_improvement: float = 0.0
    min_metric_improvement: float = 0.0
    max_metric_regression: float = 1.0
    min_precision_macro: float = 0.0
    min_roc_auc_ovr: float = 0.0
    min_balanced_accuracy: float = 0.0
    min_precision: float = 0.0
    min_roc_auc: float = 0.0


class _Paths(SimpleNamespace):
    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        path = self.runtime_dir / "training_progress" / f"{model_type}__{symbol}__{interval}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


class _StubModel:
    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.25, 0.75] for _ in range(len(X))], dtype=float)


def _make_paths(tmp_path: Path) -> _Paths:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42),
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        trading=SimpleNamespace(min_profit_atr_mult=0.3, fees_bps=3.0, slippage_bps=2.0),
        labels=SimpleNamespace(horizon_bars=3, atr_multiplier=0.25, atr_floor=0.0004, interval_overrides={}),
        training=SimpleNamespace(
            gbm=SimpleNamespace(
                random_state=42,
                tail_rows=100,
                min_rows=20,
                test_size=0.2,
                n_estimators=50,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=5,
                interval_overrides={},
            ),
            validation=SimpleNamespace(walk_forward_folds=2, purge_rows=0, embargo_rows=0),
            promotion=_DummyPromotionCfg(),
        ),
    )
    return _Paths(project_root=project_root, model_storage_dir=model_storage_dir, runtime_dir=runtime_dir, logs_dir=logs_dir, config=config)


def test_gbm_train_smoke_writes_bundle_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("training.gbm_training_engine.joblib.dump", lambda model, path: Path(path).write_bytes(b"model"))

    trainer = GBMTrainer(_make_paths(tmp_path))

    fused = pd.DataFrame({"timestamp": [1, 2, 3], "15m_close": [100.0, 101.0, 102.0], "15m_atr_pct_14": [0.01, 0.01, 0.01]})
    monkeypatch.setattr(trainer, "load_fused_frame", lambda symbol, interval, tail_rows: fused)
    monkeypatch.setattr(
        trainer,
        "_train_direction_model",
        lambda symbol, interval, fused, gbm_cfg, labels_cfg, random_state, candidate_id: (
            {
                "model": _StubModel(),
                "feature_columns": ["feat_a", "feat_b"],
                "rows_total": 12,
                "feature_set": {"version": "feature_store_v2", "signature": "dirsig", "columns": ["feat_a", "feat_b"], "column_count": 2},
                "validation_summary": {"fold_count": 2, "fold_metrics": [{"fold_index": 0}, {"fold_index": 1}]},
            },
            {"roc_auc_ovr": 0.7, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 2},
            pd.DataFrame({"direction_oos_pred": [0.0, 1.0], "future_return": [0.01, -0.01], "thr_dyn": [0.001, 0.001]}),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_train_meta_model",
        lambda symbol, interval, labeled, gbm_cfg, random_state, candidate_id: (
            {
                "model": _StubModel(),
                "feature_columns": ["feat_c"],
                "rows_total": 10,
                "feature_set": {"version": "feature_store_v2", "signature": "metasig", "columns": ["feat_c"], "column_count": 1},
                "validation_summary": {"fold_count": 1, "fold_metrics": [{"fold_index": 0}]},
            },
            {"roc_auc": 0.68, "precision": 0.55, "balanced_accuracy": 0.57, "fold_count": 1, "skipped": False},
        ),
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_candidate_promotion",
        lambda **kwargs: {"attempted": True, "approved": True, "promoted": False, "reasons": [], "candidate_score": 0.6, "incumbent_score": None},
        raising=False,
    )

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert result.model_type == "gbm"
    manifest_path = Path(result.artifacts["manifest"])
    payload = load_json(manifest_path)
    assert payload is not None
    assert payload["model_type"] == "gbm"
    assert payload["promotion"]["approved"] is True
    assert payload["promotion"]["promoted"] is True
    assert payload["governance"]["live_eligible"] is True
    assert payload["governance"]["approval_tier"] == "full_bundle_live_candidate"
    assert payload["governance"]["model_operational_role"] == "direction_plus_meta"

    direction_path = trainer.paths.project_root / payload["artifacts"]["direction_manifest"]
    meta_path = trainer.paths.project_root / payload["meta"]["manifest"]
    direction_payload = load_json(direction_path)
    meta_payload = load_json(meta_path)
    assert direction_payload is not None
    assert meta_payload is not None
    assert payload["feature_set"] == direction_payload["feature_set"]
    assert payload["feature_contract"]["execution_authority"] == "direction"
    assert payload["feature_contract"]["direction_signature"] == direction_payload["feature_set"]["signature"]
    assert payload["feature_contract"]["meta_signature"] == meta_payload["feature_set"]["signature"]

    valid, reason = validate_artifact_bundle(trainer.paths, manifest_path)
    assert valid is True
    assert reason == "fresh"

    payload["feature_set"]["signature"] = "corrupt-bundle-signature"
    save_json(manifest_path, payload)
    valid, reason = validate_artifact_bundle(trainer.paths, manifest_path)
    assert valid is False
    assert reason == "direction_feature_set_mismatch"


def test_lightgbm_estimator_switch_writes_candidate_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("training.gbm_training_engine.joblib.dump", lambda model, path: Path(path).write_bytes(b"model"))

    paths = _make_paths(tmp_path)
    paths.config.training.gbm.estimator = "lightgbm"
    trainer = GBMTrainer(paths)

    fused = pd.DataFrame({"timestamp": [1, 2, 3], "15m_close": [100.0, 101.0, 102.0], "15m_atr_pct_14": [0.01, 0.01, 0.01]})
    monkeypatch.setattr(trainer, "load_fused_frame", lambda symbol, interval, tail_rows: fused)
    monkeypatch.setattr(
        trainer,
        "_train_direction_model",
        lambda symbol, interval, fused, gbm_cfg, labels_cfg, random_state, candidate_id: (
            {
                "model": _StubModel(),
                "feature_columns": ["feat_a", "feat_b"],
                "rows_total": 12,
                "feature_set": {"version": "feature_store_v2", "signature": "dirsig-lgbm", "columns": ["feat_a", "feat_b"], "column_count": 2},
                "validation_summary": {"fold_count": 2, "fold_metrics": [{"fold_index": 0}, {"fold_index": 1}]},
            },
            {"roc_auc_ovr": 0.7, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 2},
            pd.DataFrame({"direction_oos_pred": [0.0, 1.0], "future_return": [0.01, -0.01], "thr_dyn": [0.001, 0.001]}),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_train_meta_model",
        lambda symbol, interval, labeled, gbm_cfg, random_state, candidate_id: (
            None,
            {"rows_total": 10, "model_role": "meta", "skipped": True, "skip_reason": "test_meta_skipped"},
        ),
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_candidate_promotion",
        lambda **kwargs: {"attempted": True, "approved": False, "promoted": False, "reasons": ["insufficient_paper_trades:0"], "candidate_score": 0.6, "incumbent_score": None},
        raising=False,
    )

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert result.metrics["estimator_family"] == "lightgbm"
    manifest_path = Path(result.artifacts["manifest"])
    payload = load_json(manifest_path)
    assert payload is not None
    assert payload["estimator_family"] == "lightgbm"
    assert payload["estimator_contract"] == {
        "family": "lightgbm",
        "direction_family": "lightgbm",
        "meta_family": None,
        "selector_key": "training.gbm.estimator",
        "rollback_family": "histgbm",
    }
    assert payload["training"]["estimator_family"] == "lightgbm"
    direction_path = trainer.paths.project_root / payload["artifacts"]["direction_manifest"]
    direction_payload = load_json(direction_path)
    assert direction_payload is not None
    assert direction_payload["estimator_family"] == "lightgbm"
    assert direction_payload["estimator_contract"]["rollback_family"] == "histgbm"

    valid, reason = validate_artifact_bundle(trainer.paths, manifest_path)
    assert valid is True
    assert reason == "fresh"


def test_gbm_bundle_rejects_direction_estimator_family_mismatch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("training.gbm_training_engine.joblib.dump", lambda model, path: Path(path).write_bytes(b"model"))

    paths = _make_paths(tmp_path)
    paths.config.training.gbm.estimator = "lightgbm"
    trainer = GBMTrainer(paths)

    fused = pd.DataFrame({"timestamp": [1, 2, 3], "15m_close": [100.0, 101.0, 102.0], "15m_atr_pct_14": [0.01, 0.01, 0.01]})
    monkeypatch.setattr(trainer, "load_fused_frame", lambda symbol, interval, tail_rows: fused)
    monkeypatch.setattr(
        trainer,
        "_train_direction_model",
        lambda symbol, interval, fused, gbm_cfg, labels_cfg, random_state, candidate_id: (
            {
                "model": _StubModel(),
                "feature_columns": ["feat_a", "feat_b"],
                "rows_total": 12,
                "feature_set": {"version": "feature_store_v2", "signature": "dirsig-lgbm", "columns": ["feat_a", "feat_b"], "column_count": 2},
                "validation_summary": {"fold_count": 2, "fold_metrics": [{"fold_index": 0}, {"fold_index": 1}]},
            },
            {"roc_auc_ovr": 0.7, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 2},
            pd.DataFrame({"direction_oos_pred": [0.0, 1.0], "future_return": [0.01, -0.01], "thr_dyn": [0.001, 0.001]}),
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_train_meta_model",
        lambda symbol, interval, labeled, gbm_cfg, random_state, candidate_id: (None, {"rows_total": 10, "model_role": "meta", "skipped": True, "skip_reason": "test_meta_skipped"}),
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_candidate_promotion",
        lambda **kwargs: {"attempted": True, "approved": False, "promoted": False, "reasons": ["insufficient_paper_trades:0"], "candidate_score": 0.6, "incumbent_score": None},
        raising=False,
    )

    result = trainer.train("BTCUSDT", "15m")
    manifest_path = Path(result.artifacts["manifest"])
    payload = load_json(manifest_path)
    assert payload is not None
    payload["estimator_family"] = "histgbm"
    save_json(manifest_path, payload)

    valid, reason = validate_artifact_bundle(trainer.paths, manifest_path)
    assert valid is False
    assert reason == "direction_estimator_family_mismatch"


def test_lightgbm_estimator_switch_builds_lgbm_classifier(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    paths.config.training.gbm.estimator = "lightgbm"
    model = GBMTrainer(paths)._build_model(paths.config.training.gbm, "15m", 42)

    assert model.__class__.__name__ == "LGBMClassifier"
