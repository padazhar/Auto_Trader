from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.artifact_contract import load_json, validate_artifact_bundle
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

        return np.array([[0.25, 0.5, 0.25] for _ in range(len(X))], dtype=float)


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


def test_gbm_train_succeeds_when_meta_dataset_is_too_small(monkeypatch, tmp_path: Path) -> None:
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
            pd.DataFrame(
                {
                    "direction_oos_pred": [0.0, 1.0, 2.0, 1.0],
                    "future_return": [0.0, 0.02, -0.03, 0.015],
                    "thr_dyn": [0.001, 0.001, 0.001, 0.001],
                }
            ),
        ),
    )

    original_prepare = trainer._prepare_dataset_with_fallback

    def _fake_prepare(frame, interval, cfg, target_column):
        if target_column == "meta_label":
            raise ValueError("not enough rows after preparation: 345 < 550")
        return original_prepare(frame, interval, cfg, target_column)

    monkeypatch.setattr(trainer, "_prepare_dataset_with_fallback", _fake_prepare)
    monkeypatch.setattr(
        trainer,
        "evaluate_candidate_promotion",
        lambda **kwargs: {"attempted": True, "approved": True, "promoted": False, "reasons": [], "candidate_score": 0.6, "incumbent_score": None},
        raising=False,
    )

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert result.model_type == "gbm"
    assert result.metrics["meta"]["skipped"] is True
    assert result.metrics["meta"]["skip_reason"] == "meta_not_enough_rows_after_preparation"
    manifest_path = Path(result.artifacts["manifest"])
    payload = load_json(manifest_path)
    assert payload is not None
    assert payload["meta"]["skipped"] is True
    assert payload["meta"]["skip_reason"] == "meta_not_enough_rows_after_preparation"
    assert result.metrics["promotion"]["governance"]["paper_eligible"] is True
    assert result.metrics["promotion"]["governance"]["live_eligible"] is False
    assert result.metrics["promotion"]["governance"]["approval_tier"] == "paper_only_direction_fallback"
    assert payload["governance"]["live_eligible"] is False
    assert payload["governance"]["model_operational_role"] == "direction_only_fallback"

    direction_path = trainer.paths.project_root / payload["artifacts"]["direction_manifest"]
    direction_payload = load_json(direction_path)
    assert direction_payload is not None
    assert payload["feature_set"] == direction_payload["feature_set"]
    assert payload["feature_contract"]["execution_authority"] == "direction"
    assert payload["feature_contract"]["direction_signature"] == direction_payload["feature_set"]["signature"]
    assert payload["feature_contract"]["meta_signature"] is None

    valid, reason = validate_artifact_bundle(trainer.paths, manifest_path)
    assert valid is True
    assert reason == "fresh"
