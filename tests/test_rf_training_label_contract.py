from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from training.rf_training_engine import RandomForestTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


class _Paths(SimpleNamespace):
    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        path = self.runtime_dir / "training_progress" / f"{model_type}__{symbol}__{interval}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


class _StubCalibratedModel:
    def predict(self, X):
        return [0 if idx % 3 == 0 else (1 if idx % 3 == 1 else 2) for idx in range(len(X))]

    def predict_proba(self, X):
        import numpy as np

        rows = []
        for idx in range(len(X)):
            if idx % 3 == 0:
                rows.append([0.8, 0.1, 0.1])
            elif idx % 3 == 1:
                rows.append([0.1, 0.8, 0.1])
            else:
                rows.append([0.1, 0.1, 0.8])
        return np.array(rows, dtype=float)


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
        frames=SimpleNamespace(anchor_interval="15m"),
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        labels=SimpleNamespace(horizon_bars=2, atr_multiplier=0.25, atr_floor=0.0004, interval_overrides={}),
        training=SimpleNamespace(
            rf=SimpleNamespace(
                random_state=42,
                tail_rows=100,
                min_rows=20,
                test_size=0.2,
                n_estimators=20,
                max_depth=4,
                min_samples_leaf=2,
                max_samples=0.8,
                n_jobs=1,
                class_weight="balanced_subsample",
                interval_overrides={},
            ),
            validation=SimpleNamespace(walk_forward_folds=2, purge_rows=0, embargo_rows=0),
            promotion=SimpleNamespace(
                enabled=False,
                allow_initial_promotion=True,
                require_outperform_incumbent=False,
                min_fold_count=1,
                min_score_improvement=0.0,
                min_metric_improvement=0.0,
                max_metric_regression=1.0,
                min_precision_macro=0.0,
                min_roc_auc_ovr=0.0,
                min_balanced_accuracy=0.0,
                min_precision=0.0,
                min_roc_auc=0.0,
            ),
        ),
    )
    return _Paths(project_root=project_root, model_storage_dir=model_storage_dir, runtime_dir=runtime_dir, logs_dir=logs_dir, config=config)


def test_rf_trainer_uses_rf_label_without_direction_label(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("training.rf_training_engine.joblib.dump", lambda model, path: Path(path).write_bytes(b"model"))

    trainer = RandomForestTrainer(_make_paths(tmp_path))

    fused = pd.DataFrame(
        {
            "timestamp": list(range(30)),
            "15m_close": [100.0 + idx for idx in range(30)],
            "15m_atr_pct_14": [0.01 for _ in range(30)],
        }
    )
    monkeypatch.setattr(trainer, "load_fused_frame", lambda symbol, interval, tail_rows: fused)

    labeled = fused.copy()
    labeled["future_return"] = [0.01 if idx % 3 == 0 else (0.0 if idx % 3 == 1 else -0.01) for idx in range(30)]
    labeled["thr_dyn"] = [0.001 for _ in range(30)]
    labeled["rf_label"] = [0.0 if idx % 3 == 0 else (1.0 if idx % 3 == 1 else 2.0) for idx in range(30)]
    monkeypatch.setattr("training.rf_training_engine.build_rf_labels", lambda *args, **kwargs: labeled.copy())

    captured: dict[str, object] = {}

    def fake_prepare_supervised_dataset(df, **kwargs):
        captured["target_column"] = kwargs["target_column"]
        assert kwargs["target_column"] == "rf_label"
        frame = labeled[["timestamp", "future_return", "thr_dyn", "rf_label"]].copy()
        frame["feat_a"] = [float(idx) for idx in range(30)]
        frame["feat_b"] = [float(idx % 5) for idx in range(30)]
        split_train = frame.iloc[:24].copy()
        split_valid = frame.iloc[24:].copy()
        walk_train = frame.iloc[:18].copy()
        walk_valid = frame.iloc[18:24].copy()
        from features.dataset_preparation import TrainValidSplit, WalkForwardFold

        return {
            "frame": frame,
            "feature_columns": ["feat_a", "feat_b"],
            "feature_set": {"version": "feature_store_v2", "signature": "abc123", "columns": ["feat_a", "feat_b"], "column_count": 2},
            "split": TrainValidSplit(train=split_train, valid=split_valid),
            "walk_forward_splits": [
                WalkForwardFold(
                    fold_index=0,
                    train=walk_train,
                    valid=walk_valid,
                    train_start_ts=0.0,
                    train_end_ts=17.0,
                    valid_start_ts=18.0,
                    valid_end_ts=23.0,
                )
            ],
            "rows_total": len(frame),
        }

    monkeypatch.setattr("training.rf_training_engine.prepare_supervised_dataset", fake_prepare_supervised_dataset)
    monkeypatch.setattr(trainer, "fit_calibrated_classifier", lambda *args, **kwargs: _StubCalibratedModel())
    monkeypatch.setattr(
        trainer,
        "save_training_bundle",
        lambda **kwargs: {"manifest": {}, "promoted": None, "promotion": {"attempted": False, "approved": False, "promoted": False, "reasons": ["promotion_disabled_by_config"]}},
    )

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert captured["target_column"] == "rf_label"
    assert result.metrics["label_column"] == "rf_label"
    assert result.metrics["rows_total"] == 30
    quality = result.metrics["directional_quality"]
    assert quality["available"] is True
    assert "long_precision" in quality
    assert "short_precision" in quality
    assert "confusion_matrix" in quality
