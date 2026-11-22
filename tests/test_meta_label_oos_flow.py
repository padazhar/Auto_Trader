from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from features.dataset_preparation import TrainValidSplit, WalkForwardFold
from training.gbm_training_engine import GBMTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _Paths(SimpleNamespace):
    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        path = self.runtime_dir / "training_progress" / f"{model_type}__{symbol}__{interval}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


class _StubModel:
    def predict(self, X):
        return [0 if idx % 2 == 0 else 1 for idx in range(len(X))]

    def predict_proba(self, X):
        import numpy as np
        rows = []
        for idx in range(len(X)):
            rows.append([0.8, 0.2] if idx % 2 == 0 else [0.2, 0.8])
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
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        trading=SimpleNamespace(min_profit_atr_mult=0.3, fees_bps=3.0, slippage_bps=2.0),
        labels=SimpleNamespace(horizon_bars=3, atr_multiplier=0.25, atr_floor=0.0004, interval_overrides={}),
        training=SimpleNamespace(
            gbm=SimpleNamespace(random_state=42, tail_rows=100, min_rows=20, test_size=0.2, interval_overrides={}),
            validation=SimpleNamespace(walk_forward_folds=2, purge_rows=0, embargo_rows=0),
            promotion=SimpleNamespace(enabled=False),
        ),
    )
    return _Paths(project_root=project_root, model_storage_dir=model_storage_dir, runtime_dir=runtime_dir, logs_dir=logs_dir, config=config)


def test_prepare_direction_dataset_records_oos_predictions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_make_paths(tmp_path))

    labeled = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "future_return": [0.02, 0.01, -0.01, -0.02],
            "thr_dyn": [0.001, 0.001, 0.001, 0.001],
            "direction_label": [0.0, 1.0, 2.0, 1.0],
        }
    )
    fold = WalkForwardFold(
        fold_index=0,
        train=labeled.iloc[:2].copy(),
        valid=labeled.iloc[2:].copy(),
        train_start_ts=1.0,
        train_end_ts=2.0,
        valid_start_ts=3.0,
        valid_end_ts=4.0,
    )

    monkeypatch.setattr("training.gbm_training_engine.build_direction_labels", lambda *args, **kwargs: labeled.copy())
    monkeypatch.setattr(
        trainer,
        "_prepare_dataset_with_fallback",
        lambda fused, interval, cfg, target_column: (
            {
                "frame": labeled.copy(),
                "feature_columns": ["feat_a"],
                "rows_total": len(labeled),
                "feature_set": {"version": "feature_store_v2", "signature": "sig1", "columns": ["feat_a"], "column_count": 1},
                "walk_forward_splits": [fold],
                "split": TrainValidSplit(train=labeled.iloc[:2].copy(), valid=labeled.iloc[2:].copy()),
            },
            "full_fused",
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_evaluate_direction_fold",
        lambda fold, feature_columns, gbm_cfg, interval, random_state: (
            {"fold_index": 0, "accuracy": 0.5, "balanced_accuracy": 0.5, "f1_macro": 0.5, "precision_macro": 0.5, "recall_macro": 0.5, "roc_auc_ovr": 0.6},
            pd.Series([2.0, 1.0], index=fold.valid.index, dtype="float64"),
        ),
    )

    prepared, frame = trainer._prepare_direction_dataset("BTCUSDT", "15m", labeled.copy(), trainer.paths.config.training.gbm, trainer.paths.config.labels)

    assert prepared["walk_forward_summary"]["fold_count"] == 1
    assert pd.isna(frame.loc[0, "direction_oos_pred"])
    assert pd.isna(frame.loc[1, "direction_oos_pred"])
    assert frame.loc[2, "direction_oos_pred"] == 2.0
    assert frame.loc[3, "direction_oos_pred"] == 1.0


def test_train_meta_model_uses_oos_direction_predictions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_make_paths(tmp_path))

    captured: dict[str, object] = {}

    labeled = pd.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "future_return": [0.02, -0.01, 0.03, -0.02],
            "thr_dyn": [0.001, 0.001, 0.001, 0.001],
            "direction_oos_pred": [2.0, 0.0, 2.0, 0.0],
        }
    )

    def fake_build_meta_label_from_direction(df, *, direction_pred_col, trading_cfg, **kwargs):
        captured["direction_pred_col"] = direction_pred_col
        captured["trading_cfg"] = trading_cfg
        return pd.Series([1.0, 0.0, 1.0, 0.0], index=df.index, dtype="float64")

    monkeypatch.setattr("training.gbm_training_engine.build_meta_label_from_direction", fake_build_meta_label_from_direction)

    prepared_meta = labeled.copy()
    prepared_meta["meta_label"] = [1.0, 0.0, 1.0, 0.0]
    split = TrainValidSplit(train=prepared_meta.iloc[:2].copy(), valid=prepared_meta.iloc[2:].copy())
    monkeypatch.setattr(
        trainer,
        "_prepare_dataset_with_fallback",
        lambda fused, interval, cfg, target_column: (
            {
                "frame": prepared_meta.copy(),
                "feature_columns": ["feat_a"],
                "rows_total": len(prepared_meta),
                "feature_set": {"version": "feature_store_v2", "signature": "sig2", "columns": ["feat_a"], "column_count": 1},
                "walk_forward_splits": [
                    WalkForwardFold(
                        fold_index=0,
                        train=prepared_meta.iloc[:2].copy(),
                        valid=prepared_meta.iloc[2:].copy(),
                        train_start_ts=1.0,
                        train_end_ts=2.0,
                        valid_start_ts=3.0,
                        valid_end_ts=4.0,
                    )
                ],
                "split": split,
            },
            "full_fused",
        ),
    )
    monkeypatch.setattr(
        trainer,
        "_evaluate_meta_fold",
        lambda fold, feature_columns, gbm_cfg, interval, random_state: {"fold_index": 0, "accuracy": 0.5, "balanced_accuracy": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5, "roc_auc": 0.6},
    )
    monkeypatch.setattr(trainer, "fit_calibrated_classifier", lambda *args, **kwargs: _StubModel())
    monkeypatch.setattr(trainer, "_build_model", lambda *args, **kwargs: object())

    bundle, metrics = trainer._train_meta_model("BTCUSDT", "15m", labeled, trainer.paths.config.training.gbm, 42, "cand123")

    assert bundle is not None
    assert metrics is not None
    assert metrics["skipped"] is False
    assert captured["direction_pred_col"] == "direction_oos_pred"
    assert captured["trading_cfg"] is trainer.paths.config.trading
