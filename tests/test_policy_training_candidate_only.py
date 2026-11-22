from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from training.dqn_training_engine import PolicyConfirmTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


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


class _ValidationCfg(SimpleNamespace):
    walk_forward_folds: int = 4
    purge_rows: int = 0
    embargo_rows: int = 0


class _DqnCfg(SimpleNamespace):
    enabled: bool = True
    min_rows: int = 10
    max_age_minutes: int = 120
    tail_rows: int = 200
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 10
    max_depth: int = 3
    interval_overrides: dict = {}


class _Paths(SimpleNamespace):
    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        return self.runtime_dir / f"progress_{model_type}_{symbol}_{interval}.json"


def _paths(tmp_path: Path) -> _Paths:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    for path in (model_storage_dir, runtime_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42),
        labels=SimpleNamespace(dqn_horizon_bars=1, dqn_hold_threshold=0.0015),
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        training=SimpleNamespace(
            dqn=_DqnCfg(),
            validation=_ValidationCfg(),
            promotion=_PromotionCfg(),
        ),
    )
    return _Paths(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        config=config,
    )


def test_policy_training_succeeds_as_candidate_only_when_fold_count_is_insufficient(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())

    paths = _paths(tmp_path)
    trainer = PolicyConfirmTrainer(paths)

    fused = pd.DataFrame(
        {
            "timestamp": list(range(60)),
            "feat_a": [float(i % 5) for i in range(60)],
            "feat_b": [float((i * 2) % 7) for i in range(60)],
        }
    )

    def _load_fused_frame(symbol: str, interval: str, tail_rows: int):
        return fused.copy()

    def _add_dqn_labels(df: pd.DataFrame, anchor_interval: str, horizon: int, hold_threshold: float) -> pd.DataFrame:
        out = df.copy()
        out["dqn_action"] = [0 if i % 3 == 0 else 1 if i % 3 == 1 else 2 for i in range(len(out))]
        return out

    from features.dataset_preparation import WalkForwardFold, TrainValidSplit

    def _walk_forward(train_df, **kwargs):
        return [
            WalkForwardFold(
                fold_index=0,
                train=train_df.iloc[:30].copy(),
                valid=train_df.iloc[30:45].copy(),
                train_start_ts=float(train_df.iloc[0]["timestamp"]),
                train_end_ts=float(train_df.iloc[29]["timestamp"]),
                valid_start_ts=float(train_df.iloc[30]["timestamp"]),
                valid_end_ts=float(train_df.iloc[44]["timestamp"]),
            )
        ]

    def _train_valid(train_df, **kwargs):
        return TrainValidSplit(train=train_df.iloc[:45].copy(), valid=train_df.iloc[45:].copy())

    monkeypatch.setattr(trainer, "load_fused_frame", _load_fused_frame)
    monkeypatch.setattr("training.dqn_training_engine.add_dqn_labels", _add_dqn_labels)
    monkeypatch.setattr("training.dqn_training_engine.build_walk_forward_splits", _walk_forward)
    monkeypatch.setattr("training.dqn_training_engine.build_train_valid_split", _train_valid)
    monkeypatch.setattr(trainer, "fit_calibrated_classifier", lambda model, X_train, y_train, X_valid, y_valid: model.fit(X_train, y_train) or model)

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert result.metrics["promotion_eligible"] is False
    assert result.metrics["candidate_only_reasons"] == ["insufficient_fold_count:1"]
    assert result.metrics["promotion"]["attempted"] is False
    assert result.metrics["promotion"]["reasons"] == ["promotion_not_requested"]
    assert "manifest" in result.artifacts
