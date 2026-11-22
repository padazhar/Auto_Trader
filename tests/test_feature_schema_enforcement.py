from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from core.artifact_contract import artifact_paths
from training.base_trainer import BaseTrainer
from training.dqn_training_engine import PolicyConfirmTrainer


class _DummyPaths(SimpleNamespace):
    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        return self.runtime_dir / f"progress_{model_type}_{symbol}_{interval}.json"


class _DummyTrainer(BaseTrainer):
    model_type = "rf"

    def train(self, symbol: str, interval: str):
        raise NotImplementedError


def _paths(tmp_path: Path) -> _DummyPaths:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    for path in (model_storage_dir, runtime_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return _DummyPaths(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        config=SimpleNamespace(feature_store=SimpleNamespace(recipe_version="feature_store_v2")),
    )


def test_save_training_bundle_rejects_incompatible_feature_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    trainer = _DummyTrainer(paths)
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id="cand-1")
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"dummy")

    with pytest.raises(ValueError, match="incompatible_feature_set_metadata"):
        trainer.save_training_bundle(
            symbol="BTCUSDT",
            interval="15m",
            rows_used=10,
            feature_columns=["feat_a", "feat_b"],
            artifacts=artifacts,
            metrics={"precision_macro": 0.2},
            candidate_id="cand-1",
            model_role="direction",
            promote=False,
            extra_manifest={
                "feature_set": {
                    "schema_version": 1,
                    "recipe_version": "feature_store_v1",
                    "version": "feature_store_v1",
                    "signature": "deadbeef",
                    "column_count": 2,
                    "columns": ["feat_a", "feat_b"],
                    "anchor_interval": "15m",
                    "optional_context_fields": [],
                }
            },
        )


def test_policy_trainer_accepts_raw_ohlcv_aliases_for_anchor_label_building(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    paths.config = SimpleNamespace(
        system=SimpleNamespace(random_seed=42),
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        labels=SimpleNamespace(dqn_horizon_bars=1, dqn_hold_threshold=0.0015),
        training=SimpleNamespace(
            dqn=SimpleNamespace(
                tail_rows=1000,
                min_rows=20,
                test_size=0.2,
                random_state=42,
                n_estimators=10,
                max_depth=4,
                interval_overrides={},
            ),
            validation=SimpleNamespace(walk_forward_folds=3, purge_rows=0, embargo_rows=0),
            promotion=SimpleNamespace(min_fold_count=3),
        ),
    )
    trainer = PolicyConfirmTrainer(paths)
    fused = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=80, freq="15min").astype("int64") // 10**9,
            "open": [100 + i * 0.01 for i in range(80)],
            "high": [100.5 + i * 0.01 for i in range(80)],
            "low": [99.5 + i * 0.01 for i in range(80)],
            "close": [100 + i * 0.01 for i in range(80)],
            "volume": [1000 + i for i in range(80)],
            "feat_a": [float(i % 7) for i in range(80)],
            "feat_b": [float((i * 2) % 5) for i in range(80)],
        }
    )
    monkeypatch.setattr(trainer, "load_fused_frame", lambda symbol, interval, tail_rows: fused.copy())

    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    assert result.metrics["model_operational_role"] == "confirmatory_optional"
