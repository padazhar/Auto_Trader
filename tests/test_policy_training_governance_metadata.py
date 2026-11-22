from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from training.dqn_training_engine import PolicyConfirmTrainer


class _DummyPaths(SimpleNamespace):
    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _paths(tmp_path: Path) -> _DummyPaths:
    project_root = tmp_path / "repo"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return _DummyPaths(
        project_root=project_root,
        logs_dir=logs_dir,
        config=SimpleNamespace(
            system=SimpleNamespace(random_seed=42),
            feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
            labels=SimpleNamespace(dqn_horizon_bars=1, dqn_hold_threshold=0.0015),
            training=SimpleNamespace(
                dqn=SimpleNamespace(
                    tail_rows=1000,
                    min_rows=20,
                    test_size=0.2,
                    random_state=42,
                    n_estimators=25,
                    max_depth=5,
                    interval_overrides={},
                ),
                validation=SimpleNamespace(
                    walk_forward_folds=4,
                    purge_rows=0,
                    embargo_rows=0,
                ),
                promotion=SimpleNamespace(min_fold_count=3),
            ),
        ),
    )


class _PolicyTrainerUnderTest(PolicyConfirmTrainer):
    def __init__(self, paths, fused: pd.DataFrame) -> None:
        super().__init__(paths)
        self._fused = fused

    def load_fused_frame(self, symbol: str, interval: str, tail_rows: int) -> pd.DataFrame:
        return self._fused.copy()

    def save_training_bundle(
        self,
        *,
        symbol: str,
        interval: str,
        rows_used: int,
        feature_columns: list[str],
        artifacts: dict,
        metrics: dict,
        candidate_id: str,
        model_role: str,
        promote: bool,
        extra_manifest: dict,
    ) -> dict:
        self._captured_metrics = dict(metrics)
        self._captured_manifest = dict(extra_manifest)
        return {
            "manifest": "dummy_manifest.json",
            "promoted": False,
            "promotion": {"attempted": False, "approved": False, "reasons": ["promotion_not_requested"]},
        }


def _make_frame(rows: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": ts.astype("int64") // 10**9,
            "close": [100 + (i * 0.01) for i in range(rows)],
            "open": [100 + (i * 0.01) for i in range(rows)],
            "high": [100.5 + (i * 0.01) for i in range(rows)],
            "low": [99.5 + (i * 0.01) for i in range(rows)],
            "volume": [1000 + i for i in range(rows)],
            "feat_a": [float(i % 7) for i in range(rows)],
            "feat_b": [float((i * 2) % 5) for i in range(rows)],
        }
    )


def test_policy_trainer_emits_candidate_only_governance_metadata(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    trainer = _PolicyTrainerUnderTest(paths, _make_frame())
    result = trainer.train("BTCUSDT", "15m")

    assert result.ok is True
    metrics = trainer._captured_metrics
    manifest = trainer._captured_manifest

    assert metrics["promotion_eligible"] is False
    assert "insufficient_fold_count:1" in metrics["candidate_only_reasons"]
    assert metrics["approval_tier"] == "candidate_only_optional"
    assert metrics["live_blocking"] is False
    assert metrics["live_eligible"] is False
    assert metrics["model_operational_role"] == "confirmatory_optional"
    assert metrics["retrain_min_new_bars"] == 3

    governance = manifest["governance"]
    assert governance["promotion_eligible"] is False
    assert "insufficient_fold_count:1" in governance["candidate_only_reasons"]
    assert governance["approval_tier"] == "candidate_only_optional"
    assert governance["live_blocking"] is False
    assert governance["live_eligible"] is False
    assert governance["retrain_min_new_bars"] == 3
