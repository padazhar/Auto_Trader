from types import SimpleNamespace

import numpy as np
import pandas as pd

from training.base_trainer import BaseTrainer


class DummyTrainer(BaseTrainer):
    model_type = "dummy"

    def train(self, symbol: str, interval: str):  # pragma: no cover - not used by these unit tests
        raise NotImplementedError


def _trainer(tmp_path, **overrides):
    promotion = SimpleNamespace(
        min_fold_count=1,
        min_precision_macro=0.0,
        min_roc_auc_ovr=0.0,
        min_balanced_accuracy=0.0,
        min_precision=0.0,
        min_roc_auc=0.0,
        require_paper_expectancy=False,
        require_directional_quality=False,
        require_calibration_quality=True,
        max_expected_calibration_error=0.20,
        max_brier_score=0.60,
        **overrides,
    )
    paths = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(promotion=promotion)), log_dir=tmp_path)
    trainer = DummyTrainer.__new__(DummyTrainer)
    trainer.paths = paths
    trainer.logger = SimpleNamespace(exception=lambda *args, **kwargs: None)
    return trainer


def test_calibration_quality_report_measures_reliable_probabilities():
    y_true = pd.Series([0, 1, 2, 2])
    y_prob = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.10, 0.85],
            [0.10, 0.10, 0.80],
        ]
    )

    report = BaseTrainer.calibration_quality_report(y_true, y_prob, class_labels=[0, 1, 2])["calibration"]

    assert report["available"] is True
    assert report["expected_calibration_error"] < 0.20
    assert report["brier_score"] < 0.10


def test_promotion_rejects_bad_calibration(tmp_path):
    trainer = _trainer(tmp_path)
    metrics = {
        "fold_count": 1,
        "calibration": {
            "available": True,
            "expected_calibration_error": 0.45,
            "brier_score": 0.80,
        },
    }

    reasons = trainer._promotion_floor_reasons(metrics, model_role="direction", symbol="BTCUSDT")

    assert "calibration_ece_exceeded" in reasons
    assert "calibration_brier_exceeded" in reasons


def test_promotion_rejects_missing_calibration_when_required(tmp_path):
    trainer = _trainer(tmp_path)
    metrics = {"fold_count": 1}

    reasons = trainer._promotion_floor_reasons(metrics, model_role="direction", symbol="BTCUSDT")

    assert "calibration_quality_missing" in reasons
