from __future__ import annotations

import numpy as np
import pandas as pd

from inference.model_artifact_loader import LoadedGBMArtifact, LoadedGBMRoleArtifact, LoadedRFArtifact
from inference.prediction_engines import predict_gbm, predict_rf, quality_gate_from_metrics


class FakeMultiClassModel:
    classes_ = np.array([0, 1, 2])

    def predict(self, X):
        return np.array([2])

    def predict_proba(self, X):
        return np.array([[0.1, 0.2, 0.7]])


class FakeBinaryModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.25, 0.75]])


class FakeMetaModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.array([1])

    def predict_proba(self, X):
        return np.array([[0.1, 0.9]])


def test_rf_prediction_outputs_three_probabilities():
    artifact = LoadedRFArtifact(model=FakeMultiClassModel(), feature_columns=["a"], metrics={}, manifest={})
    pred = predict_rf(artifact, pd.Series({"a": 1.0}))
    assert pred.available is True
    assert pred.short_probability == 0.1
    assert pred.hold_probability == 0.2
    assert pred.long_probability == 0.7
    assert round(pred.direction_score, 4) == 0.6


def test_gbm_effective_score_uses_meta_probability():
    direction = LoadedGBMRoleArtifact(model=FakeMultiClassModel(), feature_columns=["a"], metrics={}, manifest={})
    meta = LoadedGBMRoleArtifact(model=FakeMetaModel(), feature_columns=["a"], metrics={}, manifest={})
    artifact = LoadedGBMArtifact(direction=direction, meta=meta)
    pred = predict_gbm(artifact, pd.Series({"a": 1.0}))
    assert pred.available is True
    assert round(pred.direction_score, 4) == 0.6
    assert round(pred.meta_probability, 4) == 0.9
    assert round(pred.effective_score, 4) == 0.54


def test_quality_gate_uses_walk_forward_stability_metrics():
    metrics = {
        "rows_total": 500,
        "precision_macro": 0.30,
        "roc_auc_ovr": 0.60,
        "balanced_accuracy": 0.42,
        "fold_count": 3,
        "walk_forward_precision_macro_median": 0.28,
        "walk_forward_roc_auc_ovr_median": 0.58,
        "walk_forward_balanced_accuracy_median": 0.40,
        "walk_forward_precision_macro_min": 0.27,
        "walk_forward_roc_auc_ovr_min": 0.56,
        "walk_forward_balanced_accuracy_min": 0.38,
    }
    gate = quality_gate_from_metrics(
        metrics,
        min_rows=100,
        min_precision_macro=0.20,
        min_roc_auc_ovr=0.55,
        min_balanced_accuracy=0.36,
    )
    assert gate.passed is True
    assert gate.reasons == []


def test_quality_gate_rejects_unstable_walk_forward_metrics():
    metrics = {
        "rows_total": 500,
        "precision_macro": 0.30,
        "roc_auc_ovr": 0.60,
        "balanced_accuracy": 0.42,
        "fold_count": 3,
        "walk_forward_precision_macro_median": 0.28,
        "walk_forward_roc_auc_ovr_median": 0.58,
        "walk_forward_balanced_accuracy_median": 0.40,
        "walk_forward_precision_macro_min": 0.10,
        "walk_forward_roc_auc_ovr_min": 0.40,
        "walk_forward_balanced_accuracy_min": 0.20,
    }
    gate = quality_gate_from_metrics(
        metrics,
        min_rows=100,
        min_precision_macro=0.20,
        min_roc_auc_ovr=0.55,
        min_balanced_accuracy=0.36,
    )
    assert gate.passed is False
    assert any("walk_forward_precision_min" in reason for reason in gate.reasons)
    assert any("walk_forward_roc_auc_min" in reason for reason in gate.reasons)
    assert any("walk_forward_balanced_accuracy_min" in reason for reason in gate.reasons)
