from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from features.label_builder import HOLD_CLASS, LONG_CLASS, SHORT_CLASS
from inference.model_artifact_loader import LoadedGBMArtifact, LoadedGBMRoleArtifact, LoadedPolicyArtifact, LoadedRFArtifact


@dataclass(slots=True)
class RFPrediction:
    available: bool
    label: int
    short_probability: float
    hold_probability: float
    long_probability: float
    direction_score: float
    metrics: dict[str, Any]


@dataclass(slots=True)
class GBMPrediction:
    available: bool
    label: int
    short_probability: float
    hold_probability: float
    long_probability: float
    direction_score: float
    meta_probability: float
    effective_score: float
    metrics: dict[str, Any]
    meta_available: bool


@dataclass(slots=True)
class PolicyPrediction:
    available: bool
    action: int
    confidence: float
    score: float
    metrics: dict[str, Any]


@dataclass(slots=True)
class QualityGateResult:
    passed: bool
    reasons: list[str]


_POLICY_ACTION_SCORE_MAP = {0: 0.0, 1: 1.0, 2: -1.0}
# Backward-compatible constant alias for older callers.
_DQN_ACTION_SCORE_MAP = _POLICY_ACTION_SCORE_MAP
_PRECISION_TOLERANCE = 0.05
_AUC_TOLERANCE = 0.05
_BALANCED_ACCURACY_TOLERANCE = 0.03


def _prepare_feature_row(feature_frame: pd.DataFrame | pd.Series, feature_columns: list[str]) -> pd.DataFrame:
    """Construct a single-row numeric feature frame aligned to an artifact contract."""
    if isinstance(feature_frame, pd.Series):
        row = feature_frame.to_frame().T
    else:
        row = feature_frame.tail(1).copy()
    missing = [c for c in feature_columns if c not in row.columns]
    for col in missing:
        row[col] = 0.0
    X = row.loc[:, feature_columns].copy()
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _safe_predict(model: Any, X: pd.DataFrame) -> tuple[bool, np.ndarray | None]:
    """Safely invoke model.predict without allowing caller-facing exceptions."""
    try:
        return True, model.predict(X)
    except Exception:
        return False, None


def _safe_predict_proba(model: Any, X: pd.DataFrame) -> tuple[bool, np.ndarray | None]:
    """Safely invoke model.predict_proba when the estimator supports it."""
    if not hasattr(model, "predict_proba"):
        return False, None
    try:
        return True, model.predict_proba(X)
    except Exception:
        return False, None


def _extract_class_probabilities(model: Any, X: pd.DataFrame) -> tuple[float, float, float]:
    """Extract normalized short/hold/long probabilities from a multiclass estimator."""
    short_probability = 0.0
    hold_probability = 1.0
    long_probability = 0.0
    ok, proba = _safe_predict_proba(model, X)
    if not ok or proba is None:
        return short_probability, hold_probability, long_probability
    row = proba[0]
    classes = [int(c) for c in getattr(model, "classes_", [])]
    mapping = {cls: float(row[idx]) for idx, cls in enumerate(classes) if idx < len(row)}
    short_probability = mapping.get(SHORT_CLASS, 0.0)
    hold_probability = mapping.get(HOLD_CLASS, 0.0)
    long_probability = mapping.get(LONG_CLASS, 0.0)
    total = short_probability + hold_probability + long_probability
    if total <= 0:
        return 0.0, 1.0, 0.0
    return short_probability / total, hold_probability / total, long_probability / total


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    """Safely coerce a metrics payload value to float."""
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def quality_gate_from_metrics(
    metrics: dict[str, Any] | None,
    *,
    min_rows: int,
    min_precision_macro: float,
    min_roc_auc_ovr: float,
    min_balanced_accuracy: float = 0.0,
) -> QualityGateResult:
    """Apply a stricter inference-time quality gate using fold stability when present."""
    metrics = dict(metrics or {})
    reasons: list[str] = []

    rows_total = int(metrics.get("rows_total") or metrics.get("rows_used") or 0)
    if rows_total < int(min_rows):
        reasons.append(f"rows<{min_rows}")

    precision_value = _metric_float(metrics, "precision_macro")
    if precision_value is None:
        precision_value = _metric_float(metrics, "precision")
    if precision_value is None or precision_value < float(min_precision_macro):
        reasons.append(f"precision<{min_precision_macro}")

    auc_value = _metric_float(metrics, "roc_auc_ovr")
    if auc_value is None:
        auc_value = _metric_float(metrics, "roc_auc")
    if auc_value is None or auc_value < float(min_roc_auc_ovr):
        reasons.append(f"roc_auc<{min_roc_auc_ovr}")

    bal_acc = _metric_float(metrics, "balanced_accuracy")
    if bal_acc is None or bal_acc < float(min_balanced_accuracy):
        reasons.append(f"balanced_accuracy<{min_balanced_accuracy}")

    fold_count = int(metrics.get("fold_count") or 0)
    has_walk_forward = fold_count > 0 or any(str(key).startswith("walk_forward_") for key in metrics.keys())
    if has_walk_forward:
        if fold_count < 2:
            reasons.append("walk_forward_fold_count<2")

        wf_precision_median = _metric_float(metrics, "walk_forward_precision_macro_median")
        if wf_precision_median is None:
            wf_precision_median = _metric_float(metrics, "walk_forward_precision_median")
        if wf_precision_median is None or wf_precision_median < float(min_precision_macro):
            reasons.append(f"walk_forward_precision_median<{min_precision_macro}")

        wf_auc_median = _metric_float(metrics, "walk_forward_roc_auc_ovr_median")
        if wf_auc_median is None:
            wf_auc_median = _metric_float(metrics, "walk_forward_roc_auc_median")
        if wf_auc_median is None or wf_auc_median < float(min_roc_auc_ovr):
            reasons.append(f"walk_forward_roc_auc_median<{min_roc_auc_ovr}")

        wf_bal_acc_median = _metric_float(metrics, "walk_forward_balanced_accuracy_median")
        if wf_bal_acc_median is None or wf_bal_acc_median < float(min_balanced_accuracy):
            reasons.append(f"walk_forward_balanced_accuracy_median<{min_balanced_accuracy}")

        wf_precision_min = _metric_float(metrics, "walk_forward_precision_macro_min")
        if wf_precision_min is None:
            wf_precision_min = _metric_float(metrics, "walk_forward_precision_min")
        min_precision_floor = max(0.0, float(min_precision_macro) - _PRECISION_TOLERANCE)
        if wf_precision_min is None or wf_precision_min < min_precision_floor:
            reasons.append(f"walk_forward_precision_min<{min_precision_floor}")

        wf_auc_min = _metric_float(metrics, "walk_forward_roc_auc_ovr_min")
        if wf_auc_min is None:
            wf_auc_min = _metric_float(metrics, "walk_forward_roc_auc_min")
        min_auc_floor = max(0.0, float(min_roc_auc_ovr) - _AUC_TOLERANCE)
        if wf_auc_min is None or wf_auc_min < min_auc_floor:
            reasons.append(f"walk_forward_roc_auc_min<{min_auc_floor}")

        wf_bal_acc_min = _metric_float(metrics, "walk_forward_balanced_accuracy_min")
        min_bal_acc_floor = max(0.0, float(min_balanced_accuracy) - _BALANCED_ACCURACY_TOLERANCE)
        if wf_bal_acc_min is None or wf_bal_acc_min < min_bal_acc_floor:
            reasons.append(f"walk_forward_balanced_accuracy_min<{min_bal_acc_floor}")

    return QualityGateResult(passed=not reasons, reasons=reasons)


def predict_rf(artifact: LoadedRFArtifact | None, feature_frame: pd.DataFrame | pd.Series) -> RFPrediction:
    """Run inference for a loaded random-forest artifact."""
    if artifact is None:
        return RFPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, {})
    try:
        X = _prepare_feature_row(feature_frame, artifact.feature_columns)
    except Exception:
        return RFPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, dict(artifact.metrics or {}))
    ok_pred, pred = _safe_predict(artifact.model, X)
    if not ok_pred or pred is None:
        return RFPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, dict(artifact.metrics or {}))
    label = int(pred[0])
    short_p, hold_p, long_p = _extract_class_probabilities(artifact.model, X)
    return RFPrediction(True, label, short_p, hold_p, long_p, float(long_p - short_p), dict(artifact.metrics or {}))


def predict_gbm(artifact: LoadedGBMArtifact | None, feature_frame: pd.DataFrame | pd.Series) -> GBMPrediction:
    """Run inference for a loaded GBM bundle, including optional meta filtering."""
    if artifact is None or artifact.direction is None:
        return GBMPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, {}, False)
    try:
        X = _prepare_feature_row(feature_frame, artifact.direction.feature_columns)
    except Exception:
        return GBMPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, dict(artifact.direction.metrics or {}), artifact.meta is not None)
    ok_pred, pred = _safe_predict(artifact.direction.model, X)
    if not ok_pred or pred is None:
        return GBMPrediction(False, HOLD_CLASS, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, dict(artifact.direction.metrics or {}), artifact.meta is not None)
    label = int(pred[0])
    short_p, hold_p, long_p = _extract_class_probabilities(artifact.direction.model, X)
    direction_score = float(long_p - short_p)

    meta_probability = 1.0
    meta_metrics: dict[str, Any] = {}
    meta_available = artifact.meta is not None
    if artifact.meta is not None:
        try:
            X_meta = _prepare_feature_row(feature_frame, artifact.meta.feature_columns)
            ok_proba, proba = _safe_predict_proba(artifact.meta.model, X_meta)
            if ok_proba and proba is not None:
                classes = [int(c) for c in getattr(artifact.meta.model, "classes_", [])]
                row = proba[0]
                class_map = {cls: float(row[idx]) for idx, cls in enumerate(classes) if idx < len(row)}
                meta_probability = float(class_map.get(1, 0.0))
            else:
                ok_meta_pred, meta_pred = _safe_predict(artifact.meta.model, X_meta)
                if ok_meta_pred and meta_pred is not None:
                    meta_probability = 1.0 if int(meta_pred[0]) == 1 else 0.0
                else:
                    meta_probability = 0.0
            meta_metrics = dict(artifact.meta.metrics or {})
        except Exception:
            meta_probability = 0.0
    metrics = dict(artifact.direction.metrics or {})
    if meta_metrics:
        metrics["meta"] = meta_metrics
    effective_score = float(direction_score * meta_probability)
    return GBMPrediction(True, label, short_p, hold_p, long_p, direction_score, meta_probability, effective_score, metrics, meta_available)


def predict_policy(artifact: LoadedPolicyArtifact | None, feature_frame: pd.DataFrame | pd.Series) -> PolicyPrediction:
    """Run inference for the confirmatory policy classifier compatibility slot."""
    if artifact is None:
        return PolicyPrediction(False, 0, 0.0, 0.0, {})
    try:
        X = _prepare_feature_row(feature_frame, artifact.feature_columns)
    except Exception:
        return PolicyPrediction(False, 0, 0.0, 0.0, dict(artifact.metrics or {}))
    ok_pred, pred = _safe_predict(artifact.model, X)
    if not ok_pred or pred is None:
        return PolicyPrediction(False, 0, 0.0, 0.0, dict(artifact.metrics or {}))
    action = int(pred[0])
    confidence = 0.0
    ok_proba, proba = _safe_predict_proba(artifact.model, X)
    if ok_proba and proba is not None:
        confidence = float(np.max(proba[0]))
    return PolicyPrediction(True, action, confidence, float(_POLICY_ACTION_SCORE_MAP.get(action, 0.0)), dict(artifact.metrics or {}))


PolicyCompatibilityPrediction = PolicyPrediction
DQNPrediction = PolicyPrediction


def predict_dqn(artifact: LoadedPolicyArtifact | None, feature_frame: pd.DataFrame | pd.Series) -> PolicyPrediction:
    """Backward-compatible alias for the legacy dqn compatibility slot."""
    return predict_policy(artifact, feature_frame)
