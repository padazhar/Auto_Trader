from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from inference.model_artifact_loader import LoadedHMMArtifact
from training.hmm_regime_training_engine import build_hmm_observation_frame


@dataclass(slots=True)
class HMMRegimePrediction:
    available: bool
    state: int | None
    confidence: float
    posteriors: dict[str, float] = field(default_factory=dict)
    shadow_only: bool = True
    decision_influence_allowed: bool = False
    mandatory_gate_enabled: bool = False
    live_influence_enabled: bool = False
    reason: str = "unavailable"
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "state": self.state,
            "confidence": float(self.confidence),
            "posteriors": dict(self.posteriors),
            "shadow_only": bool(self.shadow_only),
            "decision_influence_allowed": bool(self.decision_influence_allowed),
            "mandatory_gate_enabled": bool(self.mandatory_gate_enabled),
            "live_influence_enabled": bool(self.live_influence_enabled),
            "reason": self.reason,
            "metrics": dict(self.metrics or {}),
            "artifact": dict(self.artifact or {}),
        }


def _manifest_summary(artifact: LoadedHMMArtifact | None) -> dict[str, Any]:
    manifest = dict(getattr(artifact, "manifest", {}) or {}) if artifact else {}
    lifecycle = manifest.get("lifecycle") if isinstance(manifest.get("lifecycle"), dict) else {}
    hmm_contract = manifest.get("hmm_contract") if isinstance(manifest.get("hmm_contract"), dict) else {}
    feature_set = manifest.get("feature_set") if isinstance(manifest.get("feature_set"), dict) else {}
    return {
        "model_type": manifest.get("model_type"),
        "model_role": manifest.get("model_role"),
        "candidate_id": lifecycle.get("candidate_id"),
        "stage": lifecycle.get("stage"),
        "schema_version": hmm_contract.get("schema_version") or manifest.get("schema_version"),
        "feature_set_signature": feature_set.get("signature"),
        "shadow_only": hmm_contract.get("shadow_only", True),
        "decision_influence_allowed": hmm_contract.get("decision_influence_allowed", False),
        "governance_role": "shadow_candidate",
        "artifact_mandatory_paper_gate_enabled": bool(hmm_contract.get("mandatory_paper_gate_enabled", False)),
        "artifact_live_influence_enabled": bool(hmm_contract.get("live_influence_enabled", False)),
    }


def predict_hmm_regime(
    artifact: LoadedHMMArtifact | None,
    feature_frame: pd.DataFrame | pd.Series,
    *,
    anchor_interval: str,
    min_confidence: float = 0.60,
) -> HMMRegimePrediction:
    """Return HMM shadow telemetry without decision influence."""
    if artifact is None:
        return HMMRegimePrediction(False, None, 0.0, reason="hmm_artifact_missing")
    manifest = dict(artifact.manifest or {})
    hmm_contract = manifest.get("hmm_contract") if isinstance(manifest.get("hmm_contract"), dict) else {}
    if bool(hmm_contract.get("mandatory_paper_gate_enabled", False)) or bool(hmm_contract.get("live_influence_enabled", False)):
        return HMMRegimePrediction(False, None, 0.0, reason="hmm_artifact_governance_forbidden", artifact=_manifest_summary(artifact))
    try:
        if isinstance(feature_frame, pd.Series):
            frame = feature_frame.to_frame().T
        else:
            frame = feature_frame.tail(1).copy()
        observations, _schema = build_hmm_observation_frame(
            frame,
            anchor_interval=str(anchor_interval),
            schema_version=str(hmm_contract.get("schema_version") or "hmm_observation_v1"),
        )
        if observations.empty:
            # Fallback for single-row inference: align directly to recorded observation columns.
            row = frame.tail(1).copy()
            for column in artifact.feature_columns:
                if column not in row.columns:
                    row[column] = 0.0
            observations = row.loc[:, artifact.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        else:
            for column in artifact.feature_columns:
                if column not in observations.columns:
                    observations[column] = 0.0
            observations = observations.loc[:, artifact.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = observations.tail(1).to_numpy(dtype=float)
        if X.size == 0:
            return HMMRegimePrediction(False, None, 0.0, reason="hmm_observation_empty", artifact=_manifest_summary(artifact))
        if hasattr(artifact.model, "predict_proba"):
            posterior = np.asarray(artifact.model.predict_proba(X), dtype=float)[0]
        elif hasattr(artifact.model, "score_samples"):
            _score, posterior_matrix = artifact.model.score_samples(X)
            posterior = np.asarray(posterior_matrix, dtype=float)[0]
        else:
            return HMMRegimePrediction(False, None, 0.0, reason="hmm_posterior_unavailable", artifact=_manifest_summary(artifact))
        if posterior.ndim != 1 or posterior.size <= 0:
            return HMMRegimePrediction(False, None, 0.0, reason="hmm_posterior_unavailable", artifact=_manifest_summary(artifact))
        state = int(np.argmax(posterior))
        confidence = float(posterior[state])
        posteriors = {str(idx): float(value) for idx, value in enumerate(posterior.tolist())}
        reason = "hmm_shadow_ready" if confidence >= float(min_confidence) else "hmm_shadow_low_confidence"
        return HMMRegimePrediction(
            True,
            state,
            confidence,
            posteriors=posteriors,
            reason=reason,
            metrics=dict(artifact.metrics or {}),
            artifact=_manifest_summary(artifact),
        )
    except Exception as exc:
        return HMMRegimePrediction(False, None, 0.0, reason=f"hmm_prediction_error:{exc}", artifact=_manifest_summary(artifact))
