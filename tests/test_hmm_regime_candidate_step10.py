from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.artifact_contract import artifact_paths, load_json, validate_artifact_bundle
from core.path_manager import PathManager
from inference.hmm_regime_confirmer import predict_hmm_regime
from inference.model_artifact_loader import ModelArtifactLoader
from services.training_engine import TrainingEngine
from training.hmm_regime_training_engine import HMMRegimeTrainer, build_hmm_observation_frame, train_hmm_model


class FakeGaussianHMM:
    def __init__(self, *, n_components: int, covariance_type: str, n_iter: int, random_state: int) -> None:
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.n_iter = int(n_iter)
        self.random_state = int(random_state)
        self.monitor_ = SimpleNamespace(converged=True)

    def fit(self, X):
        self.n_features_in_ = int(np.asarray(X).shape[1])
        self.classes_ = list(range(self.n_components))
        return self

    def predict(self, X):
        rows = int(np.asarray(X).shape[0])
        return np.arange(rows) % self.n_components

    def predict_proba(self, X):
        rows = int(np.asarray(X).shape[0])
        out = np.full((rows, self.n_components), 0.05, dtype=float)
        out[:, 0] = 1.0 - 0.05 * (self.n_components - 1)
        return out

    def score(self, X):
        return -123.0


def _paths(tmp_path: Path) -> PathManager:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "system_settings.yaml").write_text(
        """
system:
  environment: test
runtime:
  execution_mode: paper
training:
  gbm:
    estimator: histgbm
  hmm:
    enabled: true
    candidate_training_enabled: true
    mandatory_paper_gate_enabled: false
    live_influence_enabled: false
    shadow_only: true
    observation_schema_version: hmm_observation_v1
    n_states: 3
    covariance_type: diag
    min_rows: 20
    tail_rows: 80
    random_state: 7
    max_iter: 25
    min_posterior_confidence: 0.60
inference:
  hmm:
    enabled: true
    candidate_shadow_enabled: true
    mandatory_paper_gate_enabled: false
    live_influence_enabled: false
    require_valid_artifact_for_entry: false
""".strip(),
        encoding="utf-8",
    )
    return PathManager.from_project_root(root)


def _fused_rows(rows: int = 80) -> pd.DataFrame:
    close = np.linspace(100.0, 110.0, rows) + np.sin(np.arange(rows) / 3.0)
    return pd.DataFrame(
        {
            "timestamp": np.arange(rows, dtype=int),
            "15m_open": close - 0.2,
            "15m_high": close + 0.8,
            "15m_low": close - 0.9,
            "15m_close": close,
            "15m_volume": np.linspace(1000.0, 1400.0, rows),
        }
    )


def test_hmm_observation_schema_uses_trailing_ohlcv_features_only() -> None:
    observations, schema = build_hmm_observation_frame(_fused_rows(40), anchor_interval="15m", schema_version="hmm_observation_v1")

    assert schema.version == "hmm_observation_v1"
    assert list(schema.columns) == list(observations.columns)
    assert "future_return" not in observations.columns
    assert "dqn_action" not in observations.columns
    assert len(observations) > 0


def test_hmm_candidate_trainer_writes_shadow_only_candidate_artifact(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("training.hmm_regime_training_engine._GaussianHMM", FakeGaussianHMM)
    monkeypatch.setattr("training.hmm_regime_training_engine.build_fused_frame", lambda *args, **kwargs: _fused_rows(80))

    result = train_hmm_model(paths, "BTCUSDT", "15m")

    assert result.ok is True
    assert result.model_type == "hmm"
    assert result.model_role == "regime_confirmer"
    assert result.metrics["shadow_only"] is True
    assert result.metrics["decision_influence_allowed"] is False
    assert result.metrics["mandatory_paper_gate_enabled"] is False
    assert result.metrics["live_influence_enabled"] is False
    manifest_path = Path(result.artifacts["manifest"])
    manifest = load_json(manifest_path) or {}
    assert manifest["model_type"] == "hmm"
    assert manifest["model_role"] == "regime_confirmer"
    assert manifest["hmm_contract"]["shadow_only"] is True
    assert manifest["hmm_contract"]["decision_influence_allowed"] is False
    assert manifest["lifecycle"]["stage"] == "candidate"
    valid, reason = validate_artifact_bundle(paths, manifest_path)
    assert (valid, reason) == (True, "fresh")


def test_hmm_candidate_loader_and_shadow_prediction_have_no_decision_effect(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("training.hmm_regime_training_engine._GaussianHMM", FakeGaussianHMM)
    monkeypatch.setattr("training.hmm_regime_training_engine.build_fused_frame", lambda *args, **kwargs: _fused_rows(80))
    result = HMMRegimeTrainer(paths).train("ETHUSDT", "15m")
    assert result.ok is True

    artifact = ModelArtifactLoader(paths).load_candidate_hmm("ETHUSDT", "15m")
    prediction = predict_hmm_regime(
        artifact,
        _fused_rows(2).iloc[-1],
        anchor_interval="15m",
        min_confidence=0.60,
    )

    assert prediction.available is True
    assert prediction.shadow_only is True
    assert prediction.decision_influence_allowed is False
    assert prediction.mandatory_gate_enabled is False
    assert prediction.live_influence_enabled is False
    assert prediction.artifact["governance_role"] == "shadow_candidate"
    assert prediction.artifact["artifact_mandatory_paper_gate_enabled"] is False
    assert prediction.artifact["artifact_live_influence_enabled"] is False
    assert prediction.state == 0
    assert prediction.confidence >= 0.60


def test_training_engine_routes_hmm_without_using_policy_slot(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    engine = TrainingEngine(paths)
    called = {}

    def fake_train(paths_arg, symbol, interval):
        called["payload"] = (paths_arg, symbol, interval)
        return SimpleNamespace(ok=True, model_type="hmm", symbol=symbol, interval=interval, rows_used=1, artifacts={}, metrics={}, error=None)

    monkeypatch.setattr("services.training_engine.train_hmm_model", fake_train)
    result = engine.train_model("hmm", "SOLUSDT", "15m")

    assert result.model_type == "hmm"
    assert called["payload"][1:] == ("SOLUSDT", "15m")
