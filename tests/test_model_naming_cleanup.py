from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import incumbent_pointer_path, model_base_dir
from core.job_definition import TrainingJob
from core.model_naming import (
    canonical_model_type,
    is_policy_model,
    model_display_name,
    model_short_label,
    model_storage_directory,
)
from inference.model_artifact_loader import LoadedDQNArtifact, LoadedPolicyArtifact, ModelArtifactLoader
from inference.prediction_engines import predict_dqn, predict_policy
from services.signal_engine import EngineDecision
from services.training_engine import TrainingEngine
from training.dqn_training_engine import train_dqn_model, train_policy_model
from training.hmm_regime_training_engine import train_hmm_model


class _StubArtifact:
    def __init__(self) -> None:
        self.feature_columns = ["feat"]
        self.metrics = {"quality": "ok"}
        self.model = self

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.1, 0.9]], dtype=float)


def test_policy_aliases_normalize_to_dqn() -> None:
    assert canonical_model_type("policy") == "dqn"
    assert canonical_model_type("policy_confirmer") == "dqn"
    assert canonical_model_type("confirmatory_policy_classifier") == "dqn"
    assert model_display_name("dqn") == "Confirmatory Policy"
    assert model_short_label("dqn") == "P"
    assert model_storage_directory("policy") == "reinforcement_agents"
    assert is_policy_model("policy") is True


def test_training_job_canonicalizes_policy_alias() -> None:
    job = TrainingJob(model_type="policy", symbol="BTCUSDT", interval="15m", priority=1, requested_at=0.0, reason="test")
    assert job.model_type == "dqn"


def test_artifact_paths_accept_policy_alias(tmp_path: Path) -> None:
    paths = SimpleNamespace(model_storage_dir=tmp_path / "model_storage", project_root=tmp_path)
    base = model_base_dir(paths, "policy", "BTCUSDT", "15m")
    assert str(base).endswith("reinforcement_agents/BTCUSDT/15m")
    pointer = incumbent_pointer_path(paths, "policy", "BTCUSDT", "15m")
    assert str(pointer).endswith("reinforcement_agents/BTCUSDT/15m/incumbent.json")


def test_artifact_paths_accept_hmm_regime_type(tmp_path: Path) -> None:
    paths = SimpleNamespace(model_storage_dir=tmp_path / "model_storage", project_root=tmp_path)
    base = model_base_dir(paths, "hmm_regime_confirmer", "BTCUSDT", "15m")
    assert str(base).endswith("hmm_regime_models/BTCUSDT/15m")
    pointer = incumbent_pointer_path(paths, "hmm", "BTCUSDT", "15m")
    assert str(pointer).endswith("hmm_regime_models/BTCUSDT/15m/incumbent.json")


def test_training_engine_exposes_policy_display_name() -> None:
    engine = TrainingEngine(SimpleNamespace(config=SimpleNamespace(frames=SimpleNamespace(anchor_interval="15m"))))
    assert engine.model_display_name("dqn") == "Confirmatory Policy"
    assert engine.model_display_name("hmm") == "HMM Regime Confirmer"


def test_policy_prediction_aliases_match() -> None:
    artifact = _StubArtifact()
    import pandas as pd

    row = pd.Series({"feat": 1.0})
    policy_pred = predict_policy(artifact, row)
    dqn_pred = predict_dqn(artifact, row)
    assert policy_pred == dqn_pred
    assert policy_pred.available is True
    assert policy_pred.action == 1


def test_signal_engine_pretty_line_accepts_legacy_dqn_key() -> None:
    decision = EngineDecision(
        symbol="BTCUSDT",
        interval="15m",
        action="ENTER_LONG",
        score=0.12,
        reason="test_reason",
        mode="Medium",
        components={
            "dqn": {"used": True, "action": "long", "confidence": 0.8, "score": 1.0},
        },
    )
    line = decision.pretty_line()
    assert "Policy action=long" in line


def test_loader_exposes_policy_methods_and_legacy_aliases() -> None:
    assert LoadedDQNArtifact is LoadedPolicyArtifact
    assert hasattr(ModelArtifactLoader, "load_policy")
    assert hasattr(ModelArtifactLoader, "load_candidate_policy")
    assert hasattr(ModelArtifactLoader, "load_dqn")
    assert hasattr(ModelArtifactLoader, "load_candidate_dqn")


def test_policy_training_entrypoint_preserves_legacy_alias() -> None:
    assert train_dqn_model.__name__ == "train_dqn_model"
    assert train_policy_model.__name__ == "train_policy_model"


def test_hmm_training_entrypoint_is_separate_from_legacy_policy_slot() -> None:
    assert train_hmm_model.__name__ == "train_hmm_model"
    assert is_policy_model("hmm") is False
