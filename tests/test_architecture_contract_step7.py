from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config_contract import ConfigError, load_app_config, validate_settings
from core.model_naming import canonical_model_type, is_policy_model, model_display_name, model_storage_directory
from core.path_manager import PathManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _repository_config():
    return load_app_config(PathManager(project_root=PROJECT_ROOT, settings={}, config=None))


def test_step12_architecture_contract_enables_hmm_paper_gate_only() -> None:
    """Step 12 may enable mandatory HMM paper-entry gating, but not live influence."""
    config = _repository_config()

    assert config.training.gbm.estimator == "histgbm"
    assert config.training.hmm.enabled is True
    assert config.training.hmm.candidate_training_enabled is True
    assert config.training.hmm.mandatory_paper_gate_enabled is False
    assert config.training.hmm.live_influence_enabled is False
    assert config.training.hmm.shadow_only is True
    assert config.training.hmm.observation_schema_version == "hmm_observation_v1"

    assert config.inference.hmm.enabled is True
    assert config.inference.hmm.candidate_shadow_enabled is True
    assert config.inference.hmm.mandatory_paper_gate_enabled is True
    assert config.inference.hmm.live_influence_enabled is False
    assert config.inference.hmm.require_valid_artifact_for_entry is True

    assert config.inference.weights.rf == 0.50
    assert config.inference.weights.gbm == 0.40
    assert config.inference.weights.dqn == 0.00
    assert config.inference.ev.dqn_adjustment_strength == 0.0
    assert config.runtime.execution_mode == "paper"
    assert config.runtime.allow_unreconciled_live_mode is False


def test_step12_yaml_declares_hmm_paper_gate_and_histgbm_rollback_contracts() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "config" / "system_settings.yaml").read_text(encoding="utf-8"))

    assert payload["training"]["gbm"]["estimator"] == "histgbm"
    assert payload["training"]["hmm"]["enabled"] is True
    assert payload["training"]["hmm"]["candidate_training_enabled"] is True
    assert payload["training"]["hmm"]["mandatory_paper_gate_enabled"] is False
    assert payload["training"]["hmm"]["live_influence_enabled"] is False
    assert payload["training"]["hmm"]["shadow_only"] is True

    assert payload["inference"]["hmm"]["enabled"] is True
    assert payload["inference"]["hmm"]["candidate_shadow_enabled"] is True
    assert payload["inference"]["hmm"]["mandatory_paper_gate_enabled"] is True
    assert payload["inference"]["hmm"]["live_influence_enabled"] is False
    assert payload["inference"]["hmm"]["require_valid_artifact_for_entry"] is True


def test_step7_config_rejects_unsupported_gbm_estimator() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "config" / "system_settings.yaml").read_text(encoding="utf-8"))
    payload["training"]["gbm"]["estimator"] = "catboost"

    with pytest.raises(ConfigError, match="training.gbm.estimator"):
        validate_settings(payload)


def test_step7_hmm_model_naming_contract_preserves_legacy_policy_slot() -> None:
    assert canonical_model_type("lightgbm") == "gbm"
    assert canonical_model_type("lgbm") == "gbm"
    assert canonical_model_type("hmm_regime_confirmer") == "hmm"
    assert canonical_model_type("regime_hmm") == "hmm"

    assert model_display_name("hmm") == "HMM Regime Confirmer"
    assert model_storage_directory("hmm") == "hmm_regime_models"
    assert is_policy_model("hmm") is False

    assert canonical_model_type("policy_confirmer") == "dqn"
    assert model_storage_directory("policy") == "reinforcement_agents"
    assert is_policy_model("policy") is True


def test_step7_requirements_declare_candidate_dependencies_without_importing_them() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "lightgbm" in requirements
    assert "hmmlearn" in requirements
