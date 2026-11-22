from __future__ import annotations

from pathlib import Path

import yaml

from core.config_contract import load_app_config
from core.path_manager import PathManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_yaml_contains_governance_critical_sections(project_root: Path) -> None:
    """Verify the repository YAML explicitly declares governance-critical config blocks."""
    config_path = project_root / "config" / "system_settings.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    assert "feature_store" in payload
    assert "regime" in payload
    assert "training" in payload
    assert "validation" in payload["training"]
    assert "promotion" in payload["training"]

    assert payload["feature_store"]["recipe_version"] == "feature_store_v2"
    assert payload["runtime"]["execution_mode"] in {"paper", "live"}


def test_load_app_config_reflects_explicit_yaml_values(project_root: Path) -> None:
    """Verify explicit YAML values are reflected into the runtime config object."""
    paths = PathManager(project_root=project_root, settings={}, config=None)
    config = load_app_config(paths)

    assert config.feature_store.recipe_version == "feature_store_v2"
    assert config.feature_store.include_optional_futures_context is True

    assert config.regime.enabled is True
    assert config.regime.crisis_realized_vol_z == 3.0
    assert config.regime.high_realized_vol_z == 1.75

    assert config.training.validation.walk_forward_folds == 4
    assert config.training.validation.purge_rows == 2
    assert config.training.validation.embargo_rows == 1

    assert config.training.promotion.enabled is True
    assert config.training.promotion.allow_initial_promotion is True
    assert config.training.promotion.allow_initial_promotion_without_paper_trades is False
    assert config.training.promotion.allow_initial_paper_bootstrap_execution is True
    assert config.training.promotion.require_outperform_incumbent is True
    assert config.training.promotion.min_fold_count == 3

    assert config.runtime.execution_mode == "paper"
    assert config.runtime.allow_unreconciled_live_mode is False


def test_repository_yaml_is_paper_safe_by_default(project_root: Path) -> None:
    """The committed repository config must remain safe to start without exchange credentials."""
    config_path = project_root / "config" / "system_settings.yaml"
    raw_text = config_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text) or {}

    assert payload["runtime"]["execution_mode"] == "paper"
    assert payload["runtime"]["allow_unreconciled_live_mode"] is False
    assert payload["training"]["promotion"]["allow_initial_promotion_without_paper_trades"] is False
    assert payload["training"]["promotion"]["allow_initial_paper_bootstrap_execution"] is True
    assert payload["exchange"]["api_key"] == ""
    assert payload["exchange"]["api_secret"] == ""
    assert payload["exchange"]["use_testnet"] is True
    assert payload["exchange"]["position_reconciliation_enabled"] is True
    assert payload["exchange"]["require_reconciliation_for_live"] is True
    assert "#  execution_mode: live" not in raw_text
