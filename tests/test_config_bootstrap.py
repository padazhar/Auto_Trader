from __future__ import annotations

from pathlib import Path

from core.config_contract import AppConfig, load_app_config
from core.path_manager import PathManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_app_config_includes_feature_store_and_regime() -> None:
    """Load the repository config and verify all required AppConfig sections exist."""
    paths = PathManager(project_root=PROJECT_ROOT, settings={}, config=None)

    config = load_app_config(paths)

    assert isinstance(config, AppConfig)
    assert config.feature_store.recipe_version
    assert isinstance(config.feature_store.include_optional_futures_context, bool)
    assert isinstance(config.regime.enabled, bool)
    assert config.regime.crisis_realized_vol_z > 0
    assert config.training.promotion.allow_initial_promotion_without_paper_trades is False
    assert config.training.promotion.allow_initial_paper_bootstrap_execution is True


def test_path_manager_bootstraps_repository_config() -> None:
    """Build a real PathManager from the repository root without config constructor failure."""
    paths = PathManager.from_project_root(PROJECT_ROOT)

    assert isinstance(paths.config, AppConfig)
    assert paths.config.feature_store.recipe_version
    assert isinstance(paths.config.regime.enabled, bool)
    assert paths.runtime_dir.exists()
