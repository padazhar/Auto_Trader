from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config_contract import ConfigError, load_app_config, validate_settings
from core.path_manager import PathManager


def test_shutdown_policy_yaml_is_paper_safe_by_default(project_root: Path) -> None:
    payload = yaml.safe_load((project_root / "config" / "system_settings.yaml").read_text(encoding="utf-8")) or {}

    assert payload["shutdown"]["enabled"] is True
    assert payload["shutdown"]["report_on_exit"] is True
    assert payload["shutdown"]["paper_position_policy"] == "snapshot_only"
    assert payload["shutdown"]["live_position_policy"] == "fail_closed_snapshot_only"
    assert payload["shutdown"]["live_force_close_requires_reconciliation"] is True


def test_load_app_config_exposes_shutdown_policy(project_root: Path) -> None:
    paths = PathManager(project_root=project_root, settings={}, config=None)
    config = load_app_config(paths)

    assert config.shutdown.enabled is True
    assert config.shutdown.report_on_exit is True
    assert config.shutdown.paper_position_policy == "snapshot_only"
    assert config.shutdown.live_position_policy == "fail_closed_snapshot_only"
    assert config.shutdown.live_force_close_requires_reconciliation is True
    assert config.shutdown.coordinator_join_timeout_sec == 15.0


def test_shutdown_policy_rejects_unreconciled_live_force_close() -> None:
    settings = {
        "system": {"timezone": "UTC", "random_seed": 42, "environment": "dev"},
        "market": {"futures_base_url": "https://fapi.binance.com", "use_binance_http_updater": True},
        "frames": {"anchor_interval": "15m", "fusion_intervals": ["1m"], "lookback_rows": {"1m": 10}},
        "labels": {"class_encoding": {"short": 0, "hold": 1, "long": 2}},
        "training": {"rf": {}, "gbm": {}, "dqn": {}},
        "scheduling": {},
        "inference": {"quality": {}, "consensus": {}, "weights": {}, "dqn": {}, "ev": {}},
        "runtime": {"execution_mode": "paper"},
        "monitoring": {"auth_token": "token"},
        "shutdown": {
            "enabled": True,
            "report_on_exit": True,
            "paper_position_policy": "snapshot_only",
            "live_position_policy": "force_close_without_reconciliation",
            "live_force_close_requires_reconciliation": False,
        },
    }

    with pytest.raises(ConfigError, match="shutdown.live_position_policy"):
        validate_settings(settings)
