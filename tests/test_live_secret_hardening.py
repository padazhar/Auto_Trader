from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config_contract import ConfigError, SecretValidationError, validate_runtime_secrets, validate_settings


def _load_repo_yaml(project_root: Path) -> dict:
    path = project_root / "config" / "system_settings.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_validate_settings_rejects_inline_exchange_api_key(project_root: Path) -> None:
    payload = _load_repo_yaml(project_root)
    payload.setdefault("exchange", {})["api_key"] = "inline-secret"

    with pytest.raises(ConfigError) as exc_info:
        validate_settings(payload)

    assert "exchange.api_key" in str(exc_info.value)
    assert "AUTO_TRADER_BINANCE_API_KEY" in str(exc_info.value)


def test_validate_settings_rejects_inline_exchange_api_secret(project_root: Path) -> None:
    payload = _load_repo_yaml(project_root)
    payload.setdefault("exchange", {})["api_secret"] = "inline-secret"

    with pytest.raises(ConfigError) as exc_info:
        validate_settings(payload)

    assert "exchange.api_secret" in str(exc_info.value)
    assert "AUTO_TRADER_BINANCE_API_SECRET" in str(exc_info.value)


def test_validate_runtime_secrets_requires_env_credentials_for_live(monkeypatch, project_root: Path) -> None:
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_SECRET", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    payload = _load_repo_yaml(project_root)
    payload["runtime"]["execution_mode"] = "live"
    payload.setdefault("exchange", {})["position_reconciliation_enabled"] = True
    payload["exchange"]["api_key"] = ""
    payload["exchange"]["api_secret"] = ""
    config = validate_settings(payload)

    with pytest.raises(SecretValidationError) as exc_info:
        validate_runtime_secrets(config)

    assert "AUTO_TRADER_BINANCE_API_KEY" in str(exc_info.value)
    assert "AUTO_TRADER_BINANCE_API_SECRET" in str(exc_info.value)


def test_validate_runtime_secrets_allows_paper_without_exchange_credentials(monkeypatch, project_root: Path) -> None:
    """Paper startup must not require Binance credentials."""
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_SECRET", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)

    payload = _load_repo_yaml(project_root)
    payload["runtime"]["execution_mode"] = "paper"
    payload.setdefault("exchange", {})["api_key"] = ""
    payload["exchange"]["api_secret"] = ""
    config = validate_settings(payload)

    validate_runtime_secrets(config)
