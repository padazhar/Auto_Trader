
from __future__ import annotations

from pathlib import Path

from scripts.live_trading_readiness_check import run_live_trading_readiness_check


def test_live_trading_readiness_check_is_fail_closed_without_explicit_live_enablement(project_root: Path, monkeypatch) -> None:
    """Verify the final live-trading readiness gate stays fail-closed by default."""
    monkeypatch.delenv("AUTO_TRADER_ENABLE_LIVE_MODE", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("AUTO_TRADER_BINANCE_API_SECRET", raising=False)

    report = run_live_trading_readiness_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "fail"
    assert payload["live_trading_ready"] is False

    names = {item["name"] for item in payload["items"]}
    assert "repo_default_safety" in names
    assert "paper_trading_gate" in names
    assert "live_guard_gate" in names
    assert "live_candidate_operational_gate" in names
    assert "deliberate_live_enablement" in names
    assert "deliberate_live_canary_enablement" in names
    assert "live_canary_restrictions" in names


def test_live_trading_readiness_check_passes_with_explicit_live_canary_enablement(project_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTO_TRADER_ENABLE_LIVE_MODE", "true")
    monkeypatch.setenv("AUTO_TRADER_ENABLE_LIVE_CANARY", "true")
    monkeypatch.setenv("AUTO_TRADER_BINANCE_API_KEY", "env-key")
    monkeypatch.setenv("AUTO_TRADER_BINANCE_API_SECRET", "env-secret")

    report = run_live_trading_readiness_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["live_trading_ready"] is True
