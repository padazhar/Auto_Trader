from __future__ import annotations

from pathlib import Path

from scripts.paper_trading_readiness_check import run_paper_trading_readiness_check


class _StubReport:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return dict(self._payload)


def test_paper_trading_readiness_check_aggregates_success(monkeypatch, project_root: Path) -> None:
    """Verify the paper-trading readiness gate passes when all underlying validations pass."""
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_readiness_check",
        lambda root: _StubReport({"overall_status": "pass", "paper_trading_ready": True}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_paper_mode_smoke_check",
        lambda root: _StubReport({"overall_status": "pass", "paper_mode_ready": True}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_paper_mode_soak_check",
        lambda root: _StubReport({"overall_status": "pass", "paper_mode_soak_ready": True}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_runtime_self_check",
        lambda root: _StubReport({"overall_status": "pass", "runtime_ready": True}),
    )

    report = run_paper_trading_readiness_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "pass"
    assert payload["paper_trading_ready"] is True

    names = {item["name"] for item in payload["items"]}
    assert names == {"repository_readiness", "paper_smoke", "paper_soak", "runtime_self_check"}


def test_paper_trading_readiness_check_fails_when_any_gate_fails(monkeypatch, project_root: Path) -> None:
    """Verify the paper-trading readiness gate fails closed when any underlying validation fails."""
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_readiness_check",
        lambda root: _StubReport({"overall_status": "pass", "paper_trading_ready": True}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_paper_mode_smoke_check",
        lambda root: _StubReport({"overall_status": "fail", "paper_mode_ready": False}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_paper_mode_soak_check",
        lambda root: _StubReport({"overall_status": "pass", "paper_mode_soak_ready": True}),
    )
    monkeypatch.setattr(
        "scripts.paper_trading_readiness_check.run_runtime_self_check",
        lambda root: _StubReport({"overall_status": "pass", "runtime_ready": True}),
    )

    report = run_paper_trading_readiness_check(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] == "fail"
    assert payload["paper_trading_ready"] is False
