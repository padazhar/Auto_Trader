from __future__ import annotations

from types import SimpleNamespace

from scripts.registry_readiness_verification import RegistryReadinessVerificationReport, run_registry_readiness_verification
from scripts.step9_validation_pipeline import run_step9_validation_pipeline


class _Report(SimpleNamespace):
    pass


def test_registry_readiness_verification_writes_fail_closed_report(project_root):
    report = run_registry_readiness_verification(project_root)
    payload = report.to_dict()

    assert payload["overall_status"] in {"pass", "fail"}
    assert "registry_gate_consistent" in payload
    assert "false_ready_symbols" in payload
    assert (project_root / "data" / "runtime" / "registry_readiness_verification_report.json").exists()


def test_step9_pipeline_includes_registry_readiness_verification(project_root, monkeypatch):
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_registry_readiness_verification",
        lambda root: RegistryReadinessVerificationReport(
            project_root=str(root),
            overall_status="pass",
            registry_gate_consistent=True,
            active_symbols_count=10,
            symbol_state_execution_eligible_count=0,
            registry_execution_ready_count=0,
        ),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    item = next(item for item in report.to_dict()["items"] if item["name"] == "registry_readiness_verification")

    assert item["status"] == "pass"
    assert item["details"]["registry_gate_consistent"] is True


def test_step9_pipeline_fails_when_symbol_state_is_false_ready(project_root, monkeypatch):
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_runtime_self_check",
        lambda root: _Report(overall_status="pass", runtime_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_registry_readiness_verification",
        lambda root: RegistryReadinessVerificationReport(
            project_root=str(root),
            overall_status="fail",
            registry_gate_consistent=False,
            active_symbols_count=10,
            symbol_state_execution_eligible_count=1,
            registry_execution_ready_count=0,
            false_ready_symbols=["BTCUSDT"],
        ),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_smoke_check",
        lambda root: _Report(overall_status="pass", paper_mode_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_paper_mode_soak_check",
        lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True),
    )
    monkeypatch.setattr(
        "scripts.step9_validation_pipeline.run_final_live_validation_gate",
        lambda root: _Report(overall_status="warn", controlled_live_candidate_ready=True, live_trading_ready=False),
    )

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["overall_status"] == "fail"
    item = next(item for item in payload["items"] if item["name"] == "registry_readiness_verification")
    assert item["details"]["false_ready_symbols"] == ["BTCUSDT"]


def test_registry_readiness_verification_exposes_per_model_diagnostics(project_root):
    report = run_registry_readiness_verification(project_root)
    payload = report.to_dict()

    assert "model_diagnostics" in payload
    assert "blocker_diagnostics" in payload
    for row in payload["model_diagnostics"]:
        assert "symbol" in row
        assert "model_type" in row
        assert "execution_ready" in row
        assert "blocker_reason" in row
        assert "active_manifest_exists" in row
        assert "active_manifest_valid" in row
        assert "registry_entry_exists" in row
        assert "registry_truth_status" in row
        assert "registry_execution_ready" in row
        assert "registry_promotion_approved" in row
        assert "registry_active_manifest_valid" in row
        assert "feature_compatible" in row


def test_model_readiness_payload_contains_registry_diagnostics(project_root):
    from inference.readiness import signal_readiness_snapshot
    from core.path_manager import PathManager

    paths = PathManager.from_project_root(project_root)
    snapshot = signal_readiness_snapshot(paths, symbols=["BTCUSDT"], interval="15m", required_models=("rf", "gbm"))
    row = snapshot["rows"][0]
    for model in row["models"].values():
        assert "active_manifest_exists" in model
        assert "registry_entry_exists" in model
        assert "registry_truth_status" in model
        assert "registry_execution_ready" in model
        assert "registry_promotion_approved" in model
        assert "registry_active_manifest_valid" in model


def test_registry_readiness_verification_exposes_feature_contract_diagnostics(project_root):
    report = run_registry_readiness_verification(project_root)
    payload = report.to_dict()

    assert "model_diagnostics" in payload
    for row in payload["model_diagnostics"]:
        assert "feature_contract_diagnostics" in row
        diagnostics = row["feature_contract_diagnostics"]
        assert isinstance(diagnostics, dict)
        if diagnostics:
            assert diagnostics.get("schema_version") == "feature_contract_diagnostics.v1"
            assert "manifest_path" in diagnostics
            assert "compatible" in diagnostics or diagnostics.get("manifest_readable") is False
