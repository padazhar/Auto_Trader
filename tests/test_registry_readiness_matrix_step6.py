from __future__ import annotations

from types import SimpleNamespace

from core.training_registry import RegistryEntry, TrainingRegistry
from inference.readiness import signal_readiness_snapshot
from monitoring.publisher import MonitorPublisher


def test_signal_readiness_snapshot_exposes_execution_readiness_matrix(monkeypatch) -> None:
    def fake_anchor(paths, *, symbol: str, interval: str, required_models=("rf", "gbm")):
        if symbol == "BTCUSDT":
            return SimpleNamespace(
                ready=True,
                reason="anchor_ready",
                models={
                    "rf": SimpleNamespace(model_type="rf", ready=True, status="success", reason="ready", manifest_path="rf.json", deployment_stage="incumbent"),
                    "gbm": SimpleNamespace(model_type="gbm", ready=True, status="success", reason="ready", manifest_path="gbm.json", deployment_stage="incumbent"),
                },
            )
        return SimpleNamespace(
            ready=False,
            reason="anchor_training_incomplete:gbm:gbm_direction_active_manifest_missing",
            models={
                "rf": SimpleNamespace(model_type="rf", ready=True, status="success", reason="ready", manifest_path="rf.json", deployment_stage="incumbent"),
                "gbm": SimpleNamespace(model_type="gbm", ready=False, status="success", reason="gbm_direction_active_manifest_missing", manifest_path=None, deployment_stage="unknown"),
            },
        )

    monkeypatch.setattr("inference.readiness.anchor_model_readiness", fake_anchor)

    payload = signal_readiness_snapshot(SimpleNamespace(), symbols=["BTCUSDT", "ETHUSDT"], interval="15m")

    assert payload["execution_eligible_count"] == 1
    assert payload["execution_eligible_symbols"] == ["BTCUSDT"]
    assert payload["execution_readiness_matrix"] == payload["rows"]
    blocked = [row for row in payload["execution_readiness_matrix"] if row["symbol"] == "ETHUSDT"][0]
    assert blocked["execution_ready"] is False
    assert blocked["candidate_only_is_execution_ready"] is False
    assert blocked["blocking_models"] == ["gbm"]
    assert blocked["models"]["gbm"]["active_manifest_valid"] is False


def test_training_registry_uses_gbm_direction_incumbent_for_execution_truth(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_active_truth(paths, model_type, symbol, interval, *, model_role=None):
        calls.append({"model_type": model_type, "symbol": symbol, "interval": interval, "model_role": model_role})
        return {"execution_ready": True, "active_manifest_path": "active-direction.json", "reason": "ready"}

    monkeypatch.setattr("core.training_registry.active_manifest_truth", fake_active_truth)

    registry = object.__new__(TrainingRegistry)
    registry.paths = SimpleNamespace()
    result = registry._active_truth_locked(RegistryEntry(model_type="gbm", symbol="ETHUSDT", interval="15m"))

    assert result["execution_ready"] is True
    assert calls == [{"model_type": "gbm", "symbol": "ETHUSDT", "interval": "15m", "model_role": "direction"}]


def test_monitoring_trading_readiness_uses_signal_readiness_matrix_counts() -> None:
    publisher = object.__new__(MonitorPublisher)
    publisher._snapshot = {
        "market_data": {
            "symbol_universe": {
                "active_symbols": ["BTCUSDT", "ETHUSDT"],
                "active_count": 2,
                "target_active_symbols": 2,
                "execution_eligible_count": 2,
                "blocked_symbols": [],
            },
            "runtime_price_snapshot": {"ready": True},
        },
        "training_registry": {"diagnostic_counts": {}},
        "rejection_visibility": {},
        "control_plane": {},
        "signal_runtime": {
            "readiness": {
                "ready_count": 1,
                "execution_eligible_count": 1,
                "blocked_count": 1,
                "execution_readiness_matrix": [
                    {"symbol": "BTCUSDT", "execution_ready": True},
                    {"symbol": "ETHUSDT", "execution_ready": False, "blocking_models": ["gbm"]},
                ],
            }
        },
    }

    publisher._refresh_trading_readiness_locked()

    readiness = publisher._snapshot["trading_readiness"]
    assert readiness["status"] == "blocked"
    assert readiness["execution_eligible_count"] == 1
    assert "execution_eligible_symbols:1/2" in readiness["blockers"]
    assert "signal_readiness_blocked:1" in readiness["blockers"]
    assert readiness["execution_readiness_matrix"][1]["symbol"] == "ETHUSDT"


def test_monitoring_trading_readiness_does_not_hard_block_on_optional_registry_diagnostics_when_anchor_ready() -> None:
    publisher = object.__new__(MonitorPublisher)
    publisher._snapshot = {
        "market_data": {
            "symbol_universe": {
                "active_symbols": ["BTCUSDT", "ETHUSDT"],
                "active_count": 2,
                "target_active_symbols": 2,
                "execution_eligible_count": 2,
                "blocked_symbols": [],
            },
            "runtime_price_snapshot": {"ready": True},
        },
        "training_registry": {"diagnostic_counts": {"missing_active_manifest": 14, "candidate_only": 16}},
        "rejection_visibility": {},
        "control_plane": {},
        "signal_runtime": {
            "readiness": {
                "ready_count": 2,
                "execution_eligible_count": 2,
                "blocked_count": 0,
                "execution_readiness_matrix": [
                    {"symbol": "BTCUSDT", "execution_ready": True},
                    {"symbol": "ETHUSDT", "execution_ready": True},
                ],
            }
        },
    }

    publisher._refresh_trading_readiness_locked()

    readiness = publisher._snapshot["trading_readiness"]
    assert readiness["status"] == "ready"
    assert readiness["ready"] is True
    assert not any(str(blocker).startswith("registry_missing_active_manifest") for blocker in readiness["blockers"])


def test_monitoring_trading_readiness_still_blocks_invalid_registry_diagnostics_when_anchor_ready() -> None:
    publisher = object.__new__(MonitorPublisher)
    publisher._snapshot = {
        "market_data": {
            "symbol_universe": {
                "active_symbols": ["BTCUSDT"],
                "active_count": 1,
                "target_active_symbols": 1,
                "execution_eligible_count": 1,
                "blocked_symbols": [],
            },
            "runtime_price_snapshot": {"ready": True},
        },
        "training_registry": {"diagnostic_counts": {"active_manifest_invalid": 1, "feature_incompatible": 1}},
        "rejection_visibility": {},
        "control_plane": {},
        "signal_runtime": {
            "readiness": {
                "ready_count": 1,
                "execution_eligible_count": 1,
                "blocked_count": 0,
                "execution_readiness_matrix": [{"symbol": "BTCUSDT", "execution_ready": True}],
            }
        },
    }

    publisher._refresh_trading_readiness_locked()

    readiness = publisher._snapshot["trading_readiness"]
    assert readiness["status"] == "blocked"
    assert "registry_active_manifest_invalid:1" in readiness["blockers"]
    assert "registry_feature_incompatible:1" in readiness["blockers"]


def test_registry_diagnostic_execution_ready_uses_registry_truth_not_manifest_only(monkeypatch) -> None:
    """A valid active manifest without registry approval must remain not execution-ready."""
    registry = object.__new__(TrainingRegistry)
    registry.paths = SimpleNamespace()
    entry = RegistryEntry(model_type="rf", symbol="BTCUSDT", interval="15m", status="success")
    entry.active_manifest_valid = True
    entry.promotion_approved = False
    entry.execution_ready = False
    entry.registry_truth_status = "active_without_registry_approval"

    monkeypatch.setattr("core.training_registry.resolve_active_manifest_path", lambda *args, **kwargs: SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda *args, **kwargs: (True, "fresh"))

    diagnostic = registry._entry_diagnostic_locked(entry)

    assert diagnostic["diagnostic_category"] == "active_without_registry_approval"
    assert diagnostic["active_manifest_valid"] is True
    assert diagnostic["execution_ready"] is False


def test_registry_gate_requires_registry_truth_execution_ready() -> None:
    from inference.readiness import _registry_execution_gate_reason

    entry = {
        "status": "success",
        "registry_truth_status": "active_without_registry_approval",
        "execution_ready": True,
        "promotion_approved": True,
        "active_manifest_valid": True,
    }

    reason = _registry_execution_gate_reason({"rf:BTCUSDT:15m": entry}, entry, model_type="rf")

    assert reason == "registry_not_execution_ready:active_without_registry_approval"
