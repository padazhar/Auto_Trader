from __future__ import annotations

from types import SimpleNamespace

from inference.readiness import signal_readiness_snapshot


def test_signal_readiness_snapshot_explains_rf_gbm_blockers(monkeypatch) -> None:
    def fake_anchor(paths, *, symbol: str, interval: str, required_models=("rf", "gbm")):
        if symbol == "BTCUSDT":
            return SimpleNamespace(
                ready=True,
                reason="anchor_ready",
                models={
                    "rf": SimpleNamespace(model_type="rf", ready=True, status="success", reason="ready"),
                    "gbm": SimpleNamespace(model_type="gbm", ready=True, status="success", reason="ready"),
                },
            )
        return SimpleNamespace(
            ready=False,
            reason="anchor_training_incomplete:rf:registry_status:idle,gbm:active_manifest_missing",
            models={
                "rf": SimpleNamespace(model_type="rf", ready=False, status="idle", reason="registry_status:idle"),
                "gbm": SimpleNamespace(model_type="gbm", ready=False, status="success", reason="active_manifest_missing"),
            },
        )

    monkeypatch.setattr("inference.readiness.anchor_model_readiness", fake_anchor)

    payload = signal_readiness_snapshot(SimpleNamespace(), symbols=["BTCUSDT", "ETHUSDT"], interval="15m")

    assert payload["total_symbols"] == 2
    assert payload["ready_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["ready_symbols"] == ["BTCUSDT"]
    assert payload["blocked_symbols"] == ["ETHUSDT"]
    assert payload["model_blockers"] == {"rf": 1, "gbm": 1}
    eth_row = [row for row in payload["rows"] if row["symbol"] == "ETHUSDT"][0]
    assert eth_row["ready"] is False
    assert eth_row["models"]["rf"]["reason"] == "registry_status:idle"
    assert eth_row["models"]["gbm"]["reason"] == "active_manifest_missing"


def test_model_readiness_payload_includes_feature_contract_diagnostics(monkeypatch) -> None:
    from inference.readiness import AnchorReadiness, ModelReadiness, anchor_readiness_to_payload

    readiness = AnchorReadiness(
        ready=False,
        reason="anchor_training_incomplete:gbm:feature_incompatible:feature_set_missing",
        models={
            "gbm": ModelReadiness(
                model_type="gbm",
                ready=False,
                status="success",
                reason="feature_incompatible:feature_set_missing",
                feature_contract_diagnostics={
                    "schema_version": "feature_contract_diagnostics.v1",
                    "manifest_path": "data/model_storage/example/manifest.json",
                    "compatible": False,
                    "compatibility_reasons": ["feature_set_missing"],
                },
            )
        },
    )

    payload = anchor_readiness_to_payload("BTCUSDT", "15m", readiness)
    diagnostics = payload["models"]["gbm"]["feature_contract_diagnostics"]
    assert diagnostics["schema_version"] == "feature_contract_diagnostics.v1"
    assert diagnostics["compatibility_reasons"] == ["feature_set_missing"]


def test_signal_readiness_default_anchor_contract_remains_rf_gbm_only(monkeypatch) -> None:
    """Step 6 baseline guard: RF and GBM remain the required execution anchors.

    Future LightGBM/HMM stages must be added through their explicit gates; the
    corrected baseline readiness snapshot must not silently require DQN/HMM or
    any additional model family before those future steps are approved.
    """
    observed_required: list[tuple[str, ...]] = []

    def fake_anchor(paths, *, symbol: str, interval: str, required_models=("rf", "gbm")):
        observed_required.append(tuple(required_models))
        return SimpleNamespace(
            ready=True,
            reason="anchor_ready",
            models={
                "rf": SimpleNamespace(model_type="rf", ready=True, status="success", reason="ready"),
                "gbm": SimpleNamespace(model_type="gbm", ready=True, status="success", reason="ready"),
            },
        )

    monkeypatch.setattr("inference.readiness.anchor_model_readiness", fake_anchor)

    payload = signal_readiness_snapshot(SimpleNamespace(), symbols=["BTCUSDT"], interval="15m")

    assert observed_required == [("rf", "gbm")]
    assert payload["required_models"] == ["rf", "gbm"]
    assert payload["model_blockers"] == {"rf": 0, "gbm": 0}
    row = payload["rows"][0]
    assert row["required_active_models"] == ["rf", "gbm"]
    assert sorted(row["models"].keys()) == ["gbm", "rf"]
