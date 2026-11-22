from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import artifact_paths, incumbent_pointer_path, save_json, write_gbm_combined_manifest
from features.feature_store_contract import build_feature_store_metadata, resolve_feature_recipe_version
from inference.readiness import anchor_model_readiness, model_registry_readiness
from runners.tech_rev_controller import TechRevController


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=tmp_path,
        model_storage_dir=tmp_path / "data" / "model_storage",
        registry_path=tmp_path / "data" / "runtime" / "training_registry.json",
        config=SimpleNamespace(
            runtime=SimpleNamespace(execution_mode="paper"),
            frames=SimpleNamespace(anchor_interval="15m"),
            training=SimpleNamespace(
                rf=SimpleNamespace(enabled=True),
                gbm=SimpleNamespace(enabled=True),
                dqn=SimpleNamespace(enabled=True, allowed_intervals=["15m"]),
            ),
        ),
    )


def _write_standard_manifest(paths: SimpleNamespace, model_type: str, symbol: str, interval: str, *, role: str | None = None, stage: str = "incumbent") -> Path:
    artifacts = artifact_paths(paths, model_type, symbol, interval, model_role=role, stage="incumbent")
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_text("model", encoding="utf-8")
    save_json(artifacts["feature_columns"], {"feature_columns": ["close"]})
    save_json(artifacts["metrics"], {"balanced_accuracy": 0.6})
    feature_set = build_feature_store_metadata(
        ["close"],
        recipe_version=resolve_feature_recipe_version(paths.config),
        anchor_interval=interval,
    )
    payload = {
        "schema_version": 4,
        "model_type": model_type,
        "model_role": role,
        "symbol": symbol,
        "interval": interval,
        "artifacts": {
            "model": artifacts["model"].relative_to(paths.project_root).as_posix(),
            "feature_columns": artifacts["feature_columns"].relative_to(paths.project_root).as_posix(),
            "metrics": artifacts["metrics"].relative_to(paths.project_root).as_posix(),
        },
        "feature_set": feature_set,
        "lifecycle": {
            "stage": stage,
            "paper_execution_only": stage == "paper_bootstrap",
            "live_eligible": stage != "paper_bootstrap",
            "candidate_id": "candidate-active",
            "promoted_at": "2026-05-01T00:00:00+00:00",
        },
        "timestamps": {"created_at": "2026-05-01T00:00:00+00:00"},
    }
    save_json(artifacts["manifest"], payload)
    pointer = incumbent_pointer_path(paths, model_type, symbol, interval, model_role=role)
    save_json(
        pointer,
        {
            "schema_version": 1,
            "model_type": model_type,
            "model_role": role,
            "symbol": symbol,
            "interval": interval,
            "manifest_path": artifacts["manifest"].relative_to(paths.project_root).as_posix(),
        },
    )
    return artifacts["manifest"]



def _write_registry_ready(paths: SimpleNamespace, entries: list[tuple[str, str, str]]) -> None:
    paths.registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for model_type, symbol, interval in entries:
        payload[f"{model_type}:{symbol}:{interval}"] = {
            "model_type": model_type,
            "symbol": symbol,
            "interval": interval,
            "status": "success",
            "promotion_attempted": True,
            "promotion_approved": True,
            "candidate_saved": True,
            "candidate_manifest_valid": True,
            "active_manifest_valid": True,
            "execution_ready": True,
            "registry_truth_status": "execution_ready",
        }
    save_json(paths.registry_path, payload)

def test_anchor_readiness_requires_active_rf_and_gbm_direction_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_standard_manifest(paths, "rf", "BTCUSDT", "15m")
    _write_registry_ready(paths, [("rf", "BTCUSDT", "15m")])

    gbm = model_registry_readiness(paths, model_type="gbm", symbol="BTCUSDT", interval="15m")
    assert gbm.ready is False
    assert gbm.reason == "gbm_direction_active_manifest_missing"

    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert "gbm:gbm_direction_active_manifest_missing" in readiness.reason

    _write_standard_manifest(paths, "gbm", "BTCUSDT", "15m", role="direction")
    _write_registry_ready(paths, [("rf", "BTCUSDT", "15m"), ("gbm", "BTCUSDT", "15m")])
    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")
    assert readiness.ready is True
    assert readiness.reason == "anchor_ready"


def test_gbm_bundle_readiness_requires_direction_bound_feature_contract(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    direction_manifest = _write_standard_manifest(paths, "gbm", "BTCUSDT", "15m", role="direction")
    _write_registry_ready(paths, [("gbm", "BTCUSDT", "15m")])
    bundle = write_gbm_combined_manifest(
        paths,
        symbol="BTCUSDT",
        interval="15m",
        direction_manifest=direction_manifest,
        meta_manifest=None,
        rows_used=100,
        meta_skipped=True,
        meta_skip_reason="direction_only_paper_fallback",
        stage="incumbent",
    )

    readiness = model_registry_readiness(paths, model_type="gbm", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is True
    assert readiness.reason == "ready"

    bundle["feature_set"]["signature"] = "mismatched-bundle-feature-contract"
    bundle_path = paths.model_storage_dir / "gradient_boost_models" / "BTCUSDT" / "15m" / "manifest.json"
    save_json(bundle_path, bundle)

    readiness = model_registry_readiness(paths, model_type="gbm", symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert readiness.reason == "gbm_bundle:direction_feature_set_mismatch"


def test_techrev_execution_readiness_checks_anchor_rf_gbm_only(monkeypatch) -> None:
    cfg = SimpleNamespace(
        frames=SimpleNamespace(anchor_interval="15m"),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=True),
            gbm=SimpleNamespace(enabled=True),
            dqn=SimpleNamespace(enabled=True, allowed_intervals=["15m"]),
        ),
    )
    controller = object.__new__(TechRevController)
    controller.paths = SimpleNamespace(config=cfg)

    calls: list[tuple[str, str]] = []

    def fake_readiness(paths, *, model_type: str, symbol: str, interval: str):
        calls.append((model_type, interval))
        return SimpleNamespace(ready=True, reason="ready")

    monkeypatch.setattr("runners.tech_rev_controller.model_registry_readiness", fake_readiness)

    ready, reason = controller._symbol_execution_readiness("BTCUSDT")
    assert ready is True
    assert reason == "anchor_ready"
    assert calls == [("rf", "15m"), ("gbm", "15m")]


def test_anchor_readiness_blocks_active_manifest_without_registry_approval(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_standard_manifest(paths, "rf", "BTCUSDT", "15m")

    rf = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")
    assert rf.ready is False
    assert rf.reason == "registry_unavailable"


def test_paper_bootstrap_anchor_is_execution_ready_only_in_paper_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_standard_manifest(paths, "rf", "BTCUSDT", "15m", stage="paper_bootstrap")
    _write_standard_manifest(paths, "gbm", "BTCUSDT", "15m", role="direction", stage="paper_bootstrap")
    _write_registry_ready(paths, [("rf", "BTCUSDT", "15m"), ("gbm", "BTCUSDT", "15m")])

    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")
    assert readiness.ready is True
    assert readiness.models["rf"].paper_execution_only is True
    assert readiness.models["gbm"].paper_execution_only is True
    assert readiness.models["rf"].reason == "paper_bootstrap_ready"

    paths.config.runtime.execution_mode = "live"
    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert "paper_bootstrap_forbidden_outside_paper_mode" in readiness.reason


def _write_stale_registry(paths: SimpleNamespace, entries: list[tuple[str, str, str]]) -> None:
    paths.registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for model_type, symbol, interval in entries:
        payload[f"{model_type}:{symbol}:{interval}"] = {
            "model_type": model_type,
            "symbol": symbol,
            "interval": interval,
            "status": "success",
            "promotion_attempted": True,
            "promotion_approved": False,
            "candidate_saved": True,
            "candidate_manifest_valid": True,
            "active_manifest_valid": False,
            "execution_ready": False,
            "registry_truth_status": "promotion_rejected",
            "promotion_reasons": ["below_min_paper_expectancy"],
        }
    save_json(paths.registry_path, payload)


def _prove_paper_bootstrap_pointer(paths: SimpleNamespace, model_type: str, symbol: str, interval: str, *, role: str | None = None) -> None:
    manifest_path = artifact_paths(paths, model_type, symbol, interval, model_role=role, stage="incumbent")["manifest"]
    save_json(
        incumbent_pointer_path(paths, model_type, symbol, interval, model_role=role),
        {
            "schema_version": 1,
            "model_type": model_type,
            "model_role": role,
            "symbol": symbol,
            "interval": interval,
            "candidate_id": "candidate-active",
            "manifest_path": manifest_path.relative_to(paths.project_root).as_posix(),
            "promoted_at": "2026-05-01T00:00:00+00:00",
            "deployment_stage": "paper_bootstrap",
            "paper_execution_only": True,
            "paper_evidence_status": "paper_bootstrap_execution_ready",
            "live_eligible": False,
        },
    )


def test_pointer_proven_paper_bootstrap_recovers_from_stale_rejected_registry_in_paper_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_standard_manifest(paths, "rf", "BTCUSDT", "15m", stage="paper_bootstrap")
    _write_standard_manifest(paths, "gbm", "BTCUSDT", "15m", role="direction", stage="paper_bootstrap")
    _prove_paper_bootstrap_pointer(paths, "rf", "BTCUSDT", "15m")
    _prove_paper_bootstrap_pointer(paths, "gbm", "BTCUSDT", "15m", role="direction")
    _write_stale_registry(paths, [("rf", "BTCUSDT", "15m"), ("gbm", "BTCUSDT", "15m")])

    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")

    assert readiness.ready is True
    assert readiness.models["rf"].reason == "paper_bootstrap_ready"
    assert readiness.models["rf"].registry_reconciled_from_active_authority is True
    assert readiness.models["rf"].active_authority_source == "active_paper_bootstrap_pointer"
    assert readiness.models["gbm"].registry_reconciled_from_active_authority is True

    paths.config.runtime.execution_mode = "live"
    readiness = anchor_model_readiness(paths, symbol="BTCUSDT", interval="15m")
    assert readiness.ready is False
    assert "paper_bootstrap_forbidden_outside_paper_mode" in readiness.reason


def test_stale_registry_is_not_bypassed_when_paper_bootstrap_pointer_provenance_mismatches(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_standard_manifest(paths, "rf", "BTCUSDT", "15m", stage="paper_bootstrap")
    _prove_paper_bootstrap_pointer(paths, "rf", "BTCUSDT", "15m")
    pointer_path = incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m")
    save_json(
        pointer_path,
        {
            "schema_version": 1,
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": "different-candidate",
            "manifest_path": artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="incumbent")["manifest"].relative_to(paths.project_root).as_posix(),
            "promoted_at": "2026-05-01T00:00:00+00:00",
            "deployment_stage": "paper_bootstrap",
            "paper_execution_only": True,
            "live_eligible": False,
        },
    )
    _write_stale_registry(paths, [("rf", "BTCUSDT", "15m")])

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")

    assert readiness.ready is False
    assert readiness.reason == "registry_not_execution_ready:promotion_rejected"
