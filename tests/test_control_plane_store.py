from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import incumbent_pointer_path, save_json
from monitoring.control_plane_store import ControlPlaneStore


def _paths(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    runtime_dir = project_root / "data" / "runtime"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=project_root,
        runtime_dir=runtime_dir,
        model_storage_dir=model_storage_dir,
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        symbol_state_path=runtime_dir / "symbol_state.json",
    )


def _write_manifest(path: Path, *, model_type: str, symbol: str, interval: str, candidate_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = path.parent / "metrics.json"
    save_json(metrics_path, {"precision_macro": 0.25, "roc_auc_ovr": 0.61})
    save_json(
        path,
        {
            "schema_version": 4,
            "model_type": model_type,
            "symbol": symbol,
            "interval": interval,
            "metrics_path": str(metrics_path.relative_to(path.parents[5])),
            "feature_set": {"version": "feature_store_v2", "signature": "abc123"},
            "lifecycle": {"stage": "candidate", "candidate_id": candidate_id, "promoted_at": None},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": "2026-04-03T00:00:00+00:00"},
            "artifacts": {"model": "model.joblib", "feature_columns": "feature_columns.json", "metrics": str(metrics_path.relative_to(path.parents[5]))},
        },
    )


def test_control_plane_store_tracks_pause_and_actions(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = ControlPlaneStore(paths)

    summary = store.set_pause(paused=True, pause_reason="maintenance", updated_by="tester")

    assert summary["paused"] is True
    assert summary["pause_reason"] == "maintenance"
    assert summary["recent_actions"][0]["action"] == "pause"


def test_control_plane_store_can_activate_candidate_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = ControlPlaneStore(paths)

    current_manifest = paths.model_storage_dir / "random_forest_models" / "BTCUSDT" / "15m" / "candidates" / "old" / "manifest.json"
    rollback_manifest = paths.model_storage_dir / "random_forest_models" / "BTCUSDT" / "15m" / "candidates" / "new" / "manifest.json"
    _write_manifest(current_manifest, model_type="rf", symbol="BTCUSDT", interval="15m", candidate_id="old")
    _write_manifest(rollback_manifest, model_type="rf", symbol="BTCUSDT", interval="15m", candidate_id="new")

    save_json(
        incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"),
        {
            "schema_version": 1,
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": "old",
            "manifest_path": str(current_manifest.relative_to(paths.project_root)),
            "promoted_at": "2026-04-03T00:00:00+00:00",
            "metrics_summary": {"precision_macro": 0.20},
        },
    )

    result = store.rollback_model_version(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        manifest_path=str(rollback_manifest.relative_to(paths.project_root)),
        updated_by="tester",
    )

    pointer = incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m")
    payload = __import__("json").loads(pointer.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert payload["candidate_id"] == "new"
    assert payload["manifest_path"].endswith("new/manifest.json")
    assert store.recent_actions(limit=1)[0]["action"] == "model_rollback"


def test_control_plane_summary_exposes_runtime_state_owners(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    store = ControlPlaneStore(paths)

    summary = store.summary()

    assert summary["runtime_state_owners"]["market_data"] == "market_data_controller"
    assert summary["runtime_state_owners"]["paper_execution_runtime"] == "live_trading_controller"
    assert summary["runtime_state_owners"]["control_plane"] == "control_plane_store"


def test_step5_decision_lineage_preview_promotes_lineage_fields(tmp_path: Path) -> None:
    import json

    paths = _paths(tmp_path)
    journal = paths.decision_journal_path
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        json.dumps(
            {
                "ts": 123.0,
                "symbol": "BTCUSDT",
                "interval": "15m",
                "action": "ENTER_LONG",
                "reason": "ev_enter_long",
                "score": 0.12,
                "components": {
                    "lineage": {
                        "decision_id": "dec-1",
                        "order_id": "ord-1",
                        "close_trigger": "take_profit",
                        "rf": {"candidate_id": "rf-1"},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = ControlPlaneStore(paths).summary()["decision_lineage_preview"][0]

    assert row["lineage"]["decision_id"] == "dec-1"
    assert row["decision_id"] == "dec-1"
    assert row["order_id"] == "ord-1"
    assert row["close_trigger"] == "take_profit"

