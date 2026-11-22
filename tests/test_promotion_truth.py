from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import incumbent_pointer_path, save_json
from core.training_registry import TrainingRegistry


class _DummyLogger:
    def warning(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.events: list[dict[str, object]] = []
        self.training_results: list[dict[str, object]] = []
        self.registry_updates: list[dict[str, object]] = []
        self.model_versions: list[dict[str, object]] = []

    def publish_event(self, **payload) -> None:
        self.events.append(payload)

    def append_training_result(self, payload) -> None:
        self.training_results.append(dict(payload))

    def update_registry(self, **payload) -> None:
        self.registry_updates.append(dict(payload))

    def update_model_versions(self, payload) -> None:
        self.model_versions.append(dict(payload))


def _paths(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    project_root.mkdir(parents=True, exist_ok=True)
    runtime_dir = project_root / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=project_root,
        registry_path=runtime_dir / "training_registry.json",
    )


def test_training_registry_only_emits_promoted_event_when_candidate_is_really_promoted(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    candidate_manifest = paths.project_root / "candidate_manifest.json"
    save_json(
        candidate_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig123"},
            "lifecycle": {"stage": "candidate", "candidate_id": "cand123", "promoted_at": None},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": "2026-04-03T00:00:00+00:00"},
        },
    )

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": str(candidate_manifest)},
        metrics={
            "precision_macro": 0.22,
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "cand123",
                "manifest_path": str(candidate_manifest),
                "reasons": ["candidate_score_not_improved"],
            },
        },
    )

    categories = [event.get("category") for event in monitor.events]
    assert "candidate_written" in categories
    assert "promotion_rejected" in categories
    assert "artifact_promoted" not in categories

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.promotion_attempted is True
    assert entry.promotion_approved is False
    assert entry.promoted_at is None
    assert entry.promotion_reasons == ["candidate_score_not_improved"]


def test_training_registry_emits_promoted_event_only_for_approved_candidate(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    candidate_manifest = paths.project_root / "approved_manifest.json"
    save_json(
        candidate_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "ETHUSDT",
            "interval": "5m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig999"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "cand999", "promoted_at": "2026-04-03T01:00:00+00:00"},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": "2026-04-03T01:00:00+00:00"},
        },
    )

    registry.mark_success(
        model_type="rf",
        symbol="ETHUSDT",
        interval="5m",
        rows_used=120,
        artifacts={"manifest": str(candidate_manifest)},
        metrics={
            "precision_macro": 0.25,
            "promotion": {
                "attempted": True,
                "approved": True,
                "promoted": True,
                "candidate_id": "cand999",
                "manifest_path": str(candidate_manifest),
                "pointer_path": str(paths.project_root / "incumbent.json"),
                "pointer": {"promoted_at": "2026-04-03T01:00:00+00:00"},
                "reasons": [],
            },
        },
    )

    categories = [event.get("category") for event in monitor.events]
    assert "candidate_written" in categories
    assert "artifact_promoted" not in categories
    assert "promotion_rejected" in categories

    entry = registry.get_entry("rf", "ETHUSDT", "5m")
    assert entry.promotion_attempted is True
    assert entry.promotion_approved is False
    assert entry.promoted_at == "2026-04-03T01:00:00+00:00"
    assert any("active_manifest_truth_error" in reason for reason in entry.truth_reasons)


def test_training_registry_recovers_approval_from_matching_active_incumbent(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    active_manifest = paths.project_root / "active_manifest.json"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig123"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "cand123", "promoted_at": "2026-04-03T01:00:00+00:00"},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": "2026-04-03T01:00:00+00:00"},
        },
    )
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True,
            "active_manifest_path": str(active_manifest),
            "reason": "ready",
        },
    )

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": str(active_manifest)},
        metrics={
            "precision_macro": 0.33,
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "cand123",
                "manifest_path": str(active_manifest),
                "reasons": [],
            },
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.promotion_approved is True
    assert entry.execution_ready is True
    assert entry.registry_truth_status == "execution_ready"


def test_training_registry_does_not_recover_approval_for_candidate_id_mismatch(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    active_manifest = paths.project_root / "active_manifest.json"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig123"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "different", "promoted_at": "2026-04-03T01:00:00+00:00"},
        },
    )
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True,
            "active_manifest_path": str(active_manifest),
            "reason": "ready",
        },
    )

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": str(active_manifest)},
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "cand123",
                "manifest_path": str(active_manifest),
                "reasons": [],
            },
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.promotion_approved is False
    assert entry.execution_ready is False
    assert entry.registry_truth_status == "active_without_registry_approval"


def test_training_registry_retains_pointer_proven_incumbent_when_new_candidate_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    paths.model_storage_dir = paths.project_root / "data" / "model_storage"
    paths.model_storage_dir.mkdir(parents=True, exist_ok=True)
    active_manifest = paths.project_root / "old_incumbent_manifest.json"
    active_promoted_at = "2026-04-03T01:00:00+00:00"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-old"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "incumbent-old", "promoted_at": active_promoted_at},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": active_promoted_at},
        },
    )
    save_json(
        incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"),
        {
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": "incumbent-old",
            "manifest_path": str(active_manifest),
            "promoted_at": active_promoted_at,
        },
    )
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True,
            "active_manifest_path": str(active_manifest),
            "reason": "ready",
        },
    )

    rejected_manifest = paths.project_root / "rejected_candidate_manifest.json"
    save_json(
        rejected_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-new"},
            "lifecycle": {"stage": "candidate", "candidate_id": "candidate-new", "promoted_at": None},
            "timestamps": {"created_at": "2026-04-04T00:00:00+00:00", "updated_at": "2026-04-04T00:00:00+00:00"},
        },
    )

    registry = TrainingRegistry(paths)
    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=120,
        artifacts={"manifest": str(rejected_manifest)},
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "candidate-new",
                "manifest_path": str(rejected_manifest),
                "reasons": ["candidate_score_not_improved"],
            },
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.execution_ready is True
    assert entry.promotion_approved is True
    assert entry.latest_candidate_promotion_approved is False
    assert entry.active_incumbent_candidate_id == "incumbent-old"
    assert entry.active_incumbent_approval_source == "active_incumbent_pointer"
    assert entry.registry_truth_status == "execution_ready"
    assert "active_incumbent_retained_while_latest_candidate_not_approved" in entry.truth_reasons
    categories = [event.get("category") for event in monitor.events]
    assert "promotion_rejected" in categories
    assert "artifact_promoted" not in categories

    registry.mark_running("rf", "BTCUSDT", "15m")
    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.status == "running"
    assert entry.registry_truth_status == "execution_ready"
    assert entry.execution_ready is True


def test_training_registry_load_migrates_pointer_proven_incumbent_after_rejected_candidate(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    paths.model_storage_dir = paths.project_root / "data" / "model_storage"
    paths.model_storage_dir.mkdir(parents=True, exist_ok=True)
    active_manifest = paths.project_root / "migrated_incumbent_manifest.json"
    promoted_at = "2026-04-03T01:00:00+00:00"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-old"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "old-incumbent", "promoted_at": promoted_at},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": promoted_at},
        },
    )
    save_json(
        incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"),
        {
            "model_type": "rf", "symbol": "BTCUSDT", "interval": "15m",
            "candidate_id": "old-incumbent", "manifest_path": str(active_manifest), "promoted_at": promoted_at,
        },
    )
    rejected_manifest = paths.project_root / "latest_rejected.json"
    save_json(
        rejected_manifest,
        {
            "schema_version": 4, "model_type": "rf", "model_role": "direction",
            "symbol": "BTCUSDT", "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-new"},
            "lifecycle": {"stage": "candidate", "candidate_id": "latest-rejected", "promoted_at": None},
            "timestamps": {"created_at": "2026-04-04T00:00:00+00:00", "updated_at": "2026-04-04T00:00:00+00:00"},
        },
    )
    paths.registry_path.write_text(json.dumps({
        "rf:BTCUSDT:15m": {
            "model_type": "rf", "symbol": "BTCUSDT", "interval": "15m", "status": "success",
            "candidate_id": "latest-rejected", "manifest_path": str(rejected_manifest),
            "metrics": {"promotion": {"attempted": True, "approved": False, "promoted": False, "candidate_id": "latest-rejected", "manifest_path": str(rejected_manifest), "reasons": ["candidate_score_not_improved"]}},
            "promotion_attempted": True, "promotion_approved": False, "promotion_reasons": ["candidate_score_not_improved"],
            "active_manifest_path": str(active_manifest), "active_manifest_valid": True,
            "execution_ready": False, "registry_truth_status": "active_without_registry_approval",
        }
    }), encoding="utf-8")
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True, "active_manifest_path": str(active_manifest), "reason": "ready"
        },
    )

    registry = TrainingRegistry(paths)
    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    persisted = json.loads(paths.registry_path.read_text(encoding="utf-8"))["rf:BTCUSDT:15m"]
    assert entry.execution_ready is True
    assert entry.registry_truth_status == "execution_ready"
    assert entry.active_incumbent_candidate_id == "old-incumbent"
    assert persisted["execution_ready"] is True
    assert persisted["latest_candidate_promotion_approved"] is False
    assert persisted["active_incumbent_approval_source"] == "active_incumbent_pointer"


def test_training_registry_does_not_recover_mismatched_incumbent_pointer(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    paths.model_storage_dir = paths.project_root / "data" / "model_storage"
    paths.model_storage_dir.mkdir(parents=True, exist_ok=True)
    active_manifest = paths.project_root / "active_manifest.json"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig123"},
            "lifecycle": {"stage": "incumbent", "candidate_id": "old", "promoted_at": "2026-04-03T01:00:00+00:00"},
            "timestamps": {"created_at": "2026-04-03T00:00:00+00:00", "updated_at": "2026-04-03T01:00:00+00:00"},
        },
    )
    save_json(
        incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"),
        {
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": "not-the-manifest-candidate",
            "manifest_path": str(active_manifest),
            "promoted_at": "2026-04-03T01:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True,
            "active_manifest_path": str(active_manifest),
            "reason": "ready",
        },
    )
    registry = TrainingRegistry(paths)
    rejected_manifest = paths.project_root / "rejected.json"
    save_json(
        rejected_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-new"},
            "lifecycle": {"stage": "candidate", "candidate_id": "new", "promoted_at": None},
            "timestamps": {"created_at": "2026-04-04T00:00:00+00:00", "updated_at": "2026-04-04T00:00:00+00:00"},
        },
    )
    registry.mark_success(
        model_type="rf", symbol="BTCUSDT", interval="15m", rows_used=100,
        artifacts={"manifest": str(rejected_manifest)},
        metrics={"promotion": {"attempted": True, "approved": False, "promoted": False, "candidate_id": "new", "manifest_path": str(rejected_manifest), "reasons": ["rejected"]}},
    )
    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.execution_ready is False
    assert entry.promotion_approved is False
    assert entry.registry_truth_status == "active_without_registry_approval"


def test_training_registry_exposes_paper_bootstrap_candidate_as_non_executable(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    candidate_manifest = paths.project_root / "bootstrap_candidate_manifest.json"
    save_json(
        candidate_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-bootstrap"},
            "lifecycle": {"stage": "candidate", "candidate_id": "bootstrap-candidate", "promoted_at": None},
            "timestamps": {"created_at": "2026-04-05T00:00:00+00:00", "updated_at": "2026-04-05T00:00:00+00:00"},
        },
    )

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": str(candidate_manifest)},
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "bootstrap-candidate",
                "manifest_path": str(candidate_manifest),
                "reasons": ["paper_bootstrap_not_authorized_for_execution"],
                "paper_evidence_status": "paper_bootstrap_not_execution_ready",
                "paper_bootstrap_only": True,
                "paper_evidence_sufficient": False,
            },
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.latest_candidate_paper_evidence_status == "paper_bootstrap_not_execution_ready"
    assert entry.latest_candidate_paper_bootstrap_only is True
    assert entry.latest_candidate_paper_evidence_sufficient is False
    assert entry.execution_ready is False
    assert entry.promotion_approved is False
    assert entry.registry_truth_status in {"promotion_rejected", "candidate_only", "candidate_invalid"}

    result = monitor.training_results[-1]
    assert result["promotion"]["paper_bootstrap_only"] is True
    assert result["promotion"]["paper_evidence_status"] == "paper_bootstrap_not_execution_ready"


def test_training_registry_accepts_pointer_proven_paper_bootstrap_only_for_paper_execution(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "ready"))

    paths = _paths(tmp_path)
    paths.config = SimpleNamespace(runtime=SimpleNamespace(execution_mode="paper"))
    paths.model_storage_dir = paths.project_root / "data" / "model_storage"
    paths.model_storage_dir.mkdir(parents=True, exist_ok=True)
    active_manifest = paths.project_root / "paper_bootstrap_manifest.json"
    promoted_at = "2026-05-29T01:00:00+00:00"
    save_json(
        active_manifest,
        {
            "schema_version": 4,
            "model_type": "rf",
            "model_role": "direction",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "feature_set": {"version": "feature_store_v2", "signature": "sig-paper"},
            "lifecycle": {
                "stage": "paper_bootstrap",
                "candidate_id": "paper-first",
                "promoted_at": promoted_at,
                "paper_execution_only": True,
                "live_eligible": False,
            },
        },
    )
    save_json(
        incumbent_pointer_path(paths, "rf", "BTCUSDT", "15m"),
        {
            "model_type": "rf",
            "symbol": "BTCUSDT",
            "interval": "15m",
            "candidate_id": "paper-first",
            "manifest_path": str(active_manifest),
            "promoted_at": promoted_at,
            "deployment_stage": "paper_bootstrap",
            "paper_execution_only": True,
            "live_eligible": False,
        },
    )
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": True,
            "active_manifest_path": str(active_manifest),
            "reason": "paper_bootstrap_ready",
        },
    )

    registry = TrainingRegistry(paths)
    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": str(active_manifest)},
        metrics={
            "promotion": {
                "attempted": True,
                "approved": True,
                "promoted": True,
                "candidate_id": "paper-first",
                "manifest_path": str(active_manifest),
                "paper_evidence_status": "paper_bootstrap_execution_ready",
                "paper_bootstrap_only": True,
                "paper_evidence_sufficient": False,
                "reasons": [],
            }
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.execution_ready is True
    assert entry.promotion_approved is True
    assert entry.active_incumbent_approval_source == "active_paper_bootstrap_pointer"
    assert entry.active_incumbent_deployment_stage == "paper_bootstrap"
    assert entry.active_incumbent_paper_execution_only is True
    assert "paper_bootstrap_execution_only_non_live" in entry.truth_reasons


def test_training_registry_repairs_stale_paper_bootstrap_truth_from_active_pointer_authority(tmp_path: Path, monkeypatch) -> None:
    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr(
        "core.training_registry.active_manifest_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "execution_ready": False,
            "active_manifest_path": "/runtime/paper-bootstrap-active.json",
            "reason": "deployment_stage:paper_bootstrap",
        },
    )
    monkeypatch.setattr(
        "core.training_registry.paper_bootstrap_active_authority_truth",
        lambda paths, model_type, symbol, interval, model_role=None: {
            "authority_approved": True,
            "authority_reason": "paper_bootstrap_pointer_authority_ready",
            "authority_source": "active_paper_bootstrap_pointer",
            "active_manifest_path": "/runtime/paper-bootstrap-active.json",
            "candidate_id": "paper-active",
            "promoted_at": "2026-05-29T01:00:00+00:00",
        },
    )
    monkeypatch.setattr("core.training_registry.validate_artifact_bundle", lambda paths, manifest_path: (True, "fresh"))

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        artifacts={"manifest": "/runtime/latest-rejected.json"},
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "promoted": False,
                "candidate_id": "latest-rejected",
                "manifest_path": "/runtime/latest-rejected.json",
                "reasons": ["below_min_paper_expectancy"],
            }
        },
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    assert entry.execution_ready is True
    assert entry.promotion_approved is True
    assert entry.registry_truth_status == "execution_ready"
    assert entry.active_manifest_valid is True
    assert entry.active_incumbent_candidate_id == "paper-active"
    assert entry.active_incumbent_approval_source == "active_paper_bootstrap_pointer"
    assert entry.active_incumbent_paper_execution_only is True
    assert "active_incumbent_retained_while_latest_candidate_not_approved" in entry.truth_reasons
    assert "paper_bootstrap_execution_only_non_live" in entry.truth_reasons
