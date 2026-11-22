from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.training_registry import TrainingRegistry
from scheduling.priority_manager import build_training_backlog_diagnostics


class _Paths(SimpleNamespace):
    def load_yaml(self, path: Path):
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def market_db_path(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{symbol.upper()}.db"

    def rf_model_dir(self, symbol: str, interval: str) -> Path:
        return self.model_storage_dir / "random_forest_models" / symbol.upper() / interval

    def gbm_model_dir(self, symbol: str, interval: str) -> Path:
        return self.model_storage_dir / "gradient_boost_models" / symbol.upper() / interval

    def dqn_model_dir(self, symbol: str, interval: str) -> Path:
        return self.model_storage_dir / "reinforcement_agents" / symbol.upper() / interval

    @property
    def registry_path(self) -> Path:
        return self.runtime_dir / "training_registry.json"

    @property
    def training_registry_path(self) -> Path:
        return self.registry_path

    @property
    def tradable_symbols_path(self) -> Path:
        return self.config_dir / "tradable_symbols.yaml"

    @property
    def symbol_state_path(self) -> Path:
        return self.runtime_dir / "symbol_state.json"

    def log_file_path(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"


def _paths(tmp_path: Path) -> _Paths:
    project_root = tmp_path / "repo"
    config_dir = project_root / "config"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"
    model_storage_dir = project_root / "data" / "model_storage"
    market_storage_dir = project_root / "data" / "market_storage"
    for path in (config_dir, runtime_dir, logs_dir, model_storage_dir, market_storage_dir):
        path.mkdir(parents=True, exist_ok=True)

    (config_dir / "tradable_symbols.yaml").write_text(json.dumps({"symbols": ["BTCUSDT"]}), encoding="utf-8")
    (runtime_dir / "symbol_state.json").write_text(
        json.dumps(
            {
                "active_symbols": ["BTCUSDT"],
                "ranked_candidates": ["BTCUSDT"],
                "symbols": {"BTCUSDT": {"status": "ACTIVE"}},
            }
        ),
        encoding="utf-8",
    )

    config = SimpleNamespace(
        frames=SimpleNamespace(fusion_intervals=["1m", "3m", "5m", "15m"], anchor_interval="15m"),
        scheduling=SimpleNamespace(
            priority_intervals=["15m"],
            failure_cooldown_sec=60.0,
            retrain_min_new_bars={},
        ),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=120),
            gbm=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=120),
            dqn=SimpleNamespace(enabled=False, allowed_intervals=["15m"], min_rows=10, max_age_minutes=120),
            promotion=SimpleNamespace(min_fold_count=3),
        ),
    )
    return _Paths(
        project_root=project_root,
        config_dir=config_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        model_storage_dir=model_storage_dir,
        market_storage_dir=market_storage_dir,
        config=config,
    )


def _write_market_state(paths: _Paths, symbol: str) -> None:
    import sqlite3

    db = paths.market_db_path(symbol)
    with sqlite3.connect(str(db)) as conn:
        for interval, ts in (("15m", 1_775_925_900_000), ("1m", 1_775_926_800_000), ("3m", 1_775_926_620_000), ("5m", 1_775_926_500_000)):
            conn.execute(f'CREATE TABLE IF NOT EXISTS "klines_{interval}" (timestamp INTEGER)')
            rows = 100 if interval == "15m" else 120
            conn.executemany(f'INSERT INTO "klines_{interval}" (timestamp) VALUES (?)', [(ts,)] * rows)
        conn.commit()


def test_registry_snapshot_classifies_idle_missing_active_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    registry.ensure_entry("rf", "BTCUSDT", "15m")

    rows = registry.as_dashboard_rows()

    assert rows[0]["diagnostic_category"] == "missing_active_manifest"
    assert rows[0]["diagnostic_detail"] == "idle_without_active_manifest"


def test_registry_snapshot_classifies_candidate_only_success(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    manifest = paths.model_storage_dir / "candidate" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "reasons": ["insufficient_fold_count:1"],
                "manifest_path": str(manifest),
            }
        },
        artifacts={"manifest": str(manifest)},
        rows_used=100,
    )

    rows = registry.as_dashboard_rows()

    assert rows[0]["diagnostic_category"] == "candidate_only"
    assert "insufficient_fold_count:1" in rows[0]["diagnostic_detail"]


def test_scheduler_backlog_diagnostics_are_observability_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_market_state(paths, "BTCUSDT")
    registry = TrainingRegistry(paths)
    before = registry.snapshot()

    diagnostics = build_training_backlog_diagnostics(paths, registry)

    after = registry.snapshot()
    assert before == after
    assert diagnostics["total_rows"] == 2
    categories = {row["category"] for row in diagnostics["rows"]}
    assert "registry_missing" in categories


def test_registry_dashboard_exposes_separate_model_state_contract_for_candidate_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    manifest = paths.model_storage_dir / "candidate" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        metrics={
            "promotion": {
                "attempted": True,
                "approved": False,
                "reasons": ["candidate_score_not_improved"],
                "manifest_path": str(manifest),
            }
        },
        artifacts={"manifest": str(manifest)},
        rows_used=100,
    )

    row = registry.as_dashboard_rows()[0]
    assert row["state_contract_version"] == "model_state_v1"
    assert row["candidate_state"] == "saved_invalid"
    assert row["promotion_state"] == "rejected"
    assert row["active_manifest_state"] == "missing"
    assert row["execution_eligibility_state"] == "not_eligible"
    assert row["lifecycle_state"] == "promotion_rejected"


def test_registry_training_result_exposes_model_state_contract(tmp_path: Path, monkeypatch) -> None:
    from tests.test_promotion_truth import _DummyLogger, _DummyMonitor

    monitor = _DummyMonitor(None)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: monitor)
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())

    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    manifest = paths.model_storage_dir / "candidate" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        metrics={"promotion": {"attempted": False, "approved": False, "reasons": [], "manifest_path": str(manifest)}},
        artifacts={"manifest": str(manifest)},
        rows_used=100,
    )

    result = monitor.training_results[-1]
    contract = result["model_state_contract"]
    assert contract["state_contract_version"] == "model_state_v1"
    assert contract["candidate_state"] == "saved_invalid"
    assert contract["promotion_state"] == "not_attempted"
    assert contract["execution_eligibility_state"] == "not_eligible"


def test_registry_marks_statistical_ineligibility_explicitly(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_market_state(paths, "BTCUSDT")
    registry = TrainingRegistry(paths)

    registry.mark_ineligible(
        "rf",
        "BTCUSDT",
        "15m",
        "rf_valid_label_support_invalid:insufficient_class_count:1<2",
    )

    entry = registry.get_entry("rf", "BTCUSDT", "15m")
    row = next(item for item in registry.as_dashboard_rows() if item["model_type"] == "rf")
    assert entry.status == "ineligible"
    assert entry.ineligible_reference_anchor_ts_ms is not None
    assert row["diagnostic_category"] == "ineligible"
    assert "insufficient_class_count" in row["diagnostic_detail"]
    assert row["execution_eligibility_state"] == "not_eligible"
