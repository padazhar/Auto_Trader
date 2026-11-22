from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import artifact_paths, promote_candidate_manifest, save_json, write_manifest
from core.training_registry import TrainingRegistry
from scheduling.priority_manager import build_due_training_jobs


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
    (runtime_dir / "symbol_state.json").write_text(json.dumps({"active_symbols": ["BTCUSDT"], "symbols": {}}), encoding="utf-8")
    (config_dir / "tradable_symbols.yaml").write_text(json.dumps({"symbols": ["BTCUSDT"]}), encoding="utf-8")
    config = SimpleNamespace(
        frames=SimpleNamespace(fusion_intervals=["1m", "3m", "5m", "15m"], anchor_interval="15m"),
        scheduling=SimpleNamespace(priority_intervals=["15m", "5m", "3m", "1m"], failure_cooldown_sec=60.0),
        training=SimpleNamespace(
            retrain_cycles={"15m": 1, "5m": 2, "3m": 4, "1m": 7},
            rf=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=1_000_000),
            gbm=SimpleNamespace(enabled=False, min_rows=10, max_age_minutes=1_000_000),
            dqn=SimpleNamespace(enabled=False, allowed_intervals=["15m"], min_rows=10, max_age_minutes=1_000_000),
            promotion=SimpleNamespace(min_fold_count=1),
        ),
    )
    return _Paths(project_root=project_root, config_dir=config_dir, runtime_dir=runtime_dir, logs_dir=logs_dir, model_storage_dir=model_storage_dir, market_storage_dir=market_storage_dir, config=config)


def _write_market_state(paths: _Paths, ts_by_interval: dict[str, int]) -> None:
    db = paths.market_db_path("BTCUSDT")
    with sqlite3.connect(str(db)) as conn:
        for interval, ts in ts_by_interval.items():
            conn.execute(f'DROP TABLE IF EXISTS "klines_{interval}"')
            conn.execute(f'CREATE TABLE "klines_{interval}" (timestamp INTEGER)')
            rows = 100
            conn.executemany(f'INSERT INTO "klines_{interval}" (timestamp) VALUES (?)', [(ts,)] * rows)
        conn.commit()


def _record_active_manifest(paths: _Paths, registry: TrainingRegistry, interval: str, trained_ts_ms: int) -> None:
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", interval, stage="candidate", candidate_id=f"cand-{interval}")
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a"]})
    save_json(artifacts["metrics"], {"training_reference_anchor_ts_ms": trained_ts_ms})
    write_manifest(
        paths,
        model_type="rf",
        symbol="BTCUSDT",
        interval=interval,
        rows_used=100,
        feature_count=1,
        artifacts=artifacts,
        metrics={"training_reference_anchor_ts_ms": trained_ts_ms},
        extra={
            "training_reference_anchor_ts_ms": trained_ts_ms,
            "training": {"anchor_last_ts_ms": trained_ts_ms},
            "feature_set": {"version": "feature_store_v2", "recipe_version": "feature_store_v2", "columns": ["feat_a"], "column_count": 1, "signature": "a" * 64, "optional_context_fields": []},
        },
    )
    active_manifest = promote_candidate_manifest(paths, artifacts["manifest"])
    registry.mark_success(
        model_type="rf",
        symbol="BTCUSDT",
        interval=interval,
        metrics={"training_reference_anchor_ts_ms": trained_ts_ms, "promotion": {"attempted": True, "approved": True, "manifest_path": str(active_manifest), "reasons": []}},
        artifacts={"manifest": str(active_manifest)},
        rows_used=100,
    )


def test_step9_retrain_cycles_are_interval_owned(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)
    base = 1_775_925_900_000
    _write_market_state(paths, {"15m": base, "5m": base, "3m": base, "1m": base})
    for interval in ("15m", "5m", "3m", "1m"):
        _record_active_manifest(paths, registry, interval, base)

    _write_market_state(paths, {"15m": base + 900_000, "5m": base + 300_000, "3m": base + 180_000, "1m": base + 60_000})
    jobs = build_due_training_jobs(paths, registry)
    due = {(job.model_type, job.interval, job.reason) for job in jobs}
    assert ("rf", "15m", "new_bars:1/1") in due
    assert not any(item[1] == "5m" for item in due)
    assert not any(item[1] == "3m" for item in due)
    assert not any(item[1] == "1m" for item in due)

    _write_market_state(paths, {"15m": base + 900_000, "5m": base + 600_000, "3m": base + 720_000, "1m": base + 420_000})
    jobs = build_due_training_jobs(paths, registry)
    due = {(job.model_type, job.interval, job.reason) for job in jobs}
    assert ("rf", "5m", "new_bars:2/2") in due
    assert ("rf", "3m", "new_bars:4/4") in due
    assert ("rf", "1m", "new_bars:7/7") in due
