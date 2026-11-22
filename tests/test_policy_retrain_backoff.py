from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import artifact_paths, save_json, write_manifest
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

    (config_dir / "tradable_symbols.yaml").write_text(json.dumps({"symbols": ["BTCUSDT"]}), encoding="utf-8")

    config = SimpleNamespace(
        frames=SimpleNamespace(fusion_intervals=["1m", "3m", "5m", "15m"], anchor_interval="15m"),
        scheduling=SimpleNamespace(priority_intervals=["15m", "5m", "3m", "1m"], failure_cooldown_sec=60.0),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=False, min_rows=10, max_age_minutes=120),
            gbm=SimpleNamespace(enabled=False, min_rows=10, max_age_minutes=120),
            dqn=SimpleNamespace(enabled=True, allowed_intervals=["15m"], min_rows=10, max_age_minutes=120),
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


def _write_market_state(paths: _Paths, symbol: str, anchor_ts_ms: int) -> None:
    import sqlite3

    db = paths.market_db_path(symbol)
    with sqlite3.connect(str(db)) as conn:
        for interval, ts in (
            ("15m", anchor_ts_ms),
            ("1m", anchor_ts_ms + 60_000),
            ("3m", anchor_ts_ms),
            ("5m", anchor_ts_ms),
        ):
            conn.execute(f'CREATE TABLE IF NOT EXISTS "klines_{interval}" (timestamp INTEGER)')
            rows = 100 if interval == "15m" else 120
            conn.executemany(f'INSERT INTO "klines_{interval}" (timestamp) VALUES (?)', [(ts,)] * rows)
        conn.commit()


def _record_dqn_candidate_only_success(paths: _Paths, registry: TrainingRegistry) -> None:
    artifacts = artifact_paths(paths, "dqn", "BTCUSDT", "15m", stage="candidate", candidate_id="cand-1")
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"x")
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a", "feat_b"]})
    save_json(artifacts["metrics"], {"fold_count": 1})
    write_manifest(
        paths,
        model_type="dqn",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=100,
        feature_count=2,
        artifacts=artifacts,
        metrics={"fold_count": 1},
        extra={
            "feature_set": {
                "version": "feature_store_v2",
                "recipe_version": "feature_store_v2",
                "columns": ["feat_a", "feat_b"],
                "column_count": 2,
                "signature": "a" * 64,
                "optional_context_fields": [],
            }
        },
    )
    registry.mark_success(
        model_type="dqn",
        symbol="BTCUSDT",
        interval="15m",
        metrics={
            "fold_count": 1,
            "candidate_only_reasons": ["insufficient_fold_count:1"],
            "retrain_min_new_bars": 3,
            "promotion": {
                "attempted": False,
                "approved": False,
                "reasons": ["promotion_not_requested"],
                "manifest_path": str(artifacts["manifest"]),
            },
        },
        artifacts={"manifest": str(artifacts["manifest"])},
        rows_used=100,
    )


def test_candidate_only_policy_uses_backoff_before_requeue(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    anchor_ts_ms = 1_775_925_900_000
    _write_market_state(paths, "BTCUSDT", anchor_ts_ms)
    _record_dqn_candidate_only_success(paths, registry)

    jobs = build_due_training_jobs(paths, registry)
    due_keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("dqn", "BTCUSDT", "15m") not in due_keys

    # Advance by one 15m bar only -> still below retrain_min_new_bars=3.
    _write_market_state(paths, "BTCUSDT", anchor_ts_ms + 900_000)
    jobs = build_due_training_jobs(paths, registry)
    due_keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("dqn", "BTCUSDT", "15m") not in due_keys

    # Advance by three 15m bars total -> now due.
    _write_market_state(paths, "BTCUSDT", anchor_ts_ms + (3 * 900_000))
    jobs = build_due_training_jobs(paths, registry)
    due_keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("dqn", "BTCUSDT", "15m") in due_keys
