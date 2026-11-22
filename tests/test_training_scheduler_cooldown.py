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
            priority_intervals=["15m", "5m", "3m", "1m"],
            failure_cooldown_sec=60.0,
        ),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=120),
            gbm=SimpleNamespace(enabled=False, min_rows=10, max_age_minutes=120),
            dqn=SimpleNamespace(enabled=True, allowed_intervals=["15m"], min_rows=10, max_age_minutes=120),
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


def _record_success(paths: _Paths, registry: TrainingRegistry, model_type: str, symbol: str, interval: str) -> None:
    artifacts = artifact_paths(paths, model_type, symbol, interval, stage="candidate", candidate_id="cand-1")
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"x")
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a", "feat_b"]})
    save_json(artifacts["metrics"], {"fold_count": 1})
    write_manifest(
        paths,
        model_type=model_type,
        symbol=symbol,
        interval=interval,
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
        model_type=model_type,
        symbol=symbol,
        interval=interval,
        metrics={
            "fold_count": 1,
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


def test_scheduler_does_not_immediately_requeue_successful_candidate_only_jobs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_market_state(paths, "BTCUSDT")
    registry = TrainingRegistry(paths)

    _record_success(paths, registry, "rf", "BTCUSDT", "15m")
    _record_success(paths, registry, "dqn", "BTCUSDT", "15m")

    jobs = build_due_training_jobs(paths, registry)

    due_keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("rf", "BTCUSDT", "15m") not in due_keys
    assert ("dqn", "BTCUSDT", "15m") not in due_keys


def test_scheduler_excludes_blocked_symbols_even_if_tradable_yaml_is_stale(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.tradable_symbols_path.write_text(json.dumps({"symbols": ["BTCUSDT", "ETHUSDT"]}), encoding="utf-8")
    paths.symbol_state_path.write_text(
        json.dumps(
            {
                "active_symbols": ["BTCUSDT", "ETHUSDT"],
                "ranked_candidates": ["BTCUSDT", "ETHUSDT"],
                "symbols": {
                    "BTCUSDT": {"status": "BLOCKED"},
                    "ETHUSDT": {"status": "ACTIVE"},
                },
            }
        ),
        encoding="utf-8",
    )
    _write_market_state(paths, "BTCUSDT")
    _write_market_state(paths, "ETHUSDT")
    registry = TrainingRegistry(paths)

    jobs = build_due_training_jobs(paths, registry)

    assert jobs
    assert {job.symbol for job in jobs} == {"ETHUSDT"}


def test_scheduler_does_not_fallback_to_tradable_yaml_when_symbol_state_is_empty(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.tradable_symbols_path.write_text(json.dumps({"symbols": ["BTCUSDT"]}), encoding="utf-8")
    paths.symbol_state_path.write_text(
        json.dumps(
            {
                "active_symbols": [],
                "ranked_candidates": ["BTCUSDT"],
                "symbols": {"BTCUSDT": {"status": "ACTIVE"}},
            }
        ),
        encoding="utf-8",
    )
    _write_market_state(paths, "BTCUSDT")
    registry = TrainingRegistry(paths)

    jobs = build_due_training_jobs(paths, registry)

    assert jobs == []


def test_scheduler_requeues_affected_anchor_blockers_even_when_manifest_is_fresh(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    paths.symbol_state_path.write_text(
        json.dumps(
            {
                "active_symbols": ["BTCUSDT"],
                "execution_eligible_symbols": [],
                "symbols": {
                    "BTCUSDT": {
                        "status": "ACTIVE",
                        "model_ready_reason": "anchor_training_incomplete:rf:deployment_stage:candidate,gbm:feature_incompatible:feature_set_missing",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    paths.config.training.gbm.enabled = True
    registry = TrainingRegistry(paths)

    monkeypatch.setattr("scheduling.priority_manager._market_data_ready", lambda *args, **kwargs: (True, "market_data_ready", {}))
    monkeypatch.setattr("scheduling.priority_manager._job_due_reason", lambda *args, **kwargs: (False, "fresh"))

    jobs = build_due_training_jobs(paths, registry)

    keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("rf", "BTCUSDT", "15m") in keys
    assert ("gbm", "BTCUSDT", "15m") in keys
    assert all(job.reason.startswith("anchor_recovery:") for job in jobs)


def test_scheduler_does_not_recover_affected_anchor_when_market_data_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    paths.symbol_state_path.write_text(
        json.dumps(
            {
                "active_symbols": ["BTCUSDT"],
                "execution_eligible_symbols": [],
                "symbols": {
                    "BTCUSDT": {
                        "status": "ACTIVE",
                        "model_ready_reason": "anchor_training_incomplete:rf:deployment_stage:candidate",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    registry = TrainingRegistry(paths)

    monkeypatch.setattr("scheduling.priority_manager._market_data_ready", lambda *args, **kwargs: (False, "anchor_rows_missing", {}))
    monkeypatch.setattr("scheduling.priority_manager._job_due_reason", lambda *args, **kwargs: (False, "fresh"))

    jobs = build_due_training_jobs(paths, registry)

    assert jobs == []


def test_scheduler_waits_for_new_bars_after_statistical_ineligibility(tmp_path: Path) -> None:
    import sqlite3

    paths = _paths(tmp_path)
    _write_market_state(paths, "BTCUSDT")
    registry = TrainingRegistry(paths)
    registry.mark_ineligible(
        "rf",
        "BTCUSDT",
        "15m",
        "rf_valid_label_support_invalid:insufficient_class_count:1<2",
    )

    before = build_due_training_jobs(paths, registry)
    assert ("rf", "BTCUSDT", "15m") not in {(job.model_type, job.symbol, job.interval) for job in before}

    reference_ms = registry.get_entry("rf", "BTCUSDT", "15m").ineligible_reference_anchor_ts_ms
    assert reference_ms is not None
    with sqlite3.connect(str(paths.market_db_path("BTCUSDT"))) as conn:
        conn.execute('INSERT INTO "klines_15m" (timestamp) VALUES (?)', (int(reference_ms) + 900_000,))
        conn.commit()

    after = build_due_training_jobs(paths, registry)
    matching = [job for job in after if (job.model_type, job.symbol, job.interval) == ("rf", "BTCUSDT", "15m")]
    assert matching
    assert matching[0].reason.startswith("statistical_ineligible_new_bars:")
