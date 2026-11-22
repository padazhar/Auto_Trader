from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
            fairness_fast_timeframes=["1m", "3m"],
            fairness_anchor_timeframes=["15m"],
            fairness_secondary_timeframes=["5m"],
            deprioritize_anchor_after_ready=True,
            failure_cooldown_sec=60.0,
        ),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=120),
            gbm=SimpleNamespace(enabled=True, min_rows=10, max_age_minutes=120),
            dqn=SimpleNamespace(enabled=False, allowed_intervals=[], min_rows=10, max_age_minutes=120),
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


def test_scheduler_prioritizes_fast_missing_manifests_over_stale_anchor_when_anchor_is_active(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    monkeypatch.setattr("scheduling.priority_manager._market_data_ready", lambda *args, **kwargs: (True, "market_data_ready", {}))
    monkeypatch.setattr(
        "scheduling.priority_manager._job_due_reason",
        lambda *args, **kwargs: (True, "new_bars:4/1" if kwargs["interval"] == "15m" else "manifest_missing"),
    )
    monkeypatch.setattr(
        "scheduling.priority_manager._has_active_manifest",
        lambda _paths, _model_type, _symbol, interval: interval == "15m",
    )

    jobs = build_due_training_jobs(paths, registry)

    assert jobs
    first_fast_index = min(i for i, job in enumerate(jobs) if job.interval in {"1m", "3m"})
    first_anchor_index = min(i for i, job in enumerate(jobs) if job.interval == "15m")
    assert first_fast_index < first_anchor_index


def test_scheduler_keeps_missing_anchor_before_fast_timeframes(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    registry = TrainingRegistry(paths)

    monkeypatch.setattr("scheduling.priority_manager._market_data_ready", lambda *args, **kwargs: (True, "market_data_ready", {}))
    monkeypatch.setattr("scheduling.priority_manager._job_due_reason", lambda *args, **kwargs: (True, "manifest_missing"))
    monkeypatch.setattr("scheduling.priority_manager._has_active_manifest", lambda *args, **kwargs: False)

    jobs = build_due_training_jobs(paths, registry)

    assert jobs
    assert jobs[0].interval == "15m"
    assert jobs[0].model_type in {"rf", "gbm"}
