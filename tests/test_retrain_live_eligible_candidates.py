from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.result_contracts import build_success_result
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION, discover_live_eligible_jobs, run_retraining


class _DummyMonitor:
    def update_registry(self, *args, **kwargs):
        pass

    def update_model_versions(self, *args, **kwargs):
        pass

    def append_training_result(self, *args, **kwargs):
        pass

    def publish_event(self, *args, **kwargs):
        pass


class _DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


class _Paths(SimpleNamespace):
    def load_yaml(self, path: Path):
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def market_db_path(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{symbol.upper()}.db"

    @property
    def registry_path(self) -> Path:
        return self.runtime_dir / "training_registry.json"

    @property
    def training_registry_path(self) -> Path:
        return self.registry_path

    @property
    def tradable_symbols_path(self) -> Path:
        return self.config_dir / "tradable_symbols.yaml"

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
        scheduling=SimpleNamespace(priority_intervals=["15m", "5m", "3m", "1m"]),
        trading=SimpleNamespace(fees_bps=3.0, slippage_bps=2.0, spread_bps=1.0, funding_bps=0.0, min_profit_atr_mult=0.20),
        training=SimpleNamespace(
            rf=SimpleNamespace(enabled=True),
            gbm=SimpleNamespace(enabled=True),
            dqn=SimpleNamespace(enabled=True, allowed_intervals=["15m", "5m"]),
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


def _fake_success(paths: _Paths, model_type: str, symbol: str, interval: str):
    base = paths.model_storage_dir / f"{model_type}_{symbol}_{interval}"
    base.mkdir(parents=True, exist_ok=True)
    model_path = base / "model.joblib"
    features_path = base / "feature_columns.json"
    metrics_path = base / "metrics.json"
    manifest_path = base / "manifest.json"
    model_path.write_bytes(b"model")
    features_path.write_text(json.dumps({"feature_columns": ["feat_a", "feat_b"]}), encoding="utf-8")
    metrics = {
        "promotion": {"attempted": False, "approved": False, "reasons": ["promotion_not_requested"], "candidate_id": f"cand-{model_type}-{interval}"},
        "governance": {"live_eligible": model_type != "dqn" or interval == "15m"},
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "model_type": model_type,
                "symbol": symbol,
                "interval": interval,
                "feature_set": {
                    "version": "feature_store_v2",
                    "recipe_version": "feature_store_v2",
                    "columns": ["feat_a", "feat_b"],
                    "column_count": 2,
                    "signature": "a" * 64,
                    "anchor_interval": interval,
                    "optional_context_fields": [],
                },
                "artifacts": {
                    "model": str(model_path),
                    "feature_columns": str(features_path),
                    "metrics": str(metrics_path),
                },
                "metrics_path": str(metrics_path),
                "feature_columns_path": str(features_path),
                "validation": {},
                "training": {},
                "lifecycle": {"stage": "candidate", "candidate_id": f"cand-{model_type}-{interval}", "promoted_at": None},
                "timestamps": {"created_at": "2026-04-17T00:00:00+00:00", "updated_at": "2026-04-17T00:00:00+00:00"},
            }
        ),
        encoding="utf-8",
    )
    return build_success_result(
        model_type=model_type,
        symbol=symbol,
        interval=interval,
        rows_used=111,
        artifacts={"model": str(model_path), "feature_columns": str(features_path), "metrics": str(metrics_path), "manifest": str(manifest_path)},
        metrics=metrics,
    )


def test_discover_live_eligible_jobs_includes_all_enabled_model_families(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    jobs = discover_live_eligible_jobs(paths)
    keys = {(job.model_type, job.symbol, job.interval) for job in jobs}
    assert ("rf", "BTCUSDT", "15m") in keys
    assert ("rf", "BTCUSDT", "1m") in keys
    assert ("gbm", "BTCUSDT", "5m") in keys
    assert ("dqn", "BTCUSDT", "15m") in keys
    assert ("dqn", "BTCUSDT", "1m") not in keys


def test_run_retraining_writes_report_and_registry(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: _DummyMonitor())
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("scripts.retrain_live_eligible_candidates.train_rf_model", lambda paths, symbol, interval: _fake_success(paths, "rf", symbol, interval))
    monkeypatch.setattr("scripts.retrain_live_eligible_candidates.train_gbm_model", lambda paths, symbol, interval: _fake_success(paths, "gbm", symbol, interval))
    monkeypatch.setattr("scripts.retrain_live_eligible_candidates.train_dqn_model", lambda paths, symbol, interval: _fake_success(paths, "dqn", symbol, interval))

    report = run_retraining(paths, jobs=discover_live_eligible_jobs(paths))
    assert report.report_status == "pass"
    assert report.jobs_requested == 10
    assert report.jobs_failed == 0
    report_path = paths.runtime_dir / "retrain_live_eligible_report.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["jobs_succeeded"] == 10
    assert saved["retraining_contract_version"] == RETRAINING_CONTRACT_VERSION
    assert saved["registry_before"]["total_entries"] == 0
    assert saved["cost_aware_label_ev_contract"]["spread_bps"] == 1.0
    assert saved["cost_aware_label_ev_contract"]["contract_version"] == RETRAINING_CONTRACT_VERSION
    assert saved["registry_before"]["total_entries"] == 0
    assert saved["registry_after"]["total_entries"] == 10
    assert "registry_execution_ready_delta" in saved

    registry = json.loads(paths.registry_path.read_text(encoding="utf-8"))
    assert registry["rf:BTCUSDT:15m"]["status"] == "success"
    assert registry["dqn:BTCUSDT:5m"]["status"] == "success"


def test_run_retraining_fails_when_no_jobs_are_discovered(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: _DummyMonitor())
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())

    report = run_retraining(paths, jobs=[])

    assert report.report_status == "fail"
    assert report.jobs_requested == 0
    saved = json.loads((paths.runtime_dir / "retrain_live_eligible_report.json").read_text(encoding="utf-8"))
    assert saved["retraining_contract_version"] == RETRAINING_CONTRACT_VERSION
    assert saved["registry_before"]["total_entries"] == 0
    assert saved["registry_after"]["total_entries"] == 0


def test_run_retraining_report_captures_registry_governance_delta(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: _DummyMonitor())
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("scripts.retrain_live_eligible_candidates.train_rf_model", lambda paths, symbol, interval: _fake_success(paths, "rf", symbol, interval))

    jobs = [job for job in discover_live_eligible_jobs(paths) if job.model_type == "rf" and job.interval == "15m"]
    report = run_retraining(paths, jobs=jobs)
    payload = report.to_dict()

    assert payload["jobs_requested"] == 1
    assert payload["registry_before"]["total_entries"] == 0
    assert payload["registry_after"]["total_entries"] == 1
    assert payload["registry_after"]["status_counts"]["success"] == 1
    assert isinstance(payload["registry_execution_ready_delta"], int)


def test_discover_affected_anchor_jobs_targets_active_non_ready_anchor_blockers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    symbol_state = {
        "active_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BILLUSDT"],
        "execution_eligible_symbols": ["SOLUSDT"],
        "symbols": {
            "BTCUSDT": {
                "data_ready_reason": "market_data_ready",
                "model_ready_reason": "anchor_readiness_not_ready:rf:15m:registry_not_execution_ready:active_without_registry_approval,gbm:15m:feature_incompatible:feature_set_missing",
            },
            "ETHUSDT": {
                "data_ready_reason": "market_data_ready",
                "model_ready_reason": "anchor_readiness_not_ready:rf:15m:registry_not_execution_ready:active_without_registry_approval",
            },
            "SOLUSDT": {
                "data_ready_reason": "market_data_ready",
                "model_ready_reason": "anchor_ready",
            },
            "BILLUSDT": {
                "data_ready_reason": "market_data_warming:1d:16/60",
                "model_ready_reason": "anchor_readiness_not_ready:rf:15m:registry_status:idle,gbm:15m:registry_status:idle",
            },
        },
    }
    (paths.runtime_dir / "symbol_state.json").write_text(json.dumps(symbol_state), encoding="utf-8")

    from scripts.retrain_live_eligible_candidates import discover_affected_anchor_jobs

    jobs = discover_affected_anchor_jobs(paths)
    keys = {(job.model_type, job.symbol, job.interval) for job in jobs}

    assert keys == {
        ("rf", "BTCUSDT", "15m"),
        ("gbm", "BTCUSDT", "15m"),
        ("rf", "ETHUSDT", "15m"),
    }


def test_discover_affected_anchor_jobs_can_include_warming_when_explicit(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    symbol_state = {
        "active_symbols": ["BILLUSDT"],
        "execution_eligible_symbols": [],
        "symbols": {
            "BILLUSDT": {
                "data_ready_reason": "market_data_warming:1d:16/60",
                "model_ready_reason": "anchor_readiness_not_ready:rf:15m:registry_status:idle,gbm:15m:registry_status:idle",
            }
        },
    }
    (paths.runtime_dir / "symbol_state.json").write_text(json.dumps(symbol_state), encoding="utf-8")

    from scripts.retrain_live_eligible_candidates import discover_affected_anchor_jobs

    assert discover_affected_anchor_jobs(paths) == []
    jobs = discover_affected_anchor_jobs(paths, include_warming=True)
    assert {(job.model_type, job.symbol, job.interval) for job in jobs} == {
        ("rf", "BILLUSDT", "15m"),
        ("gbm", "BILLUSDT", "15m"),
    }


def test_run_retraining_records_affected_anchor_scope_and_details(tmp_path: Path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    symbol_state = {
        "active_symbols": ["BTCUSDT"],
        "execution_eligible_symbols": [],
        "symbols": {
            "BTCUSDT": {
                "data_ready_reason": "market_data_ready",
                "model_ready_reason": "anchor_readiness_not_ready:rf:15m:registry_status:idle",
            }
        },
    }
    (paths.runtime_dir / "symbol_state.json").write_text(json.dumps(symbol_state), encoding="utf-8")
    monkeypatch.setattr("core.training_registry.MonitorPublisher", lambda paths: _DummyMonitor())
    monkeypatch.setattr("core.training_registry.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("scripts.retrain_live_eligible_candidates.train_rf_model", lambda paths, symbol, interval: _fake_success(paths, "rf", symbol, interval))

    from scripts.retrain_live_eligible_candidates import discover_affected_anchor_jobs

    jobs = discover_affected_anchor_jobs(paths)
    report = run_retraining(paths, jobs=jobs, retraining_scope="affected_anchor")
    payload = report.to_dict()

    assert payload["retraining_scope"] == "affected_anchor"
    assert payload["jobs_requested"] == 1
    assert payload["affected_anchor_jobs"][0]["symbol"] == "BTCUSDT"
    assert payload["affected_anchor_jobs"][0]["model_ready_reason"].startswith("anchor_readiness_not_ready")


def test_discover_affected_anchor_jobs_supports_compact_step7_blockers(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    symbol_state = {
        "active_symbols": ["HYPEUSDT", "ZECUSDT"],
        "execution_eligible_symbols": [],
        "symbols": {
            "HYPEUSDT": {
                "data_ready_reason": "market_data_ready",
                "model_ready_reason": "anchor_training_incomplete:rf:deployment_stage:candidate,gbm:feature_incompatible:feature_set_missing",
            },
            "ZECUSDT": {
                "data_ready_reason": "market_data_ready",
                "readiness_last_reason": "anchor_training_incomplete:gbm:feature_incompatible:feature_set_missing",
            },
        },
    }
    (paths.runtime_dir / "symbol_state.json").write_text(json.dumps(symbol_state), encoding="utf-8")

    from scripts.retrain_live_eligible_candidates import discover_affected_anchor_jobs

    jobs = discover_affected_anchor_jobs(paths)
    assert {(job.model_type, job.symbol, job.interval) for job in jobs} == {
        ("rf", "HYPEUSDT", "15m"),
        ("gbm", "HYPEUSDT", "15m"),
        ("gbm", "ZECUSDT", "15m"),
    }
