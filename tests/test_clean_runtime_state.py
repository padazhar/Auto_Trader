from __future__ import annotations

from pathlib import Path

from scripts.clean_runtime_state import clean_runtime_state


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_clean_runtime_state_clears_market_and_training_but_preserves_runtime_evidence(tmp_path: Path) -> None:
    project = tmp_path / "Auto_Trader"
    _touch(project / "main.py", "print('ok')\n")
    _touch(project / "config" / "system_settings.yaml", "runtime:\n  execution_mode: paper\n")

    runtime_json = project / "data" / "runtime" / "portfolio_state.json"
    runtime_md = project / "data" / "runtime" / "reports" / "readiness.md"
    log_file = project / "data" / "logs" / "run.log"
    report_md = project / "data" / "reports" / "overnight.md"
    trade_file = project / "data" / "trade_history" / "trade_history.db"
    market_file = project / "data" / "market_storage" / "BTCUSDT.db"
    rf_manifest = project / "data" / "model_storage" / "random_forest_models" / "BTCUSDT" / "manifest.json"
    gbm_model = project / "data" / "model_storage" / "gradient_boost_models" / "BTCUSDT" / "model.pkl"
    generated_csv = project / "data" / "live_backtest_trades.csv"
    pycache_file = project / "execution" / "__pycache__" / "portfolio_state.cpython-312.pyc"
    pytest_cache_file = project / ".pytest_cache" / "v" / "cache" / "nodeids"

    for path in [
        runtime_json,
        runtime_md,
        log_file,
        report_md,
        trade_file,
        market_file,
        rf_manifest,
        gbm_model,
        generated_csv,
        pycache_file,
        pytest_cache_file,
    ]:
        _touch(path)

    report = clean_runtime_state(project)

    assert report.ok is True
    assert runtime_json.exists()
    assert runtime_md.exists()
    assert log_file.exists()
    assert report_md.exists()
    assert trade_file.exists()
    assert not market_file.exists()
    assert not rf_manifest.exists()
    assert not gbm_model.exists()
    assert not generated_csv.exists()
    assert not pycache_file.exists()
    assert not pytest_cache_file.exists()
    assert (project / "data" / "market_storage").is_dir()
    assert (project / "data" / "model_storage" / "random_forest_models").is_dir()
    assert (project / "data" / "runtime" / "monitoring").is_dir()
    assert (project / "data" / "logs").is_dir()


def test_clean_runtime_state_dry_run_does_not_delete(tmp_path: Path) -> None:
    project = tmp_path / "Auto_Trader"
    _touch(project / "main.py", "print('ok')\n")
    _touch(project / "config" / "system_settings.yaml", "runtime:\n  execution_mode: paper\n")
    market_file = project / "data" / "market_storage" / "BTCUSDT.db"
    model_file = project / "data" / "model_storage" / "random_forest_models" / "BTCUSDT" / "manifest.json"
    runtime_file = project / "data" / "runtime" / "portfolio_state.json"
    _touch(market_file)
    _touch(model_file)
    _touch(runtime_file)

    report = clean_runtime_state(project, dry_run=True)

    assert report.dry_run is True
    assert market_file.exists()
    assert model_file.exists()
    assert runtime_file.exists()


def test_clean_runtime_state_legacy_flags_are_accepted(tmp_path: Path) -> None:
    project = tmp_path / "Auto_Trader"
    _touch(project / "main.py", "print('ok')\n")
    _touch(project / "config" / "system_settings.yaml", "runtime:\n  execution_mode: paper\n")
    market_file = project / "data" / "market_storage" / "BTCUSDT.db"
    model_file = project / "data" / "model_storage" / "random_forest_models" / "BTCUSDT" / "manifest.json"
    _touch(market_file)
    _touch(model_file)

    report = clean_runtime_state(project, include_market_storage=True, include_model_artifacts=True)

    assert report.ok is True
    assert not market_file.exists()
    assert not model_file.exists()
