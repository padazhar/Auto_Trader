from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from inference.readiness import signal_eligible_symbols
from services.signal_engine import SignalEngine


class _Paths(SimpleNamespace):
    def log_file_path(self, name: str):
        return self.logs_dir / f"{name}.log"


def _build_paths(tmp_path: Path) -> _Paths:
    root = tmp_path / "repo"
    config_dir = root / "config"
    runtime_dir = root / "data" / "runtime"
    logs_dir = root / "data" / "logs"
    signal_logs = logs_dir / "signal_logs"
    for path in (config_dir, runtime_dir, logs_dir, signal_logs):
        path.mkdir(parents=True, exist_ok=True)
    (config_dir / "tradable_symbols.yaml").write_text("symbols:\n  - BTCUSDT\n  - ETHUSDT\n", encoding="utf-8")
    return _Paths(
        project_root=root,
        config_dir=config_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        tradable_symbols_path=config_dir / "tradable_symbols.yaml",
        backtest_open_trades_path=runtime_dir / "backtest_open_trades.json",
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        signal_events_log_path=signal_logs / "signal_events.log",
        monitoring_state_path=runtime_dir / "monitoring_state.json",
        monitoring_events_path=runtime_dir / "monitoring_events.json",
        config=SimpleNamespace(
            inference=SimpleNamespace(active_mode="Medium"),
            frames=SimpleNamespace(anchor_interval="15m"),
            monitoring=SimpleNamespace(
                max_events=100,
                max_alerts=50,
                max_market_updates=50,
                max_recent_training_results=50,
                max_recent_decisions=50,
                max_recent_signals=50,
                max_backtest_open_positions=50,
                max_backtest_recent_closed=50,
                max_registry_entries=100,
            ),
        ),
    )


def test_signal_eligible_symbols_filters_incomplete_anchor_symbols(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)

    def fake_anchor(paths, symbol: str, interval: str, required_models=("rf", "gbm")):
        if symbol == "BTCUSDT":
            return SimpleNamespace(ready=True, reason="anchor_ready", models={})
        return SimpleNamespace(ready=False, reason="anchor_training_incomplete:rf:registry_status:idle", models={})

    monkeypatch.setattr("inference.readiness.anchor_model_readiness", fake_anchor)
    ready, reasons = signal_eligible_symbols(paths, symbols=["BTCUSDT", "ETHUSDT"], interval="15m")
    assert ready == ["BTCUSDT"]
    assert reasons["ETHUSDT"].startswith("anchor_training_incomplete")


def test_signal_engine_load_symbols_only_returns_ready_symbols(monkeypatch, tmp_path: Path) -> None:
    paths = _build_paths(tmp_path)
    monkeypatch.setattr(
        "services.signal_engine.signal_eligible_symbols",
        lambda paths, symbols, interval: (
            ["BTCUSDT"],
            {
                "BTCUSDT": "anchor_ready",
                "ETHUSDT": "anchor_training_incomplete:rf:registry_status:idle",
            },
        ),
    )
    engine = SignalEngine(paths)
    assert engine.load_symbols() == ["BTCUSDT"]
