from __future__ import annotations

import threading
from types import SimpleNamespace

from core.path_manager import PathManager
from monitoring.publisher import MonitorPublisher
from runners.live_trading_controller import LiveTradingController


def test_monitor_publisher_compatibility_methods(project_root):
    runtime_paths = PathManager.from_project_root(project_root)
    monitor = MonitorPublisher(runtime_paths)
    monitor.update_market_connectivity(status="connected")
    monitor.update_market_universe(symbols=["BTCUSDT"], source="test", last_refresh_ts=123)
    monitor.update_market_result(symbol="BTCUSDT", timeframe="15m", rows_added=1, last_ts=123456, status="updated")
    monitor.update_training_runtime(queue_depth=1, inflight_count=2)
    monitor.update_scheduler(due_jobs=[{"model_type": "rf", "symbol": "BTCUSDT", "interval": "15m"}], queued_jobs=[])
    snapshot = monitor.snapshot()
    assert snapshot["market_data"]["connectivity"]["status"] == "connected"
    assert snapshot["market_data"]["universe"]["symbols"] == ["BTCUSDT"]
    assert snapshot["market_data"]["recent_results"][-1]["symbol"] == "BTCUSDT"
    assert snapshot["training_runtime"]["queue_depth"] == 1
    assert snapshot["scheduler"]["due_jobs_count"] == 1


def test_live_trading_controller_run_forever_calls_loop_once(monkeypatch, project_root):
    runtime_paths = PathManager.from_project_root(project_root)
    class DummyMonitor:
        def heartbeat(self, *args, **kwargs):
            pass
        def controller_error(self, *args, **kwargs):
            pass
        def publish_event(self, *args, **kwargs):
            pass
        def flush(self):
            pass

    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", lambda paths: DummyMonitor())
    monkeypatch.setattr(LiveTradingController, "_recover_execution_state", lambda self: None)
    monkeypatch.setattr("runners.live_trading_controller.SignalEngine", lambda paths: None)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", lambda paths: SimpleNamespace(pause_state=lambda: {"paused": False, "pause_reason": ""}, summary=lambda: {"paused": False, "pause_reason": ""}))
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", lambda *a, **k: SimpleNamespace(load=lambda: SimpleNamespace(positions={}), save=lambda state: None))
    monkeypatch.setattr("runners.live_trading_controller.BinanceFuturesReconciler", lambda paths: None)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", lambda paths: SimpleNamespace(load_circuit_breaker=lambda: None))
    monkeypatch.setattr("runners.live_trading_controller.OrderStore", lambda *a, **k: object())
    monkeypatch.setattr("runners.live_trading_controller.FillHandler", lambda store: object())
    monkeypatch.setattr("runners.live_trading_controller.PaperExchangeAdapter", lambda *a, **k: object())
    monkeypatch.setattr("runners.live_trading_controller.BinanceFuturesExchangeAdapter", lambda *a, **k: object())
    monkeypatch.setattr("runners.live_trading_controller.OrderRouter", lambda *a, **k: object())
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", lambda paths: object())

    controller = LiveTradingController(runtime_paths)
    calls = {"count": 0}
    stop_event = threading.Event()

    def fake_loop_once(self):
        calls["count"] += 1
        stop_event.set()
        return {"status": "running", "symbols": 0, "decisions": 0}

    monkeypatch.setattr(LiveTradingController, "loop_once", fake_loop_once)
    controller.run_forever(stop_event)
    assert calls["count"] == 1
