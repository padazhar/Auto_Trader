from __future__ import annotations

from types import SimpleNamespace

from runners.live_trading_controller import LiveTradingController


class _DummyLogger:
    def warning(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.heartbeats: list[dict[str, object]] = []
        self.inference_updates: list[dict[str, object]] = []
        self.signal_updates: list[dict[str, object]] = []
        self.successes: list[dict[str, object]] = []
        self.errors: list[tuple[str, str]] = []
        self.events: list[dict[str, object]] = []

    def heartbeat(self, name: str, *, status: str, details: dict[str, object]) -> None:
        self.heartbeats.append({"name": name, "status": status, "details": details})

    def update_inference(self, **payload) -> None:
        self.inference_updates.append(dict(payload))

    def update_signal_runtime(self, **payload) -> None:
        self.signal_updates.append(dict(payload))

    def controller_success(self, name: str, *, details: dict[str, object]) -> None:
        self.successes.append({"name": name, "details": details})

    def controller_error(self, name: str, message: str) -> None:
        self.errors.append((name, message))

    def publish_event(self, **payload) -> None:
        self.events.append(dict(payload))

    def flush(self) -> None:
        pass


class _DummyPosition:
    def __init__(self, side: str) -> None:
        self.side = side


class _DummyPortfolioState:
    def __init__(self) -> None:
        self.positions = {}


class _DummyPortfolioStore:
    def __init__(self, path, *, starting_equity_usd: float) -> None:
        self.path = path
        self.starting_equity_usd = starting_equity_usd
        self.state = _DummyPortfolioState()
        self.apply_calls = 0
        self.save_calls = 0

    def load(self):
        return self.state

    def apply_execution(self, portfolio_state, *, symbol, action, price, ts, position_plan=None):
        self.apply_calls += 1
        return {
            "symbol": symbol,
            "action": action,
            "price": price,
            "ts": ts,
            "realized_pnl_usd": 0.0,
        }

    def mark_to_market(self, portfolio_state, latest_prices, *, ts):
        return {
            "gross_exposure_usd": 0.0,
            "long_exposure_usd": 0.0,
            "short_exposure_usd": 0.0,
            "open_positions": len(portfolio_state.positions),
            "equity_usd": self.starting_equity_usd,
        }

    def save(self, portfolio_state) -> None:
        self.save_calls += 1

    def signed_positions(self, portfolio_state):
        return []


class _DummyCircuitState:
    def __init__(self) -> None:
        self.triggered = False
        self.reason = ""


class _DummyRiskDecision:
    def __init__(self, *, approved: bool, final_action: str, reason: str) -> None:
        self.approved = approved
        self.final_action = final_action
        self.reason = reason
        self.position_plan = None
        self.components = {
            "approved": approved,
            "final_action": final_action,
            "reason": reason,
        }


class _DummyRiskEngine:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.evaluate_calls = 0
        self.refresh_calls = 0

    def load_circuit_breaker(self):
        return _DummyCircuitState()

    def evaluate_trade(self, *, decision, portfolio_state, ts):
        self.evaluate_calls += 1
        return _DummyRiskDecision(approved=False, final_action="HOLD", reason="risk_hold")

    def refresh_circuit_breaker(self, portfolio_state, *, pnl_summary, ts):
        self.refresh_calls += 1
        return _DummyCircuitState()


class _DummyLogicDecision:
    def __init__(self, symbol: str, interval: str) -> None:
        self.symbol = symbol
        self.interval = interval
        self.action = "ENTER_LONG"
        self.score = 0.25
        self.reason = "test_signal"
        self.components = {"price": 100.0}


class _DummyLogic:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.calls = 0

    def decide(self, symbol: str, interval: str, previous_position: int = 0):
        self.calls += 1
        return _DummyLogicDecision(symbol, interval)


class _PausedControlStore:
    def __init__(self, paths) -> None:
        self.paths = paths

    def pause_state(self) -> dict[str, object]:
        return {
            "paused": True,
            "pause_reason": "manual_pause",
            "updated_at": 1.0,
            "updated_by": "test",
        }


class _ActiveControlStore:
    def __init__(self, paths) -> None:
        self.paths = paths

    def pause_state(self) -> dict[str, object]:
        return {
            "paused": False,
            "pause_reason": "",
            "updated_at": 1.0,
            "updated_by": "test",
        }


def _paths() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(market_poll_seconds=1.0, allow_live_refresh_fallback=False),
            risk=SimpleNamespace(account_balance_usd=1000.0),
        ),
        portfolio_state_path="portfolio_state.json",
        tradable_symbols_path="tradable_symbols.yaml",
    )


def test_live_controller_pause_blocks_inference_and_risk(monkeypatch) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _PausedControlStore)

    controller = LiveTradingController(_paths(), registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    controller.loop_once()

    assert controller.logic.calls == 0
    assert controller.risk_engine.evaluate_calls == 0
    assert len(controller.monitor.inference_updates) == 1
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "HOLD"
    assert decisions[0]["effective_action"] == "HOLD"
    assert decisions[0]["reason"] == "control_plane_paused"
    assert decisions[0]["approved"] is False
    assert decisions[0]["components"]["control_plane"]["paused"] is True
    assert controller.monitor.heartbeats[0]["status"] == "paused"


def test_live_controller_unpaused_still_executes_normal_path(monkeypatch) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)
    monkeypatch.setattr(
        LiveTradingController,
        "_execution_order_price_gate",
        lambda self, symbol, action, **kwargs: {
            "required": True,
            "ok": True,
            "blocked": False,
            "symbol": symbol,
            "action": action,
            "reason": "execution_price_ready",
            "price": 100.0,
            "price_source": "runtime_market_mark",
            "price_ts_ms": 1_700_000_000_000,
            "price_age_seconds": 0.001,
            "max_age_seconds": 2.0,
        },
    )

    controller = LiveTradingController(_paths(), registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    controller.loop_once()

    assert controller.logic.calls == 1
    assert controller.risk_engine.evaluate_calls == 1
    assert len(controller.monitor.inference_updates) == 1
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["reason"] == "test_signal"
    assert decisions[0]["components"]["control_plane"]["paused"] is False
    assert controller.monitor.heartbeats[0]["status"] == "running"
