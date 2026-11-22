from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.portfolio_state import PortfolioStateStore
from execution.risk_engine import RiskEngine
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


from dataclasses import dataclass


@dataclass
class _DummyCircuitState:
    triggered: bool = False
    reason: str = ""


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

    def live_authority_summary(self, portfolio_state):
        return {
            "execution_mode": getattr(self.paths.config.runtime, "execution_mode", "paper"),
            "authoritative": True,
            "guard_required": False,
            "reason": "paper_local_authority",
            "reconciled": False,
        }

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


class _DummyPortfolioState:
    def __init__(self) -> None:
        self.positions = {}
        self.authority_mode = "paper_local"
        self.reconciliation_status = "paper_local"
        self.last_reconciled_at = None


class _DummyPortfolioStore:
    def __init__(self, path, *, starting_equity_usd: float) -> None:
        self.path = path
        self.starting_equity_usd = starting_equity_usd
        self.state = _DummyPortfolioState()
        self.apply_calls = 0
        self.save_calls = 0

    def load(self):
        return self.state

    def mark_authority_mode(self, state, *, execution_mode: str, reconciled: bool) -> None:
        if execution_mode == "paper":
            state.authority_mode = "paper_local"
            state.reconciliation_status = "paper_local"
            state.last_reconciled_at = None
        else:
            state.authority_mode = "live_exchange_required"
            state.reconciliation_status = "reconciled" if reconciled else "unreconciled"
            state.last_reconciled_at = 1.0 if reconciled else None

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
            "authority_mode": portfolio_state.authority_mode,
            "reconciliation_status": portfolio_state.reconciliation_status,
            "last_reconciled_at": portfolio_state.last_reconciled_at,
        }

    def save(self, portfolio_state) -> None:
        self.save_calls += 1

    def signed_positions(self, portfolio_state):
        return []


def _paths(tmp_path: Path, *, execution_mode: str = "paper", allow_unreconciled_live_mode: bool = False) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(
                market_poll_seconds=1.0,
                allow_live_refresh_fallback=False,
                execution_mode=execution_mode,
                allow_unreconciled_live_mode=allow_unreconciled_live_mode,
            ),
            risk=SimpleNamespace(
                account_balance_usd=1000.0,
                risk_per_trade=0.01,
                sl_atr_mult=1.2,
                tp_atr_mult=1.8,
                max_notional_usd=5000.0,
                min_notional_usd=25.0,
                max_total_exposure_usd=7500.0,
                max_concurrent_positions=6,
                max_symbol_notional_usd=1500.0,
                max_same_direction_exposure_usd=4500.0,
                max_risk_distortion_fraction=0.35,
                daily_loss_limit_usd=75.0,
                max_drawdown_fraction=0.12,
                max_correlated_positions=2,
                correlation_groups={},
            ),
        ),
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        tradable_symbols_path=runtime_dir / "tradable_symbols.yaml",
    )


def test_risk_engine_blocks_unreconciled_live_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path, execution_mode="live", allow_unreconciled_live_mode=False)
    engine = RiskEngine(paths)
    portfolio_state = engine.load_portfolio()
    engine.portfolio_store.mark_authority_mode(portfolio_state, execution_mode="live", reconciled=False)

    authority = engine.live_authority_summary(portfolio_state)

    assert authority["execution_mode"] == "live"
    assert authority["exchange_authority_required"] is True
    assert authority["authoritative"] is False
    assert authority["guard_required"] is True
    assert authority["reason"] == "live_reconciliation_required"


def test_risk_engine_allows_paper_local_authority(tmp_path: Path) -> None:
    paths = _paths(tmp_path, execution_mode="paper")
    engine = RiskEngine(paths)
    portfolio_state = engine.load_portfolio()
    engine.portfolio_store.mark_authority_mode(portfolio_state, execution_mode="paper", reconciled=False)

    authority = engine.live_authority_summary(portfolio_state)

    assert authority["execution_mode"] == "paper"
    assert authority["exchange_authority_required"] is False
    assert authority["authoritative"] is True
    assert authority["guard_required"] is False
    assert authority["reason"] == "paper_local_authority"


def test_live_controller_guards_unreconciled_live_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)

    controller = LiveTradingController(_paths(tmp_path, execution_mode="live", allow_unreconciled_live_mode=False), registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]
    controller.risk_engine.live_authority_summary = lambda portfolio_state: {
        "execution_mode": "live",
        "authoritative": False,
        "guard_required": True,
        "reason": "live_reconciliation_required",
        "reconciled": False,
    }

    controller.loop_once()

    assert controller.logic.calls == 0
    assert controller.risk_engine.evaluate_calls == 0
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert decisions[0]["reason"] == "live_reconciliation_required"
    assert decisions[0]["components"]["authority"]["guard_required"] is True
    assert controller.monitor.signal_updates == []
    assert controller.monitor.heartbeats[0]["status"] == "guarded"


def test_live_controller_paper_mode_runs_local_execution_path(monkeypatch, tmp_path: Path) -> None:
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

    controller = LiveTradingController(_paths(tmp_path, execution_mode="paper"), registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    assert controller.exchange_adapter.requires_exchange_authority is False

    controller.loop_once()

    assert controller.logic.calls == 1
    assert controller.risk_engine.evaluate_calls == 1
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert decisions[0]["reason"] == "test_signal"
    assert decisions[0]["components"]["authority"]["reason"] == "paper_local_authority"
    assert controller.monitor.signal_updates == []
    assert controller.monitor.heartbeats[0]["status"] == "running"
