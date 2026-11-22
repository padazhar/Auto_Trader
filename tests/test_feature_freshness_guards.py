from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd

from inference.trade_decision_logic import TradeDecisionLogic, TradeDecision
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

    def debug(self, *args, **kwargs) -> None:
        pass


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.paths = paths
        self.inference_updates: list[dict] = []
        self.heartbeats: list[dict] = []

    def update_inference(self, **payload) -> None:
        self.inference_updates.append(dict(payload))

    def update_signal_runtime(self, **payload) -> None:
        pass

    def controller_success(self, name: str, *, details: dict) -> None:
        pass

    def controller_error(self, name: str, message: str) -> None:
        pass

    def publish_event(self, **payload) -> None:
        pass

    def heartbeat(self, name: str, *, status: str, details: dict) -> None:
        self.heartbeats.append({"name": name, "status": status, "details": dict(details)})

    def flush(self) -> None:
        pass


class _DummyPortfolioState:
    def __init__(self) -> None:
        self.positions = {}


class _DummyPortfolioStore:
    def __init__(self, path, *, starting_equity_usd: float) -> None:
        self.state = _DummyPortfolioState()
        self.starting_equity_usd = starting_equity_usd

    def load(self):
        return self.state

    def apply_execution(self, *args, **kwargs):
        raise AssertionError("stale decisions must not execute")

    def mark_to_market(self, portfolio_state, latest_prices, *, ts):
        return {
            "gross_exposure_usd": 0.0,
            "long_exposure_usd": 0.0,
            "short_exposure_usd": 0.0,
            "open_positions": 0,
            "equity_usd": self.starting_equity_usd,
        }

    def save(self, portfolio_state) -> None:
        pass

    def signed_positions(self, portfolio_state):
        return []


class _DummyCircuitState:
    triggered = False
    reason = ""


class _DummyRiskEngine:
    def __init__(self, paths) -> None:
        self.evaluate_calls = 0

    def load_circuit_breaker(self):
        return _DummyCircuitState()

    def evaluate_trade(self, *, decision, portfolio_state, ts):
        self.evaluate_calls += 1
        return SimpleNamespace(approved=True, final_action="ENTER_LONG", reason="approved", position_plan=None)

    def refresh_circuit_breaker(self, portfolio_state, *, pnl_summary, ts):
        return _DummyCircuitState()


class _ActiveControlStore:
    def __init__(self, paths) -> None:
        pass

    def pause_state(self) -> dict[str, object]:
        return {"paused": False, "pause_reason": "", "updated_at": 1.0, "updated_by": "test"}


def _logic_paths() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            inference=SimpleNamespace(
                tail_rows=100,
                minimum_models_required=2,
                quality=SimpleNamespace(min_rows=10, min_precision_macro=0.1, min_roc_auc_ovr=0.5, min_balanced_accuracy=0.34),
                weights=SimpleNamespace(rf=0.5, gbm=0.4, dqn=0.1),
                dqn=SimpleNamespace(gray_band_low=0.45, gray_band_high=0.55),
                ev=SimpleNamespace(threshold=0.0, meta_threshold=0.55, dqn_adjustment_strength=0.15, strong_anchor_probability=0.62, weak_anchor_probability=0.55, weak_anchor_lower_tf_scale=0.60),
                consensus=SimpleNamespace(enter_threshold=0.18, flip_threshold=0.12, exit_band=0.06, direction_score_threshold=0.05, require_agreement=True),
            ),
            trading=SimpleNamespace(
                fees_bps=3.0,
                slippage_bps=2.0,
                min_profit_atr_mult=0.3,
                ev_threshold=0.0,
                meta_threshold=0.55,
                min_atr_fraction=0.0,
                short_probability_threshold=0.05,
                long_probability_threshold=0.05,
            ),
            risk=SimpleNamespace(account_balance_usd=1000.0, risk_per_trade=0.01, sl_atr_mult=1.5, tp_atr_mult=2.0, max_notional_usd=5000.0, min_notional_usd=25.0),
            feature_store=SimpleNamespace(max_anchor_staleness_intervals=2, max_source_staleness_intervals=2),
        )
    )


def test_trade_decision_logic_holds_on_stale_fused_features(monkeypatch) -> None:
    paths = _logic_paths()
    old_ts = int((time.time() - 10 * 3600) * 1000)
    frame = pd.DataFrame([{"timestamp": old_ts, "15m_close": 100.0, "15m_atr_pct_14": 0.01}])
    frame.attrs["freshness"] = {
        "anchor_interval": "15m",
        "anchor_timestamp_ms": old_ts,
        "source_timestamps_ms": {"15m": old_ts},
        "context_timestamps_ms": {},
    }
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr("inference.trade_decision_logic.ModelArtifactLoader", lambda paths: object())
    logic = TradeDecisionLogic(paths)
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "HOLD"
    assert decision.reason == "stale_source_features"
    assert decision.components["freshness"]["ok"] is False


class _DummyLogic:
    def __init__(self, paths) -> None:
        pass

    def decide(self, symbol: str, interval: str, previous_position: int = 0):
        return TradeDecision(symbol=symbol, interval=interval, action="ENTER_LONG", score=0.25, reason="test_signal", components={"price": 100.0})


def test_live_controller_blocks_execution_on_stale_market_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)
    monkeypatch.setattr("features.market_loader.latest_market_timestamp", lambda *args, **kwargs: 1_700_000_000_000)
    monkeypatch.setattr("runners.live_trading_controller.validate_market_data_freshness", lambda *args, **kwargs: {"ok": False, "reason": "stale_market_data"})

    paths = SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", market_poll_seconds=1.0, allow_live_refresh_fallback=False),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            feature_store=SimpleNamespace(max_anchor_staleness_intervals=2, max_source_staleness_intervals=2),
        ),
        portfolio_state_path=tmp_path / "portfolio_state.json",
        tradable_symbols_path=tmp_path / "tradable_symbols.yaml",
        runtime_dir=tmp_path,
    )
    controller = LiveTradingController(paths, registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]
    controller.loop_once()
    assert controller.risk_engine.evaluate_calls == 0
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert decisions[0]["action"] == "HOLD"
    assert decisions[0]["reason"] == "stale_market_data"
    assert decisions[0]["components"]["freshness"]["ok"] is False


def test_live_controller_blocks_new_entry_when_order_boundary_price_is_stale(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)
    monkeypatch.setattr("features.market_loader.latest_market_timestamp", lambda *args, **kwargs: 1_700_000_000_000)
    monkeypatch.setattr("runners.live_trading_controller.validate_market_data_freshness", lambda *args, **kwargs: {"ok": True, "reason": "fresh"})
    monkeypatch.setattr(
        LiveTradingController,
        "_execution_order_price_gate",
        lambda self, symbol, action, **kwargs: {
            "required": True,
            "ok": False,
            "blocked": True,
            "symbol": symbol,
            "action": action,
            "reason": "execution_price_stale",
            "price_source": "runtime_market_mark",
            "price_age_seconds": 266.627,
            "max_age_seconds": 2.0,
        },
    )

    paths = SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", market_poll_seconds=1.0, allow_live_refresh_fallback=False),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            feature_store=SimpleNamespace(max_anchor_staleness_intervals=2, max_source_staleness_intervals=2),
        ),
        portfolio_state_path=tmp_path / "portfolio_state.json",
        tradable_symbols_path=tmp_path / "tradable_symbols.yaml",
        runtime_dir=tmp_path,
    )
    controller = LiveTradingController(paths, registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    controller.loop_once()

    assert controller.risk_engine.evaluate_calls == 0
    decisions = controller.monitor.inference_updates[0]["decisions"]
    assert decisions[0]["action"] == "HOLD"
    assert decisions[0]["reason"] == "execution_price_stale"
    assert decisions[0]["execution_status"] == "rejected_price"
    assert decisions[0]["execution_blocker_stage"] == "price"
    assert decisions[0]["components"]["execution_price_gate"]["price_age_seconds"] == 266.627


def test_live_controller_uses_final_fresh_execution_price_for_risk_plan(monkeypatch, tmp_path) -> None:
    class _CapturingRiskEngine(_DummyRiskEngine):
        def evaluate_trade(self, *, decision, portfolio_state, ts):
            self.evaluate_calls += 1
            self.observed_price = decision.components.get("price")
            return SimpleNamespace(approved=False, final_action="HOLD", reason="risk_hold", position_plan=None)

    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _CapturingRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)
    monkeypatch.setattr("features.market_loader.latest_market_timestamp", lambda *args, **kwargs: 1_700_000_000_000)
    monkeypatch.setattr("runners.live_trading_controller.validate_market_data_freshness", lambda *args, **kwargs: {"ok": True, "reason": "fresh"})
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
            "price": 101.5,
            "price_source": "runtime_market_mark",
            "price_ts_ms": 1_700_000_009_999,
            "price_age_seconds": 0.001,
            "max_age_seconds": 2.0,
        },
    )

    paths = SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", market_poll_seconds=1.0, allow_live_refresh_fallback=False),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            feature_store=SimpleNamespace(max_anchor_staleness_intervals=2, max_source_staleness_intervals=2),
        ),
        portfolio_state_path=tmp_path / "portfolio_state.json",
        tradable_symbols_path=tmp_path / "tradable_symbols.yaml",
        runtime_dir=tmp_path,
    )
    controller = LiveTradingController(paths, registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    controller.loop_once()

    assert controller.risk_engine.evaluate_calls == 1
    assert controller.risk_engine.observed_price == 101.5
    decision = controller.monitor.inference_updates[0]["decisions"][0]
    assert decision["components"]["execution_price_gate"]["reason"] == "execution_price_ready"
    assert decision["components"]["price"] == 101.5
