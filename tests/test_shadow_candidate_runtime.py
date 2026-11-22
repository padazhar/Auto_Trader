from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from runners.live_trading_controller import LiveTradingController
from monitoring.control_plane_store import ControlPlaneStore


class _DummyLogger:
    def warning(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def exception(self, *args, **kwargs):
        pass


class _DummyMonitor:
    def __init__(self, paths) -> None:
        self.inference_updates = []
        self.signal_updates = []
        self.heartbeats = []
        self.events = []
    def update_inference(self, **payload):
        self.inference_updates.append(dict(payload))
    def update_signal_runtime(self, **payload):
        self.signal_updates.append(dict(payload))
    def heartbeat(self, name: str, *, status: str, details: dict[str, object]):
        self.heartbeats.append({"name": name, "status": status, "details": details})
    def publish_event(self, *args, **kwargs):
        self.events.append({"args": args, "kwargs": kwargs})
    def update_control_plane(self, payload):
        pass
    def update_model_versions(self, payload):
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
    def save(self, state):
        return None
    def mark_to_market(self, portfolio_state, latest_prices, *, ts):
        return {"equity_usd": self.starting_equity_usd, "open_positions": 0}
    def apply_execution(self, *args, **kwargs):
        return {}


class _DummyCircuitState:
    triggered = False
    reason = ""


class _DummyRiskDecision:
    approved = False
    final_action = "HOLD"
    reason = "risk_hold"
    position_plan = None


class _DummyRiskEngine:
    def __init__(self, paths):
        self.paths = paths
    def load_circuit_breaker(self):
        return _DummyCircuitState()
    def evaluate_trade(self, *, decision, portfolio_state, ts):
        return _DummyRiskDecision()
    def refresh_circuit_breaker(self, portfolio_state, *, pnl_summary, ts):
        return _DummyCircuitState()
    def live_authority_summary(self, portfolio_state):
        return {"execution_mode": "paper", "authoritative": True, "guard_required": False, "reason": "paper_local_authority", "reconciled": False}


class _DummyDecision:
    def __init__(self, symbol: str, interval: str, action: str, reason: str, score: float):
        self.symbol = symbol
        self.interval = interval
        self.action = action
        self.reason = reason
        self.score = score
        self.components = {"price": 100.0, "lineage": {"deployment_stage": "candidate_shadow" if reason.startswith("candidate") else "incumbent"}}


class _DummyLogic:
    def __init__(self, paths):
        self.paths = paths
    def decide(self, symbol: str, interval: str, previous_position: int = 0, *, use_candidate_artifacts: bool = False):
        if use_candidate_artifacts:
            return _DummyDecision(symbol, interval, "ENTER_SHORT", "candidate_shadow_signal", 0.31)
        return _DummyDecision(symbol, interval, "ENTER_LONG", "incumbent_signal", 0.25)


class _ActiveControlStore:
    def __init__(self, paths):
        self.paths = paths
    def pause_state(self):
        return {"paused": False, "pause_reason": "", "updated_at": 1.0, "updated_by": "test"}
    def summary(self):
        return {"paused": False, "pause_reason": "", "runtime_gate": {}}


def _paths(tmp_path: Path) -> SimpleNamespace:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=tmp_path,
        runtime_dir=runtime_dir,
        decision_journal_path=runtime_dir / "decision_journal.jsonl",
        order_state_path=runtime_dir / "order_state.json",
        execution_journal_path=runtime_dir / "execution_journal.jsonl",
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        control_plane_state_path=runtime_dir / "control_plane_state.json",
        control_plane_actions_path=runtime_dir / "control_plane_actions.jsonl",
        circuit_breaker_state_path=runtime_dir / "circuit_breaker_state.json",
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", allow_live_refresh_fallback=False),
            risk=SimpleNamespace(account_balance_usd=1000.0),
        ),
    )


def test_live_controller_records_shadow_candidate_payload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("runners.live_trading_controller.get_logger", lambda name, paths: _DummyLogger())
    monkeypatch.setattr("runners.live_trading_controller.MonitorPublisher", _DummyMonitor)
    monkeypatch.setattr("runners.live_trading_controller.TradeDecisionLogic", _DummyLogic)
    monkeypatch.setattr("runners.live_trading_controller.SignalEngine", lambda paths: (_ for _ in ()).throw(RuntimeError("disabled")))
    monkeypatch.setattr("runners.live_trading_controller.RiskEngine", _DummyRiskEngine)
    monkeypatch.setattr("runners.live_trading_controller.PortfolioStateStore", _DummyPortfolioStore)
    monkeypatch.setattr("runners.live_trading_controller.ControlPlaneStore", _ActiveControlStore)

    controller = LiveTradingController(_paths(tmp_path), registry=SimpleNamespace())
    controller.load_symbols = lambda: ["BTCUSDT"]

    controller.loop_once()

    assert controller.monitor.signal_updates == []
    decision = controller.monitor.inference_updates[0]["decisions"][0]
    shadow = decision["components"]["shadow_candidate"]
    assert shadow["available"] is True
    assert shadow["action"] == "ENTER_SHORT"
    assert shadow["different_action"] is True

    journal_rows = [json.loads(line) for line in controller.paths.decision_journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert journal_rows
    assert journal_rows[0]["decision_stage"] == "execution"
    assert journal_rows[0]["journal_writer"] == "live_trading_controller"
    assert journal_rows[0]["components"]["shadow_candidate"]["available"] is True


def test_control_plane_summary_exposes_shadow_candidate_preview(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    row = {
        "ts": 1.0,
        "symbol": "BTCUSDT",
        "interval": "15m",
        "action": "ENTER_LONG",
        "reason": "incumbent_signal",
        "components": {
            "shadow_candidate": {
                "available": True,
                "action": "ENTER_SHORT",
                "reason": "candidate_shadow_signal",
                "score": 0.31,
                "different_action": True,
                "lineage": {"deployment_stage": "candidate_shadow"},
            }
        },
    }
    paths.decision_journal_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    store = ControlPlaneStore(paths)
    summary = store.summary()
    preview = summary["shadow_candidate_preview"]
    assert preview
    assert preview[0]["candidate_available"] is True
    assert preview[0]["different_action"] is True
