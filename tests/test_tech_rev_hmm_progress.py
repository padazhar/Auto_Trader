from __future__ import annotations

from types import SimpleNamespace

from core.job_definition import TrainingJob
from runners.tech_rev_controller import TechRevController
from services.training_engine import TrainingProgressSnapshot


class _FakeMonitor:
    def __init__(self) -> None:
        self.training_payload = {}

    def update_training_runtime(self, **payload):
        self.training_payload = dict(payload)


def _job(model_type: str, symbol: str = "BTCUSDT", interval: str = "15m") -> TrainingJob:
    return TrainingJob(
        model_type=model_type,
        symbol=symbol,
        interval=interval,
        priority=1,
        requested_at=1.0,
        reason="test",
    )


def test_techrev_snapshot_tracks_hmm_progress_without_policy_slot(monkeypatch) -> None:
    controller = object.__new__(TechRevController)
    controller.snapshot = TrainingProgressSnapshot()
    controller.inflight = {object(): _job("hmm")}
    controller.engine = SimpleNamespace(read_progress_payload=lambda *_args, **_kwargs: {"percent": 42.0, "message": "hmm fitting"})
    controller.monitor = _FakeMonitor()
    controller.max_failures_before_block = 3
    controller.symbol_backoff_first_sec = 10
    controller.symbol_backoff_repeat_sec = 20
    controller.target_active_symbols = 10
    controller.ranked_universe_size = 20
    controller.auto_replace_symbols = True
    controller._timed_out_symbols_payload = lambda *_args, **_kwargs: []
    controller.paths = SimpleNamespace()
    controller.registry = SimpleNamespace()

    monkeypatch.setattr("runners.tech_rev_controller.load_symbol_state", lambda _paths: {"active_symbols": ["BTCUSDT"]})
    monkeypatch.setattr(
        "runners.tech_rev_controller.build_training_backlog_diagnostics",
        lambda *_args, **_kwargs: {"counts": {}, "rows": [], "total_rows": 0, "truncated": False},
    )

    controller._build_snapshot(due_jobs=[_job("hmm", "ETHUSDT", "5m")], blocked_jobs=[])

    payload = controller.monitor.training_payload
    assert payload["hmm_pending"] == 1
    assert payload["current_hmm"] == "BTCUSDT 15m"
    assert payload["hmm_progress"] == 42.0
    assert payload["last_hmm_message"] == "hmm fitting"
    assert payload["current_dqn"] == "--"
    assert payload["dqn_progress"] == 0.0
