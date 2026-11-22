from __future__ import annotations

from types import SimpleNamespace

from core.job_definition import TrainingJob
from runners.tech_rev_controller import TechRevController


def _job(model_type: str = "rf") -> TrainingJob:
    return TrainingJob(
        model_type=model_type,
        symbol="BTCUSDT",
        interval="1m",
        priority=1,
        requested_at=1.0,
        reason="test",
    )


class _Registry:
    def __init__(self) -> None:
        self.ineligible = []
        self.failed = []

    def mark_ineligible(self, model_type, symbol, interval, reason):
        self.ineligible.append((model_type, symbol, interval, reason))

    def mark_failed(self, model_type, symbol, interval, reason):
        self.failed.append((model_type, symbol, interval, reason))

    def mark_success(self, **_kwargs):
        raise AssertionError("unexpected success")


class _Logger:
    def warning(self, *_args, **_kwargs):
        pass


def test_techrev_classifies_rf_label_support_failure_as_statistical_ineligible(monkeypatch) -> None:
    controller = object.__new__(TechRevController)
    future = object()
    job = _job("rf")
    controller.inflight = {future: job}
    controller.registry = _Registry()
    controller.logger = _Logger()
    symbol_failures = []
    controller._record_symbol_failure = lambda *_args, **_kwargs: symbol_failures.append(True)

    result = SimpleNamespace(
        ok=False,
        error="rf_valid_label_support_invalid:insufficient_class_count:1<2",
    )
    monkeypatch.setattr(
        "runners.tech_rev_controller.collect_completed_futures",
        lambda _inflight: [(future, job, result, None)],
    )

    controller._drain()

    assert len(controller.registry.ineligible) == 1
    assert controller.registry.failed == []
    assert symbol_failures == []
    assert controller.inflight == {}


def test_techrev_keeps_non_statistical_training_failure_on_existing_failure_path(monkeypatch) -> None:
    controller = object.__new__(TechRevController)
    future = object()
    job = _job("rf")
    controller.inflight = {future: job}
    controller.registry = _Registry()
    controller.logger = _Logger()
    symbol_failures = []
    controller._record_symbol_failure = lambda _job, reason: symbol_failures.append(reason)

    result = SimpleNamespace(ok=False, error="model artifact write failed")
    monkeypatch.setattr(
        "runners.tech_rev_controller.collect_completed_futures",
        lambda _inflight: [(future, job, result, None)],
    )

    controller._drain()

    assert controller.registry.ineligible == []
    assert len(controller.registry.failed) == 1
    assert symbol_failures == ["model artifact write failed"]
