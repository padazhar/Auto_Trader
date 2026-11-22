from __future__ import annotations

from core.symbol_management import (
    EXECUTION_ACTIVE,
    TRAINING_REQUIRED,
    WARMING_DATA,
    classify_symbol_universe,
    default_symbol_lifecycle_entry,
    mark_symbol_warming_entry,
    record_symbol_data_readiness_entry,
    record_symbol_execution_readiness_entry,
)


def test_promoted_symbol_enters_warming_until_data_is_ready() -> None:
    entry = mark_symbol_warming_entry(
        default_symbol_lifecycle_entry(),
        now_ts=100.0,
        reason="promoted_symbol_data_warmup_required",
    )

    assert entry["activation_stage"] == WARMING_DATA
    assert entry["data_ready"] is False
    assert entry["execution_eligible"] is False
    assert entry["warmup_requested_ts"] == 100.0

    entry, action = record_symbol_data_readiness_entry(
        entry,
        ready=False,
        reason="market_data_warming:15m:0/580",
        now_ts=120.0,
    )
    assert action == "waiting"
    assert entry["activation_stage"] == WARMING_DATA
    assert entry["execution_eligible"] is False


def test_data_ready_symbol_waits_for_model_readiness_before_execution_active() -> None:
    entry = mark_symbol_warming_entry(default_symbol_lifecycle_entry(), now_ts=100.0)
    entry, action = record_symbol_data_readiness_entry(entry, ready=True, reason="market_data_ready", now_ts=200.0)

    assert action == "training_required"
    assert entry["activation_stage"] == TRAINING_REQUIRED
    assert entry["data_ready"] is True
    assert entry["execution_eligible"] is False

    entry, action = record_symbol_execution_readiness_entry(
        entry,
        ready=False,
        reason="readiness_not_ready:rf:15m:registry_status:idle",
        now_ts=250.0,
    )
    assert action == "training_required"
    assert entry["activation_stage"] == TRAINING_REQUIRED
    assert entry["execution_eligible"] is False

    entry, action = record_symbol_execution_readiness_entry(entry, ready=True, reason="ready", now_ts=300.0)
    assert action == "execution_active"
    assert entry["activation_stage"] == EXECUTION_ACTIVE
    assert entry["execution_eligible"] is True


def test_universe_buckets_separate_warming_from_execution_eligible() -> None:
    warming = mark_symbol_warming_entry(default_symbol_lifecycle_entry(), now_ts=100.0)
    ready = default_symbol_lifecycle_entry()
    ready["activation_stage"] = EXECUTION_ACTIVE
    ready["execution_eligible"] = True
    state = {"symbols": {"BTCUSDT": ready, "ETHUSDT": warming}}

    buckets = classify_symbol_universe(
        ranked_candidates=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        active_symbols=["BTCUSDT", "ETHUSDT"],
        state=state,
        ranked_universe_size=20,
    )

    assert buckets["active_symbols"] == ["BTCUSDT", "ETHUSDT"]
    assert buckets["warming_symbols"] == ["ETHUSDT"]
    assert buckets["execution_eligible_symbols"] == ["BTCUSDT"]
    assert buckets["candidate_symbols"] == ["SOLUSDT"]

from core.symbol_management import BLOCKED, record_symbol_warmup_timeout_entry, execution_eligible_symbols


def test_default_execution_active_stage_is_not_enough_for_execution_eligibility() -> None:
    entry = default_symbol_lifecycle_entry()
    entry["activation_stage"] = EXECUTION_ACTIVE
    entry["execution_eligible"] = False
    state = {"active_symbols": ["BTCUSDT"], "symbols": {"BTCUSDT": entry}}

    buckets = classify_symbol_universe(
        ranked_candidates=["BTCUSDT"],
        active_symbols=["BTCUSDT"],
        state=state,
        ranked_universe_size=20,
    )

    assert buckets["execution_eligible_symbols"] == []
    assert execution_eligible_symbols(state) == []


def test_warming_symbol_blocks_only_when_explicit_timeout_is_recorded() -> None:
    entry = mark_symbol_warming_entry(default_symbol_lifecycle_entry(), now_ts=100.0)
    blocked = record_symbol_warmup_timeout_entry(
        entry,
        reason="warmup_timeout:data_not_ready_after_14400s",
        now_ts=14_600.0,
    )

    assert blocked["status"] == BLOCKED
    assert blocked["execution_eligible"] is False
    assert blocked["last_failed_model"] == "warmup"
    assert blocked["blocked_position_close_required"] is True


def test_warmup_readiness_summary_reports_active_warming_symbols_only() -> None:
    from core.symbol_management import warmup_readiness_summary

    warming = mark_symbol_warming_entry(default_symbol_lifecycle_entry(), now_ts=100.0)
    warming, _ = record_symbol_data_readiness_entry(
        warming,
        ready=False,
        reason="market_data_warming:1d:16/60",
        now_ts=120.0,
    )
    ready = default_symbol_lifecycle_entry()
    ready["activation_stage"] = EXECUTION_ACTIVE
    ready["execution_eligible"] = True
    inactive_warming = mark_symbol_warming_entry(default_symbol_lifecycle_entry(), now_ts=100.0)
    state = {
        "active_symbols": ["BILLUSDT", "BTCUSDT"],
        "symbols": {
            "BILLUSDT": warming,
            "BTCUSDT": ready,
            "OLDUSDT": inactive_warming,
        },
    }

    summary = warmup_readiness_summary(state)

    assert summary["schema_version"] == "symbol_warmup_readiness.v1"
    assert summary["active_symbol_count"] == 2
    assert summary["warming_symbol_count"] == 1
    assert summary["warming_symbols"] == ["BILLUSDT"]
    assert summary["rows"][0]["data_ready_reason"] == "market_data_warming:1d:16/60"
    assert summary["rows"][0]["execution_eligible"] is False
