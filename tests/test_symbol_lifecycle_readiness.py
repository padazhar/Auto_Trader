from __future__ import annotations

from core.symbol_management import (
    ACTIVE,
    BLOCKED,
    default_symbol_lifecycle_entry,
    record_symbol_readiness_result_entry,
)


def test_readiness_not_ready_is_observed_before_counted() -> None:
    entry = default_symbol_lifecycle_entry()
    updated, action = record_symbol_readiness_result_entry(
        entry,
        ready=False,
        reason="readiness_not_ready:rf:15m:active_manifest_missing",
        max_readiness_failures=3,
        min_not_ready_age_sec=900.0,
        check_interval_sec=300.0,
        now_ts=1_000.0,
    )

    assert action == "observing"
    assert updated["status"] == ACTIVE
    assert updated["readiness_failures"] == 0
    assert updated["readiness_not_ready_since_ts"] == 1_000.0


def test_readiness_not_ready_blocks_only_after_repeated_mature_checks() -> None:
    entry = default_symbol_lifecycle_entry()
    entry["readiness_not_ready_since_ts"] = 1_000.0
    entry["readiness_last_checked_ts"] = 1_000.0

    updated = entry
    actions: list[str] = []
    for now_ts in (2_000.0, 2_400.0, 2_800.0):
        updated, action = record_symbol_readiness_result_entry(
            updated,
            ready=False,
            reason="readiness_not_ready:gbm:5m:active_manifest_missing",
            max_readiness_failures=3,
            min_not_ready_age_sec=900.0,
            check_interval_sec=300.0,
            now_ts=now_ts,
        )
        actions.append(action)

    assert actions == ["counted", "counted", "blocked"]
    assert updated["status"] == BLOCKED
    assert updated["last_failed_model"] == "readiness"
    assert updated["last_failed_interval"] == "all"
    assert updated["manually_blocked"] is True


def test_readiness_ready_resets_part_failure_state() -> None:
    entry = default_symbol_lifecycle_entry()
    entry.update(
        {
            "readiness_failures": 2,
            "readiness_not_ready_since_ts": 1_000.0,
            "readiness_last_checked_ts": 2_000.0,
            "readiness_last_reason": "readiness_not_ready:rf:15m:active_manifest_missing",
        }
    )

    updated, action = record_symbol_readiness_result_entry(
        entry,
        ready=True,
        reason="ready",
        max_readiness_failures=3,
        min_not_ready_age_sec=900.0,
        check_interval_sec=300.0,
        now_ts=3_000.0,
    )

    assert action == "reset"
    assert updated["status"] == ACTIVE
    assert updated["readiness_failures"] == 0
    assert updated["readiness_not_ready_since_ts"] is None
    assert updated["readiness_last_reason"] == "--"
