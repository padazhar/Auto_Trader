from core.symbol_management import (
    BLOCKED,
    ACTIVE,
    TIMED_OUT,
    clear_expired_symbol_backoff_entry,
    compute_active_symbols,
    record_symbol_failure_entry,
    record_symbol_success_entry,
    request_symbol_retry_entry,
)


def test_symbol_failure_progresses_to_backoff_then_blocked():
    entry, action = record_symbol_failure_entry(
        {},
        model_type="rf",
        interval="15m",
        reason="timeout",
        max_failures_before_block=3,
        backoff_first_sec=60,
        backoff_repeat_sec=120,
        now_ts=1000.0,
    )
    assert action == "timed_out"
    assert entry["status"] == TIMED_OUT
    assert entry["consecutive_failures"] == 1
    assert entry["retry_not_before_ts"] == 1060.0

    entry, action = record_symbol_failure_entry(
        entry,
        model_type="gbm",
        interval="5m",
        reason="missing_artifact",
        max_failures_before_block=3,
        backoff_first_sec=60,
        backoff_repeat_sec=120,
        now_ts=1100.0,
    )
    assert action == "timed_out"
    assert entry["status"] == TIMED_OUT
    assert entry["consecutive_failures"] == 2
    assert entry["retry_not_before_ts"] == 1220.0

    entry, action = record_symbol_failure_entry(
        entry,
        model_type="dqn",
        interval="5m",
        reason="part_not_ready",
        max_failures_before_block=3,
        backoff_first_sec=60,
        backoff_repeat_sec=120,
        now_ts=1300.0,
    )
    assert action == "blocked"
    assert entry["status"] == BLOCKED
    assert entry["retry_not_before_ts"] is None
    assert entry["eligible_for_auto_retry"] is False


def test_manual_retry_then_success_resets_lifecycle_entry():
    blocked_entry = {"status": BLOCKED, "consecutive_failures": 3, "last_error": "failed"}
    retry_entry = request_symbol_retry_entry(blocked_entry)
    assert retry_entry["status"] == ACTIVE
    assert retry_entry["retry_requested"] is True

    success_entry = record_symbol_success_entry(retry_entry)
    assert success_entry["status"] == ACTIVE
    assert success_entry["consecutive_failures"] == 0
    assert success_entry["retry_requested"] is False
    assert success_entry["last_error"] == "--"


def test_expired_backoff_reactivates_symbol():
    entry = {"status": TIMED_OUT, "retry_not_before_ts": 1000.0, "eligible_for_auto_retry": True}
    updated, changed = clear_expired_symbol_backoff_entry(entry, now_ts=1001.0)
    assert changed is True
    assert updated["status"] == ACTIVE
    assert updated["retry_not_before_ts"] is None


def test_compute_active_symbols_excludes_blocked_and_fills_from_ranked_candidates():
    state = {"symbols": {"ETHUSDT": {"status": BLOCKED}}}
    active = compute_active_symbols(
        ranked_candidates=["ETHUSDT", "SOLUSDT", "BNBUSDT"],
        target_active=3,
        pinned_symbols=["BTCUSDT"],
        skip_symbols=set(),
        state=state,
        current_active_symbols=["ETHUSDT", "ADAUSDT"],
    )
    # Step 4: once a ranked universe exists, active symbols must come only
    # from that bounded ranked pool. Pinned/current symbols outside the pool
    # cannot leak into the execution-active universe.
    assert active == ["SOLUSDT", "BNBUSDT"]


def test_compute_active_symbols_maintains_exact_limit_from_top_twenty_replacements():
    ranked = [f"SYM{i}USDT" for i in range(1, 21)] + ["SYM21USDT"]
    state = {
        "symbols": {
            "SYM2USDT": {"status": BLOCKED},
            "SYM4USDT": {"status": BLOCKED},
        }
    }
    active = compute_active_symbols(
        ranked_candidates=ranked,
        target_active=10,
        ranked_universe_size=20,
        pinned_symbols=[],
        skip_symbols=set(),
        state=state,
        current_active_symbols=ranked[:10],
    )
    assert len(active) == 10
    assert "SYM2USDT" not in active
    assert "SYM4USDT" not in active
    assert active[-2:] == ["SYM11USDT", "SYM12USDT"]
    assert "SYM21USDT" not in active

from types import SimpleNamespace
from pathlib import Path
import yaml

from core.symbol_management import (
    EXECUTION_ACTIVE,
    TRAINING_REQUIRED,
    reconcile_symbol_lifecycle_with_readiness_snapshot,
    write_tradable_symbols_yaml,
)


def test_reconcile_lifecycle_execution_eligibility_from_signal_readiness() -> None:
    ready_entry = record_symbol_success_entry({})
    ready_entry["data_ready"] = True
    ready_entry["activation_stage"] = EXECUTION_ACTIVE
    ready_entry["execution_eligible"] = True
    stale_entry = record_symbol_success_entry({})
    stale_entry["data_ready"] = True
    stale_entry["activation_stage"] = EXECUTION_ACTIVE
    stale_entry["execution_eligible"] = True
    state = {
        "active_symbols": ["BTCUSDT", "ETHUSDT"],
        "execution_eligible_symbols": ["BTCUSDT", "ETHUSDT"],
        "symbols": {"BTCUSDT": ready_entry, "ETHUSDT": stale_entry},
    }
    readiness = {
        "execution_readiness_matrix": [
            {"symbol": "BTCUSDT", "ready": True, "reason": "anchor_ready"},
            {"symbol": "ETHUSDT", "ready": False, "reason": "anchor_training_incomplete:gbm:registry_status:idle"},
        ]
    }

    reconciled, changed = reconcile_symbol_lifecycle_with_readiness_snapshot(state, readiness, now_ts=500.0)

    assert changed is True
    assert reconciled["execution_eligible_symbols"] == ["BTCUSDT"]
    assert reconciled["symbols"]["BTCUSDT"]["execution_eligible"] is True
    assert reconciled["symbols"]["ETHUSDT"]["execution_eligible"] is False
    assert reconciled["symbols"]["ETHUSDT"]["activation_stage"] == TRAINING_REQUIRED
    assert reconciled["symbols"]["ETHUSDT"]["model_ready_reason"] == "anchor_training_incomplete:gbm:registry_status:idle"


def test_write_tradable_symbols_yaml_uses_unique_atomic_temp_path(tmp_path: Path) -> None:
    target = tmp_path / "tradable_symbols.yaml"
    stale_fixed_tmp = tmp_path / "tradable_symbols.tmp"
    stale_fixed_tmp.write_text("stale", encoding="utf-8")
    paths = SimpleNamespace(tradable_symbols_path=target)

    write_tradable_symbols_yaml(
        paths,
        symbols=["BTCUSDT", "ETHUSDT", "BTCUSDT"],
        pinned_symbols=["BTCUSDT"],
        skip_symbols={"ETHUSDT"},
        source="test_export",
    )

    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["symbols"] == ["BTCUSDT"]
    assert payload["pinned_symbols"] == ["BTCUSDT"]
    assert payload["skip_symbols"] == ["ETHUSDT"]
    assert stale_fixed_tmp.read_text(encoding="utf-8") == "stale"
