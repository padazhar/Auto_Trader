"""Step 1 — Runtime-contract regression tripwires (FINAL Execution Plan).

PURPOSE
-------
Lock the runtime contracts that the 21.4h forensic audit proved are CURRENTLY
PASSING in steady state, so that the subsequent governance/data edits (Step 2
HMM-gate honesty, Step 3 single outcome-source authority, Step 4 websocket
stability) cannot silently regress them.

This module is intentionally ADDITIVE and self-contained. It does not modify or
refactor any existing test or production module.

REGIME AWARENESS (correction after Step-1 validation)
-----------------------------------------------------
The contracts being guarded are *steady-state, post-warmup* invariants (a fully
warmed run: executed orders exist; all active symbols anchor-ready). A freshly
booted or just-rotated universe legitimately has no executed orders yet and
symbols whose anchors are not trained — that is correct fail-closed behaviour,
not a regression. Each guard therefore validates its invariant only when the
system is in the regime that invariant describes, and SKIPS otherwise. In every
regime the guards still assert the dangerous *false-green* and *internal
consistency* conditions, which must never hold regardless of warmup state.

CONTRACTS GUARDED
-----------------
A. Order-boundary price freshness: no EXECUTED decision may be filled on a stale
   price. (Vacuously satisfied, and skipped, when there are no executed orders.)
B. Readiness truth: the system must never be false-green (ready while blockers
   exist; eligible greater than active). When the snapshot reports itself READY,
   the full steady-state lock applies (eligible == active, zero blockers).
C. Feature-contract parity: every execution-ready model shares one feature-set
   version/signature family.
D. Execution price-gate configuration: the order-boundary freshness threshold
   must not be relaxed (runtime.execution_price_max_age_seconds == 2).

These are guards, not fixes. They assert invariants only; they change no
production behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "data" / "runtime"
DECISION_JOURNAL = RUNTIME / "decision_journal.jsonl"
MONITORING_STATE = RUNTIME / "monitoring" / "monitoring_state.json"
TRAINING_REGISTRY = RUNTIME / "training_registry.json"
SYSTEM_SETTINGS = ROOT / "config" / "system_settings.yaml"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require(path: Path) -> None:
    """Skip the guard when its runtime evidence artifact is not present."""
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip(f"runtime artifact not present for tripwire: {path}")


def _iter_executed_decisions(path: Path) -> Iterator[dict[str, Any]]:
    """Yield only decision records that were approved for an order.

    A cheap substring pre-filter avoids JSON-parsing the full journal: only
    lines that contain an approved_for_order=true marker are decoded.
    """
    markers = ('"approved_for_order": true', '"approved_for_order":true')
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if not any(marker in line for marker in markers):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A malformed journal line is itself a regression; surface it.
                raise AssertionError("decision_journal contains a non-JSON line")
            if record.get("approved_for_order") is True:
                yield record


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------- #
# Tripwire A — order-boundary price freshness on every executed order
# --------------------------------------------------------------------------- #
def test_tripwire_no_executed_order_filled_on_stale_price() -> None:
    """No approved/executed decision may carry a stale price at routing time.

    Locks the order-boundary freshness contract. A violation means new exposure
    was opened on stale alpha — a CRITICAL execution-safety regression. When the
    bundle contains no executed orders (boot/smoke run), the contract is
    vacuously satisfied and the guard skips rather than false-failing.
    """
    _require(DECISION_JOURNAL)

    executed = 0
    stale_fresh_flag = 0
    age_over_gate = 0
    worst_age = 0.0

    for record in _iter_executed_decisions(DECISION_JOURNAL):
        executed += 1
        components = record.get("components") or {}
        price_fresh = components.get("price_fresh")
        price_age = components.get("price_age_seconds")
        max_age = components.get("price_max_age_seconds")

        if price_fresh is False:
            stale_fresh_flag += 1
        if isinstance(price_age, (int, float)):
            worst_age = max(worst_age, float(price_age))
            if isinstance(max_age, (int, float)) and price_age > max_age:
                age_over_gate += 1

    if executed == 0:
        pytest.skip("no executed orders in this bundle; freshness contract vacuously satisfied")

    assert stale_fresh_flag == 0, (
        f"{stale_fresh_flag} executed orders filled with price_fresh=False "
        "(order-boundary freshness contract violated)"
    )
    assert age_over_gate == 0, (
        f"{age_over_gate} executed orders exceeded price_max_age_seconds "
        f"(worst observed age {worst_age:.3f}s)"
    )


# --------------------------------------------------------------------------- #
# Tripwire B — readiness truth: never false-green; steady-state lock when ready
# --------------------------------------------------------------------------- #
def test_tripwire_readiness_truth_consistency() -> None:
    """Readiness must never be false-green, and must fully lock when ready.

    Regime-independent (always asserted):
      * the snapshot must not report READY while carrying blockers;
      * execution_eligible_count must not exceed active_count.
    Steady-state (asserted only when the snapshot reports READY):
      * zero blocked symbols, empty blocker list, all blocker layers zero,
        and execution_eligible_count == active_count.

    A non-ready snapshot (cold boot / just-rotated universe) is honest
    fail-closed behaviour, so the steady-state lock is skipped, but the
    false-green consistency checks still apply.
    """
    _require(MONITORING_STATE)
    state = _load_json(MONITORING_STATE)
    readiness = state.get("trading_readiness") or {}

    active_count = readiness.get("active_count")
    eligible = readiness.get("execution_eligible_count")
    blocked = readiness.get("blocked_symbol_count")
    blockers = readiness.get("blockers")
    by_layer = readiness.get("blocker_summary_by_layer") or {}
    ready = readiness.get("ready")
    status = str(readiness.get("status") or "").lower()

    has_blockers = bool(blockers) or bool(
        {k: v for k, v in by_layer.items() if isinstance(v, (int, float)) and v}
    )

    # ---- Regime-independent: never false-green --------------------------- #
    is_ready = (ready is True) or (status == "ready")
    assert not (is_ready and has_blockers), (
        "false-green readiness: snapshot reports READY while blockers are present "
        f"(blockers={blockers}, layers={by_layer})"
    )
    if isinstance(eligible, int) and isinstance(active_count, int):
        assert eligible <= active_count, (
            f"execution_eligible_count ({eligible}) exceeds active_count ({active_count})"
        )

    # ---- Steady-state lock: only when the system claims readiness -------- #
    if not is_ready:
        pytest.skip(f"readiness not in steady ready state (ready={ready}, status={status})")

    nonzero_layers = {k: v for k, v in by_layer.items() if isinstance(v, (int, float)) and v}
    assert blocked == 0, f"ready snapshot reports {blocked} blocked symbols"
    assert blockers in ([], None), f"ready snapshot has non-empty blocker list: {blockers}"
    assert not nonzero_layers, f"ready snapshot has non-zero blocker layers: {nonzero_layers}"
    if isinstance(eligible, int) and isinstance(active_count, int):
        assert eligible == active_count, (
            f"ready snapshot: execution_eligible_count ({eligible}) != active_count ({active_count})"
        )


# --------------------------------------------------------------------------- #
# Tripwire C — feature-contract parity across execution-ready models
# --------------------------------------------------------------------------- #
def test_tripwire_execution_ready_models_share_single_feature_version() -> None:
    """Every execution-ready model must share one feature-set version family.

    Locks feature-contract integrity. A second feature_set_version appearing on
    an execution-ready model indicates training/inference skew risk. Skips when
    no execution-ready models exist yet (cold bundle).
    """
    _require(TRAINING_REGISTRY)
    registry = _load_json(TRAINING_REGISTRY)
    assert isinstance(registry, dict) and registry, "training_registry is empty or malformed"

    versions: set[str] = set()
    missing_signature: list[str] = []

    for key, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("execution_ready") is not True:
            continue
        version = entry.get("feature_set_version")
        signature = entry.get("feature_set_signature")
        if version:
            versions.add(version)
        if not signature:
            missing_signature.append(key)

    if not versions:
        pytest.skip("no execution-ready models in this bundle; nothing to lock")

    assert len(versions) == 1, (
        f"execution-ready models span multiple feature_set_versions: {sorted(versions)}"
    )
    assert not missing_signature, (
        f"execution-ready models missing feature_set_signature: {missing_signature[:5]}"
    )


# --------------------------------------------------------------------------- #
# Tripwire D — execution price-gate threshold must not be relaxed (config-only)
# --------------------------------------------------------------------------- #
def test_tripwire_execution_price_gate_threshold_not_relaxed() -> None:
    """The order-boundary freshness threshold must remain at 2 seconds.

    Pure configuration invariant (always runnable). Guards Step 4 websocket work
    from weakening the fail-closed order gate. The audit and config both record
    runtime.execution_price_max_age_seconds == 2.
    """
    _require(SYSTEM_SETTINGS)
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(SYSTEM_SETTINGS.read_text(encoding="utf-8"))

    runtime_cfg = (config or {}).get("runtime") or {}
    max_age = runtime_cfg.get("execution_price_max_age_seconds")

    assert max_age is not None, "runtime.execution_price_max_age_seconds is missing from config"
    assert float(max_age) == 2.0, (
        f"execution_price_max_age_seconds was changed to {max_age}; "
        "the fail-closed order-boundary freshness gate must remain at 2s"
    )
