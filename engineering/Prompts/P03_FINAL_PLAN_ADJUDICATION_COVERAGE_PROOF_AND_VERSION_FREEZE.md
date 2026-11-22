> **Required attachment:** `GLOBAL_CONTRACT.md` plus the current repository, audit, approved plan, and latest handoff artifacts. This prompt inherits the entire Global Contract.

# P03 — FINAL PLAN ADJUDICATION, COVERAGE PROOF, AND VERSION FREEZE

**Mode: READ-ONLY FINAL PLAN GATE. No implementation.**

Read and compare P01 forensic output, P02 Planner and Auditor reports, current repository and the validated historical S00–S26 plan. Challenge all three roles again: **Does each of the 24 original findings and every NEW finding have a planned corrective step or a precisely scoped diagnostic gate? Are all 29 dimensions mapped to an actual measurable observation, test or consciously unresolved scientific question?** Produce automated coverage counts and list every missing row; counts are not proof of remediation. Resolve any conflict by code/runtime evidence or label it BLOCKED_BY_EVIDENCE, rather than making the Auditor silently agree.

Publish a single **versioned implementation plan**, with:

- One step per smallest independently testable contract; unique `Snn` IDs and strict merge/test order. For all steps specify actual files and possible new files, main change, linked finding IDs, system contracts, dependencies, unapproved parallel work, runtime/config/API/schema/manifest impact, retraining, migration, smoke/integration/failure/reconciliation tests, exact pass/fail criteria, pre/post state inspection, stage permissions, alerts, rollback and paper/testnet needs.
- A mandatory **release blocking register**: P0/P1 failures, unknown exchange/order/stop state, missing provenance, missing runtime artifacts, unsafe UI controls, stale cache, invalid model artifacts, incomplete fills and PnL, failed auth negative cases, unsafe test assertions and unreconciled migration.
- A data lineage dependency: after fixing features/labels/splits/calibration, mark affected old features, training datasets, RF/GBM/confirmatory/HMM artifacts, calibration records and promotion eligibility as obsolete *until validated regeneration*. Preserve prior evidence, but do not re-promote contaminated artifacts for convenience.
- An execution dependency: code-level signer correction, durable idempotency and order/fill ledger, fee/funding accounting, protective coverage, circuit-exit rules and authoritative reconciliation must all be integrated before any testnet submission. Production activation is a separate, currently unauthorized decision.
- A migration plan for JSON/SQLite state, idempotency keys, order/position/fill enums and manifest schemas. Keep immutable backups and idempotent replay; **never restore a stale local position snapshot as authoritative over exchange positions**.
- A plan for assessing unproven model efficacy and optimized trade frequency using prospective, frozen-version, out-of-sample paper evidence—not by forcing tuning or fabricating promised returns.
- A risk matrix and `CR-###` process. A version freeze prevents **unreviewed** changes, not evidence-backed amendments for newly found defects or API evolution.

**Acceptance test:** For every known defect show an unbroken relation: `finding → evidence → root cause → change step → exact test oracle → operational/replay gate → migration/retrain implications → Auditor acceptance → current state`. No rows may vanish when steps are merged. Identify blockers that cannot yet be coded or independently verified. Explicitly distinguish a test plan from a test already executed.

**Final decision:** `PLAN_REJECTED`, `PLAN_APPROVED_WITH_NONLIVE_CONDITIONS`, or `PLAN_APPROVED_FOR_NONLIVE_IMPLEMENTATION`. Name each condition and owner. Output final `04_EXECUTION_PLAN_vX.Y.md`, updated trackers, release blockers and artifact manifest. State `LIVE CAPITAL: NO-GO`. **Stop without modifying code.**
