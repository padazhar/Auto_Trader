> **Required attachment:** `GLOBAL_CONTRACT.md` plus the current repository, audit, approved plan, and latest handoff artifacts. This prompt inherits the entire Global Contract.

# P06 — INDEPENDENT STEP VALIDATION, REPAIR LOOP AND NEXT-STEP GATE

**Inputs:** `STEP_ID = S__`; approved plan/version; *before* repo ZIP; *after* repo ZIP; patch/change list; test logs/fixtures; updated trackers; required sanitized runtime/DB evidence if applicable.

**Mode: AUDIT THE COMPLETED STEP; do not implement a future step.**

Read the GLOBAL CONTRACT. Independently inspect before/after file hashes, source diffs and changed file list. Confirm no hidden changes, no unauthorized config shifts, no secret leaks, no unsafe production endpoint flips, no unrelated refactors, no dropped tests and no plan/CR violations. Trace actual changed behavior across callers, state schema, persistence, controllers, paper/live authority, risk, manifests and monitoring. Challenge both old and new test assertions: do they observe the economic effect or merely assert that a mock function was invoked? Reproduce the original failing case and at least one adversarial case relevant to the change.

**Run required validation where safely possible:** parse/type/import, targeted tests, full affected regression suite, contract tests, deterministic data replay, simulated network/ack/fill/restart/partial-fill faults and historical feature availability/ML split checks as applicable. Confirm tests actually ran (not collected-only/skipped/xfailed), record count, time, exit status and artifact references. Check test results for flaky retries and threshold changes. For state/schema changes, run migration → read/write → rollback/forward-recovery tests on disposable copies, and independently verify authoritative fill/PnL reconstruction on synthetic fixtures. Ensure a protected exchange position is never silently downgraded to unprotected when the controller disappears.

**Explicit regression dimensions:** runtime entry points, config/env precedence, scheduler/controller orchestration, WS reconnect/staleness, data timestamps and lineage, RF/GBM/HMM/confirmatory, inference/labels/manifests, signal/action and HOLD calibration, risk circuit/exits, idempotency and order state, partial fills/protective stops, accounting and funding, universe/blocked-symbol life, dashboard/auth/event freshness and error reporting. Mark `NOT_APPLICABLE` only with justification and `NOT_TESTED` if evidence absent.

**If failure:** Produce exact failing evidence, impacted finding IDs, rollback/containment, correction guidance and a `CR-###` if scope must expand. If the step-specific correction is within approved scope and the operator explicitly authorizes repair, apply **only that same step** on a new version; produce full updated repo and run the entire P06 gate again. Never silently implement a later step or ignore a critical regression to retain sequence momentum.

**If pass:** Update all linked finding states appropriately (`TESTED` or `INTEGRATION_VERIFIED`, not automatically `VERIFIED_CLOSED` where forward/runtime/exchange proofs remain), step ledger, actual coverage and evidence hashes. Give precise next-step prerequisites and whether retraining/paper/testnet remains outstanding.

**Output verdict:** `STEP_PASS_FOR_NEXT_NONLIVE_STEP`, `STEP_FAIL_REMEDIATE_SAME_STEP`, or `STEP_BLOCKED_NEED_EVIDENCE`. **Stop after reporting. Do not auto-implement the next step.**
