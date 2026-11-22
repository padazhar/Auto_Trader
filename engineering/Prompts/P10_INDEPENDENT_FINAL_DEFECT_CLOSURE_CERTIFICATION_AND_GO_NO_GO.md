> **Required attachment:** `GLOBAL_CONTRACT.md` plus the current repository, audit, approved plan, and latest handoff artifacts. This prompt inherits the entire Global Contract.

# P10 — INDEPENDENT FINAL DEFECT-CLOSURE CERTIFICATION AND GO/NO-GO

**Mode: INDEPENDENT FINAL AUDIT. Do not change production trading state.**

Reopen the final complete repository and all documents/tests/logs/run results from P01–P09; do not rely on statements like “Step N completed.” Independently re-run safely executable tests and inspect the final actual source, configuration and manifests at pinned hashes. Compare the shipped artifact to the last individually verified commit; validate that the restored/compiled/tested files are precisely the files to be deployed. Any checkout mismatch invalidates previous qualification evidence until reproduced.

**A. Proof of closure:** For **all 24 original findings + every NEW ID**, audit code, failure-before/success-after tests, cross-contract integration, migrations, telemetry, paper, testnet and Auditor approval. Print counts by status and all remaining unresolved P0/P1 items. A finding can be verified closed only if its applicable gates are actually supported by artifacts; do not aggregate “24/24 covered” into “24/24 fixed.” For unobservable model alpha, report `UNPROVEN` until independent prospective evidence exists. For all **29 original dimensions**, print measured results, missing artifacts and readiness separately.

**B. End-to-end ledger certification:** Verify lineage from exchange timestamped data → causal features → immutable prediction/model manifest → decision/risk/intent → transmitted request/ACK/stream → fills/trade IDs and protections → exchange positions/fees/funding → recognized/unrecognized/net equity and dashboard. Review startup, shutdown, downtime/recovery, symbol exit supervision, emergency stop, auth/control access and alerts. No unknown order can be silently assigned terminal; no open exchange position may be ignored due to rotation or block; no code restore can overwrite real economic state.

**C. Statistical and operational claims:** Validate no historical future data, correct per-timeframe labels, proper training-only transforms, untouched test, calibration reliability and drift controls; report trading frequency, turnover, realized/slippage assumptions, class imbalance/HOLD, net expected value, risk-adjusted returns and confidence intervals only when real artifacts support them. Distinguish simulated paper performance from testnet fills. Require live liquidity/slippage uncertainties explicitly acknowledged. Neither a high win rate nor green dashboard alone is proof of alpha.

**D. Independent release safety:** Verify documented operator roles, minimal permissions, masked secrets and rotation, dependency/CI health, immutable artifact/provenance hashes, runbook, alerts, emergency entry-disable, open-position monitoring and recovery, release rollback plus external state reconciliation. Any production real-money action requires a **separate future explicit authorization with accountable human risk and engineering approval**, not a decision made by this prompt. If any requirement is unmeasured, classify accordingly.

**E. Final outcomes (choose one, never combine incompatible claims):**

1. `NOT_READY — CRITICAL/HIGH DEFECTS OR MISSING REQUIRED EVIDENCE`.
2. `NONLIVE_CORRECTNESS_VERIFIED — FURTHER PAPER/TESTNET/STATISTICAL EVIDENCE NEEDED`.
3. `PAPER_AND_TESTNET_QUALIFIED — PRODUCTION DECISION REQUIRES SEPARATE HUMAN APPROVAL`.
4. `RELEASE_REVIEW_PASSED_FOR_SPECIFIED_ARTIFACT — STILL NOT AUTOMATIC LIVE AUTHORIZATION`.

For failures, generate exact corrective `CR-###` and route back to P02/P03 for changed plan, or to P05/P06 for approved limited repairs, then repeat affected P07–P10 qualification. **Do not end with a false declaration that everything has been solved.**

**Deliverables:** `FINAL_DEFECT_CLOSURE_CERTIFICATE.md` (including all open exceptions), `FINAL_RECONCILIATION_REPORT.md`, `FINAL_RELEASE_GATE.md`, archived exact test evidence and source hashes, final updated finding/coverage/step matrices. State whether the system is objectively safer and what cannot yet be known.
