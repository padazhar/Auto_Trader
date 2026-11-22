> **Required attachment:** `GLOBAL_CONTRACT.md` plus the current repository, audit, approved plan, and latest handoff artifacts. This prompt inherits the entire Global Contract.

# P11 — ONGOING PRODUCTION-CHANGE / INCIDENT FORENSICS (REUSABLE)

**Mode: REVIEW AN ACTUAL NEW OBSERVATION OR PROPOSED CHANGE. No automatic live action.**

Use the GLOBAL CONTRACT and the latest certified version of the repository, manifest, config and closure register. Inventory actual new logs/exchange records and code since the last audit. Evaluate regressions, newly documented Binance API changes, stale price and WS health, order-unknown events, protective coverage, duplicate/lost partial fills, drift in features/probability calibration/trade cost, model selection, symbol/blocked position states, funding/fees/equity/PnL, dashboard/telemetry integrity and secret/control-plane exposure. Do not assume the previous certification guarantees a new build or regime. Assess severity and contain increased risk first; **never casually cancel protection for an existing exchange position**.

Every discovered issue receives `NEW-###` or `INC-###` with precise source/runtime evidence, timeline, root-cause hypothesis and confidence, owner, affected original findings/dimensions, blast radius, containment, regression oracle, proposed CR, migrations/retraining, rollback and paper/testnet requalification requirements. Only initiate code implementation via renewed P03 approval and P05/P06 process. Produce an incident/change report and a precise next safe action. If no actual new observations exist, state NO_NEW_EVIDENCE and avoid claiming monitoring took place.
