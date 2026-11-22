# Auto_Trader — Final Institutional Plan Validation & Change-Control Baseline
**Prepared:** 2026-11-22  |  **Mode:** PLAN ONLY — no source edited, no orders, no training, no API calls.  |  **Approval:** **NOT APPROVED as submitted; revised plan APPROVED WITH CHANGES for isolated, non-live implementation planning only. NO LIVE APPROVAL.**

**Source of truth for evidence:** original repository ZIP SHA-256 `98746368b520b9a71d04bf60bff6f649a9785621ad2e24fdd21d9b0693ec83d7`; previous forensic audit SHA-256 `76799fe1fd7d562b354d0d3065b54533e037f37258ab07c98bd7a3aa6b85af1d`. Source paths below are relative to `Auto_Trader/`. These hashes pin the audited baseline; they do not establish whether a different deployed checkout matches it.

**Verification:** reopened actual archive and prior audit, inspected source and test files. Static-only validation. Archived runtime data directories are unpopulated; three process logs are present. No exchange fills, model artifacts, historical market DB, active positions, or complete operational telemetry in archive. All prospective edits below are *authorized scopes for later review*, not modifications already made.

## 1. Executive diagnosis — Planner
The architecture already has deliberate paper/testnet defaults, a training scheduler, model governance scaffolding, order store, risk/circuit controls, universe lifecycle and monitoring. Those foundations should be preserved, not rewritten wholesale. Live capital is unsafe until the identified execution, research, control-plane and reconciliation failures are resolved. Historical research reports cannot be used as evidence of alpha while MTF, label, imputation, and calibration leakage remain. A ~110-minute sample with 1,128 successful training logs does not demonstrate live reliability, independent model performance or profitability.

**Separation of gates:** an engineering defect can be *code-fixed* before it is *integrated*, *paper-validated*, *testnet-validated* and *approved for real capital*. Restoring a signed Binance connection while fill recovery and stops are unsafe is specifically prohibited. Do not treat all findings as proof of actual financial losses; some are confirmed faulty code paths and others are evidence gaps or conditional risks.

## 2. Independent Auditor — 24 claims checked and closure register
`CONFIRMED` means code/config supports the structural claim; it is not proof of real-money impact. `PARTIAL` means only part is source-supported or efficacy requires unavailable runtime data. No contradictory claim found in this prior report at the reviewed source-level, but the actual deployed/runtime state is unverified. A future contradictory result must trigger an audited plan amendment.

| Finding | Severity | Auditor verdict | Root cause / institutional and runtime impact | Evidence | Closure steps |
|---|---|---|---|---|---|
| P0-01 | CRITICAL | CONFIRMED | Higher-timeframe opening-time as-of merge | `features/timeframe_fusion.py:289–346` | S12,S13,S14,S15,S23 |
| P0-02 | CRITICAL | CONFIRMED | RF target uses global anchor instead of per-model interval | `training/rf_training_engine.py:184–196` | S13,S14,S15,S23 |
| P0-03 | CRITICAL | CONFIRMED | Signed payload and transmitted parameter order differ | `execution/exchange_adapter.py:149–162`; `execution/exchange_reconciliation.py:75–88` | S02,S23,S25 |
| P0-04 | CRITICAL | CONFIRMED | Dashboard auth bypass for mutation endpoints | `monitoring/auth.py:19–49`; `monitoring/web_app.py:780–954`; `tests/test_monitoring_auth.py` | S01,S20,S23 |
| P0-05 | CRITICAL | CONFIRMED | Protective exits require live controller execution | `runners/live_trading_controller.py:523–610` | S08,S09,S23,S25 |
| P0-06 | CRITICAL | CONFIRMED | ACK recovery, false terminalization, incomplete order-fill lifecycle | `execution/exchange_adapter.py:164–262`; `execution/order_router.py:89–146` | S04,S05,S23,S25 |
| P0-07 | CRITICAL | CONFIRMED | Repeated economic intents can collide in client ID | `execution/order_store.py:55–81` | S03,S23 |
| P0-08 | CRITICAL | CONFIRMED | Risk circuit evaluated before EXIT allowance | `execution/risk_engine.py:275–295` | S09,S23 |
| P0-09 | CRITICAL | CONFIRMED | Old bar-close price can be now-stamped as fresh | `runners/market_data_controller.py:497–531`; `core/market_io.py:271–342` | S10,S23 |
| P1-10 | HIGH | CONFIRMED | Validation sample reused to fit probability calibrator | `training/base_trainer.py:87–106`; `training/rf_training_engine.py:249–260` | S14,S15,S22,S23 |
| P1-11 | HIGH | CONFIRMED | Backward-fill and preprocessing before chronological split | `features/dataset_preparation.py:69–76` | S13,S23 |
| P1-12 | HIGH | CONFIRMED | Configured purge can be shorter than label horizon | `config/system_settings.yaml:72–87`; `features/dataset_preparation.py` | S13,S14,S23 |
| P1-13 | HIGH | CONFIRMED design gap | Horizon directional labels and EV not executable net trade payoffs | `features/label_builder.py`; `inference/expected_value.py` | S16,S22,S23,S24 |
| P1-14 | HIGH | CONFIRMED | Partial-fill position/PnL updates and fee/funding authority incomplete | `execution/portfolio_state.py:861–1025`; `runners/live_trading_controller.py:601–610` | S05,S06,S23,S25 |
| P1-15 | HIGH | PARTIAL | Execution-side exchange filters/account modes not demonstrated | `execution/exchange_adapter.py:164–185` | S07,S23,S25 |
| P1-16 | HIGH | CONFIRMED | Paper/backtest simplistic fixed cost assumptions | `config/system_settings.yaml:339–343`; `services/backtest_engine.py` | S16,S23,S24 |
| P1-17 | HIGH | PARTIAL | Promotion framework exists; active manifest evidence unavailable | `core/artifact_contract.py`; `config/system_settings.yaml:88–127` | S15,S23,S24 |
| P1-18 | HIGH | CONFIRMED | Default correlation groups empty and equity/margin authority limited | `config/system_settings.yaml:352–367`; `execution/risk_engine.py` | S17,S23,S25 |
| P1-19 | HIGH | CONFIRMED | Event broadcast depends on capped array length | `monitoring/web_app.py:648–651` | S20,S23 |
| P1-20 | HIGH | CONFIRMED design gap | Blocked live symbol close fails closed but does not close/supervise automatically | `execution/portfolio_state.py:488–519` | S11,S21,S23 |
| P2-21 | MEDIUM | PARTIAL effectiveness | Confirmatory blend zero, HMM paper-confidence gate not proven as directional benefit | `config/system_settings.yaml:166–196,258–282`; `inference/trade_decision_logic.py` | S22,S24 |
| P2-22 | MEDIUM | PARTIAL evidence | Feature drift/importance and live statistical usefulness unproven | `features/timeframe_fusion.py`; `features/feature_store_contract.py` | S20,S22,S24 |
| P2-23 | MEDIUM | CONFIRMED / efficiency unverified | Bar-based training triggers; redundant work and actual promotion impact unproven | `scheduling/priority_manager.py`; archived training log | S18,S23,S24 |
| P2-24 | MEDIUM | PARTIAL | Universe rotation/controller fault tolerance implemented but not empirically verified | `runners/market_data_controller.py`; `core/symbol_management.py`; `main.py` | S19,S21,S23,S24 |

**Severity interpretation:** P0 defects threaten correctness, financial state, execution safety or control access; P1 defects invalidate research results or materially weaken capital/accounting controls; P2 findings are material but often require further empirical evidence before modifying policy. The root-cause explanations and detailed runtime effects for each ID are preserved in the original forensic audit; the table above is the non-droppable remediation register.

### New hazards/unsafe assumptions the Auditor requires addressed
1. **Tests can prove the wrong behavior.** `tests/test_monitoring_auth.py` explicitly asserts disabled authentication; `tests/test_rf_training_label_contract.py` uses a 15m model and mocks the label builder, so it cannot prove 1m/3m/5m target semantics. Existing green tests are not an acceptance gate until these oracle errors are corrected.
2. **Code rollback is not state rollback.** Once Binance fills an order, restoring old JSON, an old model, or an earlier Git checkout cannot reverse that trade. Use exchange-authoritative snapshots plus forward reconstruction; do not overwrite canonical fills.
3. **Safety and model evaluation conflict when mixed.** A dynamic retraining/promotion process can contaminate a supposedly frozen prospective experiment. Freeze model versions for evidence windows; train candidates in shadow only.
4. **Unproven performance is not a defect to "fix" with tuning.** RF, GBM, confirmatory and HMM incremental value, feature drift, optimal HOLD threshold, rotation quality and trading frequency need data and independent experiment designs; do not implement speculative changes as mandatory refactoring.
5. **A plan cannot claim completeness by forbidding future steps.** New critical exchange behavior/regressions need controlled scope change with explicit evidence, revised impact and reapproval. Never silently expand scope or abandon a finding.
6. **Historical feature availability also affects optional market context.** `features/timeframe_fusion.py:311–322` joins optional context by timestamp; its actual event/publication time must be established rather than assuming all timestamps represent data availability.
7. **The repository is not the deployed state.** Active mode, deployed commit/config, credential privileges, open positions, order state and model pointers require direct authorized runtime observation, absent from the archive.
8. **Role independence must be real for release signoff.** PLANNER/AUDITOR/IMPLEMENTER personas in a single model execution provide structured adversarial review, not genuinely independent operators. Production capital authorization requires separate accountable engineering and risk signoff, plus independently generated test evidence.
9. **Secret hygiene and control auth are linked.** A static monitoring token appears in the committed configuration, while authentication is disabled; rotate/redact before making that mechanism active.

## 3. Final frozen sequence — Planner reconciled with Auditor
**Freeze semantics:** `Plan v1.0` is the approved baseline for scope, safety rules and evidence requirements, not an absolute bar on evidence-driven change. Every adjustment needs a numbered change request, evidence, impacts on previous tests/steps, and renewed approval. NO code implementation has started. All files listed exist in the submitted repo except any explicitly future-proposed test fixtures. Files are scope for **inspection and potential modification**, not a claim that every one must be changed.

### Auditor verdict for every proposed plan step (source claim versus prospective design)

A confirmed defect does not make every **proposed implementation method** already proven; high-impact design choices remain conditional on tests and current exchange documentation.

| Step | Auditor status | Evidence-bounded correction / condition |
|---|---|---|
| S00 | CONFIRMED | Archive contains missing runtime/state/artifacts; baseline capture necessary. |
| S01 | CONFIRMED | Auth functions and existing tests explicitly bypass tokens. |
| S02 | CONFIRMED | Mismatch in signing/transmitted byte ordering reproduced in audit. |
| S03 | CONFIRMED | Fallback client ID lacks unique economic-intent identity. |
| S04 | CONFIRMED | Order recovery treats missing open order as terminal and masks transport failure. |
| S05 | CONFIRMED | No complete authoritative event/fill ledger established in inspected path. |
| S06 | CONFIRMED | Partial-close and fees/funding authority gaps. |
| S07 | PARTIAL | Missing exchange-filter/account-mode evidence; testnet requirements to verify. |
| S08 | CONFIRMED | Controller-side stop trigger not persistent native exchange protection. |
| S09 | CONFIRMED | Circuit ordering blocks normal EXIT path. |
| S10 | CONFIRMED | DB-close fallback shares artificially fresh timestamp; live WS uptime unknown. |
| S11 | CONFIRMED | Blocked live position does not automatically become flat; deployment policy unknown. |
| S12 | CONFIRMED | Opening-time MTF merge; optional context publication time unresolved. |
| S13 | CONFIRMED | RF anchor, bfill, insufficient configured purge confirmed. |
| S14 | CONFIRMED | Calibrator fits on the sample used for final metrics. |
| S15 | PARTIAL | Manifest/promotion scaffolding source-confirmed; active artifacts unavailable. |
| S16 | PARTIAL | Simplified costs and heuristic EV confirmed; actual magnitude of improvement unproven. |
| S17 | PARTIAL | Empty group default and fixed equity confirmed; advanced factor-cap choice evidence-dependent. |
| S18 | PARTIAL | Bar-driven cadence confirmed; duplicate workload and churn unproven. |
| S19 | PARTIAL | Controller supervisors/reconnect exist; actual crash recovery and uptime unverified. |
| S20 | CONFIRMED | Count-based event cap bug; end-to-end telemetry/provenance still need runtime proof. |
| S21 | PARTIAL | Volume-based rotation and warmup confirmed; liquidity-value optimization requires data. |
| S22 | UNSUPPORTED as performance claim | Neither feature alpha nor HMM/confirmatory improvement shown; only ablation is approved. |
| S23 | PARTIAL | Acceptance tests are proposed checks and must be created/validated later. |
| S24 | PARTIAL | Prospective paper sample missing; experimental efficacy not yet knowable. |
| S25 | PARTIAL | Binance testnet contract must be validated with current docs and testnet execution. |
| S26 | PARTIAL | Approvals are governance controls, not an observed existing deployed process. |

### Dependency graph and unsafe parallel work
`S00 → S01, S02 → S03 → S04 → S05 → S06 → S07 → S08 → S09 → S10 → S11 → S12 → S13 → S14 → S15 → S16 → S17 → S18 → S19 → S20 → S21 → S22 → S23 → S24 → S25 → S26` is the conservative **validation/merge order**; some code exploration may proceed in parallel in isolated branches without activating cross-contract behavior. It is **unsafe** to enable signing/order submission before recovery+protection+ledger; deploy a new portfolio schema without migration; promote a model while point-in-time dataset rules are changing; introduce model optimization into a frozen paper evidence window; or remove a symbol while an exchange position remains open.

### 1. S00 — Non-live containment, immutable evidence and baseline
**Findings:** ALL  |  **Nature:** PRESERVES / HARDENS  |  **Implementation risk:** CRITICAL if bypassed
**Dependencies / order reason:** None. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `config/system_settings.yaml`; `core/config_contract.py`; `main.py`; `core/live_readiness_guard.py`; `core/path_manager.py`; `runners/live_trading_controller.py`; `scripts/` (live-readiness and self-check entry points).
**Config change / governance:** Freeze paper/testnet/canary parameters; record effective environment and config hash; do not add unverified toggles.
**Expected runtime effect:** No exchange activity; preserve current paper/testnet behavior.
**Tests and regression focus:** Inventory and dependency lock; source hashes; baseline test coverage audit (no production run); offline checks of all entry points.
**Acceptance / evidence gate:** Written no-live declaration, archive SHA-256, source and state snapshots (if provided), test baseline and known failing/unsafe assertions recorded.
**Failure risk:** A baseline hash or deployment-state mismatch is mistaken for the audited checkout.  |  **Rollback complexity:** LOW for read-only evidence capture.
**Rollback and state recovery:** Restore code/config snapshot only; never overwrite exchange state.
**Verification flags:** retrain=N | paper=N | live-candidate=N | recon=READ-ONLY if exchange account is available | telemetry=BASELINE | compatibility=Y.
**Hard stop:** Do not enable signed order submission, live credentials, active model promotion, or dashboard mutations during remediation.

### 2. S01 — Authorize and audit the control plane
**Findings:** P0-04  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S00. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `monitoring/auth.py`; `monitoring/web_app.py`; `monitoring/control_plane_store.py`; `monitoring/static/app.js`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_monitoring_auth.py`; `tests/test_control_plane_store.py`.
**Config change / governance:** Move secret out of static configuration; rotate any previously embedded token; default deny privileged REST/WebSocket actions; distinguish read and write roles.
**Expected runtime effect:** Unauthenticated UI mutation requests must fail; dashboard clients require explicit authorized flows.
**Tests and regression focus:** Invalid/missing/expired token; privilege boundaries; CSRF/session and websocket bypass; immutable actor audit; reverse-proxy and localhost access.
**Acceptance / evidence gate:** All state-changing endpoints reject unauthenticated callers; privileged websocket access cannot bypass auth; actions attributed and tamper-evident.
**Failure risk:** The dashboard becomes inaccessible or accidentally permits privileged actions.  |  **Rollback complexity:** MEDIUM for UI compatibility; security bypass must never be restored.
**Rollback and state recovery:** Revert UI/login compatibility only while control mutations remain disabled; NEVER restore unauthenticated write paths.
**Verification flags:** retrain=N | paper=UI smoke | live-candidate=N | recon=N | telemetry=Y | compatibility=VERSIONED/UI.
**Hard stop:** Do not ship a temporary authentication bypass in any environment exposed to users.

### 3. S02 — Canonical exchange signing and request transport
**Findings:** P0-03  |  **Nature:** CORRECTS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S00. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/exchange_reconciliation.py`; `core/config_contract.py`; `config/system_settings.yaml`; `tests/test_execution_order_lifecycle.py`; `tests/test_live_secret_hardening.py`.
**Config change / governance:** Preserve testnet default; validate timestamp/recv-window and key scoping; no automatic production endpoint override.
**Expected runtime effect:** Previously invalid signed calls may start succeeding; keep transport disconnected from live capital until S03–S09 and S23–S25.
**Tests and regression focus:** Prepared-request HMAC exact-byte parity, GET/POST/encoding/precision, -1021/-1022 recovery, retries without accidental resubmission.
**Acceptance / evidence gate:** Identical serialized bytes signed and transmitted for every signed request; testnet read-only account query succeeds when credentials supplied.
**Failure risk:** Previously rejected REST orders suddenly execute or a signature retry resubmits an uncertain order.  |  **Rollback complexity:** HIGH; economic effects may be irreversible.
**Rollback and state recovery:** Disable new entries first; restore transport code only after exchange-authoritative reconciliation; cannot undo executed trades.
**Verification flags:** retrain=N | paper=contract | live-candidate=Y | recon=Y | telemetry=Y | compatibility=API.
**Hard stop:** Never activate S02 alone in real live trading.

### 4. S03 — Unique economic intents, durable order outbox and idempotency
**Findings:** P0-07  |  **Nature:** CORRECTS / COMPLETES  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S00, S02 contract. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/order_store.py`; `execution/order_models.py`; `execution/order_router.py`; `runners/live_trading_controller.py`; `core/path_manager.py`; `tests/test_execution_order_recovery.py`; `tests/test_execution_order_lifecycle.py`; `tests/test_trade_lineage_contract_step3.py`.
**Config change / governance:** Declare versioned persistence schema and intent lifetime; never use wall-clock-only retry keys.
**Expected runtime effect:** Distinct legitimate trades route independently; retransmission of the same intent is exactly-once in local outbox.
**Tests and regression focus:** Same-quantity new trades; retry after timeout; restart; partial fill; reversal; concurrent intents; duplicate response.
**Acceptance / evidence gate:** Economic intent ID identifies decision+leg; same intent keeps client ID across retries; new intent obtains a new ID; persisted audit lineage stable.
**Failure risk:** Legacy idempotency entries collide with new intents or retry creates a duplicate exchange order.  |  **Rollback complexity:** HIGH; schema and live-order replay required.
**Rollback and state recovery:** Freeze entries; archive legacy order index; replay/migrate once; never replace live outbox with old snapshot while orders are unresolved.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=Y | telemetry=Y | compatibility=SCHEMA-MIGRATION.
**Hard stop:** Unknown or orphaned old order IDs must be resolved before unfreezing entries.

### 5. S04 — Authoritative order state machine and crash recovery
**Findings:** P0-06  |  **Nature:** CORRECTS / COMPLETES  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S02, S03. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/order_router.py`; `execution/order_store.py`; `execution/order_models.py`; `execution/fill_handler.py`; `runners/live_trading_controller.py`; `tests/test_execution_order_recovery.py`; `tests/test_live_fail_closed_authority.py`.
**Config change / governance:** Separate transport retries from exchange outcome; set maximum recovery age/health gates based on measured evidence.
**Expected runtime effect:** API outage, ACK-only response, filled-but-not-open order, and unknown outcome cannot be recorded as successful/terminal without authority.
**Tests and regression focus:** ACK→fill; 503/timeout after send; missing open order; failed open-order query; canceled/expired; restart at every transition.
**Acceptance / evidence gate:** UNKNOWN is preserved across outages; exchange order/trade lookups reconcile uncertain statuses; no duplicate submission on retry.
**Failure risk:** REST outage or absent open order falsely clears an already filled order.  |  **Rollback complexity:** HIGH; unresolved order state requires exchange-authoritative recovery.
**Rollback and state recovery:** Stop new orders; restore compatible code reader; preserve current exchange order log; replay uncertain events before reopening.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=Y | telemetry=Y | compatibility=STATE-VERSION.
**Hard stop:** No authoritative response must be represented as an empty or terminal result.

### 6. S05 — Exactly-once execution ledger and exchange reconciliation
**Findings:** P0-06, P1-14  |  **Nature:** COMPLETES  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S02–S04. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/exchange_reconciliation.py`; `execution/fill_handler.py`; `execution/order_store.py`; `execution/order_models.py`; `execution/portfolio_state.py`; `runners/live_trading_controller.py`; `core/path_manager.py`; `tests/test_execution_ledger_authority_step1.py`; `tests/test_execution_trade_source_step1.py`.
**Config change / governance:** Require exchange-authoritative reconciliation for live; define venue event retention and replay cursor; separate credentials errors from zero positions.
**Expected runtime effect:** All unique trade fills and external balance changes recorded once and reconciled after disconnect/restart.
**Tests and regression focus:** User-data stream gap plus REST backfill; duplicate/out-of-order trade IDs; fee assets; partial fills; funding/income; recovery during API outage.
**Acceptance / evidence gate:** Canonical ledger exactly matches unique exchange fill IDs with deterministic replay; unresolved mismatches stop increased exposure.
**Failure risk:** Duplicate/missing fill postings distort economic state.  |  **Rollback complexity:** VERY HIGH; append-only economic reconstruction, never snapshot overwrite.
**Rollback and state recovery:** Append-only migration and compensating corrections; no rollback of economic events; exchange-authoritative re-sync before resume.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=MANDATORY | compatibility=LEDGER-SCHEMA.
**Hard stop:** Order status or position totals alone do not establish fill/fee authority.

### 7. S06 — Partial position lifecycle, fees/funding and accurate net PnL
**Findings:** P1-14  |  **Nature:** CORRECTS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S05. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/portfolio_state.py`; `execution/fill_handler.py`; `execution/risk_engine.py`; `runners/live_trading_controller.py`; `reporting/run_report.py`; `monitoring/publisher.py`; `tests/test_closed_trade_attribution_step5.py`; `tests/test_risk_controls.py`; `tests/test_trade_excursion_report_step7.py`.
**Config change / governance:** Make explicit valuation basis, fee/funding accounting currencies, margin/equity data authority and precision tolerances.
**Expected runtime effect:** Partially filled entries/exits update actual size, weighted basis, fee/funding-adjusted PnL and open exposure; no premature full close.
**Tests and regression focus:** 40% close leaves 60%; multi-fill weighted average; short reductions; fee+funding+transfers; mark/unrealized vs realized; liquidation/ADL recording.
**Acceptance / evidence gate:** Ledger-to-position-to-PnL-to-equity bridge reconciles within documented precision/currency tolerance; no double charge for slippage.
**Failure risk:** Partial exit inadvertently removes residual position or double-charges costs.  |  **Rollback complexity:** VERY HIGH; ledger-backed forward reconstruction.
**Rollback and state recovery:** Freeze entries and external mutations; preserve event ledger and rebuild positions forward; do not restore stale portfolio JSON over exchange facts.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=Y | compatibility=PORTFOLIO-SCHEMA.
**Hard stop:** Any accounting mismatch blocks new risk.

### 8. S07 — Instrument filters, exchange account mode and preflight compatibility
**Findings:** P1-15  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S02, S04–S06. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/order_models.py`; `execution/position_sizing.py`; `execution/risk_engine.py`; `execution/exchange_reconciliation.py`; `runners/live_trading_controller.py`; `tests/test_execution_order_lifecycle.py`; `tests/test_live_startup_preflight.py`.
**Config change / governance:** Add validated symbol/exchange filter snapshots and account-mode readiness; preserve canary limits.
**Expected runtime effect:** Reject unrepresentable quantities/notional/modes locally; safely construct reduction orders for verified account mode.
**Tests and regression focus:** LOT_SIZE/MARKET_LOT_SIZE/MIN_NOTIONAL/tick; Decimal rounding; hedge vs one-way; reduce-only semantics; leverage/margin readback; changed filters.
**Acceptance / evidence gate:** Submitted order parameters comply with actual symbol and account constraints; reject with reason and evidence when unsupported.
**Failure risk:** Wrong symbol rounding, positionSide or reduce-only parameters reject protection.  |  **Rollback complexity:** HIGH; active exchange orders/positions must be inspected.
**Rollback and state recovery:** Disable order submission; restore compatible contract under no-live; refresh exchange-info snapshot and reconcile before resume.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=Y | telemetry=Y | compatibility=ORDER-PARAMETERS.
**Hard stop:** Account-mode incompatibility is a hard live blocker.

### 9. S08 — Verified exchange-native protective exits and protection reconciliation
**Findings:** P0-05  |  **Nature:** COMPLETES / HARDENS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S04–S07. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/exchange_reconciliation.py`; `execution/order_models.py`; `execution/order_store.py`; `execution/portfolio_state.py`; `runners/live_trading_controller.py`; `execution/risk_engine.py`; `tests/test_execution_order_lifecycle.py`; `tests/test_shutdown_policy_step5.py`.
**Config change / governance:** Configure protection policy, trigger basis, invalidation/rearming and explicit unsupported-mode rejection; verify current Binance API capabilities on testnet before coding.
**Expected runtime effect:** Open positions retain verified exchange protection even if Python process or WS dies; partial fills resize protection safely.
**Tests and regression focus:** Entry ACK/fill then stop verification; kill process; partial close; stop triggers; rejected protective order; stale/deleted stop; reverse; OCO-like race.
**Acceptance / evidence gate:** Every nonzero exchange position has correct verified protective coverage or causes no-new-entry emergency state with operator alert and controlled reduction.
**Failure risk:** Stop creation/replace/cancel race leaves exposure unprotected.  |  **Rollback complexity:** VERY HIGH; do not remove verified protective orders during code rollback.
**Rollback and state recovery:** Never cancel existing verified stops as a naive revert; freeze new entries and preserve/reconcile stops before switching code.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=MANDATORY | compatibility=EXCHANGE-ORDERS.
**Hard stop:** Controller-only polling cannot qualify as live exchange-native protection.

### 10. S09 — Risk circuit: freeze increases, preserve safe reduce-only exits
**Findings:** P0-08  |  **Nature:** CORRECTS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S05–S08. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/risk_engine.py`; `execution/circuit_breaker.py`; `execution/exposure_model.py`; `execution/order_router.py`; `runners/live_trading_controller.py`; `tests/test_risk_controls.py`; `tests/test_pause_enforcement_live_path.py`; `tests/test_live_fail_closed_authority.py`.
**Config change / governance:** Define reducing vs increasing exposure, unknown-authority handling, kill switch and emergency-operator workflow.
**Expected runtime effect:** Circuit, pause, stale model or symbol blocks additional exposure without blocking verified reductions.
**Tests and regression focus:** Circuit-trip exit while long/short; partial reduce; invalid symbol/quantity; uncertain exchange authority; replay; protection failover.
**Acceptance / evidence gate:** Entry rejected under circuit; authorized exchange-verified reduce-only exit succeeds; never turns reduction into new exposure.
**Failure risk:** Emergency reduce accidentally increases exposure or is denied.  |  **Rollback complexity:** HIGH; reconcile verified risk reduction after recovery.
**Rollback and state recovery:** Safe-mode entries disabled; preserve emergency close route; revert risk policy only if reduction remains possible; reconcile positions.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=Y | telemetry=Y | compatibility=RISK-DECISION.
**Hard stop:** Unsafe exposure-increasing actions remain impossible while authority uncertain.

### 11. S10 — Market websocket health and source-typed execution price freshness
**Findings:** P0-09  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S00; exchange price types confirmed; coordinate S08. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `runners/market_data_controller.py`; `core/market_io.py`; `runners/live_trading_controller.py`; `features/feature_store_contract.py`; `monitoring/publisher.py`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_market_websocket_snapshot.py`; `tests/test_execution_price_refresh_contract.py`; `tests/test_market_latest_snapshot.py`.
**Config change / governance:** Source-specific event-time TTL and clock tolerance; exclude DB close and synthetic refreshed time from live execution readiness.
**Expected runtime effect:** Fresh exchange mark/book required for entries; stale DB close cannot satisfy fast-execution freshness; existing protective orders operate independently.
**Tests and regression focus:** 15m stale DB row with now-written snapshot; WS dead/reordered/duplicate; clock skew and future timestamp; missing symbol/partial snapshot; rest reconnect.
**Acceptance / evidence gate:** Live entry gate requires supported exchange source with verified event timestamp; all market-data freshness failures fail closed for new risk.
**Failure risk:** Live entry uses stale price or excessive data rejection stalls model/venue.  |  **Rollback complexity:** MEDIUM to HIGH; revert to restrictive source gate, not permissive fallback.
**Rollback and state recovery:** Restore price format reader in paper only; in live, retain restrictive freshness policy and exchange protection; verify telemetry.
**Verification flags:** retrain=Y if feature timestamps change | paper=Y | live-candidate=Y | recon=N | telemetry=MANDATORY | compatibility=SNAPSHOT-SCHEMA.
**Hard stop:** Data availability and entry-execution price readiness must be distinct metrics.

### 12. S11 — Blocked-symbol and rotating-universe position supervision
**Findings:** P1-20  |  **Nature:** CORRECTS / COMPLETES  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S05–S10. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `core/symbol_management.py`; `execution/portfolio_state.py`; `runners/live_trading_controller.py`; `runners/tech_rev_controller.py`; `monitoring/control_plane_store.py`; `tests/test_blocked_symbol_forced_close_step6.py`; `tests/test_symbol_management_lifecycle.py`.
**Config change / governance:** Separate entry-eligibility, training, blocked and still-open-risk-supervision states.
**Expected runtime effect:** Blocked or removed symbols cannot open new risk but are still marked, protected, and reconciled until flat.
**Tests and regression focus:** Blocked position remains exchange-open; symbol eviction while stop active; return after restart; force-reduce under pause/circuit; failed close alert.
**Acceptance / evidence gate:** No open exchange position disappears from protection, marking, or reconciliation due to a symbol-list change.
**Failure risk:** Removed/blocked symbol vanishes from active risk supervision.  |  **Rollback complexity:** HIGH while positions are open.
**Rollback and state recovery:** Restore symbol-state snapshot for entry eligibility only; never discard open-position supervision state.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=Y | compatibility=SYMBOL-STATE.
**Hard stop:** Fail-closed status alone is not proof that an exchange position is closed.

### 13. S12 — Causal multi-timeframe synchronization and real feature-age metadata
**Findings:** P0-01  |  **Nature:** CORRECTS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S00, S10 for market timestamp contract. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `features/timeframe_fusion.py`; `features/market_loader.py`; `features/feature_store_contract.py`; `runners/market_data_controller.py`; `core/market_io.py`; `tests/test_feature_store_and_regime.py`; `tests/test_feature_freshness_guards.py`.
**Config change / governance:** Version feature recipe; define candle-open, close, publication/receive and decision-as-of times for each interval.
**Expected runtime effect:** Changes historical training feature values, potentially sample availability and runtime model readiness.
**Tests and regression focus:** Future 1h/4h/1d mutation cannot alter earlier 1m decision; missing source TTL; event-time age; optional context point-in-time join.
**Acceptance / evidence gate:** Point-in-time batch/replay feature equality; no feature available before source bar close+observed publication delay; true nonzero age values.
**Failure risk:** As-of rewrites break feature schema and invalidate models.  |  **Rollback complexity:** MEDIUM code / HIGH model migration.
**Rollback and state recovery:** Archive old features only for comparison; never re-promote contaminated artifact; restore prior recipe in quarantined research branch, not live.
**Verification flags:** retrain=MANDATORY RF/GBM/HMM/confirmatory if affected | paper=Y | live-candidate=Y | recon=N | telemetry=Y | compatibility=FEATURE-VERSION.
**Hard stop:** Old feature_store_v2 model scores/manifests cannot remain eligible by default.

### 14. S13 — Correct horizon labels, causal imputations and event-time leakage purging
**Findings:** P0-02, P1-11, P1-12  |  **Nature:** CORRECTS  |  **Implementation risk:** CRITICAL
**Dependencies / order reason:** S12. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `training/rf_training_engine.py`; `training/gbm_training_engine.py`; `training/dqn_training_engine.py`; `features/label_builder.py`; `features/dataset_preparation.py`; `training/base_trainer.py`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_rf_training_label_contract.py`; `tests/test_directional_labels.py`; `tests/test_training_statistical_ineligibility_step5.py`.
**Config change / governance:** Label horizon explicitly per interval; split event-end-aware purge; causal only forward fill with TTL; no global pre-split feature inference.
**Expected runtime effect:** Expected class frequencies and target outcomes change; legacy RF/GBM metrics no longer comparable without re-evaluation.
**Tests and regression focus:** 1m/3m/5m/15m target close and label end; no bfill before t; fold-boundary overlap; time gaps; absent required features.
**Acceptance / evidence gate:** Each target at t uses designated interval t+h; no feature at t derives from t+Δ; no label-event overlap across validation boundaries.
**Failure risk:** Target changes create silent train/inference incompatibility.  |  **Rollback complexity:** HIGH model invalidation; no rollback to contaminated authority.
**Rollback and state recovery:** Keep old datasets archived but not inference-eligible; restore only as read-only investigation, not promoted live authority.
**Verification flags:** retrain=MANDATORY all impacted roles | paper=Y | live-candidate=Y | recon=N | telemetry=Y | compatibility=LABEL/VERSION.
**Hard stop:** Versioned schema and explicit model ineligibility before retraining.

### 15. S14 — Independent model fitting, calibration, model selection and test
**Findings:** P1-10  |  **Nature:** CORRECTS  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S12–S13. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `training/base_trainer.py`; `training/rf_training_engine.py`; `training/gbm_training_engine.py`; `training/dqn_training_engine.py`; `features/dataset_preparation.py`; `core/artifact_contract.py`; `tests/test_probability_calibration_step6.py`; `tests/test_step9_validation_pipeline.py`.
**Config change / governance:** Train/calibration/untouched test periods plus purge/embargo and pre-registered metrics; calibrator never shares evaluation rows.
**Expected runtime effect:** Headline historical metrics may fall; prevents contaminated promotions.
**Tests and regression focus:** No row identity/time overlap among fitting, tuning, calibration, selection, test; blocked contaminated manifest; paired incumbent comparisons.
**Acceptance / evidence gate:** Independent post-calibration OOS metrics with confidence intervals; calibration reliability and HOLD/LONG/SHORT class distributions recorded per interval.
**Failure risk:** Holdout leakage remains hidden, or insufficient data leaves no valid model.  |  **Rollback complexity:** MEDIUM; retrain and govern reduced readiness.
**Rollback and state recovery:** No promotion of old score artifacts; revert only code for offline baselines; preserve immutable experiment lineage.
**Verification flags:** retrain=MANDATORY | paper=Y | live-candidate=Y | recon=N | telemetry=Y | compatibility=METRICS-SCHEMA.
**Hard stop:** A passing validation suite must prove independence, not merely high scores.

### 16. S15 — Manifest readiness, immutable lineage and controlled promotion governance
**Findings:** P1-17  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S12–S14. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `core/artifact_contract.py`; `core/training_registry.py`; `core/model_approval.py`; `inference/model_artifact_loader.py`; `inference/readiness.py`; `core/live_readiness_guard.py`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_promotion_truth.py`; `tests/test_active_manifest_readiness_step7.py`; `tests/test_registry_readiness_verification_step4.py`; `tests/test_model_artifact_loader.py`.
**Config change / governance:** Versioned evidence thresholds and strict candidate→paper→live-candidate→approved state machine; require explicit human/owner signoff.
**Expected runtime effect:** Old/insufficiently evidenced models become ineligible; no silent promotion or accidental live fallback.
**Tests and regression focus:** Recipe/schema hashes; missing file; corrupt manifest; incorrect interval; partial promotion crash; rollback pointer; stale active pointer; signoff.
**Acceptance / evidence gate:** Every active model links immutable data/code/feature/label/split/calibration/evidence hashes, stage, approval actor and rollback parent.
**Failure risk:** Atomic active-pointer/manifest updates diverge on crash.  |  **Rollback complexity:** HIGH; only verified compatible pointer swaps.
**Rollback and state recovery:** Atomic pointer swap only to an already verified compatible artifact; never promote pre-leak-fix artifact or mutate candidate in place.
**Verification flags:** retrain=MANDATORY for requalification | paper=Y | live-candidate=Y | recon=N | telemetry=Y | compatibility=MANIFEST-SCHEMA.
**Hard stop:** Do not assume a signature validates statistical quality or candle availability.

### 17. S16 — Realistic paper/backtest costs and action-conditioned EV/label research
**Findings:** P1-13, P1-16  |  **Nature:** CORRECTS / VALIDATES  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S05–S07, S10, S12–S14. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `services/backtest_engine.py`; `execution/exchange_adapter.py`; `features/label_builder.py`; `inference/expected_value.py`; `inference/trade_decision_logic.py`; `training/rf_training_engine.py`; `training/gbm_training_engine.py`; `config/system_settings.yaml`; `tests/test_backtest_realism.py`; `tests/test_paper_execution_cost_realism_step6.py`.
**Config change / governance:** Keep modeled execution separate from exchange-realized fills; calibrate costs using observed tier, spread, depth, latency and funding (missing data is blocker).
**Expected runtime effect:** May reduce estimated strategy expectancy and trading frequency; directional labels and executable trade policy evaluated separately.
**Tests and regression focus:** Maker/taker fee and funding; size-sensitive slip; partial fill/rejection; stop gap; fee once; 1×/2×/3× transaction cost; paired policies.
**Acceptance / evidence gate:** Net execution-conditioned expectancy and fill feasibility demonstrated prospectively; slippage attribution not double-counted.
**Failure risk:** Cost labels/EV thresholds create spurious edge and overtrade.  |  **Rollback complexity:** MEDIUM paper / HIGH if in live policy.
**Rollback and state recovery:** Revert only in offline simulation branch; never substitute paper fills as exchange authority; old promotion score ineligible.
**Verification flags:** retrain=CONDITIONAL if labels/policy change | paper=MANDATORY | live-candidate=Y | recon=Y for cost evidence | telemetry=Y | compatibility=PERFORMANCE-METRICS.
**Hard stop:** No threshold tuning on untouched holdout.

### 18. S17 — Portfolio capital, correlation, margin and tail-risk controls
**Findings:** P1-18  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S06–S11, S16 evidence. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/risk_engine.py`; `execution/exposure_model.py`; `execution/portfolio_allocator.py`; `execution/position_sizing.py`; `execution/portfolio_state.py`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_risk_controls.py`; `tests/test_portfolio_allocation.py`.
**Config change / governance:** Equity-based budget, margin/maintenance/leverage readiness, nonempty supported correlation groups, factor/beta concentration and liquidation stress caps.
**Expected runtime effect:** Smaller/more restrictive position sizing and more explicit risk holds; safer portfolio-wide behavior.
**Tests and regression focus:** Correlated losses; long/short netting vs gross; leverage/liquidation buffer; unavailable equity; partial close while circuit tripped.
**Acceptance / evidence gate:** Risk metrics derive from authoritative live equity and exposure; correlated/aggregate risk scenario rejections deterministic.
**Failure risk:** Correlation/margin policy blocks legitimate reductions or mis-sizes risk.  |  **Rollback complexity:** HIGH if live exposures exist.
**Rollback and state recovery:** Revert to safer zero-new-risk configuration, not permissive earlier limits; reconcile all open exposures.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=Y | compatibility=RISK-SCHEMA.
**Hard stop:** Do not assume gross exposure cap bounds liquidation risk.

### 19. S18 — Training cadence, dedupe, queue fairness and promotion churn
**Findings:** P2-23  |  **Nature:** HARDENS / VALIDATES  |  **Implementation risk:** MEDIUM
**Dependencies / order reason:** S12–S15. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `scheduling/priority_manager.py`; `scheduling/job_dispatcher.py`; `runners/training_controller.py`; `runners/tech_rev_controller.py`; `core/training_registry.py`; `config/system_settings.yaml`; `core/config_contract.py`; `tests/test_training_scheduler_cooldown.py`; `tests/test_retraining_frequency_rules_step9.py`; `tests/test_training_scheduler_fairness_step9.py`.
**Config change / governance:** One documented precedence rule for retrain_cycles vs retrain_min_new_bars; bounded queue and CPU budget; stable model promotion windows.
**Expected runtime effect:** Reduce duplicate work, train contention and stale-model latency without silently skipping required new candles.
**Tests and regression focus:** Stable unchanged last-bar hash; new unique bars; out-of-order bars; duplicate events; trainer hung/crashed; fairness & backoff.
**Acceptance / evidence gate:** Each train job has unique input snapshot/hash, schedule reason, duration, CPU/IO load and eligible result status; no duplicate data retrain.
**Failure risk:** Over-throttling causes stale models or starvation while dedupe hides new bars.  |  **Rollback complexity:** LOW to MEDIUM with paper-only changes.
**Rollback and state recovery:** Restore known cadence in paper only, with promotion disabled; preserve registry/event audit.
**Verification flags:** retrain=NO forced by cadence change | paper=Y | live-candidate=Y | recon=N | telemetry=MANDATORY | compatibility=SCHEDULER-CONTRACT.
**Hard stop:** 1,128 successes in 110 minutes are workload observations, not model efficacy proof.

### 20. S19 — Websocket/controller fault recovery and process-state ownership
**Findings:** P2-24 + runtime/controller coverage  |  **Nature:** HARDENS / COMPLETES  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S04–S11, S18. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `main.py`; `start_auto_trader_roles.sh`; `runners/market_data_controller.py`; `runners/live_trading_controller.py`; `runners/tech_rev_controller.py`; `runners/signal_generator_controller.py`; `runners/state_bridge_controller.py`; `scheduling/job_dispatcher.py`; `tests/test_launcher_runtime_contracts.py`; `tests/test_shutdown_coordinator_step3.py`; `tests/test_market_websocket_snapshot.py`.
**Config change / governance:** Health SLOs, deterministic startup/replay, bounded reconnect backoff, single-writer state ownership; no unattended live restart before authority confirmed.
**Expected runtime effect:** Dead/hung controllers fail safely; stale snapshots cannot authorize entries; restarts rebuild state before accepting new risk.
**Tests and regression focus:** Kill controller on send/ACK/fill/protection; stalled WS; invalid sequence; missing state; trainer deadlock; shutdown while order uncertain.
**Acceptance / evidence gate:** All crash points enter safe state; recovery replays exactly once and checks exchange positions/protection before entries resume.
**Failure risk:** Automatic restarts resubmit unknown orders or proceed with stale authority.  |  **Rollback complexity:** HIGH due to pending exchange state.
**Rollback and state recovery:** Stop services, freeze new entries, restore supervisor logic with authority gates; preserve append-only order log.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=MANDATORY | telemetry=MANDATORY | compatibility=STARTUP/STATE.
**Hard stop:** Thread heartbeat/death detection alone is not successful state recovery.

### 21. S20 — Dashboard event correctness and end-to-end decision/execution telemetry
**Findings:** P1-19 + all observability/tracing coverage  |  **Nature:** CORRECTS / HARDENS  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S01, S03–S19 telemetry contracts. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `monitoring/web_app.py`; `monitoring/publisher.py`; `monitoring/control_plane_store.py`; `monitoring/static/app.js`; `reporting/run_report.py`; `runners/live_trading_controller.py`; `core/path_manager.py`; `tests/test_runtime_observability.py`; `tests/test_monitoring_web_app_snapshot.py`; `tests/test_dashboard_snapshot_state_retention.py`.
**Config change / governance:** Event IDs/cursors, retention policy, latency/lag SLOs and mandatory alert thresholds; secure redaction and operator action audit.
**Expected runtime effect:** Dashboard views canonical positions/PnL and catches all event updates, including after capped event window.
**Tests and regression focus:** 51st/500th event; reconnect replay; WS sends heartbeat but data stale; duplicate events; source-of-truth matching; alert on unprotected position.
**Acceptance / evidence gate:** Monotonic event sequencing and full decision→intent→order→fill→PnL lineage; dashboard readouts reconcile to canonical ledger.
**Failure risk:** Dashboard projects stale economics as current or operator authorization drifts.  |  **Rollback complexity:** MEDIUM for read projection; HIGH for controls.
**Rollback and state recovery:** Switch to stable read-only projection with alerts preserved; never roll back control security or canonical ledger.
**Verification flags:** retrain=N | paper=Y | live-candidate=Y | recon=Y | telemetry=MANDATORY | compatibility=WEBSOCKET/UI-SCHEMA.
**Hard stop:** Connected UI heartbeat cannot be taken as evidence that events or prices are fresh.

### 22. S21 — Active universe, liquidity rotation and still-open symbol retention
**Findings:** P2-24  |  **Nature:** HARDENS / VALIDATES  |  **Implementation risk:** HIGH
**Dependencies / order reason:** S07, S10–S11, S15, S17, S20. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `runners/market_data_controller.py`; `runners/tech_rev_controller.py`; `core/symbol_management.py`; `config/tradable_symbols.yaml`; `config/system_settings.yaml`; `tests/test_market_data_universe_refresh.py`; `tests/test_symbol_lifecycle_readiness.py`; `tests/test_symbol_warmup_gate_step5.py`.
**Config change / governance:** Validate spread/depth/funding/liquidity, contract status, warmup, model readiness, cooldown and blocked-state eligibility before entry.
**Expected runtime effect:** Potentially smaller universe and trade frequency; no opened exposure dropped by rotation.
**Tests and regression focus:** Volume-rank flip, thin market, widened spread, delisted pair, blocked symbol with open position, source snapshot missing, restart.
**Acceptance / evidence gate:** Every active entry-eligible symbol passes current exchange, data, model, and liquidity gates; removal reason and supervision state recorded.
**Failure risk:** Rotation exits an eligible instrument or abandons an open exposure.  |  **Rollback complexity:** HIGH when open positions exist.
**Rollback and state recovery:** Revert entry ranking logic only behind disabled entries; preserve open symbol risk supervision and reconciled state.
**Verification flags:** retrain=CONDITIONAL for new symbol | paper=MANDATORY | live-candidate=Y | recon=Y | telemetry=Y | compatibility=SYMBOL-LIFECYCLE.
**Hard stop:** Rotation quality improvements are not presumed alpha; they require observed spread/depth evidence.

### 23. S22 — Feature drift, RF/GBM/confirmatory/HMM ablations and HOLD/LONG/SHORT calibration
**Findings:** P2-21, P2-22 + signal/regime dimensions  |  **Nature:** VALIDATES / OPTIMIZES ONLY ON EVIDENCE  |  **Implementation risk:** MEDIUM research / HIGH if incorrectly promoted
**Dependencies / order reason:** S12–S16, S18–S21. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `features/timeframe_fusion.py`; `features/feature_store_contract.py`; `training/rf_training_engine.py`; `training/gbm_training_engine.py`; `training/dqn_training_engine.py`; `training/hmm_regime_training_engine.py`; `inference/trade_decision_logic.py`; `inference/regime_filter.py`; `inference/hmm_regime_confirmer.py`; `monitoring/publisher.py`; `tests/test_hmm_shadow_evaluation_step11.py`; `tests/test_trade_decision_logic.py`.
**Config change / governance:** Keep current confirmatory blend weight zero and regime shadow until an independent preregistered benefit test warrants change.
**Expected runtime effect:** Research diagnostics and drift holds can change readiness; *no* automatic HMM/DQN influence or threshold optimization.
**Tests and regression focus:** Paired candidate/incumbent on same OOS dates, RF vs GBM vs blend, HMM on/off, confirmatory on/off, class reliability, selective net EV, drift/missingness.
**Acceptance / evidence gate:** Each feature/model policy has independent effect estimate, uncertainty, cost-aware ablation and drift baselines; non-beneficial complexity rejected.
**Failure risk:** Overfitting from many ablations silently selects an ineffective policy.  |  **Rollback complexity:** LOW research / HIGH if blindly promoted.
**Rollback and state recovery:** Keep previous validated candidate pointer under unchanged recipe; disable optional model influence; preserve independent test rows.
**Verification flags:** retrain=Y for experiments | paper=MANDATORY for any effect | live-candidate=CONDITIONAL | recon=N | telemetry=MANDATORY | compatibility=SIGNAL/SCHEMA.
**Hard stop:** Do not promote policy on high backtest precision alone or rename supervised policy as reinforcement learning.

### 24. S23 — Independent integrated replay and fault-injection acceptance
**Findings:** ALL  |  **Nature:** VALIDATES  |  **Implementation risk:** HIGH gate
**Dependencies / order reason:** S01–S22. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `tests/` (all existing tests, preserve relevant assertions except known-wrong ones); `services/backtest_engine.py`; `runners/live_trading_controller.py`; `execution/` modules; `monitoring/` modules; `main.py`.
**Config change / governance:** Freeze candidate config and dependency set; deterministic fixture seeds; no real-capital credentials.
**Expected runtime effect:** No deployment; validate end-to-end contract under deterministic replay and bounded stress.
**Tests and regression focus:** All original 24 findings; all user dimensions; 0 future leak; 0 missing fills; kill every order transition; reconcile balances; security; latency; stale prices.
**Acceptance / evidence gate:** All must-pass assertions pass; defects have evidence links, coverage owners, and reproducible test outputs; no suppressed failures.
**Failure risk:** Tests pass despite wrong or incomplete oracles.  |  **Rollback complexity:** LOW rollback; repeat verification after fixing tests.
**Rollback and state recovery:** No exchange state; on failure, return to responsible step with controlled plan version amendment.
**Verification flags:** retrain=AS REQUIRED | paper=Y fixtures | live-candidate=N | recon=Y simulated | telemetry=MANDATORY | compatibility=REGRESSION-SUITE.
**Hard stop:** A test suite that asserts an intentional bug is not valid proof of safety.

### 25. S24 — Forward paper validation with frozen candidate and realistic cost replay
**Findings:** P1-10, P1-13, P1-16, P2-21–P2-24  |  **Nature:** VALIDATES  |  **Implementation risk:** MEDIUM methodological
**Dependencies / order reason:** S23. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `config/system_settings.yaml`; `runners/live_trading_controller.py`; `services/backtest_engine.py`; `monitoring/publisher.py`; `reporting/run_report.py`; `core/training_registry.py`.
**Config change / governance:** Paper mode, frozen model/feature/label/strategy policy, preregistered evaluation window/stop rules, fixed cost assumptions reviewed from evidence.
**Expected runtime effect:** No actual exchange orders; evaluate real-time data availability, decision rate, actual HOLD/LONG/SHORT selection and cost-conditioned outcomes.
**Tests and regression focus:** Data quality/WS uptime, drift, latency, proposed vs filled simulated opportunity, hit rate, drawdown, trading frequency, all symbol lifecycles.
**Acceptance / evidence gate:** Forward paper sample reaches preregistered statistical power and operational observation window; no critical incident or unexplained telemetry gap.
**Failure risk:** Non-frozen candidates or cost assumptions bias the forward evaluation.  |  **Rollback complexity:** LOW financial; repeat test window.
**Rollback and state recovery:** Deactivate candidate and revert versioned paper model pointer; archive all telemetry/metrics and failures.
**Verification flags:** retrain=FROZEN for eval except governed shadow experiments | paper=MANDATORY | live-candidate=PRECONDITION | recon=SIMULATED | telemetry=MANDATORY | compatibility=EVAL-ARTIFACT.
**Hard stop:** 5 trades and 2-hour logs are categorically insufficient to prove edge or reliability.

### 26. S25 — Binance testnet trade/stop/fill/reconciliation qualification
**Findings:** P0-03, P0-05–P0-09, P1-14–P1-15  |  **Nature:** VALIDATES  |  **Implementation risk:** CRITICAL operational if endpoint misconfigured
**Dependencies / order reason:** S23–S24, exchange testnet credentials and latest API documentation. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `execution/exchange_adapter.py`; `execution/exchange_reconciliation.py`; `execution/order_router.py`; `runners/live_trading_controller.py`; `core/live_readiness_guard.py`; `config/system_settings.yaml`; `reporting/run_report.py`.
**Config change / governance:** Use testnet only; small bound notional and isolated account, no production endpoint; explicit operator signoff and alarm channel.
**Expected runtime effect:** Real testnet exchange orders/fills/conditional protection without real capital; confirms venue-contract and recovery behaviors.
**Tests and regression focus:** Signed read + market order; ACK/partial fill; stop installed and survives process kill; exchange fault/REST outage; account mode; wallet/equity bridge.
**Acceptance / evidence gate:** All testnet orders, exchange positions, fees/funding events where available and protective coverage reconcile; no UNKNOWN unresolved and no unaudited orders.
**Failure risk:** Testnet endpoint/account-mode configuration accidentally targets production.  |  **Rollback complexity:** VERY HIGH if endpoint wrong; otherwise confined to testnet.
**Rollback and state recovery:** Disable new testnet orders, reconcile/close or intentionally protect existing testnet positions, preserve all order IDs and ledger.
**Verification flags:** retrain=N | paper=ALREADY QUALIFIED | live-candidate=NEXT | recon=MANDATORY | telemetry=MANDATORY | compatibility=EXCHANGE API.
**Hard stop:** Testnet validity does not imply real-money alpha or guaranteed venue behavior.

### 27. S26 — Formal live-candidate governance gate (NO DEPLOYMENT in this plan)
**Findings:** ALL  |  **Nature:** GOVERNANCE  |  **Implementation risk:** CRITICAL governance
**Dependencies / order reason:** S00–S25 complete and verified; external operator/legal/exchange readiness. This step is assessed at this point to keep dependent contracts and external side effects controlled.
**Exact existing files to inspect / potential change:** `core/live_readiness_guard.py`; `core/artifact_contract.py`; `core/training_registry.py`; `config/system_settings.yaml`; `main.py`; `monitoring/control_plane_store.py`; `reporting/run_report.py`.
**Config change / governance:** Keep production off unless separate audited release change authorizes real capital, explicit limits, staffed incident response and verified permissions.
**Expected runtime effect:** Decision only; no live orders implied or authorized by this plan.
**Tests and regression focus:** Independent release review: zero open findings, emergency exit drill, reconciliation, alerting, model/feature hashes, paper/testnet evidence, account restrictions.
**Acceptance / evidence gate:** Signed engineering+risk approval record OR reject; if rejected, remain paper/testnet. Record deployment criteria separately.
**Failure risk:** Approval mistaken for authorization to deploy real-capital trading.  |  **Rollback complexity:** LOW procedural until activation; STRICT release-control dependency.
**Rollback and state recovery:** Approval can be withdrawn; any later real-capital release requires distinct controlled activation and operational unwind procedure.
**Verification flags:** retrain=N | paper=Y evidence | live-candidate=GATE ONLY | recon=MANDATORY | telemetry=MANDATORY | compatibility=FULL SYSTEM.
**Hard stop:** Do not conflate approval to implement with approval to trade live.

## 4. All originally requested dimensions — zero-orphan coverage
| Requested dimension | Required remediation and validation steps |
|---|---|
| Multi-timeframe synchronization | S12,S13,S23 |
| Price cache freshness | S10,S23 |
| Websocket reliability | S10,S19,S20,S23 |
| Training cadence correctness | S18,S23 |
| RF/GBM/Confirmatory Policy effectiveness | S14,S15,S22,S24 |
| Feature usefulness | S22,S24 |
| Feature drift | S20,S22,S24 |
| Label quality | S13,S16,S23 |
| Signal calibration | S14,S22,S24 |
| HOLD/LONG/SHORT decision quality | S14,S16,S22,S24 |
| Slippage realism | S16,S24 |
| Execution latency | S19,S20,S23,S25 |
| Manifest readiness integrity | S15,S23 |
| Promotion governance | S15,S22,S26 |
| Active universe correctness | S11,S21,S23 |
| Blocked-symbol handling | S11,S21,S23 |
| Trade lifecycle traceability | S03,S04,S05,S20,S23 |
| Dashboard telemetry correctness | S01,S20,S23 |
| Runtime observability completeness | S19,S20,S23 |
| Risk engine behavior | S06,S09,S17,S23 |
| Scheduler robustness | S18,S19,S23 |
| Controller fault tolerance | S19,S23 |
| Exchange reconciliation integrity | S02,S04,S05,S06,S23,S25 |
| PnL correctness | S05,S06,S16,S23 |
| Symbol rotation quality | S11,S21,S24 |
| Training/inference skew | S12,S13,S14,S15,S23 |
| Regime adaptation quality | S22,S24 |
| Trade frequency realism | S16,S18,S22,S24 |
| Portfolio-level behavior | S06,S09,S17,S23,S25 |
| Authentication/control-plane security (additional) | S01,S20,S23 |
| Exchange order compatibility (additional) | S02,S07,S08,S25 |
| State migrations / rollback correctness (additional) | S00,S03–S06,S08,S15,S23 |

**Completeness proof:** 24/24 numbered original findings and all 29 original requested dimensions have at least one explicit step mapping; 3 additional cross-cutting domains are mapped. This is a *plan coverage assertion*, not evidence that a fix is complete or that no unknown problems exist.

## 5. Required additional forensic evidence before individual gates
No missing data is silently inferred; a missing artifact remains an explicit `BLOCKED BY EVIDENCE` item. The original ZIP contains `data/runtime/` and `data/market_storage/` as directories without state/database files, while `logs/process_roles/` contains three archived logs. Source paths here are expected application outputs, not files claimed to be inside the ZIP.

| Evidence category | File / database / log / telemetry / controller output to inspect | Why needed |
|---|---|---|
| Historical market database | `data/market_storage/<SYMBOL>.db` and its kline tables; original OHLCV raw event and optional context captures | Verify event-time contracts, MTF availability, actual coverage gaps and re-sample labels. |
| Order and execution state | `data/runtime/order_state.json`; `data/runtime/execution_journal.jsonl` | Recover exact client/exchange IDs, duplicates, pending ACKs, partial fills and crash ordering. |
| Position and circuit state | `data/runtime/portfolio_state.json`; `data/runtime/circuit_breaker_state.json` | Reconcile positions/fees/risk and test safe reduction without overwriting exchange state. |
| Symbol and control-plane state | `data/runtime/symbol_state.json`; `data/runtime/symbol_actions.json`; `data/runtime/control_plane_state.json`; `data/runtime/control_plane_actions.jsonl` | Confirm active/blocked symbols, operator controls, and evidence of forced close. |
| Decision and market price telemetry | `data/runtime/decision_journal.jsonl`; `data/runtime/market/latest.json`; WS event timestamps/sequence and controller output | Trace market observation→feature→prediction→risk→intent and freshness. |
| Model manifests and trained artifacts | `data/model_storage/**` model binaries, manifests, incumbents and `data/runtime/training_registry.json` | Confirm actual active model identity/readiness, feature versions, leakage/promotion status and rollback pointers. |
| Training progress | `data/runtime/training_progress/<MODEL>__<SYMBOL>__<INTERVAL>.json`; trainer scheduling decisions, CPU/load metrics | Determine if 1,128 successful training events were duplicate/redundant or actually selected. |
| Exchange-authoritative exports | USDⓈ-M exchange account/order history/trade IDs/commission asset/funding/income/equity/margin/position-mode/leverage; sanitized exports | Certify real fills, PnL, reconciliation and protection. Never provide API secrets. |
| Live venue market microstructure | Timestamped best bid/ask, depth, trade prints, market-volume, basis and funding history | Validate slippage, price freshness, realistic fill opportunity, liquidity constraints and trade frequency. |
| Dashboard monitoring history | `data/runtime/monitoring/monitoring_events.json`; `monitoring_state.json`; producer snapshot/event IDs; browser WS recordings | Verify dashboard event cap, source-of-truth readouts and consumer lag. |
| Operating controller logs | Archived `logs/process_roles/run_runtime.log`, `run_trainer.log`, `run_dashboard.log` are present, but need extended incident logs/restart probes | Measure end-to-end uptime and recovery, not just two-hour training success window. |
| Release baseline | Exact git commit/dependency lock, OS/Python versions, CI results and read-only deployment/runtime manifest | Reproducible implementation, rollback and change-control proof; not established by ZIP. |

**Security constraint:** obtain only masked/sanitized exports; never upload Binance API keys, full signing secrets, session tokens or personal identifiers. Production exchange queries are not necessary for this static plan and are out of scope without separate explicit authorization.

## 6. Isolation, staged rollout and observation requirements
- **Offline only:** S00–S07 unit/contracts; isolated HMAC/signature prepared-request test; no real credentials; instrument mode/filter fixtures; deterministic event ledger and data replay.
- **Exchange testnet only:** S02 signature smoke once safe; S04–S09 order/recovery/protection flows only after full integrated preflight; S25 end-to-end final qualification. Never point an intermediate branch to a production account.
- **Paper-only:** S10–S24 price availability, data/feature/label contracts, score calibration, manifest promotion, universe rotation, EV/hold policy and all experimental regimes. Testing account mechanics and demonstrating strategy edge are separate criteria.
- **Fault injection:** kill/timeout between intent creation, send, ACK, fill, state persistence, stop placement, fill reception, reconciliation and dashboard publication; network partition, clock skew, stale price and missing model artifact. Verify no duplicate fill or orphaned unprotected position.
- **Observation windows:** define from prospective SLO/stats design in S23 before test begins (market availability, latency p50/p95/p99, reconnect, model freshness, zero severity-one incidents, sample size/power and regime variety); do not invent a fixed 24-hour "institutional" threshold.
- **Staged rollout:** S23 deterministic replay → S24 shadow/forward paper → S25 Binance testnet (limited account) → S26 formal approval evaluation. A separate future release would be required for real capital.

## 7. Detailed rollback policy
1. **Read-only & control plane:** tag source/config, archive current UI event schema; rollback can retain read-only dashboard; never re-enable auth bypass (S01, S20).
2. **Economic events:** persist append-only ledger, known order IDs, fill IDs, protection IDs and external reconciliation snapshot. A trade cannot be undone by source rollback. On defect, disable entries, reconcile, leave stops intact, and rebuild internal projections forward (S02–S09, S11, S17, S19, S25).
3. **Persistence/schema:** version all state reader/writer contracts; double-read or offline migration only if provably lossless. Preserve old state for audit, but never restore obsolete portfolio JSON over real exchange positions (S03–S06, S11, S15).
4. **ML state:** archive previous candidates and pointers; promote only models compatible with causal feature/label versions. A pre-leakage-fix model is not an acceptable rollback target (S12–S16, S22).
5. **Price/universe operations:** rollback to the more conservative no-entry policy; continue monitoring and protecting pre-existing symbols until exchange flat (S10–S11, S21).
6. **Audit rollback event:** capture timestamp, actor, reason, config/code hash, old/new state hashes, reconciled positions/protective coverage, control status and post-rollback WS/telemetry health. Any unresolved mismatch remains a release blocker.

## 8. Auditor decision and implementation authorization
**As-submitted prompt:** **NOT APPROVED** as sole execution authority. It can omit finding closure, retain unsafe tests, ignore migrations, and over-freeze a plan despite newly observed failures.

**Corrected non-live plan:** **APPROVED WITH CHANGES** as a staged implementation blueprint **conditional on** S00 baseline containment and per-step evidence gates. The word "approved" does not authorize running the bot, using Binance credentials, changing production endpoints, placing orders, or deploying a change to live capital.

**Final live decision:** **NOT APPROVED**. Separate completion proofs are required for S23 integrated, S24 forward paper, S25 Binance testnet and S26 risk/engineering signoff.

### Definition of done for each defect
Each issue must have: finding ID → root-cause evidence → owner and code changes (or explicit evidence blocker) → targeted unit test → cross-component regression → integrated replay → configuration/state migration proof where needed → runtime/testnet observation → independent auditor signoff. Allowed states: OPEN, BLOCKED_BY_EVIDENCE, IN_PROGRESS, CODE_FIXED, TESTED, PAPER_VERIFIED, TESTNET_VERIFIED, VERIFIED_CLOSED, or ACCEPTED_RISK_WITH_APPROVAL. No status can be marked closed solely because a test file exists.

### Plan amendment rule
If a hidden defect, Binance API contract change, runtime mismatch, regression, unreviewed dependency or statistical contamination is found, stop the affected release, register a numbered change request with new evidence and severity, map it to impacted original finding/dimension, run Auditor review, update this plan version and rollback/acceptance criteria, and resume only after approval. This preserves traceability without pretending the original plan could predict every defect.

## 9. Prompt sufficiency verdict and indispensable amendments
Keep the user’s Planner–Auditor–Implementer split, but describe the Implementer in **read-only feasibility mode** during plan validation. Require the Planner to enumerate the 24 finding IDs, the Auditor to challenge every one against source and tests, and the Implementer to perform **no edits** until a later explicit implementation request and step-wise acceptance. Demand independent coverage proof across all 29 dimensions, baseline hash, staging tests, migration/rollback and explicit missing runtime evidence. Reject absolute "ONLY source of truth / no extra steps" wording; use a versioned, change-controlled baseline instead.

**Final recommendation:** implement only after the owner authorizes a specific step on a non-live branch and its required evidence is available. Begin with S00 and S01; do not activate any exchange signing or live-trading functionality until S02–S11 are integrated with S23–S25 acceptance.
