# Copy into each new implementation/review session

Run prompt: `P__`  |  Requested step (if applicable): `S__`  |  Prompt mode (if applicable): `PREPARE` / `REVIEW` / `PLAN`

## Required attached files (attach real files, not just filenames)
- `GLOBAL_CONTRACT.md` and the corresponding `Prompts/P__...md` (or complete playbook).
- Current **complete** Auto_Trader repository ZIP: <actual filename> and SHA-256: <value>.
- Prior forensic report and approved plan version: <name/hash>.
- Latest `02_FINDING_REGISTER.csv`, `03_DIMENSION_COVERAGE.csv`, `05_STEP_LEDGER.csv`, `06_CHANGE_REQUESTS.md`, regression and release gate records.
- Latest test/run outputs and before/after repository ZIPs if reviewing a step.
- Sanitized runtime DB/export/WS/order/PnL evidence if the stage requires it.

## Facts, authorization and constraints
- Active branch/commit, Python/dependency versions: <verified values or UNKNOWN>.
- Approved mode: **NO LIVE ORDERS**. TESTNET authorized? <NO unless separately explicitly approved>.
- Current stage and prior gate verdict: <exact previous verdict + evidence>.
- New incidents, revisions, config migrations and blocking evidence: <exact items>.
- Secret hygiene verified: <yes/no; do not include actual secrets>.

**Instruction:** Reopen every attached actual artifact. Apply Global Contract and this stage prompt. Do not assume older chat files are present. Do not skip gates or invent runtime proof. Preserve all 24 original finding IDs, all newly discovered IDs, and all 29 dimensions.
