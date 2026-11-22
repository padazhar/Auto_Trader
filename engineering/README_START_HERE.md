# Auto_Trader Complete Repair Prompt Kit — v3.0

Contains the rewritten six user prompts, expanded into 11 sequential stages, plus a mandatory evidence/safety contract and prepopulated tracking templates.

**What to do:** Attach `GLOBAL_CONTRACT.md`, the current Auto_Trader repository ZIP, `Auto_Trader_Institutional_Forensic_Audit.md`, `Auto_Trader_Validated_Execution_Plan_v1.md`, and the chosen `Prompts/Pxx_*.md`. Run P01, P02, P03, P04. Then run P05 **and P06 once for each approved step** (initial baseline contains S00 through S26); failed validation returns to the same step. Run P07 only after integrated code passes; P08 is paper PREPARE then REVIEW after actual evidence is available; P09 is testnet PLAN then REVIEW **only with explicit testnet authorization**; P10 is final audit. Use P11 for later changes.

**Prompt mapping:** source originals 1→P01, 2→P02, 3→P03, 4→P04/P05, 5→P06, 6→P07. P08–P11 are safety, real-world verification and lifecycle stages absent from the source.

**Why not promise a perfect bot:** A prompt cannot guarantee catching all future unknown faults or earning trading profits. Successful completion means each known issue has an evidence-backed disposition, tests and integrations have been conducted, and unresolved facts are transparently reported. The process **blocks false approval** rather than pretending missing data is a pass. Live-money deployment is never automatically authorized.

**Evidence limitations of the original uploaded materials:** The baseline archive is `Auto_Trader_1.zip` and the earlier audit has static source evidence with archived ~110-minute logs, not proof of a 24-hour forward runtime. An original draft references `Auto_Trader_80.zip`, UI PDF and a forensic checklist, which cannot be presumed present. The tracker records prior findings as OPEN until checked against the *latest* code and validated.

**Files:** `GLOBAL_CONTRACT.md` is the non-droppable role/safety/findings/dimension contract; `Trackers/KNOWN_FINDINGS_24.csv`, `DIMENSIONS_29.csv`, `STEPS_S00_S26.csv` are baselines, not evidence that anything was fixed; `RUN_HANDOFF_TEMPLATE.md` is a simple handoff envelope; `Prompts/P01...P11` are full, paste-ready prompts. The standalone all-in-one playbook is separately provided as `Auto_Trader_Complete_Repair_Prompt_Playbook.md`.

**Always preserve:** actual test output and code hashes, config/manifest/migration provenance, exchange-authoritative economic state, security protections, and change approval. Use testnet only after permission. Avoid including real API keys/secrets in the repository package.
