# Autonomous mission specification — v5

## What the owner authorized
Unattended **offline and paper engineering** of the Auto_Trader repository, including proactive online technical research, feature/model/strategy replacement, architecture refactoring, tests, reproducible experiments, paper runtime launch *by the supervisor*, monitoring, and iterative fault remediation. Changes to originally proposed methodologies are encouraged when facts demonstrate a better alternative.

## What this does not authorize
Real-money trading, handling production Binance keys, managing accounts or transferring funds, bypassing authentication, bypassing safety checks, accessing personal secrets, unreviewed destructive database migrations, modifying the supervisor's safety enforcement, or pretending that a paper run or backtest certifies performance.

## Operating sequence and ownership
The supervisor must repeatedly execute:

```
P01 evidence inventory & source audit
  -> P02 planning + alternative-method research
  -> P03 independent challenge + versioned plan
  -> P04/S00 baseline & containment
  -> P05 implement one current approved change
  -> deterministic unit/integration/fault tests
  -> P06 independent adversarial validation
  -> continue repairs until all required code gates pass
  -> P07 cumulative end-to-end replay
  -> P08 bounded paper run (supervisor only) + independent review
  -> P09 read-only testnet/API contract review
  -> P10 independent closure assessment
  -> P11 recurring surveillance and new finding intake
  -> replan automatically on new supported fault or materially better approach
```

The 24 prior findings and 29 dimensions remain traceable. S00–S26 are the **starting plan**, not final truth. Add `NEW-###` and changed steps under a dated change request, while preserving previous IDs. Implement different changes only in isolated branches or worktrees; no parallel write access to coupled execution/risk/accounting modules.

## Freedom to replace old methods — evidence test
When considering a material replacement (e.g. RF -> new model, HMM removal, feature store rewrite, adaptive thresholds, a new risk model, new portfolio allocator, a redesigned event-driven architecture):

1. Explain the failure of the present method with code/runtime evidence.
2. Identify at least two credible alternatives, including the minimal-fix baseline.
3. Verify version-specific external documentation and primary methodological evidence (time-stamped links).
4. Define a falsifiable experiment **before** training candidates.
5. Use point-in-time datasets and untouched chronological holdouts; avoid repeated holdout reuse by maintaining a new forward evidence window.
6. Measure cost-aware performance, calibration, coverage, drawdown, turnover, leakage, latency, robustness and statistical uncertainty; test no-trade behavior.
7. Select the safest validated candidate, or retain the baseline if superiority is not established.
8. Independently verify the diff, model manifest and compatibility, and document the reversible rollback.

Do not rewrite the complete system merely because an agent prefers a fashionable architecture. Do not preserve a provably flawed approach merely because it appears in the original plan.

## Continuous work and convergence
A `PASS` requires file paths and real successful commands. `FAIL`/`BLOCKED` must record exactly what to fix/obtain. Set cost/compute/time and retry budgets in the **external** supervisor policy. Persist state between sessions. When no further safe action is possible, stop with `BLOCKED` and produce the blocker list rather than improvising a trade credential or silently lowering required tests. A stopped system can be resumed after an external configuration change.

## Paper observation schema
Each paper session should capture: repo commit/config hash, market-data source and age, exchange server clock offset, websocket gap/reconnects, candle as-of timestamps, feature/model/manifest versions, decision IDs, HOLD/LONG/SHORT probabilities, risk decisions/reasons, simulated orders/fills/costs, equity/position state, timeouts/retries, controller liveness, instrument universe and block reasons, dashboards and event-stream continuity, process exit code and log checksum.

A paper run **with no market feed or missing durable traces** is not evidence of healthy runtime. Record missing observations as gaps and reopen telemetry/quality findings.

## Independent completion criteria
Never claim 'all problems fixed' based only on a passed pytest suite or a 24-hour paper run. Report which code issues are genuinely closed, which were closed only in paper/testnet, which efficacy questions remain open, and which risks are accepted by an external authority. Optimizing a strategy is an ongoing empirical process, not a finite guaranteed repair job.
