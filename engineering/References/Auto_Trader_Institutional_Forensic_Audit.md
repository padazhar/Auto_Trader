# Auto_Trader — Institutional Quantitative Trading System Forensic Audit

**Audit date:** 22 November 2026  
**Input:** `Auto_Trader_1.zip` (supplied repository archive)  
**Method:** source-level code tracing, configuration cross-check, static examination of archived process logs, comparison with Binance USDⓈ-M API documentation. The application was **not started**, no API keys were used, and no exchange, model-training, backtest, or execution traffic was generated.  
**Overall decision:** **NO-GO for live capital.** Default committed configuration is paper/testnet-oriented, which should be preserved until blocking defects are remediated and independently tested.

## 1. Audit scope and evidence limitations

The archive contains approximately **223 Python source files** and **119 Python test files**, configuration, dashboard assets, and three archived process logs. It does **not** include an independently verifiable exchange trade/fill export, full market OHLCV database, promoted binary model artifacts, full runtime order/portfolio journals, or populated live dashboard state. Therefore, this audit **can identify implementation defects**, but **cannot certify actual profitability, fill quality, trade frequency, active-model quality, or realized live performance**. File paths below are relative to the extracted repository root `Auto_Trader/`; line numbers refer to the submitted revision.

**Log window:** 27 October 2026, 20:12:49–22:03:16 (~110.5 minutes). The runtime log contains **15,138** lines and **13,232** lines reporting a market websocket snapshot write; there was an initial incomplete snapshot warning. The training log contains **8,937** lines and **1,128** `training success` records: RF **377**, GBM **318**, HMM **310**, legacy-DQN/confirmatory **123**. These are training-success events, **not** evidence of winning trades, genuinely independent evaluated models, or promotion into live authority. The dashboard log has only a startup line. These logs are not recent operating proof as of the audit date.

### Severity and evidence language

- **P0 / Critical:** live-trading stop condition, unprotected exchange exposure, security/control-plane issue, material trading-state corruption, or severe statistical contamination.
- **P1 / High:** materially impairs forecast validity, portfolio control, reconciliation, or production reliability.
- **P2 / Medium:** incomplete, uncalibrated, unverified, or operationally weak feature; should have assigned owner and test before broader deployment.
- **Confirmed:** behavior directly visible in code or file. **Conditional risk:** follows from code under specified runtime conditions. **Not verifiable:** absent required datasets, artifacts, or operational evidence.

## 2. Executive summary

The architecture is **more sophisticated than a toy trading script**: it separates market-data ingestion, technical-review/training coordination, inference, execution, risk management, reporting, and dashboard functions; uses model manifests and feature-set signatures; persists orders; includes idempotency and trade lineage; handles partial websocket snapshots, warmup/rotation, circuit breakers, and default paper trading; and has explicit live-startup and canary controls. Those are meaningful foundations.

However, the implementation currently fails essential institutional controls in four independent layers:

1. **Research data validity:** higher-timeframe feature joins use opening times; RF uses a mismatched timeframe for labels; pre-split backfill leaks future feature values; validation data is reused for calibration and claimed performance; purge length is shorter than label horizon.
2. **Execution correctness:** signed Binance requests likely fail due to byte-string mismatch; the default market-order response is ACK while the system expects immediate fills; failed open-order discovery is interpreted as no open orders; idempotency keys can collide across separate legitimate trades; exits and fills do not support accurate partial lifecycle accounting.
3. **Live position safety:** protective exits are controller-triggered market orders rather than durable exchange-native conditional protection; reconciliation is position-level, not a full trade/fee/funding ledger; ordinary EXIT risk approval is evaluated after the circuit breaker and can be blocked by it.
4. **Operations/security:** dashboard authentication is disabled while state-changing endpoints remain enabled; a DB-close fallback receives a current timestamp and can satisfy the supposedly strict execution price freshness gate; dashboard event broadcasting has a capped-queue update defect.

**Do not infer that the strategy has or lacks alpha** from this evidence. Statistical validity must be restored before the model can be meaningfully evaluated.

## 3. Prioritized code-confirmed findings

### P0-01 — Multi-timeframe look-ahead contamination

**Evidence:** `runners/market_data_controller.py:959–967`; `features/timeframe_fusion.py:289–294,297–356` (especially `339–346`).

Binance OHLCV `timestamp` is the **candle opening time**. `build_fused_frame` engineers indicators for the entire source candle and runs `pd.merge_asof` with `left_on="timestamp"`, `right_on=f"{interval}_timestamp"`, `direction="backward"`. A 10:00–11:00 1h candle can therefore appear in 10:00–10:59 lower-timeframe historical rows even though its final high, low, close, and volume were unavailable until after 11:00. This is **historical look-ahead bias** and a train/inference availability mismatch; the primary live collector separately restricts OHLCV updates to last-closed bars (`market_data_controller.py:1031–1059`), which does not repair historical merges. `features/timeframe_fusion.py:293` also sets every `{source}_age_bars_vs_{anchor}` feature to **zero**, so purported age metadata is not informative.

**Fix:** label every source observation with `available_at_ms = candle_open_ms + interval_ms + allowed_publication_delay_ms`; label anchor row by the **decision time after the anchor closes**; backward-asof join using availability time, not open time. Reject or mark genuinely absent source frames; compute actual age per row; write point-in-time replay tests that assert zero dependence on future candles. Increment `feature_store.recipe_version` and invalidate affected artifacts.

### P0-02 — RF labeling uses the wrong timeframe

**Evidence:** `training/rf_training_engine.py:184–196`, notably `anchor_interval=cfg.frames.anchor_interval`; `config/system_settings.yaml:49–51,72–76`; comparison `training/gbm_training_engine.py:283–292`.

RF trainers loop over 1m/3m/5m/15m models but generate targets against the configured **15m anchor close/ATR**. For example, on 1m rows the label's forward shift of five rows is applied to a stepwise 15m close rather than the 1m close. The resulting target semantics are inconsistent with the declared 1m/3m/5m modeling objective and with GBM labeling, which uses the actual `interval` argument. This can alter class balance, HOLD propensity, and forward-return magnitudes.

**Fix:** use `anchor_interval=interval`; verify `future_return[t]` matches the actual close from the proper horizon on that timeframe; regenerate labels and all affected RF artifacts; fail training if target time interval or label horizon does not match the manifest.

### P0-03 — REST request signature differs from transmitted query

**Evidence:** `execution/exchange_adapter.py:149–162`; `execution/exchange_reconciliation.py:75–88`.

Both `_sign_params` functions create the HMAC from keys in **sorted** order, but `requests` sends the returned dictionary in its original insertion order. Thus the payload that was signed can differ from the actual query string, including for ordinary `place_order` calls whose parameter dictionary begins `symbol,side,type,...`. Binance validates the signature against the actual transmitted parameter string. A valid secret alone will not fix this. Proper URL escaping of values is also necessary.

**Fix:** construct one canonical `urlencode`-encoded payload, sign **those exact bytes** and submit the same serialized string; add deterministic signing tests comparing the transmitted prepared request to the HMAC input; apply the shared signer to order submission, order lookups, order recovery, and account reconciliation. Do not deploy live until testnet signed account reads and a test order are verified.

### P0-04 — Monitoring/control API has no authentication

**Evidence:** `monitoring/auth.py:19–49`: authentication methods explicitly return `None`/`True`; `monitoring/web_app.py:703–715` announces `auth_required: false`; state-changing operations at `:780–808`, `:851–872`, `:879–954`.

Any actor able to reach the server can invoke trading pause/resume, circuit reset, model rollback, and symbol block/unblock/retry. `127.0.0.1` binding in `config/system_settings.yaml:446` limits direct remote exposure by default but does not establish authorization for local users, SSH tunnels, reverse proxies, or accidental deployment with a wider interface. The configured token does **not** protect these endpoints.

**Fix:** implement mandatory authentication and authorization for all state-changing APIs and websocket feeds, CSRF protection where cookie-based auth is used, immutable actor attribution, secure secret loading and rotation, and default-deny controls. Separate read-only dashboards from operator control-plane privileges; protect reverse-proxy access with TLS and network policy.

### P0-05 — Live positions lack exchange-native protective orders

**Evidence:** `runners/live_trading_controller.py:523–550,565–610`; `execution/exchange_adapter.py:164–215`.

The system detects SL/TP locally using a polled/runtime price and then sends a **MARKET** reduce-only order. There is no confirmed durable exchange-native stop/take-profit order being installed when an entry fills. An application outage, network issue, stale price, exchange rejection, or an unobserved fill can prevent the protective action. The configured `execution_price_refresh_seconds: 0.5` does not guarantee a 500 ms end-to-end protection response.

**Fix:** after confirming entry fills, install and verify exchange-native conditional protection using Binance's currently supported USDⓈ-M algo-order interfaces; persist the protective order IDs, trigger type, quantity, and status; continuously reconcile positions **and protection**; repair/rearm on partial fills or replace/cancel operations; immediately alert and block new entries if any open position becomes unprotected. Test all failure transitions on testnet.

### P0-06 — Order ACK and crash recovery can orphan live fills

**Evidence:** `execution/exchange_adapter.py:164–215,217–262`; `execution/order_router.py:61–87,89–146`; `runners/live_trading_controller.py:1874–1886`.

The live adapter omits `newOrderRespType=RESULT`, while Binance defaults USDⓈ-M new orders to **ACK**. It emits a `FillEvent` only if the immediate response contains both `executedQty>0` and `avgPrice>0`; otherwise the order remains submitted and the portfolio entry is not recorded. `get_order` discards execution size/price; recovery only examines **currently open orders**, not closed/filled orders or executed trades. An order filled at Binance may be absent from the open-order list and therefore be misclassified as expired/canceled locally. More severely, `get_open_orders` returns an empty list on any exception (`execution/exchange_adapter.py:240–250`), while the router interprets a missing live order as terminal (`order_router.py:113–124`). No complete user-data `ORDER_TRADE_UPDATE` stream or replayable fill accounting was established in the submitted execution path.

**Fix:** treat order outcome as UNKNOWN until an authoritative order/trade query resolves it. Use `RESULT` when appropriate, but **do not rely on response type alone**. Implement user-data stream updates with sequence/reconnect recovery; query individual orders and account trade history; preserve every unique trade ID, partial fill, commission, fee asset, execution timestamp, and order status transition; persist atomically and idempotently. Separate transport failure from an authoritative empty order list. Reconcile before opening any further position after uncertain execution.

### P0-07 — Persisted idempotency key can suppress legitimate future trades

**Evidence:** `execution/order_store.py:55–81`; `execution/order_router.py:61–66`; `runners/live_trading_controller.py:1859–1873`.

The normal entry path does not supply an explicit unique `idempotency_key` or `client_order_id`. The fallback client ID hashes symbol, action, side, quantity, order type, optional prices, reduce-only flag, and previous position; **it omits the decision/run/order-intent identity and timestamp**. Two distinct future entries with the same tuple produce the same client ID. `route_intent` returns the prior persisted order as a duplicate without placing the new one. If the previous order is filled, the returned previous fill may be applied as if the new entry filled, creating a local-only position. The severity of occurrence depends on repeated quantities, which are plausible under fixed risk sizing.

**Fix:** make every **economic intent** globally unique (stable `decision_id` + execution-leg/attempt identifier); retain the same ID only for retries of that exact intent. Provide immutable idempotency records, order lifecycle epoch, and replay tests including repeated same-quantity trades after close, partial fills, restarts, and reversals. Do not use arbitrary current time alone as a retry key.

### P0-08 — Circuit breaker can block ordinary exits

**Evidence:** `execution/risk_engine.py:275–295`.

The risk evaluation checks authority and `circuit.triggered` **before** `action.startswith("EXIT")`, returning `HOLD` for all actions when tripped. Thus normal signal-driven position-reducing exits can be refused during a loss-limit or drawdown circuit event. Separate protective-exit routing may bypass this function but is not a substitute for reliable emergency reduction.

**Fix:** distinguish new/increasing-risk actions from reduction/close actions. Circuit, pause, model unreadiness, and stale signals must **never prevent a verified reduce-only position reduction**. Unknown exchange authority should not silently authorize arbitrary quantity; use an exchange-authoritative emergency exit workflow and operator escalation.

### P0-09 — Stale DB close is stamped as a fresh execution price

**Evidence:** `runners/market_data_controller.py:497–513,743–761,515–531`; `core/market_io.py:271–342`; `runners/live_trading_controller.py:314–347`.

When live market-price inputs are missing, fallback rows can set `price=db_close` and `price_ts_ms=created_ms` (now) rather than the close's market observation timestamp. `latest_symbol_price_status` accepts `db_close` and uses that timestamp to mark freshness; entry gating tests availability and freshness but does not require an approved market-source type. Consequently a stale 15m close may appear fresh under the 2-second execution rule, potentially bypassing the intended no-fresh-price/no-entry safeguard. The snapshot also reports `snapshot_health="ready"` when every symbol merely has a positive price, regardless of provenance.

**Fix:** associate price/event timestamps with the **selected source**, not snapshot write time; distinguish `received_at`, `exchange_event_at`, `bar_open_at`, and `bar_close_at`; exclude `db_close` from live entry price sources; require approved mark/bid-ask source with bounded market-event age and source-specific health; persist both `priced` and `execution_price_ready` counts. Reject future-dated events rather than clamping negative age to zero (`core/market_io.py:263–268`).

### P1-10 — Model validation and calibration share a sample

**Evidence:** `training/base_trainer.py:87–106`; `training/rf_training_engine.py:249–260`; `training/gbm_training_engine.py:336–339`; `training/dqn_training_engine.py:310–319`.

Models are fit on `X_train`; sigmoid calibration is fit on `X_valid`, then predictions, calibration metrics and performance metrics are reported **on `X_valid` itself**. Walk-forward diagnostics are present, but they do not turn the calibrated tail performance into independent out-of-sample evidence. Calibration ECE/Brier and improvement gates can be artificially favorable.

**Fix:** split chronological data into **model fit / probability calibration / untouched test** with label-dependent purge; optionally use nested walk-forward. Require calibration on previously unseen predictions and report a forward-only post-calibration test, including interval-specific directional reliability, Brier, log loss, selective performance and confidence intervals. Compare candidate and incumbent on **the same contiguous holdout period**.

### P1-11 — Global backfill leaks future features

**Evidence:** `features/dataset_preparation.py:69–76,311–358`; `training/dqn_training_engine.py:239–246`.

`ffill().bfill().fillna(0)` is applied to the entire supervised dataset **before** chronological splitting and cross-validation. A future source observation can populate an earlier missing value; using all data to determine constant columns is also a preprocessing-scope flaw. Filling a missing higher-timeframe feature with a future value masks actual absence and reduces real live-data sparsity.

**Fix:** build strictly point-in-time features; allow only causally valid forward fills with explicit per-source TTL and missingness flags; fit any feature selectors/imputers on the training portion of each fold; treat missing essential timeframes as non-ready rather than quietly substituting a future or zero feature.

### P1-12 — Label-horizon overlap is not fully purged

**Evidence:** `config/system_settings.yaml:72–87`; `features/dataset_preparation.py` chronological purge/split functions.

Label horizons are **5, 4, 3, 2 rows** for 1m, 3m, 5m, 15m respectively, whereas `purge_rows=2` and `embargo_rows=1`. A training observation near the boundary can have a target outcome extending into the validation interval. The purge must be based on **label event end time**, not only a fixed two-row gap.

**Fix:** store `label_start_at` and `label_end_at`, exclude all training events that overlap validation observation/label periods, impose embargo appropriate to shared information/horizon, and test boundary conditions for every timeframe.

### P1-13 — Model payoff/EV and labels are not execution-conditioned

**Evidence:** `features/label_builder.py:37–43,110–161`; `training/rf_training_engine.py:261–264`; `inference/expected_value.py:71–154`.

RF/GBM directional labels use fixed-horizon close-to-close future return and dynamic ATR threshold plus a flat estimated cost. Trading exits use SL/TP; realized outcomes depend on intrabar path, fill price, fees, funding, latency, and censored holding time. EV uses win/loss averages from **all historical positive/negative future returns** rather than class-conditional, action-conditional, **actual executable trades**. The three-class EV treats the HOLD posterior as zero economic payoff. This is a **heuristic decision score**, not validated expected net PnL.

**Fix:** retain a pure directional target for signal diagnostics, but train/gate execution policy using causally simulated, event-driven triple-barrier or path-aware net return labels; account for slippage, fill propensity, fee tier, funding, stop/take-profit precedence and liquidation constraints. Calibrate action-conditioned monetary expectancy on untouched forward data.

### P1-14 — Partial fills and fees are not portfolio-authoritative

**Evidence:** `execution/exchange_adapter.py:198–214` (live `fee_usd=0`); `runners/live_trading_controller.py:601–610,1877–1886`; `execution/portfolio_state.py:861–886,934–1002,1004–1025`.

The local portfolio opens using the risk **planned quantity**, not necessarily the actual exchange partial-fill quantity; the exit path removes the **entire position** once the router reports `partially_filled`, even if exchange quantity remains open. The ledger's close operation has no partial-close quantity. Live commissions are recorded as zero; funding and liquidation/ADL are absent from net PnL. While open, equity uses mark-to-market gross unrealized without charging already incurred entry commissions. Risk sizing also references a fixed configured balance (`execution/risk_engine.py:414–417`) rather than proven current exchange account equity.

**Fix:** use individual fill quantities and unique trade IDs, maintain signed per-symbol position lots (or equivalent netted position accounting) and weighted entry cost, book entry/exit fees when charged, realized PnL on every reduction, funding/insurance/liquidation/ADL adjustments, margin/equity reconciliation, and daily opening balances. Never close a local position more than its verified filled quantity. Any unexplained mismatch must immediately block risk increases.

### P1-15 — Exchange filters and account-mode compatibility are unproven

**Evidence:** `execution/exchange_adapter.py:164–185`; exchange information is used for symbol listing in `runners/market_data_controller.py:808–839` but no evidence of execution-side `LOT_SIZE`, `MARKET_LOT_SIZE`, `MIN_NOTIONAL`, tick-size filters, `positionSide`, leverage/margin configuration or reduce-only hedge-mode adaptation in the order adapter.

The adapter directly transmits floating quantity. Binance symbol filters, position modes, and margin controls can reject otherwise valid strategy intents; a rejected protective reduction can leave risk outstanding.

**Fix:** snapshot exchange filters at trading startup, quantize using `Decimal`, verify effective notional **after** quantization, validate account position mode and leverage, build account-mode-specific order parameters, and reject unrepresentable orders before routing. Recheck exchange order acknowledgments and protection status.

### P1-16 — Backtest and paper execution costs are simplistic

**Evidence:** `config/system_settings.yaml:339–343,388–396`; `execution/exchange_adapter.py:60–85`; `services/backtest_engine.py:565–579,773–796`.

Configured trading assumptions: fees **3 bp**, slippage **2 bp**, spread **1 bp**, funding **0 bp**; paper latency cost is `min(slippage_bps, 0.1 × market_poll_seconds)` (0.5 bp at the configured 5-second polling), not an empirical latency model. Paper orders fill instantly at a synthetic adverse price. The backtest incorporates adjusted entry/exit prices and fees into net PnL, while displaying slippage separately (not automatically a double charge). There is no evidence of depth/size impact, exchange maker/taker behavior, partial fills, volatility slippage tails, latency distribution or real funding expense.

**Fix:** reconstruct replay using trades or top-of-book/depth when possible; model event-to-routing and routing-to-fill times; enforce tick/quantity constraints; include maker/taker commissions, funding settlement, rejected orders and gap exits; stress across 1×, 2×, 3× observed transaction costs. Distinguish mark-to-market performance from executable fill-to-fill PnL.

### P1-17 — Promotion policy is scaffolded but empirical evidence is insufficient

**Evidence:** `core/artifact_contract.py:395–432,940–958`; `config/system_settings.yaml:88–127`; `training/base_trainer.py:554–566`.

There are useful artifact-existence, signature/recipe, execution-readiness, paper-only bootstrap, and promotion metric gates. Nevertheless, readiness cannot be **verified** from this archive without actual active manifests. The minimum paper evidence gate is **five trades**, which cannot estimate edge or drawdown reliably. Some promotion statistics are contaminated as above. A manifest signature validates a declared feature contract, not that its historical rows were point-in-time correct, that model performance persists out of sample, or that a fill was executable.

**Fix:** make every promoted manifest immutable and content-addressed: dataset snapshot/epoch, full feature recipe hash, label version, train/cal/test time boundaries, split definition, code commit, environment/dependencies, candidate metrics with uncertainty, actual paper fills and operating-cost measurements, approval actor and rollback pointer. Separate **candidate-trained**, **paper-eligible**, **live-candidate**, and **live-approved** states. Keep only ready models as inference-authoritative.

### P1-18 — Correlation/portfolio concentration guard is ineffective by default

**Evidence:** `config/system_settings.yaml:352–367`; `execution/risk_engine.py:375–385`; `execution/portfolio_state.py:1026–1045`.

`max_correlated_positions=2` exists, but `risk.correlation_groups={}`: the risk-side correlation condition only runs if a group is defined; portfolio telemetry initializes `correlation_groups={}` locally and so reports no cluster exposure. Default gross exposure cap **$7,500** against configured balance **$1,000** represents up to 7.5× gross notional before exchange leverage/margin/liquidation constraints; a position maximum is not a substitute for **portfolio tail-risk budgeting**. Gross/net and long/short caps alone are insufficient for futures with common crypto beta.

**Fix:** populate and audit static buckets initially; add rolling correlation and beta factor exposure with robust shrinkage/cluster thresholds, volatility-targeted gross, max liquidation-risk, liquidity shock and funding concentration budgets, margin requirements, sector/correlation caps, and account-equity-aware position sizing.

### P1-19 — Dashboard websocket events stop updating at constant queue length

**Evidence:** `monitoring/web_app.py:626–658`, especially `648–651`.

Broadcast of the last 50 events occurs only when `len(events) != last_event_count`. Once the bounded list reaches 50, new events replace old ones while the count stays 50, so the browser no longer receives updated events through this websocket path. A heartbeat can still make the dashboard look connected.

**Fix:** assign monotonic event sequence IDs, broadcast on changed highest ID (not count), support cursor-based catch-up on reconnect, and report producer/consumer lag and lost event counts. Add deterministic test for 51st and 500th events.

### P1-20 — Blocked-symbol live position handling is an incomplete reduction workflow

**Evidence:** `config/system_settings.yaml:38–47`; `execution/portfolio_state.py:488–519`; lifecycle `runners/tech_rev_controller.py`, `core/symbol_management.py`.

Symbol blocking and replacement machinery exists. In live mode, `close_blocked_positions` deliberately reports a reconciliation-required failure rather than falsely mutating local state. This is **correct fail-closed accounting**, but it is not actual liquidation: the blocked symbol can remain an open Binance position. The operator needs an explicit exchange-authoritative reduction/transfer-to-risk-supervision workflow and alerts, not merely a `failed` status. A blocked symbol must remain managed for stops/reconciliation until flattened, regardless of membership in the active entry universe.

### P2-21 — Effective HMM and confirmatory policy value is unproven

**Evidence:** `training/dqn_training_engine.py:305–330`; `config/system_settings.yaml:166–196,258–282`; `inference/trade_decision_logic.py:307–376`.

The component named `dqn` is documented as a **supervised confirmatory random-forest classifier**, not true deep Q-learning; its active blend weight and adjustment strength are **zero**. Its runtime marginal trading benefit is therefore intentionally nil under this configuration. HMM is trained for candidate/shadow use, while **inference** enables a mandatory **paper-only** confidence gate at 0.60. The HMM gate checks availability and a sufficiently confident *state*, **not directional agreement** between regime and proposed trade. HMM influence on live trading is explicitly disabled. Neither component has demonstrated incremental net expectancy, reduction in adverse selection, or better drawdown from the supplied evidence.

**Fix:** retain the accurate terminology and governance state. Run prospective controlled ablations: RF, GBM, blend, HMM gate on/off, and confirmatory policy on/off, with the **same** data, risk budget, execution cost model, and paired fill opportunities; measure incremental cost-adjusted net alpha, turnover, selective precision, and drawdown.

### P2-22 — Feature usefulness and numerical drift are not established

**Evidence:** `features/timeframe_fusion.py:~180–286` engineers multiple technical indicator families; `features/feature_store_contract.py:267–362` checks recent input timestamps; `monitoring/publisher.py` has **prediction-drift** summaries but there is no demonstrated PSI/KS/Wasserstein or feature-distribution drift enforcement over live vs train baselines.

Price, volume, ATR, MACD, Bollinger, volatility, VWAP and multiple timeframe values may be highly correlated or duplicate the same signal. A schema signature detects column changes, **not** changed statistical distributions, feature missingness, order-book regime change or rolling feature-to-label efficacy. A source freshness check examines present source timestamps, but does not by itself ensure every **required** source interval has an eligible row. This is an evidence gap, not proof that those features have no predictive power.

**Fix:** track normalized per-feature distributions and missingness by symbol/timeframe/model version, PSI/KS, rank stability, permutation/drop-column importance on **untouched test**, redundancy clustering, live input/label drift, and alerts for missing expected frames. Run feature-family ablations with confidence intervals and execution-cost-aware objective.

### P2-23 — Training cadence is bar-driven, but efficiency and authority need validation

**Evidence:** `config/system_settings.yaml:78–87,198–239`; `scheduling/priority_manager.py:502–536,625–667`; archived training log.

Due logic generally compares elapsed market bars to a retraining threshold, **not** every scheduler tick. Configured retraining thresholds are 1m→7 bars, 3m→4 bars, 5m→2 bars, 15m→1 bar. This correctly prevents training on every subsecond scheduler loop when source timestamps are valid, but can still generate **1,128 success records in ~110 minutes** across symbols and model roles; frequent retraining increases resource contention, model churn and repeated validation selection. `training.retrain_cycles` takes precedence over model-specific `scheduling.retrain_min_new_bars` when present, which can confuse operators. The logs alone do not establish whether model results are independent, promoted, or consumed by inference.

**Fix:** publish per-key last-data-close, last-trained-close, new unique bars, last successful train, training duration, promotion timestamp, model-diff score, queue age, duplicate-work count, CPU/IO use, and out-of-order data checks. Avoid retraining when data hash has not changed and impose model-change utility thresholds and stable live promotion windows.

### P2-24 — Universe rotation and scheduler are partially disciplined but not verified

**Evidence:** `runners/market_data_controller.py:808–924`; `config/system_settings.yaml:7–47`; `runners/tech_rev_controller.py`; `scheduling/priority_manager.py`; `main.py:109–153`.

The market worker filters USDT perpetual instruments in `TRADING` status, ranks by 24h quote volume, and excludes insufficient-history candidates; the market worker publishes **candidate** rankings rather than overwriting lifecycle-owned active/blocked symbols, which is a sound single-writer design. The launcher supervises critical threads, and the market-data controller has reconnect/fallback handling. Still, 24h volume rank does not ensure acceptable spread, depth, basis, funding/liquidation risk, manipulation resistance or model readiness. Actual active and blocked-state files were not supplied, and thread supervision is **death detection**, not automatic state-recovering restart with idempotent bootstrapping.

**Fix:** rotate using eligible depth/spread/volume stability, age, volatility and model readiness; log removal/addition rationale and warmup state; freeze exposure-increasing actions on stale universe or missed heartbeats; test market-worker death, trainer hang, missing state, market-partial WS, graceful shutdown, and exchange-reconciliation mismatch. Maintain separate lists for entry-eligible, training candidate, blocked, and open-position-under-supervision.

## 4. Coverage matrix — all requested dimensions

**Status:** `DEFECT` = confirmed material implementation flaw; `PARTIAL` = implemented but incomplete/conditional; `UNPROVEN` = cannot certify effectiveness from supplied evidence.

| Requested dimension | Status | Audited judgment / primary evidence |
|---|---|---|
| Multi-timeframe synchronization | **DEFECT P0** | Open-time merge produces historical look-ahead; `features/timeframe_fusion.py:339–346`. |
| Price cache freshness | **DEFECT P0** | DB close can inherit current snapshot timestamp and pass freshness gate; `market_data_controller.py:497–513`, `market_io.py:305–342`. |
| Websocket reliability | PARTIAL | Ping, timeout, reconnect, per-symbol cache and HTTP fallback; fallback provenance can be unsafe; `market_data_controller.py:538–687`. |
| Training cadence correctness | PARTIAL | New-bar-based due policy; unusually high training volume; `priority_manager.py:625–667`, config. |
| RF / GBM / confirmatory effectiveness | **UNPROVEN** | RF label defect and contaminated calibration; DQN is RF classifier and zero-weight; configs/trainers. |
| Feature usefulness | UNPROVEN | Engineered indicator suite exists; no verified independent marginal contribution / ablations. |
| Feature drift | PARTIAL | Prediction drift reported; feature distribution/importance drift not established. |
| Label quality | **DEFECT P0/P1** | RF wrong timeframe, horizon overlap, close-to-close outcomes not trade-path aware. |
| Signal calibration | **DEFECT P1** | Calibration and quality measured on same validation set. |
| HOLD/LONG/SHORT decision quality | UNPROVEN | Thresholds and EV exist, but biased label/EV foundations; no confusion matrix or live attribution. |
| Slippage realism | **DEFECT P1** | Fixed bps and synthetic latency; no depth/size/funding tails. |
| Execution latency | UNPROVEN | Poll/refresh settings exist; no observed decision-to-fill p50/p95/p99. |
| Manifest readiness integrity | PARTIAL | Signatures, bundle existence, active-manifest gates; actual artifacts absent; point-in-time semantics not validated. |
| Promotion governance | PARTIAL | Paper bootstrap and metric gates; tiny paper sample and contaminated calibration. |
| Active universe correctness | UNPROVEN | Quote-volume candidates/lifecycle ownership; current active state missing. |
| Blocked-symbol handling | PARTIAL | Lifecycle and manual block exist; live positions not automatically reduced. |
| Trade lifecycle traceability | **DEFECT P0** | Decision lineage and order journal exist; ACK/fill/partial/duplicate paths can diverge. |
| Dashboard telemetry correctness | **DEFECT P1** | Capped event queue broadcast flaw; no runtime snapshot to reconcile. |
| Runtime observability completeness | PARTIAL | Heartbeats, registry, risk and events; lacks full end-to-end latency/fill/protection assertions. |
| Risk engine behavior | **DEFECT P0** | Circuit can block normal exits; base balance static; correlation groups empty. |
| Scheduler robustness | PARTIAL | Fairness/backoff/worker caps; repeated trainings and dead-thread behavior need load tests. |
| Controller fault tolerance | PARTIAL | Supervisor and alerts; no verified stateful automatic restart/replay. |
| Exchange reconciliation integrity | **DEFECT P0** | Signature mismatch, position-level only, open-order recovery can falsely terminalize. |
| PnL correctness | **DEFECT P1** | Partial closes unsupported; fee/funding/entry-equity treatment incomplete. |
| Symbol rotation quality | UNPROVEN | Liquidity-volume ranking/warmup gating; no observed spread/depth/outcome cost analysis. |
| Training/inference skew | **DEFECT P0** | MTF historical availability vs live closed-bars; RF labels differ from inference interval contract. |
| Regime adaptation quality | UNPROVEN | Regime shadow only; adaptive mode disabled; HMM confidence ≠ directional confirmation. |
| Trade frequency realism | UNPROVEN | No trade journal; idempotency collisions can suppress repeat legitimate entries. |
| Portfolio-level behavior | **DEFECT P1** | Gross/net caps, but factor/correlation/margin/liquidation risk not established. |
| Security / control plane (additional) | **DEFECT P0** | Auth bypass permits control mutations to service-accessible clients. |
| Exchange order compatibility (additional) | **DEFECT P1** | Quantity filter/margin/hedge-mode support not evident in adapter. |

## 5. Model and decision-quality assessment

### RF (Random Forest)

**Do not use RF performance numbers for capital allocation** until the target-timeframe bug, historical feature availability, calibration/test contamination and label event overlap are fixed. The model training uses meaningful time-ordered folds, class-support checks, class weights and constrained trees; these are sound *components*, but invalid labels/data defeat their statistical value. Post-fix compare by symbol and interval: balanced accuracy, per-direction precision/recall, confusion matrix, class prevalence, average forward return per posterior decile, selective PnL after costs, and stability over multiple forward regimes.

### GBM (Histogram Gradient Boosting and optional meta learner)

GBM is trained with `anchor_interval=interval`, avoiding the specific RF label mismatch. It shares the contaminated MTF features, pre-split imputation, calibration/test reuse and insufficient label boundary purge. The meta learner should use only genuinely out-of-fold **past-available** base predictions, with point-in-time guard preventing any in-sample direction forecast from becoming a meta feature. Audit all GBM role-specific manifests and test direction + meta features independently; verify the meta label is derived from realizable trades, not just signed future returns.

### Confirmatory / legacy “DQN”

The trained object is a supervised classifier, **not a reinforcement-learning agent**. It is labeled as a confirmatory policy for compatibility, and current inference weights disable its economic effect. Its value should be measured by **incremental rejected-bad-trades vs rejected-good-trades** and net improvement after fewer trades, with stable reference gates. Do not represent DQN training successes as learned RL policy performance.

### HMM / regime

Candidate HMM observations do not use future labels, but the mandatory paper gate only requires the HMM output to be a confidently inferred regime state. Confidence does **not** prove that LONG/SHORT is appropriate under that state. Require forward state stability and independent conditional return / adverse-execution calibration; compare HMM-on vs HMM-off as a **shadow ablation**, not a presumed confirmer. Live HMM influence is disabled. `adaptive_mode.enabled=false`, and `regime` is shadow-only; consequently claims of proven automated regime adaptation are **unsupported** by current settings.

### HOLD/LONG/SHORT effectiveness

The system contains a three-class encoding (`short=0, hold=1, long=2`), consensus thresholds, minimum-model gates, an EV heuristic, ATR minimums and risk rejection reasons. It is not possible to claim the HOLD rate, signal-to-trade conversion, false trade suppression, directional expectancy, or turnover optimality without paired timestamps, predictions, rejection reasons, and realized outcomes. Required forward reporting: each candidate's posterior vector, model versions, gating path, action, order attempt, actual fill, markout at 1/5/15/60 minutes, net realized result and reason for HOLD. Quantify opportunity cost of HOLD, not just its frequency.

## 6. PnL and execution-ledger reconciliation specification

Treat **exchange trades/fees/funding/account equity as the economic source of truth** for live USDⓈ-M. Maintain separate views of modeled/simulated PnL, exchange-reconciled realized PnL, and marked unrealized PnL. Required fields per event:

`decision_id`, `symbol`, `interval`, `feature_recipe_hash`, `model_manifest_digest`, `signal_at`, `decision_at`, `risk_approved_at`, `order_intent_id`, `client_order_id`, `exchange_order_id`, `trade_id`, `order_sent_at`, `ack_at`, `fill_exchange_at`, `fill_received_at`, `side`, `positionSide`, `reduceOnly`, `requested_qty`, `filled_qty_delta`, `cumulative_filled_qty`, `fill_price`, `mark_at_signal`, `best_bid/ask`, `commission`, `commission_asset`, `funding`, `realized_pnl`, `protective_order_ids`, `reconciliation_status`, `account_equity`, `reason`.

For every distinct exchange fill, update the order state and position **exactly once** by its exchange trade identity. Closed-trade realized result must reflect actual entry/exit executed price × closed contract quantity, **minus** actual commissions/funding and other cash adjustments, correctly currency-converted where necessary. Slippage is already embedded in executed vs reference prices; report slippage attribution separately and **do not subtract it again** if using executed-price gross PnL. The balance/equity bridge must reconcile to Binance account income and balance snapshots subject to known wallet transfers and unrealized price marking. Break down daily PnL by symbol, model, decision regime and exit type; reconcile open positions and protection separately.

## 7. Target production architecture

Recommended control boundaries:

1. **Data plane:** Binance kline/mark/book ingestion → event-time normalized, immutable raw store → closed-bar availability clock → point-in-time feature store. Price cache records both exchange event and receive times, typed source, health and sequence number. The execution mark is **not** a substituted old bar close.
2. **ML plane:** immutable data snapshot → causal feature transform → horizon-aware labeler → purged chronological train/calibration/test → manifest with content hashes → candidate paper comparison → manual promotion after evidence threshold. Reject loading when feature availability rules, schema or hashes mismatch.
3. **Decision plane:** persisted prediction/score + model lineage → deterministic HOLD/LONG/SHORT state → risk gate with trusted exchange equity/positions → unique execution intent ID. A risk circuit blocks **risk increases**, never verified reductions.
4. **Execution plane:** serialized signed requests → durable order outbox → signed exchange ACK → authoritative user-data and REST recovery → event-sourced fill ledger → actual position lots/average cost → linked protective algo orders → reconciliation + alerting. Unknown outcomes are **UNKNOWN**, not rejected/canceled.
5. **Control plane:** authenticated and authorized operator actions with immutable audit trail; dashboard projection from canonical event stream, not independent source of economic truth.
6. **Operations:** supervised independent processes, restart replay, staleness SLOs, circuit coverage for market-data and execution faults, time synchronization and telemetry. Reconciliation failures block new risk and alert until resolved.

## 8. Required fault-injection and regression tests

The current repository contains many tests, but count alone does not prove the following acceptance criteria. Add targeted fixtures and run them in isolated testnet/simulated environments:

| Test | Pass criterion |
|---|---|
| **MTF future-candle mutation** | Changing an 11:00+ 1h close never changes a signal as-of 10:15. |
| **Source age** | 1h/4h/1d asof age is nonzero when expected; missing required sources fail readiness. |
| **RF horizon** | 1m label at row *t* uses close at *t+5* of **1m**, not 15m. |
| **Purge overlap** | No train-label end timestamp overlaps any held-out evaluation interval. |
| **Imputation** | Replaying data only through time *t* gives identical time-*t* features to the full-history batch. |
| **Calibration isolation** | Evaluation rows are neither used to train estimator nor fit calibrator, including tuning and selection. |
| **REST signature** | HMAC is computed from the **exact** transmitted serialized bytes for POST/GET, special chars and decimals. |
| **Idempotency recurrence** | Repeated identical symbol/side/quantity **new** intents place distinct orders; retries of the same intent place exactly one. |
| **ACK-first order** | ACK + later FILLED updates local portfolio exactly once; timeout/503 outcome enters UNKNOWN. |
| **Recovery API down** | Failure to obtain open orders never terminalizes the order; state remains unknown pending authoritative query. |
| **Partial close** | 40% execution of exit reduces position quantity by exactly 40%, keeps 60% protected. |
| **Protection uptime** | Process kill immediately after entry still leaves verified exchange-native stop in place. |
| **Circuit EXIT** | Loss circuit blocks new entries but allows authorized reduce-only exits. |
| **Price fallback** | 15m `db_close` with now-stamped snapshot is **not** eligible for 2s live execution gate. |
| **Authentication** | Missing/invalid token cannot call pause, reset, rollback, block/unblock or privileged websocket streams. |
| **Dashboard capped feed** | Every new event updates browser after the event array has reached its 50-item cap. |
| **PnL reconciliation** | Synthetic multi-fill fees/funding/partial close + transfer reconcile to canonical trade and exchange ledger. |
| **Universe blocked position** | Removing a symbol from entry universe preserves risk/stop/marking/reconciliation until its position is flat. |
| **Crash/restart** | Kill at send, ACK, first fill, final fill, stop installed, and portfolio save; replay produces a single correct final state. |
| **Performance stress** | Measure p50/p95/p99 data/event/decision/order/ack/fill latency, queue age, reconnect and health SLOs. |

## 9. Remediation order and release governance

### Phase 0 — Immediately (block live enablement)

1. Preserve the committed `paper`, `use_testnet=true` and canary controls. **Do not submit live orders.** Disable or isolate dashboard mutation APIs until authentication is restored. Audit any existing exchange account separately for unknown open positions; archive the raw state without sharing secrets.
2. Repair signing serialization, authoritative order/fill recovery, unknown-order handling and economic-intent idempotency **together**. Merely fixing signatures before lifecycle control can increase operational risk because previously failing orders may begin executing.
3. Build exchange-native protective exit placement and verification, partial-fill-aware local portfolio, correct circuit behavior and safe shutdown/recovery.
4. Block DB-close prices from live entry freshness and make provenance/timestamps source-specific.

### Phase 1 — Research integrity rebuild

5. Correct MTF available-at/as-of joins and age features; fix RF anchor interval; remove future-filling; implement event-time purging and independent post-calibration evaluation.
6. Version the feature/label contracts, retire contaminated models/manifests, regenerate the full history and re-run forward-only walk-forward experiments. No previously trained model score should survive these fixes without re-evaluation.
7. Rebuild promotion requirements with fixed, same-period comparison to incumbent and meaningful forward paper sample. Record statistical uncertainty, net expectancy after costs, and maximum drawdown.

### Phase 2 — Portfolio and production readiness

8. Add accounting for actual fees, funding, account equity, exchange filters, partial fills, leverage/margin/hedge mode, liquidation exposure and correlated positions.
9. Add venue liquidity checks to symbol ranking and executable spread/depth limits; enforce risk supervision for blocked but open symbols.
10. Add drift (feature/prediction/outcome), latency histograms, event sequence, reconciliation/protection health, telemetry SLOs, dashboard source-of-truth assertions and operator incident playbooks.

### Phase 3 — Controlled paper, then exchange testnet only

11. Demonstrate that a replayed event ledger reconstructs all order, fill, position and PnL states exactly. Inject all failures above and retain logs/proof.
12. Run a sufficiently long forward **paper** experiment across market regimes and a separate Binance **testnet** execution/position/protection experiment; pre-register decision metrics and stop rules. A few minutes of websocket heartbeats and five paper trades are not adequate performance validation.
13. Require written engineering/risk approval before considering real-capital deployment; first real-capital deployment, if any, should be separately gated by legally appropriate operator controls and conservative capital limits. No specific profitable-return expectation can be recommended from supplied evidence.

## 10. Release gate — minimum institutional acceptance

**HARD BLOCK until ALL true:**

- No historical future observation can affect any prior as-of feature, label-independent prediction, or decision.
- All trained models use the correct timeframe/horizon with no train–calibration–test overlap; all promoted manifests are tied to immutable data/code/feature/label versions.
- API signatures are validated against identical transmitted bytes on testnet; all exchange order responses are classified accurately (ACK/SUBMITTED/UNKNOWN/PARTIAL/FILLED/CANCELED/REJECTED).
- Live orders and exchange trade IDs map into exactly-once local fills; duplicate/new-intent semantics are tested.
- Every exchange-open position is paired with verified protective orders, correctly quantity-sized; a stopped controller cannot disable existing protection.
- Position size, entry fee, realized PnL, funding, mark equity, and account balances reconcile within an explicit currency/tick tolerance; any mismatch blocks increases in exposure.
- Circuit breaker, dashboard pause and blocked symbol logic cannot obstruct authorized position reductions.
- Only genuinely fresh source-typed exchange prices satisfy the live execution gate.
- All control-plane mutations require real authorization and record auditable actor/action/previous state/new state.
- Required end-to-end health, reconciliation, drift and latency alarms have been demonstrated by fault injection; control and PnL dashboard views equal canonical source events.
- Sufficient prospective paper/testnet data exists to establish net-of-cost behavior with uncertainty; **profitability is not presumed**.

## 11. Evidence inventory / reproduction references

- **Market and MTF:** `runners/market_data_controller.py`, `features/timeframe_fusion.py`, `features/market_loader.py`, `features/feature_store_contract.py`, `core/market_io.py`.
- **Labels, dataset, models:** `features/label_builder.py`, `features/dataset_preparation.py`, `training/base_trainer.py`, `training/rf_training_engine.py`, `training/gbm_training_engine.py`, `training/dqn_training_engine.py`, `training/hmm_regime_training_engine.py`.
- **Signal and regime:** `inference/trade_decision_logic.py`, `inference/expected_value.py`, `inference/regime_filter.py`, `inference/model_artifact_loader.py`.
- **Execution and PnL:** `execution/exchange_adapter.py`, `execution/exchange_reconciliation.py`, `execution/order_store.py`, `execution/order_router.py`, `execution/fill_handler.py`, `execution/portfolio_state.py`, `execution/risk_engine.py`, `runners/live_trading_controller.py`, `services/backtest_engine.py`.
- **Governance/ops/universe/dashboard:** `core/artifact_contract.py`, `core/training_registry.py`, `core/symbol_management.py`, `scheduling/priority_manager.py`, `runners/tech_rev_controller.py`, `main.py`, `monitoring/auth.py`, `monitoring/web_app.py`, `monitoring/publisher.py`, `reporting/run_report.py`, `config/system_settings.yaml`.
- **Runtime evidence:** `logs/process_roles/run_runtime.log`, `run_trainer.log`, `run_dashboard.log`.
- **External verification, Binance official documentation, accessed 22 Nov 2026:** USDⓈ-M REST `General Info`: https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/general-info ; USDⓈ-M Trade endpoints (`New Order`, `New Algo Order`): https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade ; `ORDER_TRADE_UPDATE` specification: https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/user-data-streams . Note official documentation currently describes conditional TP/SL via `/fapi/v1/algoOrder` and REST market-order `newOrderRespType` default `ACK`.

---

**Final assessment:** The repository offers a useful governance-and-telemetry starting point but **does not currently meet institutional research integrity, exchange execution correctness, position-protection, PnL-accounting, or control-plane security requirements**. An alpha claim, Sharpe ratio, or realistic expected monthly return cannot be certified from the supplied archive. The required work is principally **correctness and safety refactoring**, **not model tuning**.

## Appendix A — Isolated source-contract reproductions (no exchange/API calls)

To corroborate the signing defect, an isolated Python `requests.Request(...).prepare()` experiment was run with the same key insertion order used by `execution/exchange_adapter.py`, a synthetic API secret and synthetic timestamp. The source `_sign_params` algorithm signs:

```text
newClientOrderId=cli_test&quantity=0.02&recvWindow=5000&side=BUY&symbol=BTCUSDT&timestamp=1793127789000&type=MARKET
```

while `requests` prepares the transmitted payload as:

```text
symbol=BTCUSDT&side=BUY&type=MARKET&quantity=0.02&newClientOrderId=cli_test&recvWindow=5000&timestamp=1793127789000
```

The strings are **not identical**, so the existing signature algorithm does not sign the actual transmitted string in this representative case. This is a controlled local reproduction of a serialization mismatch, **not an observed exchange rejection**.

A second standalone `pandas.merge_asof` experiment reproducing `features/timeframe_fusion.py:339–346` joins a completed source 1h close stamped `00:00` into lower-timeframe bars stamped `00:00`, `00:15`, `00:30`, and `00:45`. That is causal look-ahead in a historical training dataset. The appropriate source availability timestamp for that 1h candle is at or after `01:00`, not `00:00`. This isolated reproduction does not require or use the repository's unavailable database.
