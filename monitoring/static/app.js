const state = {
  uiConfig: { dashboard_refresh_ms: 1000, fast_snapshot_interval_sec: 1, heavy_snapshot_interval_sec: 5 },
  socket: null,
  snapshot: null,
  lastGoodSnapshot: null,
  events: [],
  lastGoodEvents: [],
  pollHandle: null,
  socketConnected: false,
  lastRefreshError: null,
  pollInFlight: false,
  reconnectHandle: null,
  lastAppliedSnapshotSource: null,
  renderedHtml: new Map(),
  lastSocketMessageAt: 0,
  socketWatchdogHandle: null,
  openPopupCardId: null,
  lastPopupMarkup: null,
};

function qs(id) { return document.getElementById(id); }
function setText(id, value) { const el = qs(id); if (el) el.textContent = value ?? "--"; }
function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(digits);
}
function fmtTime(value) {
  if (!value) return "--";
  try {
    const raw = Number(value);
    if (!Number.isFinite(raw)) return "--";
    const ms = raw < 10_000_000_000 ? raw * 1000 : raw;
    const date = new Date(ms);
    return Number.isNaN(date.getTime()) ? "--" : date.toLocaleString();
  } catch { return "--"; }
}
function fmtDurationLeft(value) {
  const total = Math.max(0, Math.floor(Number(value) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}
function authHeaders() { return {}; }
async function fetchJson(path) {
  const response = await fetch(path, { headers: authHeaders(), credentials: "same-origin", cache: "no-store" });
  if (!response.ok) throw new Error(`request failed: ${response.status}`);
  return response.json();
}
async function postJson(path, payload) {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", ...authHeaders() }, credentials: "same-origin", body: JSON.stringify(payload || {}) });
  if (!response.ok) throw new Error(`request failed: ${response.status}`);
  return response.json();
}
async function logoutSession() {
  await fetch("/api/auth/logout", { method: "POST", credentials: "same-origin" });
}
function showLogin() {
  const overlay = qs("loginOverlay");
  const logout = qs("logoutButton");
  if (overlay) overlay.classList.add("hidden");
  if (logout) logout.classList.add("hidden");
}
function emptyRow(colspan, message) {
  return `<tr><td colspan="${colspan}" class="empty-table">${message}</td></tr>`;
}
function emptyFeed(message) {
  return `<div class="empty-feed">${message}</div>`;
}
function replaceHtmlStable(element, html) {
  if (!element) return false;
  const nextHtml = String(html ?? "");
  const cacheKey = element.id || null;
  if (cacheKey && state.renderedHtml.get(cacheKey) === nextHtml) return false;
  const scrollHost = element.closest(".scroll-body");
  const scrollTop = scrollHost ? scrollHost.scrollTop : 0;
  const scrollLeft = scrollHost ? scrollHost.scrollLeft : 0;
  element.innerHTML = nextHtml;
  if (cacheKey) state.renderedHtml.set(cacheKey, nextHtml);
  if (scrollHost) {
    scrollHost.scrollTop = Math.min(scrollTop, Math.max(0, scrollHost.scrollHeight - scrollHost.clientHeight));
    scrollHost.scrollLeft = scrollLeft;
  }
  return true;
}
function feedItemHtml(inner, cls = "") {
  return `<div class="feed-item compact-feed-item ${cls}">${inner}</div>`;
}
function normalizeUniverse(rawUniverse, symbolUniverse = null) {
  const activeSymbols = symbolUniverse && typeof symbolUniverse === 'object' && Array.isArray(symbolUniverse.active_symbols)
    ? symbolUniverse.active_symbols.map((symbol) => String(symbol || '--').toUpperCase())
    : [];
  if (activeSymbols.length) return activeSymbols;
  if (Array.isArray(rawUniverse)) return rawUniverse.map((symbol) => String(symbol || '--').toUpperCase());
  if (rawUniverse && typeof rawUniverse === 'object') {
    return Array.isArray(rawUniverse.symbols)
      ? rawUniverse.symbols.map((symbol) => String(symbol || '--').toUpperCase())
      : [];
  }
  return [];
}
function normalizeInferenceOpenPositions(rawOpenPositions, mode) {
  if (Array.isArray(rawOpenPositions)) return rawOpenPositions;
  if (!rawOpenPositions || typeof rawOpenPositions !== 'object') return [];
  return Object.entries(rawOpenPositions).map(([symbol, row]) => {
    const lastPrice = row?.last_price ?? row?.latest_price ?? row?.reference_price;
    const estimatedPnl = row?.estimated_pnl ?? row?.pnl_usd ?? row?.unrealized_pnl_usd ?? 0;
    return {
      ...(row || {}),
      symbol: String(symbol || '--').toUpperCase(),
      side: row?.side || 'OPEN',
      mode: row?.mode || mode,
      entry_price: row?.entry_price,
      last_price: lastPrice,
      latest_price: row?.latest_price ?? lastPrice,
      reference_price: row?.reference_price ?? lastPrice,
      last_price_ts: row?.last_price_ts || row?.updated_at || row?.opened_at,
      estimated_pnl: estimatedPnl,
    };
  });
}
function normalizeRecentSignals(existingSignals, decisions, mode) {
  if (Array.isArray(existingSignals) && existingSignals.length) return existingSignals;
  if (!Array.isArray(decisions)) return [];
  return decisions.slice(-100).map((row) => ({
    symbol: String(row?.symbol || '--').toUpperCase(),
    action: row?.effective_action || row?.action || '--',
    mode: row?.mode || mode,
    reason: row?.reason || '--',
    ts: row?.ts || null,
    raw: row?.reason || '--',
  }));
}
function normalizeSnapshot(raw) {
  const snapshot = raw || {};
  const system = snapshot.system || {};
  const mode = system.execution_mode || system.launcher_mode || 'paper';
  const inference = snapshot.inference || {};
  const signalGenerator = { ...(snapshot.signal_generator || snapshot.signal_runtime || {}) };
  signalGenerator.recent_signals = normalizeRecentSignals(signalGenerator.recent_signals, inference.decisions, mode);
  const backtest = { ...(snapshot.backtest || snapshot.backtest_runtime || {}) };
  if (!Array.isArray(backtest.open_positions) || !backtest.open_positions.length) {
    backtest.open_positions = normalizeInferenceOpenPositions(inference.open_positions, mode);
  }
  if (!backtest.summary || !Object.keys(backtest.summary).length) {
    backtest.summary = { ...(inference.pnl_summary || {}) };
  }
  if (!Array.isArray(backtest.recent_messages) || !backtest.recent_messages.length) {
    backtest.recent_messages = Array.isArray(inference.decisions)
      ? inference.decisions.slice(-20).map((row) => row?.reason || '--')
      : [];
  }
  const trainingRegistry = snapshot.training_registry || {};
  const marketData = { ...(snapshot.market_data || {}) };
  const rawUniverse = marketData.universe;
  const symbolUniverse = marketData.symbol_universe || {};
  const normalizedUniverse = normalizeUniverse(rawUniverse, symbolUniverse);
  marketData.universe = normalizedUniverse;
  marketData.universe_source = marketData.universe_source || rawUniverse?.source || marketData.source || '--';
  marketData.last_universe_refresh_ts = marketData.last_universe_refresh_ts || rawUniverse?.last_refresh_ts || null;
  marketData.latest_updates = Array.isArray(marketData.latest_updates)
    ? marketData.latest_updates
    : Array.isArray(marketData.recent_results)
      ? marketData.recent_results
      : [];
  return {
    ...snapshot,
    market_data: marketData,
    training: snapshot.training || snapshot.training_runtime || {},
    signal_generator: signalGenerator,
    backtest,
    training_registry: {
      ...trainingRegistry,
      recent_results: Array.isArray(trainingRegistry.recent_results)
        ? trainingRegistry.recent_results
        : Array.isArray(snapshot.training_results)
          ? snapshot.training_results
          : [],
      entries: Array.isArray(trainingRegistry.entries) ? trainingRegistry.entries : [],
    },
  };
}
function statusClass(status) {
  const normalized = String(status || "unknown").toLowerCase();
  if (["healthy", "running", "connected"].includes(normalized)) return "badge ok";
  if (["degraded", "warning", "starting"].includes(normalized)) return "badge warn";
  if (["error", "failed", "crashed"].includes(normalized)) return "badge bad";
  return "badge neutral";
}
function compactStatusChip(text, cls = "neutral") {
  return `<span class="tiny-chip ${cls}">${text}</span>`;
}
function renderSystem(snapshot) {
  const system = snapshot?.system || {};
  const health = snapshot?.health || {};
  const status = health.status || snapshot?.status || 'starting';
  const uptimeSec = Number.isFinite(Number(system.uptime_sec)) ? Number(system.uptime_sec) : (system.started_at ? Math.max(0, (Date.now() / 1000) - Number(system.started_at)) : 0);
  const statusEl = qs("globalStatus");
  if (statusEl) { statusEl.className = statusClass(status); statusEl.textContent = status || "--"; }
  setText("lastUpdated", fmtTime(snapshot?.updated_at));
  setText("launcherMode", system.launcher_mode || system.execution_mode || "--");
  setText("uptime", fmtNumber(uptimeSec, 1));
  setText("binanceStatus", health.api_connectivity?.binance || "--");
  setText("dbStatus", health.db_connectivity?.market_storage || "--");
  const tradingReadiness = snapshot?.trading_readiness || {};
  setText("tradingReadiness", String(tradingReadiness.status || "unknown").toUpperCase());
  setText("tradingReadinessDetail", tradingReadiness.summary || "--");
  const readinessEl = qs("tradingReadiness");
  if (readinessEl) {
    readinessEl.className = tradingReadiness.ready ? "readiness-ok" : "readiness-bad";
  }
  setText("warningCount", health.warnings ?? 0);
  setText("errorCount", health.errors ?? 0);
}
function describeController(info) {
  const details = info?.details || {};
  if (details.error) return String(details.error);
  const preferredKeys = ['execution_mode', 'phase', 'symbols', 'positions', 'rf_running', 'gbm_running', 'dqn_running', 'mode', 'timeframe'];
  const displayKeys = { dqn_running: 'policy_running' };
  const parts = [];
  for (const key of preferredKeys) {
    if (details[key] !== undefined && details[key] !== null && String(details[key]).trim() !== '') {
      parts.push(`${displayKeys[key] || key}: ${details[key]}`);
    }
  }
  return parts.length ? parts.join(' · ') : '--';
}
function renderControllers(snapshot) {
  const body = qs("controllers");
  if (!body) return;
  const controllers = snapshot?.controllers || {};
  const entries = Object.entries(controllers);
  if (!entries.length) {
    replaceHtmlStable(body, emptyFeed("No controller heartbeat published yet."));
    return;
  }
  const html = entries.map(([name, info]) => `
    <article class="controller-tile">
      <div class="controller-top">
        <div class="row-title">${escapeHtml(name)}</div>
        <span class="${statusClass(info.status)}">${escapeHtml(info.status || "--")}</span>
      </div>
      <div class="controller-sub">${escapeHtml(describeController(info))}</div>
    </article>
  `).join("");
  replaceHtmlStable(body, html);
}
function renderTraining(snapshot) {
  const scheduler = snapshot?.scheduler || {};
  const training = snapshot?.training || {};
  const registry = snapshot?.training_registry || {};
  const counts = registry.counts || {};

  setText("dueJobsCount", scheduler.due_jobs_count ?? 0);
  setText("queuedJobsCount", scheduler.queued_jobs_count ?? 0);
  setText("inflightCount", training.inflight_count ?? 0);
  setText("retryThreshold", training.symbol_failure_retry_threshold ?? training.retry_threshold ?? 0);
  setText("blockedJobsCount", training.blocked_jobs_count ?? 0);

  setText("rfPending", training.rf_pending ?? 0);
  setText("gbmPending", training.gbm_pending ?? 0);
  setText("dqnPending", training.dqn_pending ?? 0);

  setText("currentRf", training.current_rf || "--");
  setText("currentGbm", training.current_gbm || "--");
  setText("currentDqn", training.current_dqn || "--");

  const rfProgress = Number(training.rf_progress ?? 0);
  const gbmProgress = Number(training.gbm_progress ?? 0);
  const dqnProgress = Number(training.dqn_progress ?? 0);
  setText("rfProgress", `${fmtNumber(rfProgress, 1)}%`);
  setText("gbmProgress", `${fmtNumber(gbmProgress, 1)}%`);
  setText("dqnProgress", `${fmtNumber(dqnProgress, 1)}%`);
  const rfBar = qs("rfProgressBar"); if (rfBar && Number(rfBar.value) !== rfProgress) rfBar.value = rfProgress;
  const gbmBar = qs("gbmProgressBar"); if (gbmBar && Number(gbmBar.value) !== gbmProgress) gbmBar.value = gbmProgress;
  const dqnBar = qs("dqnProgressBar"); if (dqnBar && Number(dqnBar.value) !== dqnProgress) dqnBar.value = dqnProgress;
  setText("lastRfMessage", `RF: ${training.last_rf_message || "--"}`);
  setText("lastGbmMessage", `GBM: ${training.last_gbm_message || "--"}`);
  setText("lastDqnMessage", `Confirmatory Policy: ${training.last_dqn_message || "--"}`);

  const timedOut = training.timed_out_symbols || training.symbol_backoff || [];
  setText("timedOutCount", timedOut.length);

  const countsEl = qs("registryCounts");
  if (countsEl) {
    const diagnosticCounts = registry.diagnostic_counts || {};
    const basePills = ["idle", "queued", "running", "success", "failed"]
      .map((key) => `<span class="pill">${escapeHtml(key)}: ${counts[key] ?? 0}</span>`);
    const diagnosticPills = ["promoted", "candidate_only", "missing_active_manifest", "feature_incompatible", "active_manifest_invalid"]
      .filter((key) => Number(diagnosticCounts[key] ?? 0) !== 0)
      .map((key) => `<span class="pill diagnostic-pill">${escapeHtml(key)}: ${Number(diagnosticCounts[key] ?? 0)}</span>`);
    replaceHtmlStable(countsEl, [...basePills, ...diagnosticPills].join(""));
  }

  const byModelEl = qs("registryByModel");
  if (byModelEl) {
    const diagnosticByModel = registry.diagnostic_counts_by_model || {};
    const rows = [];
    for (const [model, modelCounts] of Object.entries(registry.counts_by_model || {})) {
      const diag = diagnosticByModel[model] || {};
      rows.push(feedItemHtml(`<div class="feed-head"><span>${escapeHtml(model)}</span></div><div class="feed-category">idle ${modelCounts.idle ?? 0} · queued ${modelCounts.queued ?? 0} · running ${modelCounts.running ?? 0} · success ${modelCounts.success ?? 0} · failed ${modelCounts.failed ?? 0}</div><div class="feed-message">promoted ${diag.promoted ?? 0} · candidate-only ${diag.candidate_only ?? 0} · missing-active ${diag.missing_active_manifest ?? 0} · invalid ${diag.active_manifest_invalid ?? 0} · feature-incompatible ${diag.feature_incompatible ?? 0}</div>`));
    }
    const backlog = training.registry_backlog_diagnostics || {};
    const backlogRows = Array.isArray(backlog.rows) ? backlog.rows.slice(0, 20) : [];
    if (backlogRows.length) {
      rows.push(feedItemHtml(`<div class="feed-head"><span>Scheduler backlog diagnostics</span><span>${backlog.total_rows ?? backlogRows.length}</span></div><div class="feed-category">${Object.entries(backlog.counts || {}).map(([k, v]) => `${escapeHtml(k)} ${v}`).join(" · ") || "--"}</div>`));
      for (const row of backlogRows) {
        rows.push(feedItemHtml(`<div class="feed-head"><span>${escapeHtml(row.symbol || "--")} ${escapeHtml(row.interval || "--")} ${escapeHtml(row.model_type || "--")}</span><span>${escapeHtml(row.category || "--")}</span></div><div class="feed-category">${escapeHtml(row.detail || row.due_reason || "--")} · market ${row.market_data_ready ? "ready" : "not-ready"}</div>`));
      }
    }
    replaceHtmlStable(byModelEl, rows.join(""));
  }

  const timedBody = qs("timedOutBody");
  if (timedBody) {
    const html = timedOut.map((row) => {
      const canRetry = String(row.status || "").toUpperCase() === "BLOCKED";
      return `<tr><td>${escapeHtml(row.symbol || "--")}</td><td>${escapeHtml(row.status || "--")}</td><td>${row.consecutive_failures ?? 0}</td><td>${row.cooldown_level ?? 0}</td><td>${fmtDurationLeft(row.seconds_left ?? row.left_sec ?? 0)}</td><td>${fmtTime(row.retry_at_ts ?? row.retry_at)}</td><td>${row.blocked_jobs ?? row.blocked ?? 0}</td><td>${row.eligible_for_auto_retry ? "YES" : "NO"}</td><td>${escapeHtml(row.last_error || row.last_failure || "--")}</td><td>${canRetry ? `<button type="button" class="retry-symbol-btn" data-symbol="${escapeHtml(row.symbol)}">Retry</button>` : "--"}</td></tr>`;
    }).join("");
    const changed = replaceHtmlStable(timedBody, html);
    if (changed) {
      timedBody.querySelectorAll('.retry-symbol-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const symbol = btn.getAttribute('data-symbol');
          btn.disabled = true;
          try { await postJson('/api/symbols/retry', { symbols: [symbol] }); } catch (err) { console.error(err); } finally { btn.disabled = false; }
        });
      });
    }
  }
}
function summarizeTrainingResults(snapshot) {
  const bySymbol = new Map();
  const entries = snapshot?.training_registry?.entries || [];
  const recentResults = snapshot?.training_registry?.recent_results || [];
  const universe = snapshot?.market_data?.universe || [];
  const activeSet = new Set((Array.isArray(universe) ? universe : []).map((symbol) => String(symbol || '--').toUpperCase()));

  for (const symbolRaw of universe) {
    const symbol = String(symbolRaw || '--').toUpperCase();
    if (!bySymbol.has(symbol)) {
      bySymbol.set(symbol, { symbol, lastUpdate: null, frames: { "15m": [], "5m": [], "3m": [], "1m": [] } });
    }
  }

  // Use the full registry entry state as the source of truth so older 15m states do not disappear
  // simply because the recent-results ring buffer is dominated by faster 1m/3m/5m updates.
  for (const row of entries) {
    const symbol = String(row.symbol || "--").toUpperCase();
    const interval = row.interval || "--";
    if (activeSet.size && !activeSet.has(symbol)) continue;
    if (!["15m", "5m", "3m", "1m"].includes(interval)) continue;
    if (!bySymbol.has(symbol)) {
      bySymbol.set(symbol, { symbol, lastUpdate: row.completed_at || row.last_success_at || row.started_at || row.queued_at || null, frames: { "15m": [], "5m": [], "3m": [], "1m": [] } });
    }
    const item = bySymbol.get(symbol);
    const updatedAt = row.completed_at || row.last_success_at || row.started_at || row.queued_at || null;
    item.lastUpdate = item.lastUpdate && item.lastUpdate > updatedAt ? item.lastUpdate : updatedAt;
    item.frames[interval].push({
      model_type: row.model_type,
      status: row.status,
      ok: row.status === 'success',
      updated_at: updatedAt,
      last_error: row.last_error,
    });
  }

  // Fold in recent results as a freshness overlay when they are newer than the registry entry state.
  for (const row of recentResults) {
    const symbol = String(row.symbol || "--").toUpperCase();
    const interval = row.interval || "--";
    if (activeSet.size && !activeSet.has(symbol)) continue;
    if (!["15m", "5m", "3m", "1m"].includes(interval)) continue;
    if (!bySymbol.has(symbol)) {
      bySymbol.set(symbol, { symbol, lastUpdate: row.updated_at || null, frames: { "15m": [], "5m": [], "3m": [], "1m": [] } });
    }
    const item = bySymbol.get(symbol);
    item.lastUpdate = item.lastUpdate && item.lastUpdate > row.updated_at ? item.lastUpdate : row.updated_at;
    item.frames[interval].push(row);
  }

  return Array.from(bySymbol.values()).sort((a, b) => String(a.symbol).localeCompare(String(b.symbol)));
}
function frameSummary(rows) {
  if (!rows || !rows.length) return '<span class="frame-summary empty">--</span>';
  const states = rows.map(r => String(r.status || (r.ok ? 'success' : 'failed')).toLowerCase());
  if (states.some(s => s.includes('running'))) return '<span class="frame-summary running">RUN</span>';
  if (states.every(s => s.includes('success') || s.includes('ready'))) return '<span class="frame-summary ready">READY</span>';
  if (states.every(s => s.includes('failed'))) return '<span class="frame-summary failed">FAIL</span>';
  return '<span class="frame-summary partial">PART</span>';
}
function renderTrainingResults(snapshot) {
  const body = qs("trainingResultsBody");
  if (!body) return;
  const rows = summarizeTrainingResults(snapshot);
  const html = rows.length
    ? rows.map((row) => `<tr><td>${escapeHtml(row.symbol)}</td><td>${frameSummary(row.frames['15m'])}</td><td>${frameSummary(row.frames['5m'])}</td><td>${frameSummary(row.frames['3m'])}</td><td>${frameSummary(row.frames['1m'])}</td><td>${fmtTime(row.lastUpdate)}</td></tr>`).join("")
    : emptyRow(6, "No completed training results yet.");
  replaceHtmlStable(body, html);
}
function trainingRowUpdatedAt(row) {
  return row?.updated_at || row?.completed_at || row?.last_success_at || row?.started_at || row?.queued_at || 0;
}
function ensureSignalSymbol(bySymbol, symbol, position) {
  if (!bySymbol.has(symbol)) {
    bySymbol.set(symbol, {
      symbol,
      gateMode: '--',
      position,
      action: '--',
      reason: '--',
      frames: { '1m': [], '3m': [], '5m': [], '15m': [] },
      latestByFrameModel: {},
    });
  }
  return bySymbol.get(symbol);
}
function getExecutionRuntime(snapshot) {
  // Execution surfaces are authoritative only when published by the runtime.
  // Never fall back to backtest or inference state for the Execution card.
  return snapshot?.execution_runtime || snapshot?.paper_execution_runtime || snapshot?.paper_execution || {};
}
function getExecutionSummary(runtime) {
  return runtime?.execution_summary || runtime?.summary || {};
}
function getDecisionReason(row) {
  const contract = row?.execution_contract || row?.components?.execution_contract || {};
  return row?.rejection_reason ||
    row?.execution_blocker ||
    row?.decision_blocker ||
    row?.blocker ||
    contract?.rejection_reason ||
    contract?.execution_blocker ||
    row?.components?.decision_blocker ||
    row?.components?.decision_state?.blocker ||
    row?.reason ||
    "--";
}
function getDecisionStatus(row) {
  const contract = row?.execution_contract || row?.components?.execution_contract || {};
  return row?.execution_status || contract?.execution_status || row?.status || row?.order_status || "--";
}
function getDecisionStage(row) {
  const contract = row?.execution_contract || row?.components?.execution_contract || {};
  return row?.execution_blocker_stage || contract?.execution_blocker_stage || row?.decision_stage || "--";
}
function getDecisionCategory(row) {
  return row?.category || row?.rejection_category || summarizeReason(getDecisionReason(row));
}
function formatReasonCounts(counts, limit = 3) {
  if (!counts || typeof counts !== 'object') return "--";
  const rows = Object.entries(counts).sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0)).slice(0, limit);
  if (!rows.length) return "--";
  return rows.map(([key, value]) => `${key}: ${value}`).join(" · ");
}
function getRiskBreakdown(row) {
  return row?.risk_breakdown || row?.components?.risk_breakdown || row?.components?.risk_decision?.components?.risk_breakdown || {};
}
function formatRiskBreakdown(row) {
  const risk = getRiskBreakdown(row);
  if (!risk || typeof risk !== 'object' || !Object.keys(risk).length) return "";
  const distortion = risk.risk_distortion_fraction == null ? "--" : fmtNumber(risk.risk_distortion_fraction, 4);
  const limit = risk.max_risk_distortion_fraction == null ? "--" : fmtNumber(risk.max_risk_distortion_fraction, 4);
  const entry = risk.entry_price == null ? "--" : fmtNumber(risk.entry_price, 5);
  const sl = risk.stop_loss == null ? "--" : fmtNumber(risk.stop_loss, 5);
  const tp = risk.take_profit == null ? "--" : fmtNumber(risk.take_profit, 5);
  const notional = risk.notional_usd == null ? "--" : fmtNumber(risk.notional_usd, 2);
  const clamp = escapeHtml(risk.clamp_reason || "none");
  return `<div class="feed-category">risk: distortion ${distortion}/${limit} · entry ${entry} · SL ${sl} · TP ${tp} · notional ${notional} · clamp ${clamp}</div>`;
}
function renderDecisionMessage(row) {
  const symbol = escapeHtml(row?.symbol || "--");
  const action = escapeHtml(row?.effective_action || row?.action || "--");
  const status = escapeHtml(getDecisionStatus(row));
  const stage = escapeHtml(getDecisionStage(row));
  const category = escapeHtml(getDecisionCategory(row));
  const reason = escapeHtml(getDecisionReason(row));
  const risk = formatRiskBreakdown(row);
  return `<div class="feed-head"><span>${symbol} ${action}</span><span>${status}</span></div><div class="feed-message">${reason}</div><div class="feed-category">${stage} · ${category}</div>${risk}`;
}
function summarizeSignals(snapshot) {
  const recentSignals = snapshot?.signal_generator?.recent_signals || [];
  const readinessRows = snapshot?.signal_generator?.readiness?.rows || [];
  const registryEntries = snapshot?.training_registry?.entries || [];
  const recentTrainingResults = snapshot?.training_registry?.recent_results || [];
  const executionRuntime = getExecutionRuntime(snapshot);
  const openPositions = Array.isArray(executionRuntime?.open_positions) ? executionRuntime.open_positions : [];
  const universe = snapshot?.market_data?.universe || [];
  const activeSet = new Set((Array.isArray(universe) ? universe : []).map((symbol) => String(symbol || '--').toUpperCase()));
  const openMap = new Map(openPositions.map(r => [String(r.symbol || '--').toUpperCase(), r.side || 'OPEN']));
  const bySymbol = new Map();

  for (const symbolRaw of universe) {
    const symbol = String(symbolRaw || '--').toUpperCase();
    ensureSignalSymbol(bySymbol, symbol, openMap.get(symbol) || 'Flat');
  }

  for (const row of readinessRows) {
    const symbol = String(row.symbol || "--").toUpperCase();
    if (activeSet.size && !activeSet.has(symbol)) continue;
    const item = ensureSignalSymbol(bySymbol, symbol, openMap.get(symbol) || "Flat");
    if (!row.ready && item.action === "--") {
      item.action = "HOLD";
      item.reason = row.reason || item.reason;
    }
    const interval = row.interval || "15m";
    const models = row.models || {};
    for (const [modelType, modelState] of Object.entries(models)) {
      const key = `${interval}:${modelType}`;
      item.latestByFrameModel[key] = {
        model_type: modelType,
        interval,
        status: modelState.ready ? "ready" : (modelState.status || modelState.reason || "missing"),
        ok: Boolean(modelState.ready),
        last_error: modelState.reason || "--",
        updated_at: 0,
      };
    }
  }

  for (const row of recentSignals) {
    const symbol = String(row.symbol || '--').toUpperCase();
    if (activeSet.size && !activeSet.has(symbol)) continue;
    const item = ensureSignalSymbol(bySymbol, symbol, openMap.get(symbol) || 'Flat');
    item.gateMode = row.mode || item.gateMode;
    item.action = row.action || item.action;
    item.reason = row.reason || item.reason;
  }

  for (const row of registryEntries) {
    const symbol = String(row.symbol || '--').toUpperCase();
    if (activeSet.size && !activeSet.has(symbol)) continue;
    const interval = row.interval || '--';
    if (!['15m', '5m', '3m', '1m'].includes(interval)) continue;
    const modelType = row.model_type || 'unknown';
    const item = ensureSignalSymbol(bySymbol, symbol, openMap.get(symbol) || 'Flat');
    const key = `${interval}:${modelType}`;
    const prev = item.latestByFrameModel[key];
    if (!prev || trainingRowUpdatedAt(row) >= trainingRowUpdatedAt(prev)) {
      item.latestByFrameModel[key] = {
        model_type: modelType,
        interval,
        status: row.status,
        ok: row.status === 'success',
        updated_at: trainingRowUpdatedAt(row),
        last_error: row.last_error,
      };
    }
  }

  for (const row of recentTrainingResults) {
    const symbol = String(row.symbol || '--').toUpperCase();
    if (activeSet.size && !activeSet.has(symbol)) continue;
    const interval = row.interval || '--';
    if (!['15m', '5m', '3m', '1m'].includes(interval)) continue;
    const modelType = row.model_type || 'unknown';
    const item = ensureSignalSymbol(bySymbol, symbol, openMap.get(symbol) || 'Flat');
    const key = `${interval}:${modelType}`;
    const prev = item.latestByFrameModel[key];
    if (!prev || trainingRowUpdatedAt(row) >= trainingRowUpdatedAt(prev)) {
      item.latestByFrameModel[key] = row;
    }
  }

  for (const item of bySymbol.values()) {
    item.frames = { '1m': [], '3m': [], '5m': [], '15m': [] };
    for (const row of Object.values(item.latestByFrameModel)) {
      const interval = row.interval || row.updated_interval || Object.keys(item.frames).find((iv) => row?.key?.startsWith(`${iv}:`));
      if (interval && item.frames[interval]) item.frames[interval].push(row);
    }
    delete item.latestByFrameModel;
  }

  return Array.from(bySymbol.values()).sort((a, b) => String(a.symbol).localeCompare(String(b.symbol)));
}
function compactModelCell(rows, interval) {
  if (!rows || !rows.length) return '<span class="frame-cell empty">--</span>';
  const chips = [];
  const byModel = {};
  for (const r of rows) byModel[r.model_type || 'M'] = r;
  const cfg = [['rf', 'R'], ['gbm', 'G']];
  if (['5m', '15m'].includes(interval)) cfg.push(['dqn', 'P']);
  for (const [key, label] of cfg) {
    const row = byModel[key];
    if (!row) { chips.push(compactStatusChip(label, 'neutral')); continue; }
    const status = String(row.status || (row.ok ? 'success' : 'failed')).toLowerCase();
    const cls = status.includes('success') || status.includes('ready') ? 'ok' : status.includes('running') || status.includes('queued') ? 'warn' : 'bad';
    chips.push(compactStatusChip(label, cls));
  }
  return `<div class="frame-cell">${chips.join('')}</div>`;
}
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
function summarizeReason(value) {
  const text = String(value || '--').trim();
  if (text.length <= 56) return text;
  return `${text.slice(0, 53)}…`;
}
function renderSignals(snapshot) {
  const body = qs("signalsBody");
  const messagesBody = qs("signalMessages");
  const rows = summarizeSignals(snapshot);

  if (body) {
    const html = rows.length ? rows.map((row) => {
      const fullReason = String(row.reason || '--');
      return `
        <tr>
          <td>${escapeHtml(row.symbol)}</td>
          <td>${escapeHtml(row.gateMode)}</td>
          <td>${escapeHtml(row.position)}</td>
          <td>${compactModelCell(row.frames['1m'], '1m')}</td>
          <td>${compactModelCell(row.frames['3m'], '3m')}</td>
          <td>${compactModelCell(row.frames['5m'], '5m')}</td>
          <td>${compactModelCell(row.frames['15m'], '15m')}</td>
          <td>${escapeHtml(row.action)}</td>
          <td class="reason-cell" title="${escapeHtml(fullReason)}"><span class="reason-summary">${escapeHtml(summarizeReason(fullReason))}</span></td>
        </tr>`;
    }).join("") : emptyRow(9, "No signal decisions published yet.");
    replaceHtmlStable(body, html);
  }

  if (messagesBody) {
    const recentSignals = snapshot?.signal_generator?.recent_signals || [];
    const html = recentSignals.length
      ? recentSignals.map((row) => feedItemHtml(`<div class="feed-head"><span>${escapeHtml(row.symbol || '--')}</span><span>${escapeHtml(row.action || '--')}</span><span>${escapeHtml(row.mode || '--')}</span></div><div class="feed-message signal-message-text">${escapeHtml(row.reason || '--')}</div>`, "execution-message-item")).join("")
      : feedItemHtml("No decision messages yet.", "execution-message-item");
    replaceHtmlStable(messagesBody, html);
  }
}
function renderExecution(snapshot) {
  const runtime = getExecutionRuntime(snapshot);
  const summary = getExecutionSummary(runtime);
  const openPositions = Array.isArray(runtime.open_positions) ? runtime.open_positions : [];
  setText("execOpenTrades", runtime.open_position_count ?? summary.open_positions ?? summary.open_count ?? openPositions.length ?? 0);
  setText("backtestTotalPnl", fmtNumber(summary.total_pnl ?? summary.realized_pnl_usd ?? 0, 2));
  setText("backtestWins", summary.wins ?? 0);
  setText("backtestLosses", summary.losses ?? 0);
  setText("backtestTrades", summary.completed_count ?? summary.trades ?? 0);
  setText("backtestWinRate", fmtNumber(summary.win_rate ?? 0, 2));

  const body = qs("executionOpenBody");
  if (body) {
    const html = openPositions.length ? openPositions.map((row) => {
      const normalizedLastPrice = row.last_price ?? row.latest_price ?? row.reference_price;
      const normalizedPnl = row.estimated_pnl ?? row.pnl_usd ?? row.unrealized_pnl_usd ?? 0;
      const lastPrice = normalizedLastPrice == null ? "--" : fmtNumber(normalizedLastPrice, 5);
      const staleTag = row.stale ? ' <span class="stale-tag">stale</span>' : "";
      const priceSource = row.price_source ? ` title="source: ${escapeHtml(row.price_source)}"` : "";
      return `<tr><td>${escapeHtml(row.symbol || "--")}</td><td>${escapeHtml(row.side || "--")}</td><td>${escapeHtml(row.mode || "--")}</td><td>${fmtNumber(row.entry_price, 5)}</td><td${priceSource}>${lastPrice}${staleTag}</td><td>${fmtTime(row.last_price_ts || row.updated_at)}</td><td>${fmtNumber(normalizedPnl, 2)}</td></tr>`;
    }).join("") : emptyRow(7, "No open positions.");
    replaceHtmlStable(body, html);
  }

  const msgBody = qs("executionMessages");
  if (msgBody) {
    const messages = Array.isArray(runtime.recent_messages) ? runtime.recent_messages : [];
    const decisions = Array.isArray(runtime.recent_decisions) ? runtime.recent_decisions : [];
    let html = "";
    if (messages.length) {
      html = messages.map((msg) => feedItemHtml(escapeHtml(msg || "--"), "execution-message-item")).join("");
    } else if (decisions.length) {
      html = decisions.slice(0, 20).map((row) => feedItemHtml(renderDecisionMessage(row), "execution-message-item")).join("");
    } else {
      html = emptyFeed("No execution messages yet.");
    }
    replaceHtmlStable(msgBody, html);
  }
}
function renderTradeHistory(snapshot) {
  const body = qs("tradeHistoryBody");
  if (!body) return;
  const runtime = getExecutionRuntime(snapshot);
  const rows = Array.isArray(runtime.trade_history) ? runtime.trade_history : [];
  const html = rows.map((row) => `<tr><td>${escapeHtml(row.symbol || "--")}</td><td>${escapeHtml(row.action || "--")}</td><td>${escapeHtml(row.mode || "--")}</td><td>${escapeHtml(row.entry_time || "--")}</td><td>${escapeHtml(row.exit_time || "--")}</td><td>${escapeHtml(row.duration || "--")}</td><td>${fmtNumber(row.notional_usd, 2)}</td><td>${fmtNumber(row.entry_price, 5)}</td><td>${row.exit_price == null ? "--" : fmtNumber(row.exit_price, 5)}</td><td>${fmtNumber(row.pnl_usd, 2)}</td><td>${escapeHtml(row.exit_reason || "--")}</td></tr>`).join("");
  replaceHtmlStable(body, html);
}
function renderMarket(snapshot) {
  const market = snapshot?.market_data || {};
  const priceDiag = market.runtime_price_snapshot || {};
  const expected = Number(priceDiag.expected_symbol_count ?? market.universe?.length ?? 0);
  const priced = Number(priceDiag.priced_symbol_count ?? market.priced_symbol_count ?? 0);
  const missing = Number(priceDiag.missing_symbol_count ?? priceDiag.missing_symbols?.length ?? 0);
  const stale = Number(priceDiag.stale_symbol_count ?? priceDiag.stale_symbols?.length ?? 0);
  const cacheStatus = market.price_cache_status || (priceDiag.exists ? (priceDiag.all_symbols_priced && stale === 0 ? 'ready' : 'degraded') : 'missing');
  setText("marketUniverseSource", market.universe_source || "--");
  setText("marketUniverseCount", Array.isArray(market.universe) ? market.universe.length : 0);
  setText("marketLastRefresh", fmtTime(market.last_universe_refresh_ts));
  setText("marketPriceCacheStatus", cacheStatus.toUpperCase());
  setText("marketPricedCount", `${priced}/${expected}`);
  setText("marketPriceCacheAge", priceDiag.age_seconds == null ? "--" : `${fmtNumber(priceDiag.age_seconds, 1)}s`);
  setText("marketMissingPrices", missing);
  setText("marketStalePrices", stale);
  setText("marketMaxPriceAge", priceDiag.max_price_age_seconds == null ? "--" : `${fmtNumber(priceDiag.max_price_age_seconds, 1)}s`);
  const diagEl = qs("marketPriceDiagnostics");
  if (diagEl) {
    const missingSymbols = Array.isArray(priceDiag.missing_symbols) && priceDiag.missing_symbols.length ? `missing: ${priceDiag.missing_symbols.slice(0, 8).join(', ')}` : 'missing: none';
    const staleSymbols = Array.isArray(priceDiag.stale_symbols) && priceDiag.stale_symbols.length ? `stale: ${priceDiag.stale_symbols.slice(0, 8).join(', ')}` : 'stale: none';
    const sources = priceDiag.price_source_counts && typeof priceDiag.price_source_counts === 'object'
      ? Object.entries(priceDiag.price_source_counts).map(([key, value]) => `${key}:${value}`).join(' · ')
      : '';
    const message = `${missingSymbols} · ${staleSymbols}${sources ? ` · sources: ${sources}` : ''}`;
    if (diagEl.textContent !== message) diagEl.textContent = message;
    diagEl.className = `diagnostic-strip compact-top-gap ${cacheStatus === 'ready' ? 'ok' : 'warn'}`;
  }
  const body = qs("marketUpdatesBody");
  if (!body) return;
  let updateRows = Array.isArray(market.latest_updates) ? market.latest_updates : [];
  if (!updateRows.length && Array.isArray(market.universe)) {
    const missingSet = new Set(Array.isArray(priceDiag.missing_symbols) ? priceDiag.missing_symbols : []);
    const staleSet = new Set(Array.isArray(priceDiag.stale_symbols) ? priceDiag.stale_symbols : []);
    updateRows = market.universe.map((symbol) => ({
      symbol,
      timeframe: 'price',
      rows_added: 0,
      status: missingSet.has(symbol) ? 'missing_price' : staleSet.has(symbol) ? 'stale_price' : cacheStatus,
      last_ts: null,
    }));
  }
  const html = updateRows.length
    ? updateRows.map((row) => `<tr><td>${escapeHtml(row.symbol || "--")}</td><td>${escapeHtml(row.timeframe || "--")}</td><td>${row.rows_added ?? 0}</td><td>${escapeHtml(row.status || "--")}</td><td>${fmtTime(row.last_ts)}</td></tr>`).join("")
    : emptyRow(5, "No market update rows published yet.");
  replaceHtmlStable(body, html);
}
function renderEvents(events) {
  state.events = Array.isArray(events) ? events : [];
  const body = qs("eventFeed");
  if (!body) return;
  const html = state.events.map((row) => feedItemHtml(`<div class="feed-head"><span class="feed-level">${escapeHtml(row.level || "--")}</span><span class="feed-component">${escapeHtml(row.component || "--")}</span><span class="feed-time">${escapeHtml(row.iso_ts || "--")}</span></div><div class="feed-message">${escapeHtml(row.message || "--")}</div><div class="feed-category">${escapeHtml(row.category || "--")}</div>`, `level-${String(row.level || "info").toLowerCase()}`)).join("");
  replaceHtmlStable(body, html);
}
function renderControlPlane(snapshot) {
  const control = snapshot?.control_plane || {};
  setText("cpPaused", control.paused ? "YES" : "NO");
  setText("cpPauseReason", control.pause_reason || "--");
  const circuit = control.circuit_breaker || snapshot?.inference?.circuit_breaker || {};
  setText("cpCircuit", circuit.triggered ? `${circuit.reason || "TRIGGERED"}` : "CLEAR");

  const actionsEl = qs("controlActions");
  if (actionsEl) {
    const html = (control.recent_actions || []).slice(0, 20).map((row) => feedItemHtml(`<div class="feed-head"><span>${escapeHtml(row.action || "--")}</span><span class="feed-time">${fmtTime(row.ts)}</span></div><div class="feed-message">${escapeHtml(row.message || "--")}</div><div class="feed-category">${escapeHtml(row.updated_by || "--")}</div>`)).join("");
    replaceHtmlStable(actionsEl, html);
  }

  const lineageEl = qs("decisionLineage");
  if (lineageEl) {
    const lineageRows = (snapshot?.decision_lineage_preview || control.decision_lineage_preview || snapshot?.rejection_visibility?.recent || []).slice(0, 20);
    const html = lineageRows.map((row) => {
      const lineage = row.lineage || {};
      const reason = getDecisionReason(row);
      const status = getDecisionStatus(row);
      const stage = getDecisionStage(row);
      const category = getDecisionCategory(row);
      return feedItemHtml(`<div class="feed-head"><span>${escapeHtml(row.symbol || "--")} ${escapeHtml(row.action || "--")}</span><span class="feed-time">${fmtTime(row.ts)}</span></div><div class="feed-message">${escapeHtml(reason)} | score ${fmtNumber(row.score, 6)}</div><div class="feed-category">${escapeHtml(status)} · ${escapeHtml(stage)} · ${escapeHtml(category)} · RF ${escapeHtml(lineage.rf?.candidate_id || "--")} · GBM ${escapeHtml(lineage.gbm?.direction?.candidate_id || "--")}</div>`);
    }).join("");
    replaceHtmlStable(lineageEl, html);
  }

  const observability = control.runtime_observability || {};
  const freshness = observability.freshness || {};
  const drift = observability.prediction_drift || {};
  const modelState = observability.model_state || {};
  const runtimePerf = observability.runtime_performance || {};
  const portfolioRuntime = runtimePerf.portfolio_runtime || {};
  setText("obsFreshRows", freshness.fresh_entries ?? 0);
  setText("obsStaleRows", freshness.stale_entries ?? 0);
  setText("obsShadowDivergence", fmtNumber(drift.shadow_divergence_rate ?? 0, 2));
  setText("obsAvgScore", drift.avg_score_recent == null ? "--" : fmtNumber(drift.avg_score_recent, 4));
  setText("obsActiveModels", modelState.active_model_count ?? 0);
  setText("obsRuntimeEquity", portfolioRuntime.equity_usd == null ? "--" : fmtNumber(portfolioRuntime.equity_usd, 2));

  const observabilityEl = qs("observabilitySummary");
  if (observabilityEl) {
    const priceDiag = snapshot?.market_data?.runtime_price_snapshot || {};
    const rejectionVisibility = observability.rejection_visibility || snapshot?.rejection_visibility || {};
    const tradingReadiness = snapshot?.trading_readiness || observability.trading_readiness || {};
    const items = [
      { title: "Trading Readiness", body: `${String(tradingReadiness.status || 'unknown').toUpperCase()} · ${tradingReadiness.summary || '--'}` },
      { title: "Freshness", body: `fresh ${freshness.fresh_entries ?? 0} · stale ${freshness.stale_entries ?? 0} · max age ${freshness.max_age_seconds == null ? "--" : fmtNumber(freshness.max_age_seconds, 2)}s` },
      { title: "Price Cache", body: `priced ${priceDiag.priced_symbol_count ?? 0}/${priceDiag.expected_symbol_count ?? 0} · missing ${priceDiag.missing_symbol_count ?? 0} · stale ${priceDiag.stale_symbol_count ?? 0} · age ${priceDiag.age_seconds == null ? "--" : fmtNumber(priceDiag.age_seconds, 1)}s` },
      { title: "Rejection Visibility", body: `recent ${rejectionVisibility.total_recent ?? 0} · categories ${formatReasonCounts(rejectionVisibility.by_category, 4)} · blockers ${formatReasonCounts(rejectionVisibility.by_reason, 2)}` },
      { title: "Prediction Drift", body: `recent non-hold ${fmtNumber(drift.recent_non_hold_rate ?? 0, 2)} · baseline ${fmtNumber(drift.baseline_non_hold_rate ?? 0, 2)} · shadow divergence ${fmtNumber(drift.shadow_divergence_rate ?? 0, 2)}` },
      { title: "Model State", body: `active ${modelState.active_model_count ?? 0} · candidates ${modelState.candidate_model_count ?? 0}` },
      { title: "Runtime Performance", body: `equity ${portfolioRuntime.equity_usd == null ? "--" : fmtNumber(portfolioRuntime.equity_usd, 2)} · realized ${portfolioRuntime.realized_pnl_usd == null ? "--" : fmtNumber(portfolioRuntime.realized_pnl_usd, 2)} · direction balance ${runtimePerf.recent_expected_direction_balance ?? 0}` },
    ];
    const html = items.map((row) => feedItemHtml(`<div class="feed-head"><span>${escapeHtml(row.title)}</span></div><div class="feed-message">${escapeHtml(row.body)}</div>`)).join("");
    replaceHtmlStable(observabilityEl, html);
  }

  const modelBody = qs("modelVersionsBody");
  if (modelBody) {
    const versions = snapshot?.model_versions || {};
    const rows = [];
    for (const [model, symbolMap] of Object.entries(versions)) {
      for (const [symbol, intervalMap] of Object.entries(symbolMap || {})) {
        for (const [interval, payload] of Object.entries(intervalMap || {})) {
          for (const row of payload.active || []) {
            rows.push(`<tr><td>${escapeHtml(model)}</td><td>${escapeHtml(symbol)}</td><td>${escapeHtml(interval)}</td><td>${escapeHtml(row.role || "bundle")}</td><td>${escapeHtml(row.stage || "active")}</td><td>${escapeHtml(row.candidate_id || "--")}</td><td>${escapeHtml(row.feature_set_version || "--")}</td><td>${fmtTime(row.promoted_at || row.updated_at)}</td><td>active</td></tr>`);
          }
          for (const row of payload.candidates || []) {
            rows.push(`<tr><td>${escapeHtml(model)}</td><td>${escapeHtml(symbol)}</td><td>${escapeHtml(interval)}</td><td>${escapeHtml(row.role || "bundle")}</td><td>${escapeHtml(row.stage || "candidate")}</td><td>${escapeHtml(row.candidate_id || "--")}</td><td>${escapeHtml(row.feature_set_version || "--")}</td><td>${fmtTime(row.created_at || row.updated_at)}</td><td><button type="button" class="rollback-btn" data-model="${escapeHtml(model)}" data-symbol="${escapeHtml(symbol)}" data-interval="${escapeHtml(interval)}" data-role="${escapeHtml(row.role || "")}" data-manifest="${escapeHtml(row.manifest_path)}">Make Active</button></td></tr>`);
          }
        }
      }
    }
    const changed = replaceHtmlStable(modelBody, rows.join(""));
    if (changed) {
      modelBody.querySelectorAll('.rollback-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
          btn.disabled = true;
          try {
            await postJson('/api/model-versions/rollback', {
              model_type: btn.getAttribute('data-model'),
              symbol: btn.getAttribute('data-symbol'),
              interval: btn.getAttribute('data-interval'),
              model_role: btn.getAttribute('data-role') || null,
              manifest_path: btn.getAttribute('data-manifest'),
            });
          } catch (err) {
            console.error(err);
          } finally {
            btn.disabled = false;
          }
        });
      });
    }
  }
}


function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function hasKeys(value) {
  return isPlainObject(value) && Object.keys(value).length > 0;
}
function isMeaningfulSnapshotPayload(snapshot) {
  if (!hasKeys(snapshot)) return false;
  return [
    "system",
    "controllers",
    "training",
    "training_runtime",
    "training_registry",
    "signal_generator",
    "signal_runtime",
    "backtest",
    "backtest_runtime",
    "execution_runtime",
    "paper_execution_runtime",
    "paper_execution",
    "inference",
    "market_data",
    "control_plane",
    "health",
    "trading_readiness",
  ].some((key) => {
    const value = snapshot[key];
    if (Array.isArray(value)) return value.length > 0;
    return hasKeys(value) || value === true || value === false || typeof value === "number" || typeof value === "string";
  });
}
function isPartialSnapshotPayload(snapshot) {
  if (!hasKeys(snapshot)) return true;
  const coreKeys = [
    "system",
    "controllers",
    "training",
    "training_runtime",
    "training_registry",
    "signal_generator",
    "signal_runtime",
    "execution_runtime",
    "paper_execution_runtime",
    "inference",
    "market_data",
    "control_plane",
  ];
  return coreKeys.filter((key) => Object.prototype.hasOwnProperty.call(snapshot, key)).length <= 2;
}

function meaningfulLeafScore(value) {
  if (Array.isArray(value)) return value.length > 0 ? Math.min(value.length, 10) : 0;
  if (isPlainObject(value)) {
    return Object.values(value).reduce((total, child) => total + meaningfulLeafScore(child), 0);
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    return normalized && normalized !== "--" && normalized !== "invalid date" && normalized !== "none" ? 1 : 0;
  }
  if (typeof value === "number") return Number.isFinite(value) && value !== 0 ? 1 : 0;
  if (typeof value === "boolean") return value ? 1 : 0;
  return 0;
}
function shouldPreserveRegressiveSection(path, previousSection, incomingSection) {
  const root = path[0] || "";
  const preserveRoots = new Set([
    "controllers",
    "training",
    "training_runtime",
    "training_registry",
    "scheduler",
    "signal_generator",
    "signal_runtime",
    "market_data",
    "model_versions",
    "control_plane",
  ]);
  if (path.length !== 1 || !preserveRoots.has(root)) return false;
  if (!isPlainObject(previousSection) || !isPlainObject(incomingSection)) return false;
  const previousScore = meaningfulLeafScore(previousSection);
  const incomingScore = meaningfulLeafScore(incomingSection);
  // A standalone dashboard process can briefly read a skeleton/default section while
  // the runtime process is still publishing. Keep the previous populated section instead
  // of repainting the panel with zeros/blanks. Real execution empties are handled below
  // at array-level exceptions, not by preserving the whole execution section.
  return previousScore >= 3 && incomingScore <= 1;
}

function shouldPreserveEmptyArray(path) {
  const joined = path.join('.');
  return ![
    'execution_runtime.open_positions',
    'paper_execution_runtime.open_positions',
    'paper_execution.open_positions',
    'execution_runtime.trade_history',
    'paper_execution_runtime.trade_history',
    'paper_execution.trade_history',
    'control_plane.recent_actions',
  ].includes(joined);
}
function mergeSnapshotForDisplay(previous, incoming, path = []) {
  if (!previous) return incoming;
  if (!isPlainObject(previous) || !isPlainObject(incoming)) return incoming ?? previous;
  const merged = { ...previous };
  Object.entries(incoming).forEach(([key, value]) => {
    const nextPath = [...path, key];
    if (value === null || value === undefined) return;
    if (Array.isArray(value)) {
      // Most empty arrays in partial dashboard payloads mean "section not included in this tick".
      // Keep the last known-good rows to avoid table/card blanking between refreshes.
      // Execution positions/trade history are allowed to become empty because that is a real state.
      if (!value.length && Array.isArray(previous[key]) && shouldPreserveEmptyArray(nextPath)) {
        merged[key] = previous[key];
      } else {
        merged[key] = value;
      }
      return;
    }
    if (isPlainObject(value)) {
      if (shouldPreserveRegressiveSection(nextPath, previous[key], value)) {
        merged[key] = previous[key];
        return;
      }
      merged[key] = hasKeys(value) ? mergeSnapshotForDisplay(previous[key] || {}, value, nextPath) : (previous[key] || value);
      return;
    }
    merged[key] = value;
  });
  return merged;
}
function markSnapshotRefreshStatus(message, isError = false) {
  state.lastRefreshError = isError ? message : null;
  const el = qs("snapshotRefreshStatus") || qs("lastUpdated");
  if (el) {
    if (isError) el.title = message || "Using last valid snapshot";
    else el.title = "";
  }
}
function applyDashboardSnapshot(rawSnapshot, source = "unknown") {
  const raw = rawSnapshot || {};
  if (!isMeaningfulSnapshotPayload(raw)) {
    if (state.lastGoodSnapshot) {
      state.snapshot = state.lastGoodSnapshot;
      markSnapshotRefreshStatus("Latest snapshot was empty/partial; holding last valid UI state.", true);
      return state.snapshot;
    }
    return normalizeSnapshot({});
  }
  const normalized = normalizeSnapshot(raw);
  // Always merge into last known-good state. This prevents a newer partial/empty subsection
  // from blanking a panel while another async refresh is still loading.
  const base = state.lastGoodSnapshot || state.snapshot || null;
  state.snapshot = base ? normalizeSnapshot(mergeSnapshotForDisplay(base, normalized)) : normalized;
  state.lastGoodSnapshot = state.snapshot;
  state.lastAppliedSnapshotSource = source;
  markSnapshotRefreshStatus(null, false);
  return state.snapshot;
}

function renderSnapshot(snapshot, source = "snapshot") {
  const displaySnapshot = applyDashboardSnapshot(snapshot, source);
  renderSystem(displaySnapshot);
  renderControllers(displaySnapshot);
  renderTraining(displaySnapshot);
  renderTrainingResults(displaySnapshot);
  renderSignals(displaySnapshot);
  renderExecution(displaySnapshot);
  renderTradeHistory(displaySnapshot);
  renderMarket(displaySnapshot);
  renderControlPlane(displaySnapshot);
  syncOpenPopup();
}
function mergeFastRuntimeSnapshot(payload) {
  const fast = normalizeSnapshot(payload || {});
  const current = state.lastGoodSnapshot || state.snapshot || {};
  const fastPatch = {
    updated_at: fast.updated_at || current.updated_at,
    system: fast.system || {},
    execution_runtime: fast.execution_runtime,
    paper_execution_runtime: fast.paper_execution_runtime,
    paper_execution: fast.paper_execution,
    inference: fast.inference || {},
    market_data: {
      runtime_price_snapshot: fast.market_data?.runtime_price_snapshot,
      price_cache_status: fast.market_data?.price_cache_status,
      priced_symbol_count: fast.market_data?.priced_symbol_count,
    },
    control_plane: fast.control_plane || {},
    health: fast.health || {},
    trading_readiness: fast.trading_readiness || {},
  };
  const next = mergeSnapshotForDisplay(current, fastPatch);
  state.snapshot = normalizeSnapshot(next);
  state.lastGoodSnapshot = state.snapshot;
  state.lastAppliedSnapshotSource = "fast_runtime";
  renderSystem(state.snapshot);
  renderExecution(state.snapshot);
  renderMarket(state.snapshot);
  renderControlPlane(state.snapshot);
  syncOpenPopup();
}
async function loadUiConfig() {
  try {
    const payload = await fetchJson("/api/ui-config");
    state.uiConfig = { ...(state.uiConfig || {}), ...(payload || {}) };
  } catch (err) {
    console.error("ui config load failed", err);
  }
}
async function loadInitialState(source = "snapshot") {
  const snapshot = await fetchJson("/api/snapshot");
  renderSnapshot(snapshot, source);
  try {
    const eventsPayload = await fetchJson("/api/events?limit=100");
    const events = Array.isArray(eventsPayload) ? eventsPayload : Array.isArray(eventsPayload?.events) ? eventsPayload.events : [];
    if (events.length || !state.lastGoodEvents.length) {
      state.lastGoodEvents = events;
      renderEvents(events);
    } else {
      renderEvents(state.lastGoodEvents);
    }
  } catch (err) {
    console.error("events refresh failed", err);
    if (state.lastGoodEvents.length) renderEvents(state.lastGoodEvents);
  }
}
function stopPolling() {
  if (state.pollHandle) {
    window.clearInterval(state.pollHandle);
    state.pollHandle = null;
  }
}
function startPolling(immediate = false) {
  if (state.pollHandle) return;
  let lastHeavyRefresh = 0;
  const refreshMs = Math.max(250, Number(state.uiConfig?.dashboard_refresh_ms || 1000));
  const heavyMs = Math.max(1000, Number(state.uiConfig?.heavy_snapshot_interval_sec || 5) * 1000);
  const tick = async () => {
    if (state.pollInFlight) return;
    state.pollInFlight = true;
    try {
      const now = Date.now();
      if (!state.snapshot || now - lastHeavyRefresh >= heavyMs) {
        await loadInitialState("poll_snapshot");
        lastHeavyRefresh = now;
        return;
      }
      const fastPayload = await fetchJson("/api/runtime/fast");
      mergeFastRuntimeSnapshot(fastPayload);
    } catch (err) {
      console.error("polling refresh failed", err);
      markSnapshotRefreshStatus("Polling refresh failed; holding last valid UI state.", true);
    } finally {
      state.pollInFlight = false;
    }
  };
  if (immediate) tick();
  state.pollHandle = window.setInterval(tick, refreshMs);
}
function stopSocketWatchdog() {
  if (state.socketWatchdogHandle) {
    window.clearInterval(state.socketWatchdogHandle);
    state.socketWatchdogHandle = null;
  }
}
function startSocketWatchdog() {
  stopSocketWatchdog();
  const heartbeatWindowMs = Math.max(5000, Number(state.uiConfig?.heavy_snapshot_interval_sec || 5) * 3000);
  state.socketWatchdogHandle = window.setInterval(() => {
    if (!state.socketConnected || !state.socket) return;
    if (Date.now() - state.lastSocketMessageAt > heartbeatWindowMs) {
      markSnapshotRefreshStatus("Dashboard websocket heartbeat timed out; using polling fallback.", true);
      startPolling(true);
      state.socket.close();
    }
  }, Math.max(1000, Number(state.uiConfig?.fast_snapshot_interval_sec || 1) * 2000));
}
function connectSocket() {
  if (state.reconnectHandle) {
    window.clearTimeout(state.reconnectHandle);
    state.reconnectHandle = null;
  }
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws`;
  const socket = new WebSocket(url);
  state.socket = socket;
  socket.onopen = () => {
    state.socketConnected = true;
    state.lastSocketMessageAt = Date.now();
    stopPolling();
    startSocketWatchdog();
  };
  socket.onmessage = (event) => {
    state.lastSocketMessageAt = Date.now();
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "snapshot") renderSnapshot(payload.payload || {}, "websocket_snapshot");
      else if (payload.type === "fast_runtime") mergeFastRuntimeSnapshot(payload.payload || {});
      else if (payload.type === "events") {
        const events = Array.isArray(payload.payload) ? payload.payload : [];
        if (events.length || !state.lastGoodEvents.length) {
          state.lastGoodEvents = events;
          renderEvents(events);
        } else {
          renderEvents(state.lastGoodEvents);
        }
      } else if (payload.type === "heartbeat") {
        markSnapshotRefreshStatus(null, false);
      }
    } catch (err) { console.error("socket parse failed", err); }
  };
  socket.onerror = () => {
    state.socketConnected = false;
    stopSocketWatchdog();
    markSnapshotRefreshStatus("Dashboard websocket error; using polling fallback.", true);
    startPolling(true);
  };
  socket.onclose = () => {
    state.socket = null;
    state.socketConnected = false;
    stopSocketWatchdog();
    startPolling(true);
    if (!state.reconnectHandle) {
      state.reconnectHandle = window.setTimeout(() => {
        state.reconnectHandle = null;
        connectSocket();
      }, 1500);
    }
  };
}
function setCollapseState(cardId, collapsed) {
  const card = qs(cardId);
  if (!card) return;
  card.classList.toggle('collapsed', collapsed);
  localStorage.setItem(`collapse_${cardId}`, collapsed ? '1' : '0');
  const btn = card.querySelector('.collapse-btn');
  if (btn) btn.textContent = collapsed ? '+' : '–';
}
function bindCollapseButtons() {
  document.querySelectorAll('.collapse-btn').forEach((btn) => {
    const target = btn.getAttribute('data-target');
    const saved = localStorage.getItem(`collapse_${target}`) === '1';
    setCollapseState(target, saved);
    btn.addEventListener('click', () => {
      const card = qs(target);
      setCollapseState(target, !card.classList.contains('collapsed'));
    });
  });
}

function syncOpenPopup() {
  const overlay = qs("popupOverlay");
  const content = qs("popupContent");
  const title = qs("popupTitle");
  const sourceCard = state.openPopupCardId ? qs(state.openPopupCardId) : null;
  if (!overlay || overlay.classList.contains("hidden") || !content || !title || !sourceCard) return;
  const sourceMarkup = sourceCard.innerHTML;
  if (state.lastPopupMarkup === sourceMarkup) return;
  const scrollTop = content.scrollTop;
  const scrollLeft = content.scrollLeft;
  const clone = sourceCard.cloneNode(true);
  clone.id = `${state.openPopupCardId}PopupClone`;
  clone.classList.add("popup-card-clone");
  clone.classList.remove("collapsed");
  clone.querySelectorAll('.popup-btn, .collapse-btn').forEach((el) => el.remove());
  content.replaceChildren(clone);
  title.textContent = sourceCard.querySelector('h3')?.textContent || 'Expanded View';
  content.scrollTop = scrollTop;
  content.scrollLeft = scrollLeft;
  state.lastPopupMarkup = sourceMarkup;
}
function openPopupForCard(cardId) {
  const overlay = qs("popupOverlay");
  if (!overlay || !qs(cardId)) return;
  state.openPopupCardId = cardId;
  state.lastPopupMarkup = null;
  overlay.classList.remove('hidden');
  overlay.setAttribute('aria-hidden', 'false');
  document.body.classList.add('popup-open');
  syncOpenPopup();
}
function closePopup() {
  const overlay = qs("popupOverlay");
  const content = qs("popupContent");
  if (!overlay || overlay.classList.contains('hidden')) return;
  overlay.classList.add('hidden');
  overlay.setAttribute('aria-hidden', 'true');
  if (content) content.innerHTML = '';
  state.openPopupCardId = null;
  state.lastPopupMarkup = null;
  document.body.classList.remove('popup-open');
}
function bindPopupButtons() {
  document.querySelectorAll('.popup-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      openPopupForCard(btn.getAttribute('data-popup-target'));
    });
  });
  qs('popupClose')?.addEventListener('click', closePopup);
  qs('popupOverlay')?.addEventListener('click', (event) => {
    if (event.target?.getAttribute('data-popup-close') === '1') closePopup();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closePopup();
  });
}
function bindLogin() {
  const themeSelect = qs("themeSelect");
  const logout = qs("logoutButton");
  const savedTheme = localStorage.getItem("dashboard_theme") || "current";
  document.documentElement.setAttribute("data-theme", savedTheme);
  if (themeSelect) {
    themeSelect.value = savedTheme;
    themeSelect.addEventListener("change", () => {
      const value = themeSelect.value || "current";
      document.documentElement.setAttribute("data-theme", value);
      localStorage.setItem("dashboard_theme", value);
    });
  }
  logout?.classList.add("hidden");
}
async function bootstrap() {
  bindLogin();
  bindCollapseButtons();
  bindPopupButtons();
  showLogin();
  const retryAllBlockedBtn = qs("retryAllBlockedBtn");
  qs("pauseTradingBtn")?.addEventListener("click", async () => { try { await postJson("/api/control/pause", { paused: true, reason: "manual_pause" }); } catch (err) { console.error(err); } });
  qs("resumeTradingBtn")?.addEventListener("click", async () => { try { await postJson("/api/control/pause", { paused: false, reason: "manual_resume" }); } catch (err) { console.error(err); } });
  qs("resetCircuitBtn")?.addEventListener("click", async () => { try { await postJson("/api/control/circuit/reset", { reason: "manual_reset" }); } catch (err) { console.error(err); } });
  if (retryAllBlockedBtn) {
    retryAllBlockedBtn.addEventListener("click", async () => {
      retryAllBlockedBtn.disabled = true;
      try { await postJson("/api/symbols/retry", { retry_all_blocked: true }); } catch (err) { console.error(err); } finally { retryAllBlockedBtn.disabled = false; }
    });
  }
  try {
    await loadUiConfig();
    await loadInitialState();
  } catch (err) {
    console.error("initial dashboard load failed", err);
  }
  connectSocket();
}
document.addEventListener("DOMContentLoaded", bootstrap);