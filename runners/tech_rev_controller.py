from __future__ import annotations

import concurrent.futures
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass

from core.job_definition import TrainingJob
from core.log_manager import get_logger
from core.symbol_management import (
    clear_expired_symbol_backoff_entry,
    clear_symbol_actions,
    classify_symbol_universe,
    compute_active_symbols,
    default_symbol_lifecycle_entry,
    is_symbol_backed_off_entry,
    load_symbol_actions,
    load_symbol_state,
    normalize_symbol_list,
    mark_symbol_warming_entry,
    required_market_history_rows,
    record_symbol_data_readiness_entry,
    record_symbol_execution_readiness_entry,
    record_symbol_failure_entry,
    record_symbol_readiness_result_entry,
    record_symbol_success_entry,
    record_symbol_warmup_timeout_entry,
    request_symbol_retry_entry,
    save_symbol_state,
    timed_out_symbols_payload,
    write_tradable_symbols_yaml,
)
from inference.readiness import model_registry_readiness
from monitoring.publisher import MonitorPublisher
from scheduling.future_collector import collect_completed_futures
from scheduling.priority_manager import build_due_training_jobs, build_training_backlog_diagnostics
from services.training_engine import TrainingEngine, TrainingProgressSnapshot


@dataclass(slots=True)
class SymbolRetryState:
    status: str = "ACTIVE"  # ACTIVE / TIMED_OUT / BLOCKED
    consecutive_failures: int = 0
    cooldown_level: int = 0
    retry_not_before_ts: float | None = None
    last_error: str = "--"
    last_failed_model: str = "--"
    last_failed_interval: str = "--"
    eligible_for_auto_retry: bool = True
    manually_blocked: bool = False
    retry_requested: bool = False
    blocked_since_ts: float | None = None
    readiness_failures: int = 0
    readiness_not_ready_since_ts: float | None = None
    readiness_last_checked_ts: float | None = None
    readiness_last_reason: str = "--"
    activation_stage: str = "execution_active"
    warmup_requested_ts: float | None = None
    data_ready: bool = False
    data_ready_ts: float | None = None
    data_ready_reason: str = "unchecked"
    execution_eligible: bool = False
    model_ready_ts: float | None = None
    model_ready_reason: str = "unchecked"
    blocked_position_close_required: bool = False
    blocked_position_close_status: str = "not_required"
    blocked_position_close_reason: str = ""
    blocked_position_close_ts: float | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "SymbolRetryState":
        return cls(**{k: payload.get(k, getattr(cls(), k)) for k in cls.__dataclass_fields__.keys()})

    def to_dict(self) -> dict:
        return asdict(self)


class TechRevController:
    def __init__(self, paths, registry) -> None:
        self.paths = paths
        self.registry = registry
        self.logger = get_logger("runners.tech_rev_controller", paths)
        self.monitor = MonitorPublisher(paths)
        self.engine = TrainingEngine(paths)
        self.loop_sleep_sec = float(paths.config.scheduling.loop_sleep_sec)
        self.max_rf = int(paths.config.scheduling.max_rf_workers)
        self.max_gbm = int(paths.config.scheduling.max_gbm_workers)
        self.max_dqn = int(paths.config.scheduling.max_dqn_workers)
        self.max_hmm = int(getattr(paths.config.scheduling, "max_hmm_workers", self.max_dqn) or self.max_dqn)
        self.symbol_backoff_first_sec = float(paths.config.scheduling.symbol_backoff_first_sec)
        self.symbol_backoff_repeat_sec = float(paths.config.scheduling.symbol_backoff_repeat_sec)
        self.max_failures_before_block = int(paths.config.symbols.max_failures_before_block)
        self.target_active_symbols = int(paths.config.symbols.target_active_symbols)
        self.ranked_universe_size = int(getattr(paths.config.symbols, "ranked_universe_size", 20) or 20)
        self.auto_replace_symbols = bool(paths.config.symbols.auto_replace_symbols)
        self.allow_manual_unblock = bool(paths.config.symbols.allow_manual_unblock)
        self.manual_unblock_required = bool(getattr(paths.config.symbols, "manual_unblock_required", self.allow_manual_unblock))
        self.max_submissions_per_loop = int(getattr(paths.config.scheduling, "max_submissions_per_loop", 3) or 3)
        self.max_submissions_per_symbol_per_loop = int(getattr(paths.config.scheduling, "max_submissions_per_symbol_per_loop", 1) or 1)
        self.readiness_failure_check_sec = float(getattr(paths.config.symbols, "readiness_failure_check_sec", 300.0) or 300.0)
        self.readiness_failure_min_age_sec = float(getattr(paths.config.symbols, "readiness_failure_min_age_sec", 900.0) or 900.0)
        self.warmup_timeout_sec = float(getattr(paths.config.symbols, "warmup_timeout_sec", 14400.0) or 0.0)
        raw_model_caps = getattr(paths.config.scheduling, "max_model_submissions_per_loop", {"rf": 1, "gbm": 1, "dqn": 1})
        self.max_model_submissions_per_loop = {
            "rf": int(raw_model_caps.get("rf", 1)) if isinstance(raw_model_caps, dict) else 1,
            "gbm": int(raw_model_caps.get("gbm", 1)) if isinstance(raw_model_caps, dict) else 1,
            "dqn": int(raw_model_caps.get("dqn", 1)) if isinstance(raw_model_caps, dict) else 1,
            "hmm": int(raw_model_caps.get("hmm", 1)) if isinstance(raw_model_caps, dict) else 1,
        }
        self.rf_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.gbm_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.dqn_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.hmm_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.inflight: dict[concurrent.futures.Future, TrainingJob] = {}
        self.snapshot = TrainingProgressSnapshot()
        self._symbol_retry: dict[str, SymbolRetryState] = {}
        self._load_state_from_disk()

    def _load_state_from_disk(self) -> None:
        state = load_symbol_state(self.paths)
        self._symbol_retry = {
            str(symbol).upper(): SymbolRetryState.from_dict(entry if isinstance(entry, dict) else {})
            for symbol, entry in state.get("symbols", {}).items()
        }
        for symbol in normalize_symbol_list(state.get("active_symbols", [])):
            self._get_symbol_state(symbol)

    def _load_symbol_control_config(self) -> tuple[list[str], set[str], list[str]]:
        """Read static symbol controls, not lifecycle truth.

        `tradable_symbols.yaml` is now an export/compatibility file owned by
        TechRev. We still read pinned/skip controls from it and use `symbols`
        only as a one-time bootstrap when symbol_state has neither active
        symbols nor ranked replacement candidates.
        """
        symbol_cfg = self.paths.load_yaml(self.paths.tradable_symbols_path)
        pinned = normalize_symbol_list(symbol_cfg.get("pinned_symbols", []))
        skip = set(normalize_symbol_list(symbol_cfg.get("skip_symbols", [])))
        legacy_symbols = normalize_symbol_list(symbol_cfg.get("symbols", []))
        return pinned, skip, legacy_symbols

    def _current_active_from_lifecycle_state(self, state: dict, legacy_symbols: list[str]) -> tuple[list[str], str]:
        current_active = normalize_symbol_list(state.get("active_symbols", []))
        if current_active:
            return current_active, "symbol_state.active_symbols"
        ranked = normalize_symbol_list(state.get("ranked_candidates", []))
        if ranked:
            return [], "symbol_state.ranked_candidates"
        return legacy_symbols, "legacy_tradable_symbols_bootstrap"

    def _persist_state(self) -> None:
        state = load_symbol_state(self.paths)
        state["lifecycle_owner"] = "tech_rev_controller"
        state["lifecycle_version"] = 5
        state["target_active_symbols"] = self.target_active_symbols
        state["ranked_universe_size"] = self.ranked_universe_size
        state["symbols"] = {symbol: entry.to_dict() for symbol, entry in self._symbol_retry.items()}
        ranked = normalize_symbol_list(state.get("ranked_candidates", []))
        pinned, skip, legacy_symbols = self._load_symbol_control_config()
        current_active, active_source = self._current_active_from_lifecycle_state(state, legacy_symbols)
        previous_active_set = set(normalize_symbol_list(current_active))
        active = compute_active_symbols(
            ranked_candidates=ranked if self.auto_replace_symbols else [],
            target_active=self.target_active_symbols,
            pinned_symbols=pinned,
            skip_symbols=skip,
            state=state,
            current_active_symbols=current_active,
            ranked_universe_size=self.ranked_universe_size,
        )
        now_ts = time.time()
        symbols_state = state.setdefault("symbols", {})
        for symbol in active:
            entry = symbols_state.get(symbol) if isinstance(symbols_state.get(symbol), dict) else {}
            if symbol not in previous_active_set:
                symbols_state[symbol] = mark_symbol_warming_entry(
                    entry,
                    now_ts=now_ts,
                    reason="promoted_symbol_data_warmup_required",
                )
                self.logger.info("Promoted symbol entering data warmup symbol=%s", symbol)
            elif symbol not in symbols_state:
                symbols_state[symbol] = self._get_symbol_state(symbol).to_dict()
        state["symbols"] = symbols_state
        warmup_blocked = self._refresh_active_symbol_warmup_state(state, active, now_ts=now_ts)
        if warmup_blocked:
            # A warmup timeout can block a current active symbol. Recompute the
            # active universe immediately so the replacement from the ranked
            # pool is persisted in the same lifecycle write, not one loop later.
            active = compute_active_symbols(
                ranked_candidates=ranked if self.auto_replace_symbols else [],
                target_active=self.target_active_symbols,
                pinned_symbols=pinned,
                skip_symbols=skip,
                state=state,
                current_active_symbols=active,
                ranked_universe_size=self.ranked_universe_size,
            )
            for symbol in active:
                if symbol not in state.setdefault("symbols", {}):
                    state["symbols"][symbol] = mark_symbol_warming_entry(
                        {},
                        now_ts=now_ts,
                        reason="replacement_after_warmup_timeout",
                    )
            self._refresh_active_symbol_warmup_state(state, active, now_ts=now_ts)
        for symbol, entry in (state.get("symbols") or {}).items():
            if isinstance(entry, dict):
                self._symbol_retry[str(symbol).upper()] = SymbolRetryState.from_dict(entry)
        buckets = classify_symbol_universe(
            ranked_candidates=ranked,
            active_symbols=active,
            state=state,
            ranked_universe_size=self.ranked_universe_size,
        )
        state.update(buckets)
        state["active_symbols_source"] = active_source
        save_symbol_state(self.paths, state)
        try:
            write_tradable_symbols_yaml(
                self.paths,
                symbols=active,
                pinned_symbols=pinned,
                skip_symbols=skip,
                source="tech_rev_lifecycle_export",
            )
        except Exception as exc:
            # The symbol_state.json lifecycle record is canonical.  A compatibility
            # export failure must not crash the training controller or revert the
            # lifecycle truth to a stale universe; the next loop will retry the
            # atomic export.
            self.logger.warning("Tradable symbols compatibility export failed: %s", exc)
        self._seed_registry_for_active_symbols(active)

    def _seed_registry_for_active_symbols(self, active_symbols: list[str]) -> None:
        """Pre-seed registry rows for active symbols so readiness sees managed work.

        Active symbols that are waiting for queue capacity should surface as
        idle/queued/running rather than `registry_status:missing`.
        """
        jobs: list[tuple[str, str, str]] = []
        dqn_allowed = list(self.paths.config.training.dqn.allowed_intervals or [])
        for symbol in active_symbols:
            for interval in list(self.paths.config.scheduling.priority_intervals):
                if bool(self.paths.config.training.rf.enabled):
                    jobs.append(("rf", symbol, interval))
                if bool(self.paths.config.training.gbm.enabled):
                    jobs.append(("gbm", symbol, interval))
                if bool(self.paths.config.training.dqn.enabled) and interval in dqn_allowed:
                    jobs.append(("dqn", symbol, interval))
                hmm_cfg = getattr(self.paths.config.training, "hmm", None)
                if bool(getattr(hmm_cfg, "enabled", False)) and bool(getattr(hmm_cfg, "candidate_training_enabled", False)):
                    jobs.append(("hmm", symbol, interval))
        if jobs:
            self.registry.ensure_entries(jobs)

    def _market_table_state(self, symbol: str, interval: str) -> tuple[int, int]:
        db_path = self.paths.market_db_path(symbol)
        if not db_path.exists():
            return 0, 0
        table = f"klines_{interval}"
        try:
            with sqlite3.connect(str(db_path), timeout=5) as conn:
                row = conn.execute(f'SELECT COUNT(1), COALESCE(MAX(timestamp), 0) FROM "{table}"').fetchone()
        except sqlite3.Error:
            return 0, 0
        if not row:
            return 0, 0
        try:
            return int(row[0] or 0), int(row[1] or 0)
        except Exception:
            return 0, 0

    def _symbol_data_readiness(self, symbol: str) -> tuple[bool, str]:
        missing: list[str] = []
        anchor = str(getattr(self.paths.config.frames, "anchor_interval", "15m") or "15m")
        rf_min_rows = int(getattr(self.paths.config.training.rf, "min_rows", 300) or 300)
        requirements = required_market_history_rows(
            fusion_intervals=[str(interval) for interval in list(self.paths.config.frames.fusion_intervals)],
            anchor_interval=anchor,
            rf_min_rows=rf_min_rows,
        )
        for interval, required in requirements.items():
            rows, last_ts = self._market_table_state(symbol, interval)
            if rows < required or last_ts <= 0:
                missing.append(f"{interval}:{rows}/{required}")
        if missing:
            return False, "market_data_warming:" + ",".join(missing[:8])
        return True, "market_data_ready"

    def _refresh_active_symbol_warmup_state(self, state: dict, active_symbols: list[str], *, now_ts: float) -> bool:
        symbols_state = state.setdefault("symbols", {})
        lifecycle_blocked = False
        for symbol in normalize_symbol_list(active_symbols):
            entry = symbols_state.get(symbol) if isinstance(symbols_state.get(symbol), dict) else {}
            data_ready, data_reason = self._symbol_data_readiness(symbol)
            updated, _ = record_symbol_data_readiness_entry(
                entry,
                ready=data_ready,
                reason=data_reason,
                now_ts=now_ts,
            )
            warmup_started = updated.get("warmup_requested_ts")
            warmup_age = (now_ts - float(warmup_started)) if warmup_started is not None else 0.0
            if (
                self.warmup_timeout_sec > 0
                and not data_ready
                and warmup_started is not None
                and warmup_age >= self.warmup_timeout_sec
            ):
                reason = f"warmup_timeout:data_not_ready_after_{int(self.warmup_timeout_sec)}s:{data_reason}"
                updated = record_symbol_warmup_timeout_entry(updated, reason=reason, now_ts=now_ts)
                lifecycle_blocked = True
                self.logger.warning("Symbol blocked after warmup timeout symbol=%s reason=%s", symbol, reason)
                symbols_state[symbol] = updated
                continue
            model_ready, model_reason = self._symbol_execution_readiness(symbol)
            updated, _ = record_symbol_execution_readiness_entry(
                updated,
                ready=model_ready,
                reason=model_reason,
                now_ts=now_ts,
            )
            symbols_state[symbol] = updated
        return lifecycle_blocked

    def _ensure_pools(self) -> None:
        if self.rf_pool is None:
            self.rf_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_rf, thread_name_prefix="techrev-rf")
        if self.gbm_pool is None:
            self.gbm_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_gbm, thread_name_prefix="techrev-gbm")
        if self.dqn_pool is None:
            self.dqn_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_dqn, thread_name_prefix="techrev-dqn")
        if self.hmm_pool is None:
            self.hmm_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_hmm, thread_name_prefix="techrev-hmm")

    def _submit_job(self, job: TrainingJob) -> None:
        pool = (
            self.rf_pool
            if job.model_type == "rf"
            else self.gbm_pool
            if job.model_type == "gbm"
            else self.hmm_pool
            if job.model_type == "hmm"
            else self.dqn_pool
        )
        assert pool is not None
        future = pool.submit(self.engine.train_model, job.model_type, job.symbol, job.interval)
        self.inflight[future] = job
        self.registry.mark_running(job.model_type, job.symbol, job.interval)

    def _get_symbol_state(self, symbol: str) -> SymbolRetryState:
        key = str(symbol).upper()
        if key not in self._symbol_retry:
            self._symbol_retry[key] = SymbolRetryState.from_dict(default_symbol_lifecycle_entry())
        return self._symbol_retry[key]

    def _is_symbol_backed_off(self, symbol: str, now_ts: float) -> bool:
        state = self._get_symbol_state(symbol)
        return is_symbol_backed_off_entry(state.to_dict(), now_ts)

    def _clear_expired_symbol_backoff(self, now_ts: float) -> None:
        changed = False
        for symbol, state in list(self._symbol_retry.items()):
            updated, did_change = clear_expired_symbol_backoff_entry(state.to_dict(), now_ts)
            if did_change:
                self._symbol_retry[symbol] = SymbolRetryState.from_dict(updated)
                changed = True
        if changed:
            self._persist_state()

    def _record_symbol_failure(self, job: TrainingJob, reason: str) -> None:
        now_ts = time.time()
        current = self._get_symbol_state(job.symbol)
        updated, action = record_symbol_failure_entry(
            current.to_dict(),
            model_type=job.model_type,
            interval=job.interval,
            reason=str(reason),
            max_failures_before_block=self.max_failures_before_block,
            backoff_first_sec=self.symbol_backoff_first_sec,
            backoff_repeat_sec=self.symbol_backoff_repeat_sec,
            now_ts=now_ts,
        )
        self._symbol_retry[str(job.symbol).upper()] = SymbolRetryState.from_dict(updated)
        if action == "blocked":
            self.logger.warning(
                "Symbol permanently blocked symbol=%s failures=%s reason=%s",
                job.symbol,
                updated.get("consecutive_failures"),
                updated.get("last_error"),
            )
        else:
            retry_at = updated.get("retry_not_before_ts") or now_ts
            self.logger.warning(
                "Symbol backoff activated symbol=%s retry_at=%s last_error=%s",
                job.symbol,
                int(float(retry_at)),
                updated.get("last_error"),
            )
        self._persist_state()

    def _record_symbol_success(self, symbol: str) -> None:
        state = self._get_symbol_state(symbol)
        self._symbol_retry[str(symbol).upper()] = SymbolRetryState.from_dict(record_symbol_success_entry(state.to_dict()))
        self._persist_state()

    def _timed_out_symbols_payload(self, blocked_jobs: list[TrainingJob], now_ts: float) -> list[dict[str, object]]:
        due_counts: dict[str, int] = {}
        for job in blocked_jobs:
            due_counts[job.symbol] = due_counts.get(job.symbol, 0) + 1
        return timed_out_symbols_payload(
            {symbol: state.to_dict() for symbol, state in self._symbol_retry.items()},
            due_counts,
            now_ts,
        )

    def _ensure_managed_universe(self) -> None:
        """Enforce TechRev-owned symbol lifecycle state.

        This is safe to call every loop. It gives the scheduler a deterministic
        active universe, fills holes after BLOCKED symbols when auto replacement
        is enabled, and exports `tradable_symbols.yaml` only as a compatibility
        view of the canonical `symbol_state.json` lifecycle state.
        """
        before = load_symbol_state(self.paths)
        before_active = [str(s).upper() for s in before.get("active_symbols", []) if s]
        self._persist_state()
        after = load_symbol_state(self.paths)
        after_active = [str(s).upper() for s in after.get("active_symbols", []) if s]
        if before_active != after_active:
            self.logger.info(
                "Active symbol universe refreshed count=%s symbols=%s",
                len(after_active),
                after_active,
            )

    def _required_anchor_readiness_models(self) -> list[str]:
        """Return the execution-critical anchor models.

        Step 7 intentionally keeps execution eligibility tied only to the 15m
        anchor RF and GBM incumbents. Lower timeframes and the confirmatory
        policy can enhance decisions, but they must not keep a symbol out of the
        execution-eligible set when the anchor contract is ready.
        """
        required: list[str] = []
        training_cfg = self.paths.config.training
        if bool(training_cfg.rf.enabled):
            required.append("rf")
        if bool(training_cfg.gbm.enabled):
            required.append("gbm")
        return required

    def _symbol_execution_readiness(self, symbol: str) -> tuple[bool, str]:
        blocking: list[str] = []
        anchor_interval = str(getattr(self.paths.config.frames, "anchor_interval", "15m") or "15m")
        for model_type in self._required_anchor_readiness_models():
            readiness = model_registry_readiness(self.paths, model_type=model_type, symbol=symbol, interval=anchor_interval)
            if not readiness.ready:
                blocking.append(f"{model_type}:{anchor_interval}:{readiness.reason}")
        if blocking:
            return False, "anchor_readiness_not_ready:" + ",".join(blocking)
        return True, "anchor_ready"

    def _evaluate_active_symbol_readiness(self) -> None:
        """Block and replace symbols that remain PART/not-ready repeatedly.

        This is deliberately lifecycle-level logic rather than scheduler logic.
        The scheduler should never decide symbol ownership; it only schedules the
        active symbols TechRev exposes through symbol_state.json.
        """
        state = load_symbol_state(self.paths)
        active_symbols = normalize_symbol_list(state.get("active_symbols", []))
        now_ts = time.time()
        changed = False
        for symbol in active_symbols:
            current = self._get_symbol_state(symbol)
            current_payload = current.to_dict()
            if str(current.status).upper() == "BLOCKED":
                continue
            if str(current_payload.get("activation_stage") or "").lower() == "warming_data" or not bool(current_payload.get("data_ready", False)):
                continue
            last_checked = current.readiness_last_checked_ts
            if last_checked is not None and (now_ts - float(last_checked)) < max(1.0, self.readiness_failure_check_sec):
                continue
            ready, reason = self._symbol_execution_readiness(symbol)
            updated, action = record_symbol_readiness_result_entry(
                current.to_dict(),
                ready=ready,
                reason=reason,
                max_readiness_failures=self.max_failures_before_block,
                min_not_ready_age_sec=self.readiness_failure_min_age_sec,
                check_interval_sec=self.readiness_failure_check_sec,
                now_ts=now_ts,
            )
            if action in {"reset", "observing", "counted", "blocked"}:
                self._symbol_retry[str(symbol).upper()] = SymbolRetryState.from_dict(updated)
                changed = True
            if action == "blocked":
                self.logger.warning(
                    "Symbol blocked after repeated readiness failures symbol=%s failures=%s reason=%s",
                    symbol,
                    updated.get("readiness_failures"),
                    updated.get("readiness_last_reason"),
                )
            elif action == "counted":
                self.logger.warning(
                    "Symbol readiness failure counted symbol=%s failures=%s reason=%s",
                    symbol,
                    updated.get("readiness_failures"),
                    updated.get("readiness_last_reason"),
                )
        if changed:
            self._persist_state()

    @staticmethod
    def _is_statistical_ineligibility(job: TrainingJob, reason: str) -> bool:
        """Classify evidence-backed RF/GBM target-support failures.

        These are model/timeframe statistical eligibility outcomes, not symbol or
        infrastructure failures. Treating them as symbol failures caused repeated
        backoff/blocking during the 24-hour forensic run.
        """
        if str(job.model_type).lower() not in {"rf", "gbm"}:
            return False
        text = str(reason or "").lower()
        return (
            "_label_support_invalid:" in text
            or "walk-forward validation produced no usable folds" in text
        )

    def _drain(self) -> None:
        for future, job, result, error_text in collect_completed_futures(self.inflight):
            self.inflight.pop(future, None)
            if error_text or result is None or not result.ok:
                reason = error_text or getattr(result, "error", "unknown")
                if self._is_statistical_ineligibility(job, str(reason)):
                    self.registry.mark_ineligible(job.model_type, job.symbol, job.interval, str(reason))
                    self.logger.warning(
                        "Training tuple statistically ineligible model=%s symbol=%s interval=%s reason=%s",
                        job.model_type,
                        job.symbol,
                        job.interval,
                        reason,
                    )
                    continue
                self.registry.mark_failed(job.model_type, job.symbol, job.interval, str(reason))
                self._record_symbol_failure(job, str(reason))
                continue
            self.registry.mark_success(model_type=result.model_type, symbol=result.symbol, interval=result.interval, metrics=result.metrics, artifacts=result.artifacts, rows_used=result.rows_used)
            self._record_symbol_success(result.symbol)

    def _apply_model_progress(self, job: TrainingJob, payload: dict[str, object], progress_groups: dict[str, list[float]]) -> None:
        label = f"{job.symbol} {job.interval}"
        msg = str(payload.get("message", "--"))
        pct = float(payload.get("percent", 0.0) or 0.0)
        model_type = str(job.model_type)
        progress_groups.setdefault(model_type, []).append(pct)
        if model_type == "rf":
            self.snapshot.current_rf = label
            self.snapshot.last_rf_message = msg
        elif model_type == "gbm":
            self.snapshot.current_gbm = label
            self.snapshot.last_gbm_message = msg
        elif model_type == "hmm":
            self.snapshot.current_hmm = label
            self.snapshot.last_hmm_message = msg
        else:
            self.snapshot.current_dqn = label
            self.snapshot.last_dqn_message = msg

    def _build_snapshot(self, due_jobs: list[TrainingJob], blocked_jobs: list[TrainingJob]) -> None:
        now_ts = time.time()
        self.snapshot.rf_pending = sum(1 for j in due_jobs if j.model_type == "rf")
        self.snapshot.gbm_pending = sum(1 for j in due_jobs if j.model_type == "gbm")
        self.snapshot.dqn_pending = sum(1 for j in due_jobs if j.model_type == "dqn")
        self.snapshot.hmm_pending = sum(1 for j in due_jobs if j.model_type == "hmm")
        self.snapshot.current_rf = self.snapshot.current_gbm = self.snapshot.current_dqn = self.snapshot.current_hmm = "--"
        self.snapshot.rf_progress = self.snapshot.gbm_progress = self.snapshot.dqn_progress = self.snapshot.hmm_progress = 0.0
        self.snapshot.last_rf_message = self.snapshot.last_gbm_message = self.snapshot.last_dqn_message = self.snapshot.last_hmm_message = "--"
        progress_groups: dict[str, list[float]] = {"rf": [], "gbm": [], "dqn": [], "hmm": []}
        for job in self.inflight.values():
            payload = self.engine.read_progress_payload(job.model_type, job.symbol, job.interval)
            self._apply_model_progress(job, payload, progress_groups)
        for model_type in progress_groups:
            if progress_groups[model_type]:
                setattr(self.snapshot, f"{model_type}_progress", sum(progress_groups[model_type]) / len(progress_groups[model_type]))
        symbol_rows = self._timed_out_symbols_payload(blocked_jobs, now_ts)
        blocked_count = sum(1 for r in symbol_rows if r["status"] == "BLOCKED")
        snapshot_dict = asdict(self.snapshot)
        try:
            backlog_diagnostics = build_training_backlog_diagnostics(self.paths, self.registry)
        except Exception as exc:
            self.logger.warning("Failed to build training backlog diagnostics: %s", exc)
            backlog_diagnostics = {"counts": {}, "rows": [], "total_rows": 0, "truncated": False, "error": str(exc)}

        snapshot_dict.update({
            "inflight_count": len(self.inflight),
            "timed_out_symbols": symbol_rows,
            "registry_backlog_diagnostics": backlog_diagnostics,
            "registry_backlog_counts": dict(backlog_diagnostics.get("counts", {})) if isinstance(backlog_diagnostics, dict) else {},
            "timed_out_symbol_count": len(symbol_rows),
            "blocked_symbol_count": blocked_count,
            "blocked_job_count": len(blocked_jobs),
            "retry_threshold": self.max_failures_before_block,
            "retry_backoff_first_sec": self.symbol_backoff_first_sec,
            "retry_backoff_repeat_sec": self.symbol_backoff_repeat_sec,
            "target_active_symbols": self.target_active_symbols,
            "ranked_universe_size": self.ranked_universe_size,
            "active_symbol_count": len(load_symbol_state(self.paths).get("active_symbols", [])),
            "auto_replace_symbols": self.auto_replace_symbols,
        })
        self.monitor.update_training_runtime(**snapshot_dict)

    def _process_symbol_actions(self) -> None:
        actions = load_symbol_actions(self.paths)
        if not actions:
            return
        symbols: set[str] = set(str(s).upper() for s in actions.get("symbols", []) if s)
        if actions.get("retry_all_blocked"):
            symbols.update(symbol for symbol, state in self._symbol_retry.items() if state.status == "BLOCKED")
        changed = False
        for symbol in symbols:
            state = self._get_symbol_state(symbol)
            self._symbol_retry[symbol] = SymbolRetryState.from_dict(request_symbol_retry_entry(state.to_dict()))
            changed = True
        if changed:
            self._persist_state()
        clear_symbol_actions(self.paths)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self._ensure_pools()
        self.logger.info("TechRevController started")
        while not stop_event.is_set():
            try:
                self._process_symbol_actions()
                self._ensure_managed_universe()
                self._drain()
                self._evaluate_active_symbol_readiness()
                now_ts = time.time()
                self._clear_expired_symbol_backoff(now_ts)
                raw_due_jobs = build_due_training_jobs(self.paths, self.registry)
                blocked_jobs: list[TrainingJob] = []
                due_jobs: list[TrainingJob] = []
                for job in raw_due_jobs:
                    if self._is_symbol_backed_off(job.symbol, now_ts):
                        blocked_jobs.append(job)
                    else:
                        due_jobs.append(job)
                active_keys = {(j.model_type, j.symbol, j.interval) for j in self.inflight.values()}
                rf_running = sum(1 for j in self.inflight.values() if j.model_type == "rf")
                gbm_running = sum(1 for j in self.inflight.values() if j.model_type == "gbm")
                dqn_running = sum(1 for j in self.inflight.values() if j.model_type == "dqn")
                hmm_running = sum(1 for j in self.inflight.values() if j.model_type == "hmm")
                queued_jobs: list[TrainingJob] = []
                skipped_by_backpressure = 0
                submitted_by_symbol: dict[str, int] = {}
                submitted_by_model: dict[str, int] = {"rf": 0, "gbm": 0, "dqn": 0, "hmm": 0}
                for job in due_jobs:
                    if len(queued_jobs) >= self.max_submissions_per_loop:
                        skipped_by_backpressure += 1
                        continue

                    key = (job.model_type, job.symbol, job.interval)
                    if key in active_keys:
                        continue

                    symbol_submissions = submitted_by_symbol.get(job.symbol, 0)
                    if symbol_submissions >= self.max_submissions_per_symbol_per_loop:
                        skipped_by_backpressure += 1
                        continue

                    model_cap = max(1, int(self.max_model_submissions_per_loop.get(job.model_type, 1)))
                    if submitted_by_model.get(job.model_type, 0) >= model_cap:
                        skipped_by_backpressure += 1
                        continue

                    if job.model_type == "rf":
                        if rf_running >= self.max_rf:
                            continue
                        self.engine.warm_fused_cache(job.symbol, job.interval)
                        self._submit_job(job)
                        rf_running += 1
                    elif job.model_type == "gbm":
                        if gbm_running >= self.max_gbm:
                            continue
                        self.engine.warm_fused_cache(job.symbol, job.interval)
                        self._submit_job(job)
                        gbm_running += 1
                    elif job.model_type == "hmm":
                        if hmm_running >= self.max_hmm:
                            continue
                        self.engine.warm_fused_cache(job.symbol, job.interval)
                        self._submit_job(job)
                        hmm_running += 1
                    else:
                        if dqn_running >= self.max_dqn:
                            continue
                        self._submit_job(job)
                        dqn_running += 1

                    queued_jobs.append(job)
                    submitted_by_symbol[job.symbol] = symbol_submissions + 1
                    submitted_by_model[job.model_type] = submitted_by_model.get(job.model_type, 0) + 1
                    active_keys.add(key)

                self._build_snapshot(due_jobs, blocked_jobs)
                self.monitor.update_scheduler(due_jobs=due_jobs, queued_jobs=queued_jobs)
                self.monitor.heartbeat(
                    "tech_rev",
                    status="running",
                    details={
                        "inflight": len(self.inflight),
                        "rf_running": rf_running,
                        "gbm_running": gbm_running,
                        "dqn_running": dqn_running,
                        "hmm_running": hmm_running,
                        "due_jobs": len(due_jobs),
                        "queued_jobs": len(queued_jobs),
                        "skipped_by_backpressure": skipped_by_backpressure,
                        "max_submissions_per_loop": self.max_submissions_per_loop,
                        "timed_out_symbols": len(self._timed_out_symbols_payload(blocked_jobs, time.time())),
                    },
                )
            except Exception as exc:
                self.logger.exception("TechRev loop failed")
                self.monitor.controller_error("tech_rev", str(exc))
            stop_event.wait(self.loop_sleep_sec)
