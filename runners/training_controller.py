from __future__ import annotations

import concurrent.futures
import json
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import Dict, Set, Tuple

from core.config_contract import get_scheduling_config
from core.job_definition import TrainingJob
from core.log_manager import get_logger
from monitoring.publisher import MonitorPublisher
from scheduling.future_collector import collect_completed_futures
from scheduling.job_dispatcher import dispatch_training_job
from scheduling.priority_manager import build_due_training_jobs


@dataclass
class _FailureRecord:
    ts: float
    reason: str


class TrainingController:
    def __init__(self, paths, registry) -> None:
        self.paths = paths
        self.registry = registry
        self.logger = get_logger("runners.training_controller", paths)
        settings = paths.load_yaml(paths.config_dir / "system_settings.yaml")
        cfg = get_scheduling_config(settings)
        self.loop_sleep_sec = cfg.loop_sleep_sec
        self.max_workers_rf = cfg.max_rf_workers
        self.max_workers_gbm = cfg.max_gbm_workers
        self.max_workers_dqn = cfg.max_dqn_workers
        self.allow_parallel_symbols = cfg.allow_parallel_symbols
        self.failure_cooldown_sec = cfg.failure_cooldown_sec

        self.rf_executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.gbm_executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.dqn_executor: concurrent.futures.ProcessPoolExecutor | None = None
        self.inflight: Dict[concurrent.futures.Future, TrainingJob] = {}
        self.active_keys: Set[Tuple[str, str, str]] = set()
        self.failure_cache: Dict[Tuple[str, str, str], _FailureRecord] = {}
        self.monitor = MonitorPublisher(paths)
        self.completed_success: list[dict] = []
        self.completed_failed: list[dict] = []

    def _ensure_executors(self) -> None:
        if self.rf_executor is not None and self.gbm_executor is not None and self.dqn_executor is not None:
            return
        spawn_ctx = mp.get_context("spawn")
        self.rf_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_rf, mp_context=spawn_ctx)
        self.gbm_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_gbm, mp_context=spawn_ctx)
        self.dqn_executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers_dqn, mp_context=spawn_ctx)
        self.logger.info(
            "Created executors rf_workers=%s gbm_workers=%s dqn_workers=%s start_method=spawn",
            self.max_workers_rf,
            self.max_workers_gbm,
            self.max_workers_dqn,
        )

    def _job_key(self, job: TrainingJob) -> Tuple[str, str, str]:
        return (job.model_type, job.symbol, job.interval)

    def _progress_payload(self, model_type: str, symbol: str, interval: str) -> dict:
        path = self.paths.trainer_progress_path(model_type, symbol, interval)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _recover_stale_queued_entries(self) -> None:
        active = set(self.active_keys)
        for entry in self.registry.all_entries():
            key = (str(entry.model_type), str(entry.symbol), str(entry.interval))
            if entry.status in {"queued", "running"} and key not in active:
                self.registry.mark_idle(*key)

    def _publish_training_state(self, due_jobs: list[TrainingJob], queued_jobs: list[TrainingJob]) -> None:
        rf_pending = sum(1 for j in due_jobs if j.model_type == "rf")
        gbm_pending = sum(1 for j in due_jobs if j.model_type == "gbm")
        dqn_pending = sum(1 for j in due_jobs if j.model_type == "dqn")
        running_jobs = [{"model_type": j.model_type, "symbol": j.symbol, "interval": j.interval} for j in self.inflight.values()]
        rf_running = sum(1 for j in self.inflight.values() if j.model_type == "rf")
        gbm_running = sum(1 for j in self.inflight.values() if j.model_type == "gbm")
        dqn_running = sum(1 for j in self.inflight.values() if j.model_type == "dqn")

        last_rf = last_gbm = last_dqn = "--"
        rf_progress = gbm_progress = dqn_progress = 0.0
        current_rf = current_gbm = current_dqn = "--"
        for job in self.inflight.values():
            payload = self._progress_payload(job.model_type, job.symbol, job.interval)
            message = str(payload.get("message", "--"))
            percent = float(payload.get("percent", 0.0) or 0.0)
            label = f"{job.symbol} {job.interval}"
            if job.model_type == "rf":
                current_rf, last_rf, rf_progress = label, message, percent
            elif job.model_type == "gbm":
                current_gbm, last_gbm, gbm_progress = label, message, percent
            elif job.model_type == "dqn":
                current_dqn, last_dqn, dqn_progress = label, message, percent

        queue_depth = max(0, len(due_jobs) - len(queued_jobs) - len(self.inflight))
        self.monitor.heartbeat("training", status="running", details={"rf_running": rf_running, "gbm_running": gbm_running, "dqn_running": dqn_running})
        self.monitor.update_scheduler(due_jobs=due_jobs, queued_jobs=queued_jobs)
        self.monitor.update_training_runtime(
            queue_depth=queue_depth,
            inflight_count=len(self.inflight),
            rf_running=rf_running,
            gbm_running=gbm_running,
            dqn_running=dqn_running,
            due_jobs_preview=[{"model_type": j.model_type, "symbol": j.symbol, "interval": j.interval, "reason": getattr(j, "reason", "")} for j in due_jobs[:25]],
            queued_jobs=[{"model_type": j.model_type, "symbol": j.symbol, "interval": j.interval, "reason": getattr(j, "reason", "")} for j in queued_jobs[:25]],
            running_jobs=running_jobs[:25],
            succeeded_jobs=self.completed_success[:25],
            failed_jobs=self.completed_failed[:25],
            failure_cache_size=len(self.failure_cache),
            rf_pending=rf_pending,
            gbm_pending=gbm_pending,
            dqn_pending=dqn_pending,
            current_rf=current_rf,
            current_gbm=current_gbm,
            current_dqn=current_dqn,
            rf_progress=rf_progress,
            gbm_progress=gbm_progress,
            dqn_progress=dqn_progress,
            last_rf_message=last_rf,
            last_gbm_message=last_gbm,
            last_dqn_message=last_dqn,
        )

    def _can_submit(self, job: TrainingJob) -> bool:
        if not self.allow_parallel_symbols:
            active_symbols = {j.symbol for j in self.inflight.values()}
            if job.symbol in active_symbols:
                return False
        key = self._job_key(job)
        failure = self.failure_cache.get(key)
        if failure and (time.time() - failure.ts) < self.failure_cooldown_sec:
            return False
        rf_running = sum(1 for j in self.inflight.values() if j.model_type == "rf")
        gbm_running = sum(1 for j in self.inflight.values() if j.model_type == "gbm")
        dqn_running = sum(1 for j in self.inflight.values() if j.model_type == "dqn")
        if job.model_type == "rf":
            return rf_running < self.max_workers_rf
        if job.model_type == "gbm":
            return gbm_running < self.max_workers_gbm
        if job.model_type == "dqn":
            return dqn_running < self.max_workers_dqn
        return False

    def _submit_due_jobs(self, due_jobs: list[TrainingJob]) -> list[TrainingJob]:
        queued_jobs: list[TrainingJob] = []
        for job in due_jobs:
            if not self._can_submit(job):
                continue
            key = self._job_key(job)
            if key in self.active_keys:
                continue
            self.logger.info("Attempting submit model=%s symbol=%s interval=%s reason=%s", job.model_type, job.symbol, job.interval, getattr(job, "reason", ""))
            future = dispatch_training_job(self.rf_executor, self.gbm_executor, self.dqn_executor, str(self.paths.project_root), job)
            self.inflight[future] = job
            self.active_keys.add(key)
            self.registry.mark_queued(*key)
            self.registry.mark_running(*key)
            queued_jobs.append(job)
        return queued_jobs

    def _drain_completed(self) -> None:
        for future, job, result, error_text in collect_completed_futures(self.inflight):
            self.inflight.pop(future, None)
            key = self._job_key(job)
            self.active_keys.discard(key)
            if error_text or result is None or not result.ok:
                reason = error_text or getattr(result, "error", "unknown_failure")
                self.logger.error("Training job failed model=%s symbol=%s interval=%s reason=%s", job.model_type, job.symbol, job.interval, reason)
                self.failure_cache[key] = _FailureRecord(ts=time.time(), reason=str(reason))
                self.registry.mark_failed(job.model_type, job.symbol, job.interval, str(reason))
                self.completed_failed.insert(0, {"model_type": job.model_type, "symbol": job.symbol, "interval": job.interval, "error": str(reason), "updated_at": time.time()})
                self.completed_failed = self.completed_failed[:25]
                continue
            self.logger.info("Training job succeeded model=%s symbol=%s interval=%s rows_used=%s", result.model_type, result.symbol, result.interval, result.rows_used)
            self.failure_cache.pop(key, None)
            self.registry.mark_success(model_type=result.model_type, symbol=result.symbol, interval=result.interval, metrics=result.metrics, artifacts=result.artifacts, rows_used=result.rows_used)
            self.completed_success.insert(0, {"model_type": result.model_type, "symbol": result.symbol, "interval": result.interval, "rows_used": result.rows_used, "updated_at": time.time()})
            self.completed_success = self.completed_success[:25]

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.logger.info("TrainingController started")
        self._ensure_executors()
        try:
            while not stop_event.is_set():
                self._recover_stale_queued_entries()
                self._drain_completed()
                due_jobs = build_due_training_jobs(self.paths, self.registry)
                queued_jobs = self._submit_due_jobs(due_jobs)
                self._publish_training_state(due_jobs, queued_jobs)
                stop_event.wait(self.loop_sleep_sec)
        except KeyboardInterrupt:
            self.logger.warning("TrainingController interrupted")
        except Exception as exc:
            self.logger.exception("TrainingController crashed")
            self.monitor.controller_error("training", str(exc))
            raise
        finally:
            self.monitor.flush()
            for executor in [self.rf_executor, self.gbm_executor, self.dqn_executor]:
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)
