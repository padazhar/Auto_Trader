from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

from core.job_definition import TrainingJob


def _run_job(project_root: str, job_payload: dict):
    project_path = Path(project_root).resolve()
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

    from core.log_manager import get_logger
    from core.path_manager import PathManager
    from core.result_contracts import build_failure_result, coerce_result
    from training.dqn_training_engine import train_dqn_model
    from training.gbm_training_engine import train_gbm_model
    from training.hmm_regime_training_engine import train_hmm_model
    from training.rf_training_engine import train_rf_model

    trainer_map = {
        "rf": train_rf_model,
        "gbm": train_gbm_model,
        "dqn": train_dqn_model,
        "hmm": train_hmm_model,
    }

    model_type = str(job_payload["model_type"])
    symbol = str(job_payload["symbol"])
    interval = str(job_payload["interval"])

    trainer = trainer_map.get(model_type)
    if trainer is None:
        return build_failure_result(model_type=model_type, symbol=symbol, interval=interval, error=f"unsupported model_type: {model_type}")

    try:
        paths = PathManager.from_project_root(project_path)
        logger = get_logger(f"training.{model_type}", paths)
        logger.info("worker job start model=%s symbol=%s interval=%s", model_type, symbol, interval)
        result = coerce_result(trainer(paths, symbol, interval))
        logger.info(
            "worker job end model=%s symbol=%s interval=%s ok=%s rows_used=%s error=%s",
            model_type,
            symbol,
            interval,
            result.ok,
            result.rows_used,
            result.error,
        )
        return result
    except Exception as exc:
        try:
            paths = PathManager.from_project_root(project_path)
            logger = get_logger(f"training.{model_type}", paths)
            logger.exception("worker job crash model=%s symbol=%s interval=%s", model_type, symbol, interval)
        except Exception:
            pass
        return build_failure_result(model_type=model_type, symbol=symbol, interval=interval, error=str(exc))


def dispatch_training_job(
    rf_executor: concurrent.futures.Executor,
    gbm_executor: concurrent.futures.Executor,
    dqn_executor: concurrent.futures.Executor,
    project_root: str,
    job: TrainingJob,
) -> concurrent.futures.Future:
    payload = {"model_type": job.model_type, "symbol": job.symbol, "interval": job.interval}
    if job.model_type == "rf":
        return rf_executor.submit(_run_job, project_root, payload)
    if job.model_type == "gbm":
        return gbm_executor.submit(_run_job, project_root, payload)
    return dqn_executor.submit(_run_job, project_root, payload)
