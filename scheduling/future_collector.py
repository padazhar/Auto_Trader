from __future__ import annotations

import concurrent.futures
from typing import Dict, List, Tuple

from core.job_definition import TrainingJob
from core.result_contracts import TrainerResult, coerce_result, validate_success_result


CollectedFuture = tuple[
    concurrent.futures.Future,
    TrainingJob,
    TrainerResult | None,
    str | None,
]


def _read_future_result(
    future: concurrent.futures.Future,
    job: TrainingJob,
) -> CollectedFuture:
    try:
        raw = future.result()
    except Exception as exc:
        return future, job, None, str(exc)

    try:
        result = coerce_result(raw)
    except Exception as exc:
        return future, job, None, f"invalid trainer result type: {exc}"

    if result.model_type != job.model_type:
        return future, job, None, (
            f"trainer returned mismatched model_type: "
            f"expected={job.model_type} got={result.model_type}"
        )

    if result.symbol != job.symbol:
        return future, job, None, (
            f"trainer returned mismatched symbol: "
            f"expected={job.symbol} got={result.symbol}"
        )

    if result.interval != job.interval:
        return future, job, None, (
            f"trainer returned mismatched interval: "
            f"expected={job.interval} got={result.interval}"
        )

    if result.ok:
        ok, validation_error = validate_success_result(result)
        if not ok:
            return future, job, None, validation_error

    return future, job, result, None


def collect_completed_futures(
    inflight: Dict[concurrent.futures.Future, TrainingJob],
) -> List[CollectedFuture]:
    """
    Drain only futures that are already finished.

    Returns:
        [
            (future, job, result_or_none, error_text_or_none),
            ...
        ]

    Rules:
    - unfinished futures are ignored
    - crashed workers become (None result, error_text)
    - malformed trainer results become (None result, error_text)
    - successful results are artifact-validated through result_contracts
    """
    completed: List[CollectedFuture] = []

    if not inflight:
        return completed

    for future, job in list(inflight.items()):
        if not future.done():
            continue
        completed.append(_read_future_result(future, job))

    return completed
