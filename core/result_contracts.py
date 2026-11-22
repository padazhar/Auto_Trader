from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.model_naming import canonical_model_type


@dataclass(slots=True)
class TrainerResult:
    ok: bool
    model_type: str
    symbol: str
    interval: str
    rows_used: int
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    model_role: str | None = None
    timestamps: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_type"] = canonical_model_type(self.model_type)
        return payload


REQUIRED_ARTIFACT_KEYS: dict[str, tuple[str, ...]] = {
    "rf": ("model", "feature_columns", "metrics", "manifest"),
    "dqn": ("model", "feature_columns", "metrics", "manifest"),
    "hmm": ("model", "feature_columns", "metrics", "manifest"),
}


ROLE_REQUIRED_ARTIFACT_KEYS: dict[tuple[str, str], tuple[str, ...]] = {
    ("gbm", "direction"): ("model", "feature_columns", "metrics", "manifest"),
    ("gbm", "meta"): ("model", "feature_columns", "metrics", "manifest"),
}


def _normalize_artifacts(artifacts: Mapping[str, Any] | None) -> dict[str, str]:
    if not artifacts:
        return {}
    return {str(k): str(v) for k, v in artifacts.items() if v is not None}


def validate_result_shape(result: TrainerResult) -> tuple[bool, str | None]:
    result.model_type = canonical_model_type(result.model_type)
    if not result.model_type:
        return False, "model_type is required"
    if not result.symbol:
        return False, "symbol is required"
    if not result.interval:
        return False, "interval is required"
    if result.rows_used < 0:
        return False, "rows_used must be non-negative"
    if result.ok and result.error:
        return False, "successful result cannot also contain an error"
    return True, None


def validate_artifacts_exist(
    model_type: str,
    artifacts: Mapping[str, Any] | None,
    *,
    model_role: str | None = None,
) -> tuple[bool, str | None]:
    model_type = canonical_model_type(model_type)
    normalized = _normalize_artifacts(artifacts)
    required = ROLE_REQUIRED_ARTIFACT_KEYS.get((model_type, str(model_role or "")), REQUIRED_ARTIFACT_KEYS.get(model_type, ()))
    for key in required:
        raw = normalized.get(key)
        if not raw:
            return False, f"missing artifact key: {key}"
        if not Path(raw).exists():
            return False, f"artifact does not exist: {raw}"
    return True, None


def validate_success_result(result: TrainerResult) -> tuple[bool, str | None]:
    ok, error = validate_result_shape(result)
    if not ok:
        return False, error
    if not result.ok:
        return False, "result is not marked successful"
    return validate_artifacts_exist(result.model_type, result.artifacts, model_role=result.model_role)


def build_success_result(
    *,
    model_type: str,
    symbol: str,
    interval: str,
    rows_used: int,
    artifacts: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    model_role: str | None = None,
    timestamps: Mapping[str, Any] | None = None,
) -> TrainerResult:
    result = TrainerResult(
        ok=True,
        model_type=canonical_model_type(model_type),
        symbol=symbol,
        interval=interval,
        rows_used=int(rows_used),
        artifacts=_normalize_artifacts(artifacts),
        metrics=dict(metrics or {}),
        error=None,
        model_role=model_role,
        timestamps=dict(timestamps or {}),
    )
    ok, error = validate_success_result(result)
    if not ok:
        result.ok = False
        result.error = error
    return result


def build_failure_result(
    *,
    model_type: str,
    symbol: str,
    interval: str,
    error: str,
    rows_used: int = 0,
    artifacts: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
    model_role: str | None = None,
) -> TrainerResult:
    return TrainerResult(
        ok=False,
        model_type=canonical_model_type(model_type),
        symbol=symbol,
        interval=interval,
        rows_used=int(rows_used),
        artifacts=_normalize_artifacts(artifacts),
        metrics=dict(metrics or {}),
        error=str(error),
        model_role=model_role,
    )


def coerce_result(value: Any) -> TrainerResult:
    if isinstance(value, TrainerResult):
        return value
    if isinstance(value, dict):
        return TrainerResult(
            ok=bool(value.get("ok", False)),
            model_type=canonical_model_type(str(value.get("model_type", ""))),
            symbol=str(value.get("symbol", "")),
            interval=str(value.get("interval", "")),
            rows_used=int(value.get("rows_used", 0)),
            artifacts=_normalize_artifacts(value.get("artifacts")),
            metrics=dict(value.get("metrics") or {}),
            error=value.get("error"),
            model_role=value.get("model_role"),
            timestamps=dict(value.get("timestamps") or {}),
        )
    raise TypeError(f"cannot coerce value into TrainerResult: {type(value)!r}")
