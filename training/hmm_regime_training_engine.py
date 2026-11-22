from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from core.artifact_contract import artifact_paths
from core.result_contracts import TrainerResult
from features.dataset_preparation import build_feature_set_metadata
from features.feature_store_contract import resolve_feature_recipe_version
from features.timeframe_fusion import build_fused_frame
from training.base_trainer import BaseTrainer

try:  # optional dependency declared by Step 7; imported lazily for safe startup
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
except Exception:  # pragma: no cover - exercised when dependency is absent
    _GaussianHMM = None


@dataclass(frozen=True)
class HMMObservationSchema:
    """Versioned HMM observation contract.

    The schema uses only contemporaneous/trailing OHLCV-derived fields. It does
    not use future returns or trade outcomes, so candidate/shadow HMM telemetry
    cannot leak labels into the active RF/GBM EV path.
    """

    version: str
    anchor_interval: str
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["columns"] = list(self.columns)
        return payload


_DEFAULT_OBSERVATION_COLUMNS = (
    "hmm_return_1",
    "hmm_return_3",
    "hmm_return_10",
    "hmm_range_fraction",
    "hmm_body_fraction",
    "hmm_volatility_10",
    "hmm_volume_z_20",
)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _first_existing(frame: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    for column in candidates:
        if column in frame.columns:
            return _coerce_numeric(frame[column])
    return pd.Series([float(default)] * len(frame), index=frame.index, dtype=float)


def build_hmm_observation_frame(frame: pd.DataFrame, *, anchor_interval: str, schema_version: str) -> tuple[pd.DataFrame, HMMObservationSchema]:
    """Build a leakage-safe HMM observation matrix from a fused feature frame."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_DEFAULT_OBSERVATION_COLUMNS)), HMMObservationSchema(schema_version, anchor_interval, _DEFAULT_OBSERVATION_COLUMNS)

    src = frame.copy()
    prefix = str(anchor_interval)
    close = _first_existing(src, [f"{prefix}_close", "close"])
    open_ = _first_existing(src, [f"{prefix}_open", "open"], default=float(close.iloc[0]) if len(close) else 0.0)
    high = _first_existing(src, [f"{prefix}_high", "high"], default=float(close.iloc[0]) if len(close) else 0.0)
    low = _first_existing(src, [f"{prefix}_low", "low"], default=float(close.iloc[0]) if len(close) else 0.0)
    volume = _first_existing(src, [f"{prefix}_volume", "volume"], default=0.0)

    safe_close = close.replace(0.0, np.nan).ffill().bfill()
    return_1 = safe_close.pct_change(1)
    return_3 = safe_close.pct_change(3)
    return_10 = safe_close.pct_change(10)
    range_fraction = (high - low) / safe_close.replace(0.0, np.nan)
    body_fraction = (close - open_).abs() / safe_close.replace(0.0, np.nan)
    volatility_10 = return_1.rolling(10, min_periods=2).std()
    vol_mean = volume.rolling(20, min_periods=2).mean()
    vol_std = volume.rolling(20, min_periods=2).std().replace(0.0, np.nan)
    volume_z_20 = (volume - vol_mean) / vol_std

    observations = pd.DataFrame(
        {
            "hmm_return_1": return_1,
            "hmm_return_3": return_3,
            "hmm_return_10": return_10,
            "hmm_range_fraction": range_fraction,
            "hmm_body_fraction": body_fraction,
            "hmm_volatility_10": volatility_10,
            "hmm_volume_z_20": volume_z_20,
        },
        index=src.index,
    )
    observations = observations.replace([np.inf, -np.inf], np.nan).dropna()
    observations = observations.astype(float)
    return observations, HMMObservationSchema(schema_version, anchor_interval, tuple(observations.columns))


def _state_summary(observations: pd.DataFrame, states: np.ndarray, posteriors: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if observations.empty or len(states) == 0:
        return out
    for state in sorted(set(int(s) for s in states.tolist())):
        mask = states == state
        subset = observations.loc[mask]
        if subset.empty:
            continue
        posterior_values = posteriors[:, state] if posteriors.ndim == 2 and state < posteriors.shape[1] else np.array([])
        if posterior_values.size:
            posterior_values = posterior_values[mask]
        out[str(state)] = {
            "rows": int(mask.sum()),
            "mean_return_1": BaseTrainer.safe_float(subset["hmm_return_1"].mean()),
            "mean_volatility_10": BaseTrainer.safe_float(subset["hmm_volatility_10"].mean()),
            "mean_range_fraction": BaseTrainer.safe_float(subset["hmm_range_fraction"].mean()),
            "mean_posterior": BaseTrainer.safe_float(float(np.mean(posterior_values))) if posterior_values.size else None,
        }
    return out


class HMMRegimeTrainer(BaseTrainer):
    """Candidate-only Gaussian HMM regime confirmer trainer.

    Step 10 contract: HMM artefacts are trained for shadow telemetry only. They
    are never promoted into RF/GBM EV fusion, never size/risk trades, and never
    gain live influence in this plan.
    """

    model_type = "hmm"

    def train(self, symbol: str, interval: str) -> TrainerResult:
        try:
            cfg = self.paths.config
            hmm_cfg = cfg.training.hmm
            if not bool(hmm_cfg.enabled and hmm_cfg.candidate_training_enabled):
                return self.fail_result(symbol=symbol, interval=interval, error="hmm_candidate_training_disabled")
            if bool(hmm_cfg.mandatory_paper_gate_enabled) or bool(hmm_cfg.live_influence_enabled):
                return self.fail_result(symbol=symbol, interval=interval, error="hmm_training_gate_or_live_influence_forbidden_in_step10")
            if _GaussianHMM is None:
                return self.fail_result(symbol=symbol, interval=interval, error="hmmlearn_missing")

            self.progress(symbol, interval, 10, "loading_hmm_observation_frame")
            raw = build_fused_frame(self.paths, symbol, interval, tail_rows=int(hmm_cfg.tail_rows))
            observations, schema = build_hmm_observation_frame(
                raw,
                anchor_interval=str(interval),
                schema_version=str(hmm_cfg.observation_schema_version),
            )
            if len(observations) < int(hmm_cfg.min_rows):
                return self.fail_result(
                    symbol=symbol,
                    interval=interval,
                    rows_used=int(len(observations)),
                    error=f"insufficient_hmm_rows:{len(observations)}/{int(hmm_cfg.min_rows)}",
                )

            candidate_id = self.new_candidate_id()
            n_states = int(hmm_cfg.n_states)
            self.progress(symbol, interval, 45, "training_hmm_regime_candidate", rows=len(observations), n_states=n_states)
            model = _GaussianHMM(
                n_components=n_states,
                covariance_type=str(hmm_cfg.covariance_type),
                n_iter=int(hmm_cfg.max_iter),
                random_state=int(hmm_cfg.random_state),
            )
            X = observations.to_numpy(dtype=float)
            model.fit(X)
            try:
                posteriors = np.asarray(model.predict_proba(X), dtype=float)
            except Exception:
                posteriors = np.zeros((len(observations), n_states), dtype=float)
            try:
                states = np.asarray(model.predict(X), dtype=int)
            except Exception:
                states = np.argmax(posteriors, axis=1) if posteriors.size else np.zeros(len(observations), dtype=int)
            max_posterior = np.max(posteriors, axis=1) if posteriors.size else np.zeros(len(observations), dtype=float)
            try:
                log_likelihood = BaseTrainer.safe_float(float(model.score(X)))
            except Exception:
                log_likelihood = None
            converged = bool(getattr(getattr(model, "monitor_", None), "converged", False))
            state_counts = {str(int(state)): int((states == state).sum()) for state in sorted(set(states.tolist()))}
            metrics: dict[str, Any] = {
                "model_operational_role": "hmm_regime_confirmer_candidate_shadow",
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_paper_gate_enabled": False,
                "live_influence_enabled": False,
                "rows_total": int(len(raw)),
                "rows_used": int(len(observations)),
                "observation_schema_version": schema.version,
                "observation_columns": list(schema.columns),
                "n_states": n_states,
                "covariance_type": str(hmm_cfg.covariance_type),
                "converged": converged,
                "log_likelihood": log_likelihood,
                "avg_max_posterior": BaseTrainer.safe_float(float(np.mean(max_posterior))) if len(max_posterior) else None,
                "min_posterior_confidence": BaseTrainer.safe_float(float(hmm_cfg.min_posterior_confidence)),
                "state_counts": state_counts,
                "state_summary": _state_summary(observations, states, posteriors),
                "validation_policy": "unsupervised_gaussian_hmm_candidate_shadow",
            }

            artifacts = artifact_paths(self.paths, "hmm", symbol, interval, stage="candidate", candidate_id=candidate_id)
            artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, artifacts["model"])
            recipe_version = resolve_feature_recipe_version(getattr(self.paths, "config", None))
            feature_set = build_feature_set_metadata(
                list(schema.columns),
                recipe_version=recipe_version,
                anchor_interval=str(interval),
                optional_context_fields=[],
            )
            bundle = self.save_training_bundle(
                symbol=symbol,
                interval=interval,
                rows_used=int(len(observations)),
                feature_columns=list(schema.columns),
                artifacts=artifacts,
                metrics=metrics,
                candidate_id=candidate_id,
                model_role="regime_confirmer",
                promote=False,
                extra_manifest={
                    "feature_set": feature_set,
                    "validation": {
                        "policy": "unsupervised_gaussian_hmm_candidate_shadow",
                        "leakage_safe_observations": True,
                        "decision_effect": "none",
                    },
                    "training": {
                        "observation_schema": schema.to_dict(),
                        "candidate_training_enabled": True,
                        "shadow_only": True,
                        "mandatory_paper_gate_enabled": False,
                        "live_influence_enabled": False,
                    },
                    "hmm_contract": {
                        "schema_version": schema.version,
                        "n_states": n_states,
                        "covariance_type": str(hmm_cfg.covariance_type),
                        "min_posterior_confidence": BaseTrainer.safe_float(float(hmm_cfg.min_posterior_confidence)),
                        "shadow_only": True,
                        "decision_influence_allowed": False,
                        "mandatory_paper_gate_enabled": False,
                        "live_influence_enabled": False,
                    },
                },
            )
            self.progress(symbol, interval, 100, "hmm_candidate_shadow_ready", candidate_id=candidate_id)
            return self.ok_result(
                symbol=symbol,
                interval=interval,
                rows_used=int(len(observations)),
                artifacts=artifacts,
                metrics={**metrics, "candidate_id": candidate_id, "promotion": bundle.get("promotion", {})},
                model_role="regime_confirmer",
            )
        except Exception as exc:
            self.logger.exception("HMM regime candidate training failed symbol=%s interval=%s", symbol, interval)
            return self.fail_result(symbol=symbol, interval=interval, error=str(exc))


def train_hmm_model(paths, symbol: str, interval: str) -> TrainerResult:
    """Train a candidate-only HMM regime confirmer for shadow telemetry."""
    return HMMRegimeTrainer(paths).train(symbol, interval)


# Backward-compatible descriptive aliases for future callers.
HMMRegimeTrainerCandidate = HMMRegimeTrainer
