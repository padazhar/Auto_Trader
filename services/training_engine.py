from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from features.timeframe_fusion import build_fused_frame
from training.dqn_training_engine import train_dqn_model
from training.gbm_training_engine import train_gbm_model
from training.hmm_regime_training_engine import train_hmm_model
from core.model_naming import canonical_model_type, model_display_name
from training.rf_training_engine import train_rf_model


@dataclass(slots=True)
class TrainingProgressSnapshot:
    rf_pending: int = 0
    gbm_pending: int = 0
    dqn_pending: int = 0
    hmm_pending: int = 0
    current_rf: str = "--"
    current_gbm: str = "--"
    current_dqn: str = "--"
    current_hmm: str = "--"
    rf_progress: float = 0.0
    gbm_progress: float = 0.0
    dqn_progress: float = 0.0
    hmm_progress: float = 0.0
    last_rf_message: str = "--"
    last_gbm_message: str = "--"
    last_dqn_message: str = "--"
    last_hmm_message: str = "--"


class TrainingEngine:
    """Thin integration layer that keeps canonical trainers but exposes Tech_Rev-style progress."""

    def __init__(self, paths) -> None:
        self.paths = paths
        self.anchor_interval = paths.config.frames.anchor_interval

    def warm_fused_cache(self, symbol: str, interval: str) -> int:
        frame = build_fused_frame(self.paths, symbol, interval, tail_rows=self.paths.config.training.rf.tail_rows)
        return int(len(frame))

    def train_model(self, model_type: str, symbol: str, interval: str):
        model_type = canonical_model_type(model_type)
        if model_type == "rf":
            return train_rf_model(self.paths, symbol, interval)
        if model_type == "gbm":
            return train_gbm_model(self.paths, symbol, interval)
        if model_type == "dqn":
            return train_dqn_model(self.paths, symbol, interval)
        if model_type == "hmm":
            return train_hmm_model(self.paths, symbol, interval)
        raise ValueError(f"unsupported model_type: {model_type}")

    def read_progress_payload(self, model_type: str, symbol: str, interval: str) -> dict[str, Any]:
        path = self.paths.trainer_progress_path(model_type, symbol, interval)
        if not path.exists():
            return {}
        try:
            import json
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}


    def model_display_name(self, model_type: str) -> str:
        """Return the operator-facing display name for a model type."""
        return model_display_name(model_type)
