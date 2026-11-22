from __future__ import annotations

from typing import Final

INTERNAL_POLICY_COMPAT_KEY: Final[str] = "dqn"
HMM_REGIME_MODEL_TYPE: Final[str] = "hmm"
CANONICAL_MODEL_TYPES: Final[tuple[str, ...]] = ("rf", "gbm", INTERNAL_POLICY_COMPAT_KEY, HMM_REGIME_MODEL_TYPE)
_MODEL_TYPE_ALIASES: Final[dict[str, str]] = {
    "rf": "rf",
    "random_forest": "rf",
    "random_forest_classifier": "rf",
    "gbm": "gbm",
    "gradient_boost": "gbm",
    "gradient_boosting": "gbm",
    "xgboost": "gbm",
    "lightgbm": "gbm",
    "lgbm": "gbm",
    HMM_REGIME_MODEL_TYPE: HMM_REGIME_MODEL_TYPE,
    "hmm_regime": HMM_REGIME_MODEL_TYPE,
    "hmm_regime_confirmer": HMM_REGIME_MODEL_TYPE,
    "regime_hmm": HMM_REGIME_MODEL_TYPE,
    INTERNAL_POLICY_COMPAT_KEY: INTERNAL_POLICY_COMPAT_KEY,
    "policy": INTERNAL_POLICY_COMPAT_KEY,
    "policy_model": INTERNAL_POLICY_COMPAT_KEY,
    "policy_models": INTERNAL_POLICY_COMPAT_KEY,
    "policy_classifier": INTERNAL_POLICY_COMPAT_KEY,
    "policy_confirmer": INTERNAL_POLICY_COMPAT_KEY,
    "confirmatory_policy": INTERNAL_POLICY_COMPAT_KEY,
    "confirmatory_policy_classifier": INTERNAL_POLICY_COMPAT_KEY,
    "placeholder_policy": INTERNAL_POLICY_COMPAT_KEY,
    "reinforcement_agent": INTERNAL_POLICY_COMPAT_KEY,
}

MODEL_DISPLAY_NAMES: Final[dict[str, str]] = {
    "rf": "RF",
    "gbm": "GBM",
    HMM_REGIME_MODEL_TYPE: "HMM Regime Confirmer",
    INTERNAL_POLICY_COMPAT_KEY: "Confirmatory Policy",
}

MODEL_SHORT_LABELS: Final[dict[str, str]] = {
    "rf": "R",
    "gbm": "G",
    HMM_REGIME_MODEL_TYPE: "H",
    INTERNAL_POLICY_COMPAT_KEY: "P",
}

MODEL_STORAGE_DIRECTORIES: Final[dict[str, str]] = {
    "rf": "random_forest_models",
    "gbm": "gradient_boost_models",
    HMM_REGIME_MODEL_TYPE: "hmm_regime_models",
    # Keep the historical storage directory for backward compatibility.
    INTERNAL_POLICY_COMPAT_KEY: "reinforcement_agents",
}


def canonical_model_type(model_type: str) -> str:
    """Normalize external model names to canonical internal identifiers.

    The repository keeps ``dqn`` as the internal compatibility key for the old
    policy-confirmation slot so existing storage paths, manifests, and callers
    continue to work. Operator-facing surfaces should describe that slot as a
    confirmatory policy model rather than implying true reinforcement learning.
    """
    normalized = str(model_type or "").strip().lower()
    if not normalized:
        raise ValueError("model_type is required")
    return _MODEL_TYPE_ALIASES.get(normalized, normalized)


def model_display_name(model_type: str) -> str:
    """Return the operator-facing display name for a model type."""
    canonical = canonical_model_type(model_type)
    return MODEL_DISPLAY_NAMES.get(canonical, canonical.upper())


def model_short_label(model_type: str) -> str:
    """Return the compact operator-facing label for a model type."""
    canonical = canonical_model_type(model_type)
    return MODEL_SHORT_LABELS.get(canonical, canonical[:1].upper())


def model_storage_directory(model_type: str) -> str:
    """Return the on-disk storage directory for a model type."""
    canonical = canonical_model_type(model_type)
    return MODEL_STORAGE_DIRECTORIES[canonical]


def is_policy_model(model_type: str) -> bool:
    """Return True when the model type refers to the confirmatory policy slot."""
    return canonical_model_type(model_type) == INTERNAL_POLICY_COMPAT_KEY
