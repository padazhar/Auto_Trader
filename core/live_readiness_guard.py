from __future__ import annotations

from pathlib import Path
from typing import Any


class LiveReadinessError(Exception):
    pass


def _check_model_registry(paths) -> list[str]:
    errors = []

    model_dir = getattr(paths, "model_storage_dir", None)
    if not model_dir or not Path(model_dir).exists():
        errors.append("missing_model_storage_dir")
        return errors

    incumbents = list(Path(model_dir).rglob("incumbent.json"))
    if not incumbents:
        errors.append("missing_model_registry")

    return errors


def _check_anchor_models(paths) -> list[str]:
    errors = []

    anchor_tf = str(paths.config.frames.anchor_interval)
    model_dir = Path(paths.model_storage_dir)

    found = False
    for p in model_dir.rglob("incumbent.json"):
        if anchor_tf in str(p):
            found = True
            break

    if not found:
        errors.append("anchor_models_missing")

    return errors


def _check_rf_models(paths) -> list[str]:
    errors = []

    model_dir = Path(paths.model_storage_dir)

    rf_models = list(model_dir.rglob("*random_forest_models*incumbent.json"))

    if not rf_models:
        errors.append("rf_not_trained")

    return errors


def _check_symbol_coverage(paths) -> list[str]:
    errors = []

    try:
        from core.symbol_management import load_symbol_state

        state = load_symbol_state(paths)
        symbols = state.get("symbols", {})

        if not symbols:
            errors.append("no_symbols_loaded")
            return errors

        active = [s for s, v in symbols.items() if v.get("status") == "ACTIVE"]

        if len(active) < 2:
            errors.append("insufficient_active_symbols")

    except Exception:
        errors.append("symbol_state_unavailable")

    return errors


def validate_live_readiness(paths) -> None:
    """
    HARD GATE before live trading starts.
    Raises LiveReadinessError if any condition fails.
    """

    errors: list[str] = []

    errors += _check_model_registry(paths)
    errors += _check_anchor_models(paths)
    errors += _check_rf_models(paths)
    errors += _check_symbol_coverage(paths)

    if errors:
        raise LiveReadinessError(
            "LIVE READINESS FAILED:\n" + "\n".join(f"- {e}" for e in errors)
        )
