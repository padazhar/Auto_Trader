from __future__ import annotations

from typing import Any

from core.shared_utils import load_yaml
from inference.readiness import signal_eligible_symbols


def resolve_live_incumbent_requirements(paths) -> dict[str, Any]:
    raw_cfg = load_yaml(paths.tradable_symbols_path) if hasattr(paths, "tradable_symbols_path") else {}
    raw_cfg = raw_cfg or {}
    raw_symbols = raw_cfg.get("symbols", []) if isinstance(raw_cfg, dict) else []
    symbols = [str(item).upper() for item in raw_symbols if str(item).strip()]
    interval = str(
        getattr(getattr(paths, "config", None), "frames", None).anchor_interval
        if getattr(getattr(paths, "config", None), "frames", None)
        else "15m"
    )
    ready_symbols, reasons = signal_eligible_symbols(paths, symbols=symbols, interval=interval)
    active_incumbents = [{"symbol": symbol, "interval": interval, "reason": reasons.get(symbol, "ready")} for symbol in ready_symbols]
    return {
        "ready": bool(active_incumbents),
        "active_incumbents": active_incumbents,
        "reasons": reasons,
    }
