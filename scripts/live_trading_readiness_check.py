from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.path_manager import PathManager
from execution.circuit_breaker import CircuitBreakerStore
from scripts.live_candidate_operational_check import run_live_candidate_operational_check
from scripts.repository_readiness_check import run_readiness_check


@dataclass(slots=True)
class LiveTradingReadinessItem:
    """Single live-trading readiness item."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LiveTradingReadinessReport:
    """Structured final live-trading readiness report."""

    project_root: str
    overall_status: str
    live_trading_ready: bool
    items: list[LiveTradingReadinessItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "live_trading_ready": self.live_trading_ready,
            "items": [asdict(item) for item in self.items],
        }


def _normalize_root(paths_or_root: Any) -> Path:
    if isinstance(paths_or_root, PathManager):
        return Path(paths_or_root.project_root).resolve()
    if isinstance(paths_or_root, (str, Path)):
        return Path(paths_or_root).resolve()
    project_root = getattr(paths_or_root, "project_root", None)
    if project_root is not None:
        return Path(project_root).resolve()
    raise TypeError("Expected PathManager-like object or project root path")


def _record(items: list[LiveTradingReadinessItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(LiveTradingReadinessItem(name=name, status=status, message=message, details=dict(details)))


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _risk_and_circuit_contract(paths: PathManager) -> tuple[bool, dict[str, Any]]:
    risk_cfg = getattr(getattr(paths, "config", None), "risk", None)
    details = {
        "daily_loss_limit_usd": float(getattr(risk_cfg, "daily_loss_limit_usd", 0.0) or 0.0),
        "max_drawdown_fraction": float(getattr(risk_cfg, "max_drawdown_fraction", 0.0) or 0.0),
        "max_total_exposure_usd": float(getattr(risk_cfg, "max_total_exposure_usd", 0.0) or 0.0),
        "max_concurrent_positions": int(getattr(risk_cfg, "max_concurrent_positions", 0) or 0),
        "max_symbol_notional_usd": float(getattr(risk_cfg, "max_symbol_notional_usd", 0.0) or 0.0),
        "risk_per_trade": float(getattr(risk_cfg, "risk_per_trade", 0.0) or 0.0),
    }
    limits_ok = all((
        details["daily_loss_limit_usd"] > 0.0,
        0.0 < details["max_drawdown_fraction"] < 1.0,
        details["max_total_exposure_usd"] > 0.0,
        details["max_concurrent_positions"] > 0,
        details["max_symbol_notional_usd"] > 0.0,
        0.0 < details["risk_per_trade"] < 1.0,
    ))
    try:
        circuit = CircuitBreakerStore(paths.circuit_breaker_state_path).load()
        details["circuit_breaker_store_readable"] = True
        details["circuit_breaker_triggered"] = bool(circuit.triggered)
        details["circuit_breaker_reason"] = str(circuit.reason or "")
    except Exception as exc:  # pragma: no cover - defensive readiness reporting
        details["circuit_breaker_store_readable"] = False
        details["circuit_breaker_error"] = str(exc)

    return bool(limits_ok and details.get("circuit_breaker_store_readable", False)), details


def run_live_trading_readiness_check(paths_or_root) -> LiveTradingReadinessReport:
    root = _normalize_root(paths_or_root)
    paths = PathManager.from_project_root(root)
    items: list[LiveTradingReadinessItem] = []

    readiness = run_readiness_check(root).to_dict()
    candidate = run_live_candidate_operational_check(root).to_dict()
    enable_live = _truthy_env("AUTO_TRADER_ENABLE_LIVE_MODE")
    enable_live_canary = _truthy_env("AUTO_TRADER_ENABLE_LIVE_CANARY")

    repo_default_safe = bool(readiness.get("paper_trading_ready", False)) and not bool(readiness.get("live_candidate_ready", False))
    _record(
        items,
        "repo_default_safety",
        "pass" if repo_default_safe else "fail",
        "Repository defaults remain paper-safe and not live-candidate ready by default." if repo_default_safe else "Repository default safety contract is not satisfied.",
        readiness=readiness,
    )

    paper_gate = bool(readiness.get("paper_trading_ready", False))
    _record(
        items,
        "paper_trading_gate",
        "pass" if paper_gate else "fail",
        "Paper-trading readiness gate passed." if paper_gate else "Paper-trading readiness gate failed.",
    )

    live_guard_gate = bool(candidate.get("live_candidate_ready", False))
    _record(
        items,
        "live_guard_gate",
        "pass" if live_guard_gate else "fail",
        "Live guard contract is present." if live_guard_gate else "Live guard contract failed.",
        candidate=candidate,
    )

    live_candidate_operational_gate = bool(candidate.get("live_candidate_ready", False))
    _record(
        items,
        "live_candidate_operational_gate",
        "pass" if live_candidate_operational_gate else "fail",
        "Live-candidate operational gate passed." if live_candidate_operational_gate else "Live-candidate operational gate failed.",
    )

    deliberate_live_enablement = enable_live
    _record(
        items,
        "deliberate_live_enablement",
        "pass" if deliberate_live_enablement else "fail",
        "Explicit live enablement is present." if deliberate_live_enablement else "Explicit live enablement is absent; final live gate remains fail-closed.",
    )

    runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
    exchange_cfg = getattr(getattr(paths, "config", None), "exchange", None)
    live_canary_enabled = bool(getattr(runtime_cfg, "live_canary_enabled", True))
    live_canary_max_symbols = int(getattr(runtime_cfg, "live_canary_max_symbols", 1) or 1)
    live_canary_max_notional_usd = float(getattr(runtime_cfg, "live_canary_max_notional_usd", 50.0) or 0.0)
    live_canary_max_concurrent_positions = int(getattr(runtime_cfg, "live_canary_max_concurrent_positions", 1) or 1)
    live_canary_require_testnet = bool(getattr(runtime_cfg, "live_canary_require_testnet", True))
    canary_testnet_ok = (not live_canary_require_testnet) or bool(getattr(exchange_cfg, "use_testnet", False))
    canary_restrictions_ok = all((
        live_canary_enabled,
        enable_live_canary,
        live_canary_max_symbols <= 1,
        live_canary_max_notional_usd <= 50.0,
        live_canary_max_concurrent_positions <= 1,
        canary_testnet_ok,
    ))
    _record(
        items,
        "deliberate_live_canary_enablement",
        "pass" if enable_live_canary else "fail",
        "Explicit live canary enablement is present." if enable_live_canary else "Explicit live canary enablement is absent; live trading remains fail-closed.",
    )
    _record(
        items,
        "live_canary_restrictions",
        "pass" if canary_restrictions_ok else "fail",
        "Restricted live canary limits are configured." if canary_restrictions_ok else "Restricted live canary limits are missing or exceed the deployment envelope.",
        live_canary_enabled=live_canary_enabled,
        live_canary_max_symbols=live_canary_max_symbols,
        live_canary_max_notional_usd=live_canary_max_notional_usd,
        live_canary_max_concurrent_positions=live_canary_max_concurrent_positions,
        live_canary_require_testnet=live_canary_require_testnet,
        exchange_use_testnet=bool(getattr(exchange_cfg, "use_testnet", False)),
    )

    risk_contract_ok, risk_contract_details = _risk_and_circuit_contract(paths)
    _record(
        items,
        "risk_and_circuit_contract",
        "pass" if risk_contract_ok else "fail",
        "Risk limits and circuit-breaker persistence are active." if risk_contract_ok else "Risk limits or circuit-breaker persistence are not live-ready.",
        **risk_contract_details,
    )

    overall_status = "pass" if all(item.status == "pass" for item in items) else "fail"
    return LiveTradingReadinessReport(
        project_root=str(root),
        overall_status=overall_status,
        live_trading_ready=(overall_status == "pass"),
        items=items,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live trading readiness check")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()

    project_root = args.project_root or Path.cwd()
    report = run_live_trading_readiness_check(project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.live_trading_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
