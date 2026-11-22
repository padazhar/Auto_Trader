from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.path_manager import PathManager
from inference.readiness import signal_readiness_snapshot


@dataclass(slots=True)
class RegistryReadinessVerificationReport:
    """Registry-derived execution-readiness verification report.

    This report is diagnostic and fail-closed. It does not promote models,
    modify registry state, change symbol lifecycle, or enable live trading.
    """

    project_root: str
    overall_status: str
    registry_gate_consistent: bool
    active_symbols_count: int
    symbol_state_execution_eligible_count: int
    registry_execution_ready_count: int
    false_ready_symbols: list[str] = field(default_factory=list)
    registry_ready_not_in_symbol_state: list[str] = field(default_factory=list)
    active_symbols: list[str] = field(default_factory=list)
    symbol_state_execution_eligible_symbols: list[str] = field(default_factory=list)
    registry_execution_ready_symbols: list[str] = field(default_factory=list)
    reason_counts: dict[str, int] = field(default_factory=dict)
    model_blockers: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    model_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    blocker_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _flatten_model_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        interval = str(row.get("interval") or "")
        models = row.get("models") if isinstance(row.get("models"), dict) else {}
        for model_type, model in models.items():
            if not isinstance(model, dict):
                continue
            ready = bool(model.get("ready", False))
            reason = str(model.get("reason") or "unknown")
            item = {
                "symbol": symbol,
                "interval": interval,
                "model_type": str(model_type).lower(),
                "execution_ready": ready,
                "ready": ready,
                "blocker_reason": "ready" if ready else reason,
                "status": str(model.get("status") or "unknown"),
                "deployment_stage": str(model.get("deployment_stage") or "unknown"),
                "active_manifest_exists": bool(model.get("active_manifest_exists", bool(model.get("active_manifest_path") or model.get("manifest_path")))),
                "active_manifest_valid": bool(model.get("active_manifest_valid", False)),
                "active_manifest_path": model.get("active_manifest_path") or model.get("manifest_path"),
                "registry_entry_exists": bool(model.get("registry_entry_exists", False)),
                "registry_truth_status": str(model.get("registry_truth_status") or "unknown"),
                "registry_execution_ready": bool(model.get("registry_execution_ready", False)),
                "registry_promotion_approved": bool(model.get("registry_promotion_approved", False)),
                "registry_active_manifest_valid": bool(model.get("registry_active_manifest_valid", False)),
                "feature_compatible": bool(model.get("feature_compatible", False)),
                "feature_contract_diagnostics": dict(model.get("feature_contract_diagnostics") or {}),
            }
            diagnostics.append(item)
    return sorted(diagnostics, key=lambda item: (item["symbol"], item["model_type"], item["interval"]))

def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).upper() for item in value if str(item).strip()]


def run_registry_readiness_verification(project_root: Path | str | None = None) -> RegistryReadinessVerificationReport:
    """Verify symbol execution eligibility is derived from registry readiness.

    Step 4 contract: active manifests and symbol-state execution eligibility are
    not trusted by themselves. A symbol may be execution eligible only when the
    registry/readiness gate says the required anchor models are execution-ready.
    """
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    paths = PathManager.from_project_root(root)
    symbol_state = _load_json_mapping(paths.runtime_dir / "symbol_state.json")

    active_symbols = _string_list(symbol_state.get("active_symbols"))
    symbol_state_eligible = _string_list(symbol_state.get("execution_eligible_symbols"))
    anchor_interval = str(getattr(getattr(paths.config, "frames", None), "anchor_interval", "15m") or "15m")

    snapshot = signal_readiness_snapshot(paths, symbols=active_symbols, interval=anchor_interval, required_models=("rf", "gbm"))
    registry_ready = _string_list(snapshot.get("execution_eligible_symbols"))

    registry_ready_set = set(registry_ready)
    symbol_state_eligible_set = set(symbol_state_eligible)
    false_ready = sorted(symbol_state_eligible_set - registry_ready_set)
    registry_ready_not_in_symbol_state = sorted(registry_ready_set - symbol_state_eligible_set)
    consistent = not false_ready
    overall_status = "pass" if consistent else "fail"

    rows = list(snapshot.get("rows") or [])
    model_diagnostics = _flatten_model_diagnostics(rows)
    blocker_diagnostics = [item for item in model_diagnostics if not bool(item.get("execution_ready"))]

    report = RegistryReadinessVerificationReport(
        project_root=str(root),
        overall_status=overall_status,
        registry_gate_consistent=consistent,
        active_symbols_count=len(active_symbols),
        symbol_state_execution_eligible_count=len(symbol_state_eligible),
        registry_execution_ready_count=len(registry_ready),
        false_ready_symbols=false_ready,
        registry_ready_not_in_symbol_state=registry_ready_not_in_symbol_state,
        active_symbols=active_symbols,
        symbol_state_execution_eligible_symbols=symbol_state_eligible,
        registry_execution_ready_symbols=registry_ready,
        reason_counts=dict(snapshot.get("reason_counts") or {}),
        model_blockers=dict(snapshot.get("model_blockers") or {}),
        rows=rows,
        model_diagnostics=model_diagnostics,
        blocker_diagnostics=blocker_diagnostics,
    )

    output_path = paths.runtime_dir / "registry_readiness_verification_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify execution eligibility is registry-derived and fail-closed.")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args(argv)
    report = run_registry_readiness_verification(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.registry_gate_consistent else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
