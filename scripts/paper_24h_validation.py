from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core.path_manager import PathManager

MIN_PAPER_VALIDATION_SECONDS = 24 * 60 * 60


@dataclass(slots=True)
class Paper24hValidationItem:
    """Single 24-hour paper validation evidence check."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Paper24hValidationReport:
    """Validation result for the required post-retraining 24-hour paper run."""

    project_root: str
    report_path: str
    overall_status: str
    paper_24h_passed: bool
    items: list[Paper24hValidationItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "report_path": self.report_path,
            "overall_status": self.overall_status,
            "paper_24h_passed": self.paper_24h_passed,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[Paper24hValidationItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(Paper24hValidationItem(name=name, status=status, message=message, details=dict(details)))


def _first_number(payload: dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _first_bool(payload: dict[str, Any], names: tuple[str, ...]) -> bool | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "ready"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "blocked", "warn"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
    return None


def _first_string(payload: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = payload.get(name)
        if value is not None:
            return str(value)
    return ""


def validate_paper_24h_report(project_root: Path | str | None = None) -> Paper24hValidationReport:
    """Validate the required 24-hour paper-run evidence without enabling live trading.

    This validator intentionally does not start a paper run. It verifies that an
    operator-produced ``data/runtime/paper_24h_validation_report.json`` contains
    enough evidence to satisfy the Step 5 gate before any live-candidate review.
    """
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    paths = PathManager.from_project_root(root)
    report_path = paths.runtime_dir / "paper_24h_validation_report.json"
    items: list[Paper24hValidationItem] = []

    if not report_path.exists():
        _record(
            items,
            "report_present",
            "warn",
            "24-hour paper validation report is missing; run the required post-retraining paper validation before live-candidate review.",
            report_path=str(report_path),
        )
        return Paper24hValidationReport(
            project_root=str(root),
            report_path=str(report_path),
            overall_status="warn",
            paper_24h_passed=False,
            items=items,
        )

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _record(items, "report_parse", "fail", "24-hour paper validation report could not be parsed.", report_path=str(report_path), error=str(exc))
        return Paper24hValidationReport(str(root), str(report_path), "fail", False, items)

    _record(items, "report_present", "pass", "24-hour paper validation report found and parsed.", report_path=str(report_path))

    duration_seconds = _first_number(payload, ("duration_seconds", "runtime_seconds", "uptime_seconds", "paper_runtime_seconds"))
    duration_ok = duration_seconds >= MIN_PAPER_VALIDATION_SECONDS
    _record(
        items,
        "duration",
        "pass" if duration_ok else "warn",
        "Paper validation duration meets the 24-hour minimum." if duration_ok else "Paper validation duration is below the 24-hour minimum.",
        duration_seconds=duration_seconds,
        required_seconds=MIN_PAPER_VALIDATION_SECONDS,
    )

    execution_mode = _first_string(payload, ("execution_mode", "mode", "runtime_mode")).strip().lower()
    paper_mode_ok = execution_mode in {"paper", "paper_trading", "simulated"}
    _record(
        items,
        "paper_mode",
        "pass" if paper_mode_ok else "fail",
        "Paper validation ran in paper mode." if paper_mode_ok else "Paper validation report does not prove paper-mode execution.",
        execution_mode=execution_mode or None,
    )

    runtime_clean = _first_bool(payload, ("runtime_stability_clean", "no_controller_crashes", "controller_stability_clean"))
    crash_count = _first_number(payload, ("controller_crash_count", "crash_count", "runtime_crash_count"))
    runtime_ok = bool(runtime_clean) if runtime_clean is not None else crash_count == 0.0
    _record(
        items,
        "runtime_stability",
        "pass" if runtime_ok else "fail",
        "Paper validation reports clean runtime stability." if runtime_ok else "Paper validation reports runtime/controller instability.",
        runtime_stability_clean=runtime_clean,
        controller_crash_count=crash_count,
    )

    retraining_ok = _first_bool(payload, ("retraining_completed", "governed_retraining_completed", "retraining_ok"))
    _record(
        items,
        "governed_retraining",
        "pass" if retraining_ok is True else "warn",
        "Paper validation confirms governed retraining was completed." if retraining_ok is True else "Paper validation does not confirm governed retraining completion.",
        retraining_completed=retraining_ok,
    )

    registry_ok = _first_bool(payload, ("registry_approval_correct", "registry_gate_consistent", "registry_ok"))
    _record(
        items,
        "registry_approval",
        "pass" if registry_ok is True else "fail",
        "Paper validation confirms registry-approved readiness." if registry_ok is True else "Paper validation does not prove registry-approved readiness.",
        registry_approval_correct=registry_ok,
    )

    costs_ok = _first_bool(payload, ("costs_included", "net_pnl_after_costs", "fees_slippage_spread_included"))
    _record(
        items,
        "cost_realism",
        "pass" if costs_ok is True else "fail",
        "Paper validation confirms fees/slippage/spread were included." if costs_ok is True else "Paper validation does not prove realistic trading costs were included.",
        costs_included=costs_ok,
    )

    explicit_pass = bool(payload.get("paper_24h_passed"))
    _record(
        items,
        "explicit_operator_result",
        "pass" if explicit_pass else "warn",
        "Paper validation report explicitly marks the 24-hour run as passed." if explicit_pass else "Paper validation report does not explicitly mark the 24-hour run as passed.",
        paper_24h_passed=payload.get("paper_24h_passed"),
    )

    hard_fail = any(item.status == "fail" for item in items)
    paper_24h_passed = bool(explicit_pass and duration_ok and paper_mode_ok and runtime_ok and retraining_ok is True and registry_ok is True and costs_ok is True and not hard_fail)
    overall_status = "pass" if paper_24h_passed else "fail" if hard_fail else "warn"
    return Paper24hValidationReport(str(root), str(report_path), overall_status, paper_24h_passed, items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the required 24-hour paper-run evidence")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    report = validate_paper_24h_report(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.paper_24h_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
