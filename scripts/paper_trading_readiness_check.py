from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from scripts.paper_mode_smoke_check import run_paper_mode_smoke_check
from scripts.paper_mode_soak_check import run_paper_mode_soak_check
from scripts.repository_readiness_check import run_readiness_check
from scripts.runtime_self_check import run_runtime_self_check


@dataclass(slots=True)
class PaperReadinessItem:
    """Single paper-trading readiness check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperTradingReadinessReport:
    """Structured paper-trading readiness report."""

    project_root: str
    overall_status: str
    paper_trading_ready: bool
    items: list[PaperReadinessItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the readiness report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "paper_trading_ready": self.paper_trading_ready,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[PaperReadinessItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a readiness item to the report."""
    items.append(PaperReadinessItem(name=name, status=status, message=message, details=dict(details)))


def run_paper_trading_readiness_check(project_root: Path | str | None = None) -> PaperTradingReadinessReport:
    """Run the combined paper-trading readiness gate for the repository."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[PaperReadinessItem] = []

    repository_report = run_readiness_check(root)
    repository_payload = repository_report.to_dict()
    repository_ok = bool(repository_payload.get("overall_status") == "pass" and repository_payload.get("paper_trading_ready") is True)
    _record(
        items,
        "repository_readiness",
        "pass" if repository_ok else "fail",
        "Repository readiness gate passed." if repository_ok else "Repository readiness gate failed.",
        report=repository_payload,
    )

    smoke_report = run_paper_mode_smoke_check(root)
    smoke_payload = smoke_report.to_dict()
    smoke_ok = bool(smoke_payload.get("paper_mode_ready") is True)
    _record(
        items,
        "paper_smoke",
        "pass" if smoke_ok else "fail",
        "Paper smoke check passed." if smoke_ok else "Paper smoke check failed.",
        report=smoke_payload,
    )

    soak_report = run_paper_mode_soak_check(root)
    soak_payload = soak_report.to_dict()
    soak_ok = bool(soak_payload.get("paper_mode_soak_ready") is True)
    _record(
        items,
        "paper_soak",
        "pass" if soak_ok else "fail",
        "Paper soak check passed." if soak_ok else "Paper soak check failed.",
        report=soak_payload,
    )

    runtime_report = run_runtime_self_check(root)
    runtime_payload = runtime_report.to_dict()
    runtime_ok = bool(runtime_payload.get("runtime_ready") is True)
    _record(
        items,
        "runtime_self_check",
        "pass" if runtime_ok else "fail",
        "Runtime self-check passed." if runtime_ok else "Runtime self-check failed.",
        report=runtime_payload,
    )

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    return PaperTradingReadinessReport(
        project_root=str(root),
        overall_status=overall_status,
        paper_trading_ready=(overall_status == "pass"),
        items=items,
    )


def main() -> int:
    """CLI entrypoint for combined paper-trading readiness validation."""
    parser = argparse.ArgumentParser(description="Paper-trading readiness check")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args()

    report = run_paper_trading_readiness_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.paper_trading_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
