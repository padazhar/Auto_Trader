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
from scripts.paper_24h_validation import validate_paper_24h_report


@dataclass(slots=True)
class PostPaperAttributionItem:
    """Single post-paper attribution evidence check."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostPaperAttributionReport:
    """Post-24h-paper attribution review gate.

    This gate does not tune strategy parameters and does not enable live trading.
    It only proves that the required attribution artifacts exist after the
    validated paper run so a human/operator can judge trade quality before any
    live-candidate review.
    """

    project_root: str
    report_path: str | None
    overall_status: str
    attribution_analysis_passed: bool
    items: list[PostPaperAttributionItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "report_path": self.report_path,
            "overall_status": self.overall_status,
            "attribution_analysis_passed": self.attribution_analysis_passed,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[PostPaperAttributionItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(PostPaperAttributionItem(name=name, status=status, message=message, details=dict(details)))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_run_report(paths: PathManager) -> Path | None:
    reports_dir = paths.reports_dir
    if not reports_dir.exists():
        return None
    reports = sorted(reports_dir.glob("run_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0] if reports else None


def _resolve_run_report_path(paths: PathManager, paper_payload: dict[str, Any] | None) -> Path | None:
    if isinstance(paper_payload, dict):
        for key in ("run_report_path", "report_path", "source_run_report_path"):
            value = paper_payload.get(key)
            if not value:
                continue
            candidate = Path(str(value))
            if not candidate.is_absolute():
                candidate = paths.project_root / candidate
            if candidate.exists():
                return candidate
    return _latest_run_report(paths)


def analyze_post_paper_attribution(project_root: Path | str | None = None) -> PostPaperAttributionReport:
    """Validate post-paper attribution evidence without changing trading state."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    paths = PathManager.from_project_root(root)
    items: list[PostPaperAttributionItem] = []

    paper_report = validate_paper_24h_report(root)
    paper_ok = bool(paper_report.paper_24h_passed)
    _record(
        items,
        "paper_24h_gate",
        "pass" if paper_ok else str(paper_report.overall_status),
        "24-hour paper validation passed before attribution analysis." if paper_ok else "Post-paper attribution analysis is blocked until complete 24-hour paper evidence passes.",
        paper_24h_passed=paper_ok,
        paper_report_path=str(paper_report.report_path),
    )

    if not paper_ok:
        report = PostPaperAttributionReport(str(root), None, "warn", False, items)
        output_path = paths.runtime_dir / "post_paper_attribution_report.json"
        output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return report

    paper_payload = _load_json(Path(paper_report.report_path)) if Path(paper_report.report_path).exists() else None
    run_report_path = _resolve_run_report_path(paths, paper_payload)
    if run_report_path is None:
        _record(items, "run_report_present", "warn", "No run report was found for post-paper attribution analysis.", reports_dir=str(paths.reports_dir))
        return PostPaperAttributionReport(str(root), None, "warn", False, items)

    run_report = _load_json(run_report_path)
    if run_report is None:
        _record(items, "run_report_parse", "fail", "Run report could not be parsed.", report_path=str(run_report_path))
        return PostPaperAttributionReport(str(root), str(run_report_path), "fail", False, items)

    _record(items, "run_report_present", "pass", "Run report found and parsed for attribution analysis.", report_path=str(run_report_path))

    execution = run_report.get("execution") if isinstance(run_report.get("execution"), dict) else {}
    completed_trades = _safe_int(execution.get("completed_trades"), 0)
    net_pnl = _safe_float(execution.get("total_pnl"), 0.0)
    fees = _safe_float(execution.get("fees_usd"), 0.0)
    slippage = _safe_float(execution.get("slippage_usd"), 0.0)
    _record(
        items,
        "execution_summary",
        "pass" if completed_trades > 0 else "warn",
        "Closed trades are available for post-paper attribution." if completed_trades > 0 else "No closed trades are available for attribution; paper result is not decision-useful.",
        completed_trades=completed_trades,
        net_pnl=net_pnl,
        fees_usd=fees,
        slippage_usd=slippage,
    )

    closed_attr = run_report.get("closed_trade_attribution") if isinstance(run_report.get("closed_trade_attribution"), dict) else {}
    closed_attr_count = _safe_int(closed_attr.get("closed_trade_count"), 0)
    missing_decisions = _safe_int(closed_attr.get("missing_decision_id_count"), 0)
    match_counts = closed_attr.get("prediction_match_counts") if isinstance(closed_attr.get("prediction_match_counts"), dict) else {}
    closed_ok = closed_attr_count > 0 and missing_decisions == 0 and bool(match_counts)
    _record(
        items,
        "closed_trade_attribution",
        "pass" if closed_ok else "warn",
        "Closed-trade attribution is complete enough for review." if closed_ok else "Closed-trade attribution is missing, empty, or has missing decision IDs.",
        closed_trade_count=closed_attr_count,
        missing_decision_id_count=missing_decisions,
        prediction_match_counts=match_counts,
    )

    excursion = run_report.get("trade_excursion_attribution") if isinstance(run_report.get("trade_excursion_attribution"), dict) else {}
    excursion_count = _safe_int(excursion.get("closed_trade_count", excursion.get("trade_count")), 0)
    excursion_ok = excursion_count > 0 and ("avg_mfe_bps" in excursion or "avg_mae_bps" in excursion)
    _record(
        items,
        "mae_mfe_attribution",
        "pass" if excursion_ok else "warn",
        "MAE/MFE attribution is available for closed trades." if excursion_ok else "MAE/MFE attribution is missing or empty.",
        closed_trade_count=excursion_count,
        avg_mfe_bps=excursion.get("avg_mfe_bps"),
        avg_mae_bps=excursion.get("avg_mae_bps"),
    )

    risk_attr = run_report.get("risk_rejection_attribution") if isinstance(run_report.get("risk_rejection_attribution"), dict) else {}
    risk_rejections = _safe_int(risk_attr.get("risk_rejection_count"), 0)
    risk_ok = "reason_counts" in risk_attr and "clamp_counts" in risk_attr
    _record(
        items,
        "risk_rejection_attribution",
        "pass" if risk_ok else "warn",
        "Risk rejection attribution is available for review." if risk_ok else "Risk rejection attribution is missing or incomplete.",
        risk_rejection_count=risk_rejections,
        reason_counts=risk_attr.get("reason_counts", {}),
        clamp_counts=risk_attr.get("clamp_counts", {}),
    )

    cost_realism_ok = ("gross_pnl_usd" in execution) and ("fees_usd" in execution) and ("slippage_usd" in execution)
    _record(
        items,
        "cost_attribution",
        "pass" if cost_realism_ok else "fail",
        "Execution summary carries gross/net cost fields." if cost_realism_ok else "Execution summary is missing gross/fee/slippage fields required for post-paper review.",
        gross_pnl_usd=execution.get("gross_pnl_usd"),
        fees_usd=execution.get("fees_usd"),
        slippage_usd=execution.get("slippage_usd"),
        total_pnl=execution.get("total_pnl"),
    )

    hard_fail = any(item.status == "fail" for item in items)
    attribution_analysis_passed = bool(paper_ok and closed_ok and excursion_ok and risk_ok and cost_realism_ok and not hard_fail)
    if hard_fail:
        overall_status = "fail"
    elif attribution_analysis_passed:
        overall_status = "pass"
    else:
        overall_status = "warn"

    report = PostPaperAttributionReport(str(root), str(run_report_path), overall_status, attribution_analysis_passed, items)
    output_path = paths.runtime_dir / "post_paper_attribution_report.json"
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate post-24h-paper attribution evidence before live-candidate review")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    report = analyze_post_paper_attribution(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.attribution_analysis_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
