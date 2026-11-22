from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import yaml
except Exception:  # pragma: no cover - repository requirements include PyYAML
    yaml = None  # type: ignore[assignment]


@dataclass(slots=True)
class Step0CheckItem:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Step0BaselineReport:
    project_root: str
    overall_status: str
    safe_to_start_step1: bool
    items: list[Step0CheckItem]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "safe_to_start_step1": self.safe_to_start_step1,
            "summary": self.summary,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[Step0CheckItem], name: str, status: str, message: str, **details: Any) -> None:
    items.append(Step0CheckItem(name=name, status=status, message=message, details=dict(details)))


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


def _runtime_file_status(project_root: Path) -> dict[str, bool]:
    required = {
        "symbol_state": project_root / "data" / "runtime" / "symbol_state.json",
        "training_registry": project_root / "data" / "runtime" / "training_registry.json",
        "portfolio_state": project_root / "data" / "runtime" / "portfolio_state.json",
        "market_latest": project_root / "data" / "runtime" / "market" / "latest.json",
        "live_backtest_trades": project_root / "data" / "live_backtest_trades.csv",
        "system_settings": project_root / "config" / "system_settings.yaml",
        "tradable_symbols": project_root / "config" / "tradable_symbols.yaml",
    }
    return {name: path.exists() for name, path in required.items()}


def _repo_extraction_status(project_root: Path) -> dict[str, Any]:
    required_paths = [
        "main.py",
        "config/system_settings.yaml",
        "config/tradable_symbols.yaml",
        "core/symbol_management.py",
        "inference/readiness.py",
        "monitoring/publisher.py",
        "reporting/run_report.py",
        "scripts/runtime_self_check.py",
        "scripts/live_trading_readiness_check.py",
        "scripts/paper_trading_readiness_check.py",
        "tests",
    ]
    missing = [rel for rel in required_paths if not (project_root / rel).exists()]
    return {"repo_extracted_successfully": not missing, "missing_required_paths": missing}


def _summarize_symbol_state(symbol_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(symbol_state, dict):
        return {"present": False}
    active = list(symbol_state.get("active_symbols") or [])
    eligible = list(symbol_state.get("execution_eligible_symbols") or [])
    blocked = list(symbol_state.get("blocked_symbols") or [])
    warming = list(symbol_state.get("warming_symbols") or [])
    historical = list(symbol_state.get("historical_symbols") or [])
    target = symbol_state.get("target_active_symbols")
    symbols = symbol_state.get("symbols") if isinstance(symbol_state.get("symbols"), dict) else {}
    not_eligible = [sym for sym in active if sym not in set(eligible)]
    not_eligible_reasons = {}
    for sym in not_eligible:
        payload = symbols.get(sym, {}) if isinstance(symbols, dict) else {}
        if isinstance(payload, dict):
            not_eligible_reasons[sym] = {
                "activation_stage": payload.get("activation_stage"),
                "data_ready": payload.get("data_ready"),
                "data_ready_reason": payload.get("data_ready_reason"),
                "model_ready_reason": payload.get("model_ready_reason"),
                "execution_eligible": payload.get("execution_eligible"),
                "status": payload.get("status"),
            }
    return {
        "present": True,
        "target_active_symbols": target,
        "active_symbols": active,
        "active_symbols_count": len(active),
        "execution_eligible_symbols": eligible,
        "execution_eligible_symbols_count": len(eligible),
        "blocked_symbols": blocked,
        "warming_symbols": warming,
        "historical_symbols": historical,
        "active_not_execution_eligible": not_eligible,
        "active_not_execution_eligible_reasons": not_eligible_reasons,
        "active_count_matches_target": target is None or len(active) == int(target),
    }


def _summarize_market_latest(market_latest: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(market_latest, dict):
        return {"present": False}
    symbols_payload = market_latest.get("symbols") if isinstance(market_latest.get("symbols"), dict) else {}
    active_symbols = list(market_latest.get("active_symbols") or [])
    open_position_symbols = list(market_latest.get("open_position_symbols") or [])
    price_symbols = list(market_latest.get("price_symbols") or [])
    required_price_symbols = sorted(set(active_symbols) | set(open_position_symbols))
    missing_required = sorted(set(required_price_symbols) - set(price_symbols))
    missing_payload = sorted(set(price_symbols) - set(symbols_payload.keys()))
    return {
        "present": True,
        "source": market_latest.get("source"),
        "stream": market_latest.get("stream"),
        "active_symbols": active_symbols,
        "open_position_symbols": open_position_symbols,
        "price_symbols": price_symbols,
        "expected_symbol_count": market_latest.get("expected_symbol_count"),
        "priced_symbol_count": market_latest.get("priced_symbol_count"),
        "missing_symbols": list(market_latest.get("missing_symbols") or []),
        "required_price_symbols": required_price_symbols,
        "missing_required_price_symbols": missing_required,
        "price_symbol_payload_missing": missing_payload,
        "preserves_open_position_price_coverage": not missing_required,
        "snapshot_health": market_latest.get("snapshot_health"),
    }


def _summarize_portfolio(portfolio_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(portfolio_state, dict):
        return {"present": False}
    positions = portfolio_state.get("positions")
    if isinstance(positions, dict):
        open_symbols = sorted(positions.keys())
        position_count = len(positions)
    elif isinstance(positions, list):
        open_symbols = sorted({str(item.get("symbol")) for item in positions if isinstance(item, dict) and item.get("symbol")})
        position_count = len(positions)
    else:
        open_symbols = []
        position_count = 0
    closed = portfolio_state.get("closed_trades") if isinstance(portfolio_state.get("closed_trades"), list) else []
    lineage_count = 0
    missing_lineage_fields: Counter[str] = Counter()
    required_lineage = ["decision_id", "rf", "gbm", "risk", "close_event"]
    for trade in closed:
        if not isinstance(trade, dict):
            continue
        lineage = trade.get("lineage") if isinstance(trade.get("lineage"), dict) else {}
        if lineage:
            lineage_count += 1
        for field_name in required_lineage:
            if field_name not in lineage:
                missing_lineage_fields[field_name] += 1
    return {
        "present": True,
        "authority_mode": portfolio_state.get("authority_mode"),
        "open_position_count": position_count,
        "open_position_symbols": open_symbols,
        "closed_trade_count": len(closed),
        "closed_trades_with_lineage": lineage_count,
        "lineage_missing_field_counts": dict(missing_lineage_fields),
        "realized_pnl_usd": portfolio_state.get("realized_pnl_usd"),
        "daily_realized_pnl_usd": portfolio_state.get("daily_realized_pnl_usd"),
        "reconciliation_status": portfolio_state.get("reconciliation_status"),
    }


def _summarize_registry(registry: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(registry, dict):
        return {"present": False}
    status_counts: Counter[str] = Counter()
    truth_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    missing_active_manifest = []
    active_without_registry_approval = []
    execution_ready = []
    by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    for key, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        model = str(entry.get("model_type") or key.split(":", 1)[0])
        symbol = str(entry.get("symbol") or "")
        status = str(entry.get("status") or "unknown")
        truth = str(entry.get("registry_truth_status") or "unknown")
        status_counts[status] += 1
        truth_counts[truth] += 1
        model_counts[model] += 1
        if symbol:
            by_symbol[symbol][truth] += 1
        if entry.get("execution_ready") is True:
            execution_ready.append(key)
        reasons = entry.get("truth_reasons") if isinstance(entry.get("truth_reasons"), list) else []
        for reason in reasons:
            blocker_counts[str(reason)] += 1
        if entry.get("active_manifest_valid") is not True:
            missing_active_manifest.append(key)
        if truth == "active_without_registry_approval":
            active_without_registry_approval.append(key)
    return {
        "present": True,
        "entry_count": len(registry),
        "status_counts": dict(status_counts),
        "registry_truth_status_counts": dict(truth_counts),
        "model_type_counts": dict(model_counts),
        "truth_reason_counts": dict(blocker_counts),
        "missing_active_manifest_count": len(missing_active_manifest),
        "missing_active_manifest_examples": missing_active_manifest[:20],
        "active_without_registry_approval_count": len(active_without_registry_approval),
        "active_without_registry_approval_examples": active_without_registry_approval[:20],
        "execution_ready_count": len(execution_ready),
        "by_symbol_truth_counts": {symbol: dict(counter) for symbol, counter in sorted(by_symbol.items())},
    }


def _summarize_trades_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    pnl_values: list[float] = []
    fee_values: list[float] = []
    slippage_values: list[float] = []
    exit_reasons: Counter[str] = Counter()
    wins = 0
    losses = 0
    for row in rows:
        try:
            pnl = float(row.get("pnl_usd") or row.get("realized_pnl_usd") or 0.0)
        except Exception:
            pnl = 0.0
        pnl_values.append(pnl)
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        for target, values in [("fees_usd", fee_values), ("slippage_usd", slippage_values)]:
            try:
                values.append(float(row.get(target) or 0.0))
            except Exception:
                values.append(0.0)
        exit_reasons[str(row.get("exit_reason") or row.get("reason") or "unknown")] += 1
    return {
        "present": True,
        "row_count": len(rows),
        "wins": wins,
        "losses": losses,
        "total_pnl_usd": round(sum(pnl_values), 8),
        "win_rate": round(wins / len(rows), 6) if rows else 0.0,
        "fees_all_zero": all(value == 0.0 for value in fee_values) if rows else None,
        "slippage_all_zero": all(value == 0.0 for value in slippage_values) if rows else None,
        "exit_reason_counts": dict(exit_reasons),
    }


def _summarize_config(project_root: Path) -> dict[str, Any]:
    settings = _load_yaml(project_root / "config" / "system_settings.yaml")
    tradable = _load_yaml(project_root / "config" / "tradable_symbols.yaml")
    runtime = settings.get("runtime") if isinstance(settings.get("runtime"), dict) else {}
    market = settings.get("market") if isinstance(settings.get("market"), dict) else {}
    return {
        "system_settings_present": bool(settings),
        "tradable_symbols_present": bool(tradable),
        "execution_mode": runtime.get("execution_mode") or settings.get("execution_mode"),
        "paper_mode_preserved": str(runtime.get("execution_mode") or settings.get("execution_mode") or "").lower() != "live",
        "allow_unreconciled_live_mode": runtime.get("allow_unreconciled_live_mode"),
        "market_top_n": market.get("top_n"),
        "runtime_price_websocket_enabled": market.get("runtime_price_websocket_enabled") or runtime.get("runtime_price_websocket_enabled"),
    }


def _run_optional_command(project_root: Path, command: list[str], timeout_sec: int = 120) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "passed": False, "error": str(exc)}


def build_step0_baseline(project_root: Path, run_checks: bool = False) -> Step0BaselineReport:
    project_root = Path(project_root).resolve()
    items: list[Step0CheckItem] = []

    extraction = _repo_extraction_status(project_root)
    _record(
        items,
        "repo_extraction",
        "pass" if extraction["repo_extracted_successfully"] else "fail",
        "Repository extraction baseline checked.",
        **extraction,
    )

    runtime_files = _runtime_file_status(project_root)
    missing_runtime = [name for name, present in runtime_files.items() if not present]
    _record(
        items,
        "runtime_artifacts",
        "pass" if not missing_runtime else "fail",
        "Runtime artifacts required for Step 0 were checked.",
        files=runtime_files,
        missing=missing_runtime,
    )

    symbol_state = _summarize_symbol_state(_load_json(project_root / "data" / "runtime" / "symbol_state.json"))
    _record(
        items,
        "symbol_state_baseline",
        "pass" if symbol_state.get("present") else "fail",
        "Symbol lifecycle baseline captured without mutating runtime state.",
        **symbol_state,
    )

    portfolio = _summarize_portfolio(_load_json(project_root / "data" / "runtime" / "portfolio_state.json"))
    _record(
        items,
        "portfolio_state_baseline",
        "pass" if portfolio.get("present") else "fail",
        "Portfolio/open-position baseline captured.",
        **portfolio,
    )

    market_latest = _summarize_market_latest(_load_json(project_root / "data" / "runtime" / "market" / "latest.json"))
    price_coverage_ok = bool(market_latest.get("present")) and bool(market_latest.get("preserves_open_position_price_coverage"))
    _record(
        items,
        "price_subscription_baseline",
        "pass" if price_coverage_ok else "fail",
        "Price subscription baseline checked: required symbols are active_symbols union open_position_symbols.",
        **market_latest,
    )

    registry = _summarize_registry(_load_json(project_root / "data" / "runtime" / "training_registry.json"))
    _record(
        items,
        "registry_baseline",
        "pass" if registry.get("present") else "fail",
        "Training registry baseline captured for Step 1/2 planning.",
        **registry,
    )

    trades = _summarize_trades_csv(project_root / "data" / "live_backtest_trades.csv")
    _record(
        items,
        "closed_trade_csv_baseline",
        "pass" if trades.get("present") else "fail",
        "Closed-trade CSV baseline captured.",
        **trades,
    )

    config = _summarize_config(project_root)
    _record(
        items,
        "config_safety_baseline",
        "pass" if config.get("system_settings_present") and config.get("paper_mode_preserved") else "fail",
        "Configuration safety baseline captured; Step 0 does not enable live trading.",
        **config,
    )

    optional_commands: list[dict[str, Any]] = []
    if run_checks:
        optional_commands = [
            _run_optional_command(project_root, [sys.executable, "-m", "scripts.runtime_self_check"]),
            _run_optional_command(project_root, [sys.executable, "-m", "scripts.live_trading_readiness_check"]),
            _run_optional_command(project_root, [sys.executable, "scripts/paper_trading_readiness_check.py"]),
        ]
        _record(
            items,
            "optional_command_baseline",
            "pass" if all(item.get("passed") for item in optional_commands) else "fail",
            "Optional readiness commands were executed because --run-checks was supplied.",
            commands=optional_commands,
        )

    blocking_failures = [item.name for item in items if item.status == "fail" and item.name not in {"optional_command_baseline"}]
    summary = {
        "repo_extracted_successfully": extraction["repo_extracted_successfully"],
        "runtime_artifacts_present": not missing_runtime,
        "active_symbols_count": symbol_state.get("active_symbols_count"),
        "target_active_symbols": symbol_state.get("target_active_symbols"),
        "execution_eligible_symbols_count": symbol_state.get("execution_eligible_symbols_count"),
        "active_not_execution_eligible": symbol_state.get("active_not_execution_eligible"),
        "open_position_symbols": portfolio.get("open_position_symbols"),
        "price_subscription_symbols": market_latest.get("price_symbols"),
        "price_subscription_preserves_open_positions": market_latest.get("preserves_open_position_price_coverage"),
        "registry_blocker_summary": {
            "missing_active_manifest_count": registry.get("missing_active_manifest_count"),
            "active_without_registry_approval_count": registry.get("active_without_registry_approval_count"),
            "truth_reason_counts": registry.get("truth_reason_counts"),
        },
        "paper_trade_summary": trades,
        "config_safety": config,
        "current_failing_baseline_items": blocking_failures,
        "safe_to_start_step1": not blocking_failures,
    }
    if optional_commands:
        summary["optional_command_results"] = optional_commands

    return Step0BaselineReport(
        project_root=str(project_root),
        overall_status="pass" if not blocking_failures else "fail",
        safe_to_start_step1=not blocking_failures,
        items=items,
        summary=summary,
    )


def _render_text(report: Step0BaselineReport) -> str:
    payload = report.to_dict()
    summary = payload["summary"]
    lines = [
        "STEP 0 BASELINE REPORT",
        "",
        f"1. Repo extracted successfully: {'YES' if summary.get('repo_extracted_successfully') else 'NO'}",
        f"2. Runtime artifacts present: {'YES' if summary.get('runtime_artifacts_present') else 'NO'}",
        f"3. Active symbols count: {summary.get('active_symbols_count')} / target {summary.get('target_active_symbols')}",
        f"4. Execution eligible symbols count: {summary.get('execution_eligible_symbols_count')}",
        f"5. Open position symbols: {', '.join(summary.get('open_position_symbols') or []) or '--'}",
        f"6. Price subscription symbols: {', '.join(summary.get('price_subscription_symbols') or []) or '--'}",
        "7. Registry blocker summary: " + json.dumps(summary.get("registry_blocker_summary"), sort_keys=True),
        f"8. Current failing baseline items: {', '.join(summary.get('current_failing_baseline_items') or []) or 'none'}",
        f"9. Safe to start Step 1: {'YES' if summary.get('safe_to_start_step1') else 'NO'}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Step 0 evidence baseline before Auto_Trader implementation work.")
    parser.add_argument("--project-root", default=str(_PROJECT_ROOT), help="Auto_Trader project root. Defaults to this repository root.")
    parser.add_argument("--json", action="store_true", help="Print full JSON instead of the short text baseline.")
    parser.add_argument("--output-json", default="", help="Optional path to write the full JSON report.")
    parser.add_argument("--run-checks", action="store_true", help="Also run lightweight readiness commands. This can take longer.")
    args = parser.parse_args(argv)

    report = build_step0_baseline(Path(args.project_root), run_checks=bool(args.run_checks))
    payload = report.to_dict()
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_text(report))
    return 0 if report.safe_to_start_step1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
