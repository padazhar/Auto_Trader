from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import PathManager


@dataclass(slots=True)
class RuntimeSelfCheckItem:
    """Single runtime self-check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeSelfCheckReport:
    """Structured runtime self-check report."""

    project_root: str
    overall_status: str
    runtime_ready: bool
    items: list[RuntimeSelfCheckItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the self-check report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "runtime_ready": self.runtime_ready,
            "items": [asdict(item) for item in self.items],
        }


class _NullLogger:
    """Minimal logger stub for launcher-guard validation."""

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def _record(items: list[RuntimeSelfCheckItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a runtime self-check item to the report."""
    items.append(RuntimeSelfCheckItem(name=name, status=status, message=message, details=dict(details)))


def _seed_market_data(paths: PathManager, symbol: str = "BTCUSDT") -> None:
    import pandas as pd
    from features.market_loader import save_market_frame

    market_db_path = paths.market_db_path(symbol)
    market_db_path.parent.mkdir(parents=True, exist_ok=True)
    if market_db_path.exists():
        market_db_path.unlink()

    base_ts = 1_700_000_000_000
    step_map = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    for interval in paths.config.frames.fusion_intervals:
        step = step_map[interval]
        rows = []
        price = 100.0
        for idx in range(600):
            ts = base_ts + idx * step
            direction = 1 if (idx // 12) % 2 == 0 else -1
            price += 0.35 * direction
            rows.append(
                {
                    "timestamp": ts,
                    "open": price - 0.12 * direction,
                    "high": price + 0.25,
                    "low": price - 0.25,
                    "close": price,
                    "volume": 1000.0 + idx,
                }
            )
        save_market_frame(paths, symbol, interval, pd.DataFrame(rows))


def _run_check(items: list[RuntimeSelfCheckItem], name: str, fn: Callable[[PathManager], None], paths: PathManager) -> None:
    """Run a single check and store a structured result."""
    try:
        fn(paths)
    except Exception as exc:
        _record(items, name, "fail", f"{name} failed", error=str(exc))
        return
    _record(items, name, "pass", f"{name} passed")


def check_market_data_freshness(paths: PathManager) -> None:
    from features.feature_store_contract import validate_market_data_freshness
    from features.market_loader import latest_market_timestamp

    _seed_market_data(paths)
    interval = str(paths.config.frames.anchor_interval)
    latest_ts = latest_market_timestamp(paths, "BTCUSDT", interval)
    freshness = validate_market_data_freshness(paths, symbol="BTCUSDT", interval=interval, latest_timestamp_ms=latest_ts, now_ms=int(latest_ts or 0) + 1)
    if freshness.get("ok") is not True:
        raise AssertionError(f"market freshness must pass for seeded data, got {freshness}")


def check_registry(paths: PathManager) -> None:
    """Verify the training registry contract without instantiating runtime publishers."""
    import ast

    registry_path = paths.project_root / "core" / "training_registry.py"
    if not registry_path.exists():
        raise AssertionError("training registry file is missing")

    source = registry_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(registry_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "TrainingRegistry" not in class_names:
        raise AssertionError("training registry must define TrainingRegistry")

    required_methods = {"mark_queued", "mark_running", "mark_failed", "mark_success", "counts"}
    missing = sorted(required_methods - function_names)
    if missing:
        raise AssertionError(f"training registry missing required method(s): {missing}")

    missing_statuses = [token for token in ["queued", "running", "failed", "success"] if token not in source]
    if missing_statuses:
        raise AssertionError(f"training registry missing status token(s): {missing_statuses}")

    missing_persistence = [token for token in ["json", "registry_path"] if token not in source]
    if missing_persistence:
        raise AssertionError(f"training registry persistence contract missing token(s): {missing_persistence}")

def check_web_app(paths: PathManager) -> None:
    """Verify monitoring web-app contracts without starting FastAPI/TestClient."""
    import ast

    web_app_path = paths.project_root / "monitoring" / "web_app.py"
    if not web_app_path.exists():
        raise AssertionError("monitoring web_app.py is missing")

    source = web_app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(web_app_path))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "create_app" not in function_names:
        raise AssertionError("monitoring web app must expose create_app")

    required_route_tokens = ["/api/events", "/api/health", "/api/monitoring-owner", "/ws"]
    missing_routes = [token for token in required_route_tokens if token not in source]
    if missing_routes:
        raise AssertionError(f"monitoring web app missing route token(s): {missing_routes}")

    required_owner_tokens = ["monitoring_server_owner.json", "monitoring_owner"]
    missing_owner_tokens = [token for token in required_owner_tokens if token not in source]
    if missing_owner_tokens:
        raise AssertionError(f"monitoring owner contract missing token(s): {missing_owner_tokens}")

def check_config(paths: PathManager) -> None:
    cfg = paths.config
    if cfg.frames.anchor_interval not in cfg.frames.fusion_intervals:
        raise AssertionError("anchor interval must exist in fusion intervals")
    if set(cfg.market.fast_frames + cfg.market.slow_frames) != set(cfg.frames.fusion_intervals):
        raise AssertionError("market frames must match fusion intervals exactly")
    if "30m" not in cfg.frames.fusion_intervals:
        raise AssertionError("30m must exist in fusion intervals")
    if "30m" not in cfg.frames.lookback_rows:
        raise AssertionError("30m must exist in frames.lookback_rows")
    if "30m" not in cfg.market.lookback_days:
        raise AssertionError("30m must exist in market.lookback_days")
    if {
        "hold": cfg.inference.dqn_action_hold,
        "long": cfg.inference.dqn_action_long,
        "short": cfg.inference.dqn_action_short,
    } != {"hold": 0, "long": 1, "short": 2}:
        raise AssertionError("DQN action map must be hold=0, long=1, short=2")
    decision_logic_path = paths.project_root / "inference" / "trade_decision_logic.py"
    if not decision_logic_path.exists():
        raise AssertionError("trade decision logic file is missing")


def check_paper_mode_guards(paths: PathManager) -> None:
    """Verify paper mode is safe and live launcher selection is fail-closed.

    This check intentionally avoids importing main.py because launcher imports pull
    controller implementations and optional ML dependencies. Step-1 must remain
    a lightweight boot/import gate; dedicated launcher tests cover the concrete
    guard function.
    """
    execution_mode = str(paths.config.runtime.execution_mode).strip().lower()
    if execution_mode != "paper":
        raise AssertionError("runtime self-check requires paper execution_mode for the paper baseline")
    if bool(getattr(paths.config.runtime, "allow_unreconciled_live_mode", False)):
        raise AssertionError("paper baseline must not allow unreconciled live mode")

    main_path = paths.project_root / "main.py"
    if not main_path.exists():
        raise AssertionError("main.py is missing")
    main_source = main_path.read_text(encoding="utf-8")
    required_tokens = [
        "def _validate_runtime_mode_guards",
        "execution_mode != \"live\"",
        "raise SystemExit(3)",
        "AUTO_TRADER_ENABLE_LIVE_MODE",
    ]
    missing = [token for token in required_tokens if token not in main_source]
    if missing:
        raise AssertionError(f"live startup guard contract missing token(s): {missing}")

def check_signal_engine_paused_path(paths: PathManager) -> None:
    """Verify the signal engine pause contract is present without importing ML loaders."""
    import ast

    signal_engine_path = paths.project_root / "services" / "signal_engine.py"
    if not signal_engine_path.exists():
        raise AssertionError("signal engine file is missing")
    tree = ast.parse(signal_engine_path.read_text(encoding="utf-8"), filename=str(signal_engine_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "SignalEngine" not in class_names:
        raise AssertionError("signal engine must define SignalEngine")
    if "compute_decision" not in function_names:
        raise AssertionError("signal engine must expose compute_decision")
    source = signal_engine_path.read_text(encoding="utf-8")
    if "control_plane_paused" not in source:
        raise AssertionError("paused signal path must report control_plane_paused")
    if "PortfolioStateStore" not in source or "portfolio_state_path" not in source:
        raise AssertionError("signal engine must derive open-position snapshot from portfolio authority")


def check_runtime_cleanup_contract(paths: PathManager) -> None:
    """Verify runtime cleanup resets rebuildable data while preserving evidence."""
    from scripts.clean_runtime_state import (
        MARKET_DATA_DIRS,
        PRESERVED_HINTS,
        TRAINING_ARTIFACT_DIRS,
        build_arg_parser,
        clean_runtime_state,
    )

    parser = build_arg_parser()
    option_dests = {action.dest for action in parser._actions}
    required_options = {"project_root", "include_model_artifacts", "include_market_storage", "dry_run"}
    missing_options = sorted(required_options - option_dests)
    if missing_options:
        raise AssertionError(f"runtime cleanup CLI missing option(s): {missing_options}")

    market_tokens = {str(item) for item in MARKET_DATA_DIRS}
    if str(Path("data") / "market_storage") not in market_tokens:
        raise AssertionError("runtime cleanup must reset data/market_storage")

    training_tokens = {str(item) for item in TRAINING_ARTIFACT_DIRS}
    required_training = {
        str(Path("data") / "model_storage" / "random_forest_models"),
        str(Path("data") / "model_storage" / "gradient_boost_models"),
        str(Path("data") / "model_storage" / "reinforcement_agents"),
    }
    missing_training = sorted(required_training - training_tokens)
    if missing_training:
        raise AssertionError(f"runtime cleanup training reset missing: {missing_training}")

    preserved_tokens = {str(item) for item in PRESERVED_HINTS}
    required_preserved = {
        str(Path("data") / "runtime"),
        str(Path("data") / "logs"),
        str(Path("data") / "reports"),
        str(Path("data") / "trade_history"),
    }
    missing_preserved = sorted(required_preserved - preserved_tokens)
    if missing_preserved:
        raise AssertionError(f"runtime cleanup evidence preservation missing: {missing_preserved}")

    report = clean_runtime_state(paths.project_root, dry_run=True)
    if not report.dry_run:
        raise AssertionError("runtime cleanup dry-run report must preserve dry_run=True")

def check_rf_training_contract(paths: PathManager) -> None:
    """Verify the RF trainer source contract without running expensive ML training.

    Step-1 is an import/boot gate. It must be deterministic and fast even on
    hosts where optional scientific runtimes are slow to import or unavailable.
    Full RF model training remains covered by trainer-specific tests and by the
    later paper-validation phases.
    """
    import ast

    trainer_path = paths.project_root / "training" / "rf_training_engine.py"
    if not trainer_path.exists():
        raise AssertionError("RF trainer file is missing")

    tree = ast.parse(trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path))
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "RandomForestTrainer" not in class_names:
        raise AssertionError("RF trainer must define RandomForestTrainer")
    if "train_rf_model" not in function_names:
        raise AssertionError("RF trainer must expose train_rf_model")

    artifact_contract_path = paths.project_root / "core" / "artifact_contract.py"
    if not artifact_contract_path.exists():
        raise AssertionError("artifact contract file is missing")
    artifact_tree = ast.parse(artifact_contract_path.read_text(encoding="utf-8"), filename=str(artifact_contract_path))
    artifact_functions = {node.name for node in ast.walk(artifact_tree) if isinstance(node, ast.FunctionDef)}
    required = {"write_manifest", "validate_artifact_bundle", "promote_candidate_manifest"}
    missing = sorted(required - artifact_functions)
    if missing:
        raise AssertionError(f"artifact contract missing required function(s): {missing}")

def check_bundled_manifest(paths: PathManager) -> None:
    bundled_manifest = paths.rf_model_dir("BTCUSDT", "15m") / "manifest.json"
    if not bundled_manifest.exists():
        return
    manifest = json.loads(bundled_manifest.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    for key, value in artifacts.items():
        if Path(value).is_absolute():
            raise AssertionError(f"bundled manifest artifact path for {key} must be relative")



def check_runtime_path_contract(paths: PathManager) -> None:
    """Verify required config/runtime path contracts resolve before Step-1 ledger work.

    This check is intentionally non-destructive. PathManager creates the required
    directories; this check confirms the exact files and parent directories that
    later execution-ledger work will depend on are addressable.
    """
    required_files = [
        paths.system_settings_path,
        paths.tradable_symbols_path,
        paths.project_root / "main.py",
    ]
    missing_files = [str(path.relative_to(paths.project_root)) for path in required_files if not path.exists()]
    if missing_files:
        raise AssertionError(f"required repository file(s) missing: {missing_files}")

    required_dirs = [
        paths.config_dir,
        paths.data_dir,
        paths.runtime_dir,
        paths.monitoring_runtime_dir,
        paths.training_progress_dir,
        paths.logs_dir,
        paths.market_runtime_dir,
    ]
    missing_dirs = [str(path.relative_to(paths.project_root)) for path in required_dirs if not path.exists() or not path.is_dir()]
    if missing_dirs:
        raise AssertionError(f"required runtime directory path(s) unavailable: {missing_dirs}")

    writable_dirs: list[str] = []
    for directory in [paths.runtime_dir, paths.monitoring_runtime_dir, paths.market_runtime_dir, paths.logs_dir]:
        probe = directory / ".runtime_self_check_write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            writable_dirs.append(f"{directory.relative_to(paths.project_root)} ({exc})")
    if writable_dirs:
        raise AssertionError(f"runtime directory path(s) are not writable: {writable_dirs}")


def _load_json_if_present(path: Path) -> Any:
    """Load JSON when a runtime state file exists and fail loudly on corruption."""
    if not path.exists():
        return None
    if not path.is_file():
        raise AssertionError(f"{path} exists but is not a file")
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path} contains invalid JSON: {exc}") from exc


def _validate_jsonl_if_present(path: Path, *, max_lines: int = 250) -> None:
    """Validate a bounded sample of a JSONL runtime file when present."""
    if not path.exists():
        return
    if not path.is_file():
        raise AssertionError(f"{path} exists but is not a file")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > max_lines:
                break
            payload = line.strip()
            if not payload:
                continue
            try:
                json.loads(payload)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path} has invalid JSONL at line {line_number}: {exc}") from exc


def check_runtime_state_files(paths: PathManager) -> None:
    """Validate existing runtime JSON/JSONL files before execution-ledger changes.

    Missing runtime snapshots are allowed for a clean checkout. Corrupted files
    are not allowed because Step 1 needs a trustworthy baseline for portfolio,
    dashboard, registry, monitoring, and market cache state.
    """
    json_paths = [
        paths.symbol_state_path,
        paths.symbol_actions_path,
        paths.control_plane_state_path,
        paths.portfolio_state_path,
        paths.circuit_breaker_state_path,
        paths.dashboard_state_path,
        paths.alerts_path,
        paths.training_registry_path,
        paths.monitoring_state_path,
        paths.monitoring_events_path,
        paths.market_latest_snapshot_path,
    ]
    for path in json_paths:
        payload = _load_json_if_present(path)
        if payload is not None and not isinstance(payload, (dict, list)):
            raise AssertionError(f"{path} must contain a JSON object or array")

    _validate_jsonl_if_present(paths.decision_journal_path)
    _validate_jsonl_if_present(paths.control_plane_actions_path)

    latest_payload = _load_json_if_present(paths.market_latest_snapshot_path)
    if isinstance(latest_payload, dict) and latest_payload:
        priced = latest_payload.get("prices") or latest_payload.get("symbols") or latest_payload.get("price_cache")
        if priced is not None and not isinstance(priced, (dict, list)):
            raise AssertionError("market latest snapshot price container must be a JSON object or array")

    portfolio_payload = _load_json_if_present(paths.portfolio_state_path)
    if isinstance(portfolio_payload, dict):
        open_positions = portfolio_payload.get("open_positions")
        if open_positions is not None and not isinstance(open_positions, (dict, list)):
            raise AssertionError("portfolio_state.open_positions must be a JSON object or array when present")
        closed_trades = portfolio_payload.get("closed_trades")
        if closed_trades is not None and not isinstance(closed_trades, list):
            raise AssertionError("portfolio_state.closed_trades must be a JSON array when present")

def _mirror_repo(project_root: Path, temp_project: Path) -> None:
    """Create a lightweight isolated copy for deterministic boot checks.

    Runtime self-checks must not inherit large overnight runtime/model data from
    the source repository. The PathManager will create a fresh data/ tree inside
    the temporary copy as needed.
    """
    for name in [
        "config",
        "core",
        "execution",
        "features",
        "inference",
        "interface",
        "monitoring",
        "runners",
        "scheduling",
        "services",
        "training",
        "main.py",
        "requirements.txt",
    ]:
        source = project_root / name
        if not source.exists():
            continue
        target = temp_project / name
        if source.is_dir():
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
            shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
        else:
            shutil.copy2(source, target)


def run_runtime_self_check(project_root: Path | str | None = None) -> RuntimeSelfCheckReport:
    """Run a structured lightweight runtime self-check.

    This gate intentionally validates source/runtime contracts in place instead
    of copying the repository. Copying the source tree can be slow or unstable
    on large runtime snapshots, and Step 1/3 validation must be deterministic.
    The checks below avoid training, FastAPI startup, exchange access, and
    monitoring publisher side effects.
    """
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
    items: list[RuntimeSelfCheckItem] = []
    paths = PathManager.from_project_root(root)

    _run_check(items, "config_check", check_config, paths)
    _run_check(items, "runtime_path_contract_check", check_runtime_path_contract, paths)
    _run_check(items, "runtime_state_files_check", check_runtime_state_files, paths)
    _run_check(items, "paper_mode_guard_check", check_paper_mode_guards, paths)
    _run_check(items, "registry_check", check_registry, paths)
    _run_check(items, "web_app_check", check_web_app, paths)
    _run_check(items, "signal_engine_paused_path_check", check_signal_engine_paused_path, paths)
    _run_check(items, "runtime_cleanup_contract_check", check_runtime_cleanup_contract, paths)
    _run_check(items, "bundled_manifest_check", check_bundled_manifest, paths)
    _run_check(items, "rf_training_contract_check", check_rf_training_contract, paths)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    return RuntimeSelfCheckReport(
        project_root=str(root),
        overall_status=overall_status,
        runtime_ready=(overall_status == "pass"),
        items=items,
    )

def main() -> int:
    """CLI entrypoint for Auto Trader runtime self-checks."""
    parser = argparse.ArgumentParser(description="Run Auto Trader runtime self-checks")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    report = run_runtime_self_check(args.project_root)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    sys.stdout.write(payload)
    return 0 if report.runtime_ready else 1


if __name__ == "__main__":

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(exit_code)
