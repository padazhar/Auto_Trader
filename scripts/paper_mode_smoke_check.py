from __future__ import annotations


import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import main as main_module
from core.path_manager import PathManager
from core.training_registry import TrainingRegistry
from monitoring.control_plane_store import ControlPlaneStore
from runners.live_trading_controller import LiveTradingController
from services.signal_engine import SignalEngine


@dataclass(slots=True)
class SmokeCheckItem:
    """Single paper-mode smoke check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperModeSmokeReport:
    """Structured paper-mode smoke validation report."""

    project_root: str
    overall_status: str
    paper_mode_ready: bool
    items: list[SmokeCheckItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the smoke report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "paper_mode_ready": self.paper_mode_ready,
            "items": [asdict(item) for item in self.items],
        }


class _NullLogger:
    """Minimal logger used for launcher-guard smoke validation."""

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass


def _record(items: list[SmokeCheckItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a smoke-check item to the report."""
    items.append(SmokeCheckItem(name=name, status=status, message=message, details=dict(details)))


def _check_config_and_paths(project_root: Path, items: list[SmokeCheckItem]) -> tuple[bool, Any | None]:
    """Verify repository config/bootstrap works in the current paper-mode default."""
    try:
        paths = PathManager.from_project_root(project_root)
        execution_mode = str(getattr(getattr(paths.config, "runtime", None), "execution_mode", "paper") or "paper").strip().lower()
        if execution_mode != "paper":
            _record(
                items,
                "config_bootstrap",
                "warn",
                "Repository config loaded, but default execution mode is not paper.",
                execution_mode=execution_mode,
            )
            return True, paths
        _record(
            items,
            "config_bootstrap",
            "pass",
            "Repository config and paths loaded successfully in paper mode.",
            execution_mode=execution_mode,
        )
        return True, paths
    except Exception as exc:
        _record(items, "config_bootstrap", "fail", "Repository config/bootstrap failed.", error=str(exc))
        return False, None


def _check_module_imports(items: list[SmokeCheckItem]) -> None:
    """Verify key paper-runtime modules import successfully."""
    modules = [
        "main",
        "inference.model_artifact_loader",
        "inference.trade_decision_logic",
        "services.signal_engine",
        "runners.live_trading_controller",
        "monitoring.web_app",
    ]
    imported: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            _record(
                items,
                "module_imports",
                "fail",
                f"Failed to import required module: {module_name}",
                module=module_name,
                error=str(exc),
                imported=imported,
            )
            return
    _record(items, "module_imports", "pass", "Key paper-runtime modules import successfully.", imported=imported)


def _check_launcher_guard_all_mode(paths, items: list[SmokeCheckItem]) -> None:
    """Verify paper-safe orchestration allows --mode all without tripping live guards."""
    try:
        main_module._validate_runtime_mode_guards(paths, _NullLogger(), "all")
    except SystemExit as exc:
        _record(
            items,
            "launcher_guard_all_mode",
            "fail",
            "Paper-safe --mode all orchestration was incorrectly blocked.",
            exit_code=getattr(exc, "code", None),
        )
        return
    except Exception as exc:
        _record(items, "launcher_guard_all_mode", "fail", "Launcher guard validation failed unexpectedly.", error=str(exc))
        return
    _record(items, "launcher_guard_all_mode", "pass", "Paper-safe --mode all orchestration is allowed.")


def _check_constructor_safety(paths, items: list[SmokeCheckItem]) -> None:
    """Verify key runtime objects can be constructed in paper configuration."""
    registry = None
    try:
        registry = TrainingRegistry(paths)
    except Exception as exc:
        _record(items, "constructor_training_registry", "fail", "TrainingRegistry construction failed.", error=str(exc))
        return
    _record(items, "constructor_training_registry", "pass", "TrainingRegistry constructed successfully.")

    try:
        ControlPlaneStore(paths)
    except Exception as exc:
        _record(items, "constructor_control_plane_store", "fail", "ControlPlaneStore construction failed.", error=str(exc))
        return
    _record(items, "constructor_control_plane_store", "pass", "ControlPlaneStore constructed successfully.")

    try:
        SignalEngine(paths)
    except Exception as exc:
        _record(items, "constructor_signal_engine", "fail", "SignalEngine construction failed.", error=str(exc))
        return
    _record(items, "constructor_signal_engine", "pass", "SignalEngine constructed successfully.")

    try:
        LiveTradingController(paths, registry)
    except Exception as exc:
        _record(items, "constructor_live_trading_controller", "fail", "LiveTradingController construction failed.", error=str(exc))
        return
    _record(items, "constructor_live_trading_controller", "pass", "LiveTradingController constructed successfully in paper configuration.")


def run_paper_mode_smoke_check(project_root: Path | str | None = None) -> PaperModeSmokeReport:
    """Run a lightweight paper-mode smoke validation for the repository."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[SmokeCheckItem] = []

    config_ok, paths = _check_config_and_paths(root, items)
    if paths is not None:
        _check_module_imports(items)
        _check_launcher_guard_all_mode(paths, items)
        _check_constructor_safety(paths, items)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    paper_mode_ready = bool(config_ok and overall_status == "pass")
    return PaperModeSmokeReport(
        project_root=str(root),
        overall_status=overall_status,
        paper_mode_ready=paper_mode_ready,
        items=items,
    )


def main() -> int:
    """CLI entrypoint for paper-mode smoke validation."""
    parser = argparse.ArgumentParser(description="Paper-mode smoke check")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args()

    report = run_paper_mode_smoke_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.paper_mode_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
