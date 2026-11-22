from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Purpose of this script:
#   Reset market/model/training artifacts so the system can retrain/requalify from
#   a clean data base, while preserving runtime evidence and operator context.
#
# Preserved intentionally:
#   - data/runtime/**/*.json, *.md
#   - data/logs/**/*.log
#   - reports / markdown notes
#   - trade history / runtime evidence
#
# Fully destructive cleanup lives in scripts/clean_all_including_market_data.py.

MARKET_DATA_DIRS = (
    Path("data") / "market_storage",
)

TRAINING_ARTIFACT_DIRS = (
    Path("data") / "model_storage" / "random_forest_models",
    Path("data") / "model_storage" / "gradient_boost_models",
    Path("data") / "model_storage" / "reinforcement_agents",
)

# Generated tabular artifacts that are safe to rebuild and are not runtime logs/json/md.
GENERATED_DATA_GLOBS = (
    Path("data") / "*.csv",
    Path("data") / "*.parquet",
    Path("data") / "*.feather",
)

CACHE_FILE_GLOBS = (
    "*.pyc",
    "*.pyo",
)

CACHE_DIR_GLOBS = (
    "__pycache__",
    ".pytest_cache",
)

PRESERVED_HINTS = (
    Path("data") / "runtime",
    Path("data") / "logs",
    Path("data") / "reports",
    Path("data") / "trade_history",
)

RECREATE_DIRS = (
    Path("data") / "logs",
    Path("data") / "runtime",
    Path("data") / "runtime" / "monitoring",
    Path("data") / "runtime" / "market",
    Path("data") / "runtime" / "process_roles",
    Path("data") / "runtime" / "reports",
    Path("data") / "runtime" / "training_progress",
    Path("data") / "market_storage",
    Path("data") / "model_storage",
    Path("data") / "model_storage" / "random_forest_models",
    Path("data") / "model_storage" / "gradient_boost_models",
    Path("data") / "model_storage" / "reinforcement_agents",
    Path("data") / "reports",
    Path("data") / "trade_history",
)

PACKAGE_MARKERS = (
    Path("data") / "__init__.py",
    Path("data") / "model_storage" / "__init__.py",
)


@dataclass(slots=True)
class CleanRuntimeStateReport:
    """Structured result for a non-destructive requalification cleanup pass."""

    project_root: str
    dry_run: bool
    removed: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    recreated: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def _resolve_project_root(project_root: Path | str | None = None) -> Path:
    root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT.resolve()
    if not (root / "main.py").exists():
        raise ValueError(f"Refusing cleanup: {root} does not look like Auto_Trader root; main.py is missing")
    if not (root / "config" / "system_settings.yaml").exists():
        raise ValueError(
            f"Refusing cleanup: {root} does not look like Auto_Trader root; config/system_settings.yaml is missing"
        )
    return root


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _remove_path(path: Path, root: Path, report: CleanRuntimeStateReport, *, dry_run: bool) -> None:
    resolved = path.resolve()
    if not _is_within(resolved, root):
        report.warnings.append(f"refused_outside_project_root:{resolved}")
        return
    if not path.exists() and not path.is_symlink():
        return

    report.removed.append(str(path))
    if dry_run:
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
        return

    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists() or path.is_symlink():
            path.unlink()


def _iter_existing(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.exists() or path.is_symlink():
            yield path


def clean_runtime_state(
    project_root: Path | str | None = None,
    *,
    include_model_artifacts: bool | None = None,
    include_market_storage: bool | None = None,
    dry_run: bool = False,
) -> CleanRuntimeStateReport:
    """Clear rebuildable market/training artifacts while preserving runtime evidence.

    This is the safe requalification cleanup. It removes downloaded market DBs,
    model artifacts, and generated training/tabular data so training can restart
    cleanly, but it deliberately keeps runtime JSON/MD/log evidence for forensic
    review and UI continuity.

    ``include_model_artifacts`` and ``include_market_storage`` are accepted for
    backward compatibility only; both categories are now always included by this
    script. Use ``clean_all_including_market_data.py`` for a fully destructive
    data wipe.
    """
    root = _resolve_project_root(project_root)
    report = CleanRuntimeStateReport(project_root=str(root), dry_run=dry_run)

    for rel in PRESERVED_HINTS:
        target = root / rel
        if target.exists():
            report.preserved.append(str(target))

    if include_model_artifacts is False:
        report.warnings.append("include_model_artifacts_false_ignored:runtime_cleanup_always_resets_training_artifacts")
    if include_market_storage is False:
        report.warnings.append("include_market_storage_false_ignored:runtime_cleanup_always_resets_market_storage")

    remove_dirs = [root / rel for rel in MARKET_DATA_DIRS]
    remove_dirs.extend(root / rel for rel in TRAINING_ARTIFACT_DIRS)

    for target in _iter_existing(remove_dirs):
        _remove_path(target, root, report, dry_run=dry_run)

    for pattern in GENERATED_DATA_GLOBS:
        for target in root.glob(str(pattern)):
            if target.suffix.lower() in {".json", ".md", ".log"}:
                continue
            _remove_path(target, root, report, dry_run=dry_run)

    for pattern in CACHE_FILE_GLOBS:
        for target in root.rglob(pattern):
            _remove_path(target, root, report, dry_run=dry_run)

    for pattern in CACHE_DIR_GLOBS:
        for target in sorted(root.rglob(pattern), key=lambda item: len(item.parts), reverse=True):
            _remove_path(target, root, report, dry_run=dry_run)

    for rel in RECREATE_DIRS:
        target = root / rel
        report.recreated.append(str(target))
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)

    for rel in PACKAGE_MARKERS:
        target = root / rel
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
        if str(target) not in report.recreated:
            report.recreated.append(str(target))

    return report


def _print_report(report: CleanRuntimeStateReport) -> None:
    print(f"Project root: {report.project_root}")
    print(f"Dry run: {report.dry_run}")
    print("Mode: clear market/training artifacts; preserve runtime JSON/MD/log evidence")

    if report.preserved:
        print("Preserved runtime evidence/context:")
        for item in report.preserved:
            print(f" - {item}")

    if report.removed:
        print("Removed rebuildable market/training artifacts:")
        for item in report.removed:
            print(f" - {item}")
    else:
        print("Removed: none")

    print("Recreated/verified directories:")
    for item in report.recreated:
        print(f" - {item}")

    if report.warnings:
        print("Warnings:", file=sys.stderr)
        for warning in report.warnings:
            print(f" - {warning}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean Auto_Trader rebuildable market/training artifacts while preserving "
            "runtime JSON/MD/log evidence."
        )
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Auto_Trader project root. Defaults to this repository.")
    parser.add_argument(
        "--include-model-artifacts",
        action="store_true",
        help="Backward-compatible no-op: model artifacts are always reset by this script.",
    )
    parser.add_argument(
        "--include-market-storage",
        action="store_true",
        help="Backward-compatible no-op: market storage is always reset by this script.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = clean_runtime_state(
        args.project_root,
        include_model_artifacts=bool(args.include_model_artifacts),
        include_market_storage=bool(args.include_market_storage),
        dry_run=bool(args.dry_run),
    )
    _print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
