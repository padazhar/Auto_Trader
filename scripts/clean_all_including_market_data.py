from __future__ import annotations

from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

# Purpose of this script:
#   Fully wipe generated data under data/. This is the destructive reset. It
#   removes market data, model/training artifacts, runtime JSON, reports, logs,
#   trade history, CSV/DB/parquet artifacts, then recreates the empty scaffold.
#
# For a safer requalification reset that preserves runtime JSON/MD/log evidence,
# use scripts/clean_runtime_state.py instead.

# Keep only Python package markers and code files that are part of the repository.
KEEP_RELATIVE_PATHS = {
    Path("__init__.py"),
    Path("model_storage") / "__init__.py",
}

# Directories expected by the application after cleanup.
RECREATE_DIRS = [
    DATA_ROOT / "logs",
    DATA_ROOT / "runtime",
    DATA_ROOT / "runtime" / "monitoring",
    DATA_ROOT / "runtime" / "training_progress",
    DATA_ROOT / "market_storage",
    DATA_ROOT / "model_storage" / "random_forest_models",
    DATA_ROOT / "model_storage" / "gradient_boost_models",
    DATA_ROOT / "model_storage" / "reinforcement_agents",
    DATA_ROOT / "reports",
    DATA_ROOT / "trade_history",
]


def _should_keep(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rel = path.relative_to(DATA_ROOT)
    except ValueError:
        return False
    return rel in KEEP_RELATIVE_PATHS



def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8 compatibility fallback; harmless on 3.12+
            if path.exists() or path.is_symlink():
                path.unlink()



def _cleanup_data_root() -> None:
    if not DATA_ROOT.exists():
        return

    for child in DATA_ROOT.iterdir():
        rel = child.relative_to(DATA_ROOT)
        if rel in KEEP_RELATIVE_PATHS:
            continue

        if child.is_dir() and not child.is_symlink():
            if child.name == "model_storage":
                for nested in child.iterdir():
                    nested_rel = nested.relative_to(DATA_ROOT)
                    if nested_rel in KEEP_RELATIVE_PATHS:
                        continue
                    print(f"Removing: {nested}")
                    _remove_path(nested)
                continue

            print(f"Removing: {child}")
            _remove_path(child)
            continue

        if not _should_keep(child):
            print(f"Removing: {child}")
            _remove_path(child)



def _restore_scaffolding() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in RECREATE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)

    for rel_path in KEEP_RELATIVE_PATHS:
        full_path = DATA_ROOT / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.touch(exist_ok=True)



def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print("Removing all generated data artifacts while preserving configuration and code...")
    _cleanup_data_root()
    _restore_scaffolding()
    print("Full cleanup complete. Market data, models, runtime state, reports, trade history, logs, and generated CSV/DB artifacts were removed.")


if __name__ == "__main__":
    main()
