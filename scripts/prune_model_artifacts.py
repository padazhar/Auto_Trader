from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.artifact_contract import MODEL_DIR_MAP, prune_candidate_artifacts
from core.path_manager import PathManager


MODEL_ROLES: dict[str, list[str | None]] = {
    "rf": [None],
    "gbm": [None, "direction", "meta"],
    "dqn": [None],
}


def _candidate_roots(paths: PathManager, model_type: str) -> list[tuple[str, str, str | None]]:
    """Discover symbol/interval/role slots that may contain candidates."""
    storage_root = paths.model_storage_dir / MODEL_DIR_MAP[model_type]
    if not storage_root.exists():
        return []
    slots: set[tuple[str, str, str | None]] = set()
    for symbol_dir in storage_root.iterdir():
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name.upper()
        for interval_dir in symbol_dir.iterdir():
            if not interval_dir.is_dir():
                continue
            interval = interval_dir.name
            if (interval_dir / "candidates").exists():
                slots.add((symbol, interval, None))
            if model_type == "gbm":
                for role in ("direction", "meta"):
                    if (interval_dir / role / "candidates").exists():
                        slots.add((symbol, interval, role))
            # RF/DQN candidates live directly below symbol/interval/candidates.
    return sorted(slots, key=lambda item: (item[0], item[1], str(item[2] or "")))


def run_prune(
    project_root: Path,
    *,
    apply: bool = False,
    model_type: str | None = None,
    symbol: str | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    paths = PathManager.from_project_root(project_root)
    model_types = [model_type] if model_type else ["rf", "gbm", "dqn"]
    results: list[dict[str, Any]] = []
    for mt in model_types:
        if symbol and interval:
            roles = MODEL_ROLES.get(mt, [None])
            slots = [(symbol.upper(), interval, role) for role in roles]
        else:
            slots = _candidate_roots(paths, mt)
        for slot_symbol, slot_interval, role in slots:
            if symbol and slot_symbol.upper() != symbol.upper():
                continue
            if interval and slot_interval != interval:
                continue
            results.append(
                prune_candidate_artifacts(
                    paths,
                    model_type=mt,
                    symbol=slot_symbol,
                    interval=slot_interval,
                    model_role=role,
                    dry_run=not apply,
                )
            )
    return {
        "applied": bool(apply),
        "project_root": str(project_root.resolve()),
        "results": results,
        "deleted_count": sum(len(result.get("deleted", [])) for result in results if isinstance(result, dict)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune stale Auto_Trader model candidate artifacts safely.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--apply", action="store_true", help="Delete files. Without this flag the script is dry-run only.")
    parser.add_argument("--model-type", choices=["rf", "gbm", "dqn"], default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--interval", default=None)
    args = parser.parse_args()
    payload = run_prune(
        Path(args.project_root),
        apply=bool(args.apply),
        model_type=args.model_type,
        symbol=args.symbol,
        interval=args.interval,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
