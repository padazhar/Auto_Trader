from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "FLIP_TO_LONG", "FLIP_TO_SHORT"}
READINESS_PREFIXES = ("anchor_training_incomplete", "insufficient_models", "active_manifest_missing", "candidate_artifact_missing")
PRICE_PREFIXES = ("price_unavailable", "price_missing", "market_price", "stale_price", "missing_price")
THRESHOLD_REASONS = {"negative_ev", "movement_filter", "weak_short_filter", "weak_long_filter", "model_disagreement", "regime_blocked"}


def _load_yaml_runtime_defaults(project_root: Path) -> dict[str, Any]:
    path = project_root / "config" / "system_settings.yaml"
    defaults = {
        "paper_validation_min_decisions": 100,
        "paper_validation_max_readiness_blocker_fraction": 0.35,
        "paper_validation_min_entry_candidates": 5,
    }
    if not path.exists():
        return defaults
    try:
        import yaml  # type: ignore
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
        if isinstance(runtime, dict):
            for key in defaults:
                if key in runtime:
                    defaults[key] = runtime[key]
    except Exception:
        pass
    return defaults


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                yield value


def _reason(row: dict[str, Any]) -> str:
    for key in ("rejection_reason", "execution_blocker", "reason", "execution_blocker_stage"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def _classify_reason(reason: str) -> str:
    normalized = str(reason or "unknown")
    if normalized.startswith(READINESS_PREFIXES):
        return "readiness"
    if normalized.startswith(PRICE_PREFIXES) or "price" in normalized:
        return "price"
    if normalized in THRESHOLD_REASONS:
        return "threshold"
    if "risk" in normalized or "exposure" in normalized or "cooldown" in normalized or "position" in normalized:
        return "risk_runtime"
    if normalized.startswith("ev_"):
        return "entry_candidate"
    return "other"


def analyze(project_root: Path) -> dict[str, Any]:
    runtime = _load_yaml_runtime_defaults(project_root)
    journal = project_root / "data" / "runtime" / "decision_journal.jsonl"
    reasons: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    stages: Counter[str] = Counter()
    total = 0
    entry_candidates = 0
    for row in _iter_jsonl(journal) or []:
        total += 1
        action = str(row.get("action") or row.get("effective_action") or "UNKNOWN")
        reason = _reason(row)
        category = _classify_reason(reason)
        actions[action] += 1
        reasons[reason] += 1
        categories[category] += 1
        if action in ENTRY_ACTIONS or category == "entry_candidate":
            entry_candidates += 1
        stage = row.get("execution_blocker_stage")
        if stage not in (None, ""):
            stages[str(stage)] += 1

    min_decisions = int(runtime.get("paper_validation_min_decisions", 100) or 100)
    max_readiness_fraction = float(runtime.get("paper_validation_max_readiness_blocker_fraction", 0.35) or 0.35)
    min_entry_candidates = int(runtime.get("paper_validation_min_entry_candidates", 5) or 5)
    readiness_fraction = (categories["readiness"] / total) if total else 0.0

    threshold_tuning_allowed = (
        total >= min_decisions
        and entry_candidates >= min_entry_candidates
        and readiness_fraction <= max_readiness_fraction
        and categories["price"] == 0
    )
    if total < min_decisions:
        recommendation = "collect_more_paper_decisions_before_tuning"
    elif categories["price"]:
        recommendation = "fix_price_pipeline_before_threshold_tuning"
    elif readiness_fraction > max_readiness_fraction:
        recommendation = "fix_model_readiness_before_threshold_tuning"
    elif entry_candidates < min_entry_candidates:
        recommendation = "fix_signal_generation_or_model_gates_before_threshold_tuning"
    else:
        recommendation = "threshold_tuning_allowed_use_small_paper_only_adjustments"

    return {
        "journal": str(journal),
        "total_decisions": total,
        "entry_candidates": entry_candidates,
        "readiness_blocker_fraction": readiness_fraction,
        "threshold_tuning_allowed": threshold_tuning_allowed,
        "recommendation": recommendation,
        "minimums": {
            "paper_validation_min_decisions": min_decisions,
            "paper_validation_max_readiness_blocker_fraction": max_readiness_fraction,
            "paper_validation_min_entry_candidates": min_entry_candidates,
        },
        "actions": dict(actions.most_common()),
        "categories": dict(categories.most_common()),
        "reasons": dict(reasons.most_common(25)),
        "stages": dict(stages.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze paper-trading rejection distribution before threshold tuning.")
    parser.add_argument("--project-root", default=".", help="Auto_Trader project root")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args(argv)
    result = analyze(Path(args.project_root).resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Paper rejection analysis")
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
