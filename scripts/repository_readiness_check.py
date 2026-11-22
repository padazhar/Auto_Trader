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

import yaml

from core.config_contract import validate_settings
from core.path_manager import PathManager


@dataclass(slots=True)
class ReadinessItem:
    """Single readiness check result."""

    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReadinessReport:
    """Structured repository readiness report."""

    project_root: str
    overall_status: str
    paper_trading_ready: bool
    live_candidate_ready: bool
    live_candidate_reason: str
    items: list[ReadinessItem]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into a JSON-serializable payload."""
        return {
            "project_root": self.project_root,
            "overall_status": self.overall_status,
            "paper_trading_ready": self.paper_trading_ready,
            "live_candidate_ready": self.live_candidate_ready,
            "live_candidate_reason": self.live_candidate_reason,
            "items": [asdict(item) for item in self.items],
        }


def _record(items: list[ReadinessItem], name: str, status: str, message: str, **details: Any) -> None:
    """Append a readiness item to the report."""
    items.append(ReadinessItem(name=name, status=status, message=message, details=dict(details)))


def _load_repo_yaml(project_root: Path) -> dict[str, Any]:
    """Load the canonical repository system settings YAML."""
    config_path = project_root / "config" / "system_settings.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("system_settings.yaml must load as a mapping")
    return payload


def _check_config(project_root: Path, items: list[ReadinessItem]) -> tuple[bool, Any | None, Any | None, dict[str, Any] | None]:
    """Validate config loading and governance-critical YAML visibility."""
    try:
        raw = _load_repo_yaml(project_root)
        config = validate_settings(raw)
        paths = PathManager.from_project_root(project_root)

        missing_blocks: list[str] = []
        if "feature_store" not in raw:
            missing_blocks.append("feature_store")
        if "regime" not in raw:
            missing_blocks.append("regime")
        if "exchange" not in raw:
            missing_blocks.append("exchange")
        if "training" not in raw or not isinstance(raw.get("training"), dict):
            missing_blocks.append("training")
        else:
            training = raw["training"]
            if "validation" not in training:
                missing_blocks.append("training.validation")
            if "promotion" not in training:
                missing_blocks.append("training.promotion")

        if missing_blocks:
            _record(
                items,
                "config_schema",
                "fail",
                "Governance-critical config blocks are missing.",
                missing_blocks=missing_blocks,
            )
            return False, None, None, raw

        _record(
            items,
            "config_schema",
            "pass",
            "Configuration schema loaded successfully and governance-critical blocks are explicit.",
            execution_mode=config.runtime.execution_mode,
            allow_unreconciled_live_mode=config.runtime.allow_unreconciled_live_mode,
            feature_recipe_version=config.feature_store.recipe_version,
            regime_enabled=config.regime.enabled,
            walk_forward_folds=config.training.validation.walk_forward_folds,
            reconciliation_enabled=config.exchange.position_reconciliation_enabled,
            reconciliation_required=config.exchange.require_reconciliation_for_live,
        )
        return True, config, paths, raw
    except Exception as exc:
        _record(items, "config_schema", "fail", "Configuration loading failed.", error=str(exc))
        return False, None, None, None


def _check_launcher(items: list[ReadinessItem]) -> None:
    """Verify launcher modes are present and parseable."""
    try:
        main_module = importlib.import_module("main")
        parser = main_module.build_arg_parser()
        mode_action = next((action for action in parser._actions if getattr(action, "dest", "") == "mode"), None)
        choices = sorted(list(mode_action.choices)) if mode_action is not None and getattr(mode_action, "choices", None) else []
        expected = {"update", "tech_rev", "trainer", "signal", "runtime", "backtest", "dashboard", "monitor", "live", "all"}
        choice_set = set(choices)
        missing = sorted(expected - choice_set)
        allowed_aliases = {"monitoring"}
        unsupported = sorted(choice_set - expected - allowed_aliases)
        if missing or unsupported:
            _record(
                items,
                "launcher_modes",
                "fail",
                "Launcher modes do not match the expected control surface.",
                choices=choices,
                missing=missing,
                unsupported=unsupported,
                allowed_aliases=sorted(allowed_aliases),
            )
            return
        _record(
            items,
            "launcher_modes",
            "pass",
            "Launcher modes are wired for combined and isolated process-role operation.",
            choices=choices,
            aliases=sorted(choice_set & allowed_aliases),
            isolated_process_roles=["runtime", "trainer"],
        )
    except Exception as exc:
        _record(items, "launcher_modes", "fail", "Launcher import or mode parsing failed.", error=str(exc))


def _check_training_governance(items: list[ReadinessItem]) -> None:
    """Verify training promotion and bundle hooks exist."""
    try:
        base_module = importlib.import_module("training.base_trainer")
        rf_module = importlib.import_module("training.rf_training_engine")
        gbm_module = importlib.import_module("training.gbm_training_engine")
        policy_module = importlib.import_module("training.dqn_training_engine")
        hmm_module = importlib.import_module("training.hmm_regime_training_engine")

        required = [
            hasattr(base_module.BaseTrainer, "save_training_bundle"),
            hasattr(base_module.BaseTrainer, "evaluate_candidate_promotion"),
            hasattr(rf_module, "train_rf_model"),
            hasattr(gbm_module, "train_gbm_model"),
            hasattr(policy_module, "train_policy_model"),
            hasattr(hmm_module, "train_hmm_model"),
        ]
        if not all(required):
            _record(items, "training_governance", "fail", "Training governance hooks are incomplete.")
            return

        _record(
            items,
            "training_governance",
            "pass",
            "Candidate bundle, promotion, and model training entrypoints are present.",
            policy_entrypoint="train_policy_model",
            legacy_policy_alias="train_dqn_model",
            hmm_candidate_entrypoint="train_hmm_model",
        )
    except Exception as exc:
        _record(items, "training_governance", "fail", "Training governance import failed.", error=str(exc))


def _check_feature_compatibility(items: list[ReadinessItem]) -> None:
    """Verify feature compatibility enforcement exists at inference load time."""
    try:
        artifact_module = importlib.import_module("core.artifact_contract")
        loader_module = importlib.import_module("inference.model_artifact_loader")

        if not hasattr(artifact_module, "validate_manifest_feature_compatibility"):
            _record(items, "feature_compatibility", "fail", "Manifest feature compatibility validator is missing.")
            return
        if not hasattr(loader_module.ModelArtifactLoader, "_ensure_manifest_compatible"):
            _record(items, "feature_compatibility", "fail", "Inference loader compatibility guard is missing.")
            return

        _record(
            items,
            "feature_compatibility",
            "pass",
            "Inference artifact loading enforces feature compatibility.",
        )
    except Exception as exc:
        _record(items, "feature_compatibility", "fail", "Feature compatibility validation import failed.", error=str(exc))


def _check_canonical_futures_context(items: list[ReadinessItem]) -> None:
    """Verify funding/open-interest ingestion and fusion ownership hooks exist."""
    try:
        market_loader = importlib.import_module("features.market_loader")
        fusion_module = importlib.import_module("features.timeframe_fusion")
        controller_module = importlib.import_module("runners.market_data_controller")

        required = [
            hasattr(market_loader, "refresh_funding_rate_data"),
            hasattr(market_loader, "refresh_open_interest_data"),
            hasattr(fusion_module, "_load_optional_context_frame"),
            hasattr(controller_module.MarketDataController, "update_optional_context_for_symbol"),
        ]
        if not all(required):
            _record(items, "canonical_futures_context", "fail", "Canonical futures-context ownership is incomplete.")
            return

        _record(
            items,
            "canonical_futures_context",
            "pass",
            "Funding/open-interest ingestion and fusion ownership hooks are present.",
        )
    except Exception as exc:
        _record(items, "canonical_futures_context", "fail", "Canonical futures-context import failed.", error=str(exc))


def _check_exchange_reconciliation(items: list[ReadinessItem]) -> None:
    """Verify exchange reconciliation exists and is wired into the live path."""
    try:
        recon_module = importlib.import_module("execution.exchange_reconciliation")
        portfolio_module = importlib.import_module("execution.portfolio_state")
        risk_module = importlib.import_module("execution.risk_engine")
        live_module = importlib.import_module("runners.live_trading_controller")

        required = [
            hasattr(recon_module, "BinanceFuturesReconciler"),
            hasattr(portfolio_module.PortfolioStateStore, "record_reconciliation"),
            hasattr(risk_module.RiskEngine, "live_authority_summary"),
            hasattr(live_module.LiveTradingController, "_run_reconciliation")
            or hasattr(live_module.LiveTradingController, "_maybe_reconcile"),
        ]
        if not all(required):
            _record(items, "exchange_reconciliation", "fail", "Exchange reconciliation wiring is incomplete.")
            return

        _record(
            items,
            "exchange_reconciliation",
            "pass",
            "Exchange reconciliation foundation exists and is wired into live runtime authority.",
        )
    except Exception as exc:
        _record(items, "exchange_reconciliation", "fail", "Exchange reconciliation import failed.", error=str(exc))


def _check_control_plane_and_live_guard(items: list[ReadinessItem]) -> None:
    """Verify pause authority, control-plane summary, and live-mode guard hooks exist."""
    try:
        control_module = importlib.import_module("monitoring.control_plane_store")
        risk_module = importlib.import_module("execution.risk_engine")
        portfolio_module = importlib.import_module("execution.portfolio_state")
        live_module = importlib.import_module("runners.live_trading_controller")

        required = [
            hasattr(control_module.ControlPlaneStore, "pause_state"),
            hasattr(control_module.ControlPlaneStore, "_live_runtime_summary"),
            hasattr(risk_module.RiskEngine, "live_authority_summary"),
            hasattr(portfolio_module.PortfolioStateStore, "mark_authority_mode"),
            hasattr(live_module.LiveTradingController, "loop_once"),
        ]
        if not all(required):
            _record(items, "control_plane_and_live_guard", "fail", "Control-plane or live guard hooks are incomplete.")
            return

        _record(
            items,
            "control_plane_and_live_guard",
            "pass",
            "Pause authority, live guard, and live runtime summary hooks are present.",
        )
    except Exception as exc:
        _record(items, "control_plane_and_live_guard", "fail", "Control-plane or live guard import failed.", error=str(exc))


def _check_operator_naming(items: list[ReadinessItem]) -> None:
    """Verify operator-facing policy naming is cleaned up while compatibility is preserved."""
    try:
        naming_module = importlib.import_module("core.model_naming")
        display_name = naming_module.model_display_name("dqn")
        short_label = naming_module.model_short_label("dqn")
        canonical = naming_module.canonical_model_type("confirmatory_policy_classifier")

        if display_name != "Confirmatory Policy" or short_label != "P" or canonical != "dqn":
            _record(
                items,
                "operator_naming",
                "fail",
                "Operator naming cleanup is inconsistent.",
                display_name=display_name,
                short_label=short_label,
                canonical=canonical,
            )
            return

        _record(
            items,
            "operator_naming",
            "pass",
            "Operator-facing policy naming is cleaned up while legacy compatibility remains intact.",
            display_name=display_name,
            short_label=short_label,
            canonical=canonical,
        )
    except Exception as exc:
        _record(items, "operator_naming", "fail", "Operator naming import failed.", error=str(exc))


def run_readiness_check(project_root: Path | str | None = None) -> ReadinessReport:
    """Run a repository-level readiness check."""
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    items: list[ReadinessItem] = []

    config_ok, config, _paths, raw = _check_config(root, items)
    _check_launcher(items)
    _check_training_governance(items)
    _check_feature_compatibility(items)
    _check_canonical_futures_context(items)
    _check_exchange_reconciliation(items)
    _check_control_plane_and_live_guard(items)
    _check_operator_naming(items)

    statuses = [item.status for item in items]
    overall_status = "pass"
    if any(status == "fail" for status in statuses):
        overall_status = "fail"
    elif any(status == "warn" for status in statuses):
        overall_status = "warn"

    paper_trading_ready = bool(config_ok and overall_status == "pass")

    live_candidate_ready = False
    live_candidate_reason = (
        "Repository defaults remain paper-oriented. Enable exchange reconciliation, provide exchange credentials, "
        "and complete staged operational burn-in before using live mode."
    )
    if config is not None:
        exchange_cfg = config.exchange
        runtime_cfg = config.runtime
        has_exchange_secrets = bool(str(exchange_cfg.api_key or "").strip()) and bool(str(exchange_cfg.api_secret or "").strip())
        if (
            overall_status == "pass"
            and runtime_cfg.execution_mode == "live"
            and exchange_cfg.position_reconciliation_enabled
            and exchange_cfg.require_reconciliation_for_live
            and has_exchange_secrets
        ):
            live_candidate_ready = True
            live_candidate_reason = (
                "Live-candidate prerequisites are configured. Exchange reconciliation is enabled, but staged rollout "
                "and operational validation are still required before real-capital deployment."
            )

    return ReadinessReport(
        project_root=str(root),
        overall_status=overall_status,
        paper_trading_ready=paper_trading_ready,
        live_candidate_ready=live_candidate_ready,
        live_candidate_reason=live_candidate_reason,
        items=items,
    )


def main() -> int:
    """CLI entrypoint for repository readiness validation."""
    parser = argparse.ArgumentParser(description="Repository readiness check")
    parser.add_argument("--project-root", default=None, help="Repository root to validate")
    args = parser.parse_args()

    report = run_readiness_check(args.project_root)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.overall_status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
