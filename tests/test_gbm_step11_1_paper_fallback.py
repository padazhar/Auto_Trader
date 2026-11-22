from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core.artifact_contract import artifact_paths, incumbent_pointer_path, save_json
from training.gbm_training_engine import GBMTrainer


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


class _PromotionCfg(SimpleNamespace):
    enabled: bool = True
    allow_initial_promotion: bool = True
    require_outperform_incumbent: bool = True
    min_fold_count: int = 2
    min_score_improvement: float = 0.0
    min_metric_improvement: float = 0.0
    max_metric_regression: float = 1.0
    min_precision_macro: float = 0.0
    min_roc_auc_ovr: float = 0.0
    min_balanced_accuracy: float = 0.0
    min_precision: float = 0.0
    min_roc_auc: float = 0.0


def _paths(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "repo"
    model_storage_dir = root / "data" / "model_storage"
    runtime_dir = root / "data" / "runtime"
    logs_dir = root / "data" / "logs"
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        project_root=root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        config=SimpleNamespace(
            feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
            training=SimpleNamespace(promotion=_PromotionCfg()),
        ),
    )


def _write_active_gbm_role(paths: SimpleNamespace, *, symbol: str = "BTCUSDT", interval: str = "15m", role: str = "direction") -> Path:
    artifacts = artifact_paths(paths, "gbm", symbol, interval, model_role=role, stage="incumbent")
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["model"].write_bytes(b"model")
    save_json(artifacts["feature_columns"], {"feature_columns": ["close"]})
    save_json(artifacts["metrics"], {"roc_auc_ovr": 0.7, "precision_macro": 0.4, "balanced_accuracy": 0.55})
    payload = {
        "schema_version": 4,
        "model_type": "gbm",
        "model_role": role,
        "symbol": symbol,
        "interval": interval,
        "artifacts": {
            "model": artifacts["model"].relative_to(paths.project_root).as_posix(),
            "feature_columns": artifacts["feature_columns"].relative_to(paths.project_root).as_posix(),
            "metrics": artifacts["metrics"].relative_to(paths.project_root).as_posix(),
        },
        "feature_set": {"version": "feature_store_v2", "signature": "sig", "columns": ["close"]},
        "lifecycle": {"stage": "incumbent", "candidate_id": "old"},
        "timestamps": {"created_at": "2026-05-01T00:00:00+00:00"},
    }
    save_json(artifacts["manifest"], payload)
    save_json(
        incumbent_pointer_path(paths, "gbm", symbol, interval, model_role=role),
        {
            "schema_version": 1,
            "model_type": "gbm",
            "model_role": role,
            "symbol": symbol,
            "interval": interval,
            "candidate_id": "old",
            "manifest_path": artifacts["manifest"].relative_to(paths.project_root).as_posix(),
        },
    )
    return artifacts["manifest"]


def _paper_bootstrap_direction_decision() -> dict:
    return {
        "attempted": True,
        "approved": True,
        "promoted": False,
        "has_incumbent": False,
        "reasons": [],
        "paper_bootstrap_only": True,
        "paper_execution_only": True,
        "paper_evidence_status": "paper_bootstrap_execution_ready",
        "paper_evidence_sufficient": False,
        "deployment_stage": "paper_bootstrap",
        "live_eligible": False,
    }


def test_step11_1_allows_initial_paper_direction_when_meta_has_only_insufficient_fold_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_paths(tmp_path))

    def _evaluate(**kwargs):
        if kwargs.get("model_role") == "direction":
            return _paper_bootstrap_direction_decision()
        return {
            "attempted": True,
            "approved": False,
            "promoted": False,
            "has_incumbent": False,
            "reasons": ["insufficient_fold_count:1"],
            "deployment_stage": "candidate",
            "paper_execution_only": False,
            "live_eligible": False,
        }

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _evaluate)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="paper-direction-only",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.7, "precision_macro": 0.4, "balanced_accuracy": 0.55, "fold_count": 3},
        meta_manifest=tmp_path / "candidate_meta_manifest.json",
        meta_metrics={"roc_auc": 0.7, "precision": 0.4, "balanced_accuracy": 0.55, "fold_count": 1},
        promote_requested=True,
    )

    assert decision["approved"] is True
    assert decision["paper_bootstrap_only"] is True
    assert decision["paper_execution_only"] is True
    assert decision["deployment_stage"] == "paper_bootstrap"
    assert decision["reasons"] == []
    assert decision["live_eligible"] is False
    assert decision["meta"]["approved"] is False
    assert decision["meta"]["paper_direction_fallback_allowed"] is True
    assert decision["meta"]["blocking_for_paper_bundle"] is False
    assert decision["governance"]["paper_eligible"] is True
    assert decision["governance"]["paper_execution_authorized"] is True
    assert decision["governance"]["live_eligible"] is False
    assert decision["governance"]["meta_required_for_live"] is True
    assert decision["governance"]["meta_rejected_for_paper_direction_fallback"] is True
    assert decision["governance"]["approval_tier"] == "paper_only_direction_fallback"


def test_step11_1_does_not_bypass_meta_quality_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_paths(tmp_path))

    def _evaluate(**kwargs):
        if kwargs.get("model_role") == "direction":
            return _paper_bootstrap_direction_decision()
        return {
            "attempted": True,
            "approved": False,
            "promoted": False,
            "has_incumbent": False,
            "reasons": ["insufficient_fold_count:1", "below_min_roc_auc"],
            "deployment_stage": "candidate",
            "paper_execution_only": False,
            "live_eligible": False,
        }

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _evaluate)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="rejected-meta-quality",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.7, "precision_macro": 0.4, "balanced_accuracy": 0.55, "fold_count": 3},
        meta_manifest=tmp_path / "candidate_meta_manifest.json",
        meta_metrics={"roc_auc": 0.4, "precision": 0.4, "balanced_accuracy": 0.55, "fold_count": 1},
        promote_requested=True,
    )

    assert decision["approved"] is False
    assert "meta:insufficient_fold_count:1" in decision["reasons"]
    assert "meta:below_min_roc_auc" in decision["reasons"]
    assert decision["governance"].get("meta_rejected_for_paper_direction_fallback") is not True


def test_step11_1_does_not_replace_existing_gbm_incumbent_with_direction_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    incumbent = _write_active_gbm_role(paths)
    trainer = GBMTrainer(paths)

    def _evaluate(**kwargs):
        if kwargs.get("model_role") == "direction":
            decision = _paper_bootstrap_direction_decision()
            decision["has_incumbent"] = True
            return decision
        return {
            "attempted": True,
            "approved": False,
            "promoted": False,
            "has_incumbent": False,
            "reasons": ["insufficient_fold_count:1"],
            "deployment_stage": "candidate",
            "paper_execution_only": False,
            "live_eligible": False,
        }

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _evaluate)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="incumbent-protected",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.7, "precision_macro": 0.4, "balanced_accuracy": 0.55, "fold_count": 3},
        meta_manifest=tmp_path / "candidate_meta_manifest.json",
        meta_metrics={"roc_auc": 0.7, "precision": 0.4, "balanced_accuracy": 0.55, "fold_count": 1},
        promote_requested=True,
    )

    assert decision["approved"] is False
    assert decision["incumbent_retained"] is True
    assert decision["active_incumbents_before"]["direction"] == str(incumbent)
    assert "meta:insufficient_fold_count:1" in decision["reasons"]
