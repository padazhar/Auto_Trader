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
    min_fold_count: int = 1
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


def test_gbm_rejected_candidate_reports_incumbent_retained(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    paths = _paths(tmp_path)
    incumbent = _write_active_gbm_role(paths)
    trainer = GBMTrainer(paths)

    def _reject(**kwargs):
        return {
            "attempted": True,
            "approved": False,
            "promoted": False,
            "has_incumbent": True,
            "reasons": ["candidate_score_not_improved"],
            "candidate_score": 0.4,
            "incumbent_score": 0.5,
        }

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _reject)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="new",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.6, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 3},
        meta_manifest=None,
        meta_metrics=None,
        promote_requested=True,
    )

    assert decision["approved"] is False
    assert decision["promoted"] is False
    assert decision["incumbent_retained"] is True
    assert decision["initial_promotion"] is False
    assert decision["active_incumbents_before"]["direction"] == str(incumbent)
    assert decision["governance"]["incumbent_retained"] is True


def test_gbm_initial_promotion_is_explicit_when_no_incumbent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_paths(tmp_path))

    def _approve(**kwargs):
        return {
            "attempted": True,
            "approved": True,
            "promoted": False,
            "has_incumbent": False,
            "reasons": [],
            "candidate_score": 0.6,
            "incumbent_score": None,
        }

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _approve)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="first",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.6, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 3},
        meta_manifest=None,
        meta_metrics={"skipped": True, "skip_reason": "meta_not_enough_rows_after_preparation"},
        promote_requested=True,
    )

    assert decision["approved"] is True
    assert decision["initial_promotion"] is True
    assert decision["incumbent_retained"] is False
    assert decision["governance"]["paper_eligible"] is True
    assert decision["governance"]["live_eligible"] is False
    assert decision["governance"]["approval_tier"] == "paper_only_direction_fallback"


def test_gbm_initial_paper_bootstrap_is_paper_eligible_but_not_live(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _DummyLogger())
    trainer = GBMTrainer(_paths(tmp_path))

    def _approve_paper_bootstrap(**kwargs):
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

    monkeypatch.setattr(trainer, "evaluate_candidate_promotion", _approve_paper_bootstrap)
    decision = trainer._bundle_promotion_decision(
        symbol="BTCUSDT",
        interval="15m",
        candidate_id="paper-first",
        direction_manifest=tmp_path / "candidate_direction_manifest.json",
        direction_metrics={"roc_auc_ovr": 0.6, "precision_macro": 0.3, "balanced_accuracy": 0.5, "fold_count": 3},
        meta_manifest=None,
        meta_metrics={"skipped": True, "skip_reason": "meta_not_enough_rows_after_preparation"},
        promote_requested=True,
    )

    assert decision["approved"] is True
    assert decision["paper_bootstrap_only"] is True
    assert decision["paper_execution_only"] is True
    assert decision["deployment_stage"] == "paper_bootstrap"
    assert decision["governance"]["paper_eligible"] is True
    assert decision["governance"]["paper_execution_authorized"] is True
    assert decision["governance"]["live_eligible"] is False
    assert decision["governance"]["approval_tier"] == "paper_bootstrap_execution_only"
