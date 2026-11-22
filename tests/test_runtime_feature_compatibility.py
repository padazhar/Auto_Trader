from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib

from core.artifact_contract import artifact_paths, save_json, write_manifest
from inference.readiness import anchor_model_readiness, model_registry_readiness
from inference.trade_decision_logic import TradeDecisionLogic


class _DummyModel:
    def predict(self, X):
        return [1] * len(X)

    def predict_proba(self, X):
        import numpy as np

        return np.array([[0.1, 0.2, 0.7] for _ in range(len(X))], dtype=float)


def _paths(tmp_path: Path) -> SimpleNamespace:
    project_root = tmp_path / "repo"
    model_storage_dir = project_root / "data" / "model_storage"
    runtime_dir = project_root / "data" / "runtime"
    logs_dir = project_root / "data" / "logs"

    model_storage_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    registry_path = runtime_dir / "training_registry.json"
    registry_path.write_text("{}", encoding="utf-8")

    config = SimpleNamespace(
        feature_store=SimpleNamespace(recipe_version="feature_store_v2"),
        frames=SimpleNamespace(anchor_interval="15m"),
        inference=SimpleNamespace(
            tail_rows=100,
            minimum_models_required=2,
            quality=SimpleNamespace(
                min_rows=10,
                min_precision_macro=0.1,
                min_roc_auc_ovr=0.5,
                min_balanced_accuracy=0.34,
            ),
            weights=SimpleNamespace(rf=0.5, gbm=0.4, dqn=0.1),
            dqn=SimpleNamespace(gray_band_low=0.45, gray_band_high=0.55),
            ev=SimpleNamespace(
                threshold=0.0,
                meta_threshold=0.55,
                dqn_adjustment_strength=0.15,
                strong_anchor_probability=0.62,
                weak_anchor_probability=0.55,
                weak_anchor_lower_tf_scale=0.60,
            ),
            consensus=SimpleNamespace(
                enter_threshold=0.18,
                flip_threshold=0.12,
                exit_band=0.06,
                direction_score_threshold=0.05,
                require_agreement=True,
            ),
        ),
        trading=SimpleNamespace(
            fees_bps=3.0,
            slippage_bps=2.0,
            min_profit_atr_mult=0.30,
            ev_threshold=0.0,
            meta_threshold=0.55,
            min_atr_fraction=0.0,
            short_probability_threshold=0.60,
            long_probability_threshold=0.55,
        ),
        risk=SimpleNamespace(
            account_balance_usd=1000.0,
            risk_per_trade=0.01,
            sl_atr_mult=1.5,
            tp_atr_mult=2.0,
            max_notional_usd=5000.0,
            min_notional_usd=25.0,
        ),
    )

    return SimpleNamespace(
        project_root=project_root,
        model_storage_dir=model_storage_dir,
        runtime_dir=runtime_dir,
        logs_dir=logs_dir,
        registry_path=registry_path,
        config=config,
        log_file_path=lambda name: logs_dir / f"{name}.log",
    )


def _write_incompatible_rf_manifest(paths, *, candidate_id: str = "cand-bad-feature") -> Path:
    artifacts = artifact_paths(paths, "rf", "BTCUSDT", "15m", stage="candidate", candidate_id=candidate_id)
    artifacts["model"].parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(_DummyModel(), artifacts["model"])
    save_json(artifacts["feature_columns"], {"feature_columns": ["feat_a", "feat_b"]})
    save_json(
        artifacts["metrics"],
        {
            "precision_macro": 0.25,
            "roc_auc_ovr": 0.62,
            "balanced_accuracy": 0.48,
            "rows_total": 50,
        },
    )

    write_manifest(
        paths,
        model_type="rf",
        symbol="BTCUSDT",
        interval="15m",
        rows_used=50,
        feature_count=2,
        artifacts=artifacts,
        metrics={
            "precision_macro": 0.25,
            "roc_auc_ovr": 0.62,
            "balanced_accuracy": 0.48,
            "rows_total": 50,
        },
        extra={
            "feature_set": {
                "version": "feature_store_v1",
                "recipe_version": "feature_store_v1",
                "signature": "not-a-real-signature",
                "columns": ["feat_a", "feat_b"],
                "column_count": 2,
                "optional_context_fields": [],
            },
            "model_role": "direction",
        },
    )
    return artifacts["manifest"]


def _write_registry_entry(paths, *, model_type: str, symbol: str, interval: str, manifest_path: Path) -> None:
    payload = {
        f"{model_type}:{symbol}:{interval}": {
            "status": "success",
            "artifacts": {
                "manifest": manifest_path.relative_to(paths.project_root).as_posix(),
            },
        }
    }
    paths.registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_model_registry_readiness_rejects_incompatible_feature_manifest(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest_path = _write_incompatible_rf_manifest(paths)
    _write_registry_entry(paths, model_type="rf", symbol="BTCUSDT", interval="15m", manifest_path=manifest_path)

    readiness = model_registry_readiness(paths, model_type="rf", symbol="BTCUSDT", interval="15m")

    assert readiness.ready is False
    assert readiness.status == "success"
    assert "feature_incompatible" in readiness.reason


def test_trade_decision_logic_holds_when_anchor_artifact_is_feature_incompatible(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manifest_path = _write_incompatible_rf_manifest(paths)
    _write_registry_entry(paths, model_type="rf", symbol="BTCUSDT", interval="15m", manifest_path=manifest_path)

    logic = TradeDecisionLogic(paths)

    monkeypatch.setattr(
        "inference.trade_decision_logic.anchor_model_readiness",
        lambda paths, symbol, interval: anchor_model_readiness(
            paths,
            symbol=symbol,
            interval=interval,
            required_models=("rf",),
        ),
    )

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.reason.startswith("anchor_training_incomplete:")
    assert "feature_incompatible" in decision.reason
