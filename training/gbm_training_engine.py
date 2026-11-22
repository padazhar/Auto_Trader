from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core.artifact_contract import (
    delete_incumbent_pointer,
    gbm_combined_manifest_path,
    gbm_role_artifact_paths,
    incumbent_pointer_path,
    load_json,
    promote_candidate_manifest,
    prune_related_candidate_artifacts,
    resolve_active_manifest_path,
    save_json,
    write_gbm_combined_manifest,
)
from core.result_contracts import TrainerResult
from features.dataset_preparation import WalkForwardFold, prepare_supervised_dataset, validate_target_class_support
from features.label_builder import (
    build_direction_labels,
    build_meta_label_from_direction,
    get_class_distribution,
    labels_config_value,
)
from training.base_trainer import BaseTrainer


class GBMTrainer(BaseTrainer):
    """Gradient-boost trainer with leakage-safe meta labeling and bundle-level promotion."""

    model_type = "gbm"

    @staticmethod
    def _cfg_value(model_cfg, interval: str, key: str, default: Any = None) -> Any:
        overrides = getattr(model_cfg, "interval_overrides", {}) or {}
        interval_payload = overrides.get(interval, {}) if isinstance(overrides, dict) else {}
        if key in interval_payload and interval_payload[key] is not None:
            return interval_payload[key]
        return getattr(model_cfg, key, default)

    @staticmethod
    def _metrics_multiclass(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, Any]:
        metrics = {
            "accuracy": BaseTrainer.safe_float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": BaseTrainer.safe_float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": BaseTrainer.safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": BaseTrainer.safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": BaseTrainer.safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        if y_prob is not None:
            try:
                classes = sorted(pd.Series(y_true).dropna().astype(int).unique().tolist())
                from sklearn.preprocessing import label_binarize

                y_bin = label_binarize(y_true, classes=classes)
                metrics["roc_auc_ovr"] = BaseTrainer.safe_float(
                    roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                )
            except Exception:
                metrics["roc_auc_ovr"] = None
            metrics.update(BaseTrainer.calibration_quality_report(y_true, y_prob, class_labels=[0, 1, 2]))
        else:
            metrics["roc_auc_ovr"] = None
            metrics.update({"calibration": {"available": False, "reason": "probabilities_unavailable"}})
        metrics.update(BaseTrainer.directional_quality_report(y_true, y_pred, class_encoding={0: "short", 1: "hold", 2: "long"}))
        return metrics

    @staticmethod
    def _metrics_binary(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, Any]:
        metrics = {
            "accuracy": BaseTrainer.safe_float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": BaseTrainer.safe_float(balanced_accuracy_score(y_true, y_pred)),
            "f1": BaseTrainer.safe_float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": BaseTrainer.safe_float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": BaseTrainer.safe_float(recall_score(y_true, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = BaseTrainer.safe_float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = None
            metrics.update(BaseTrainer.calibration_quality_report(y_true, y_prob, class_labels=[0, 1]))
        else:
            metrics["roc_auc"] = None
            metrics.update({"calibration": {"available": False, "reason": "probabilities_unavailable"}})
        return metrics

    def _prepare_dataset_with_fallback(self, fused: pd.DataFrame, interval: str, cfg, target_column: str) -> tuple[dict, str]:
        validation_cfg = self.paths.config.training.validation
        exclude_columns = {"rf_label", "dqn_action", "future_return", "thr_dyn", "meta_label", "direction_oos_pred"}
        common_kwargs = dict(
            target_column=target_column,
            exclude_columns=exclude_columns,
            timestamp_column="timestamp",
            valid_fraction=float(self._cfg_value(cfg, interval, "test_size", 0.2) or 0.2),
            min_rows_required=int(self._cfg_value(cfg, interval, "min_rows", 100)),
            passthrough_columns=["future_return", "thr_dyn", "direction_oos_pred"],
            walk_forward_folds=int(validation_cfg.walk_forward_folds),
            walk_forward_purge_rows=int(validation_cfg.purge_rows),
            walk_forward_embargo_rows=int(validation_cfg.embargo_rows),
            feature_recipe_version=self.paths.config.feature_store.recipe_version,
            anchor_interval=interval,
            optional_context_fields=["funding_rate", "open_interest"],
        )
        try:
            return prepare_supervised_dataset(fused, **common_kwargs), "full_fused"
        except ValueError as exc:
            message = str(exc)
            if "constant" not in message and "feature" not in message:
                raise
            anchor_columns = [
                c
                for c in fused.columns
                if c in {"timestamp", target_column, "future_return", "thr_dyn", "direction_oos_pred"} or c.startswith(f"{interval}_")
            ]
            anchor_only = fused.loc[:, [c for c in anchor_columns if c in fused.columns]].copy()
            prepared = prepare_supervised_dataset(anchor_only, **common_kwargs)
            return prepared, "anchor_only"

    @staticmethod
    def _estimator_family(gbm_cfg) -> str:
        raw = str(getattr(gbm_cfg, "estimator", None) or "histgbm").strip().lower().replace("-", "_")
        aliases = {
            "histgbm": "histgbm",
            "hist_gradient_boosting": "histgbm",
            "histgradientboosting": "histgbm",
            "lightgbm": "lightgbm",
            "lgbm": "lightgbm",
        }
        family = aliases.get(raw)
        if family is None:
            raise ValueError(f"unsupported GBM estimator family: {raw}")
        return family

    def _estimator_training_contract(self, gbm_cfg, interval: str, random_state: int) -> dict[str, Any]:
        family = self._estimator_family(gbm_cfg)
        return {
            "estimator_family": family,
            "estimator_selector": "training.gbm.estimator",
            "rollback_family": "histgbm",
            "random_state": int(random_state),
            "params": {
                "n_estimators": int(self._cfg_value(gbm_cfg, interval, "n_estimators", 300) or 300),
                "learning_rate": float(self._cfg_value(gbm_cfg, interval, "learning_rate", 0.04) or 0.04),
                "max_depth": int(self._cfg_value(gbm_cfg, interval, "max_depth", 3) or 3),
                "min_samples_leaf": int(self._cfg_value(gbm_cfg, interval, "min_samples_leaf", 20) or 20),
            },
        }

    def _build_model(self, gbm_cfg, interval: str, random_state: int):
        family = self._estimator_family(gbm_cfg)
        n_estimators = int(self._cfg_value(gbm_cfg, interval, "n_estimators", 300) or 300)
        learning_rate = float(self._cfg_value(gbm_cfg, interval, "learning_rate", 0.04) or 0.04)
        max_depth = int(self._cfg_value(gbm_cfg, interval, "max_depth", 3) or 3)
        min_samples_leaf = int(self._cfg_value(gbm_cfg, interval, "min_samples_leaf", 20) or 20)
        if family == "histgbm":
            return HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_iter=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=int(random_state),
            )
        if family == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("training.gbm.estimator=lightgbm requires the lightgbm package") from exc
            n_jobs = getattr(gbm_cfg, "n_jobs", None)
            return LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_samples=min_samples_leaf,
                random_state=int(random_state),
                n_jobs=int(n_jobs) if n_jobs is not None else None,
                verbose=-1,
            )
        raise ValueError(f"unsupported GBM estimator family: {family}")

    def _evaluate_direction_fold(self, fold: WalkForwardFold, feature_columns: list[str], gbm_cfg, interval: str, random_state: int) -> tuple[dict[str, Any] | None, pd.Series | None]:
        X_train = fold.train[feature_columns]
        y_train = fold.train["direction_label"].astype(int)
        X_valid = fold.valid[feature_columns]
        y_valid = fold.valid["direction_label"].astype(int)
        if X_train.empty or X_valid.empty:
            return None, None
        if len(set(y_train.unique().tolist())) < 3 or len(set(y_valid.unique().tolist())) < 2:
            return None, None
        model = self._build_model(gbm_cfg, interval, random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        prob = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
        metrics = self._metrics_multiclass(y_valid, pred, prob)
        metrics.update(
            {
                "fold_index": int(fold.fold_index),
                "rows_train": int(len(X_train)),
                "rows_valid": int(len(X_valid)),
                "train_start_ts": fold.train_start_ts,
                "train_end_ts": fold.train_end_ts,
                "valid_start_ts": fold.valid_start_ts,
                "valid_end_ts": fold.valid_end_ts,
            }
        )
        return metrics, pd.Series(pred, index=fold.valid.index, dtype="float64")

    @staticmethod
    def _aggregate_fold_metrics(fold_metrics: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
        aggregate: dict[str, Any] = {"fold_count": len(fold_metrics), "fold_metrics": fold_metrics}
        for key in keys:
            values = [float(item[key]) for item in fold_metrics if item.get(key) is not None]
            aggregate[f"walk_forward_{key}_median"] = BaseTrainer.safe_float(np.median(values)) if values else None
            aggregate[f"walk_forward_{key}_min"] = BaseTrainer.safe_float(np.min(values)) if values else None
        return aggregate

    @staticmethod
    def _meta_skip_metrics(clean_meta: pd.DataFrame, *, reason: str, detail: str | None = None) -> dict[str, Any]:
        """Build a consistent skipped-meta payload for non-fatal meta-model exits."""
        y_meta = pd.to_numeric(clean_meta.get("meta_label"), errors="coerce") if "meta_label" in clean_meta.columns else pd.Series(dtype="float64")
        return {
            "rows_total": int(len(clean_meta)),
            "model_role": "meta",
            "model_operational_role": "meta_unavailable_optional",
            "approval_tier": "meta_skipped_direction_only_fallback",
            "live_blocking": False,
            "live_eligible": False,
            "skipped": True,
            "skip_reason": str(reason),
            "skip_detail": str(detail or ""),
            "class_distribution_train": {
                "0": int((y_meta == 0).sum()) if not y_meta.empty else 0,
                "1": int((y_meta == 1).sum()) if not y_meta.empty else 0,
            },
        }

    @staticmethod
    def _bundle_governance(direction_decision: dict[str, Any], meta_metrics: dict[str, Any] | None, approved: bool) -> dict[str, Any]:
        """Build explicit governance metadata for the GBM bundle.

        The current repository allows paper-safe direction-only fallback when the
        meta model is unavailable, but that fallback should not be treated as
        live-eligible. This helper makes that contract explicit in metrics and
        manifests without collapsing the bundle into a hard failure.
        """
        meta_available = bool(meta_metrics) and not bool((meta_metrics or {}).get("skipped", False))
        meta_skipped = bool((meta_metrics or {}).get("skipped", False))
        paper_bootstrap_only = bool(direction_decision.get("paper_execution_only", False))
        live_eligible = bool(approved) and meta_available and not paper_bootstrap_only
        paper_eligible = bool(approved)
        if bool(approved) and paper_bootstrap_only:
            approval_tier = "paper_bootstrap_execution_only"
            model_operational_role = "direction_paper_bootstrap"
        elif bool(approved) and meta_available:
            approval_tier = "full_bundle_live_candidate"
            model_operational_role = "direction_plus_meta"
        elif bool(approved) and meta_skipped:
            approval_tier = "paper_only_direction_fallback"
            model_operational_role = "direction_only_fallback"
        else:
            approval_tier = "candidate_rejected"
            model_operational_role = "direction_candidate_rejected"
        return {
            "meta_required_for_live": True,
            "direction_approved": bool(direction_decision.get("approved", False)),
            "meta_available": meta_available,
            "meta_skipped": meta_skipped,
            "paper_eligible": paper_eligible,
            "paper_bootstrap_only": paper_bootstrap_only,
            "paper_execution_authorized": bool(paper_eligible and paper_bootstrap_only),
            "live_eligible": live_eligible,
            "live_blocking": bool(approved) and (meta_skipped or paper_bootstrap_only),
            "approval_tier": approval_tier,
            "model_operational_role": model_operational_role,
        }

    def _prepare_direction_dataset(self, symbol: str, interval: str, fused: pd.DataFrame, gbm_cfg, labels_cfg) -> tuple[dict[str, Any], pd.DataFrame]:
        labeled = build_direction_labels(
            fused.loc[:, ~fused.columns.duplicated()].copy(),
            anchor_interval=interval,
            horizon=int(labels_config_value(labels_cfg, "horizon_bars", interval, 3)),
            atr_multiplier=float(labels_config_value(labels_cfg, "atr_multiplier", interval, 0.25)),
            atr_floor=float(labels_config_value(labels_cfg, "atr_floor", interval, 0.0004)),
            trading_cfg=getattr(self.paths.config, "trading", None),
        )
        prepared, dataset_variant = self._prepare_dataset_with_fallback(labeled, interval, gbm_cfg, "direction_label")
        feature_columns = list(prepared["feature_columns"])
        fold_metrics: list[dict[str, Any]] = []
        oos_predictions = pd.Series(np.nan, index=prepared["frame"].index, dtype="float64")
        random_state = self._cfg_value(gbm_cfg, interval, "random_state", self.paths.config.system.random_seed)
        for fold in prepared.get("walk_forward_splits", []):
            metrics, preds = self._evaluate_direction_fold(fold, feature_columns, gbm_cfg, interval, int(random_state))
            if metrics is None or preds is None:
                continue
            fold_metrics.append(metrics)
            oos_predictions.loc[preds.index] = preds.astype(float)
        prepared["frame"] = prepared["frame"].copy()
        prepared["frame"]["direction_oos_pred"] = oos_predictions
        prepared["dataset_variant"] = dataset_variant
        prepared["walk_forward_summary"] = self._aggregate_fold_metrics(
            fold_metrics,
            ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"],
        )
        if not fold_metrics:
            raise ValueError("direction walk-forward validation produced no usable folds")
        return prepared, prepared["frame"]

    def _train_direction_model(self, symbol: str, interval: str, fused: pd.DataFrame, gbm_cfg, labels_cfg, random_state: int, candidate_id: str) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
        prepared, labeled = self._prepare_direction_dataset(symbol, interval, fused, gbm_cfg, labels_cfg)
        feature_columns = list(prepared["feature_columns"])
        split = prepared["split"]
        X_train = split.train[feature_columns]
        y_train = split.train["direction_label"].astype(int)
        X_valid = split.valid[feature_columns]
        y_valid = split.valid["direction_label"].astype(int)
        if X_train.empty or X_valid.empty:
            raise ValueError("direction train/valid split produced an empty partition")
        label_validation_train = validate_target_class_support(
            y_train,
            min_classes=3,
            required_classes=[0, 1, 2],
            context="gbm_direction_train",
        )
        label_validation_valid = validate_target_class_support(
            y_valid,
            min_classes=2,
            context="gbm_direction_valid",
        )

        calibrated_model = self.fit_calibrated_classifier(self._build_model(gbm_cfg, interval, random_state), X_train, y_train, X_valid, y_valid)
        valid_pred = calibrated_model.predict(X_valid)
        valid_prob = calibrated_model.predict_proba(X_valid) if hasattr(calibrated_model, "predict_proba") else None
        metrics = self._metrics_multiclass(y_valid, valid_pred, valid_prob)
        metrics.update(
            {
                "rows_train": int(len(X_train)),
                "rows_valid": int(len(X_valid)),
                "rows_total": int(prepared["rows_total"]),
                "feature_count": int(len(feature_columns)),
                "class_distribution_train": get_class_distribution(y_train),
                "class_distribution_valid": get_class_distribution(y_valid),
                "model_role": "direction",
                "model_operational_role": "direction_primary",
                "approval_tier": "direction_candidate",
                "live_blocking": True,
                "live_eligible": False,
                "dataset_variant": prepared.get("dataset_variant", "full_fused"),
                "candidate_id": candidate_id,
                "calibration_method": "sigmoid_prefit",
                "validation_policy": "walk_forward_plus_tail_calibration",
                "estimator_family": self._estimator_family(gbm_cfg),
                "estimator_contract": self._estimator_training_contract(gbm_cfg, interval, random_state),
                "label_validation": {
                    "promotion_eligible": True,
                    "train": label_validation_train,
                    "valid": label_validation_valid,
                    "reasons": [],
                },
                "avg_win_return": BaseTrainer.safe_float(float(labeled.loc[labeled["future_return"] > 0, "future_return"].mean())) if "future_return" in labeled.columns else None,
                "avg_loss_return": BaseTrainer.safe_float(abs(float(labeled.loc[labeled["future_return"] < 0, "future_return"].mean()))) if "future_return" in labeled.columns else None,
                **prepared["walk_forward_summary"],
            }
        )
        bundle = {
            "model": calibrated_model,
            "feature_columns": feature_columns,
            "rows_total": int(prepared["rows_total"]),
            "feature_set": prepared["feature_set"],
            "validation_summary": prepared["walk_forward_summary"],
            "estimator_family": self._estimator_family(gbm_cfg),
            "estimator_contract": self._estimator_training_contract(gbm_cfg, interval, random_state),
        }
        return bundle, metrics, labeled

    def _evaluate_meta_fold(self, fold: WalkForwardFold, feature_columns: list[str], gbm_cfg, interval: str, random_state: int) -> dict[str, Any] | None:
        X_train = fold.train[feature_columns]
        y_train = fold.train["meta_label"].astype(int)
        X_valid = fold.valid[feature_columns]
        y_valid = fold.valid["meta_label"].astype(int)
        if X_train.empty or X_valid.empty or y_train.nunique() < 2 or y_valid.nunique() < 2:
            return None
        model = self._build_model(gbm_cfg, interval, random_state)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        prob = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = self._metrics_binary(y_valid, pred, prob)
        metrics.update(
            {
                "fold_index": int(fold.fold_index),
                "rows_train": int(len(X_train)),
                "rows_valid": int(len(X_valid)),
                "train_start_ts": fold.train_start_ts,
                "train_end_ts": fold.train_end_ts,
                "valid_start_ts": fold.valid_start_ts,
                "valid_end_ts": fold.valid_end_ts,
            }
        )
        return metrics

    def _train_meta_model(self, symbol: str, interval: str, labeled: pd.DataFrame, gbm_cfg, random_state: int, candidate_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        prepared_meta = labeled.copy()
        prepared_meta = prepared_meta.dropna(subset=["direction_oos_pred", "future_return", "thr_dyn"]).copy()
        if prepared_meta.empty:
            return None, {"rows_total": 0, "model_role": "meta", "skipped": True, "skip_reason": "no_oos_direction_predictions"}

        prepared_meta["meta_label"] = build_meta_label_from_direction(
            prepared_meta,
            direction_pred_col="direction_oos_pred",
            trading_cfg=self.paths.config.trading,
        )
        clean_meta = prepared_meta.dropna(subset=["meta_label"]).copy()
        if clean_meta.empty:
            return None, {"rows_total": 0, "model_role": "meta", "skipped": True, "skip_reason": "meta_labels_empty"}
        if clean_meta["meta_label"].astype(int).nunique() < 2:
            return None, {
                "rows_total": int(len(clean_meta)),
                "model_role": "meta",
                "skipped": True,
                "skip_reason": "degenerate_meta_labels",
                "class_distribution_train": {"0": int((clean_meta["meta_label"] == 0).sum()), "1": int((clean_meta["meta_label"] == 1).sum())},
            }

        try:
            prepared, dataset_variant = self._prepare_dataset_with_fallback(clean_meta, interval, gbm_cfg, "meta_label")
        except ValueError as exc:
            message = str(exc)
            lowered = message.lower()
            if "not enough rows after preparation" in lowered:
                return None, self._meta_skip_metrics(
                    clean_meta,
                    reason="meta_not_enough_rows_after_preparation",
                    detail=message,
                )
            if "feature" in lowered or "constant" in lowered:
                return None, self._meta_skip_metrics(
                    clean_meta,
                    reason="meta_dataset_unusable",
                    detail=message,
                )
            raise
        feature_columns = list(prepared["feature_columns"])
        fold_metrics = [
            metrics
            for metrics in (
                self._evaluate_meta_fold(fold, feature_columns, gbm_cfg, interval, random_state)
                for fold in prepared.get("walk_forward_splits", [])
            )
            if metrics is not None
        ]
        if not fold_metrics:
            return None, {"rows_total": int(prepared["rows_total"]), "model_role": "meta", "skipped": True, "skip_reason": "meta_walk_forward_unusable"}

        walk_forward_summary = self._aggregate_fold_metrics(fold_metrics, ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "roc_auc"])
        split = prepared["split"]
        X_train = split.train[feature_columns]
        y_train = split.train["meta_label"].astype(int)
        X_valid = split.valid[feature_columns]
        y_valid = split.valid["meta_label"].astype(int)
        if X_train.empty or X_valid.empty or y_train.nunique() < 2 or y_valid.nunique() < 2:
            return None, {"rows_total": int(prepared["rows_total"]), "model_role": "meta", "skipped": True, "skip_reason": "meta_train_valid_unusable"}

        calibrated_model = self.fit_calibrated_classifier(self._build_model(gbm_cfg, interval, random_state), X_train, y_train, X_valid, y_valid)
        valid_pred = calibrated_model.predict(X_valid)
        valid_prob = calibrated_model.predict_proba(X_valid)[:, 1] if hasattr(calibrated_model, "predict_proba") else None
        metrics = self._metrics_binary(y_valid, valid_pred, valid_prob)
        metrics.update(
            {
                "rows_train": int(len(X_train)),
                "rows_valid": int(len(X_valid)),
                "rows_total": int(prepared["rows_total"]),
                "feature_count": int(len(feature_columns)),
                "class_distribution_train": {"0": int((y_train == 0).sum()), "1": int((y_train == 1).sum())},
                "class_distribution_valid": {"0": int((y_valid == 0).sum()), "1": int((y_valid == 1).sum())},
                "model_role": "meta",
                "model_operational_role": "meta_optional",
                "approval_tier": "meta_candidate",
                "live_blocking": False,
                "live_eligible": False,
                "dataset_variant": dataset_variant,
                "candidate_id": candidate_id,
                "calibration_method": "sigmoid_prefit",
                "validation_policy": "walk_forward_plus_tail_calibration",
                "estimator_family": self._estimator_family(gbm_cfg),
                "estimator_contract": self._estimator_training_contract(gbm_cfg, interval, random_state),
                "skipped": False,
                **walk_forward_summary,
            }
        )
        bundle = {
            "model": calibrated_model,
            "feature_columns": feature_columns,
            "rows_total": int(prepared["rows_total"]),
            "feature_set": prepared["feature_set"],
            "validation_summary": walk_forward_summary,
            "estimator_family": self._estimator_family(gbm_cfg),
            "estimator_contract": self._estimator_training_contract(gbm_cfg, interval, random_state),
        }
        return bundle, metrics

    def _active_incumbent_snapshot(self, symbol: str, interval: str) -> dict[str, str | None]:
        """Return current GBM incumbent manifests without changing pointers.

        Step 7 depends on incumbent retention being auditable.  Candidate
        training may save and reject many bundles overnight, but a rejected
        candidate must never erase or replace an already active direction/meta
        incumbent.
        """
        def _path(role: str | None) -> str | None:
            try:
                manifest = resolve_active_manifest_path(self.paths, "gbm", symbol, interval, model_role=role)
            except Exception:
                manifest = None
            return str(manifest) if manifest is not None else None

        return {
            "bundle": _path(None),
            "direction": _path("direction"),
            "meta": _path("meta"),
        }

    @staticmethod
    def _meta_rejection_allows_paper_direction_fallback(
        *,
        direction_decision: dict[str, Any],
        meta_decision: dict[str, Any] | None,
        incumbent_before: dict[str, str | None],
    ) -> bool:
        """Return True when a rejected meta role may be non-blocking for initial paper bootstrap.

        Step 11.1 is intentionally narrow: a GBM meta model with too little
        walk-forward support must not block an otherwise valid initial
        paper-bootstrap direction anchor.  The exception is paper-only, initial
        activation only, and never grants live eligibility.  Any actual meta
        quality failure, incumbent replacement attempt, or direction failure
        remains fail-closed.
        """
        if not isinstance(meta_decision, dict):
            return False
        if not bool(direction_decision.get("approved", False)):
            return False
        if not bool(direction_decision.get("paper_execution_only", False)):
            return False
        if bool(direction_decision.get("live_eligible", False)):
            return False
        if any(bool(value) for value in (incumbent_before or {}).values()):
            return False
        reasons = [str(reason) for reason in (meta_decision.get("reasons") or [])]
        if not reasons:
            return False
        allowed_prefixes = ("insufficient_fold_count:",)
        return all(reason.startswith(allowed_prefixes) for reason in reasons)

    def _bundle_promotion_decision(
        self,
        *,
        symbol: str,
        interval: str,
        candidate_id: str,
        direction_manifest,
        direction_metrics: dict[str, Any],
        meta_manifest,
        meta_metrics: dict[str, Any] | None,
        promote_requested: bool,
    ) -> dict[str, Any]:
        incumbent_before = self._active_incumbent_snapshot(symbol, interval)
        direction_decision = self.evaluate_candidate_promotion(
            symbol=symbol,
            interval=interval,
            metrics=direction_metrics,
            manifest_path=direction_manifest,
            model_role="direction",
            promote_requested=promote_requested,
        )
        direction_decision["candidate_id"] = candidate_id

        meta_decision: dict[str, Any] | None = None
        reasons: list[str] = []
        attempted = bool(direction_decision.get("attempted", False))
        if not direction_decision.get("approved", False):
            reasons.extend([f"direction:{reason}" for reason in direction_decision.get("reasons", [])])

        if meta_metrics is not None:
            if bool(meta_metrics.get("skipped", False)):
                meta_decision = {
                    "attempted": False,
                    "approved": True,
                    "promoted": False,
                    "candidate_id": candidate_id,
                    "model_role": "meta",
                    "skipped": True,
                    "skip_reason": meta_metrics.get("skip_reason"),
                    "has_incumbent": bool(incumbent_before.get("meta")),
                    "incumbent_manifest_path": incumbent_before.get("meta"),
                    "incumbent_retained": bool(incumbent_before.get("meta")),
                    "reasons": [],
                }
            else:
                meta_decision = self.evaluate_candidate_promotion(
                    symbol=symbol,
                    interval=interval,
                    metrics=meta_metrics,
                    manifest_path=meta_manifest,
                    model_role="meta",
                    promote_requested=promote_requested,
                )
                meta_decision["candidate_id"] = candidate_id
                attempted = attempted or bool(meta_decision.get("attempted", False))
                if not meta_decision.get("approved", False):
                    if self._meta_rejection_allows_paper_direction_fallback(
                        direction_decision=direction_decision,
                        meta_decision=meta_decision,
                        incumbent_before=incumbent_before,
                    ):
                        meta_decision.update(
                            {
                                "paper_direction_fallback_allowed": True,
                                "blocking_for_paper_bundle": False,
                                "promoted": False,
                                "live_eligible": False,
                                "fallback_scope": "initial_paper_bootstrap_only",
                            }
                        )
                    else:
                        reasons.extend([f"meta:{reason}" for reason in meta_decision.get("reasons", [])])

        approved = attempted and not reasons
        initial_promotion = bool(approved and not incumbent_before.get("direction"))
        incumbent_retained = bool(attempted and not approved and incumbent_before.get("direction"))
        governance = self._bundle_governance(direction_decision, meta_metrics, approved)
        meta_direction_fallback_allowed = bool(
            isinstance(meta_decision, dict) and meta_decision.get("paper_direction_fallback_allowed")
        )
        if meta_direction_fallback_allowed:
            governance.update(
                {
                    "meta_available": False,
                    "meta_skipped": True,
                    "meta_rejected_for_paper_direction_fallback": True,
                    "meta_rejection_reasons": list(meta_decision.get("reasons") or []),
                    "approval_tier": "paper_only_direction_fallback",
                    "model_operational_role": "direction_only_fallback",
                    "paper_eligible": bool(approved),
                    "paper_execution_authorized": bool(approved),
                    "live_eligible": False,
                    "live_blocking": bool(approved),
                }
            )
        governance.update(
            {
                "initial_promotion": initial_promotion,
                "incumbent_retained": incumbent_retained,
                "active_incumbents_before": dict(incumbent_before),
            }
        )
        return {
            "attempted": attempted,
            "approved": approved,
            "promoted": False,
            "candidate_id": candidate_id,
            "model_role": "bundle",
            "direction": direction_decision,
            "meta": meta_decision,
            "reasons": reasons,
            "paper_evidence_status": str(direction_decision.get("paper_evidence_status") or "not_evaluated"),
            "paper_bootstrap_only": bool(direction_decision.get("paper_execution_only", False)),
            "paper_execution_only": bool(direction_decision.get("paper_execution_only", False)),
            "paper_evidence_sufficient": direction_decision.get("paper_evidence_sufficient"),
            "deployment_stage": "paper_bootstrap" if bool(direction_decision.get("paper_execution_only", False)) else "incumbent",
            "live_eligible": bool(governance.get("live_eligible", False)),
            "initial_promotion": initial_promotion,
            "incumbent_retained": incumbent_retained,
            "active_incumbents_before": dict(incumbent_before),
            "governance": governance,
        }

    def train(self, symbol: str, interval: str) -> TrainerResult:
        try:
            cfg = self.paths.config
            gbm_cfg = cfg.training.gbm
            labels_cfg = cfg.labels
            random_state = int(self._cfg_value(gbm_cfg, interval, "random_state", cfg.system.random_seed))
            tail_rows = int(self._cfg_value(gbm_cfg, interval, "tail_rows", gbm_cfg.tail_rows))
            candidate_id = self.new_candidate_id()
            estimator_contract = self._estimator_training_contract(gbm_cfg, interval, random_state)
            estimator_family = str(estimator_contract["estimator_family"])

            self.progress(symbol, interval, 10, "loading_features")
            fused = self.load_fused_frame(symbol, interval, tail_rows=tail_rows)
            if fused.empty:
                return self.fail_result(symbol=symbol, interval=interval, error="fused frame is empty", model_role="direction")

            self.progress(symbol, interval, 35, "training_direction_model")
            direction_bundle, direction_metrics, labeled = self._train_direction_model(symbol, interval, fused, gbm_cfg, labels_cfg, random_state, candidate_id)
            direction_artifacts = gbm_role_artifact_paths(self.paths, symbol, interval, "direction", stage="candidate", candidate_id=candidate_id)
            direction_artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(direction_bundle["model"], direction_artifacts["model"])
            self.save_training_bundle(
                symbol=symbol,
                interval=interval,
                rows_used=direction_bundle["rows_total"],
                feature_columns=direction_bundle["feature_columns"],
                artifacts=direction_artifacts,
                metrics=direction_metrics,
                candidate_id=candidate_id,
                model_role="direction",
                promote=False,
                extra_manifest={
                    "class_encoding": {"0": "short", "1": "hold", "2": "long"},
                    "feature_set": direction_bundle["feature_set"],
                    "validation": {
                        "policy": "walk_forward",
                        "fold_count": direction_bundle["validation_summary"]["fold_count"],
                        "fold_metrics": direction_bundle["validation_summary"]["fold_metrics"],
                    },
                    "training": {"calibration_method": "sigmoid_prefit", "random_state": random_state, "tail_rows": tail_rows, **estimator_contract},
                    "estimator_family": estimator_family,
                    "estimator_contract": estimator_contract,
                },
            )

            self.progress(symbol, interval, 70, "training_meta_model")
            meta_bundle, meta_metrics = self._train_meta_model(symbol, interval, labeled, gbm_cfg, random_state, candidate_id)
            meta_artifacts = None
            meta_manifest = None
            meta_skipped = True
            meta_skip_reason = None
            if meta_metrics is not None:
                meta_skipped = bool(meta_metrics.get("skipped", False))
                meta_skip_reason = meta_metrics.get("skip_reason")
            if meta_bundle is not None and meta_metrics is not None:
                meta_artifacts = gbm_role_artifact_paths(self.paths, symbol, interval, "meta", stage="candidate", candidate_id=candidate_id)
                meta_artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(meta_bundle["model"], meta_artifacts["model"])
                self.save_training_bundle(
                    symbol=symbol,
                    interval=interval,
                    rows_used=meta_bundle["rows_total"],
                    feature_columns=meta_bundle["feature_columns"],
                    artifacts=meta_artifacts,
                    metrics=meta_metrics,
                    candidate_id=candidate_id,
                    model_role="meta",
                    promote=False,
                    extra_manifest={
                        "feature_set": meta_bundle["feature_set"],
                        "validation": {
                            "policy": "walk_forward",
                            "fold_count": meta_bundle["validation_summary"]["fold_count"],
                            "fold_metrics": meta_bundle["validation_summary"]["fold_metrics"],
                        },
                        "training": {"calibration_method": "sigmoid_prefit", "random_state": random_state, "tail_rows": tail_rows, **estimator_contract},
                        "estimator_family": estimator_family,
                        "estimator_contract": estimator_contract,
                    },
                )
                meta_manifest = meta_artifacts["manifest"]

            candidate_bundle_manifest = gbm_combined_manifest_path(self.paths, symbol, interval, stage="candidate", candidate_id=candidate_id)
            write_gbm_combined_manifest(
                self.paths,
                symbol=symbol,
                interval=interval,
                direction_manifest=direction_artifacts["manifest"],
                meta_manifest=meta_manifest,
                rows_used=direction_bundle["rows_total"],
                meta_skipped=meta_skipped,
                meta_skip_reason=meta_skip_reason,
                stage="candidate",
                candidate_id=candidate_id,
                validation={
                    "policy": "walk_forward",
                    "direction_fold_count": direction_bundle["validation_summary"]["fold_count"],
                    "meta_fold_count": meta_bundle["validation_summary"]["fold_count"] if meta_bundle is not None else 0,
                },
                training={
                    "random_state": random_state,
                    "tail_rows": tail_rows,
                    "candidate_id": candidate_id,
                    **estimator_contract,
                    "governance": {
                        "meta_required_for_live": True,
                        "fallback_mode": "direction_only_allowed_for_paper",
                    },
                },
            )

            bundle_payload = self._bundle_promotion_decision(
                symbol=symbol,
                interval=interval,
                candidate_id=candidate_id,
                direction_manifest=direction_artifacts["manifest"],
                direction_metrics=direction_metrics,
                meta_manifest=meta_manifest,
                meta_metrics=meta_metrics,
                promote_requested=True,
            )

            promoted = False
            if bundle_payload["approved"]:
                direction_decision = bundle_payload.get("direction") if isinstance(bundle_payload.get("direction"), dict) else {}
                direction_stage = str(direction_decision.get("deployment_stage") or "incumbent")
                direction_paper_only = bool(direction_decision.get("paper_execution_only", False))
                promote_candidate_manifest(
                    self.paths,
                    model_type="gbm",
                    symbol=symbol,
                    interval=interval,
                    manifest_path=direction_artifacts["manifest"],
                    candidate_id=candidate_id,
                    model_role="direction",
                    metrics=direction_metrics,
                    deployment_stage=direction_stage,
                    paper_execution_only=direction_paper_only,
                    paper_evidence_status=str(direction_decision.get("paper_evidence_status") or "not_evaluated"),
                )
                bundle_payload["direction"]["promoted"] = True
                bundle_payload["direction"]["pointer_path"] = str(incumbent_pointer_path(self.paths, "gbm", symbol, interval, model_role="direction"))

                meta_decision = bundle_payload.get("meta") if isinstance(bundle_payload.get("meta"), dict) else {}
                meta_promotable = bool(
                    meta_artifacts is not None
                    and meta_metrics is not None
                    and not bool(meta_metrics.get("skipped", False))
                    and bool(meta_decision.get("approved", False))
                    and not bool(meta_decision.get("paper_direction_fallback_allowed", False))
                )
                if meta_promotable:
                    meta_stage = str(meta_decision.get("deployment_stage") or direction_stage)
                    meta_paper_only = bool(meta_decision.get("paper_execution_only", direction_paper_only))
                    promote_candidate_manifest(
                        self.paths,
                        model_type="gbm",
                        symbol=symbol,
                        interval=interval,
                        manifest_path=meta_artifacts["manifest"],
                        candidate_id=candidate_id,
                        model_role="meta",
                        metrics=meta_metrics,
                        deployment_stage=meta_stage,
                        paper_execution_only=meta_paper_only,
                        paper_evidence_status=str((meta_decision or {}).get("paper_evidence_status") or direction_decision.get("paper_evidence_status") or "not_evaluated"),
                    )
                    if isinstance(bundle_payload.get("meta"), dict):
                        bundle_payload["meta"]["promoted"] = True
                        bundle_payload["meta"]["pointer_path"] = str(incumbent_pointer_path(self.paths, "gbm", symbol, interval, model_role="meta"))
                else:
                    delete_incumbent_pointer(self.paths, "gbm", symbol, interval, model_role="meta")

                pointer = promote_candidate_manifest(
                    self.paths,
                    model_type="gbm",
                    symbol=symbol,
                    interval=interval,
                    manifest_path=candidate_bundle_manifest,
                    candidate_id=candidate_id,
                    model_role=None,
                    metrics={"direction": direction_metrics, "meta": meta_metrics or {}},
                    deployment_stage="paper_bootstrap" if direction_paper_only else "incumbent",
                    paper_execution_only=direction_paper_only,
                    paper_evidence_status=str(direction_decision.get("paper_evidence_status") or "not_evaluated"),
                )
                bundle_payload["promoted"] = True
                bundle_payload["pointer_path"] = str(incumbent_pointer_path(self.paths, "gbm", symbol, interval, model_role=None))
                bundle_payload["pointer"] = pointer
                promoted = True

            manifest_payload = load_json(candidate_bundle_manifest) or {}
            manifest_payload["promotion"] = bundle_payload
            manifest_payload["governance"] = bundle_payload.get("governance", {})
            save_json(candidate_bundle_manifest, manifest_payload)

            prune_results: list[dict[str, Any]] = []
            try:
                prune_results = prune_related_candidate_artifacts(
                    self.paths,
                    model_type="gbm",
                    symbol=symbol,
                    interval=interval,
                    model_role=None,
                )
            except Exception:
                self.logger.exception("GBM artifact lifecycle pruning failed symbol=%s interval=%s", symbol, interval)

            self.progress(
                symbol,
                interval,
                100,
                "done",
                meta_skipped=meta_skipped,
                meta_skip_reason=meta_skip_reason,
                candidate_id=candidate_id,
                promoted=promoted,
                promotion_reasons=bundle_payload["reasons"],
                estimator_family=estimator_family,
            )
            combined_metrics = {
                "direction": direction_metrics,
                "promotion": bundle_payload,
                "governance": bundle_payload.get("governance", {}),
                "artifact_pruning": prune_results,
                "estimator_family": estimator_family,
                "estimator_contract": estimator_contract,
            }
            if meta_metrics is not None:
                combined_metrics["meta"] = meta_metrics
            combined_artifacts = {
                "manifest": str(candidate_bundle_manifest),
                "direction_manifest": str(direction_artifacts["manifest"]),
                "direction_model": str(direction_artifacts["model"]),
                "direction_feature_columns": str(direction_artifacts["feature_columns"]),
                "direction_metrics": str(direction_artifacts["metrics"]),
            }
            if meta_artifacts is not None:
                combined_artifacts.update(
                    {
                        "meta_manifest": str(meta_artifacts["manifest"]),
                        "meta_model": str(meta_artifacts["model"]),
                        "meta_feature_columns": str(meta_artifacts["feature_columns"]),
                        "meta_metrics": str(meta_artifacts["metrics"]),
                    }
                )
            return self.ok_result(symbol=symbol, interval=interval, rows_used=direction_bundle["rows_total"], artifacts=combined_artifacts, metrics=combined_metrics)
        except Exception as exc:
            self.logger.exception("GBM training failed symbol=%s interval=%s", symbol, interval)
            return self.fail_result(symbol=symbol, interval=interval, error=str(exc), model_role="direction")


def train_gbm_model(paths, symbol: str, interval: str) -> TrainerResult:
    """Train the GBM direction/meta bundle for a single symbol/interval pair."""
    return GBMTrainer(paths).train(symbol, interval)
