from __future__ import annotations

from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from core.artifact_contract import artifact_paths
from core.result_contracts import TrainerResult
from features.dataset_preparation import WalkForwardFold, prepare_supervised_dataset, validate_target_class_support
from features.label_builder import build_rf_labels, get_class_distribution, labels_config_value
from training.base_trainer import BaseTrainer


class RandomForestTrainer(BaseTrainer):
    """Random-forest direction trainer with walk-forward validation and gated promotion."""

    model_type = "rf"
    target_label_column = "rf_label"

    @staticmethod
    def _cfg_value(model_cfg, interval: str, key: str, default: Any = None) -> Any:
        overrides = getattr(model_cfg, "interval_overrides", {}) or {}
        interval_payload = overrides.get(interval, {}) if isinstance(overrides, dict) else {}
        if key in interval_payload and interval_payload[key] is not None:
            return interval_payload[key]
        return getattr(model_cfg, key, default)

    @staticmethod
    def _metrics_multiclass(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray | None) -> Dict[str, Any]:
        metrics = {
            "macro_f1": BaseTrainer.safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": BaseTrainer.safe_float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "balanced_accuracy": BaseTrainer.safe_float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": BaseTrainer.safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": BaseTrainer.safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        if y_prob is not None:
            try:
                classes = sorted(pd.Series(y_true).dropna().astype(int).unique().tolist())
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
        return metrics

    @staticmethod
    def _aggregate_fold_metrics(fold_metrics: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
        summary: dict[str, Any] = {"fold_count": len(fold_metrics), "fold_metrics": fold_metrics}
        for key in metric_keys:
            values = [float(metric[key]) for metric in fold_metrics if metric.get(key) is not None]
            if not values:
                summary[f"walk_forward_{key}_median"] = None
                summary[f"walk_forward_{key}_min"] = None
                continue
            summary[f"walk_forward_{key}_median"] = BaseTrainer.safe_float(np.median(values))
            summary[f"walk_forward_{key}_min"] = BaseTrainer.safe_float(np.min(values))
        return summary

    @staticmethod
    def _resolve_target_column(frame: pd.DataFrame) -> str:
        """Return the target label column used by the RF trainer.

        The current canonical RF label is ``rf_label``. A legacy ``direction_label``
        column is still accepted as a backward-compatible fallback when loading
        older intermediate frames or tests.
        """
        if "rf_label" in frame.columns:
            return "rf_label"
        if "direction_label" in frame.columns:
            return "direction_label"
        raise ValueError("missing_rf_target_label")

    def _build_model(self, rf_cfg, interval: str, random_state: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=int(self._cfg_value(rf_cfg, interval, "n_estimators", 200)),
            max_depth=int(self._cfg_value(rf_cfg, interval, "max_depth", 10)),
            min_samples_leaf=int(self._cfg_value(rf_cfg, interval, "min_samples_leaf", 15)),
            max_samples=float(self._cfg_value(rf_cfg, interval, "max_samples", 0.7)),
            random_state=int(random_state),
            n_jobs=int(self._cfg_value(rf_cfg, interval, "n_jobs", 1)),
            class_weight=self._cfg_value(rf_cfg, interval, "class_weight", "balanced_subsample"),
        )

    def _evaluate_fold(
        self,
        fold: WalkForwardFold,
        feature_columns: list[str],
        rf_cfg,
        interval: str,
        random_state: int,
        target_column: str,
    ) -> dict[str, Any]:
        X_train = fold.train[feature_columns]
        y_train = fold.train[target_column].astype(int)
        X_valid = fold.valid[feature_columns]
        y_valid = fold.valid[target_column].astype(int)
        model = self._build_model(rf_cfg, interval, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        y_prob = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
        metrics = self._metrics_multiclass(y_valid, y_pred, y_prob)
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

    def _prepare_dataset_with_fallback(self, labeled: pd.DataFrame, interval: str, rf_cfg, target_column: str):
        preferred_tail = int(self._cfg_value(rf_cfg, interval, "tail_rows", rf_cfg.tail_rows))
        preferred_min_rows = int(self._cfg_value(rf_cfg, interval, "min_rows", rf_cfg.min_rows))
        validation_cfg = self.paths.config.training.validation
        candidates = [
            {"tail_rows": preferred_tail, "min_rows": preferred_min_rows, "dataset_variant": "configured"},
            {
                "tail_rows": max(preferred_tail, preferred_min_rows * 3),
                "min_rows": max(80, min(preferred_min_rows, 120)),
                "dataset_variant": "fallback_relaxed_min_rows",
            },
        ]
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                prepared = prepare_supervised_dataset(
                    labeled,
                    target_column=target_column,
                    exclude_columns={"direction_label", "rf_label", "dqn_action", "future_return", "thr_dyn", "meta_label", "direction_oos_pred"},
                    timestamp_column="timestamp",
                    valid_fraction=float(self._cfg_value(rf_cfg, interval, "test_size", 0.2) or 0.2),
                    min_rows_required=int(candidate["min_rows"]),
                    passthrough_columns=["future_return", "thr_dyn"],
                    walk_forward_folds=int(validation_cfg.walk_forward_folds),
                    walk_forward_purge_rows=int(validation_cfg.purge_rows),
                    walk_forward_embargo_rows=int(validation_cfg.embargo_rows),
                    feature_recipe_version=self.paths.config.feature_store.recipe_version,
                    anchor_interval=interval,
                    optional_context_fields=["funding_rate", "open_interest"],
                )
                return prepared, str(candidate["dataset_variant"])
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "RF dataset preparation retry symbol=%s interval=%s variant=%s error=%s",
                    labeled.get("symbol", pd.Series(["unknown"])).iloc[0] if "symbol" in labeled.columns and not labeled.empty else "unknown",
                    interval,
                    candidate["dataset_variant"],
                    exc,
                )
        if last_error is not None:
            raise last_error
        raise ValueError("RF dataset preparation failed unexpectedly")

    def train(self, symbol: str, interval: str) -> TrainerResult:
        try:
            cfg = self.paths.config
            rf_cfg = cfg.training.rf
            labels_cfg = cfg.labels
            random_state = int(self._cfg_value(rf_cfg, interval, "random_state", cfg.system.random_seed))
            tail_rows = int(self._cfg_value(rf_cfg, interval, "tail_rows", rf_cfg.tail_rows))
            candidate_id = self.new_candidate_id()

            self.progress(symbol, interval, 10, "loading_features")
            fused = self.load_fused_frame(symbol, interval, tail_rows=tail_rows)
            if fused.empty:
                return self.fail_result(symbol=symbol, interval=interval, error="fused frame is empty", model_role="direction")

            horizon = labels_config_value(labels_cfg, "horizon_bars", interval, labels_cfg.horizon_bars)
            atr_multiplier = labels_config_value(labels_cfg, "atr_multiplier", interval, labels_cfg.atr_multiplier)
            atr_floor = labels_config_value(labels_cfg, "atr_floor", interval, labels_cfg.atr_floor)

            self.progress(symbol, interval, 25, "building_labels")
            labeled = build_rf_labels(
                fused,
                anchor_interval=cfg.frames.anchor_interval,
                horizon=int(horizon),
                atr_multiplier=float(atr_multiplier),
                atr_floor=float(atr_floor),
                trading_cfg=getattr(cfg, "trading", None),
            )
            target_column = self._resolve_target_column(labeled)
            labeled = labeled.dropna(subset=[target_column]).copy()
            if labeled.empty:
                return self.fail_result(symbol=symbol, interval=interval, error="no labeled rows", model_role="direction")

            self.progress(symbol, interval, 40, "preparing_dataset")
            prepared, dataset_variant = self._prepare_dataset_with_fallback(labeled, interval, rf_cfg, target_column)
            feature_columns = list(prepared["feature_columns"])
            fold_metrics = [
                self._evaluate_fold(fold, feature_columns, rf_cfg, interval, random_state, target_column)
                for fold in prepared.get("walk_forward_splits", [])
            ]
            aggregate_walk_forward = self._aggregate_fold_metrics(
                fold_metrics,
                ["macro_f1", "weighted_f1", "balanced_accuracy", "precision_macro", "recall_macro", "roc_auc_ovr"],
            )

            split = prepared["split"]
            X_train = split.train[feature_columns]
            y_train = split.train[target_column].astype(int)
            X_valid = split.valid[feature_columns]
            y_valid = split.valid[target_column].astype(int)

            if X_train.empty or X_valid.empty:
                return self.fail_result(
                    symbol=symbol,
                    interval=interval,
                    rows_used=int(prepared["rows_total"]),
                    error="train/valid split produced an empty partition",
                    model_role="direction",
                )
            try:
                label_validation_train = validate_target_class_support(
                    y_train,
                    min_classes=3,
                    required_classes=[0, 1, 2],
                    context="rf_train",
                )
                label_validation_valid = validate_target_class_support(
                    y_valid,
                    min_classes=2,
                    context="rf_valid",
                )
            except ValueError as exc:
                return self.fail_result(
                    symbol=symbol,
                    interval=interval,
                    rows_used=int(prepared["rows_total"]),
                    error=str(exc),
                    model_role="direction",
                )

            self.progress(symbol, interval, 70, "training_calibrated_random_forest")
            calibrated_model = self.fit_calibrated_classifier(
                self._build_model(rf_cfg, interval, int(random_state)),
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
            valid_pred = calibrated_model.predict(X_valid)
            valid_prob = calibrated_model.predict_proba(X_valid) if hasattr(calibrated_model, "predict_proba") else None
            metrics = self._metrics_multiclass(y_valid, valid_pred, valid_prob)
            metrics.update(self.directional_quality_report(y_valid, valid_pred, class_encoding={0: "short", 1: "hold", 2: "long"}))
            payoff_metrics = {
                "avg_win_return": BaseTrainer.safe_float(float(labeled.loc[labeled["future_return"] > 0, "future_return"].mean())) if "future_return" in labeled.columns else None,
                "avg_loss_return": BaseTrainer.safe_float(abs(float(labeled.loc[labeled["future_return"] < 0, "future_return"].mean()))) if "future_return" in labeled.columns else None,
            }
            metrics.update(
                {
                    "rows_train": int(len(X_train)),
                    "rows_valid": int(len(X_valid)),
                    "rows_total": int(prepared["rows_total"]),
                    "feature_count": int(len(feature_columns)),
                    "class_distribution_train": get_class_distribution(y_train),
                    "class_distribution_valid": get_class_distribution(y_valid),
                    "model_role": "direction",
                    "dataset_variant": dataset_variant,
                    "candidate_id": candidate_id,
                    "label_column": target_column,
                    "label_validation": {
                        "promotion_eligible": True,
                        "train": label_validation_train,
                        "valid": label_validation_valid,
                        "reasons": [],
                    },
                    "calibration_method": "sigmoid_prefit",
                    "validation_policy": "walk_forward_plus_tail_calibration",
                    **aggregate_walk_forward,
                    **payoff_metrics,
                }
            )

            self.progress(symbol, interval, 90, "saving_candidate_artifacts")
            artifacts = artifact_paths(self.paths, "rf", symbol, interval, stage="candidate", candidate_id=candidate_id)
            artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibrated_model, artifacts["model"])
            bundle_state = self.save_training_bundle(
                symbol=symbol,
                interval=interval,
                rows_used=int(prepared["rows_total"]),
                feature_columns=feature_columns,
                artifacts=artifacts,
                metrics=metrics,
                candidate_id=candidate_id,
                model_role="direction",
                promote=True,
                extra_manifest={
                    "class_encoding": {"0": "short", "1": "hold", "2": "long"},
                    "feature_set": prepared["feature_set"],
                    "validation": {
                        "policy": "walk_forward",
                        "fold_count": aggregate_walk_forward["fold_count"],
                        "fold_metrics": fold_metrics,
                    },
                    "training": {
                        "calibration_method": "sigmoid_prefit",
                        "random_state": int(random_state),
                        "tail_rows": int(tail_rows),
                    },
                },
            )
            promotion_state = bundle_state.get(
                "promotion",
                {
                    "attempted": True,
                    "approved": bool(bundle_state.get("promoted")),
                    "promoted": bool(bundle_state.get("promoted")),
                    "reasons": [],
                },
            )
            metrics["promotion"] = promotion_state
            self.progress(
                symbol,
                interval,
                100,
                "done",
                candidate_id=candidate_id,
                promoted=bool(promotion_state.get("promoted")),
                promotion_reasons=promotion_state.get("reasons", []),
            )
            return self.ok_result(symbol=symbol, interval=interval, rows_used=int(prepared["rows_total"]), artifacts=artifacts, metrics=metrics)
        except Exception as exc:
            self.logger.exception("RF training failed symbol=%s interval=%s", symbol, interval)
            return self.fail_result(symbol=symbol, interval=interval, error=str(exc))


def train_rf_model(paths, symbol: str, interval: str) -> TrainerResult:
    """Train the RF direction model for a single symbol/interval pair."""
    return RandomForestTrainer(paths).train(symbol, interval)
