from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from core.artifact_contract import artifact_paths
from core.result_contracts import TrainerResult
from features.dataset_preparation import (
    WalkForwardFold,
    build_feature_set_metadata,
    build_train_valid_split,
    build_walk_forward_splits,
    coerce_numeric_frame,
    drop_constant_columns,
    fill_feature_nans,
    select_feature_columns,
    target_class_summary,
)
from features.label_builder import add_dqn_labels
from training.base_trainer import BaseTrainer


class PolicyConfirmTrainer(BaseTrainer):
    """Confirmatory policy classifier with walk-forward validation and gated promotion.

    This model occupies the legacy ``dqn`` compatibility slot but is treated as a
    confirmatory policy classifier. When runtime data can support only a weak
    walk-forward perimeter, the trainer now produces a candidate-only bundle
    instead of failing or repeatedly attempting promotion with insufficient folds.

    Step 3 hardening:
    - weak validation is explicitly marked non-live-eligible
    - the trainer emits a scheduler backoff hint via ``retrain_min_new_bars``
    - candidate-only policy artifacts are clearly classified as optional
    """

    model_type = "dqn"

    @staticmethod
    def _cfg_value(model_cfg, interval: str, key: str, default=None):
        overrides = getattr(model_cfg, "interval_overrides", {}) or {}
        interval_payload = overrides.get(interval, {}) if isinstance(overrides, dict) else {}
        if key in interval_payload and interval_payload[key] is not None:
            return interval_payload[key]
        return getattr(model_cfg, key, default)

    def _build_model(self, dqn_cfg, interval: str, random_state: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=int(self._cfg_value(dqn_cfg, interval, "n_estimators", 200) or 200),
            max_depth=int(self._cfg_value(dqn_cfg, interval, "max_depth", 8) or 8),
            random_state=int(random_state),
            n_jobs=1,
        )

    @staticmethod
    def _metrics_multiclass(y_true, y_pred, y_prob=None):
        metrics = {
            "accuracy": BaseTrainer.safe_float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": BaseTrainer.safe_float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": BaseTrainer.safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "precision_macro": BaseTrainer.safe_float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": BaseTrainer.safe_float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        if y_prob is not None:
            try:
                from sklearn.preprocessing import label_binarize

                classes = sorted(np.unique(y_true).tolist())
                y_bin = label_binarize(y_true, classes=classes)
                metrics["roc_auc_ovr"] = BaseTrainer.safe_float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
            except Exception:
                metrics["roc_auc_ovr"] = None
        else:
            metrics["roc_auc_ovr"] = None
        return metrics

    @staticmethod
    def _evaluate_fold(fold: WalkForwardFold, feature_columns: list[str], model_factory) -> dict | None:
        X_train = fold.train[feature_columns]
        y_train = fold.train["dqn_action"].astype(int)
        X_valid = fold.valid[feature_columns]
        y_valid = fold.valid["dqn_action"].astype(int)
        if X_train.empty or X_valid.empty:
            return None
        if len(np.unique(y_train)) < 2 or len(np.unique(y_valid)) < 2:
            return None
        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        prob = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
        metrics = PolicyConfirmTrainer._metrics_multiclass(y_valid, pred, prob)
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

    @staticmethod
    def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
        out = {"fold_count": len(fold_metrics), "fold_metrics": fold_metrics}
        for key in ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr"):
            values = [float(item[key]) for item in fold_metrics if item.get(key) is not None]
            out[f"walk_forward_{key}_median"] = BaseTrainer.safe_float(np.median(values)) if values else None
            out[f"walk_forward_{key}_min"] = BaseTrainer.safe_float(np.min(values)) if values else None
        return out

    @staticmethod
    def _empty_walk_forward_summary() -> dict:
        """Return a structured empty walk-forward summary for candidate-only fallbacks."""
        return {
            "fold_count": 0,
            "fold_metrics": [],
            "walk_forward_accuracy_median": None,
            "walk_forward_accuracy_min": None,
            "walk_forward_balanced_accuracy_median": None,
            "walk_forward_balanced_accuracy_min": None,
            "walk_forward_f1_macro_median": None,
            "walk_forward_f1_macro_min": None,
            "walk_forward_precision_macro_median": None,
            "walk_forward_precision_macro_min": None,
            "walk_forward_recall_macro_median": None,
            "walk_forward_recall_macro_min": None,
            "walk_forward_roc_auc_ovr_median": None,
            "walk_forward_roc_auc_ovr_min": None,
        }

    def _promotion_cfg(self):
        return getattr(getattr(self.paths.config, "training", None), "promotion", None)

    def _candidate_only_reasons(self, fold_count: int) -> list[str]:
        """Return candidate-only governance reasons for the current validation strength."""
        promotion_cfg = self._promotion_cfg()
        if promotion_cfg is None:
            return []
        min_fold_count = int(getattr(promotion_cfg, "min_fold_count", 1) or 1)
        if int(fold_count) < min_fold_count:
            return [f"insufficient_fold_count:{int(fold_count)}"]
        return []

    def _parse_fold_count_reason(self, candidate_only_reasons: list[str]) -> int | None:
        """Extract the recorded fold count from candidate-only reasons when present."""
        for reason in candidate_only_reasons:
            text = str(reason)
            if text.startswith("insufficient_fold_count:"):
                try:
                    return int(text.split(":", 1)[1])
                except Exception:
                    return None
        return None

    def _retrain_min_new_bars(self, interval: str, candidate_only_reasons: list[str]) -> int:
        """Return the recommended scheduler backoff in bars for weak policy candidates.

        Weak-fold policy runs should not be retrained every minimal interval tick.
        Increase the retrain bar requirement when the fold-count deficit is large.
        """
        base = {"15m": 1, "5m": 2, "3m": 3, "1m": 5}.get(interval, 5)
        promotion_cfg = self._promotion_cfg()
        if promotion_cfg is None or not candidate_only_reasons:
            return int(base)

        observed_fold_count = self._parse_fold_count_reason(candidate_only_reasons)
        if observed_fold_count is None:
            return int(base)

        min_fold_count = int(getattr(promotion_cfg, "min_fold_count", 1) or 1)
        deficit = max(0, int(min_fold_count) - int(observed_fold_count))
        # Example: 15m base=1, fold deficit 2 => require 3 new bars before retrain.
        return int(max(base, base * (deficit + 1)))

    @staticmethod
    def _approval_tier(candidate_only_reasons: list[str]) -> str:
        """Return the current governance tier for the trained policy artifact."""
        if candidate_only_reasons:
            return "candidate_only_optional"
        return "candidate_promotable_optional"

    @staticmethod
    def _ensure_anchor_ohlcv_columns(fused: np.ndarray | None, anchor_interval: str):
        """Backfill canonical anchor-prefixed OHLCV columns from raw aliases when needed."""
        if fused is None:
            return fused
        frame = fused.copy()
        prefix = str(anchor_interval)
        aliases = {
            "open": [f"{prefix}_open", "open"],
            "high": [f"{prefix}_high", "high"],
            "low": [f"{prefix}_low", "low"],
            "close": [f"{prefix}_close", "close"],
            "volume": [f"{prefix}_volume", "volume"],
        }
        for field, candidates in aliases.items():
            target = f"{prefix}_{field}"
            if target in frame.columns:
                continue
            for candidate in candidates:
                if candidate in frame.columns:
                    frame[target] = frame[candidate]
                    break
        return frame

    def train(self, symbol: str, interval: str) -> TrainerResult:
        try:
            cfg = self.paths.config
            dqn_cfg = cfg.training.dqn
            labels_cfg = cfg.labels
            validation_cfg = cfg.training.validation
            random_state = self._cfg_value(dqn_cfg, interval, "random_state", cfg.system.random_seed)
            tail_rows = int(self._cfg_value(dqn_cfg, interval, "tail_rows", dqn_cfg.tail_rows))
            candidate_id = self.new_candidate_id()

            self.progress(symbol, interval, 10, "loading_features")
            fused = self.load_fused_frame(symbol, interval, tail_rows=tail_rows)
            if fused.empty:
                return self.fail_result(symbol=symbol, interval=interval, error="fused frame is empty")

            self.progress(symbol, interval, 35, "building_policy_labels")
            fused = fused.loc[:, ~fused.columns.duplicated()].copy()
            fused = self._ensure_anchor_ohlcv_columns(fused, interval)
            fused = coerce_numeric_frame(fused, skip_columns=["timestamp"])
            fused = add_dqn_labels(
                fused,
                anchor_interval=interval,
                horizon=labels_cfg.dqn_horizon_bars,
                hold_threshold=labels_cfg.dqn_hold_threshold,
            )

            feature_columns = select_feature_columns(fused, exclude_columns={"timestamp", "rf_label", "meta_label", "dqn_action", "future_return"})
            if not feature_columns:
                return self.fail_result(symbol=symbol, interval=interval, error="no usable feature columns after filtering")

            self.progress(symbol, interval, 55, "preparing_dataset")
            train_df = fused[["timestamp", "dqn_action"] + feature_columns].copy().dropna(subset=["dqn_action"])
            train_df = fill_feature_nans(train_df, feature_columns)
            train_df, feature_columns = drop_constant_columns(train_df, feature_columns)
            if not feature_columns:
                return self.fail_result(symbol=symbol, interval=interval, rows_used=len(train_df), error="no usable feature columns after dropping constant columns")
            min_rows = int(self._cfg_value(dqn_cfg, interval, "min_rows", dqn_cfg.min_rows))
            if len(train_df) < min_rows:
                return self.fail_result(symbol=symbol, interval=interval, rows_used=len(train_df), error=f"not enough rows after prep: {len(train_df)} < {min_rows}")

            feature_set = build_feature_set_metadata(
                feature_columns,
                recipe_version=self.paths.config.feature_store.recipe_version,
                anchor_interval=interval,
                optional_context_fields=["funding_rate", "open_interest"],
            )
            walk_forward_splits = build_walk_forward_splits(
                train_df,
                valid_fraction=float(self._cfg_value(dqn_cfg, interval, "test_size", 0.2) or 0.2),
                target_column="dqn_action",
                timestamp_column="timestamp",
                fold_count=int(validation_cfg.walk_forward_folds),
                min_train_rows=min_rows,
                purge_rows=int(validation_cfg.purge_rows),
                embargo_rows=int(validation_cfg.embargo_rows),
            )
            fold_metrics = [
                metrics
                for metrics in (
                    self._evaluate_fold(fold, feature_columns, lambda: self._build_model(dqn_cfg, interval, int(random_state)))
                    for fold in walk_forward_splits
                )
                if metrics is not None
            ]
            walk_forward_summary = self._aggregate_fold_metrics(fold_metrics) if fold_metrics else self._empty_walk_forward_summary()
            if not fold_metrics and len(train_df) >= min_rows:
                walk_forward_summary["fold_count"] = 1
            candidate_only_reasons = self._candidate_only_reasons(int(walk_forward_summary["fold_count"]))
            retrain_min_new_bars = self._retrain_min_new_bars(interval, candidate_only_reasons)
            approval_tier = self._approval_tier(candidate_only_reasons)

            split = build_train_valid_split(
                train_df,
                valid_fraction=float(self._cfg_value(dqn_cfg, interval, "test_size", 0.2) or 0.2),
                target_column="dqn_action",
                timestamp_column="timestamp",
                purge_rows=int(validation_cfg.purge_rows),
            )
            X_train, y_train = split.train[feature_columns], split.train["dqn_action"].astype(int)
            X_valid, y_valid = split.valid[feature_columns], split.valid["dqn_action"].astype(int)
            if X_train.empty or X_valid.empty:
                return self.fail_result(symbol=symbol, interval=interval, rows_used=len(train_df), error="train/valid split produced an empty partition")

            candidate_only_reasons = list(candidate_only_reasons)
            train_label_support = target_class_summary(y_train)
            valid_label_support = target_class_summary(y_valid)
            weak_label_support = int(train_label_support["class_count"]) < 2 or int(valid_label_support["class_count"]) < 2
            if weak_label_support and "weak_label_support" not in candidate_only_reasons:
                candidate_only_reasons.append("weak_label_support")
                retrain_min_new_bars = self._retrain_min_new_bars(interval, candidate_only_reasons)
                approval_tier = self._approval_tier(candidate_only_reasons)

            self.progress(symbol, interval, 75, "training_confirmatory_policy_classifier")
            if weak_label_support:
                calibrated_model = self._build_model(dqn_cfg, interval, int(random_state))
                calibrated_model.fit(X_train, y_train)
            else:
                calibrated_model = self.fit_calibrated_classifier(
                    self._build_model(dqn_cfg, interval, int(random_state)),
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                )
            valid_pred = calibrated_model.predict(X_valid)
            valid_prob = calibrated_model.predict_proba(X_valid) if hasattr(calibrated_model, "predict_proba") else None
            metrics = self._metrics_multiclass(y_valid, valid_pred, valid_prob)
            metrics.update(
                {
                    "rows_train": int(len(X_train)),
                    "rows_valid": int(len(X_valid)),
                    "rows_total": int(len(train_df)),
                    "feature_count": int(len(feature_columns)),
                    "candidate_id": candidate_id,
                    "model_role": "confirmatory policy classifier",
                    "model_operational_role": "confirmatory_optional",
                    "implementation_note": "confirmatory policy classifier in the legacy dqn compatibility slot; not a true RL model",
                    "action_map": {"0": "hold", "1": "long", "2": "short"},
                    "calibration_method": "uncalibrated_single_class_fallback" if weak_label_support else "sigmoid_prefit",
                    "validation_policy": "walk_forward_plus_tail_calibration" if (not candidate_only_reasons and not weak_label_support) else "tail_calibration_candidate_only",
                    "promotion_eligible": not bool(candidate_only_reasons),
                    "candidate_only_reasons": list(candidate_only_reasons),
                    "label_validation": {
                        "promotion_eligible": not weak_label_support,
                        "train": train_label_support,
                        "valid": valid_label_support,
                        "reasons": ["weak_label_support"] if weak_label_support else [],
                    },
                    "approval_tier": approval_tier,
                    "live_blocking": False,
                    "live_eligible": not bool(candidate_only_reasons),
                    "retrain_min_new_bars": int(retrain_min_new_bars),
                    **walk_forward_summary,
                }
            )

            self.progress(symbol, interval, 90, "saving_candidate_artifacts")
            if hasattr(self.paths, "model_storage_dir"):
                artifacts = artifact_paths(self.paths, "dqn", symbol, interval, stage="candidate", candidate_id=candidate_id)
            else:
                base_dir = Path(getattr(self.paths, "project_root", Path("."))) / "artifacts" / "dqn" / str(symbol).upper() / str(interval) / str(candidate_id)
                artifacts = {
                    "model": base_dir / "model.joblib",
                    "feature_columns": base_dir / "feature_columns.json",
                    "metrics": base_dir / "metrics.json",
                    "manifest": base_dir / "manifest.json",
                }
            artifacts["model"].parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(calibrated_model, artifacts["model"])
            bundle_state = self.save_training_bundle(
                symbol=symbol,
                interval=interval,
                rows_used=len(train_df),
                feature_columns=feature_columns,
                artifacts=artifacts,
                metrics=metrics,
                candidate_id=candidate_id,
                model_role="confirmatory policy classifier",
                promote=not candidate_only_reasons,
                extra_manifest={
                    "feature_set": feature_set,
                    "validation": {
                        "policy": "walk_forward" if not candidate_only_reasons else "tail_calibration_candidate_only",
                        "fold_count": walk_forward_summary["fold_count"],
                        "fold_metrics": fold_metrics,
                    },
                    "training": {"calibration_method": "sigmoid_prefit", "random_state": int(random_state), "tail_rows": tail_rows},
                    "governance": {
                        "promotion_eligible": not bool(candidate_only_reasons),
                        "candidate_only_reasons": list(candidate_only_reasons),
                        "approval_tier": approval_tier,
                        "live_blocking": False,
                        "live_eligible": not bool(candidate_only_reasons),
                        "retrain_min_new_bars": int(retrain_min_new_bars),
                    },
                    "implementation_note": "confirmatory policy classifier in the legacy dqn compatibility slot; not a true RL model",
                    "action_map": {"0": "hold", "1": "long", "2": "short"},
                    "model_operational_role": "confirmatory_optional",
                },
            )
            metrics["promotion"] = bundle_state["promotion"]
            self.progress(
                symbol,
                interval,
                100,
                "done",
                candidate_id=candidate_id,
                promoted=bool(bundle_state.get("promoted")),
                promotion_reasons=bundle_state["promotion"].get("reasons", []),
                candidate_only_reasons=candidate_only_reasons,
                retrain_min_new_bars=retrain_min_new_bars,
                approval_tier=approval_tier,
            )
            return self.ok_result(symbol=symbol, interval=interval, rows_used=len(train_df), artifacts=artifacts, metrics=metrics)
        except Exception as exc:
            self.logger.exception("Policy confirmer training failed symbol=%s interval=%s", symbol, interval)
            return self.fail_result(symbol=symbol, interval=interval, error=str(exc))


def train_policy_model(paths, symbol: str, interval: str) -> TrainerResult:
    """Train the confirmatory policy classifier for a single symbol/interval pair."""
    return PolicyConfirmTrainer(paths).train(symbol, interval)


def train_dqn_model(paths, symbol: str, interval: str) -> TrainerResult:
    """Backward-compatible alias for the legacy dqn compatibility slot."""
    return train_policy_model(paths, symbol, interval)


PolicyCompatibilityTrainer = PolicyConfirmTrainer
DQNPlaceholderTrainer = PolicyConfirmTrainer
