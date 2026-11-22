from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


class SecretValidationError(ConfigError):
    """Raised when required runtime secrets are missing or unsafe."""


MONITORING_TOKEN_ENV_VARS = ("AUTO_TRADER_MONITORING_AUTH_TOKEN",)
TELEGRAM_BOT_TOKEN_ENV_VARS = ("AUTO_TRADER_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID_ENV_VARS = ("AUTO_TRADER_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")
BINANCE_API_KEY_ENV_VARS = ("AUTO_TRADER_BINANCE_API_KEY", "BINANCE_API_KEY")
BINANCE_API_SECRET_ENV_VARS = ("AUTO_TRADER_BINANCE_API_SECRET", "BINANCE_API_SECRET")

DEFAULT_MONITORING_TOKEN = "change-this-token"

DEV_ENVIRONMENTS = {"dev", "development", "local", "test", "testing"}
VALID_ENVIRONMENTS = {
    "dev",
    "development",
    "local",
    "test",
    "testing",
    "stage",
    "staging",
    "prod",
    "production",
}
VALID_EXECUTION_MODES = {"paper", "live"}
VALID_PAPER_SHUTDOWN_POLICIES = {"snapshot_only", "close_positions", "close_positions_and_snapshot"}
VALID_LIVE_SHUTDOWN_POLICIES = {"fail_closed_snapshot_only", "force_close_after_reconciliation"}
VALID_GBM_ESTIMATORS = {"histgbm", "lightgbm"}
VALID_HMM_COVARIANCE_TYPES = {"diag", "full", "tied", "spherical"}



def _expect_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{path}' must be a mapping")
    return value


def _get_section(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key, {})
    return _expect_mapping(value, key)


def _get_str(section: dict[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"config key must be a non-empty string: {key}")
    return value.strip()


def _get_optional_str(section: dict[str, Any], key: str, default: str = "") -> str:
    value = section.get(key, default)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError(f"config key must be a string when provided: {key}")
    return value.strip()


def _get_bool(section: dict[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if isinstance(value, bool):
        return value
    raise ConfigError(f"config key must be boolean: {key}")


def _get_int(section: dict[str, Any], key: str, default: int, minimum: int | None = None) -> int:
    try:
        out = int(section.get(key, default))
    except Exception as exc:
        raise ConfigError(f"config key must be integer: {key}") from exc
    if minimum is not None and out < minimum:
        raise ConfigError(f"config key must be >= {minimum}: {key}")
    return out


def _get_float(section: dict[str, Any], key: str, default: float, minimum: float | None = None) -> float:
    try:
        out = float(section.get(key, default))
    except Exception as exc:
        raise ConfigError(f"config key must be numeric: {key}") from exc
    if minimum is not None and out < minimum:
        raise ConfigError(f"config key must be >= {minimum}: {key}")
    return out


def _get_str_list(section: dict[str, Any], key: str, default: list[str]) -> list[str]:
    raw = section.get(key, default)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"config key must be non-empty list: {key}")
    return [str(v) for v in raw]


def _get_int_mapping(section: dict[str, Any], key: str, default: dict[str, int]) -> dict[str, int]:
    raw = section.get(key, default)
    if not isinstance(raw, dict):
        raise ConfigError(f"config key must be mapping: {key}")
    return {str(k): int(v) for k, v in raw.items()}


def _get_nested_float_mapping(section: dict[str, Any], key: str, default: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    raw = section.get(key, default)
    if not isinstance(raw, dict):
        raise ConfigError(f"config key must be nested mapping: {key}")
    out: dict[str, dict[str, float]] = {}
    for outer_key, inner in raw.items():
        if not isinstance(inner, dict):
            raise ConfigError(f"config key must be nested mapping: {key}.{outer_key}")
        out[str(outer_key)] = {str(inner_key): float(inner_value) for inner_key, inner_value in inner.items()}
    return out


def _get_nested_int_mapping(section: dict[str, Any], key: str, default: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    raw = section.get(key, default)
    if not isinstance(raw, dict):
        raise ConfigError(f"config key must be nested mapping: {key}")
    out: dict[str, dict[str, int]] = {}
    for outer_key, inner in raw.items():
        if not isinstance(inner, dict):
            raise ConfigError(f"config key must be nested mapping: {key}.{outer_key}")
        out[str(outer_key)] = {str(inner_key): int(inner_value) for inner_key, inner_value in inner.items()}
    return out


def _resolve_secret(config_value: str, env_var_names: tuple[str, ...]) -> str:
    """
    Resolve a secret from config first, then environment.

    Resolution order:
    1. YAML / config value if non-empty
    2. First non-empty matching environment variable
    3. Empty string
    """
    if isinstance(config_value, str) and config_value.strip():
        return config_value.strip()
    for env_name in env_var_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return ""


def _resolve_env_only_secret(env_var_names: tuple[str, ...]) -> str:
    """Resolve a secret from environment variables only."""
    for env_name in env_var_names:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value
    return ""


def _reject_inline_secret(config_value: str, key_path: str, env_var_names: tuple[str, ...]) -> None:
    if isinstance(config_value, str) and config_value.strip():
        env_hint = " or ".join(env_var_names)
        raise ConfigError(
            f"{key_path} must not be set inline; provide it via environment variable {env_hint}"
        )


def _normalize_environment(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VALID_ENVIRONMENTS:
        raise ConfigError(f"system.environment must be one of {sorted(VALID_ENVIRONMENTS)}")
    return normalized


def _normalize_execution_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in VALID_EXECUTION_MODES:
        raise ConfigError(f"runtime.execution_mode must be one of {sorted(VALID_EXECUTION_MODES)}")
    return normalized


def _normalize_shutdown_policy(value: str, *, valid: set[str], key_path: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in valid:
        raise ConfigError(f"{key_path} must be one of {sorted(valid)}")
    return normalized


def _normalize_choice(value: str, *, valid: set[str], key_path: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in valid:
        raise ConfigError(f"{key_path} must be one of {sorted(valid)}")
    return normalized


@dataclass(frozen=True)
class SystemConfig:
    timezone: str
    random_seed: int
    environment: str

    @property
    def is_dev_mode(self) -> bool:
        return self.environment.strip().lower() in DEV_ENVIRONMENTS


@dataclass(frozen=True)
class MarketConfig:
    futures_base_url: str
    use_binance_http_updater: bool
    refresh_universe_enabled: bool
    refresh_universe_sec: int
    top_n: int
    fast_frames: list[str]
    slow_frames: list[str]
    sem_fast: int
    sem_slow: int
    fetch_timeout_sec: float
    max_retries: int
    backoff_base_sec: float
    klines_limit: int
    request_timeout_sec: int
    request_retries: int
    bar_close_delay_ms: int
    server_time_cache_ms: int
    lookback_days: dict[str, int]
    runtime_snapshot_interval_sec: float
    runtime_price_websocket_enabled: bool
    runtime_price_websocket_url: str
    runtime_price_websocket_reconnect_sec: float


@dataclass(frozen=True)
class FramesConfig:
    anchor_interval: str
    fusion_intervals: list[str]
    lookback_rows: dict[str, int]


@dataclass(frozen=True)
class LabelsConfig:
    horizon_bars: int
    atr_multiplier: float
    atr_floor: float
    dqn_horizon_bars: int
    dqn_hold_threshold: float
    class_encoding: dict[str, int]
    interval_overrides: dict[str, dict[str, float | int]] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingModelConfig:
    enabled: bool
    min_rows: int
    max_age_minutes: int
    tail_rows: int
    random_state: int | None = None
    test_size: float | None = None
    n_estimators: int | None = None
    max_depth: int | None = None
    min_samples_leaf: int | None = None
    n_jobs: int | None = None
    max_samples: float | None = None
    learning_rate: float | None = None
    class_weight: str | None = None
    allowed_intervals: list[str] | None = None
    interval_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    estimator: str | None = None


@dataclass(frozen=True)
class TrainingHMMConfig:
    enabled: bool
    candidate_training_enabled: bool
    mandatory_paper_gate_enabled: bool
    live_influence_enabled: bool
    shadow_only: bool
    observation_schema_version: str
    n_states: int
    covariance_type: str
    min_rows: int
    max_age_minutes: int
    tail_rows: int
    random_state: int
    max_iter: int
    min_posterior_confidence: float


@dataclass(frozen=True)
class TrainingValidationConfig:
    walk_forward_folds: int
    purge_rows: int
    embargo_rows: int


@dataclass(frozen=True)
class TrainingPromotionConfig:
    enabled: bool
    allow_initial_promotion: bool
    allow_initial_promotion_without_paper_trades: bool
    allow_initial_paper_bootstrap_execution: bool
    require_outperform_incumbent: bool
    min_fold_count: int
    min_score_improvement: float
    min_metric_improvement: float
    max_metric_regression: float
    min_precision_macro: float
    min_roc_auc_ovr: float
    min_balanced_accuracy: float
    min_precision: float
    min_roc_auc: float
    require_directional_quality: bool
    min_long_precision: float
    min_short_precision: float
    min_long_recall: float
    min_short_recall: float
    max_hold_prediction_fraction: float
    min_directional_prediction_fraction: float
    require_paper_expectancy: bool
    paper_lookback_trades: int
    min_paper_trades: int
    min_paper_expectancy_usd: float
    min_paper_win_rate: float
    max_paper_drawdown_usd: float
    require_calibration_quality: bool
    max_expected_calibration_error: float
    max_brier_score: float


@dataclass(frozen=True)
class TrainingArtifactLifecycleConfig:
    enabled: bool
    prune_after_training: bool
    keep_last_candidates: int
    max_candidate_age_hours: float
    max_deleted_per_run: int
    protect_promoted: bool


@dataclass(frozen=True)
class TrainingConfig:
    rf: TrainingModelConfig
    gbm: TrainingModelConfig
    dqn: TrainingModelConfig
    hmm: TrainingHMMConfig
    validation: TrainingValidationConfig
    promotion: TrainingPromotionConfig
    artifact_lifecycle: TrainingArtifactLifecycleConfig
    retrain_cycles: dict[str, int]


@dataclass(frozen=True)
class SymbolsConfig:
    target_active_symbols: int
    ranked_universe_size: int
    max_failures_before_block: int
    auto_replace_symbols: bool
    allow_manual_unblock: bool
    manual_unblock_required: bool
    close_position_on_block: bool
    live_block_close_fail_closed: bool
    warmup_timeout_sec: float


@dataclass(frozen=True)
class SchedulingConfig:
    max_rf_workers: int
    max_gbm_workers: int
    max_dqn_workers: int
    loop_sleep_sec: float
    priority_intervals: list[str]
    allow_parallel_symbols: bool
    failure_cooldown_sec: float
    symbol_failure_retry_threshold: int
    symbol_backoff_first_sec: float
    symbol_backoff_repeat_sec: float
    max_submissions_per_loop: int
    max_submissions_per_symbol_per_loop: int
    max_model_submissions_per_loop: dict[str, int]
    retrain_min_new_bars: dict[str, dict[str, int]]


@dataclass(frozen=True)
class QualityGateConfig:
    min_rows: int
    min_precision_macro: float
    min_roc_auc_ovr: float
    min_balanced_accuracy: float
    anchor_interval: str


@dataclass(frozen=True)
class ConsensusConfig:
    enter_threshold: float
    flip_threshold: float
    exit_band: float
    direction_score_threshold: float
    require_agreement: bool


@dataclass(frozen=True)
class WeightConfig:
    rf: float
    gbm: float
    dqn: float


@dataclass(frozen=True)
class DQNInferenceConfig:
    gray_band_low: float
    gray_band_high: float
    action_hold: int
    action_long: int
    action_short: int


@dataclass(frozen=True)
class EVConfig:
    threshold: float
    meta_threshold: float
    dqn_adjustment_strength: float
    strong_anchor_probability: float
    weak_anchor_probability: float
    weak_anchor_lower_tf_scale: float


@dataclass(frozen=True)
class HMMInferenceConfig:
    enabled: bool
    candidate_shadow_enabled: bool
    mandatory_paper_gate_enabled: bool
    live_influence_enabled: bool
    require_valid_artifact_for_entry: bool
    min_posterior_confidence: float


@dataclass(frozen=True)
class InferenceConfig:
    minimum_models_required: int
    tail_rows: int
    quality: QualityGateConfig
    consensus: ConsensusConfig
    weights: WeightConfig
    dqn: DQNInferenceConfig
    hmm: HMMInferenceConfig
    ev: EVConfig
    active_mode: str = "Medium"
    mode_profiles: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def consensus_enter_threshold(self) -> float:
        return self.consensus.enter_threshold

    @property
    def consensus_flip_threshold(self) -> float:
        return self.consensus.flip_threshold

    @property
    def exit_band(self) -> float:
        return self.consensus.exit_band

    @property
    def rf_weight(self) -> float:
        return self.weights.rf

    @property
    def gbm_weight(self) -> float:
        return self.weights.gbm

    @property
    def dqn_weight(self) -> float:
        return self.weights.dqn

    @property
    def dqn_action_hold(self) -> int:
        return self.dqn.action_hold

    @property
    def dqn_action_long(self) -> int:
        return self.dqn.action_long

    @property
    def dqn_action_short(self) -> int:
        return self.dqn.action_short


@dataclass(frozen=True)
class FeatureStoreConfig:
    recipe_version: str
    include_optional_futures_context: bool


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool
    crisis_realized_vol_z: float
    high_realized_vol_z: float
    funding_abs_z_block: float
    open_interest_abs_z_block: float
    weak_trend_strength: float
    squeeze_width_threshold: float
    high_vol_score_scale: float
    stress_score_scale: float
    rangebound_score_scale: float


@dataclass(frozen=True)
class TradingConfig:
    fees_bps: float
    slippage_bps: float
    spread_bps: float
    funding_bps: float
    min_profit_atr_mult: float
    ev_threshold: float
    meta_threshold: float
    min_atr_fraction: float
    short_probability_threshold: float
    long_probability_threshold: float
    signal_cooldown_sec: float


@dataclass(frozen=True)
class AdaptiveModeConfig:
    enabled: bool
    hold_minutes: float
    cooldown_minutes: float
    medium_to_light_when_no_trades: bool
    heavy_on_loss_streak: int
    heavy_win_rate_lookback: int
    heavy_on_win_rate_below: float


@dataclass(frozen=True)
class RiskConfig:
    account_balance_usd: float
    risk_per_trade: float
    sl_atr_mult: float
    tp_atr_mult: float
    max_notional_usd: float
    min_notional_usd: float
    max_total_exposure_usd: float
    max_concurrent_positions: int
    max_symbol_notional_usd: float
    max_same_direction_exposure_usd: float
    max_risk_distortion_fraction: float
    daily_loss_limit_usd: float
    max_drawdown_fraction: float
    max_correlated_positions: int
    correlation_groups: dict[str, str]
    symbol_expectancy_guard: dict[str, object]


@dataclass(frozen=True)
class RuntimeConfig:
    dashboard_refresh_ms: int
    monitoring_fast_snapshot_interval_sec: float
    monitoring_heavy_snapshot_interval_sec: float
    market_poll_seconds: float
    execution_price_refresh_seconds: float
    execution_price_max_age_seconds: float
    allow_live_refresh_fallback: bool
    price_max_age_seconds: float
    paper_validation_min_decisions: int
    paper_validation_max_readiness_blocker_fraction: float
    paper_validation_min_entry_candidates: int
    execution_mode: str
    allow_unreconciled_live_mode: bool
    live_canary_enabled: bool
    live_canary_max_symbols: int
    live_canary_max_notional_usd: float
    live_canary_max_concurrent_positions: int
    live_canary_require_testnet: bool


@dataclass(frozen=True)
class ExchangeConfig:
    futures_rest_base_url: str
    api_key: str
    api_secret: str
    recv_window_ms: int
    use_testnet: bool
    position_reconciliation_enabled: bool
    require_reconciliation_for_live: bool
    max_reconciliation_age_sec: int


@dataclass(frozen=True)
class MonitoringConfig:
    host: str
    port: int
    embedded_in_all_mode: bool
    uvicorn_log_level: str
    auth_token: str
    max_events: int
    max_alerts: int
    max_market_updates: int
    max_recent_training_results: int
    max_recent_decisions: int
    max_recent_signals: int
    max_backtest_open_positions: int
    max_backtest_recent_closed: int
    max_registry_entries: int


@dataclass(frozen=True)
class ShutdownConfig:
    enabled: bool
    report_on_exit: bool
    paper_position_policy: str
    live_position_policy: str
    live_force_close_requires_reconciliation: bool
    coordinator_join_timeout_sec: float


@dataclass(frozen=True)
class TelegramConfig:
    enabled: bool
    bot_token: str
    chat_id: str
    poll_interval_sec: float
    command_cooldown_sec: float


@dataclass(frozen=True)
class AppConfig:
    system: SystemConfig
    market: MarketConfig
    symbols: SymbolsConfig
    frames: FramesConfig
    labels: LabelsConfig
    training: TrainingConfig
    scheduling: SchedulingConfig
    inference: InferenceConfig
    feature_store: FeatureStoreConfig
    regime: RegimeConfig
    trading: TradingConfig
    adaptive_mode: AdaptiveModeConfig
    risk: RiskConfig
    runtime: RuntimeConfig
    exchange: ExchangeConfig
    monitoring: MonitoringConfig
    shutdown: ShutdownConfig
    telegram: TelegramConfig


def _training_model(section: dict[str, Any], *, model_type: str = "") -> TrainingModelConfig:
    estimator: str | None = None
    if "estimator" in section:
        estimator = _get_optional_str(section, "estimator") or None
        if model_type == "gbm" and estimator is not None:
            estimator = _normalize_choice(estimator, valid=VALID_GBM_ESTIMATORS, key_path="training.gbm.estimator")
    return TrainingModelConfig(
        enabled=_get_bool(section, "enabled", True),
        estimator=estimator,
        min_rows=_get_int(section, "min_rows", 100, minimum=1),
        max_age_minutes=_get_int(section, "max_age_minutes", 120, minimum=1),
        tail_rows=_get_int(section, "tail_rows", 2000, minimum=1),
        random_state=int(section["random_state"]) if section.get("random_state") is not None else None,
        test_size=float(section["test_size"]) if section.get("test_size") is not None else None,
        n_estimators=int(section["n_estimators"]) if section.get("n_estimators") is not None else None,
        max_depth=int(section["max_depth"]) if section.get("max_depth") is not None else None,
        min_samples_leaf=int(section["min_samples_leaf"]) if section.get("min_samples_leaf") is not None else None,
        n_jobs=int(section["n_jobs"]) if section.get("n_jobs") is not None else None,
        max_samples=float(section["max_samples"]) if section.get("max_samples") is not None else None,
        learning_rate=float(section["learning_rate"]) if section.get("learning_rate") is not None else None,
        class_weight=str(section["class_weight"]) if section.get("class_weight") is not None else None,
        allowed_intervals=[str(v) for v in section.get("allowed_intervals", [])] or None,
        interval_overrides=section.get("interval_overrides", {})
        if isinstance(section.get("interval_overrides", {}), dict)
        else {},
    )


def _training_hmm(section: dict[str, Any]) -> TrainingHMMConfig:
    covariance_type = _normalize_choice(
        _get_str(section, "covariance_type", "diag"),
        valid=VALID_HMM_COVARIANCE_TYPES,
        key_path="training.hmm.covariance_type",
    )
    return TrainingHMMConfig(
        enabled=_get_bool(section, "enabled", False),
        candidate_training_enabled=_get_bool(section, "candidate_training_enabled", False),
        mandatory_paper_gate_enabled=_get_bool(section, "mandatory_paper_gate_enabled", False),
        live_influence_enabled=_get_bool(section, "live_influence_enabled", False),
        shadow_only=_get_bool(section, "shadow_only", True),
        observation_schema_version=_get_str(section, "observation_schema_version", "hmm_observation_v1"),
        n_states=_get_int(section, "n_states", 4, minimum=2),
        covariance_type=covariance_type,
        min_rows=_get_int(section, "min_rows", 1000, minimum=1),
        max_age_minutes=_get_int(section, "max_age_minutes", 240, minimum=1),
        tail_rows=_get_int(section, "tail_rows", 3000, minimum=1),
        random_state=_get_int(section, "random_state", 42),
        max_iter=_get_int(section, "max_iter", 200, minimum=1),
        min_posterior_confidence=_get_float(section, "min_posterior_confidence", 0.60, minimum=0.0),
    )


def _inference_hmm(section: dict[str, Any]) -> HMMInferenceConfig:
    return HMMInferenceConfig(
        enabled=_get_bool(section, "enabled", False),
        candidate_shadow_enabled=_get_bool(section, "candidate_shadow_enabled", False),
        mandatory_paper_gate_enabled=_get_bool(section, "mandatory_paper_gate_enabled", False),
        live_influence_enabled=_get_bool(section, "live_influence_enabled", False),
        require_valid_artifact_for_entry=_get_bool(section, "require_valid_artifact_for_entry", False),
        min_posterior_confidence=_get_float(section, "min_posterior_confidence", 0.60, minimum=0.0),
    )


def validate_settings(raw_settings: dict[str, Any]) -> AppConfig:
    root = _expect_mapping(raw_settings, "root")

    system_section = _get_section(root, "system")
    market_section = _get_section(root, "market")
    frames_section = _get_section(root, "frames")
    labels_section = _get_section(root, "labels")
    training_section = _get_section(root, "training")
    scheduling_section = _get_section(root, "scheduling")
    symbols_section = _get_section(root, "symbols") if isinstance(root.get("symbols", {}), dict) else {}
    inference_section = _get_section(root, "inference")
    feature_store_section = _get_section(root, "feature_store") if isinstance(root.get("feature_store", {}), dict) else {}
    regime_section = _get_section(root, "regime") if isinstance(root.get("regime", {}), dict) else {}
    trading_section = _get_section(root, "trading") if isinstance(root.get("trading", {}), dict) else {}
    adaptive_mode_section = _get_section(root, "adaptive_mode") if isinstance(root.get("adaptive_mode", {}), dict) else {}
    risk_section = _get_section(root, "risk") if isinstance(root.get("risk", {}), dict) else {}
    runtime_section = _get_section(root, "runtime")
    exchange_section = _get_section(root, "exchange") if isinstance(root.get("exchange", {}), dict) else {}
    monitoring_section = _get_section(root, "monitoring")
    shutdown_section = _get_section(root, "shutdown") if isinstance(root.get("shutdown", {}), dict) else {}
    telegram_section = _get_section(root, "telegram") if isinstance(root.get("telegram", {}), dict) else {}

    environment = _normalize_environment(_get_str(system_section, "environment", "dev"))
    system = SystemConfig(
        timezone=_get_str(system_section, "timezone", "UTC"),
        random_seed=_get_int(system_section, "random_seed", 42),
        environment=environment,
    )

    market = MarketConfig(
        futures_base_url=_get_str(market_section, "futures_base_url", "https://fapi.binance.com"),
        use_binance_http_updater=_get_bool(market_section, "use_binance_http_updater", True),
        refresh_universe_enabled=_get_bool(market_section, "refresh_universe_enabled", True),
        refresh_universe_sec=_get_int(market_section, "refresh_universe_sec", 300, minimum=1),
        top_n=_get_int(market_section, "top_n", 10, minimum=1),
        fast_frames=_get_str_list(market_section, "fast_frames", ["1m", "3m", "5m", "15m"]),
        slow_frames=_get_str_list(market_section, "slow_frames", ["30m", "1h", "4h", "1d"]),
        sem_fast=_get_int(market_section, "sem_fast", 8, minimum=1),
        sem_slow=_get_int(market_section, "sem_slow", 4, minimum=1),
        fetch_timeout_sec=_get_float(market_section, "fetch_timeout_sec", 10.0, minimum=0.1),
        max_retries=_get_int(market_section, "max_retries", 4, minimum=1),
        backoff_base_sec=_get_float(market_section, "backoff_base_sec", 0.75, minimum=0.0),
        klines_limit=_get_int(market_section, "klines_limit", 1000, minimum=1),
        request_timeout_sec=_get_int(market_section, "request_timeout_sec", 10, minimum=1),
        request_retries=_get_int(market_section, "request_retries", 3, minimum=1),
        bar_close_delay_ms=_get_int(market_section, "bar_close_delay_ms", 2000, minimum=0),
        server_time_cache_ms=_get_int(market_section, "server_time_cache_ms", 180000, minimum=0),
        lookback_days=_get_int_mapping(
            market_section,
            "lookback_days",
            {"1m": 5, "3m": 5, "5m": 7, "15m": 10, "30m": 10, "1h": 30, "4h": 60, "1d": 120},
        ),
        runtime_snapshot_interval_sec=_get_float(market_section, "runtime_snapshot_interval_sec", 5.0, minimum=1.0),
        runtime_price_websocket_enabled=_get_bool(market_section, "runtime_price_websocket_enabled", False),
        runtime_price_websocket_url=_get_str(
            market_section,
            "runtime_price_websocket_url",
            "wss://fstream.binance.com/ws/!markPrice@arr@1s",
        ),
        runtime_price_websocket_reconnect_sec=_get_float(
            market_section,
            "runtime_price_websocket_reconnect_sec",
            5.0,
            minimum=0.5,
        ),
    )

    frames = FramesConfig(
        anchor_interval=_get_str(frames_section, "anchor_interval", "15m"),
        fusion_intervals=_get_str_list(
            frames_section,
            "fusion_intervals",
            ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
        ),
        lookback_rows=_get_int_mapping(
            frames_section,
            "lookback_rows",
            {"1m": 3000, "3m": 2000, "5m": 1500, "15m": 1200, "30m": 1100, "1h": 1000, "4h": 800, "1d": 500},
        ),
    )

    class_encoding_raw = labels_section.get("class_encoding", {"short": 0, "hold": 1, "long": 2})
    if not isinstance(class_encoding_raw, dict):
        raise ConfigError("labels.class_encoding must be a mapping")

    labels = LabelsConfig(
        horizon_bars=_get_int(labels_section, "horizon_bars", 3, minimum=1),
        atr_multiplier=_get_float(labels_section, "atr_multiplier", 0.25, minimum=0.0),
        atr_floor=_get_float(labels_section, "atr_floor", 0.0004, minimum=0.0),
        dqn_horizon_bars=_get_int(labels_section, "dqn_horizon_bars", 1, minimum=1),
        dqn_hold_threshold=_get_float(labels_section, "dqn_hold_threshold", 0.0015, minimum=0.0),
        class_encoding={str(k): int(v) for k, v in class_encoding_raw.items()},
        interval_overrides=labels_section.get("interval_overrides", {})
        if isinstance(labels_section.get("interval_overrides", {}), dict)
        else {},
    )

    symbols = SymbolsConfig(
        target_active_symbols=_get_int(
            symbols_section,
            "target_active_symbols",
            _get_int(market_section, "top_n", 10, minimum=1),
            minimum=1,
        ),
        ranked_universe_size=_get_int(symbols_section, "ranked_universe_size", 20, minimum=1),
        max_failures_before_block=_get_int(symbols_section, "max_failures_before_block", 3, minimum=1),
        auto_replace_symbols=_get_bool(symbols_section, "auto_replace_symbols", True),
        allow_manual_unblock=_get_bool(symbols_section, "allow_manual_unblock", True),
        manual_unblock_required=_get_bool(
            symbols_section,
            "manual_unblock_required",
            _get_bool(symbols_section, "allow_manual_unblock", True),
        ),
        close_position_on_block=_get_bool(symbols_section, "close_position_on_block", True),
        live_block_close_fail_closed=_get_bool(symbols_section, "live_block_close_fail_closed", True),
        warmup_timeout_sec=_get_float(symbols_section, "warmup_timeout_sec", 14400.0, minimum=0.0),
    )

    validation_section = (
        _expect_mapping(training_section.get("validation", {}), "training.validation")
        if isinstance(training_section.get("validation", {}), dict)
        else {}
    )
    promotion_section = (
        _expect_mapping(training_section.get("promotion", {}), "training.promotion")
        if isinstance(training_section.get("promotion", {}), dict)
        else {}
    )
    artifact_lifecycle_section = (
        _expect_mapping(training_section.get("artifact_lifecycle", {}), "training.artifact_lifecycle")
        if isinstance(training_section.get("artifact_lifecycle", {}), dict)
        else {}
    )
    training_hmm_section = (
        _expect_mapping(training_section.get("hmm", {}), "training.hmm")
        if isinstance(training_section.get("hmm", {}), dict)
        else {}
    )

    training = TrainingConfig(
        rf=_training_model(_get_section(training_section, "rf"), model_type="rf"),
        gbm=_training_model(_get_section(training_section, "gbm"), model_type="gbm"),
        dqn=_training_model(_get_section(training_section, "dqn"), model_type="dqn"),
        hmm=_training_hmm(training_hmm_section),
        validation=TrainingValidationConfig(
            walk_forward_folds=_get_int(validation_section, "walk_forward_folds", 4, minimum=1),
            purge_rows=_get_int(validation_section, "purge_rows", 2, minimum=0),
            embargo_rows=_get_int(validation_section, "embargo_rows", 1, minimum=0),
        ),
        promotion=TrainingPromotionConfig(
            enabled=_get_bool(promotion_section, "enabled", True),
            allow_initial_promotion=_get_bool(promotion_section, "allow_initial_promotion", True),
            # Outcome-free promotion must never be enabled implicitly. Any future
            # bootstrap workflow remains paper-only and requires explicit governance.
            allow_initial_promotion_without_paper_trades=_get_bool(
                promotion_section,
                "allow_initial_promotion_without_paper_trades",
                False,
            ),
            # A paper-only bootstrap incumbent is an explicit deadlock breaker for
            # collecting clean paper outcomes. It is never implicitly enabled and
            # must remain non-live in artifact/readiness validation.
            allow_initial_paper_bootstrap_execution=_get_bool(
                promotion_section,
                "allow_initial_paper_bootstrap_execution",
                False,
            ),
            require_outperform_incumbent=_get_bool(promotion_section, "require_outperform_incumbent", True),
            min_fold_count=_get_int(promotion_section, "min_fold_count", 3, minimum=1),
            min_score_improvement=_get_float(promotion_section, "min_score_improvement", 0.002),
            min_metric_improvement=_get_float(promotion_section, "min_metric_improvement", 0.001),
            max_metric_regression=_get_float(promotion_section, "max_metric_regression", 0.01, minimum=0.0),
            min_precision_macro=_get_float(promotion_section, "min_precision_macro", 0.16, minimum=0.0),
            min_roc_auc_ovr=_get_float(promotion_section, "min_roc_auc_ovr", 0.56, minimum=0.0),
            min_balanced_accuracy=_get_float(promotion_section, "min_balanced_accuracy", 0.38, minimum=0.0),
            min_precision=_get_float(promotion_section, "min_precision", 0.52, minimum=0.0),
            min_roc_auc=_get_float(promotion_section, "min_roc_auc", 0.56, minimum=0.0),
            require_directional_quality=_get_bool(promotion_section, "require_directional_quality", True),
            min_long_precision=_get_float(promotion_section, "min_long_precision", 0.20, minimum=0.0),
            min_short_precision=_get_float(promotion_section, "min_short_precision", 0.20, minimum=0.0),
            min_long_recall=_get_float(promotion_section, "min_long_recall", 0.05, minimum=0.0),
            min_short_recall=_get_float(promotion_section, "min_short_recall", 0.05, minimum=0.0),
            max_hold_prediction_fraction=_get_float(promotion_section, "max_hold_prediction_fraction", 0.85, minimum=0.0),
            min_directional_prediction_fraction=_get_float(promotion_section, "min_directional_prediction_fraction", 0.05, minimum=0.0),
            require_paper_expectancy=_get_bool(promotion_section, "require_paper_expectancy", True),
            paper_lookback_trades=_get_int(promotion_section, "paper_lookback_trades", 25, minimum=1),
            min_paper_trades=_get_int(promotion_section, "min_paper_trades", 5, minimum=0),
            min_paper_expectancy_usd=_get_float(promotion_section, "min_paper_expectancy_usd", 0.0),
            min_paper_win_rate=_get_float(promotion_section, "min_paper_win_rate", 0.45, minimum=0.0),
            max_paper_drawdown_usd=_get_float(promotion_section, "max_paper_drawdown_usd", -50.0),
            require_calibration_quality=_get_bool(promotion_section, "require_calibration_quality", True),
            max_expected_calibration_error=_get_float(promotion_section, "max_expected_calibration_error", 0.25, minimum=0.0),
            max_brier_score=_get_float(promotion_section, "max_brier_score", 0.75, minimum=0.0),
        ),
        artifact_lifecycle=TrainingArtifactLifecycleConfig(
            enabled=_get_bool(artifact_lifecycle_section, "enabled", True),
            prune_after_training=_get_bool(artifact_lifecycle_section, "prune_after_training", True),
            keep_last_candidates=_get_int(artifact_lifecycle_section, "keep_last_candidates", 3, minimum=0),
            max_candidate_age_hours=_get_float(artifact_lifecycle_section, "max_candidate_age_hours", 24.0, minimum=0.0),
            max_deleted_per_run=_get_int(artifact_lifecycle_section, "max_deleted_per_run", 200, minimum=1),
            protect_promoted=_get_bool(artifact_lifecycle_section, "protect_promoted", True),
        ),
        retrain_cycles=_get_int_mapping(
            training_section,
            "retrain_cycles",
            {"15m": 1, "5m": 2, "3m": 4, "1m": 7},
        ),
    )

    scheduling = SchedulingConfig(
        max_rf_workers=_get_int(
            scheduling_section,
            "max_rf_workers",
            int(scheduling_section.get("max_tree_workers", 2)),
            minimum=1,
        ),
        max_gbm_workers=_get_int(
            scheduling_section,
            "max_gbm_workers",
            int(scheduling_section.get("max_tree_workers", 2)),
            minimum=1,
        ),
        max_dqn_workers=_get_int(scheduling_section, "max_dqn_workers", 1, minimum=1),
        loop_sleep_sec=_get_float(scheduling_section, "loop_sleep_sec", 1.0, minimum=0.1),
        priority_intervals=_get_str_list(scheduling_section, "priority_intervals", ["15m", "5m", "3m", "1m"]),
        allow_parallel_symbols=_get_bool(scheduling_section, "allow_parallel_symbols", True),
        failure_cooldown_sec=_get_float(scheduling_section, "failure_cooldown_sec", 60.0, minimum=0.0),
        symbol_failure_retry_threshold=_get_int(
            scheduling_section,
            "symbol_failure_retry_threshold",
            5,
            minimum=1,
        ),
        symbol_backoff_first_sec=_get_float(
            scheduling_section,
            "symbol_backoff_first_sec",
            600.0,
            minimum=1.0,
        ),
        symbol_backoff_repeat_sec=_get_float(
            scheduling_section,
            "symbol_backoff_repeat_sec",
            1800.0,
            minimum=1.0,
        ),
        max_submissions_per_loop=_get_int(
            scheduling_section,
            "max_submissions_per_loop",
            3,
            minimum=1,
        ),
        max_submissions_per_symbol_per_loop=_get_int(
            scheduling_section,
            "max_submissions_per_symbol_per_loop",
            1,
            minimum=1,
        ),
        max_model_submissions_per_loop=_get_int_mapping(
            scheduling_section,
            "max_model_submissions_per_loop",
            {"rf": 1, "gbm": 1, "dqn": 1},
        ),
        retrain_min_new_bars=_get_nested_int_mapping(
            scheduling_section,
            "retrain_min_new_bars",
            {
                "rf": {"1m": 60, "3m": 40, "5m": 24, "15m": 12},
                "gbm": {"1m": 60, "3m": 40, "5m": 24, "15m": 12},
                "dqn": {"1m": 30, "3m": 20, "5m": 12, "15m": 8},
            },
        ),
    )

    quality_section = _expect_mapping(inference_section.get("quality", {}), "inference.quality")
    consensus_section = _expect_mapping(inference_section.get("consensus", {}), "inference.consensus")
    weights_section = _expect_mapping(inference_section.get("weights", {}), "inference.weights")
    dqn_section = _expect_mapping(inference_section.get("dqn", {}), "inference.dqn")
    inference_hmm_section = (
        _expect_mapping(inference_section.get("hmm", {}), "inference.hmm")
        if isinstance(inference_section.get("hmm", {}), dict)
        else {}
    )
    ev_section = (
        _expect_mapping(inference_section.get("ev", {}), "inference.ev")
        if isinstance(inference_section.get("ev", {}), dict)
        else {}
    )

    feature_store = FeatureStoreConfig(
        recipe_version=_get_str(feature_store_section, "recipe_version", "feature_store_v2"),
        include_optional_futures_context=_get_bool(
            feature_store_section,
            "include_optional_futures_context",
            True,
        ),
    )

    regime = RegimeConfig(
        enabled=_get_bool(regime_section, "enabled", True),
        crisis_realized_vol_z=_get_float(regime_section, "crisis_realized_vol_z", 3.0),
        high_realized_vol_z=_get_float(regime_section, "high_realized_vol_z", 1.75),
        funding_abs_z_block=_get_float(regime_section, "funding_abs_z_block", 3.0),
        open_interest_abs_z_block=_get_float(regime_section, "open_interest_abs_z_block", 3.0),
        weak_trend_strength=_get_float(regime_section, "weak_trend_strength", 0.35),
        squeeze_width_threshold=_get_float(regime_section, "squeeze_width_threshold", 0.025),
        high_vol_score_scale=_get_float(regime_section, "high_vol_score_scale", 0.70),
        stress_score_scale=_get_float(regime_section, "stress_score_scale", 0.60),
        rangebound_score_scale=_get_float(regime_section, "rangebound_score_scale", 0.75),
    )

    trading = TradingConfig(
        fees_bps=_get_float(trading_section, "fees_bps", 3.0, minimum=0.0),
        slippage_bps=_get_float(trading_section, "slippage_bps", 2.0, minimum=0.0),
        spread_bps=_get_float(trading_section, "spread_bps", 0.0, minimum=0.0),
        funding_bps=_get_float(trading_section, "funding_bps", 0.0, minimum=0.0),
        min_profit_atr_mult=_get_float(trading_section, "min_profit_atr_mult", 0.30, minimum=0.0),
        ev_threshold=_get_float(trading_section, "ev_threshold", 0.0),
        meta_threshold=_get_float(trading_section, "meta_threshold", 0.55, minimum=0.0),
        min_atr_fraction=_get_float(trading_section, "min_atr_fraction", 0.0, minimum=0.0),
        short_probability_threshold=_get_float(
            trading_section,
            "short_probability_threshold",
            0.60,
            minimum=0.0,
        ),
        long_probability_threshold=_get_float(
            trading_section,
            "long_probability_threshold",
            0.55,
            minimum=0.0,
        ),
        signal_cooldown_sec=_get_float(trading_section, "signal_cooldown_sec", 45.0, minimum=0.0),
    )

    inference = InferenceConfig(
        minimum_models_required=_get_int(inference_section, "minimum_models_required", 2, minimum=1),
        tail_rows=_get_int(inference_section, "tail_rows", 500, minimum=1),
        quality=QualityGateConfig(
            min_rows=_get_int(quality_section, "min_rows", 250, minimum=1),
            min_precision_macro=_get_float(quality_section, "min_precision_macro", 0.10, minimum=0.0),
            min_roc_auc_ovr=_get_float(quality_section, "min_roc_auc_ovr", 0.50, minimum=0.0),
            min_balanced_accuracy=_get_float(
                quality_section,
                "min_balanced_accuracy",
                0.34,
                minimum=0.0,
            ),
            anchor_interval=_get_str(quality_section, "anchor_interval", frames.anchor_interval),
        ),
        consensus=ConsensusConfig(
            enter_threshold=_get_float(consensus_section, "enter_threshold", 0.18, minimum=0.0),
            flip_threshold=_get_float(consensus_section, "flip_threshold", 0.12, minimum=0.0),
            exit_band=_get_float(consensus_section, "exit_band", 0.06, minimum=0.0),
            direction_score_threshold=_get_float(
                consensus_section,
                "direction_score_threshold",
                0.05,
                minimum=0.0,
            ),
            require_agreement=_get_bool(consensus_section, "require_agreement", True),
        ),
        weights=WeightConfig(
            rf=_get_float(weights_section, "rf", 0.5, minimum=0.0),
            gbm=_get_float(weights_section, "gbm", 0.4, minimum=0.0),
            dqn=_get_float(weights_section, "dqn", 0.1, minimum=0.0),
        ),
        dqn=DQNInferenceConfig(
            gray_band_low=_get_float(dqn_section, "gray_band_low", 0.45, minimum=0.0),
            gray_band_high=_get_float(dqn_section, "gray_band_high", 0.55, minimum=0.0),
            action_hold=_get_int(dqn_section, "action_hold", 0),
            action_long=_get_int(dqn_section, "action_long", 1),
            action_short=_get_int(dqn_section, "action_short", 2),
        ),
        hmm=_inference_hmm(inference_hmm_section),
        ev=EVConfig(
            threshold=_get_float(ev_section, "threshold", trading.ev_threshold),
            meta_threshold=_get_float(ev_section, "meta_threshold", trading.meta_threshold, minimum=0.0),
            dqn_adjustment_strength=_get_float(
                ev_section,
                "dqn_adjustment_strength",
                0.15,
                minimum=0.0,
            ),
            strong_anchor_probability=_get_float(
                ev_section,
                "strong_anchor_probability",
                0.62,
                minimum=0.0,
            ),
            weak_anchor_probability=_get_float(
                ev_section,
                "weak_anchor_probability",
                0.55,
                minimum=0.0,
            ),
            weak_anchor_lower_tf_scale=_get_float(
                ev_section,
                "weak_anchor_lower_tf_scale",
                0.60,
                minimum=0.0,
            ),
        ),
        active_mode=_get_str(inference_section, "active_mode", "Medium"),
        mode_profiles=_get_nested_float_mapping(
            inference_section,
            "mode_profiles",
            {
                "Light": {
                    "ev_threshold": 0.00015,
                    "min_atr_fraction": 0.0012,
                    "short_probability_threshold": 0.57,
                    "long_probability_threshold": 0.52,
                },
                "Medium": {
                    "ev_threshold": trading.ev_threshold,
                    "min_atr_fraction": trading.min_atr_fraction,
                    "short_probability_threshold": trading.short_probability_threshold,
                    "long_probability_threshold": trading.long_probability_threshold,
                },
                "Heavy": {
                    "ev_threshold": 0.00035,
                    "min_atr_fraction": 0.0018,
                    "short_probability_threshold": 0.64,
                    "long_probability_threshold": 0.60,
                },
            },
        ),
    )

    adaptive_mode = AdaptiveModeConfig(
        enabled=_get_bool(adaptive_mode_section, "enabled", False),
        hold_minutes=_get_float(adaptive_mode_section, "hold_minutes", 30.0, minimum=1.0),
        cooldown_minutes=_get_float(adaptive_mode_section, "cooldown_minutes", 30.0, minimum=1.0),
        medium_to_light_when_no_trades=_get_bool(adaptive_mode_section, "medium_to_light_when_no_trades", True),
        heavy_on_loss_streak=_get_int(adaptive_mode_section, "heavy_on_loss_streak", 3, minimum=1),
        heavy_win_rate_lookback=_get_int(adaptive_mode_section, "heavy_win_rate_lookback", 8, minimum=1),
        heavy_on_win_rate_below=_get_float(adaptive_mode_section, "heavy_on_win_rate_below", 0.25, minimum=0.0),
    )

    correlation_groups_raw = risk_section.get("correlation_groups", {})
    if not isinstance(correlation_groups_raw, dict):
        raise ConfigError("risk.correlation_groups must be a mapping")

    risk = RiskConfig(
        account_balance_usd=_get_float(risk_section, "account_balance_usd", 1000.0, minimum=0.0),
        risk_per_trade=_get_float(risk_section, "risk_per_trade", 0.01, minimum=0.0),
        sl_atr_mult=_get_float(risk_section, "sl_atr_mult", 1.5, minimum=0.0),
        tp_atr_mult=_get_float(risk_section, "tp_atr_mult", 2.0, minimum=0.0),
        max_notional_usd=_get_float(risk_section, "max_notional_usd", 5000.0, minimum=0.0),
        min_notional_usd=_get_float(risk_section, "min_notional_usd", 25.0, minimum=0.0),
        max_total_exposure_usd=_get_float(
            risk_section,
            "max_total_exposure_usd",
            7500.0,
            minimum=0.0,
        ),
        max_concurrent_positions=_get_int(risk_section, "max_concurrent_positions", 6, minimum=1),
        max_symbol_notional_usd=_get_float(
            risk_section,
            "max_symbol_notional_usd",
            _get_float(risk_section, "max_notional_usd", 5000.0, minimum=0.0),
            minimum=0.0,
        ),
        max_same_direction_exposure_usd=_get_float(
            risk_section,
            "max_same_direction_exposure_usd",
            4500.0,
            minimum=0.0,
        ),
        max_risk_distortion_fraction=_get_float(
            risk_section,
            "max_risk_distortion_fraction",
            0.35,
            minimum=0.0,
        ),
        daily_loss_limit_usd=_get_float(risk_section, "daily_loss_limit_usd", 75.0, minimum=0.0),
        max_drawdown_fraction=_get_float(risk_section, "max_drawdown_fraction", 0.12, minimum=0.0),
        max_correlated_positions=_get_int(risk_section, "max_correlated_positions", 2, minimum=1),
        correlation_groups={str(k).upper(): str(v) for k, v in correlation_groups_raw.items()},
        symbol_expectancy_guard=dict(risk_section.get("symbol_expectancy_guard", {}) or {}),
    )

    runtime = RuntimeConfig(
        dashboard_refresh_ms=_get_int(runtime_section, "dashboard_refresh_ms", 400, minimum=100),
        monitoring_fast_snapshot_interval_sec=_get_float(runtime_section, "monitoring_fast_snapshot_interval_sec", 0.5, minimum=0.25),
        monitoring_heavy_snapshot_interval_sec=_get_float(runtime_section, "monitoring_heavy_snapshot_interval_sec", 5.0, minimum=1.0),
        market_poll_seconds=_get_float(runtime_section, "market_poll_seconds", 10.0, minimum=0.1),
        execution_price_refresh_seconds=_get_float(runtime_section, "execution_price_refresh_seconds", 1.0, minimum=0.1),
        execution_price_max_age_seconds=_get_float(runtime_section, "execution_price_max_age_seconds", 3.0, minimum=0.5),
        allow_live_refresh_fallback=_get_bool(runtime_section, "allow_live_refresh_fallback", False),
        price_max_age_seconds=_get_float(runtime_section, "price_max_age_seconds", 180.0, minimum=1.0),
        paper_validation_min_decisions=_get_int(runtime_section, "paper_validation_min_decisions", 100, minimum=1),
        paper_validation_max_readiness_blocker_fraction=_get_float(
            runtime_section,
            "paper_validation_max_readiness_blocker_fraction",
            0.35,
            minimum=0.0,
        ),
        paper_validation_min_entry_candidates=_get_int(
            runtime_section,
            "paper_validation_min_entry_candidates",
            5,
            minimum=0,
        ),
        execution_mode=_normalize_execution_mode(_get_str(runtime_section, "execution_mode", "paper")),
        allow_unreconciled_live_mode=_get_bool(runtime_section, "allow_unreconciled_live_mode", False),
        live_canary_enabled=_get_bool(runtime_section, "live_canary_enabled", True),
        live_canary_max_symbols=_get_int(runtime_section, "live_canary_max_symbols", 1, minimum=1),
        live_canary_max_notional_usd=_get_float(runtime_section, "live_canary_max_notional_usd", 50.0, minimum=0.0),
        live_canary_max_concurrent_positions=_get_int(runtime_section, "live_canary_max_concurrent_positions", 1, minimum=1),
        live_canary_require_testnet=_get_bool(runtime_section, "live_canary_require_testnet", True),
    )

    inline_exchange_api_key = _get_optional_str(exchange_section, "api_key", "")
    inline_exchange_api_secret = _get_optional_str(exchange_section, "api_secret", "")
    _reject_inline_secret(inline_exchange_api_key, "exchange.api_key", BINANCE_API_KEY_ENV_VARS)
    _reject_inline_secret(inline_exchange_api_secret, "exchange.api_secret", BINANCE_API_SECRET_ENV_VARS)

    exchange = ExchangeConfig(
        futures_rest_base_url=_get_str(exchange_section, "futures_rest_base_url", market.futures_base_url),
        api_key=_resolve_env_only_secret(BINANCE_API_KEY_ENV_VARS),
        api_secret=_resolve_env_only_secret(BINANCE_API_SECRET_ENV_VARS),
        recv_window_ms=_get_int(exchange_section, "recv_window_ms", 5000, minimum=1),
        use_testnet=_get_bool(exchange_section, "use_testnet", False),
        position_reconciliation_enabled=_get_bool(
            exchange_section,
            "position_reconciliation_enabled",
            False,
        ),
        require_reconciliation_for_live=_get_bool(
            exchange_section,
            "require_reconciliation_for_live",
            True,
        ),
        max_reconciliation_age_sec=_get_int(
            exchange_section,
            "max_reconciliation_age_sec",
            30,
            minimum=1,
        ),
    )

    monitoring = MonitoringConfig(
        host=_get_str(monitoring_section, "host", "127.0.0.1"),
        port=_get_int(monitoring_section, "port", 8000, minimum=1),
        embedded_in_all_mode=_get_bool(monitoring_section, "embedded_in_all_mode", False),
        uvicorn_log_level=_get_str(monitoring_section, "uvicorn_log_level", "info"),
        auth_token=_resolve_secret(
            _get_optional_str(monitoring_section, "auth_token", ""),
            MONITORING_TOKEN_ENV_VARS,
        ),
        max_events=_get_int(monitoring_section, "max_events", 300, minimum=1),
        max_alerts=_get_int(monitoring_section, "max_alerts", 100, minimum=1),
        max_market_updates=_get_int(monitoring_section, "max_market_updates", 100, minimum=1),
        max_recent_training_results=_get_int(
            monitoring_section,
            "max_recent_training_results",
            50,
            minimum=1,
        ),
        max_recent_decisions=_get_int(monitoring_section, "max_recent_decisions", 50, minimum=1),
        max_recent_signals=_get_int(monitoring_section, "max_recent_signals", 100, minimum=1),
        max_backtest_open_positions=_get_int(
            monitoring_section,
            "max_backtest_open_positions",
            100,
            minimum=1,
        ),
        max_backtest_recent_closed=_get_int(
            monitoring_section,
            "max_backtest_recent_closed",
            50,
            minimum=1,
        ),
        max_registry_entries=_get_int(monitoring_section, "max_registry_entries", 300, minimum=1),
    )

    shutdown = ShutdownConfig(
        enabled=_get_bool(shutdown_section, "enabled", True),
        report_on_exit=_get_bool(shutdown_section, "report_on_exit", True),
        paper_position_policy=_normalize_shutdown_policy(
            _get_str(shutdown_section, "paper_position_policy", "snapshot_only"),
            valid=VALID_PAPER_SHUTDOWN_POLICIES,
            key_path="shutdown.paper_position_policy",
        ),
        live_position_policy=_normalize_shutdown_policy(
            _get_str(shutdown_section, "live_position_policy", "fail_closed_snapshot_only"),
            valid=VALID_LIVE_SHUTDOWN_POLICIES,
            key_path="shutdown.live_position_policy",
        ),
        live_force_close_requires_reconciliation=_get_bool(
            shutdown_section,
            "live_force_close_requires_reconciliation",
            True,
        ),
        coordinator_join_timeout_sec=_get_float(
            shutdown_section,
            "coordinator_join_timeout_sec",
            5.0,
            minimum=0.1,
        ),
    )

    telegram = TelegramConfig(
        enabled=_get_bool(telegram_section, "enabled", False),
        bot_token=_resolve_secret(
            _get_optional_str(telegram_section, "bot_token", ""),
            TELEGRAM_BOT_TOKEN_ENV_VARS,
        ),
        chat_id=_resolve_secret(
            _get_optional_str(telegram_section, "chat_id", ""),
            TELEGRAM_CHAT_ID_ENV_VARS,
        ),
        poll_interval_sec=_get_float(telegram_section, "poll_interval_sec", 2.0, minimum=0.5),
        command_cooldown_sec=_get_float(
            telegram_section,
            "command_cooldown_sec",
            2.0,
            minimum=0.0,
        ),
    )

    return AppConfig(
        system=system,
        market=market,
        symbols=symbols,
        frames=frames,
        labels=labels,
        training=training,
        scheduling=scheduling,
        inference=inference,
        feature_store=feature_store,
        regime=regime,
        trading=trading,
        adaptive_mode=adaptive_mode,
        risk=risk,
        runtime=runtime,
        exchange=exchange,
        monitoring=monitoring,
        shutdown=shutdown,
        telegram=telegram,
    )


def load_settings(paths_or_path: Any) -> dict[str, Any]:
    if isinstance(paths_or_path, Path):
        import yaml

        with paths_or_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        data = payload if isinstance(payload, dict) else {}
    else:
        data = paths_or_path.load_yaml(paths_or_path.system_settings_path)

    _ = validate_settings(data)
    return data


def load_app_config(paths) -> AppConfig:
    raw = load_settings(paths)
    return validate_settings(raw)


def validate_runtime_secrets(config: AppConfig) -> None:
    """Validate startup secret requirements for the active runtime mode."""
    environment = config.system.environment.strip().lower()

    if config.telegram.enabled:
        missing: list[str] = []
        if not config.telegram.bot_token.strip():
            missing.append("telegram.bot_token or env AUTO_TRADER_TELEGRAM_BOT_TOKEN")
        if not config.telegram.chat_id.strip():
            missing.append("telegram.chat_id or env AUTO_TRADER_TELEGRAM_CHAT_ID")
        if missing:
            raise SecretValidationError("missing required Telegram secrets: " + ", ".join(missing))

    if config.runtime.execution_mode == "live":
        missing_exchange: list[str] = []
        if not config.exchange.api_key.strip():
            missing_exchange.append("env AUTO_TRADER_BINANCE_API_KEY")
        if not config.exchange.api_secret.strip():
            missing_exchange.append("env AUTO_TRADER_BINANCE_API_SECRET")
        if missing_exchange:
            raise SecretValidationError(
                "missing required Binance exchange secrets: " + ", ".join(missing_exchange)
            )

