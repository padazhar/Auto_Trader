from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.artifact_contract import artifact_paths
from core.config_contract import load_app_config


@dataclass(frozen=True)
class PathManager:
    project_root: Path
    settings: dict[str, Any]
    config: Any

    @classmethod
    def from_project_root(cls, project_root: Path) -> "PathManager":
        project_root = Path(project_root).resolve()
        settings_path = project_root / "config" / "system_settings.yaml"
        settings: dict[str, Any] = {}
        if settings_path.exists():
            with settings_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if isinstance(loaded, dict):
                settings = loaded
        temp = cls(project_root=project_root, settings=settings, config=None)
        config = load_app_config(temp)
        instance = cls(project_root=project_root, settings=settings, config=config)
        instance.ensure_directories()
        return instance

    def load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}

    def ensure_directories(self) -> None:
        for directory in [
            self.config_dir,
            self.data_dir,
            self.runtime_dir,
            self.monitoring_runtime_dir,
            self.training_progress_dir,
            self.logs_dir,
            self.market_storage_dir,
            self.model_storage_dir,
            self.rf_models_dir,
            self.gbm_models_dir,
            self.dqn_models_dir,
            self.hmm_models_dir,
            self.reports_dir,
            self.signal_logs_dir,
            self.trade_history_dir,
            self.scripts_dir,
            self.control_plane_actions_path.parent,
            self.market_runtime_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def system_settings_path(self) -> Path:
        return self.config_dir / "system_settings.yaml"

    @property
    def tradable_symbols_path(self) -> Path:
        return self.config_dir / "tradable_symbols.yaml"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def monitoring_runtime_dir(self) -> Path:
        return self.runtime_dir / "monitoring"

    @property
    def training_progress_dir(self) -> Path:
        return self.runtime_dir / "training_progress"

    @property
    def monitoring_static_dir(self) -> Path:
        return self.project_root / "monitoring" / "static"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def signal_logs_dir(self) -> Path:
        return self.logs_dir / "signal_logs"

    @property
    def market_storage_dir(self) -> Path:
        return self.data_dir / "market_storage"

    @property
    def model_storage_dir(self) -> Path:
        return self.data_dir / "model_storage"

    @property
    def rf_models_dir(self) -> Path:
        return self.model_storage_dir / "random_forest_models"

    @property
    def gbm_models_dir(self) -> Path:
        return self.model_storage_dir / "gradient_boost_models"

    @property
    def dqn_models_dir(self) -> Path:
        return self.model_storage_dir / "reinforcement_agents"

    @property
    def hmm_models_dir(self) -> Path:
        return self.model_storage_dir / "hmm_regime_models"

    @property
    def reports_dir(self) -> Path:
        return self.runtime_dir / "reports"

    @property
    def trade_history_dir(self) -> Path:
        return self.data_dir / "trade_history"

    @property
    def scripts_dir(self) -> Path:
        return self.project_root / "scripts"

    @property
    def symbol_state_path(self) -> Path:
        return self.runtime_dir / "symbol_state.json"

    @property
    def symbol_actions_path(self) -> Path:
        return self.runtime_dir / "symbol_actions.json"

    @property
    def control_plane_state_path(self) -> Path:
        return self.runtime_dir / "control_plane_state.json"

    @property
    def decision_journal_path(self) -> Path:
        return self.runtime_dir / "decision_journal.jsonl"

    @property
    def portfolio_state_path(self) -> Path:
        return self.runtime_dir / "portfolio_state.json"

    @property
    def circuit_breaker_state_path(self) -> Path:
        return self.runtime_dir / "circuit_breaker_state.json"


    @property
    def control_plane_actions_path(self) -> Path:
        return self.runtime_dir / "control_plane_actions.jsonl"

    @property
    def dashboard_state_path(self) -> Path:
        return self.runtime_dir / "dashboard_state.json"

    @property
    def alerts_path(self) -> Path:
        return self.runtime_dir / "alerts.json"

    @property
    def training_registry_path(self) -> Path:
        return self.runtime_dir / "training_registry.json"

    @property
    def monitoring_state_path(self) -> Path:
        return self.monitoring_runtime_dir / "monitoring_state.json"

    @property
    def monitoring_events_path(self) -> Path:
        return self.monitoring_runtime_dir / "monitoring_events.json"

    @property
    def registry_path(self) -> Path:
        return self.training_registry_path

    @property
    def dashboard_path(self) -> Path:
        return self.dashboard_state_path

    @property
    def alerts_file_path(self) -> Path:
        return self.alerts_path

    @property
    def monitoring_dir(self) -> Path:
        return self.monitoring_runtime_dir

    @property
    def state_file_path(self) -> Path:
        return self.monitoring_state_path

    @property
    def events_file_path(self) -> Path:
        return self.monitoring_events_path


    @property
    def market_runtime_dir(self) -> Path:
        return self.runtime_dir / "market"

    @property
    def market_latest_snapshot_path(self) -> Path:
        return self.market_runtime_dir / "latest.json"

    @property
    def signal_events_log_path(self) -> Path:
        return self.signal_logs_dir / "signal_events.log"

    @property
    def trade_history_db_path(self) -> Path:
        return self.trade_history_dir / "trade_history.db"

    @property
    def live_backtest_trades_path(self) -> Path:
        return self.data_dir / "live_backtest_trades.csv"

    @property
    def backtest_open_trades_path(self) -> Path:
        return self.runtime_dir / "backtest_open_trades.json"

    def log_file_path(self, name: str) -> Path:
        safe_name = str(name).replace("/", "_").replace("\\", "_").replace(".", "_").replace("..", "_")
        return self.logs_dir / f"{safe_name}.log"

    def market_db_path(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{symbol.upper()}.db"

    def rf_model_dir(self, symbol: str, interval: str) -> Path:
        return self.rf_models_dir / symbol.upper() / interval

    def gbm_model_dir(self, symbol: str, interval: str) -> Path:
        return self.gbm_models_dir / symbol.upper() / interval

    def dqn_model_dir(self, symbol: str, interval: str) -> Path:
        return self.dqn_models_dir / symbol.upper() / interval

    def hmm_model_dir(self, symbol: str, interval: str) -> Path:
        return self.hmm_models_dir / symbol.upper() / interval

    def trainer_progress_path(self, model_type: str, symbol: str, interval: str) -> Path:
        safe = f"{model_type.lower()}__{symbol.upper()}__{interval}.json"
        return self.training_progress_dir / safe

    def rf_artifact_path(self, symbol: str, interval: str) -> dict[str, Path]:
        return artifact_paths(self, "rf", symbol, interval)

    def gbm_artifact_path(self, symbol: str, interval: str) -> dict[str, Path]:
        return artifact_paths(self, "gbm", symbol, interval)

    def dqn_artifact_path(self, symbol: str, interval: str) -> dict[str, Path]:
        return artifact_paths(self, "dqn", symbol, interval)

    def hmm_artifact_path(self, symbol: str, interval: str) -> dict[str, Path]:
        return artifact_paths(self, "hmm", symbol, interval)
