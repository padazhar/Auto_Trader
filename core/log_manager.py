from __future__ import annotations

import logging
import multiprocessing as mp
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.path_manager import PathManager
from monitoring.logging_handler import MonitoringLogHandler

_LOGGERS: dict[str, logging.Logger] = {}


def _safe_logger_file_name(name: str) -> str:
    normalized = str(name).strip().replace(".", "_").replace("/", "_").replace("\\", "_")
    if normalized.startswith("training_"):
        return normalized
    if normalized.startswith("training."):
        return normalized.replace("training.", "training_")
    return normalized


def _fallback_log_file_path(paths: PathManager, name: str) -> Path:
    """Resolve a safe log path for lightweight fixtures that lack PathManager helpers."""
    helper = getattr(paths, "log_file_path", None)
    if callable(helper):
        return Path(helper(_safe_logger_file_name(name)))
    logs_dir = getattr(paths, "logs_dir", getattr(paths, "runtime_dir", getattr(paths, "project_root", Path(".")) / "data" / "logs"))
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / f"{_safe_logger_file_name(name)}.log"


def get_logger(name: str, paths: PathManager, level: int = logging.INFO) -> logging.Logger:
    process_name = mp.current_process().name
    cache_key = f"{name}:{process_name}"
    if cache_key in _LOGGERS:
        return _LOGGERS[cache_key]

    logger = logging.getLogger(cache_key)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        file_handler = RotatingFileHandler(
            _fallback_log_file_path(paths, name),
            maxBytes=5_000_000,
            backupCount=10,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        if process_name == "MainProcess":
            monitoring_handler = MonitoringLogHandler(paths)
            monitoring_handler.setFormatter(formatter)
            logger.addHandler(monitoring_handler)

    _LOGGERS[cache_key] = logger
    return logger
