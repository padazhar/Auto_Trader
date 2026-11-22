from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import threading
import urllib.error
import urllib.request
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.config_contract import SecretValidationError, validate_runtime_secrets
from core.log_manager import get_logger
from core.path_manager import PathManager
from interface.terminal_dashboard import TerminalDashboard
from core.shared_utils import now_utc_ts, set_random_seed
from core.training_registry import TrainingRegistry
from execution.exchange_reconciliation import BinanceFuturesReconciler
from execution.portfolio_state import PortfolioStateStore
from monitoring.publisher import MonitorPublisher
from monitoring.web_app import run_monitoring_server
from runners.backtest_controller import BacktestController
from runners.live_trading_controller import LiveTradingController
from runners.market_data_controller import MarketDataController
from runners.signal_generator_controller import SignalGeneratorController
from runners.state_bridge_controller import StateBridgeController
from runners.tech_rev_controller import TechRevController
from runners.telegram_controller import TelegramController
from runtime.shutdown_coordinator import ShutdownCoordinator
from reporting.run_report import write_run_report
from scripts.live_trading_readiness_check import run_live_trading_readiness_check


@dataclass(slots=True)
class ControllerRuntime:
    """Runtime metadata for a managed controller thread."""

    name: str
    target: Callable[[threading.Event], None]
    critical: bool = True
    thread: threading.Thread | None = None
    error: Exception | None = None
    has_been_reported_dead: bool = field(default=False)


def build_paths() -> PathManager:
    """Build application paths and load the validated configuration."""
    project_root = Path(__file__).resolve().parent
    paths = PathManager.from_project_root(project_root)
    set_random_seed(paths.config.system.random_seed)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the launcher CLI parser."""
    parser = argparse.ArgumentParser(description="Auto Trader launcher")
    parser.add_argument(
        "--mode",
        choices=["update", "tech_rev", "trainer", "signal", "runtime", "backtest", "dashboard", "monitor", "monitoring", "live", "all"],
        default="all",
    )
    return parser


def _runtime_execution_mode(paths) -> str:
    """Return the normalized runtime execution mode with a safe paper default."""
    runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
    return str(getattr(runtime_cfg, "execution_mode", "paper") or "paper").strip().lower()


def _allow_unreconciled_live_mode(paths) -> bool:
    """Return whether unreconciled live mode is explicitly allowed."""
    runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
    return bool(getattr(runtime_cfg, "allow_unreconciled_live_mode", False))


def _exchange_cfg(paths):
    """Return the exchange config object when present."""
    return getattr(getattr(paths, "config", None), "exchange", None)


def _exchange_reconciliation_enabled(paths) -> bool:
    """Return whether exchange position reconciliation is enabled."""
    exchange_cfg = _exchange_cfg(paths)
    return bool(getattr(exchange_cfg, "position_reconciliation_enabled", False))


def _exchange_reconciliation_required(paths) -> bool:
    """Return whether exchange reconciliation is required for live mode."""
    exchange_cfg = _exchange_cfg(paths)
    return bool(getattr(exchange_cfg, "require_reconciliation_for_live", True))


def _normalize_launcher_mode(mode: str) -> str:
    """Collapse legacy launcher mode names into their canonical runtime targets."""
    normalized = str(mode or "all").strip().lower()
    if normalized in {"dashboard", "monitor", "monitoring"}:
        return "monitoring"
    if normalized == "trainer":
        return "tech_rev"
    return normalized


def _run_single(name: str, target: Callable[[threading.Event], None], stop_event: threading.Event, logger, monitor: MonitorPublisher) -> None:
    """Run a single controller in the foreground with shared monitoring hooks."""
    logger.info("Controller %s entering run loop", name)
    try:
        target(stop_event)
    except Exception as exc:
        monitor.controller_error(name, str(exc))
        logger.exception("Controller %s crashed", name)
        raise


def _start_controller(runtime: ControllerRuntime, stop_event: threading.Event, logger, monitor: MonitorPublisher) -> None:
    """Start a controller thread and capture failures in shared state."""

    def _runner() -> None:
        try:
            runtime.target(stop_event)
        except Exception as exc:
            runtime.error = exc
            monitor.controller_error(runtime.name, str(exc))
            if runtime.critical:
                stop_event.set()
            logger.exception("Thread %s crashed", runtime.name)

    runtime.thread = threading.Thread(target=_runner, daemon=False, name=runtime.name)
    runtime.thread.start()


def _supervise_controllers(runtimes: list[ControllerRuntime], stop_event: threading.Event, monitor: MonitorPublisher, logger) -> None:
    """Supervise controllers without letting non-critical failures stop trading."""
    while not stop_event.is_set():
        for runtime in runtimes:
            if runtime.thread is None or runtime.thread.is_alive():
                continue
            if runtime.has_been_reported_dead:
                continue
            runtime.has_been_reported_dead = True
            logger.error("Controller %s is no longer alive", runtime.name)
            monitor.controller_error(runtime.name, "thread_dead")
            if runtime.critical:
                logger.error("Critical controller %s died; shutting down the launcher", runtime.name)
                stop_event.set()
                break
            logger.warning("Non-critical controller %s died; continuing with degraded service", runtime.name)
        time.sleep(1.0)


def _validate_startup_secrets(paths: PathManager, logger) -> None:
    """Fail fast on missing or unsafe secrets for the current runtime mode."""
    validate_runtime_secrets(paths.config)


def _mode_includes_live_trading(mode: str) -> bool:
    """Return whether the selected launcher mode explicitly requests live startup."""
    normalized = str(mode or "").strip().lower()
    return normalized == "live"


def _live_mode_explicitly_enabled() -> bool:
    """Return whether the operator explicitly enabled live startup for this process."""
    raw = os.getenv("AUTO_TRADER_ENABLE_LIVE_MODE", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _live_canary_explicitly_enabled() -> bool:
    """Return whether the operator explicitly enabled restricted live canary trading."""
    raw = os.getenv("AUTO_TRADER_ENABLE_LIVE_CANARY", "")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _validate_runtime_mode_guards(paths: PathManager, logger, launcher_mode: str) -> None:
    """Block unsafe live launcher selections while allowing paper-safe orchestration.

    Explicit ``--mode live`` is guarded. ``--mode all`` is treated as the
    paper-safe default orchestration and never requires live runtime authority.
    """
    normalized_mode = str(launcher_mode or "").strip().lower()
    execution_mode = _runtime_execution_mode(paths)

    if normalized_mode != "live":
        return

    allow_unreconciled_live_mode = _allow_unreconciled_live_mode(paths)
    reconciliation_required = _exchange_reconciliation_required(paths)
    reconciliation_enabled = _exchange_reconciliation_enabled(paths)

    if execution_mode != "live":
        logger.error(
            "Startup blocked: launcher mode '%s' includes live trading, but runtime.execution_mode is '%s'. "
            "Set runtime.execution_mode=live explicitly before using a live launcher mode.",
            launcher_mode,
            execution_mode,
        )
        raise SystemExit(3)

    if reconciliation_required and not reconciliation_enabled and not allow_unreconciled_live_mode:
        logger.error(
            "Startup blocked: launcher mode '%s' includes live trading, but exchange position reconciliation is not enabled.",
            launcher_mode,
        )
        raise SystemExit(4)

    if reconciliation_required and reconciliation_enabled:
        logger.info("Live launcher mode enabled with exchange reconciliation configured.")
    elif allow_unreconciled_live_mode:
        logger.warning("Live launcher mode enabled with allow_unreconciled_live_mode=true. Use with extreme caution.")

    if not _live_mode_explicitly_enabled():
        logger.error(
            "Startup blocked: launcher mode '%s' includes live trading, but AUTO_TRADER_ENABLE_LIVE_MODE is not explicitly enabled.",
            launcher_mode,
        )
        raise SystemExit(5)

    runtime_cfg = getattr(getattr(paths, "config", None), "runtime", None)
    exchange_cfg = _exchange_cfg(paths)
    live_canary_enabled = bool(getattr(runtime_cfg, "live_canary_enabled", True))
    live_canary_max_symbols = int(getattr(runtime_cfg, "live_canary_max_symbols", 1) or 1)
    live_canary_max_notional_usd = float(getattr(runtime_cfg, "live_canary_max_notional_usd", 50.0) or 0.0)
    live_canary_max_concurrent_positions = int(getattr(runtime_cfg, "live_canary_max_concurrent_positions", 1) or 1)
    live_canary_require_testnet = bool(getattr(runtime_cfg, "live_canary_require_testnet", True))

    if not live_canary_enabled:
        logger.error("Startup blocked: live launcher mode requires runtime.live_canary_enabled=true.")
        raise SystemExit(5)
    if not _live_canary_explicitly_enabled():
        logger.error(
            "Startup blocked: launcher mode '%s' includes live trading, but AUTO_TRADER_ENABLE_LIVE_CANARY is not explicitly enabled.",
            launcher_mode,
        )
        raise SystemExit(5)
    if live_canary_max_symbols > 1:
        logger.error("Startup blocked: live canary must limit symbols to 1. configured=%s", live_canary_max_symbols)
        raise SystemExit(5)
    if live_canary_max_concurrent_positions > 1:
        logger.error(
            "Startup blocked: live canary must limit concurrent positions to 1. configured=%s",
            live_canary_max_concurrent_positions,
        )
        raise SystemExit(5)
    if live_canary_max_notional_usd > 50.0:
        logger.error(
            "Startup blocked: live canary max notional must be <= 50 USD. configured=%s",
            live_canary_max_notional_usd,
        )
        raise SystemExit(5)
    if live_canary_require_testnet and not bool(getattr(exchange_cfg, "use_testnet", False)):
        logger.error("Startup blocked: live canary requires exchange.use_testnet=true.")
        raise SystemExit(5)


def _live_startup_preflight_if_applicable(paths, logger, launcher_mode: str) -> None:
    """Run a fail-closed live startup preflight when a live launcher path is selected.

    This lightweight preflight is intentionally separate from the broader final
    live-readiness gate. It validates the immediate startup authority contract so
    explicit live launcher paths cannot start with reconciliation disabled or in
    a mismatched exchange-vs-ledger state.
    """
    normalized_mode = str(launcher_mode or "").strip().lower()
    if normalized_mode != "live":
        return

    execution_mode = _runtime_execution_mode(paths)
    if execution_mode != "live":
        return

    if _exchange_reconciliation_required(paths) and not _exchange_reconciliation_enabled(paths):
        logger.error(
            "Startup blocked: live startup preflight requires exchange reconciliation to be enabled."
        )
        raise SystemExit(5)

    if not _exchange_reconciliation_enabled(paths):
        return

    portfolio_path = getattr(paths, "portfolio_state_path", None)
    if portfolio_path is None:
        logger.warning(
            "Live startup preflight is running without portfolio_state_path; "
            "skipping reconciliation persistence in this lightweight context."
        )
        return

    portfolio_store = PortfolioStateStore(
        portfolio_path,
        starting_equity_usd=float(getattr(getattr(paths.config, "risk", None), "account_balance_usd", 0.0) or 0.0),
    )
    portfolio_state = portfolio_store.load()

    try:
        reconciler = BinanceFuturesReconciler(paths)
        result = reconciler.reconcile(portfolio_state, ts=now_utc_ts())
    except Exception as exc:
        logger.error("Startup blocked: live startup preflight reconciliation errored: %s", exc)
        raise SystemExit(5) from exc

    try:
        portfolio_store.record_reconciliation(portfolio_state, result, ts=now_utc_ts())
    except Exception:
        logger.exception("Failed to persist startup reconciliation result")

    if not bool(getattr(result, "success", False)) or not bool(getattr(result, "reconciled", False)):
        logger.error(
            "Startup blocked: live startup preflight did not reconcile successfully. status=%s error=%s",
            getattr(result, "status", ""),
            getattr(result, "error", ""),
        )
        raise SystemExit(5)

    logger.info("Live startup preflight passed with reconciled exchange authority.")


def _validate_live_trading_readiness(paths: PathManager, logger, launcher_mode: str) -> None:
    """Run the final live-trading readiness gate before any live launcher path starts.

    The repository defaults to paper-safe operation. When a process is launched
    with the explicit live launcher path, the final live-readiness report must
    pass before the launcher is allowed to start live trading.
    """
    normalized_mode = str(launcher_mode or "").strip().lower()
    if normalized_mode != "live":
        return

    execution_mode = _runtime_execution_mode(paths)
    if execution_mode != "live":
        return

    report = run_live_trading_readiness_check(paths.project_root)
    payload = report.to_dict()
    if not payload.get("live_trading_ready", False):
        logger.error(
            "Startup blocked: final live-trading readiness gate failed. "
            "overall_status=%s items=%s",
            payload.get("overall_status"),
            payload.get("items"),
        )
        raise SystemExit(6)

    logger.info("Final live-trading readiness gate passed.")




def _monitoring_owner_url(paths: PathManager, endpoint: str) -> str | None:
    cfg = getattr(getattr(paths, "config", None), "monitoring", None)
    host = str(getattr(cfg, "host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(cfg, "port", 0) or 0)
    if port <= 0:
        return None
    return f"http://{host}:{port}{endpoint}"


def _read_monitoring_json_url(url: str | None, timeout_sec: float = 0.75) -> dict | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def _monitoring_external_owner(paths: PathManager) -> dict | None:
    owner = _read_monitoring_json_url(_monitoring_owner_url(paths, "/api/monitoring-owner"), timeout_sec=0.5)
    if owner:
        return owner
    health = _read_monitoring_json_url(_monitoring_owner_url(paths, "/api/health"), timeout_sec=0.5)
    if health:
        return health
    snapshot = _read_monitoring_json_url(_monitoring_owner_url(paths, "/api/snapshot"), timeout_sec=0.75)
    if not snapshot:
        return None
    system = snapshot.get("system", {}) if isinstance(snapshot, dict) else {}
    return {"ok": True, "pid": system.get("pid"), "project_root": system.get("project_root"), "started_at": system.get("started_at"), "source": "snapshot_fallback"}


def _monitoring_listener_is_compatible(paths: PathManager) -> tuple[bool, str]:
    owner = _monitoring_external_owner(paths)
    if not owner:
        return False, "existing listener did not return a valid monitoring owner payload"
    project_root = str(owner.get("project_root") or "").strip()
    current_root = str(getattr(paths, "project_root", "") or "").strip()
    if current_root and project_root and Path(project_root).resolve() != Path(current_root).resolve():
        return False, f"existing listener belongs to a different project root: {project_root}"
    return True, f"existing compatible monitoring listener pid={owner.get('pid') or 'unknown'}"


def _monitoring_bind_status(paths: PathManager) -> tuple[str, str]:
    """Return the monitoring bind state as available, listening, or unavailable."""
    cfg = getattr(getattr(paths, "config", None), "monitoring", None)
    host = str(getattr(cfg, "host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(cfg, "port", 0) or 0)
    if port <= 0:
        return "available", ""

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            return "listening", f"existing listener detected on {host}:{port}"
    except OSError:
        pass
    finally:
        with contextlib.suppress(Exception):
            probe.close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        return "available", ""
    except OSError as exc:
        return "unavailable", str(exc)
    finally:
        with contextlib.suppress(Exception):
            sock.close()


def _run_monitoring_standby_mode(
    stop_event: threading.Event,
    *,
    monitor: MonitorPublisher | None,
    logger,
    reason: str,
    message: str,
    status: str,
    category: str,
) -> None:
    """Keep launcher state explicit when monitoring is external or unavailable."""
    if monitor is not None:
        try:
            monitor.publish_event(
                level="WARNING",
                component="monitoring",
                category=category,
                message=message,
                data={"reason": str(reason)},
                alert=(status != "external"),
            )
            monitor.heartbeat("monitoring", status=status, details={"reason": str(reason)})
        except Exception:
            pass
    if logger is not None:
        if status == "external":
            logger.info("Monitoring already running externally: %s", reason)
        else:
            logger.warning("Monitoring entering standby mode: %s", reason)
    while not stop_event.is_set():
        stop_event.wait(0.1)


def _monitoring_target(
    paths: PathManager,
    *,
    monitor: MonitorPublisher | None = None,
    logger=None,
) -> Callable[[threading.Event], None]:
    """Create the monitoring controller target with the current paths bound."""

    def _target(stop_event: threading.Event) -> None:
        bind_state, bind_reason = _monitoring_bind_status(paths)
        if bind_state == "listening":
            compatible, compatibility_reason = _monitoring_listener_is_compatible(paths)
            if not compatible:
                _run_monitoring_standby_mode(
                    stop_event,
                    monitor=monitor,
                    logger=logger,
                    reason=compatibility_reason,
                    message="monitoring_external_listener_incompatible",
                    status="degraded",
                    category="monitoring_degraded",
                )
                return
            _run_monitoring_standby_mode(
                stop_event,
                monitor=monitor,
                logger=logger,
                reason=bind_reason or "monitor_already_running",
                message="monitoring_external_listener_reused",
                status="external",
                category="monitoring_external",
            )
            return
        if bind_state != "available":
            _run_monitoring_standby_mode(
                stop_event,
                monitor=monitor,
                logger=logger,
                reason=bind_reason or "monitor_bind_unavailable",
                message="monitoring_bind_unavailable",
                status="degraded",
                category="monitoring_degraded",
            )
            return
        try:
            run_monitoring_server(paths, stop_event=stop_event)
        except OSError as exc:
            post_state, post_reason = _monitoring_bind_status(paths)
            if post_state == "listening":
                compatible, compatibility_reason = _monitoring_listener_is_compatible(paths)
                _run_monitoring_standby_mode(
                    stop_event,
                    monitor=monitor,
                    logger=logger,
                    reason=(post_reason if compatible else compatibility_reason) or str(exc),
                    message=("monitoring_external_listener_reused" if compatible else "monitoring_external_listener_incompatible"),
                    status=("external" if compatible else "degraded"),
                    category=("monitoring_external" if compatible else "monitoring_degraded"),
                )
                return
            _run_monitoring_standby_mode(
                stop_event,
                monitor=monitor,
                logger=logger,
                reason=str(exc),
                message="monitoring_server_start_failed",
                status="degraded",
                category="monitoring_degraded",
            )

    return _target



def _embedded_monitoring_enabled(paths: PathManager) -> bool:
    """Return whether --mode all should start the dashboard inside the trading process.

    The default is False so the UI can be run as a permanent, isolated process
    with ``python3 main.py --mode dashboard``.  This prevents CPU-heavy training
    and signal loops from freezing dashboard refreshes.
    """
    monitoring_cfg = getattr(getattr(paths, "config", None), "monitoring", None)
    return bool(getattr(monitoring_cfg, "embedded_in_all_mode", False))


def _publish_monitoring_external_hint(monitor: MonitorPublisher, logger) -> None:
    """Publish/log the operator contract for standalone dashboard mode."""
    message = "dashboard runs as a standalone process; start it with: python3 main.py --mode dashboard"
    with contextlib.suppress(Exception):
        monitor.publish_event(
            level="INFO",
            component="monitoring",
            category="monitoring_external_required",
            message="monitoring_standalone_process_required",
            data={"command": "python3 main.py --mode dashboard"},
            alert=False,
        )
        monitor.heartbeat("monitoring", status="external_required", details={"command": "python3 main.py --mode dashboard"})
    if logger is not None:
        with contextlib.suppress(Exception):
            logger.info(message)

def _controller_target(factory: Callable, *args, runner_name: str = "run_forever", **kwargs) -> Callable[[threading.Event], None]:
    """Create a lazy controller target so orchestration can be assembled safely.

    Controllers are instantiated only when the returned target is actually run.
    This keeps launcher smoke tests lightweight and avoids premature side
    effects while building the runtime set.
    """

    def _target(stop_event: threading.Event) -> None:
        controller = factory(*args, **kwargs)
        runner = getattr(controller, runner_name)
        runner(stop_event)

    return _target



def _build_runtime_process_controllers(paths: PathManager, registry: TrainingRegistry) -> list[ControllerRuntime]:
    """Build the runtime-core controller set without training or dashboard.

    Runtime mode is intentionally separated from the trainer process so CPU-heavy
    model training cannot block market updates, signal generation, execution, or
    snapshot publishing. Live trading remains fail-closed: explicit ``--mode live``
    is still required for live authority, and runtime mode skips the live trading
    controller when config runtime.execution_mode is live.
    """
    runtimes = [
        ControllerRuntime(name="market_data", target=_controller_target(MarketDataController, paths), critical=True),
        ControllerRuntime(name="signal_generator", target=_controller_target(SignalGeneratorController, paths), critical=True),
    ]
    if _runtime_execution_mode(paths) != "live":
        runtimes.append(ControllerRuntime(name="live_trading", target=_controller_target(LiveTradingController, paths, registry), critical=True))
    runtimes.extend([
        ControllerRuntime(name="backtest", target=_controller_target(BacktestController, paths), critical=False),
        ControllerRuntime(name="state_bridge", target=_controller_target(StateBridgeController, paths), critical=True),
    ])
    if bool(getattr(getattr(paths.config, "telegram", None), "enabled", False)):
        runtimes.append(ControllerRuntime(name="telegram", target=_controller_target(TelegramController, paths), critical=False))
    return runtimes


def _build_all_process_controllers(paths: PathManager, registry: TrainingRegistry) -> list[ControllerRuntime]:
    """Build ``--mode all`` as runtime-core plus trainer, excluding dashboard by default."""
    runtimes = [
        ControllerRuntime(name="market_data", target=_controller_target(MarketDataController, paths), critical=True),
        ControllerRuntime(name="tech_rev", target=_controller_target(TechRevController, paths, registry), critical=True),
        ControllerRuntime(name="signal_generator", target=_controller_target(SignalGeneratorController, paths), critical=True),
    ]
    if _runtime_execution_mode(paths) != "live":
        runtimes.append(ControllerRuntime(name="live_trading", target=_controller_target(LiveTradingController, paths, registry), critical=True))
    runtimes.extend([
        ControllerRuntime(name="backtest", target=_controller_target(BacktestController, paths), critical=False),
        ControllerRuntime(name="state_bridge", target=_controller_target(StateBridgeController, paths), critical=True),
    ])
    if _embedded_monitoring_enabled(paths):
        runtimes.append(
            ControllerRuntime(name="monitoring", target=_monitoring_target(paths), critical=False)
        )
    if bool(getattr(getattr(paths.config, "telegram", None), "enabled", False)):
        runtimes.append(ControllerRuntime(name="telegram", target=_controller_target(TelegramController, paths), critical=False))
    return runtimes

def main(argv: list[str] | None = None) -> None:
    """Launch the selected Auto Trader runtime mode."""
    paths = build_paths()
    logger = get_logger("main", paths)
    monitor = MonitorPublisher(paths)

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.mode = _normalize_launcher_mode(args.mode)

    try:
        _validate_startup_secrets(paths, logger)
        _validate_runtime_mode_guards(paths, logger, args.mode)
        _live_startup_preflight_if_applicable(paths, logger, args.mode)
        _validate_live_trading_readiness(paths, logger, args.mode)
    except SecretValidationError as exc:
        logger.error("Startup blocked by secret validation: %s", exc)
        raise SystemExit(2) from exc

    monitor.publish_system(
        launcher_mode=args.mode,
        project_root=str(paths.project_root),
        python_version=platform.python_version(),
        pid=None,
        started_at=now_utc_ts(),
        execution_mode=_runtime_execution_mode(paths),
        allow_unreconciled_live_mode=_allow_unreconciled_live_mode(paths),
        reconciliation_enabled=_exchange_reconciliation_enabled(paths),
    )

    registry = TrainingRegistry(paths)
    stop_event = threading.Event()
    shutdown_config = getattr(paths.config, "shutdown", None)
    shutdown = ShutdownCoordinator(
        stop_event=stop_event,
        monitor=monitor,
        logger=logger,
        join_timeout_sec=float(getattr(shutdown_config, "coordinator_join_timeout_sec", 5.0) or 5.0),
    )
    shutdown.install_signal_handlers()
    runtimes: list[ControllerRuntime] = []
    try:
        if args.mode == "update":
            _run_single("market_data", _controller_target(MarketDataController, paths), stop_event, logger, monitor)
            return
        if args.mode == "tech_rev":
            _run_single("tech_rev", _controller_target(TechRevController, paths, registry), stop_event, logger, monitor)
            return
        if args.mode == "signal":
            _run_single("signal_generator", _controller_target(SignalGeneratorController, paths), stop_event, logger, monitor)
            return
        if args.mode == "backtest":
            _run_single("backtest", _controller_target(BacktestController, paths), stop_event, logger, monitor)
            return
        if args.mode == "monitoring":
            _run_single("monitoring", _monitoring_target(paths, monitor=monitor, logger=logger), stop_event, logger, monitor)
            return
        if args.mode == "live":
            _run_single("live_trading", _controller_target(LiveTradingController, paths, registry), stop_event, logger, monitor)
            return
        if args.mode == "runtime":
            runtimes = _build_runtime_process_controllers(paths, registry)
            _publish_monitoring_external_hint(monitor, logger)
        else:
            runtimes = _build_all_process_controllers(paths, registry)
            if not _embedded_monitoring_enabled(paths):
                _publish_monitoring_external_hint(monitor, logger)
            else:
                for runtime in runtimes:
                    if runtime.name == "monitoring":
                        runtime.target = _monitoring_target(paths, monitor=monitor, logger=logger)
        for runtime in runtimes:
            _start_controller(runtime, stop_event, logger, monitor)
        _supervise_controllers(runtimes, stop_event, monitor, logger)
    except KeyboardInterrupt:
        shutdown.request_shutdown("keyboard_interrupt")
    finally:
        shutdown_state = shutdown.finalize(runtimes)
        if bool(getattr(getattr(paths.config, "shutdown", None), "report_on_exit", True)):
            try:
                result = write_run_report(paths, shutdown_state=shutdown_state, monitor=monitor)
                logger.info("Final run report written: %s %s", result.get("json_path"), result.get("md_path"))
            except Exception as exc:
                logger.exception("Final run report generation failed: %s", exc)


if __name__ == "__main__":
    main()
