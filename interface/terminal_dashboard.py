from __future__ import annotations

import threading

from core.path_manager import PathManager
from monitoring.web_app import run_monitoring_server


class TerminalDashboard:
    """Compatibility wrapper for older dashboard entry points.

    Launcher-level ownership of the monitoring server now lives in the dedicated
    monitoring runtime. This wrapper restores the historical direct-call behavior
    for older imports that still instantiate ``TerminalDashboard`` directly.
    """

    def __init__(self, paths: PathManager) -> None:
        self.paths = paths

    def run_file_backed(self, stop_event: threading.Event | None = None) -> None:
        run_monitoring_server(self.paths, stop_event=stop_event)
