from __future__ import annotations

import logging
import multiprocessing

from monitoring.publisher import MonitorPublisher


class MonitoringLogHandler(logging.Handler):
    """
    Forward selected log records into the monitoring event stream.

    Important: training jobs run inside ProcessPoolExecutor worker processes.
    On fork-based systems those workers inherit the parent's MonitorPublisher
    singleton snapshot at fork time. If a worker writes monitoring files it can
    overwrite the main process's current state with stale data.
    """

    def __init__(self, paths) -> None:
        super().__init__(level=logging.INFO)
        self.publisher = MonitorPublisher(paths)

    def emit(self, record: logging.LogRecord) -> None:
        if multiprocessing.current_process().name != 'MainProcess':
            return

        try:
            level = record.levelname.upper()
            self.publisher.publish_event(
                level=level,
                component=record.name,
                category='log',
                message=record.getMessage(),
                data={},
                alert=level in {'WARNING', 'ERROR', 'CRITICAL'},
            )
        except Exception:
            pass
