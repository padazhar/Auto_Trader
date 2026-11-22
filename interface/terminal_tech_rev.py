from __future__ import annotations

from datetime import datetime, timezone


def render_tech_rev_panel(snapshot) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n".join([
        f"TECH_REV — {now}",
        f"RF pending: {snapshot.rf_pending} | GBM pending: {snapshot.gbm_pending} | DQN pending: {snapshot.dqn_pending}",
        f"Current RF: {snapshot.current_rf} | Current GBM: {snapshot.current_gbm} | Current DQN: {snapshot.current_dqn}",
        f"DQN progress: {snapshot.dqn_progress:.1f}%",
        f"Last RF: {snapshot.last_rf_message}",
        f"Last GBM: {snapshot.last_gbm_message}",
        f"Last DQN: {snapshot.last_dqn_message}",
    ])
