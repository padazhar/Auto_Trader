from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def render_signal_generator_panel(rows: list[dict[str, Any]], messages: list[str], *, mode: str = "Medium") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"SIGNAL GENERATOR — {now} | Global Mode: {mode}",
        "Symbol     Action        Score    Reason",
        "---------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(f"{row['symbol']:<10} {row['action']:<12} {row['score']:+.2f}   {row['reason']}")
    lines.append("")
    lines.append("Messages:")
    lines.extend(messages[-12:])
    return "\n".join(lines)
