from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def render_backtest_dashboard(open_rows: list[dict[str, Any]], summary, messages: list[str], *, last_error: str | None = None) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"LIVE BACKTEST DASHBOARD — {now}",
        f"Open Trades: {summary.open_count} | Completed: {summary.completed_count} | Wins: {summary.wins} | Losses: {summary.losses}",
        f"Total PnL: {summary.total_pnl:+.2f} USD | Win Rate: {summary.win_rate:.1f}%",
        "",
        "Symbol     Side   Mode     Entry        Last(ts)          Est.PnL",
        "----------------------------------------------------------------",
    ]
    for row in open_rows:
        last_price = f"{row['last_price']:.4f}" if row.get("last_price") is not None else "—"
        stale = " stale" if row.get("stale") else ""
        lines.append(
            f"{row['symbol']:<10} {row['side']:<6} {row['mode']:<8} {row['entry_price']:<12.4f} "
            f"{last_price:<10} {row['last_price_ts']}{stale:<6} {row['estimated_pnl']:+10.2f}"
        )
    lines.append("")
    lines.append("Messages:")
    lines.extend(messages[-12:])
    if last_error:
        lines.append("")
        lines.append("Last Error:")
        lines.append(last_error)
    return "\n".join(lines)
