
from __future__ import annotations

import io
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


@dataclass
class _TelegramCfg:
    enabled: bool
    bot_token: str
    chat_id: str
    poll_interval_sec: int = 2
    command_cooldown_sec: int = 3
    snapshot_url: str = "http://127.0.0.1:8000/api/snapshot"
    status_url: str = "http://127.0.0.1:8000/api/snapshot"


class TelegramController:
    """
    Telegram runner for the current Auto_Trader repo.

    Features:
    - /snap-execution
    - /snap-signal
    - /snap-training   (includes timed-out symbols section)
    - /status
    - /start_stack
    - /start_rem
    - /stop_stack
    - sends Telegram trade notifications for ENTRY / CLOSED / SL_HIT / TP_HIT / EXIT
    """

    BG = (10, 16, 28)
    PANEL = (16, 24, 40)
    BORDER = (64, 76, 104)
    TEXT = (232, 238, 247)
    MUTED = (170, 182, 204)
    GOOD = (38, 166, 91)
    BAD = (196, 72, 72)
    WARN = (207, 160, 51)
    BLUE = (67, 111, 220)

    def __init__(self, paths):
        self.paths = paths
        self.cfg = self._load_cfg(paths)
        self._last_update_id: int | None = None
        self._last_command_at: dict[str, float] = {}
        self._seen_trade_messages: set[str] = set()

    def _load_cfg(self, paths) -> _TelegramCfg:
        cfg = getattr(paths.config, "telegram", None)
        if cfg is None:
            return _TelegramCfg(False, "", "")
        return _TelegramCfg(
            enabled=bool(getattr(cfg, "enabled", False)),
            bot_token=str(getattr(cfg, "bot_token", "") or ""),
            chat_id=str(getattr(cfg, "chat_id", "") or ""),
            poll_interval_sec=int(getattr(cfg, "poll_interval_sec", 2) or 2),
            command_cooldown_sec=int(getattr(cfg, "command_cooldown_sec", 3) or 3),
            snapshot_url=str(getattr(cfg, "snapshot_url", "http://127.0.0.1:8000/api/snapshot") or "http://127.0.0.1:8000/api/snapshot"),
            status_url=str(getattr(cfg, "status_url", "http://127.0.0.1:8000/api/snapshot") or "http://127.0.0.1:8000/api/snapshot"),
        )

    @property
    def bot_api_url(self) -> str:
        return f"https://api.telegram.org/bot{self.cfg.bot_token}"

    def _send_text(self, text: str) -> None:
        if not self.cfg.enabled or not self.cfg.bot_token or not self.cfg.chat_id:
            return
        try:
            requests.get(
                f"{self.bot_api_url}/sendMessage",
                params={"chat_id": self.cfg.chat_id, "text": text},
                timeout=10,
            )
        except Exception:
            pass

    def _send_photo(self, image: Image.Image, caption: str) -> None:
        if not self.cfg.enabled or not self.cfg.bot_token or not self.cfg.chat_id:
            return
        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            requests.post(
                f"{self.bot_api_url}/sendPhoto",
                data={"chat_id": self.cfg.chat_id, "caption": caption},
                files={"photo": ("snapshot.png", buf, "image/png")},
                timeout=20,
            )
        except Exception:
            try:
                self._send_text(f"{caption}\n(snapshot render failed)")
            except Exception:
                pass

    def _fetch_snapshot(self) -> dict[str, Any]:
        try:
            resp = requests.get(self.cfg.snapshot_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _font(self, size: int, bold: bool = False):
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for c in candidates:
            p = Path(c)
            if p.exists():
                return ImageFont.truetype(str(p), size=size)
        return ImageFont.load_default()

    def _draw_panel(self, draw: ImageDraw.ImageDraw, xy, title: str):
        x1, y1, x2, y2 = xy
        draw.rounded_rectangle([x1, y1, x2, y2], radius=18, fill=self.PANEL, outline=self.BORDER, width=2)
        title_font = self._font(24, bold=True)
        draw.text((x1 + 18, y1 + 12), title, font=title_font, fill=self.TEXT)

    def _draw_table(self, draw: ImageDraw.ImageDraw, origin, col_widths, headers, rows, row_h=30, chip_cols: set[int] | None = None, max_rows: int | None = None):
        chip_cols = chip_cols or set()
        x0, y0 = origin
        header_font = self._font(15, bold=True)
        body_font = self._font(14, bold=False)
        cur_x = x0
        for i, (h, w) in enumerate(zip(headers, col_widths)):
            draw.text((cur_x + 6, y0), str(h), font=header_font, fill=self.TEXT)
            cur_x += w
        y = y0 + 28
        shown_rows = rows if max_rows is None else rows[:max_rows]
        for row in shown_rows:
            cur_x = x0
            for i, (cell, w) in enumerate(zip(row, col_widths)):
                text = "" if cell is None else str(cell)
                if i in chip_cols and text:
                    parts = [p.strip() for p in text.split() if p.strip()]
                    cx = cur_x + 6
                    for part in parts:
                        fill = self.BLUE
                        if part in {"READY", "R"}:
                            fill = self.GOOD
                        elif part in {"FAIL", "X"}:
                            fill = self.BAD
                        elif part in {"PART", "P", "D"}:
                            fill = self.BLUE
                        elif part in {"G"}:
                            fill = self.GOOD
                        elif part in {"BLOCKED"}:
                            fill = self.BAD
                        elif part in {"TIMED_OUT"}:
                            fill = self.WARN
                        draw.rounded_rectangle([cx, y - 2, cx + 24, y + 18], radius=10, fill=fill)
                        draw.text((cx + 8, y + 1), part[:1], font=body_font, fill=self.TEXT)
                        cx += 30
                else:
                    draw.text((cur_x + 6, y), text[:max(1, int(w / 8))], font=body_font, fill=self.MUTED if i == len(row) - 1 else self.TEXT)
                cur_x += w
            y += row_h
        return y

    def _snapshot_signal_image(self, snap: dict[str, Any]) -> Image.Image:
        img = Image.new("RGB", (1700, 920), self.BG)
        draw = ImageDraw.Draw(img)
        self._draw_panel(draw, (10, 10, 1690, 910), "Signal Generator")
        signal = ((snap.get("signal_generator") or {}).get("table") or [])
        rows = []
        for item in signal:
            chips = []
            tf_map = item.get("timeframes", {}) or {}
            for tf in ("1m", "3m", "5m", "15m"):
                models = tf_map.get(tf, {}) or {}
                parts = []
                for key in ("rf", "gbm", "dqn"):
                    if models.get(key):
                        parts.append({"rf": "R", "gbm": "G", "dqn": "D"}[key])
                chips.append(" ".join(parts) if parts else "—")
            rows.append([
                item.get("symbol", ""),
                item.get("mode", ""),
                item.get("position", ""),
                chips[0], chips[1], chips[2], chips[3],
                item.get("final_action", ""),
                item.get("reason", ""),
            ])
        headers = ["Symbol", "Gate/Mode", "Position", "1m", "3m", "5m", "15m", "Final Action", "Reason"]
        widths = [150, 130, 90, 90, 90, 90, 90, 120, 700]
        self._draw_table(draw, (24, 56), widths, headers, rows, row_h=34, chip_cols={3, 4, 5, 6})
        return img

    def _snapshot_execution_image(self, snap: dict[str, Any]) -> Image.Image:
        img = Image.new("RGB", (1700, 920), self.BG)
        draw = ImageDraw.Draw(img)
        self._draw_panel(draw, (10, 10, 1690, 910), "Execution")
        execution = snap.get("execution") or {}
        stats = execution.get("stats") or {}
        stat_font = self._font(18, bold=True)
        small_font = self._font(15, bold=False)
        cards = [
            ("Open Trades", stats.get("open_trades", 0)),
            ("Completed", stats.get("completed", 0)),
            ("Wins", stats.get("wins", 0)),
            ("Losses", stats.get("losses", 0)),
            ("Total PnL", stats.get("total_pnl", 0)),
            ("Win Rate", stats.get("win_rate", 0)),
        ]
        x = 24
        for title, value in cards:
            draw.rounded_rectangle([x, 56, x + 250, 116], radius=12, fill=(28, 37, 58), outline=self.BORDER, width=1)
            draw.text((x + 12, 68), title, font=small_font, fill=self.MUTED)
            draw.text((x + 206, 84), str(value), font=stat_font, fill=self.TEXT, anchor="ra")
            x += 262
        positions = execution.get("open_positions") or []
        headers = ["Symbol", "Side", "Mode", "Entry Price", "Last Price", "Last Price TS", "Est. PnL"]
        widths = [170, 110, 110, 170, 170, 180, 140]
        rows = [[p.get("symbol",""), p.get("side",""), p.get("mode",""), p.get("entry_price",""), p.get("last_price",""), p.get("last_price_ts",""), p.get("estimated_pnl","")] for p in positions]
        self._draw_table(draw, (24, 148), widths, headers, rows, row_h=34, max_rows=14)
        msgs = execution.get("messages") or []
        draw.text((24, 700), "Messages", font=self._font(18, bold=True), fill=self.TEXT)
        y = 730
        font = self._font(14)
        for line in msgs[:8]:
            draw.rounded_rectangle([24, y, 1670, y + 28], radius=10, fill=(24, 33, 54), outline=self.BORDER, width=1)
            draw.text((34, y + 6), str(line)[:220], font=font, fill=self.TEXT)
            y += 34
        return img

    def _snapshot_training_image(self, snap: dict[str, Any]) -> Image.Image:
        img = Image.new("RGB", (1700, 1080), self.BG)
        draw = ImageDraw.Draw(img)
        self._draw_panel(draw, (10, 10, 830, 1070), "Training")
        self._draw_panel(draw, (850, 10, 1690, 1070), "Training Results")

        training = snap.get("training") or {}
        summary = training.get("summary") or {}
        stat_font = self._font(16, bold=True)
        small = self._font(14)
        x = 24
        for title, value in [
            ("idle", summary.get("idle", 0)),
            ("queued", summary.get("queued", 0)),
            ("running", summary.get("running", 0)),
            ("success", summary.get("success", 0)),
            ("failed", summary.get("failed", 0)),
        ]:
            draw.rounded_rectangle([x, 56, x + 120, 92], radius=10, fill=(28, 37, 58), outline=self.BORDER, width=1)
            draw.text((x + 10, 67), f"{title}:{value}", font=small, fill=self.TEXT)
            x += 128

        prog = training.get("progress") or {}
        y = 120
        for name in ("rf", "gbm", "dqn"):
            block = prog.get(name) or {}
            draw.rounded_rectangle([24, y, 806, y + 96], radius=14, fill=(22, 31, 50), outline=self.BORDER, width=1)
            draw.text((36, y + 12), name.upper(), font=self._font(18, bold=True), fill=self.TEXT)
            draw.text((240, y + 12), f"Current {block.get('current', '—')}", font=small, fill=self.MUTED)
            draw.text((240, y + 34), f"Pending {block.get('pending', 0)}", font=small, fill=self.TEXT)
            pct = float(block.get("percent", 0.0) or 0.0)
            draw.rounded_rectangle([36, y + 60, 786, y + 74], radius=7, fill=(38, 47, 68))
            fill_w = int(750 * max(0.0, min(1.0, pct / 100.0)))
            draw.rounded_rectangle([36, y + 60, 36 + fill_w, y + 74], radius=7, fill=self.BLUE)
            draw.text((736, y + 40), f"{pct:.1f}%", font=small, fill=self.TEXT)
            y += 112

        timed = training.get("timed_out_symbols") or []
        draw.text((24, y + 8), "Timed-out symbols", font=self._font(18, bold=True), fill=self.TEXT)
        timed_rows = []
        for item in timed:
            timed_rows.append([
                item.get("symbol", ""),
                item.get("left", ""),
                item.get("retry_at", ""),
                item.get("blocked_jobs", ""),
                item.get("status", ""),
                item.get("consecutive_failures", ""),
                item.get("cooldown_level", ""),
                item.get("last_failure", ""),
            ])
        timed_headers = ["Symbol", "Left", "Retry At", "Blocked", "Status", "Fails", "CD", "Last failure"]
        timed_widths = [110, 80, 160, 80, 100, 70, 60, 330]
        self._draw_table(draw, (24, y + 40), timed_widths, timed_headers, timed_rows, row_h=30)

        results = training.get("results_rows") or training.get("results") or []
        result_rows = []
        for item in results:
            result_rows.append([
                item.get("symbol", ""),
                item.get("15m", ""),
                item.get("5m", ""),
                item.get("3m", ""),
                item.get("1m", ""),
                item.get("last_update", ""),
            ])
        result_headers = ["Symbol", "15m", "5m", "3m", "1m", "Last Update"]
        result_widths = [150, 120, 120, 120, 120, 270]
        self._draw_table(draw, (864, 56), result_widths, result_headers, result_rows, row_h=34, chip_cols={1, 2, 3, 4})
        return img

    def _handle_snapshot_command(self, cmd: str) -> None:
        snap = self._fetch_snapshot()
        if not snap:
            self._send_text("⚠️ Snapshot unavailable.")
            return
        if cmd == "/snap-signal":
            self._send_photo(self._snapshot_signal_image(snap), "Signal Generator snapshot")
        elif cmd == "/snap-execution":
            self._send_photo(self._snapshot_execution_image(snap), "Execution snapshot")
        elif cmd == "/snap-training":
            self._send_photo(self._snapshot_training_image(snap), "Training snapshot")

    def _trade_notification_text(self, line: str) -> str:
        line = line.strip()
        if " ENTRY " in line:
            return f"🟢 TRADE OPENED\n{line}"
        if " TP_HIT " in line or " TP " in line:
            return f"🎯 TAKE PROFIT\n{line}"
        if " SL_HIT " in line or " SL " in line:
            return f"🔴 STOP LOSS\n{line}"
        if " CLOSED " in line:
            return f"✅ TRADE CLOSED\n{line}"
        if " EXIT " in line:
            return f"📤 TRADE EXIT\n{line}"
        return line

    def _watch_trade_events(self) -> None:
        snap = self._fetch_snapshot()
        execution = snap.get("execution") or {}
        for line in (execution.get("messages") or [])[:30]:
            text = str(line).strip()
            if not text or text in self._seen_trade_messages:
                continue
            if any(token in text for token in (" ENTRY ", " CLOSED ", " TP_HIT ", " SL_HIT ", " EXIT ")):
                self._seen_trade_messages.add(text)
                self._send_text(self._trade_notification_text(text))

    def _status_text(self) -> str:
        snap = self._fetch_snapshot()
        if not snap:
            return "⚠️ Status unavailable."
        training = snap.get("training") or {}
        execution = snap.get("execution") or {}
        stats = execution.get("stats") or {}
        market = snap.get("market_data") or {}
        universe = market.get("universe") or []
        return (
            "📊 Auto Trader Status\n"
            f"Active symbols: {len(universe)}\n"
            f"Open trades: {stats.get('open_trades', 0)}\n"
            f"Completed: {stats.get('completed', 0)}\n"
            f"Wins: {stats.get('wins', 0)}\n"
            f"Losses: {stats.get('losses', 0)}\n"
            f"Total PnL: {stats.get('total_pnl', 0)}\n"
            f"Training success: {(training.get('summary') or {}).get('success', 0)}\n"
            f"Training failed: {(training.get('summary') or {}).get('failed', 0)}"
        )

    def _is_process_running(self, pattern: str) -> bool:
        try:
            res = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except Exception:
            return False

    def _kill_by_pattern(self, pattern: str) -> bool:
        try:
            subprocess.run(["pkill", "-f", pattern], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _start_stack_if_needed(self) -> None:
        root = Path(getattr(self.paths, "root_dir", Path.cwd()))
        if self._is_process_running("main.py --mode all"):
            self._send_text("⚠️ Stack already appears to be running.")
            return
        try:
            subprocess.Popen(["bash", "-lc", f"cd {root}; python3 -u main.py --mode all"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._send_text("🚀 Started Auto Trader stack.")
        except Exception as exc:
            self._send_text(f"❌ Failed to start stack: {exc}")

    def _start_remaining_stack(self) -> None:
        self._start_stack_if_needed()

    def _stop_stack(self) -> None:
        patterns = ["main.py --mode all", "market_data_controller", "tech_rev_controller", "signal_generator_controller", "backtest_controller", "telegram_controller"]
        touched = []
        for pat in patterns:
            if self._is_process_running(pat):
                self._kill_by_pattern(pat)
                touched.append(pat)
        self._send_text("💀 Stopped processes:\n" + ("\n".join(f"• {p}" for p in touched) if touched else "No matching processes found."))

    def _handle_command(self, text: str) -> None:
        t = (text or "").strip().lower()
        now = time.time()
        last = self._last_command_at.get(t, 0.0)
        if now - last < self.cfg.command_cooldown_sec:
            return
        self._last_command_at[t] = now

        if t in ("/snap-execution", "/snap-signal", "/snap-training"):
            self._handle_snapshot_command(t)
        elif t in ("/status", "status"):
            self._send_text(self._status_text())
        elif t in ("/start_stack", "start stack", "start bot", "run all"):
            self._start_stack_if_needed()
        elif t in ("/start_rem", "start remaining", "start_rem"):
            self._start_remaining_stack()
        elif t in ("/stop_stack", "stop stack", "stop bot", "kill stack"):
            self._stop_stack()
        else:
            self._send_text(
                "🤖 Commands:\n"
                "/status\n"
                "/snap-execution\n"
                "/snap-signal\n"
                "/snap-training\n"
                "/start_stack\n"
                "/stop_stack"
            )

    def _poll_updates(self) -> None:
        params: dict[str, Any] = {"timeout": 20}
        if self._last_update_id is not None:
            params["offset"] = self._last_update_id + 1
        resp = requests.get(f"{self.bot_api_url}/getUpdates", params=params, timeout=25)
        data = resp.json()
        if not data.get("ok"):
            return
        for update in data.get("result", []):
            self._last_update_id = update["update_id"]
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue
            chat_id = str(message["chat"]["id"])
            if chat_id != self.cfg.chat_id:
                continue
            self._handle_command(message.get("text", ""))

    def run_forever(self, stop_event=None) -> None:
        if not self.cfg.enabled or not self.cfg.bot_token or not self.cfg.chat_id:
            return
        self._send_text("🤖 Telegram controller started. Use /snap-execution, /snap-signal, /snap-training, /status.")
        while True:
            if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                break
            try:
                self._poll_updates()
            except Exception:
                pass
            try:
                self._watch_trade_events()
            except Exception:
                pass
            time.sleep(max(1, self.cfg.poll_interval_sec))
