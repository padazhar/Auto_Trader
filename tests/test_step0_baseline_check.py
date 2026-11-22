from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.step0_baseline_check import build_step0_baseline


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_minimal_repo(root: Path) -> None:
    for rel in [
        "main.py",
        "config/system_settings.yaml",
        "config/tradable_symbols.yaml",
        "core/symbol_management.py",
        "inference/readiness.py",
        "monitoring/publisher.py",
        "reporting/run_report.py",
        "scripts/runtime_self_check.py",
        "scripts/live_trading_readiness_check.py",
        "scripts/paper_trading_readiness_check.py",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "config" / "system_settings.yaml").write_text("runtime:\n  execution_mode: paper\nmarket:\n  top_n: 10\n", encoding="utf-8")
    (root / "config" / "tradable_symbols.yaml").write_text("symbols: []\n", encoding="utf-8")


def test_step0_baseline_preserves_open_position_price_coverage(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)
    _write_json(
        tmp_path / "data/runtime/symbol_state.json",
        {
            "target_active_symbols": 2,
            "active_symbols": ["BTCUSDT", "ETHUSDT"],
            "execution_eligible_symbols": ["BTCUSDT"],
            "blocked_symbols": [],
            "warming_symbols": ["ETHUSDT"],
            "historical_symbols": ["FILUSDT"],
            "symbols": {
                "ETHUSDT": {
                    "activation_stage": "warming_data",
                    "data_ready": False,
                    "data_ready_reason": "market_data_warming",
                    "model_ready_reason": "anchor_readiness_not_ready",
                    "execution_eligible": False,
                    "status": "ACTIVE",
                }
            },
        },
    )
    _write_json(
        tmp_path / "data/runtime/portfolio_state.json",
        {"authority_mode": "paper_local", "positions": {"FILUSDT": {"side": "LONG"}}, "closed_trades": []},
    )
    _write_json(
        tmp_path / "data/runtime/market/latest.json",
        {
            "active_symbols": ["BTCUSDT", "ETHUSDT"],
            "open_position_symbols": ["FILUSDT"],
            "price_symbols": ["BTCUSDT", "ETHUSDT", "FILUSDT"],
            "symbols": {"BTCUSDT": {}, "ETHUSDT": {}, "FILUSDT": {}},
        },
    )
    _write_json(
        tmp_path / "data/runtime/training_registry.json",
        {
            "rf:ETHUSDT:15m": {
                "model_type": "rf",
                "symbol": "ETHUSDT",
                "status": "idle",
                "registry_truth_status": "idle",
                "active_manifest_valid": False,
                "truth_reasons": ["active_manifest_missing"],
            }
        },
    )
    trades_path = tmp_path / "data/live_backtest_trades.csv"
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    with trades_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "pnl_usd", "fees_usd", "slippage_usd", "exit_reason"])
        writer.writeheader()
        writer.writerow({"symbol": "BTCUSDT", "pnl_usd": "1.5", "fees_usd": "0", "slippage_usd": "0", "exit_reason": "TP_HIT"})

    report = build_step0_baseline(tmp_path)
    summary = report.to_dict()["summary"]

    assert report.safe_to_start_step1 is True
    assert summary["active_symbols_count"] == 2
    assert summary["execution_eligible_symbols_count"] == 1
    assert summary["active_not_execution_eligible"] == ["ETHUSDT"]
    assert summary["open_position_symbols"] == ["FILUSDT"]
    assert summary["price_subscription_preserves_open_positions"] is True
    assert summary["registry_blocker_summary"]["missing_active_manifest_count"] == 1
    assert summary["paper_trade_summary"]["fees_all_zero"] is True


def test_step0_baseline_fails_when_required_runtime_artifact_missing(tmp_path: Path) -> None:
    _write_minimal_repo(tmp_path)
    report = build_step0_baseline(tmp_path)
    payload = report.to_dict()

    assert report.safe_to_start_step1 is False
    assert "runtime_artifacts" in payload["summary"]["current_failing_baseline_items"]
