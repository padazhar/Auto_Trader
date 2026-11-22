from __future__ import annotations

import json
from types import SimpleNamespace

from reporting.run_report import build_run_report, write_run_report
from runtime.shutdown_coordinator import ShutdownState


class DummyMonitor:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self):
        return self._snapshot

    def events(self, limit=500):
        return [
            {"level": "WARNING", "component": "runtime", "category": "shutdown_requested", "message": "stop"},
            {"level": "ERROR", "component": "test", "category": "x", "message": "boom"},
        ]


def _paths(tmp_path):
    runtime = tmp_path / "data" / "runtime"
    runtime.mkdir(parents=True)
    data = tmp_path / "data"
    csv = data / "live_backtest_trades.csv"
    csv.write_text(
        "symbol,side,mode,entry_time,entry_price,exit_time,exit_price,pnl_usd,pnl_pct,notional_usd,exit_reason\n"
        "BTCUSDT,LONG,paper,1,100,2,101,4.0,1,500,tp\n"
        "ETHUSDT,SHORT,paper,3,200,4,210,-2.0,-1,500,sl\n",
        encoding="utf-8",
    )
    portfolio_state = runtime / "portfolio_state.json"
    portfolio_state.write_text(
        json.dumps({
            "realized_pnl_usd": 7.0,
            "closed_trades": [
                {"symbol": "BTCUSDT", "side": "LONG", "pnl_usd": 9.0, "closed_at": 2},
                {"symbol": "ETHUSDT", "side": "SHORT", "pnl_usd": -2.0, "closed_at": 4},
            ],
        }),
        encoding="utf-8",
    )
    registry = runtime / "training_registry.json"
    registry.write_text(
        json.dumps(
            {
                "rf:BTCUSDT:15m": {
                    "model_type": "rf",
                    "symbol": "BTCUSDT",
                    "interval": "15m",
                    "status": "success",
                    "registry_truth_status": "execution_ready",
                    "execution_ready": True,
                    "active_manifest_valid": True,
                    "candidate_saved": True,
                    "promotion_attempted": True,
                    "promotion_approved": True,
                    "rows_used": 100,
                    "metrics": {
                        "precision_macro": 0.5,
                        "balanced_accuracy": 0.44,
                        "directional_quality": {
                            "available": True,
                            "long_precision": 0.6,
                            "long_recall": 0.4,
                            "short_precision": 0.55,
                            "short_recall": 0.45,
                            "hold_prediction_fraction": 0.2,
                            "directional_prediction_fraction": 0.8,
                            "confusion_matrix": {"short": {"short": 2}},
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    decision_journal = runtime / "decision_journal.jsonl"
    decision_journal.write_text(
        "\n".join(
            [
                json.dumps({
                    "symbol": "BTCUSDT",
                    "action": "ENTER_LONG",
                    "reason": "ev_enter_long",
                    "components": {
                        "hmm": {
                            "gate_applies": True,
                            "gate_used": True,
                            "gate_result": "passed",
                            "gate_reason": "hmm_valid_high_confidence_paper_confirmer",
                        }
                    },
                }),
                json.dumps({
                    "symbol": "ETHUSDT",
                    "action": "HOLD",
                    "reason": "negative_ev",
                    "components": {
                        "hmm": {
                            "gate_applies": False,
                            "gate_used": False,
                            "gate_result": "not_applicable",
                            "gate_reason": "not_entry_or_flip",
                        }
                    },
                }),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        project_root=tmp_path,
        runtime_dir=runtime,
        reports_dir=runtime / "reports",
        live_backtest_trades_path=csv,
        training_registry_path=registry,
        registry_path=registry,
        portfolio_state_path=portfolio_state,
        decision_journal_path=decision_journal,
        symbol_state_path=runtime / "symbol_state.json",
    )


def test_build_run_report_uses_execution_trade_source_and_registry(tmp_path):
    paths = _paths(tmp_path)
    snapshot = {
        "system": {"launcher_mode": "all", "execution_mode": "paper", "started_at": 1000.0},
        "execution_runtime": {"summary": {"completed_count": 0}, "open_position_count": 1},
        "market_data": {"runtime_price_snapshot": {"expected_symbol_count": 2, "priced_symbol_count": 2, "age_seconds": 0.5}},
        "trading_readiness": {"status": "BLOCKED", "reason": "registry_missing_active_manifest"},
        "controllers": {"market_data": {"status": "running"}},
    }

    report = build_run_report(paths, shutdown_state=ShutdownState(requested=True, reason="unit"), monitor=DummyMonitor(snapshot))

    assert report["execution"]["source"] == "paper_execution_ledger"
    assert report["execution"]["completed_trades"] == 2
    assert report["execution"]["wins"] == 1
    assert report["execution"]["losses"] == 1
    assert report["execution"]["total_pnl"] == 7.0
    assert report["execution"]["trade_result_source"]["selected_source"] == "paper_execution_ledger"
    assert report["training"]["entry_count"] == 1
    quality_rows = report["training"]["ml_quality_report"]
    assert quality_rows[0]["long_precision"] == 0.6
    assert quality_rows[0]["short_recall"] == 0.45
    assert quality_rows[0]["confusion_matrix"] == {"short": {"short": 2}}
    assert report["runtime"]["shutdown"]["reason"] == "unit"
    hmm_summary = report["signals"]["hmm_gate_summary"]
    assert hmm_summary["rows_with_gate_applicable"] == 1
    assert hmm_summary["rows_with_gate_used"] == 1
    assert hmm_summary["gate_result_counts"] == {"passed": 1, "not_applicable": 1}
    assert hmm_summary["artifact_governance_role"] == "shadow_candidate"
    assert hmm_summary["runtime_gate_scope"] == "paper_entry_or_flip"
    assert hmm_summary["live_influence_allowed"] is False
    enhanced = report["enhanced_structured_report"]
    assert enhanced["system_state_table"]
    assert enhanced["system_health_table"]
    assert enhanced["ml_training_forensic_table"][0]["symbol"] == "BTCUSDT"


def test_write_run_report_writes_json_and_markdown_under_runtime_reports(tmp_path):
    paths = _paths(tmp_path)
    snapshot = {"system": {"launcher_mode": "all", "execution_mode": "paper", "started_at": 1000.0}}

    result = write_run_report(paths, shutdown_state={"requested": True}, monitor=DummyMonitor(snapshot), timestamp="20260506_120000")

    json_path = paths.reports_dir / "run_report_20260506_120000.json"
    md_path = paths.reports_dir / "run_report_20260506_120000.md"
    assert result["json_path"] == str(json_path)
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "run_report.v1"
    assert "enhanced_structured_report" in payload
    markdown = md_path.read_text(encoding="utf-8")
    assert "Auto Trader Final Run Report" in markdown
    assert "Enhanced Structured Trading System Report" in markdown
    assert "ML Training Forensic Report" in markdown


def test_run_report_exposes_cumulative_and_training_error_evidence(tmp_path):
    paths = _paths(tmp_path)
    snapshot = {
        "system": {"launcher_mode": "all", "execution_mode": "paper", "started_at": 1000.0},
        "trading_readiness": {
            "status": "degraded",
            "blockers": ["active_universe_count_mismatch:5/10", "blocked_symbols:15"],
        },
        "runtime_event_counters": {"warnings_since_start": 7, "errors_since_start": 3},
        "error_state": {
            "active_controller_errors": {"training": {"controller": "training", "error": "boom", "ts": 2.0}},
            "recovered_controller_errors": [{"controller": "market_data", "error": "old", "ts": 1.0, "recovered_at": 2.0}],
        },
        "training_runtime": {"failed_jobs": [{"model_type": "rf", "symbol": "BTCUSDT", "error": "invalid labels"}]},
        "training_registry": {"counts": {"failed": 2}},
    }

    report = build_run_report(paths, monitor=DummyMonitor(snapshot))

    errors = report["warnings_errors"]
    assert errors["warnings_count"] >= 7
    assert errors["errors_count"] >= 3
    assert errors["errors_since_start"] >= 3
    assert errors["active_controller_errors_count"] == 1
    assert errors["recovered_controller_errors_count"] == 1
    assert errors["training_failure_evidence_count"] == 2
    assert report["readiness"]["blocker_report"]["runtime_blocker_count"] == 2
