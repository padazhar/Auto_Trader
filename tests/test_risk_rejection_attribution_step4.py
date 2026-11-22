from __future__ import annotations

import json
from types import SimpleNamespace

from reporting.run_report import build_run_report


class DummyMonitor:
    def snapshot(self):
        return {
            "system": {"execution_mode": "paper", "started_at": 1.0},
            "trading_readiness": {"status": "BLOCKED", "reason": "test"},
            "paper_execution_runtime": {"recent_decisions": []},
        }

    def events(self, limit=500):
        return []


def test_run_report_summarizes_risk_rejection_attribution(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    decision_journal = runtime / "decision_journal.jsonl"
    row = {
        "decision_id": "risk-1",
        "ts": 10.0,
        "symbol": "BTCUSDT",
        "action": "ENTER_LONG",
        "effective_action": "HOLD",
        "execution_status": "rejected_risk",
        "execution_blocker_stage": "risk",
        "rejection_reason": "risk_distortion_exceeded",
        "score": 0.42,
        "risk_breakdown": {
            "schema_version": "risk_rejection.v1",
            "reason": "risk_distortion_exceeded",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "requested_notional_usd": 12.0,
            "approved_notional_usd": 25.0,
            "risk_distortion_fraction": 0.75,
            "risk_limit": 0.35,
            "clamp_reason": "max_notional",
        },
    }
    decision_journal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    paths = SimpleNamespace(
        runtime_dir=runtime,
        decision_journal_path=decision_journal,
        training_registry_path=runtime / "training_registry.json",
        symbol_state_path=runtime / "symbol_state.json",
    )

    report = build_run_report(paths, monitor=DummyMonitor())
    attribution = report["risk_rejection_attribution"]

    assert attribution["schema_version"] == "risk_rejection_attribution.v1"
    assert attribution["risk_rejection_count"] == 1
    assert attribution["reason_counts"] == {"risk_distortion_exceeded": 1}
    assert attribution["clamp_counts"] == {"max_notional": 1}
    assert attribution["requested_notional_usd_total"] == 12.0
    assert attribution["approved_notional_usd_total"] == 25.0
    assert attribution["recent"][0]["decision_id"] == "risk-1"
