from __future__ import annotations

import json

from scripts.paper_24h_validation import MIN_PAPER_VALIDATION_SECONDS
from scripts.post_paper_attribution_analysis import analyze_post_paper_attribution
from scripts.retrain_live_eligible_candidates import RETRAINING_CONTRACT_VERSION
from scripts.step9_validation_pipeline import run_step9_validation_pipeline


class _Report:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _runtime_dir(project_root):
    path = project_root / "data" / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)
    return path


def _write_paper_report(runtime_dir, **extra):
    payload = {
        "paper_24h_passed": True,
        "duration_seconds": MIN_PAPER_VALIDATION_SECONDS,
        "execution_mode": "paper",
        "runtime_stability_clean": True,
        "governed_retraining_completed": True,
        "registry_gate_consistent": True,
        "fees_slippage_spread_included": True,
    }
    payload.update(extra)
    (runtime_dir / "paper_24h_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_run_report(runtime_dir, **extra):
    payload = {
        "execution": {
            "completed_trades": 2,
            "total_pnl": -8.83,
            "gross_pnl_usd": -8.50,
            "fees_usd": 0.20,
            "slippage_usd": 0.13,
        },
        "closed_trade_attribution": {
            "closed_trade_count": 2,
            "missing_decision_id_count": 0,
            "prediction_match_counts": {"False": 2},
        },
        "trade_excursion_attribution": {
            "closed_trade_count": 2,
            "avg_mfe_bps": 4.0,
            "avg_mae_bps": 12.0,
        },
        "risk_rejection_attribution": {
            "risk_rejection_count": 57,
            "reason_counts": {"risk_distortion_exceeded": 57},
            "clamp_counts": {"max_notional": 57},
        },
    }
    payload.update(extra)
    report_path = runtime_dir / "reports" / "run_report_step6.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def test_post_paper_attribution_blocks_until_24h_paper_passes(project_root):
    runtime_dir = _runtime_dir(project_root)
    (runtime_dir / "paper_24h_validation_report.json").unlink(missing_ok=True)

    report = analyze_post_paper_attribution(project_root)

    assert report.attribution_analysis_passed is False
    assert any(item.name == "paper_24h_gate" and item.status == "warn" for item in report.items)


def test_post_paper_attribution_accepts_complete_evidence(project_root):
    runtime_dir = _runtime_dir(project_root)
    run_report_path = _write_run_report(runtime_dir)
    _write_paper_report(runtime_dir, run_report_path=str(run_report_path))

    report = analyze_post_paper_attribution(project_root)

    assert report.overall_status == "pass"
    assert report.attribution_analysis_passed is True
    names = {item.name for item in report.items}
    assert {"closed_trade_attribution", "mae_mfe_attribution", "risk_rejection_attribution", "cost_attribution"}.issubset(names)


def test_post_paper_attribution_rejects_missing_cost_fields(project_root):
    runtime_dir = _runtime_dir(project_root)
    run_report_path = _write_run_report(runtime_dir, execution={"completed_trades": 1, "total_pnl": 1.0})
    _write_paper_report(runtime_dir, run_report_path=str(run_report_path))

    report = analyze_post_paper_attribution(project_root)

    assert report.attribution_analysis_passed is False
    assert report.overall_status == "fail"
    assert any(item.name == "cost_attribution" and item.status == "fail" for item in report.items)


def test_step9_requires_post_paper_attribution_before_live_candidate(project_root, monkeypatch):
    runtime_dir = _runtime_dir(project_root)
    (runtime_dir / "retrain_live_eligible_report.json").write_text(
        json.dumps(
            {
                "report_status": "pass",
                "retraining_contract_version": RETRAINING_CONTRACT_VERSION,
                "jobs_requested": 1,
                "jobs_succeeded": 1,
                "jobs_failed": 0,
            }
        ),
        encoding="utf-8",
    )
    _write_paper_report(runtime_dir)

    monkeypatch.setattr("scripts.step9_validation_pipeline.run_runtime_self_check", lambda root: _Report(overall_status="pass", runtime_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_registry_readiness_verification", lambda root: _Report(overall_status="pass", registry_gate_consistent=True, active_symbols_count=10, symbol_state_execution_eligible_count=10, registry_execution_ready_count=10, false_ready_symbols=[], registry_ready_not_in_symbol_state=[]))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_paper_mode_smoke_check", lambda root: _Report(overall_status="pass", paper_mode_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_paper_mode_soak_check", lambda root: _Report(overall_status="pass", paper_mode_soak_ready=True))
    monkeypatch.setattr("scripts.step9_validation_pipeline.run_final_live_validation_gate", lambda root: _Report(overall_status="pass", controlled_live_candidate_ready=True, live_trading_ready=True))

    report = run_step9_validation_pipeline(project_root, run_retrain=False)
    payload = report.to_dict()

    assert payload["ready_for_live_candidate"] is False
    assert payload["live_trading_ready"] is False
    item = next(item for item in payload["items"] if item["name"] == "post_paper_attribution_analysis")
    assert item["details"]["attribution_analysis_passed"] is False
