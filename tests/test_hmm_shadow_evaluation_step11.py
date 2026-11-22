from __future__ import annotations

import json
from pathlib import Path

from core.artifact_contract import save_json
from core.path_manager import PathManager
from scripts.hmm_shadow_evaluation import run_hmm_shadow_evaluation


def _paths(tmp_path: Path) -> PathManager:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "system_settings.yaml").write_text(
        """
training:
  gbm:
    estimator: histgbm
  hmm:
    enabled: true
    candidate_training_enabled: true
    mandatory_paper_gate_enabled: false
    live_influence_enabled: false
    shadow_only: true
    min_posterior_confidence: 0.60
inference:
  hmm:
    enabled: true
    candidate_shadow_enabled: true
    mandatory_paper_gate_enabled: false
    live_influence_enabled: false
    require_valid_artifact_for_entry: false
    min_posterior_confidence: 0.60
runtime:
  execution_mode: paper
""".strip(),
        encoding="utf-8",
    )
    return PathManager.from_project_root(root)


def _hmm_manifest(paths: PathManager, *, symbol: str = "BTCUSDT", interval: str = "15m", candidate_id: str = "hmm01") -> Path:
    base = paths.hmm_models_dir / symbol / interval / "candidates" / candidate_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "model.joblib").write_bytes(b"not-used-in-step11-test")
    save_json(base / "feature_columns.json", {"columns": ["hmm_return_1"]})
    save_json(base / "metrics.json", {"avg_max_posterior": 0.78})
    manifest = base / "manifest.json"
    save_json(
        manifest,
        {
            "schema_version": 1,
            "model_type": "hmm",
            "symbol": symbol,
            "interval": interval,
            "model": "model.joblib",
            "feature_columns": "feature_columns.json",
            "metrics": "metrics.json",
            "lifecycle": {"stage": "candidate", "candidate_id": candidate_id},
            "hmm_contract": {
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_paper_gate_enabled": False,
                "live_influence_enabled": False,
                "schema_version": "hmm_observation_v1",
                "n_states": 4,
            },
        },
    )
    return manifest


def _paper_ledger(paths: PathManager, *, closed_trades: int = 5) -> None:
    paths.portfolio_state_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"symbol": "BTCUSDT", "pnl_usd": 1.0 if idx % 2 == 0 else -0.25, "closed_at": idx}
        for idx in range(closed_trades)
    ]
    paths.portfolio_state_path.write_text(
        json.dumps(
            {
                "authority_mode": "paper_local",
                "closed_trades": rows,
                "positions": {},
                "realized_pnl_usd": sum(float(row["pnl_usd"]) for row in rows),
                "reconciliation_status": "paper_local",
                "state_domain": "execution_portfolio",
                "state_owner": "live_trading_controller",
            }
        ),
        encoding="utf-8",
    )


def _readiness_snapshot(paths: PathManager, *, execution_eligible: int = 1, gbm_ready: bool = True) -> None:
    paths.monitoring_state_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(max(1, execution_eligible)):
        rows.append(
            {
                "symbol": f"BTC{idx}USDT",
                "interval": "15m",
                "ready": bool(gbm_ready),
                "execution_ready": bool(gbm_ready),
                "models": {
                    "rf": {"ready": True, "execution_ready": True, "deployment_stage": "paper_bootstrap"},
                    "gbm": {"ready": bool(gbm_ready), "execution_ready": bool(gbm_ready), "deployment_stage": "paper_bootstrap" if gbm_ready else "candidate"},
                },
            }
        )
    paths.monitoring_state_path.write_text(
        json.dumps(
            {
                "trading_readiness": {
                    "execution_eligible_count": execution_eligible if gbm_ready else 0,
                    "blockers": [] if gbm_ready else ["execution_eligible_symbols:0/1"],
                    "anchor_blockers": {} if gbm_ready else {"anchor_training_incomplete:gbm:deployment_stage:candidate": 1},
                    "execution_readiness_matrix": rows,
                }
            }
        ),
        encoding="utf-8",
    )


def _append_decision(
    paths: PathManager,
    *,
    confidence: float = 0.72,
    state: int = 1,
    effect: str = "none",
    mandatory: bool = False,
    counterfactual_action: str = "ENTER_LONG",
) -> None:
    paths.decision_journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "action": "HOLD",
        "effective_action": "HOLD",
        "components": {
            "hmm": {
                "available": True,
                "confidence": confidence,
                "decision_effect": effect,
                "decision_influence_allowed": effect != "none",
                "live_influence_enabled": False,
                "mandatory_gate_enabled": mandatory,
                "posteriors": {"0": 1.0 - confidence, "1": confidence},
                "reason": "hmm_shadow_ready",
                "shadow_only": True,
                "state": state,
                "used": effect != "none",
            },
            "shadow_candidate": {
                "action": counterfactual_action,
                "available": True,
                "different_action": counterfactual_action != "HOLD",
                "reason": "counterfactual_entry_support",
                "score": 0.001,
            },
        },
    }
    with paths.decision_journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_hmm_shadow_evaluation_recommends_review_without_activation(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config_before = paths.system_settings_path.read_text(encoding="utf-8")
    _hmm_manifest(paths)
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    for _ in range(5):
        _append_decision(paths, confidence=0.80)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(
        paths.project_root,
        write_report=True,
        min_samples=5,
        min_available_ratio=1.0,
        min_high_confidence_ratio=1.0,
    )

    assert report.ready_for_mandatory_paper_gate_review is True
    assert report.approval_recommendation == "paper_gate_review_ready"
    assert report.activation_performed is False
    assert report.config_mutation_performed is False
    assert report.pointer_mutation_performed is False
    assert report.hmm_decision_effect_allowed is False
    assert paths.system_settings_path.read_text(encoding="utf-8") == config_before
    persisted = json.loads((paths.runtime_dir / "hmm_shadow_evaluation_report.json").read_text(encoding="utf-8"))
    assert persisted["evaluation_type"] == "evidence_only_no_activation"


def test_hmm_shadow_evaluation_blocks_missing_evidence(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    report = run_hmm_shadow_evaluation(paths.project_root, min_samples=3)

    assert report.ready_for_mandatory_paper_gate_review is False
    assert report.approval_recommendation == "keep_hmm_shadow_only"
    assert "hmm_candidate_artifacts_missing" in report.reasons
    assert any(reason.startswith("insufficient_hmm_shadow_samples") for reason in report.reasons)


def test_hmm_shadow_evaluation_blocks_decision_influence_violation(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths)
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    for _ in range(3):
        _append_decision(paths, confidence=0.75, effect="entry_block")
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(paths.project_root, min_samples=3, min_available_ratio=1.0, min_high_confidence_ratio=1.0)

    assert report.ready_for_mandatory_paper_gate_review is False
    assert "hmm_decision_influence_violations:3" in report.reasons
    assert report.activation_performed is False


def test_hmm_shadow_evaluation_blocks_live_or_mandatory_violation(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths)
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    for _ in range(3):
        _append_decision(paths, confidence=0.75, mandatory=True)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(paths.project_root, min_samples=3, min_available_ratio=1.0, min_high_confidence_ratio=1.0)

    assert report.ready_for_mandatory_paper_gate_review is False
    assert "hmm_live_or_mandatory_violations:3" in report.reasons
    assert report.live_candidate_validation_allowed is False


def test_hmm_shadow_evaluation_reports_missing_decision_interval_artifacts(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths, symbol="BTCUSDT", interval="5m", candidate_id="hmm5m")
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    _append_decision(paths, confidence=0.0)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(paths.project_root, min_samples=1, min_available_ratio=0.0, min_high_confidence_ratio=0.0)

    assert report.ready_for_mandatory_paper_gate_review is False
    assert "hmm_valid_candidate_artifacts_missing_for_decision_keys:1" in report.reasons
    assert report.artifact_summary["missing_decision_artifact_keys"] == ["BTCUSDT:15m"]
    assert report.artifact_summary["shadow_only_artifact_interval_counts"] == {"5m": 1}


def test_hmm_shadow_evaluation_blocks_when_paper_or_anchor_evidence_is_missing(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths)
    for _ in range(5):
        _append_decision(paths, confidence=0.80)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(
        paths.project_root,
        min_samples=5,
        min_available_ratio=1.0,
        min_high_confidence_ratio=1.0,
    )

    assert report.ready_for_mandatory_paper_gate_review is False
    assert "authoritative_paper_ledger_missing" in report.reasons
    assert "readiness_evidence_missing" in report.reasons
    assert any(reason.startswith("insufficient_authoritative_paper_closed_trades") for reason in report.reasons)
    assert any(reason.startswith("insufficient_execution_eligible_symbols") for reason in report.reasons)


def test_hmm_shadow_evaluation_blocks_when_gbm_anchor_is_not_accepted(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths)
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=False)
    for _ in range(5):
        _append_decision(paths, confidence=0.80)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(
        paths.project_root,
        min_samples=5,
        min_available_ratio=1.0,
        min_high_confidence_ratio=1.0,
    )

    assert report.ready_for_mandatory_paper_gate_review is False
    assert "accepted_gbm_estimator_missing_or_blocked:0/1" in report.reasons
    assert "insufficient_execution_eligible_symbols:0/1" in report.reasons


def _append_nested_shadow_candidate_decision(paths: PathManager, *, confidence: float = 0.72, state: int = 1) -> None:
    paths.decision_journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "action": "HOLD",
        "effective_action": "HOLD",
        "components": {
            "shadow_candidate": {
                "components": {
                    "hmm": {
                        "available": True,
                        "confidence": confidence,
                        "decision_effect": "none",
                        "decision_influence_allowed": False,
                        "live_influence_enabled": False,
                        "mandatory_gate_enabled": False,
                        "posteriors": {"0": 1.0 - confidence, "1": confidence},
                        "reason": "hmm_shadow_ready",
                        "shadow_only": True,
                        "state": state,
                        "used": False,
                    }
                },
                "decision_effect": "none",
            }
        },
    }
    with paths.decision_journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_hmm_shadow_evaluation_reads_nested_shadow_candidate_hmm_components(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _hmm_manifest(paths, symbol="BTCUSDT", interval="15m", candidate_id="hmm15m")
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    for _ in range(4):
        _append_nested_shadow_candidate_decision(paths, confidence=0.78)
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(
        paths.project_root,
        min_samples=4,
        min_available_ratio=1.0,
        min_high_confidence_ratio=1.0,
        min_counterfactual_entries=0,
    )

    assert report.ready_for_mandatory_paper_gate_review is True
    assert report.decision_summary["decisions_with_hmm"] == 4
    assert report.decision_summary["available_decisions"] == 4
    assert report.decision_summary["high_confidence_decisions"] == 4
    assert report.activation_performed is False
    assert report.pointer_mutation_performed is False


def test_step12_post_activation_paper_gate_is_not_reported_as_shadow_violation(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    settings = paths.system_settings_path.read_text(encoding="utf-8")
    settings = settings.replace(
        "mandatory_paper_gate_enabled: false\n    live_influence_enabled: false\n    require_valid_artifact_for_entry: false",
        "mandatory_paper_gate_enabled: true\n    live_influence_enabled: false\n    require_valid_artifact_for_entry: true",
    )
    paths.system_settings_path.write_text(settings, encoding="utf-8")
    _hmm_manifest(paths)
    _paper_ledger(paths, closed_trades=5)
    _readiness_snapshot(paths, execution_eligible=1, gbm_ready=True)
    for _ in range(5):
        _append_decision(
            paths,
            confidence=0.80,
            effect="paper_entry_gate",
            mandatory=True,
            counterfactual_action="ENTER_LONG",
        )
    monkeypatch.setattr("scripts.hmm_shadow_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_hmm_shadow_evaluation(
        paths.project_root,
        write_report=True,
        min_samples=5,
        min_available_ratio=1.0,
        min_high_confidence_ratio=1.0,
    )

    assert report.ready_for_mandatory_paper_gate_review is True
    assert report.approval_recommendation == "mandatory_paper_gate_active_validated"
    assert report.evaluation_type == "paper_gate_post_activation_validation"
    assert report.activation_performed is True
    assert report.hmm_decision_effect_allowed is True
    assert report.decision_summary["decision_influence_violations"] == 0
    assert report.decision_summary["live_or_mandatory_violations"] == 0
    assert "mandatory_paper_gate_already_enabled" not in report.reasons
    assert not any(reason.startswith("hmm_decision_influence_violations") for reason in report.reasons)
    assert not any(reason.startswith("hmm_live_or_mandatory_violations") for reason in report.reasons)
    persisted = json.loads((paths.runtime_dir / "hmm_shadow_evaluation_report.json").read_text(encoding="utf-8"))
    assert persisted["evaluation_type"] == "paper_gate_post_activation_validation"
    assert persisted["config_state"]["approved_step12_paper_gate_active"] is True
