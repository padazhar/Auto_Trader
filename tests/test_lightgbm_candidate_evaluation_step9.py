from __future__ import annotations

from pathlib import Path

from core.artifact_contract import save_json
from core.path_manager import PathManager
from scripts.lightgbm_candidate_evaluation import run_lightgbm_candidate_evaluation


def _paths(tmp_path: Path) -> PathManager:
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "system_settings.yaml").write_text(
        """
training:
  gbm:
    estimator: histgbm
  promotion:
    min_score_improvement: 0.01
runtime:
  execution_mode: paper
""".strip(),
        encoding="utf-8",
    )
    return PathManager.from_project_root(root)


def _gbm_manifest(paths: PathManager, *, symbol: str, interval: str, candidate_id: str, family: str, score: float) -> Path:
    manifest = paths.gbm_models_dir / symbol / interval / "candidates" / candidate_id / "manifest.json"
    save_json(
        manifest,
        {
            "schema_version": 4,
            "model_type": "gbm",
            "model_role": "bundle",
            "symbol": symbol,
            "interval": interval,
            "estimator_family": family,
            "estimator_contract": {
                "family": family,
                "direction_family": family,
                "meta_family": None,
                "selector_key": "training.gbm.estimator",
                "rollback_family": "histgbm",
            },
            "lifecycle": {"stage": "candidate", "candidate_id": candidate_id},
            "promotion": {"candidate_score": score, "approved": False, "promoted": False},
            "training": {"estimator_family": family, "estimator_selector": "training.gbm.estimator"},
            "timestamps": {"created_at": f"2026-01-01T00:00:{candidate_id[-2:]}+00:00"},
        },
    )
    return manifest


def _active_bundle_pointer(paths: PathManager, *, symbol: str, interval: str, manifest: Path, candidate_id: str) -> Path:
    pointer = paths.gbm_models_dir / symbol / interval / "bundle" / "incumbent.json"
    save_json(
        pointer,
        {
            "schema_version": 1,
            "model_type": "gbm",
            "model_role": "bundle",
            "symbol": symbol,
            "interval": interval,
            "candidate_id": candidate_id,
            "manifest_path": manifest.relative_to(paths.project_root).as_posix(),
            "deployment_stage": "paper_bootstrap",
            "paper_execution_only": True,
            "live_eligible": False,
        },
    )
    return pointer


def test_lightgbm_candidate_evaluation_recommends_paper_review_without_activation(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    baseline = _gbm_manifest(paths, symbol="BTCUSDT", interval="15m", candidate_id="hist01", family="histgbm", score=0.52)
    _active_bundle_pointer(paths, symbol="BTCUSDT", interval="15m", manifest=baseline, candidate_id="hist01")
    _gbm_manifest(paths, symbol="BTCUSDT", interval="15m", candidate_id="lgbm01", family="lightgbm", score=0.56)
    pointer_before = (paths.gbm_models_dir / "BTCUSDT" / "15m" / "bundle" / "incumbent.json").read_text(encoding="utf-8")

    monkeypatch.setattr("scripts.lightgbm_candidate_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_lightgbm_candidate_evaluation(paths.project_root, write_report=True)

    assert report.ready_for_lightgbm_paper_approval is True
    assert report.activation_performed is False
    assert report.items[0].status == "approved_for_paper_switch_review"
    assert report.items[0].paper_approval_recommended is True
    assert report.items[0].score_delta == 0.040000000000000036
    assert (paths.gbm_models_dir / "BTCUSDT" / "15m" / "bundle" / "incumbent.json").read_text(encoding="utf-8") == pointer_before
    payload = (paths.runtime_dir / "lightgbm_candidate_evaluation_report.json").read_text(encoding="utf-8")
    assert "evidence_report_only_no_pointer_or_config_mutation" in payload


def test_lightgbm_candidate_evaluation_retains_histgbm_when_candidate_does_not_improve(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    baseline = _gbm_manifest(paths, symbol="ETHUSDT", interval="15m", candidate_id="hist02", family="histgbm", score=0.60)
    _active_bundle_pointer(paths, symbol="ETHUSDT", interval="15m", manifest=baseline, candidate_id="hist02")
    _gbm_manifest(paths, symbol="ETHUSDT", interval="15m", candidate_id="lgbm02", family="lightgbm", score=0.605)
    monkeypatch.setattr("scripts.lightgbm_candidate_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_lightgbm_candidate_evaluation(paths.project_root)

    assert report.ready_for_lightgbm_paper_approval is False
    assert report.activation_performed is False
    assert report.items[0].status == "retain_histgbm"
    assert report.items[0].reasons == ["lightgbm_score_delta_below_threshold:0.005000<0.010000"]


def test_lightgbm_candidate_evaluation_blocks_missing_candidate(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    baseline = _gbm_manifest(paths, symbol="SOLUSDT", interval="15m", candidate_id="hist03", family="histgbm", score=0.58)
    _active_bundle_pointer(paths, symbol="SOLUSDT", interval="15m", manifest=baseline, candidate_id="hist03")
    monkeypatch.setattr("scripts.lightgbm_candidate_evaluation.validate_artifact_bundle", lambda _paths, _manifest: (True, "fresh"))

    report = run_lightgbm_candidate_evaluation(paths.project_root)

    assert report.ready_for_lightgbm_paper_approval is False
    assert report.items[0].status == "candidate_missing"
    assert report.items[0].reasons == ["lightgbm_candidate_missing"]
