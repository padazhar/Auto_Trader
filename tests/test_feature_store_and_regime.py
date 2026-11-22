from __future__ import annotations

import pandas as pd

from features.dataset_preparation import build_feature_set_metadata
from inference.regime_filter import evaluate_regime


def _shadow_cfg():
    return type("Cfg", (), {
        "enabled": True,
        "crisis_realized_vol_z": 3.0,
        "high_realized_vol_z": 1.75,
        "funding_abs_z_block": 3.0,
        "open_interest_abs_z_block": 3.0,
        "weak_trend_strength": 0.35,
        "squeeze_width_threshold": 0.025,
        "high_vol_score_scale": 0.70,
        "stress_score_scale": 0.60,
        "rangebound_score_scale": 0.75,
    })()


def test_feature_set_signature_changes_with_recipe_version():
    base = build_feature_set_metadata(["a", "b"], recipe_version="feature_store_v1")
    changed = build_feature_set_metadata(["a", "b"], recipe_version="feature_store_v2")
    assert base["signature"] != changed["signature"]
    assert changed["recipe_version"] == "feature_store_v2"


def test_regime_reports_crisis_as_shadow_telemetry_without_blocking_entries():
    row = pd.Series({"15m_realized_vol_z_20": 3.5, "15m_bb_width_20": 0.03, "15m_trend_strength_20": 0.8})
    regime = evaluate_regime(row, interval="15m", regime_cfg=_shadow_cfg())
    payload = regime.as_dict()

    assert regime.name == "crisis_volatility"
    assert regime.allow_entries is True
    assert regime.score_scale == 1.0
    assert payload["mode"] == "shadow_only"
    assert payload["decision_influence_allowed"] is False
    assert payload["shadow_candidate_allow_entries"] is False
    assert payload["shadow_candidate_score_scale"] == 0.0


def test_regime_reports_candidate_scaling_but_active_fields_remain_neutral():
    row = pd.Series({"15m_realized_vol_z_20": 2.0, "15m_bb_width_20": 0.04, "15m_trend_strength_20": 0.8})
    regime = evaluate_regime(row, interval="15m", regime_cfg=_shadow_cfg())
    payload = regime.as_dict()

    assert regime.name == "high_volatility"
    assert payload["allow_entries"] is True
    assert payload["score_scale"] == 1.0
    assert payload["shadow_candidate_allow_entries"] is True
    assert payload["shadow_candidate_score_scale"] == 0.70
