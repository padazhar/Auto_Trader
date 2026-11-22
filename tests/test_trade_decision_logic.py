from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from inference.expected_value import compute_action_evs
from inference.trade_decision_logic import TradeDecisionLogic


class DummyLoader:
    def __init__(self, rf_art, gbm_art, dqn_art=None):
        self._rf = rf_art
        self._gbm = gbm_art
        self._dqn = dqn_art

    def load_rf(self, symbol, interval):
        return self._rf

    def load_gbm(self, symbol, interval):
        return self._gbm

    def load_dqn(self, symbol, interval):
        return self._dqn

    def load_candidate_hmm(self, symbol, interval):
        return None


class DummyArtifact:
    def __init__(self, model, feature_columns, metrics):
        self.model = model
        self.feature_columns = feature_columns
        self.metrics = metrics
        self.manifest = {}


class DummyGBM:
    def __init__(self, direction, meta):
        self.direction = direction
        self.meta = meta


class RFModel:
    classes_ = [0, 1, 2]

    def predict(self, X):
        return [2]

    def predict_proba(self, X):
        return [[0.05, 0.15, 0.80]]


class GBMDirModel:
    classes_ = [0, 1, 2]

    def predict(self, X):
        return [2]

    def predict_proba(self, X):
        return [[0.10, 0.20, 0.70]]


class GBMMetaModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.05, 0.95]]


class HoldHeavyLongModel:
    classes_ = [0, 1, 2]

    def predict(self, X):
        return [2]

    def predict_proba(self, X):
        return [[0.10, 0.80, 0.10]]


class HoldHeavyGBMModel:
    classes_ = [0, 1, 2]

    def predict(self, X):
        return [2]

    def predict_proba(self, X):
        return [[0.05, 0.85, 0.10]]


class LowConfidenceMetaModel:
    classes_ = [0, 1]

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.45, 0.55]]


class StrongShortPolicyModel:
    classes_ = [0, 1, 2]

    def predict(self, X):
        return [2]

    def predict_proba(self, X):
        return [[0.05, 0.05, 0.90]]


def _paths():
    return SimpleNamespace(
        project_root=Path("/tmp/auto_trader_test"),
        runtime_dir=Path("/tmp/auto_trader_test/runtime"),
        logs_dir=Path("/tmp/auto_trader_test/logs"),
        monitoring_state_path=Path("/tmp/auto_trader_test/runtime/monitoring_state.json"),
        monitoring_events_path=Path("/tmp/auto_trader_test/runtime/monitoring_events.json"),
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            inference=SimpleNamespace(
                tail_rows=100,
                minimum_models_required=2,
                quality=SimpleNamespace(min_rows=10, min_precision_macro=0.1, min_roc_auc_ovr=0.5, min_balanced_accuracy=0.34),
                weights=SimpleNamespace(rf=0.5, gbm=0.4, dqn=0.0),
                dqn=SimpleNamespace(gray_band_low=0.45, gray_band_high=0.55),
                ev=SimpleNamespace(threshold=0.0, meta_threshold=0.55, dqn_adjustment_strength=0.0, strong_anchor_probability=0.62, weak_anchor_probability=0.55, weak_anchor_lower_tf_scale=0.60),
                consensus=SimpleNamespace(enter_threshold=0.18, flip_threshold=0.12, exit_band=0.06, direction_score_threshold=0.05, require_agreement=True),
            ),
            trading=SimpleNamespace(
                fees_bps=3.0,
                slippage_bps=2.0,
                min_profit_atr_mult=0.3,
                ev_threshold=0.0,
                meta_threshold=0.55,
                min_atr_fraction=0.0,
                short_probability_threshold=0.05,
                long_probability_threshold=0.05,
            ),
            risk=SimpleNamespace(account_balance_usd=1000.0, risk_per_trade=0.01, sl_atr_mult=1.5, tp_atr_mult=2.0, max_notional_usd=5000.0, min_notional_usd=25.0),
        )
    )


def _strong_metrics():
    return {
        "rows_total": 100,
        "precision_macro": 0.5,
        "roc_auc_ovr": 0.7,
        "balanced_accuracy": 0.5,
        "avg_win_return": 0.02,
        "avg_loss_return": 0.01,
        "fold_count": 3,
        "walk_forward_precision_macro_median": 0.45,
        "walk_forward_roc_auc_ovr_median": 0.65,
        "walk_forward_balanced_accuracy_median": 0.48,
        "walk_forward_precision_macro_min": 0.43,
        "walk_forward_roc_auc_ovr_min": 0.62,
        "walk_forward_balanced_accuracy_min": 0.45,
    }


def test_compute_action_evs_does_not_treat_hold_as_loss():
    evs = compute_action_evs(p_long=0.20, p_hold=0.70, p_short=0.10, avg_win=0.02, avg_loss=0.01)
    assert round(evs.long.value, 6) == 0.003
    assert round(evs.short.value, 6) == 0.0
    assert evs.best_overall.action == "LONG"


def test_decision_logic_enters_long(monkeypatch):
    paths = _paths()
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "ENTER_LONG"
    assert "risk_plan" in decision.components
    assert decision.components["rf"]["action_evs"]["best_overall_action"] == "LONG"


def test_decision_logic_holds_when_insufficient_models(monkeypatch):
    paths = _paths()
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(None, None, None)
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "HOLD"
    assert decision.reason == "insufficient_models"


def test_decision_logic_holds_when_hold_probability_dominates(monkeypatch):
    paths = _paths()
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(HoldHeavyLongModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(HoldHeavyGBMModel(), ["a"], _strong_metrics()),
            DummyArtifact(LowConfidenceMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "HOLD"
    assert decision.reason in {"negative_ev", "ev_hold"}
    assert decision.components["rf"]["action_evs"]["best_overall_action"] == "HOLD"




def test_decision_logic_uses_market_snapshot_price_fallback(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "runtime" / "market" / "latest.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        '{"updated_at": 1, "symbols": {"BTCUSDT": {"symbol": "BTCUSDT", "timeframe": "15m", "timestamp": 1, "close": 234.56, "volume": 9.0}}}',
        encoding="utf-8",
    )

    paths = _paths()
    paths.market_latest_snapshot_path = snapshot_path
    paths.market_db_path = lambda symbol: tmp_path / "missing.db"
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 0.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "ENTER_LONG"
    assert decision.components["price"] == 234.56
    assert decision.components["price_source"] == "market_latest_snapshot:15m"

def test_decision_logic_uses_market_db_price_fallback(monkeypatch, tmp_path):
    db_path = tmp_path / "BTCUSDT.db"
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE "klines_15m" (timestamp INTEGER, close REAL)')
        conn.execute('INSERT INTO "klines_15m" (timestamp, close) VALUES (?, ?)', (1, 123.45))
        conn.commit()

    paths = _paths()
    paths.market_db_path = lambda symbol: db_path
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 0.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    decision = logic.decide("BTCUSDT", "15m", previous_position=0)
    assert decision.action == "ENTER_LONG"
    assert decision.components["price"] == 123.45
    assert decision.components["price_source"] == "market_db_latest_close"


def test_hold_decision_carries_canonical_blocker(monkeypatch):
    paths = _paths()
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(None, None, None)

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.components["execution_eligible"] is False
    assert decision.components["decision_blocker"] == "insufficient_models"
    assert decision.components["decision_state"] == {
        "action": "HOLD",
        "reason": "insufficient_models",
        "execution_eligible": False,
        "blocker": "insufficient_models",
    }


def test_enter_decision_carries_execution_eligible_state(monkeypatch):
    paths = _paths()
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "ENTER_LONG"
    assert decision.components["execution_eligible"] is True
    assert decision.components["decision_blocker"] == ""
    assert decision.components["decision_state"]["action"] == "ENTER_LONG"
    assert decision.components["decision_state"]["execution_eligible"] is True


def test_confirmatory_policy_has_no_execution_influence_when_weight_is_zero(monkeypatch):
    paths = _paths()
    paths.config.inference.minimum_models_required = 1
    paths.config.inference.consensus.require_agreement = False
    paths.config.inference.dqn.gray_band_low = 0.0
    paths.config.inference.dqn.gray_band_high = 1.0
    paths.config.inference.weights.dqn = 0.0
    paths.config.inference.ev.dqn_adjustment_strength = 0.0

    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))

    base_logic = TradeDecisionLogic(paths)
    base_logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    base_decision = base_logic.decide("BTCUSDT", "15m", previous_position=0)

    policy_logic = TradeDecisionLogic(paths)
    policy_logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        DummyArtifact(StrongShortPolicyModel(), ["a"], _strong_metrics()),
    )
    policy_decision = policy_logic.decide("BTCUSDT", "15m", previous_position=0)

    assert policy_decision.action == base_decision.action
    assert policy_decision.score == base_decision.score
    assert policy_decision.components["dqn"]["role"] == "confirmatory_policy"
    assert policy_decision.components["dqn"]["available"] is True
    assert policy_decision.components["dqn"]["influence_enabled"] is False
    assert policy_decision.components["dqn"]["used"] is False


def test_cost_aware_ev_reduces_win_and_increases_loss():
    from inference.expected_value import compute_cost_rate, estimate_payoff_metrics

    cfg = SimpleNamespace(fees_bps=3.0, slippage_bps=2.0, spread_bps=1.0, funding_bps=0.0)
    cost_rate = compute_cost_rate(cfg)
    avg_win, avg_loss = estimate_payoff_metrics({'avg_win_return': 0.02, 'avg_loss_return': 0.01}, atr_fraction=0.01, cost_rate=cost_rate)
    assert round(cost_rate, 6) == 0.0006
    assert round(avg_win, 6) == 0.0194
    assert round(avg_loss, 6) == 0.0106


def test_regime_shadow_telemetry_does_not_change_active_decision(monkeypatch):
    frame = pd.DataFrame([{
        "timestamp": 1,
        "a": 1.0,
        "15m_close": 100.0,
        "15m_atr_pct_14": 0.01,
        "15m_realized_vol_z_20": 3.5,
        "15m_bb_width_20": 0.03,
        "15m_trend_strength_20": 0.8,
    }])
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: frame)

    shadow_paths = _paths()
    shadow_paths.config.regime = SimpleNamespace(
        enabled=True,
        crisis_realized_vol_z=3.0,
        high_realized_vol_z=1.75,
        funding_abs_z_block=3.0,
        open_interest_abs_z_block=3.0,
        weak_trend_strength=0.35,
        squeeze_width_threshold=0.025,
        high_vol_score_scale=0.70,
        stress_score_scale=0.60,
        rangebound_score_scale=0.75,
    )
    shadow_logic = TradeDecisionLogic(shadow_paths)
    shadow_logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )

    disabled_paths = _paths()
    disabled_paths.config.regime = SimpleNamespace(enabled=False)
    disabled_logic = TradeDecisionLogic(disabled_paths)
    disabled_logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )

    shadow_decision = shadow_logic.decide("BTCUSDT", "15m", previous_position=0)
    disabled_decision = disabled_logic.decide("BTCUSDT", "15m", previous_position=0)

    assert shadow_decision.action == disabled_decision.action == "ENTER_LONG"
    assert shadow_decision.reason == disabled_decision.reason == "ev_enter_long"
    assert shadow_decision.score == disabled_decision.score
    assert shadow_decision.components["consensus"]["regime_influence_applied"] is False
    assert shadow_decision.components["regime"]["name"] == "crisis_volatility"
    assert shadow_decision.components["regime"]["decision_influence_allowed"] is False
    assert shadow_decision.components["regime"]["shadow_candidate_allow_entries"] is False


def test_rf_gbm_ev_fusion_denominator_excludes_policy_and_regime_shadow(monkeypatch):
    """Step 6 baseline guard: RF and GBM remain the only numeric EV contributors.

    The confirmatory policy/DQN slot and deterministic regime output may publish
    telemetry, but while their execution influence is disabled they must not
    change the score numerator, denominator, action, or active decision reason.
    """
    frame = pd.DataFrame([{
        "timestamp": 1,
        "a": 1.0,
        "15m_close": 100.0,
        "15m_atr_pct_14": 0.01,
        "15m_realized_vol_z_20": 3.5,
        "15m_bb_width_20": 0.03,
        "15m_trend_strength_20": 0.8,
    }])
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: frame)

    paths = _paths()
    paths.config.inference.minimum_models_required = 2
    paths.config.inference.weights.dqn = 0.0
    paths.config.inference.ev.dqn_adjustment_strength = 0.0
    paths.config.regime = SimpleNamespace(
        enabled=True,
        crisis_realized_vol_z=3.0,
        high_realized_vol_z=1.75,
        funding_abs_z_block=3.0,
        open_interest_abs_z_block=3.0,
        weak_trend_strength=0.35,
        squeeze_width_threshold=0.025,
        high_vol_score_scale=0.70,
        stress_score_scale=0.60,
        rangebound_score_scale=0.75,
    )

    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        DummyArtifact(StrongShortPolicyModel(), ["a"], _strong_metrics()),
    )

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    rf_weight = float(paths.config.inference.weights.rf)
    gbm_weight = float(paths.config.inference.weights.gbm)
    rf_component = decision.components["rf"]["action_evs"]["best_directional_value"] * rf_weight
    gbm_component = decision.components["gbm"]["effective_ev"] * gbm_weight
    expected_score = (rf_component + gbm_component) / (rf_weight + gbm_weight)

    assert decision.action == "ENTER_LONG"
    assert decision.reason == "ev_enter_long"
    assert abs(decision.score - expected_score) <= 1e-12
    assert decision.components["consensus"]["contributing_models"] == 2
    assert decision.components["consensus"]["dqn_used"] is False
    assert decision.components["dqn"]["influence_enabled"] is False
    assert decision.components["regime"]["decision_influence_allowed"] is False
    assert decision.components["consensus"]["regime_influence_applied"] is False


def test_rf_anchor_and_gbm_lineage_are_preserved_in_baseline_decision(monkeypatch):
    """Step 6 guard: the corrected baseline still exposes RF as the anchor lineage.

    Later LightGBM/HMM work must compare against this baseline rather than
    silently replacing RF, changing the GBM role contract, or introducing an HMM
    contribution before the approved future gates.
    """
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]))

    paths = _paths()
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.components["lineage"]["anchor_interval"] == "15m"
    assert decision.components["rf"]["available"] is True
    assert decision.components["gbm"]["available"] is True
    assert decision.components["gbm"]["meta_available"] is True
    assert decision.components["hmm"]["decision_effect"] == "none"
    assert decision.components["hmm"]["decision_influence_allowed"] is False


def test_hmm_shadow_telemetry_is_attached_when_anchor_readiness_blocks(monkeypatch):
    """Step 11 corrective guard: HMM shadow samples must survive RF/GBM readiness HOLDs.

    The mandatory HMM gate is not enabled here. This only proves that a valid
    shadow prediction reaches the decision journal components even when the
    active RF/GBM anchor layer correctly blocks execution.
    """
    paths = _paths()
    paths.config.inference.hmm = SimpleNamespace(
        enabled=True,
        candidate_shadow_enabled=True,
        mandatory_paper_gate_enabled=False,
        live_influence_enabled=False,
        min_posterior_confidence=0.60,
    )
    monkeypatch.setattr(
        "inference.trade_decision_logic.anchor_model_readiness",
        lambda *args, **kwargs: SimpleNamespace(
            ready=False,
            reason="anchor_training_incomplete:gbm:deployment_stage:candidate",
            models={"rf": SimpleNamespace(reason="paper_bootstrap_ready"), "gbm": SimpleNamespace(reason="deployment_stage:candidate")},
        ),
    )
    monkeypatch.setattr(
        "inference.trade_decision_logic.build_fused_frame",
        lambda *args, **kwargs: pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}]),
    )
    monkeypatch.setattr(
        "inference.trade_decision_logic.predict_hmm_regime",
        lambda *args, **kwargs: SimpleNamespace(
            as_dict=lambda: {
                "available": True,
                "state": 2,
                "confidence": 0.72,
                "reason": "hmm_shadow_ready",
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_gate_enabled": False,
                "live_influence_enabled": False,
            }
        ),
    )

    logic = TradeDecisionLogic(paths)
    logic.loader = SimpleNamespace(load_candidate_hmm=lambda symbol, interval: object())

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.reason == "anchor_training_incomplete:gbm:deployment_stage:candidate"
    assert decision.components["hmm"]["available"] is True
    assert decision.components["hmm"]["reason"] == "hmm_shadow_ready"
    assert decision.components["hmm"]["used"] is False
    assert decision.components["hmm"]["decision_effect"] == "none"
    assert decision.components["hmm"]["decision_influence_allowed"] is False
    assert decision.components["anchor_readiness"]["gbm"] == "deployment_stage:candidate"


def _step12_hmm_gate_config(*, enabled: bool = True, require_valid: bool = True, live_influence: bool = False, confidence: float = 0.60):
    return SimpleNamespace(
        enabled=True,
        candidate_shadow_enabled=True,
        mandatory_paper_gate_enabled=enabled,
        live_influence_enabled=live_influence,
        require_valid_artifact_for_entry=require_valid,
        min_posterior_confidence=confidence,
    )


def _step12_entering_logic(paths):
    logic = TradeDecisionLogic(paths)
    logic.loader = DummyLoader(
        DummyArtifact(RFModel(), ["a"], _strong_metrics()),
        DummyGBM(
            DummyArtifact(GBMDirModel(), ["a"], _strong_metrics()),
            DummyArtifact(GBMMetaModel(), ["a"], _strong_metrics()),
        ),
        None,
    )
    return logic


def _step12_ready_frame():
    return pd.DataFrame([{"timestamp": 1, "a": 1.0, "15m_close": 100.0, "15m_atr_pct_14": 0.01}])


def test_step12_hmm_gate_blocks_missing_artifact_for_new_paper_entry(monkeypatch):
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: _step12_ready_frame())

    paths = _paths()
    paths.config.inference.hmm = _step12_hmm_gate_config(enabled=True, require_valid=True)
    paths.config.runtime = SimpleNamespace(execution_mode="paper")

    decision = _step12_entering_logic(paths).decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.reason == "hmm_gate_blocked:hmm_artifact_missing"
    assert decision.components["hmm"]["gate_used"] is True
    assert decision.components["hmm"]["gate_result"] == "blocked"
    assert "risk_plan" not in decision.components


def test_step12_hmm_gate_pass_preserves_rf_gbm_ev_score(monkeypatch):
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: _step12_ready_frame())

    baseline_paths = _paths()
    baseline_paths.config.inference.hmm = _step12_hmm_gate_config(enabled=False)
    baseline_paths.config.runtime = SimpleNamespace(execution_mode="paper")
    baseline = _step12_entering_logic(baseline_paths).decide("BTCUSDT", "15m", previous_position=0)

    monkeypatch.setattr(
        "inference.trade_decision_logic.predict_hmm_regime",
        lambda *args, **kwargs: SimpleNamespace(
            as_dict=lambda: {
                "available": True,
                "state": 2,
                "confidence": 0.91,
                "reason": "hmm_shadow_ready",
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_gate_enabled": True,
                "live_influence_enabled": False,
                "artifact": {"candidate_id": "hmm-step12"},
            }
        ),
    )

    gate_paths = _paths()
    gate_paths.config.inference.hmm = _step12_hmm_gate_config(enabled=True, require_valid=True)
    gate_paths.config.runtime = SimpleNamespace(execution_mode="paper")
    gate_logic = _step12_entering_logic(gate_paths)
    gate_logic.loader.load_candidate_hmm = lambda symbol, interval: object()
    gated = gate_logic.decide("BTCUSDT", "15m", previous_position=0)

    assert baseline.action == gated.action == "ENTER_LONG"
    assert baseline.reason == gated.reason == "ev_enter_long"
    assert abs(baseline.score - gated.score) <= 1e-12
    assert gated.components["hmm"]["gate_used"] is True
    assert gated.components["hmm"]["gate_result"] == "passed"
    assert gated.components["hmm"]["gate_reason"] == "hmm_valid_high_confidence_paper_confirmer"
    assert gated.components["hmm"]["artifact_governance_role"] == "shadow_candidate"
    assert gated.components["hmm"]["artifact_shadow_only"] is True
    assert gated.components["hmm"]["artifact_decision_influence_allowed"] is False
    assert gated.components["hmm"]["runtime_gate_policy"] == {
        "scope": "paper_entry_or_flip",
        "mandatory_enabled": True,
        "require_valid_artifact": True,
        "live_influence_enabled": False,
    }
    assert gated.components["consensus"]["contributing_models"] == 2


def test_step12_hmm_gate_blocks_low_confidence_candidate(monkeypatch):
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: _step12_ready_frame())
    monkeypatch.setattr(
        "inference.trade_decision_logic.predict_hmm_regime",
        lambda *args, **kwargs: SimpleNamespace(
            as_dict=lambda: {
                "available": True,
                "state": 1,
                "confidence": 0.41,
                "reason": "hmm_shadow_low_confidence",
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_gate_enabled": True,
                "live_influence_enabled": False,
            }
        ),
    )

    paths = _paths()
    paths.config.inference.hmm = _step12_hmm_gate_config(enabled=True, require_valid=True, confidence=0.60)
    paths.config.runtime = SimpleNamespace(execution_mode="paper")
    logic = _step12_entering_logic(paths)
    logic.loader.load_candidate_hmm = lambda symbol, interval: object()

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.reason == "hmm_gate_blocked:low_confidence"
    assert decision.components["hmm"]["gate_result"] == "blocked"
    assert decision.components["hmm"]["gate_reason"].startswith("hmm_low_confidence:")


def test_step12_hmm_gate_is_live_prohibited(monkeypatch):
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: _step12_ready_frame())
    monkeypatch.setattr(
        "inference.trade_decision_logic.predict_hmm_regime",
        lambda *args, **kwargs: SimpleNamespace(
            as_dict=lambda: {
                "available": True,
                "state": 2,
                "confidence": 0.91,
                "reason": "hmm_shadow_ready",
                "shadow_only": True,
                "decision_influence_allowed": False,
                "mandatory_gate_enabled": True,
                "live_influence_enabled": False,
            }
        ),
    )

    paths = _paths()
    paths.config.inference.hmm = _step12_hmm_gate_config(enabled=True, require_valid=True)
    paths.config.runtime = SimpleNamespace(execution_mode="live")
    logic = _step12_entering_logic(paths)
    logic.loader.load_candidate_hmm = lambda symbol, interval: object()

    decision = logic.decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "HOLD"
    assert decision.reason == "hmm_gate_blocked:paper_gate_live_mode_forbidden"
    assert decision.components["hmm"]["gate_result"] == "blocked"
    assert decision.components["hmm"]["live_influence_enabled"] is False


def test_step12_rollback_gate_off_allows_prior_paper_path(monkeypatch):
    monkeypatch.setattr("inference.trade_decision_logic.anchor_model_readiness", lambda *args, **kwargs: SimpleNamespace(ready=True, reason="ok", models={}))
    monkeypatch.setattr("inference.trade_decision_logic.build_fused_frame", lambda *args, **kwargs: _step12_ready_frame())

    paths = _paths()
    paths.config.inference.hmm = _step12_hmm_gate_config(enabled=False, require_valid=False)
    paths.config.runtime = SimpleNamespace(execution_mode="paper")

    decision = _step12_entering_logic(paths).decide("BTCUSDT", "15m", previous_position=0)

    assert decision.action == "ENTER_LONG"
    assert decision.reason == "ev_enter_long"
    assert decision.components["hmm"]["mandatory_gate_enabled"] is False
    assert decision.components["hmm"]["gate_result"] == "not_applicable"
