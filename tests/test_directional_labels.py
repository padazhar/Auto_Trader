
from __future__ import annotations

import pandas as pd

from features.label_builder import build_direction_labels, build_meta_label_from_direction


def test_directional_labels_short_hold_long():
    df = pd.DataFrame(
        {
            '15m_close': [100.0, 99.0, 99.3, 101.0, 101.0],
            '15m_atr_pct_14': [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    out = build_direction_labels(df, anchor_interval='15m', horizon=1, atr_multiplier=0.5, atr_floor=0.001)
    labels = out['direction_label'].tolist()
    assert labels[0] == 0.0
    assert labels[1] == 1.0
    assert labels[2] == 2.0


def test_meta_label_uses_directional_profitability():
    cfg = type('Cfg', (), {'fees_bps': 3.0, 'slippage_bps': 2.0, 'min_profit_atr_mult': 0.3})()
    df = pd.DataFrame(
        {
            'future_return': [0.02, -0.02, 0.001],
            'thr_dyn': [0.01, 0.01, 0.01],
            'direction_pred': [2.0, 0.0, 1.0],
        }
    )
    meta = build_meta_label_from_direction(df, direction_pred_col='direction_pred', trading_cfg=cfg)
    assert meta.tolist() == [1.0, 1.0, 1.0]


def test_directional_labels_apply_cost_aware_threshold():
    cfg = type('Cfg', (), {'fees_bps': 3.0, 'slippage_bps': 2.0, 'spread_bps': 5.0, 'funding_bps': 0.0})()
    df = pd.DataFrame(
        {
            '15m_close': [100.0, 100.09, 100.30],
            '15m_atr_pct_14': [0.0001, 0.0001, 0.0001],
        }
    )
    no_cost = build_direction_labels(df, anchor_interval='15m', horizon=1, atr_multiplier=0.0, atr_floor=0.0004)
    with_cost = build_direction_labels(df, anchor_interval='15m', horizon=1, atr_multiplier=0.0, atr_floor=0.0004, trading_cfg=cfg)
    assert no_cost.loc[0, 'direction_label'] == 2.0
    assert with_cost.loc[0, 'direction_label'] == 1.0
    assert round(float(with_cost.loc[0, 'cost_rate']), 6) == 0.001
    assert round(float(with_cost.loc[0, 'cost_aware_thr']), 6) == 0.0014
