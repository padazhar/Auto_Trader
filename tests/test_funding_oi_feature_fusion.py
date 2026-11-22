from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from features.market_loader import save_funding_rate_frame, save_market_frame, save_open_interest_frame
from features.timeframe_fusion import build_fused_frame


class _Paths(SimpleNamespace):
    def market_db_path(self, symbol: str) -> Path:
        return self.market_storage_dir / f"{str(symbol).upper()}.db"


def _make_paths(tmp_path: Path) -> _Paths:
    project_root = tmp_path / "repo"
    market_storage_dir = project_root / "data" / "market_storage"
    market_storage_dir.mkdir(parents=True, exist_ok=True)
    config = SimpleNamespace(
        frames=SimpleNamespace(anchor_interval="15m", fusion_intervals=["15m"], lookback_rows={"15m": 100}),
        feature_store=SimpleNamespace(include_optional_futures_context=True),
    )
    return _Paths(project_root=project_root, market_storage_dir=market_storage_dir, config=config)


def test_fused_frame_uses_canonical_funding_and_open_interest_tables(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    symbol = "BTCUSDT"

    market = pd.DataFrame(
        {
            "timestamp": [1710027000000, 1710027900000, 1710028800000],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 12.0, 11.0],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": [1710020000000, 1710028800000],
            "funding_rate": [0.0001, 0.0002],
        }
    )
    open_interest = pd.DataFrame(
        {
            "timestamp": [1710027900000, 1710028800000],
            "open_interest": [12345.0, 12375.0],
        }
    )

    save_market_frame(paths, symbol, "15m", market)
    save_funding_rate_frame(paths, symbol, funding)
    save_open_interest_frame(paths, symbol, "15m", open_interest)

    fused = build_fused_frame(paths, symbol, "15m", tail_rows=20)

    assert not fused.empty
    assert "15m_funding_rate" in fused.columns
    assert "15m_open_interest" in fused.columns
    last = fused.iloc[-1]
    assert float(last["15m_funding_rate"]) == 0.0002
    assert float(last["15m_open_interest"]) == 12375.0
