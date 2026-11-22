from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from training.base_trainer import BaseTrainer


class _Trainer(BaseTrainer):
    model_type = "rf"

    def train(self, symbol: str, interval: str):  # pragma: no cover - not needed for promotion unit tests
        raise NotImplementedError


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def _paths(tmp_path: Path, *, closed_trades: list[dict]) -> SimpleNamespace:
    portfolio_state_path = tmp_path / "portfolio_state.json"
    portfolio_state_path.write_text(json.dumps({"closed_trades": closed_trades}), encoding="utf-8")
    promotion = SimpleNamespace(
        enabled=True,
        allow_initial_promotion=True,
        allow_initial_promotion_without_paper_trades=False,
        allow_initial_paper_bootstrap_execution=False,
        require_outperform_incumbent=False,
        min_fold_count=1,
        min_score_improvement=0.0,
        min_metric_improvement=0.0,
        max_metric_regression=0.0,
        min_precision_macro=0.0,
        min_roc_auc_ovr=0.0,
        min_balanced_accuracy=0.0,
        min_precision=0.0,
        min_roc_auc=0.0,
        require_directional_quality=False,
        min_long_precision=0.0,
        min_short_precision=0.0,
        min_long_recall=0.0,
        min_short_recall=0.0,
        max_hold_prediction_fraction=1.0,
        min_directional_prediction_fraction=0.0,
        require_paper_expectancy=True,
        paper_lookback_trades=10,
        min_paper_trades=5,
        min_paper_expectancy_usd=0.0,
        min_paper_win_rate=0.45,
        max_paper_drawdown_usd=-50.0,
    )
    model_storage_dir = tmp_path / "model_storage"
    model_storage_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        portfolio_state_path=portfolio_state_path,
        model_storage_dir=model_storage_dir,
        config=SimpleNamespace(training=SimpleNamespace(promotion=promotion)),
    )


def test_promotion_rejects_negative_realized_paper_expectancy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(
        tmp_path,
        closed_trades=[
            {"symbol": "HYPEUSDT", "side": "LONG", "pnl_usd": -10.0, "closed_at": 1},
            {"symbol": "HYPEUSDT", "side": "SHORT", "pnl_usd": -8.0, "closed_at": 2},
            {"symbol": "HYPEUSDT", "side": "LONG", "pnl_usd": -7.0, "closed_at": 3},
            {"symbol": "HYPEUSDT", "side": "SHORT", "pnl_usd": -6.0, "closed_at": 4},
            {"symbol": "HYPEUSDT", "side": "LONG", "pnl_usd": 2.0, "closed_at": 5},
        ],
    )
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="HYPEUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["attempted"] is True
    assert state["approved"] is False
    assert "below_min_paper_expectancy" in state["reasons"]
    assert "below_min_paper_win_rate" in state["reasons"]
    assert state["paper_expectancy"]["trade_count"] == 5


def test_promotion_allows_positive_realized_paper_expectancy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(
        tmp_path,
        closed_trades=[
            {"symbol": "ZECUSDT", "side": "LONG", "pnl_usd": 10.0, "closed_at": 1},
            {"symbol": "ZECUSDT", "side": "SHORT", "pnl_usd": 8.0, "closed_at": 2},
            {"symbol": "ZECUSDT", "side": "LONG", "pnl_usd": -4.0, "closed_at": 3},
            {"symbol": "ZECUSDT", "side": "SHORT", "pnl_usd": 7.0, "closed_at": 4},
            {"symbol": "ZECUSDT", "side": "LONG", "pnl_usd": 6.0, "closed_at": 5},
        ],
    )
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="ZECUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is True
    assert state["paper_expectancy"]["expectancy_usd"] > 0


def test_initial_promotion_rejects_zero_paper_trades_by_default(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(tmp_path, closed_trades=[])
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is False
    assert "insufficient_paper_trades:0" in state["reasons"]
    assert state["paper_expectancy"]["trade_count"] == 0
    assert state["paper_evidence_status"] == "insufficient_paper_evidence"
    assert state["paper_bootstrap_only"] is False


def test_explicit_bootstrap_request_remains_candidate_only_and_non_executable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(tmp_path, closed_trades=[])
    paths.config.training.promotion.allow_initial_promotion_without_paper_trades = True
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is False
    assert "paper_bootstrap_not_authorized_for_execution" in state["reasons"]
    assert state["paper_evidence_status"] == "paper_bootstrap_not_execution_ready"
    assert state["paper_bootstrap_only"] is True
    assert state["paper_evidence_governance"]["execution_authorized"] is False
    assert state["paper_evidence_governance"]["live_eligible"] is False


def test_promotion_still_rejects_partial_insufficient_paper_trades(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(
        tmp_path,
        closed_trades=[{"symbol": "BTCUSDT", "side": "LONG", "pnl_usd": 5.0, "closed_at": 1}],
    )
    paths.config.training.promotion.allow_initial_promotion_without_paper_trades = True
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is False
    assert "insufficient_paper_trades:1" in state["reasons"]
    assert state["paper_evidence_status"] == "insufficient_paper_evidence"
    assert state["paper_bootstrap_only"] is False


def test_initial_paper_bootstrap_authorizes_paper_execution_only_when_no_incumbent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(tmp_path, closed_trades=[])
    paths.config.training.promotion.allow_initial_paper_bootstrap_execution = True
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is True
    assert state["paper_bootstrap_only"] is True
    assert state["paper_execution_only"] is True
    assert state["deployment_stage"] == "paper_bootstrap"
    assert state["live_eligible"] is False
    assert state["paper_evidence_status"] == "paper_bootstrap_execution_ready"
    assert state["paper_evidence_governance"]["execution_authorized"] is True
    assert state["paper_evidence_governance"]["live_eligible"] is False


def test_initial_paper_bootstrap_does_not_replace_existing_incumbent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(tmp_path, closed_trades=[])
    paths.config.training.promotion.allow_initial_paper_bootstrap_execution = True
    active_manifest = tmp_path / "already_active.json"
    active_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("training.base_trainer.resolve_active_manifest_path", lambda *args, **kwargs: active_manifest)
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is False
    assert "paper_bootstrap_initial_slot_only" in state["reasons"]
    assert state["paper_execution_only"] is False


def test_paper_promotion_never_consumes_backtest_database_rows_when_portfolio_ledger_is_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("training.base_trainer.get_logger", lambda name, paths: _Logger())
    paths = _paths(tmp_path, closed_trades=[])
    trade_db = tmp_path / "trade_history.db"
    with sqlite3.connect(trade_db) as conn:
        conn.execute(
            "CREATE TABLE closed_trades (symbol TEXT, side TEXT, mode TEXT, pnl_usd REAL, exit_reason TEXT, exit_time REAL, created_at REAL)"
        )
        conn.executemany(
            "INSERT INTO closed_trades VALUES (?, ?, ?, ?, ?, ?, ?)",
            [("BTCUSDT", "LONG", "paper", 100.0, "tp", float(i), float(i)) for i in range(1, 6)],
        )
    paths.trade_history_db_path = trade_db
    trainer = _Trainer(paths)

    state = trainer.evaluate_candidate_promotion(
        symbol="BTCUSDT",
        interval="15m",
        metrics={"fold_count": 1, "balanced_accuracy": 0.99, "precision": 0.99, "roc_auc": 0.99},
        manifest_path=tmp_path / "manifest.json",
        promote_requested=True,
    )

    assert state["approved"] is False
    assert "insufficient_paper_trades:0" in state["reasons"]
    assert state["paper_expectancy"]["trade_count"] == 0
    assert state["paper_expectancy"]["outcome_source"] == "paper_execution_ledger"
    assert state["paper_expectancy"]["authoritative_for_paper_promotion"] is True
