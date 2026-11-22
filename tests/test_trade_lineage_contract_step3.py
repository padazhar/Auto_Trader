from types import SimpleNamespace

from execution.order_models import OrderIntent
from execution.order_store import OrderStore
from execution.portfolio_state import PortfolioStateStore
from services.signal_engine import build_decision_lineage_id, normalize_lineage_payload


def test_order_lifecycle_preserves_decision_lineage(tmp_path):
    store = OrderStore(tmp_path / "orders.json", tmp_path / "orders.jsonl")
    decision_id = build_decision_lineage_id(
        symbol="BTCUSDT",
        interval="15m",
        action="ENTER_LONG",
        reason="ev_enter_long",
        score=0.001,
        previous_position=0,
        ts=1000.0,
    )
    lineage = normalize_lineage_payload({}, decision_id=decision_id, symbol="BTCUSDT", interval="15m", ts=1000.0, source="unit_test")

    order = store.create_from_intent(
        OrderIntent(
            symbol="BTCUSDT",
            action="ENTER_LONG",
            side="LONG",
            quantity=0.1,
            signal_reason="ev_enter_long",
            lineage=lineage,
        ),
        ts=1000.0,
    )

    assert order.lineage["decision_id"] == decision_id
    request = store.to_request(order)
    assert request.lineage["decision_id"] == decision_id
    journal = (tmp_path / "orders.jsonl").read_text(encoding="utf-8")
    assert decision_id in journal


def test_portfolio_open_and_close_preserve_lineage(tmp_path):
    store = PortfolioStateStore(tmp_path / "portfolio.json", starting_equity_usd=1000.0)
    state = store.load()
    lineage = {"decision_id": "dec_unit", "contract_version": "trade_lineage_v1"}

    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100.0,
        ts=10.0,
        position_plan={"quantity": 1.0, "notional_usd": 100.0, "stop_loss": 90.0, "take_profit": 110.0, "lineage": lineage},
    )
    assert state.positions["BTCUSDT"].lineage["decision_id"] == "dec_unit"

    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="EXIT_LONG",
        price=105.0,
        ts=20.0,
        position_plan={"exit_reason": "protective_take_profit"},
    )
    assert state.closed_trades[-1].lineage["decision_id"] == "dec_unit"


def test_close_event_preserves_risk_plan_and_existing_lineage(tmp_path):
    from execution.portfolio_state import PortfolioStateStore

    store = PortfolioStateStore(tmp_path / "portfolio.json", starting_equity_usd=1000.0)
    state = store.load()
    lineage = {"decision_id": "dec_entry", "order_id": "ord_entry", "contract_version": "trade_lineage_v1"}
    risk_plan = {
        "quantity": 2.0,
        "notional_usd": 200.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "risk_amount_usd": 10.0,
        "stop_distance": 5.0,
        "clamp_reason": "none",
        "lineage": lineage,
    }

    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="ENTER_LONG",
        price=100.0,
        ts=1000.0,
        position_plan=risk_plan,
    )
    store.apply_execution(
        state,
        symbol="BTCUSDT",
        action="EXIT_LONG",
        price=94.0,
        ts=1010.0,
        position_plan={"exit_reason": "protective_stop_loss", "lineage": {"close_order_id": "ord_close", "close_trigger_source": "protective_exit"}},
    )

    closed = state.closed_trades[-1]
    assert closed.lineage["decision_id"] == "dec_entry"
    assert closed.lineage["close_order_id"] == "ord_close"
    assert closed.lineage["close_trigger"] == "stop_loss"
    assert closed.close_event["trigger"] == "stop_loss"
    assert closed.close_event["is_sl_tp"] is True
    assert closed.risk_plan["stop_loss"] == 95.0
    assert closed.risk_plan["take_profit"] == 110.0
