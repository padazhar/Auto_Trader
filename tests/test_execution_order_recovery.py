from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from execution.exchange_adapter import ExchangeAdapter, ExchangeOrderResult
from execution.fill_handler import FillHandler
from execution.order_models import OrderIntent
from execution.order_router import OrderRouter
from execution.order_store import OrderStore


class _SubmittedOnlyAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        self.place_calls = 0
        self.open_orders: list[dict[str, str]] = []

    def place_order(self, request, *, ts: float) -> ExchangeOrderResult:
        self.place_calls += 1
        self.open_orders = [{
            "client_order_id": request.client_order_id,
            "exchange_order_id": f"ex-{request.order_id}",
            "symbol": request.symbol,
            "status": "NEW",
        }]
        return ExchangeOrderResult(
            accepted=True,
            status="submitted",
            exchange_order_id=f"ex-{request.order_id}",
            exchange_status="NEW",
        )

    def get_open_orders(self) -> list[dict[str, str]]:
        return list(self.open_orders)


def test_order_router_is_idempotent_for_same_intent(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "order_state.json", tmp_path / "execution_journal.jsonl")
    adapter = _SubmittedOnlyAdapter()
    router = OrderRouter(store, adapter, FillHandler(store))

    intent = OrderIntent(
        symbol="BTCUSDT",
        action="ENTER_LONG",
        side="LONG",
        quantity=0.25,
        signal_reason="unit_test",
        metadata={"reference_price": 101.5, "idempotency_key": "bar-1"},
    )

    first = router.route_intent(intent, ts=123.0, previous_position=0)
    second = router.route_intent(intent, ts=124.0, previous_position=0)

    assert first["accepted"] is True
    assert first["duplicate"] is False
    assert second["accepted"] is True
    assert second["duplicate"] is True
    assert second["order_id"] == first["order_id"]
    assert adapter.place_calls == 1
    assert len(store.list_orders()) == 1


def test_order_router_recovers_open_orders_without_resubmission(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "order_state.json", tmp_path / "execution_journal.jsonl")
    adapter = _SubmittedOnlyAdapter()
    router = OrderRouter(store, adapter, FillHandler(store))

    first = router.route_intent(
        OrderIntent(
            symbol="BTCUSDT",
            action="ENTER_LONG",
            side="LONG",
            quantity=0.25,
            signal_reason="unit_test",
            metadata={"idempotency_key": "bar-2"},
        ),
        ts=123.0,
        previous_position=0,
    )
    assert first["status"] == "submitted"

    recovered = router.recover_open_orders(ts=125.0)
    assert recovered["open_order_count"] == 1
    assert len(recovered["recovered_orders"]) == 1
    assert recovered["recovered_orders"][0]["status"] == "submitted"
    assert recovered["recovered_orders"][0]["recovery_source"] == "exchange_open_orders"
    assert recovered["recovered_orders"][0]["recovery_authoritative"] is True
    assert recovered["recovery_sources"]["exchange_open_orders"] == 1
    assert adapter.place_calls == 1


def test_live_controller_recovers_execution_state_on_init(tmp_path: Path) -> None:
    from runners.live_trading_controller import LiveTradingController
    import runners.live_trading_controller as live_module

    class _DummyMonitor:
        def __init__(self, paths):
            self.events = []
        def publish_event(self, event_type, payload):
            self.events.append((event_type, payload))

    class _DummyControlStore:
        def __init__(self, paths):
            pass
        def pause_state(self):
            return {"paused": False, "pause_reason": ""}

    class _DummyRiskEngine:
        def __init__(self, paths):
            pass

    class _DummyLogic:
        def __init__(self, paths):
            pass

    class _RecoveringAdapter(_SubmittedOnlyAdapter):
        pass

    live_module.MonitorPublisher = _DummyMonitor
    live_module.ControlPlaneStore = _DummyControlStore
    live_module.RiskEngine = _DummyRiskEngine
    live_module.TradeDecisionLogic = _DummyLogic
    live_module.PaperExchangeAdapter = _RecoveringAdapter

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    seed_store = OrderStore(runtime_dir / "order_state.json", runtime_dir / "execution_journal.jsonl")
    adapter = _RecoveringAdapter()
    OrderRouter(seed_store, adapter, FillHandler(seed_store)).route_intent(
        OrderIntent(symbol="BTCUSDT", action="ENTER_LONG", side="LONG", quantity=0.25, metadata={"idempotency_key": "restart-1"}),
        ts=123.0,
        previous_position=0,
    )

    paths = SimpleNamespace(
        config=SimpleNamespace(
            frames=SimpleNamespace(anchor_interval="15m"),
            runtime=SimpleNamespace(execution_mode="paper", market_poll_seconds=1.0),
            risk=SimpleNamespace(account_balance_usd=1000.0),
            exchange=SimpleNamespace(position_reconciliation_enabled=False),
        ),
        runtime_dir=runtime_dir,
        portfolio_state_path=runtime_dir / "portfolio_state.json",
        tradable_symbols_path=runtime_dir / "tradable_symbols.yaml",
    )

    controller = LiveTradingController(paths, registry=SimpleNamespace(), signal_engine=SimpleNamespace(load_symbols=lambda: [], logic=None))
    recovered_orders = controller.order_store.list_open_orders()
    portfolio_payload = __import__("json").loads((runtime_dir / "portfolio_state.json").read_text(encoding="utf-8"))
    assert portfolio_payload["state_domain"] == "execution_portfolio"
    assert portfolio_payload["state_owner"] == "live_trading_controller"
    assert len(recovered_orders) == 1
    assert recovered_orders[0].client_order_id.startswith("cli_")
    assert any(event == "execution_state_recovered" for event, _ in controller.monitor.events)


def test_execution_state_files_include_explicit_owner_metadata(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "order_state.json", tmp_path / "execution_journal.jsonl")
    adapter = _SubmittedOnlyAdapter()
    router = OrderRouter(store, adapter, FillHandler(store))

    router.route_intent(
        OrderIntent(symbol="BTCUSDT", action="ENTER_LONG", side="LONG", quantity=0.25, metadata={"idempotency_key": "owner-meta-1"}),
        ts=123.0,
        previous_position=0,
    )

    order_payload = __import__("json").loads((tmp_path / "order_state.json").read_text(encoding="utf-8"))
    journal_row = __import__("json").loads((tmp_path / "execution_journal.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert order_payload["state_domain"] == "execution_orders"
    assert order_payload["state_owner"] == "live_trading_controller"
    assert journal_row["state_domain"] == "execution_orders"
    assert journal_row["state_owner"] == "live_trading_controller"


def test_order_router_recovers_paper_orders_from_local_ledger_when_exchange_surface_is_empty(tmp_path: Path) -> None:
    store = OrderStore(tmp_path / "order_state.json", tmp_path / "execution_journal.jsonl")
    adapter = _SubmittedOnlyAdapter()
    router = OrderRouter(store, adapter, FillHandler(store))

    router.route_intent(
        OrderIntent(
            symbol="BTCUSDT",
            action="ENTER_LONG",
            side="LONG",
            quantity=0.25,
            signal_reason="unit_test",
            metadata={"idempotency_key": "bar-3"},
        ),
        ts=123.0,
        previous_position=0,
    )
    adapter.open_orders = []

    recovered = router.recover_open_orders(ts=126.0)
    state = store.list_open_orders()[0]

    assert recovered["recovered_orders"][0]["recovery_source"] == "paper_local_ledger"
    assert recovered["recovered_orders"][0]["recovery_authoritative"] is True
    assert recovered["recovery_sources"]["paper_local_ledger"] == 1
    assert state.metadata["last_recovery_source"] == "paper_local_ledger"
    assert state.metadata["last_recovery_authoritative"] is True
