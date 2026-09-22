from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.data.sources import SyntheticSource
from ai_finance.execution.backtest import SimulatedExecution
from ai_finance.execution.base import Order
from ai_finance.execution.paper import PaperExecution
from ai_finance.ops.alerts import engage_halt, release_halt
from ai_finance.ops.monitor import check_health
from ai_finance.ops.runner import run_once
from ai_finance.ops.state import StateStore, default_state_path
from ai_finance.risk.engine import RiskLimits

EPOCH = pd.Timestamp("2024-01-01", tz="UTC")
START = pd.Timestamp("2024-01-20 00:00", tz="UTC")
FAST_PARAMS = {"fast": 5, "slow": 10}


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AIFIN_DATA_DIR", str(tmp_path))
    return tmp_path


def feed(now: pd.Timestamp, *, drift: float = 2.0, seed: int = 4):
    """A synthetic source that behaves like a live feed frozen at ``now``."""
    return SyntheticSource(
        epoch_ms=int(EPOCH.timestamp() * 1000),
        seed=seed,
        annual_drift=drift,
        annual_vol=0.5,
        now_ms=lambda: int(now.timestamp() * 1000),
    )


def run(now: pd.Timestamp, **overrides):
    options = {
        "symbol": "PAPER",
        "interval": "1h",
        "strategy_name": "ma-crossover",
        "strategy_params": FAST_PARAMS,
        "source": feed(now),
        "costs": CostModel(),
        "limits": RiskLimits(max_position_weight=1.0, min_order_notional=10.0),
        "initial_equity": 5_000.0,
        "now": now,
    }
    options.update(overrides)
    return run_once(**options)


class TestModeGuards:
    def test_live_mode_is_refused(self):
        """There must be no code path to a real order before Phase 5."""
        with pytest.raises(NotImplementedError, match="live trading is not implemented"):
            run(START, mode="live")

    def test_the_refusal_explains_where_the_guarantee_comes_from(self):
        with pytest.raises(NotImplementedError, match=r"execution/live\.py"):
            run(START, mode="live")

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown mode"):
            run(START, mode="turbo")

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            run(START, strategy_name="clairvoyance")


class TestFirstRun:
    def test_creates_state_and_acts_on_the_latest_closed_bar(self):
        outcome = run(START)

        assert outcome.action in {"traded", "held"}
        assert outcome.bar_time is not None
        assert outcome.bar_time < START
        assert outcome.state.n_runs == 1

    def test_state_is_persisted_where_status_can_find_it(self):
        run(START)
        state = StateStore(default_state_path("PAPER", "paper")).load()

        assert state is not None
        assert state.symbol == "PAPER"
        assert state.last_bar_time is not None

    def test_writes_a_heartbeat(self):
        run(START)
        health = check_health("PAPER", "paper", 3600.0, now=START)
        assert health.status == "OK"

    def test_too_little_history_is_reported_not_crashed(self):
        """Evaluate what is stored, without syncing, when there is almost none."""
        from ai_finance.config import bars_dir
        from ai_finance.data.store import write_bars

        write_bars(
            feed(START).fetch_chunk("PAPER", "1m", int(EPOCH.timestamp() * 1000), limit=120),
            "PAPER",
            root=bars_dir(),
        )

        outcome = run(START, sync=False)

        assert outcome.action == "insufficient-data"
        assert "Back-fill more history" in outcome.message


class TestDataFailures:
    def test_a_failed_sync_holds_rather_than_trading_on_stale_prices(self):
        """An exchange outage must not be indistinguishable from a quiet market."""

        class BrokenFeed:
            name = "broken"

            def fetch_chunk(self, *args, **kwargs):
                raise ConnectionError("exchange unreachable")

        outcome = run(START, source=BrokenFeed())

        assert outcome.action == "sync-failed"
        assert "stale prices" in outcome.message
        assert not outcome.traded

    def test_a_failed_sync_is_announced(self):
        sent = []

        class Recorder:
            name = "recorder"

            def send(self, message):
                sent.append(message)
                return True

        class BrokenFeed:
            name = "broken"

            def fetch_chunk(self, *args, **kwargs):
                raise ConnectionError("exchange unreachable")

        run(START, source=BrokenFeed(), notifier=Recorder())

        assert sent, "an outage is exactly what an alert is for"
        assert "sync-failed" in sent[0]

    def test_a_failed_sync_still_writes_a_heartbeat(self):
        """The job ran; it just could not get data. That is not the same as dead."""

        class BrokenFeed:
            name = "broken"

            def fetch_chunk(self, *args, **kwargs):
                raise ConnectionError("exchange unreachable")

        run(START, source=BrokenFeed())
        assert check_health("PAPER", "paper", 3600.0, now=START).status == "OK"


class TestIdempotency:
    def test_a_second_run_on_the_same_bar_does_nothing(self):
        """What makes an overlapping schedule or a retry safe."""
        first = run(START)
        second = run(START + pd.Timedelta(minutes=2))

        assert second.action == "already-done"
        assert second.state.n_fills == first.state.n_fills
        assert second.state.units == first.state.units

    def test_running_five_times_in_a_row_trades_at_most_once(self):
        outcomes = [run(START + pd.Timedelta(minutes=i)) for i in range(5)]
        assert sum(1 for o in outcomes if o.traded) <= 1

    def test_a_new_bar_is_acted_on(self):
        run(START)
        later = run(START + pd.Timedelta(hours=2))

        assert later.action != "already-done"
        assert later.state.n_runs == 2


class TestTrading:
    def test_a_trending_market_eventually_opens_a_position(self):
        outcomes = [run(START + pd.Timedelta(hours=3 * i)) for i in range(8)]

        assert any(o.traded for o in outcomes), "a strong uptrend should trigger the crossover"
        # The position need not still be open at the end: a crossover exits
        # when the fast average drops back through the slow one.
        assert any(o.state.units > 0 for o in outcomes)

    def test_fills_are_recorded_in_state(self):
        for i in range(8):
            outcome = run(START + pd.Timedelta(hours=3 * i))
            if outcome.traded:
                break

        assert outcome.state.n_fills >= 1
        assert outcome.state.total_fees > 0
        assert outcome.state.cash < 5_000.0

    def test_equity_is_marked_at_the_live_price(self):
        outcome = run(START)
        assert outcome.equity == pytest.approx(
            outcome.state.cash + outcome.state.units * outcome.price
        )


class TestKillSwitch:
    def test_engaged_switch_forces_a_flat_target(self, data_dir):
        for i in range(8):
            outcome = run(START + pd.Timedelta(hours=3 * i))
            if outcome.state.units > 0:
                break
        assert outcome.state.units > 0, "need an open position to test flattening"

        engage_halt("test", data_dir)
        halted = run(START + pd.Timedelta(hours=40))

        assert halted.target_weight == 0.0
        assert "kill switch" in halted.message
        assert halted.state.units == pytest.approx(0.0)

    def test_switch_blocks_opening_a_new_position(self, data_dir):
        engage_halt("test", data_dir)

        outcomes = [run(START + pd.Timedelta(hours=3 * i)) for i in range(6)]

        assert all(o.state.units == 0.0 for o in outcomes)
        assert all(not o.traded for o in outcomes)

    def test_releasing_the_switch_allows_trading_again(self, data_dir):
        engage_halt("test", data_dir)
        run(START)
        release_halt(data_dir)

        outcomes = [run(START + pd.Timedelta(hours=3 * i)) for i in range(1, 8)]
        assert any(o.traded for o in outcomes)


class TestRiskPersistence:
    def test_a_drawdown_halt_survives_a_restart(self, data_dir):
        """A halt that reset itself every few hours would be worse than none."""
        run(START)

        # Simulate the risk engine having tripped in an earlier process.
        store = StateStore(default_state_path("PAPER", "paper"))
        state = store.load()
        state.halted_permanently = True
        state.halt_reason = "max drawdown 16.0% >= 15.0%"
        store.save(state)

        outcome = run(START + pd.Timedelta(hours=3))

        assert outcome.target_weight == 0.0
        assert "risk halt" in outcome.message
        assert outcome.state.halted_permanently

    def test_peak_equity_is_carried_between_runs(self):
        run(START)
        second = run(START + pd.Timedelta(hours=3))
        assert second.state.peak_equity > 0


class TestReconciliation:
    def test_paper_fills_match_backtest_fills_exactly(self):
        """The premise of Phase 4: a divergence must be about inputs, not arithmetic."""
        costs = CostModel()
        order = Order(
            symbol="BTCUSDT",
            delta_units=0.37,
            reference_price=64_250.5,
            timestamp=START,
            reason="test",
        )

        paper = PaperExecution(costs).submit(order)
        backtest = SimulatedExecution(costs).submit(order)

        assert paper == backtest

    def test_both_adapters_decline_a_zero_order(self):
        order = Order("BTCUSDT", 0.0, 100.0, START)
        assert PaperExecution(CostModel()).submit(order) is None
        assert SimulatedExecution(CostModel()).submit(order) is None

    def test_paper_execution_is_named_for_the_logs(self):
        assert PaperExecution(CostModel()).name == "paper"


class TestNotifications:
    def test_a_real_action_is_announced(self):
        sent = []

        class Recorder:
            name = "recorder"

            def send(self, message):
                sent.append(message)
                return True

        run(START, notifier=Recorder())
        assert sent, "the first decision should be reported"

    def test_a_no_op_run_stays_quiet(self):
        """Four alerts a day saying 'nothing happened' trains you to ignore them."""
        sent = []

        class Recorder:
            name = "recorder"

            def send(self, message):
                sent.append(message)
                return True

        run(START)
        run(START + pd.Timedelta(minutes=1), notifier=Recorder())

        assert sent == []
