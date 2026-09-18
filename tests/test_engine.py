"""Engine tests, including the Phase 1 gate from docs/PLAN.md.

The gate: a buy-and-hold strategy must reproduce the asset's price return, and a
random strategy must lose its cost bill and nothing else. Both are asserted as
exact identities rather than loose tolerances — if the engine is off by a basis
point, something is wrong and a vague bound would hide it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.data.schema import normalize_bars
from ai_finance.data.sources import SyntheticSource
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.base import MarketState, Signal
from ai_finance.strategy.baselines import (
    AlwaysFlat,
    BuyAndHold,
    RandomStrategy,
    TargetWeightSchedule,
)

FREE = CostModel.free()
OPEN = RiskLimits.unconstrained()


def make_bars(rows, start="2024-01-01"):
    """Build canonical bars from ``(open, high, low, close)`` tuples."""
    step = 60_000
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    return normalize_bars(
        pd.DataFrame(
            {
                "open_time": [start_ms + i * step for i in range(len(rows))],
                "open": [r[0] for r in rows],
                "high": [r[1] for r in rows],
                "low": [r[2] for r in rows],
                "close": [r[3] for r in rows],
                "volume": [1.0] * len(rows),
                "trades": [1] * len(rows),
                "close_time": [start_ms + (i + 1) * step - 1 for i in range(len(rows))],
            }
        )
    )


def continuous_bars(closes, first_open=100.0):
    """Bars where each open equals the previous close, as a real series roughly is."""
    opens = [first_open, *closes[:-1]]
    return make_bars([(o, max(o, c), min(o, c), c) for o, c in zip(opens, closes, strict=True)])


def synth(n=5000, seed=1, **kwargs):
    return SyntheticSource(epoch_ms=0, seed=seed, **kwargs).fetch_chunk("S", "1m", 0, limit=n)


class TestHandComputedArithmetic:
    """One trade, every number checked by hand."""

    def test_entry_fill_cash_units_and_fee(self):
        bars = make_bars(
            [
                (100.0, 100.0, 100.0, 100.0),
                (100.0, 110.0, 100.0, 110.0),
                (110.0, 110.0, 110.0, 110.0),
            ]
        )
        costs = CostModel(fee_rate=0.001, half_spread_bps=0.0, slippage_bps=0.0)

        result = run_backtest(
            bars,
            TargetWeightSchedule({0: 1.0}),
            symbol="X",
            initial_equity=1000.0,
            costs=costs,
            limits=OPEN,
        )

        # Signal on bar 0 fills at bar 1's open of 100. Buying 100% of equity
        # while also paying a 0.1% fee is impossible, so the buy is clamped to
        # what the cash can settle: 1000 / (100 * 1.001) units.
        expected_units = 1000.0 / (100.0 * 1.001)
        assert result.n_fills == 1
        fill = result.fills[0]
        assert fill.delta_units == pytest.approx(expected_units)
        assert fill.fill_price == 100.0
        assert fill.fee == pytest.approx(expected_units * 100.0 * 0.001)
        assert fill.cash_flow == pytest.approx(1000.0)  # spends the account exactly

        # Marked at the final close of 110.
        assert result.final_equity == pytest.approx(expected_units * 110.0)
        assert result.final_equity == pytest.approx(1098.9010989010989)
        assert result.total_return == pytest.approx(0.0989010989010989)
        assert result.total_fees == pytest.approx(0.999000999000999)
        assert result.total_price_concession == 0.0

    def test_spread_makes_the_fill_price_worse_than_the_open(self):
        bars = make_bars([(100.0, 100.0, 100.0, 100.0)] * 3)
        costs = CostModel(fee_rate=0.0, half_spread_bps=10.0, slippage_bps=0.0)

        result = run_backtest(
            bars, TargetWeightSchedule({0: 1.0}), initial_equity=1000.0, costs=costs, limits=OPEN
        )

        fill = result.fills[0]
        assert fill.reference_price == 100.0
        assert fill.fill_price == pytest.approx(100.1)
        assert result.total_price_concession == pytest.approx(fill.delta_units * 0.1)
        # Paying 100.1 for something worth 100 costs edge/(1+edge) of the
        # account, not edge: you buy fewer units, you do not lose cash.
        assert result.total_return == pytest.approx(-0.001 / 1.001, rel=1e-12)


class TestFillTiming:
    """A decision on bar i fills at the open of bar i+1. Never earlier."""

    def test_fill_lands_on_the_next_bar_open(self):
        # Gapped bars on purpose: in a continuous series open[i] == close[i-1],
        # which would make the assertion below pass for the wrong reason.
        bars = make_bars([(100.0 + 10 * i, 200.0, 50.0, 100.0 + 10 * i + 5) for i in range(10)])

        result = run_backtest(
            bars, TargetWeightSchedule({5: 1.0}), costs=FREE, limits=OPEN, symbol="S"
        )

        assert result.n_fills == 1
        fill = result.fills[0]
        assert fill.timestamp == bars["open_time"].iloc[6]
        assert fill.reference_price == pytest.approx(bars["open"].iloc[6])
        # Emphatically not the close of the bar the decision was made on.
        assert fill.reference_price != pytest.approx(bars["close"].iloc[5])

    def test_no_position_before_the_fill(self):
        bars = synth(50)
        result = run_backtest(
            bars, TargetWeightSchedule({5: 1.0}), costs=FREE, limits=OPEN, symbol="S"
        )

        assert (result.weight_curve.iloc[:6].abs() < 1e-12).all()
        assert result.weight_curve.iloc[6] == pytest.approx(1.0)

    def test_a_decision_on_the_last_bar_is_never_taken(self):
        bars = synth(20)
        last = len(bars) - 1

        result = run_backtest(
            bars, TargetWeightSchedule({last: 1.0}), costs=FREE, limits=OPEN, symbol="S"
        )

        assert result.n_fills == 0
        assert result.total_return == 0.0

    def test_one_bar_of_foresight_would_have_shown_up_as_profit(self):
        """The direct look-ahead detector.

        Prices alternate 100/110. A strategy that goes long after a green bar
        can only buy at the *next* open, which is always the top, so it loses
        on every cycle. If the engine filled at the signal bar's own price it
        would buy every bottom and this test would show a gain.
        """
        bars = continuous_bars([100.0, 110.0, 100.0, 110.0, 100.0, 110.0])

        class BuyAfterGreen:
            name = "buy-after-green"

            def on_bar(self, state: MarketState) -> Signal | None:
                green = state.bar.close > state.bar.open
                return Signal(state.symbol, 1.0 if green else 0.0, reason="green bar")

        result = run_backtest(bars, BuyAfterGreen(), initial_equity=1000.0, costs=FREE, limits=OPEN)

        # Two complete cycles, each buying at 110 and selling at 100.
        assert result.final_equity == pytest.approx(1000.0 * (100.0 / 110.0) ** 2)
        assert result.total_return == pytest.approx(-0.1735537190082645)
        # Meanwhile simply holding the asset gained 10%.
        assert result.benchmark_return == pytest.approx(0.10)
        assert not result.beat_benchmark


class TestNoLookAhead:
    """MarketState must make future data unreachable, not merely discouraged."""

    def test_history_is_cut_at_the_current_bar(self):
        bars = synth(200)
        seen: list[tuple[int, int, float]] = []

        class Recorder:
            name = "recorder"

            def on_bar(self, state: MarketState) -> Signal | None:
                seen.append((state.index, len(state.history), float(state.history.close[-1])))
                return None

        run_backtest(bars, Recorder(), costs=FREE, limits=OPEN, symbol="S")

        closes = bars["close"].to_numpy()
        for index, length, last_close in seen:
            assert length == index + 1
            assert last_close == pytest.approx(closes[index])

    def test_maximum_visible_price_never_includes_the_future(self):
        bars = synth(500)
        highs = bars["high"].to_numpy()
        breaches: list[int] = []

        class PeekChecker:
            name = "peek-checker"

            def on_bar(self, state: MarketState) -> Signal | None:
                if state.history.high.max() > highs[: state.index + 1].max() + 1e-12:
                    breaches.append(state.index)
                return None

        run_backtest(bars, PeekChecker(), costs=FREE, limits=OPEN, symbol="S")
        assert breaches == []

    def test_indexing_past_the_present_raises(self):
        bars = synth(30)
        errors: list[str] = []

        class Cheater:
            name = "cheater"

            def on_bar(self, state: MarketState) -> Signal | None:
                try:
                    _ = state.history.close[state.index + 1]
                except IndexError as exc:
                    errors.append(str(exc))
                return None

        run_backtest(bars, Cheater(), costs=FREE, limits=OPEN, symbol="S")
        assert len(errors) == len(bars) - 1

    def test_last_returns_fewer_values_early_in_the_series(self):
        bars = synth(30)
        lengths: list[tuple[int, int]] = []

        class WindowUser:
            name = "window-user"

            def on_bar(self, state: MarketState) -> Signal | None:
                lengths.append((state.index, len(state.history.last("close", 10))))
                return None

        run_backtest(bars, WindowUser(), costs=FREE, limits=OPEN, symbol="S")
        assert lengths[0] == (0, 1)
        assert lengths[4] == (4, 5)
        assert lengths[20] == (20, 10)


class TestPhase1Gate:
    """docs/PLAN.md Phase 1: the engine must be provably correct before use."""

    def test_zero_cost_buy_and_hold_reproduces_the_price_return_exactly(self):
        bars = synth(20_000)
        result = run_backtest(bars, BuyAndHold(), symbol="S", costs=FREE, limits=OPEN)

        # Entry is at bar 1's open, since the decision is made on bar 0.
        price_return = bars["close"].iloc[-1] / bars["open"].iloc[1] - 1.0

        assert result.total_return == pytest.approx(price_return, rel=1e-12)
        assert result.n_fills == 1
        assert result.total_costs == 0.0
        assert result.exposure == pytest.approx(1.0 - 1 / len(bars))

    def test_costed_buy_and_hold_differs_by_exactly_one_entry(self):
        """The discrepancy must be fully explained by the cost model."""
        bars = synth(20_000)
        costs = CostModel()

        free = run_backtest(bars, BuyAndHold(), symbol="S", costs=FREE, limits=OPEN)
        paid = run_backtest(bars, BuyAndHold(), symbol="S", costs=costs, limits=OPEN)

        # Entry buys units = cash / (open * (1 + edge) * (1 + fee)), so the two
        # final equities differ by precisely that factor.
        drag = (1.0 + costs.edge) * (1.0 + costs.fee_rate)
        assert paid.final_equity == pytest.approx(free.final_equity / drag, rel=1e-12)
        assert (1 + paid.total_return) * drag == pytest.approx(1 + free.total_return, rel=1e-12)

        # And that factor is about one side of the round trip, ~11bp.
        assert paid.total_return < free.total_return
        assert free.total_return - paid.total_return == pytest.approx(
            (1 + free.total_return) * (1 - 1 / drag), rel=1e-9
        )

    def test_always_flat_returns_exactly_zero_and_pays_nothing(self):
        bars = synth(5_000)
        result = run_backtest(bars, AlwaysFlat(), symbol="S", costs=CostModel(), limits=OPEN)

        assert result.total_return == 0.0
        assert result.final_equity == result.initial_equity
        assert result.n_fills == 0
        assert result.total_costs == 0.0
        assert result.exposure == 0.0
        assert result.max_drawdown == 0.0

    def test_random_strategy_pays_exactly_what_the_cost_model_says(self):
        """Per-fill identity: cost = notional x (edge + (1 + edge) x fee)."""
        bars = synth(5_000)
        costs = CostModel()

        result = run_backtest(
            bars,
            RandomStrategy(seed=3, every_n_bars=20),
            symbol="S",
            costs=costs,
            limits=OPEN,
        )

        assert result.n_fills > 20
        for fill in result.fills:
            reference_notional = abs(fill.delta_units) * fill.reference_price
            # A buy fills at ref x (1 + edge) and a sell at ref x (1 - edge), so
            # the fee leg differs in sign between the two.
            sign = 1.0 if fill.delta_units > 0 else -1.0
            expected = reference_notional * (
                costs.edge + (1.0 + sign * costs.edge) * costs.fee_rate
            )
            assert fill.total_cost == pytest.approx(expected, rel=1e-12)

        assert result.total_costs == pytest.approx(
            sum(f.fee + f.price_concession for f in result.fills)
        )
        assert result.total_costs > 0

    def test_random_strategy_loses_only_its_cost_bill(self):
        """With no edge and no drift, the cost bill should explain the damage."""
        bars = synth(20_000, seed=17)  # zero drift by construction
        costs = CostModel()
        gaps: list[float] = []

        for seed in range(12):
            strategy = RandomStrategy(seed=seed, every_n_bars=50)
            free = run_backtest(bars, strategy, symbol="S", costs=FREE, limits=OPEN)

            strategy = RandomStrategy(seed=seed, every_n_bars=50)
            paid = run_backtest(bars, strategy, symbol="S", costs=costs, limits=OPEN)

            # Costs are a multiplicative haircut on an identical weight path,
            # so they can only ever reduce the final equity.
            assert paid.final_equity < free.final_equity

            # The shortfall should be within a few percent of the cost bill,
            # not a multiple of it.
            shortfall = free.final_equity - paid.final_equity
            gaps.append(shortfall / paid.total_costs)

        assert 0.9 < float(np.mean(gaps)) < 1.15

    def test_trading_a_flat_market_is_exactly_free_when_costs_are_zero(self):
        """The accounting-leak detector.

        With the price pinned and no costs, any amount of trading must leave
        equity untouched to the last bit. If the engine created or destroyed
        money anywhere — double-counting a fill, marking at the wrong price —
        this is where it shows.
        """
        bars = make_bars([(100.0, 100.0, 100.0, 100.0)] * 500)

        result = run_backtest(
            bars,
            RandomStrategy(seed=2, every_n_bars=1),
            initial_equity=1000.0,
            costs=FREE,
            limits=OPEN,
        )

        assert result.n_fills > 100
        assert result.final_equity == pytest.approx(1000.0, abs=1e-9)
        assert result.total_return == pytest.approx(0.0, abs=1e-12)

    def test_trading_a_flat_market_costs_exactly_the_cost_bill(self):
        """Same setup with costs on: the loss must equal the fees and spread."""
        bars = make_bars([(100.0, 100.0, 100.0, 100.0)] * 500)

        result = run_backtest(
            bars,
            RandomStrategy(seed=2, every_n_bars=1),
            initial_equity=1000.0,
            costs=CostModel(),
            limits=OPEN,
        )

        assert result.n_fills > 100
        lost = result.initial_equity - result.final_equity
        assert lost == pytest.approx(result.total_costs, rel=1e-12)
        assert result.total_return == pytest.approx(-result.cost_drag, rel=1e-12)

    def test_random_strategy_earns_its_exposure_and_no_more(self):
        """Smoke test across price paths, with a deliberately wide band.

        A strategy long about half the time should capture about half the
        asset's move. The band is wide because the estimate is noisy: over
        ~7 days at 50% annual vol the asset moves ~7%, the strategy ~3.5%, and
        the median of 15 paths carries roughly 1% of standard error. Anything
        near or above 1.0 would mean the engine is handing out leverage.
        """
        ratios = []
        for data_seed in range(15):
            bars = synth(10_000, seed=100 + data_seed)
            result = run_backtest(
                bars,
                RandomStrategy(seed=7, every_n_bars=50),
                symbol="S",
                costs=FREE,
                limits=OPEN,
            )
            asset_return = bars["close"].iloc[-1] / bars["open"].iloc[1] - 1.0
            if abs(asset_return) > 0.01:
                ratios.append(result.total_return / asset_return)

        assert len(ratios) >= 8
        assert 0.15 < float(np.median(ratios)) < 0.85


class TestAccountingInvariants:
    def test_equity_reconciles_with_the_trade_ledger(self):
        bars = synth(5_000)
        result = run_backtest(
            bars,
            RandomStrategy(seed=5, every_n_bars=30),
            symbol="S",
            costs=CostModel(),
            limits=OPEN,
            liquidate_at_end=True,
        )

        realized = sum(t.pnl for t in result.trades)
        assert result.final_equity == pytest.approx(result.initial_equity + realized)

    def test_cash_never_goes_negative(self):
        """No leverage means the account cannot be overdrawn, fees included."""
        bars = synth(3_000)
        result = run_backtest(
            bars,
            RandomStrategy(seed=9, every_n_bars=5),
            symbol="S",
            costs=CostModel(fee_rate=0.01),  # deliberately punitive
            limits=OPEN,
        )

        # Reconstruct the cash path from the fills.
        cash = result.initial_equity
        for fill in result.fills:
            cash -= fill.cash_flow
            assert cash >= -1e-9, f"overdrawn at {fill.timestamp}"

    def test_weight_never_exceeds_the_cap(self):
        bars = synth(3_000)
        result = run_backtest(
            bars,
            RandomStrategy(seed=11, every_n_bars=10),
            symbol="S",
            costs=CostModel(),
            limits=RiskLimits(max_position_weight=0.25, min_order_notional=0.0),
        )

        # The cap binds when the order is sized, not continuously — a position
        # entered at 25% drifts as the price moves. The engine enforces the
        # former (unit-tested in test_risk.py); here we just confirm it is
        # wired in and that drift between rebalances stays modest.
        assert any("capped at 0.25" in a for a in result.risk_adjustments)
        assert result.weight_curve.max() < 0.30
        assert result.weight_curve.min() >= 0.0

    def test_identical_inputs_give_identical_results(self):
        bars = synth(2_000)
        runs = [
            run_backtest(
                bars,
                RandomStrategy(seed=4, every_n_bars=15),
                symbol="S",
                costs=CostModel(),
                limits=OPEN,
            )
            for _ in range(2)
        ]

        pd.testing.assert_series_equal(runs[0].equity_curve, runs[1].equity_curve)
        assert runs[0].total_costs == runs[1].total_costs
        assert runs[0].n_trades == runs[1].n_trades


class TestRiskIntegration:
    def test_default_limits_stop_buy_and_hold_from_going_all_in(self):
        """The backtest must measure the strategy as it would actually run."""
        bars = synth(5_000)
        result = run_backtest(bars, BuyAndHold(), symbol="S", costs=FREE)

        assert any("capped at 0.25" in a for a in result.risk_adjustments)
        # Sized at 25% on entry. Buy-and-hold never rebalances, so the weight
        # then drifts with the price rather than being pinned.
        assert result.weight_curve.iloc[1] == pytest.approx(0.25, abs=1e-3)
        assert result.weight_curve.max() < 0.30

    def test_drawdown_halt_flattens_and_stays_flat(self):
        # A series that falls 40%, well past the 15% limit.
        closes = list(np.linspace(100.0, 60.0, 400))
        bars = continuous_bars(closes)

        result = run_backtest(
            bars,
            BuyAndHold(),
            initial_equity=10_000.0,
            costs=FREE,
            limits=RiskLimits(max_position_weight=1.0, max_daily_loss=0.99, min_order_notional=0.0),
        )

        assert "max drawdown" in result.halt_reason
        assert result.weight_curve.iloc[-1] == pytest.approx(0.0)
        assert result.max_drawdown > -0.20  # halted rather than riding it to -40%

    def test_dust_rebalances_are_rejected_rather_than_bleeding_fees(self):
        bars = synth(500)
        # Alternate between two nearly identical weights so every rebalance is tiny.
        schedule = {i: (0.2 if i % 2 else 0.2001) for i in range(0, 400)}

        result = run_backtest(
            bars,
            TargetWeightSchedule(schedule),
            symbol="S",
            initial_equity=10_000.0,
            costs=CostModel(),
            limits=RiskLimits(max_position_weight=1.0, min_order_notional=20.0),
        )

        assert result.rejected_orders
        assert any("below minimum" in r for r in result.rejected_orders)
        assert result.n_fills < 10


class TestResultShape:
    def test_curves_are_aligned_and_complete(self):
        bars = synth(300)
        result = run_backtest(bars, BuyAndHold(), symbol="S", costs=FREE, limits=OPEN)

        assert len(result.equity_curve) == len(bars)
        assert len(result.weight_curve) == len(bars)
        assert len(result.benchmark_curve) == len(bars)
        pd.testing.assert_index_equal(result.equity_curve.index, result.weight_curve.index)
        assert result.equity_curve.index[0] == bars["close_time"].iloc[0]
        assert result.equity_curve.index[-1] == bars["close_time"].iloc[-1]

    def test_benchmark_is_cost_free_buy_and_hold_from_the_first_open(self):
        bars = synth(300)
        result = run_backtest(bars, AlwaysFlat(), symbol="S", costs=FREE, limits=OPEN)

        expected = bars["close"].iloc[-1] / bars["open"].iloc[0] - 1.0
        assert result.benchmark_return == pytest.approx(expected)

    def test_liquidation_closes_the_position_and_completes_the_trade(self):
        bars = synth(300)
        held = run_backtest(bars, BuyAndHold(), symbol="S", costs=FREE, limits=OPEN)
        closed = run_backtest(
            bars, BuyAndHold(), symbol="S", costs=FREE, limits=OPEN, liquidate_at_end=True
        )

        assert held.n_trades == 0  # never exited, so no completed round trip
        assert closed.n_trades == 1
        assert closed.weight_curve.iloc[-1] == pytest.approx(0.0)
        assert closed.n_fills == held.n_fills + 1

    def test_report_always_shows_the_benchmark_and_the_cost_bill(self):
        bars = synth(3_000)
        text = run_backtest(
            bars, RandomStrategy(seed=1, every_n_bars=30), symbol="S", limits=OPEN
        ).to_text()

        assert "benchmark" in text
        assert "excess" in text
        assert "total cost" in text
        assert "turnover" in text

    def test_at_least_two_bars_required(self):
        with pytest.raises(ValueError, match="at least 2 bars"):
            run_backtest(synth(1), BuyAndHold(), symbol="S")

    def test_nonpositive_starting_equity_rejected(self):
        with pytest.raises(ValueError, match="initial_equity must be positive"):
            run_backtest(synth(10), BuyAndHold(), symbol="S", initial_equity=0.0)

    def test_strategy_name_is_picked_up_for_the_report(self):
        result = run_backtest(synth(10), BuyAndHold(), symbol="S", costs=FREE, limits=OPEN)
        assert result.strategy_name == "buy-and-hold"
