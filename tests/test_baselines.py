from __future__ import annotations

import numpy as np
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.base import Bar, BarWindow, MarketState
from ai_finance.strategy.baselines import (
    PARAM_GRIDS,
    STRATEGY_FACTORIES,
    Breakout,
    MovingAverageCrossover,
    RSIMeanReversion,
    VolatilityScaledTrend,
    warmup_bars_for,
)
from tests.test_engine import continuous_bars, synth

FREE = CostModel.free()
OPEN = RiskLimits.unconstrained()


def state_from(closes, *, highs=None, lows=None, index=None, bar_seconds=14400.0):
    """A MarketState positioned at the last of ``closes``."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    arrays = {
        "open": closes,
        "high": np.asarray(highs, dtype=float) if highs is not None else closes,
        "low": np.asarray(lows, dtype=float) if lows is not None else closes,
        "close": closes,
        "volume": np.ones(n),
    }
    i = n - 1 if index is None else index
    import pandas as pd

    return MarketState(
        symbol="X",
        index=i,
        bar=Bar(
            open_time=pd.Timestamp("2024-01-01", tz="UTC"),
            close_time=pd.Timestamp("2024-01-01 03:59:59.999", tz="UTC"),
            open=float(arrays["open"][i]),
            high=float(arrays["high"][i]),
            low=float(arrays["low"][i]),
            close=float(closes[i]),
            volume=1.0,
            trades=1,
        ),
        history=BarWindow(arrays, i),
        equity=10_000.0,
        position_weight=0.0,
        bar_seconds=bar_seconds,
    )


class TestMovingAverageCrossover:
    def test_goes_long_when_fast_is_above_slow(self):
        rising = np.arange(1.0, 101.0)
        signal = MovingAverageCrossover(fast=5, slow=20).on_bar(state_from(rising))
        assert signal is not None
        assert signal.target_weight == 1.0

    def test_goes_flat_when_fast_falls_below_slow(self):
        falling = np.arange(100.0, 0.0, -1.0)
        signal = MovingAverageCrossover(fast=5, slow=20).on_bar(state_from(falling))
        assert signal is not None
        assert signal.target_weight == 0.0

    def test_silent_until_there_is_enough_history(self):
        assert MovingAverageCrossover(fast=5, slow=20).on_bar(state_from(np.arange(10.0))) is None

    def test_emits_only_when_the_target_changes(self):
        strategy = MovingAverageCrossover(fast=5, slow=20)
        rising = np.arange(1.0, 101.0)

        first = strategy.on_bar(state_from(rising))
        second = strategy.on_bar(state_from(rising + 1.0))

        assert first is not None
        assert second is None, "restating the same target would trigger a needless rebalance"

    def test_fast_must_be_shorter_than_slow(self):
        with pytest.raises(ValueError, match="must be shorter"):
            MovingAverageCrossover(fast=50, slow=20)

    def test_reason_names_both_averages(self):
        signal = MovingAverageCrossover(fast=5, slow=20).on_bar(state_from(np.arange(1.0, 101.0)))
        assert "MA5" in signal.reason and "MA20" in signal.reason


class TestRSIMeanReversion:
    def test_buys_when_oversold(self):
        falling = np.arange(200.0, 100.0, -1.0)
        signal = RSIMeanReversion(period=14, oversold=30.0).on_bar(state_from(falling))
        assert signal is not None
        assert signal.target_weight == 1.0

    def test_sells_back_into_strength(self):
        strategy = RSIMeanReversion(period=14, oversold=30.0, exit_level=50.0)
        strategy.on_bar(state_from(np.arange(200.0, 100.0, -1.0)))

        signal = strategy.on_bar(state_from(np.arange(1.0, 200.0)))

        assert signal is not None
        assert signal.target_weight == 0.0

    def test_holds_through_the_middle_without_signalling(self):
        """Between the two thresholds the strategy has no opinion."""
        values = np.array([100.0 + (1.0 if i % 2 else -1.0) for i in range(100)])
        strategy = RSIMeanReversion(period=14, oversold=30.0, exit_level=70.0)

        # This series sits around RSI 52: not oversold, not strong enough to exit.
        assert strategy.on_bar(state_from(values)) is None

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match="oversold < exit_level"):
            RSIMeanReversion(oversold=60.0, exit_level=40.0)


class TestBreakout:
    def test_goes_long_on_a_new_high(self):
        highs = np.concatenate([np.full(30, 10.0), [20.0]])
        lows = np.full(31, 5.0)
        closes = np.concatenate([np.full(30, 8.0), [20.0]])

        signal = Breakout(lookback=20, exit_lookback=10).on_bar(
            state_from(closes, highs=highs, lows=lows)
        )

        assert signal is not None
        assert signal.target_weight == 1.0

    def test_goes_flat_on_a_new_low(self):
        strategy = Breakout(lookback=20, exit_lookback=10)
        highs = np.concatenate([np.full(30, 10.0), [20.0]])
        lows = np.full(31, 5.0)
        strategy.on_bar(
            state_from(np.concatenate([np.full(30, 8.0), [20.0]]), highs=highs, lows=lows)
        )

        signal = strategy.on_bar(
            state_from(np.concatenate([np.full(30, 8.0), [1.0]]), highs=highs, lows=lows)
        )

        assert signal is not None
        assert signal.target_weight == 0.0

    def test_lookbacks_must_be_sane(self):
        with pytest.raises(ValueError, match="at least 2"):
            Breakout(lookback=1)


class TestVolatilityScaledTrend:
    def test_flat_when_the_trend_is_down(self):
        signal = VolatilityScaledTrend(lookback=20, vol_window=20).on_bar(
            state_from(np.arange(200.0, 100.0, -1.0))
        )
        assert signal is not None
        assert signal.target_weight == 0.0

    def test_size_is_target_vol_over_realised_vol(self):
        rng = np.random.default_rng(1)
        # Upward drift with controlled noise, on 4-hour bars.
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.004, 300)))

        signal = VolatilityScaledTrend(
            lookback=50, vol_window=50, target_vol=0.20, max_weight=1.0
        ).on_bar(state_from(closes))

        assert signal is not None
        assert 0.0 < signal.target_weight <= 1.0

    def test_weight_is_capped(self):
        closes = 100.0 * (1.0 + np.arange(300) * 1e-6)  # trending, almost no volatility
        signal = VolatilityScaledTrend(lookback=50, vol_window=50, max_weight=0.5).on_bar(
            state_from(closes)
        )
        assert signal.target_weight == pytest.approx(0.5)

    def test_no_trade_band_suppresses_tiny_adjustments(self):
        """Without this the target drifts every bar and the fees eat it alive."""
        rng = np.random.default_rng(2)
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.004, 300)))
        strategy = VolatilityScaledTrend(lookback=50, vol_window=50, rebalance_threshold=0.5)

        assert strategy.on_bar(state_from(closes)) is not None
        assert strategy.on_bar(state_from(closes * 1.0001)) is None

    def test_target_vol_must_be_positive(self):
        with pytest.raises(ValueError, match="target_vol must be positive"):
            VolatilityScaledTrend(target_vol=0.0)


class TestRegistryOfStrategies:
    def test_every_grid_has_a_factory_and_a_warmup(self):
        assert set(PARAM_GRIDS) == set(STRATEGY_FACTORIES)
        for name, grid in PARAM_GRIDS.items():
            params = {key: values[0] for key, values in grid.items()}
            assert warmup_bars_for(name, params) > 0

    def test_warmup_rejects_an_unknown_strategy(self):
        with pytest.raises(ValueError, match="unknown strategy"):
            warmup_bars_for("nope", {})

    @pytest.mark.parametrize("name", sorted(STRATEGY_FACTORIES))
    def test_each_baseline_runs_end_to_end(self, name):
        bars = synth(3_000)
        result = run_backtest(bars, STRATEGY_FACTORIES[name](), symbol="S", costs=FREE, limits=OPEN)
        assert len(result.equity_curve) == len(bars)
        assert np.isfinite(result.final_equity)

    def test_baselines_stay_flat_on_a_pinned_price(self):
        """No movement means no trend, no breakout and no oversold reading."""
        bars = continuous_bars([100.0] * 600)
        for name in STRATEGY_FACTORIES:
            result = run_backtest(
                bars, STRATEGY_FACTORIES[name](), symbol="S", costs=CostModel(), limits=OPEN
            )
            assert result.total_costs < 1.0, f"{name} churned on a flat market"
