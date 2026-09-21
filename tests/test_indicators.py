from __future__ import annotations

import numpy as np
import pytest

from ai_finance.strategy import indicators


class TestSMA:
    def test_known_value(self):
        assert indicators.sma(np.array([1.0, 2.0, 3.0, 4.0]), 2) == pytest.approx(3.5)

    def test_uses_only_the_most_recent_window(self):
        values = np.array([100.0, 100.0, 1.0, 1.0])
        assert indicators.sma(values, 2) == pytest.approx(1.0)

    def test_nan_when_history_is_short(self):
        assert np.isnan(indicators.sma(np.array([1.0, 2.0]), 5))

    def test_exact_window_length_is_enough(self):
        assert indicators.sma(np.array([1.0, 3.0]), 2) == pytest.approx(2.0)

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            indicators.sma(np.array([1.0]), 0)


class TestRSI:
    def test_unbroken_gains_peg_at_100(self):
        assert indicators.rsi(np.arange(1.0, 60.0), 14) == pytest.approx(100.0)

    def test_unbroken_losses_peg_at_0(self):
        assert indicators.rsi(np.arange(60.0, 1.0, -1.0), 14) == pytest.approx(0.0)

    def test_symmetric_moves_sit_near_the_middle(self):
        values = np.array([100.0 + (1.0 if i % 2 else -1.0) for i in range(100)])
        assert 40.0 < indicators.rsi(values, 14) < 60.0

    def test_nan_when_history_is_short(self):
        assert np.isnan(indicators.rsi(np.arange(10.0), 14))

    def test_flat_series_is_neutral_not_a_division_by_zero(self):
        assert indicators.rsi(np.full(50, 100.0), 14) == pytest.approx(50.0)

    def test_result_depends_on_seed_length_so_feed_it_history(self):
        """Wilder smoothing is recursive; the docstring warns about exactly this."""
        rising = np.concatenate([np.arange(1.0, 40.0), np.arange(40.0, 20.0, -1.0)])
        short = indicators.rsi(rising[-15:], 14)
        long = indicators.rsi(rising, 14)
        assert short != pytest.approx(long)


class TestRealizedVolatility:
    def test_annualises_by_the_square_root_of_periods(self):
        rng = np.random.default_rng(0)
        daily = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 5000)))

        vol = indicators.realized_volatility(daily, 4999, periods_per_year=365)

        # 1% daily moves annualise to roughly 19%.
        assert 0.17 < vol < 0.21

    def test_zero_volatility_for_a_flat_series(self):
        assert indicators.realized_volatility(np.full(100, 50.0), 50, 365) == pytest.approx(0.0)

    def test_nan_when_history_is_short(self):
        assert np.isnan(indicators.realized_volatility(np.arange(1.0, 10.0), 50, 365))

    def test_nan_for_nonpositive_prices(self):
        values = np.concatenate([np.full(50, 100.0), [0.0]])
        assert np.isnan(indicators.realized_volatility(values, 20, 365))

    def test_window_must_allow_a_return(self):
        with pytest.raises(ValueError, match="at least 2"):
            indicators.realized_volatility(np.arange(10.0), 1, 365)


class TestDonchian:
    def test_excludes_the_current_bar(self):
        """The whole point: 'did today break the range that came before it?'"""
        high = np.array([10.0, 11.0, 12.0, 99.0])
        low = np.array([5.0, 4.0, 3.0, 1.0])

        upper, lower = indicators.donchian(high, low, 3)

        assert upper == 12.0  # not 99
        assert lower == 3.0  # not 1

    def test_nan_when_history_is_short(self):
        upper, lower = indicators.donchian(np.arange(3.0), np.arange(3.0), 5)
        assert np.isnan(upper) and np.isnan(lower)

    def test_window_of_one_looks_at_the_previous_bar_only(self):
        upper, lower = indicators.donchian(np.array([1.0, 7.0, 2.0]), np.array([1.0, 6.0, 0.0]), 1)
        assert upper == 7.0
        assert lower == 6.0


class TestMomentum:
    def test_simple_return_over_the_window(self):
        assert indicators.momentum(np.array([100.0, 0.0, 0.0, 110.0]), 3) == pytest.approx(0.10)

    def test_negative_momentum(self):
        assert indicators.momentum(np.array([100.0, 90.0]), 1) == pytest.approx(-0.10)

    def test_nan_when_history_is_short(self):
        assert np.isnan(indicators.momentum(np.array([100.0]), 5))

    def test_nan_when_the_earlier_price_is_nonpositive(self):
        assert np.isnan(indicators.momentum(np.array([0.0, 100.0]), 1))
