from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.data.sources import SyntheticSource
from ai_finance.features import technical
from ai_finance.features.pipeline import (
    PointInTimeError,
    assert_point_in_time,
    build_dataset,
    directional_accuracy,
    information_coefficient,
)
from ai_finance.features.technical import build_features, forward_return
from tests.test_engine import continuous_bars

PPY_4H = 365 * 24 / 4


def bars_4h(n=800, seed=1):
    return SyntheticSource(epoch_ms=0, seed=seed).fetch_chunk("S", "4h", 0, limit=n)


class TestPointInTime:
    """The guarantee that makes precomputing a feature matrix safe."""

    def test_no_feature_changes_when_future_bars_are_removed(self):
        assert_point_in_time(bars_4h(800), PPY_4H)

    def test_holds_on_a_flat_series_too(self):
        assert_point_in_time(continuous_bars([100.0] * 400), PPY_4H)

    def test_every_column_is_checked(self):
        bars = bars_4h(400)
        full = build_features(bars, PPY_4H)
        truncated = build_features(bars.iloc[:200], PPY_4H)

        for column in full.columns:
            left = full.loc[truncated.index, column].to_numpy()
            right = truncated[column].to_numpy()
            assert np.allclose(left, right, equal_nan=True), column

    def test_the_detector_actually_catches_a_leak(self, monkeypatch):
        """Prove the check works, not merely that the features pass it."""

        real = technical.build_features

        def leaky(bars, periods_per_year):
            frame = real(bars, periods_per_year)
            # A centred rolling mean sees the future by half its window. This
            # is the classic mistake the whole module is written to avoid.
            close = pd.Series(
                bars["close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(bars["close_time"]),
            )
            frame["oops"] = close.rolling(11, center=True).mean()
            return frame

        monkeypatch.setattr("ai_finance.features.pipeline.build_features", leaky)

        with pytest.raises(PointInTimeError, match="'oops' changed"):
            assert_point_in_time(bars_4h(400), PPY_4H)

    def test_error_names_the_column_and_the_timestamp(self, monkeypatch):
        real = technical.build_features

        def leaky(bars, periods_per_year):
            frame = real(bars, periods_per_year)
            close = pd.Series(
                bars["close"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(bars["close_time"]),
            )
            frame["tomorrow"] = close.shift(-1)
            return frame

        monkeypatch.setattr("ai_finance.features.pipeline.build_features", leaky)

        with pytest.raises(PointInTimeError) as excinfo:
            assert_point_in_time(bars_4h(400), PPY_4H)
        assert "tomorrow" in str(excinfo.value)
        assert "depends on data from the future" in str(excinfo.value)


class TestFeatureValues:
    def test_expected_columns(self):
        frame = build_features(bars_4h(400), PPY_4H)
        assert list(frame.columns) == technical.feature_names()

    def test_indexed_by_close_time_not_open_time(self):
        """A row is knowable when its bar closes, not when it opens."""
        bars = bars_4h(100)
        frame = build_features(bars, PPY_4H)
        assert frame.index[0] == bars["close_time"].iloc[0]

    def test_returns_are_log_returns_over_the_stated_lookback(self):
        bars = continuous_bars([100.0, 110.0, 121.0, 133.1])
        frame = build_features(bars, PPY_4H)

        assert frame["ret_1"].iloc[1] == pytest.approx(np.log(110.0 / 100.0))
        assert frame["ret_1"].iloc[3] == pytest.approx(np.log(133.1 / 121.0))

    def test_flat_series_has_zero_volatility_and_neutral_trend(self):
        frame = build_features(continuous_bars([100.0] * 300), PPY_4H)
        assert frame["vol_12"].iloc[-1] == pytest.approx(0.0)
        assert frame["ma_ratio"].iloc[-1] == pytest.approx(0.0)

    def test_rsi_stays_in_range(self):
        frame = build_features(bars_4h(600), PPY_4H)
        rsi = frame["rsi"].dropna()
        assert rsi.min() >= 0.0
        assert rsi.max() <= 100.0

    def test_rsi_pegs_high_on_an_unbroken_climb(self):
        frame = build_features(continuous_bars(list(np.arange(100.0, 300.0))), PPY_4H)
        assert frame["rsi"].iloc[-1] > 95.0

    def test_distance_from_the_channel_high_is_never_positive(self):
        frame = build_features(bars_4h(500), PPY_4H)
        assert frame["dist_from_high"].dropna().max() <= 1e-12

    def test_distance_from_the_channel_low_is_never_negative(self):
        frame = build_features(bars_4h(500), PPY_4H)
        assert frame["dist_from_low"].dropna().min() >= -1e-12

    def test_time_features_are_cyclical(self):
        """23:00 and 00:00 must be neighbours, not opposites."""
        frame = build_features(bars_4h(200), PPY_4H)
        assert np.allclose(frame["hour_sin"] ** 2 + frame["hour_cos"] ** 2, 1.0)
        assert np.allclose(frame["dow_sin"] ** 2 + frame["dow_cos"] ** 2, 1.0)

    def test_warmup_rows_are_nan_and_later_rows_are_not(self):
        frame = build_features(bars_4h(400), PPY_4H)
        warmup = technical.warmup_rows()

        # Every row before the warm-up boundary is missing at least one value,
        # and every row from it onward is complete. Both halves matter: an
        # over-estimate silently throws away usable data.
        assert frame.iloc[:warmup].isna().any(axis=1).all()
        assert not frame.iloc[warmup:].isna().any(axis=1).any()
        assert warmup == technical.MA_SLOW - 1

    def test_no_infinities_survive(self):
        frame = build_features(continuous_bars([100.0] * 300), PPY_4H)
        assert not np.isinf(frame.to_numpy(dtype=float)).any()


class TestForwardReturn:
    def test_looks_forward_by_the_horizon(self):
        bars = continuous_bars([100.0, 110.0, 121.0, 133.1])
        labels = forward_return(bars, horizon=1)

        assert labels.iloc[0] == pytest.approx(0.10)
        assert labels.iloc[1] == pytest.approx(0.10)

    def test_last_rows_have_no_label(self):
        labels = forward_return(continuous_bars([100.0] * 10), horizon=3)
        assert labels.iloc[-3:].isna().all()
        assert labels.iloc[:-3].notna().all()

    def test_is_not_among_the_features(self):
        """The label must never leak back in as an input."""
        frame = build_features(bars_4h(200), PPY_4H)
        assert "forward_return" not in frame.columns
        assert not any("forward" in name for name in frame.columns)

    def test_horizon_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            forward_return(continuous_bars([100.0] * 5), horizon=0)


class TestDataset:
    def test_drops_warmup_and_unlabelled_rows(self):
        bars = bars_4h(500)
        dataset = build_dataset(bars, horizon=6, periods_per_year=PPY_4H)

        assert len(dataset) < len(bars)
        assert dataset.features.notna().all().all()
        assert dataset.labels.notna().all()

    def test_features_and_labels_stay_aligned(self):
        dataset = build_dataset(bars_4h(500), horizon=6, periods_per_year=PPY_4H)
        assert dataset.features.index.equals(dataset.labels.index)

    def test_no_label_reaches_past_the_data(self):
        bars = bars_4h(500)
        dataset = build_dataset(bars, horizon=6, periods_per_year=PPY_4H)
        last_usable = bars["close_time"].iloc[-7]
        assert dataset.index[-1] <= last_usable

    def test_slice_returns_arrays(self):
        dataset = build_dataset(bars_4h(500), horizon=6, periods_per_year=PPY_4H)
        x, y = dataset.slice(np.array([0, 1, 2]))

        assert x.shape == (3, len(dataset.feature_names))
        assert y.shape == (3,)

    def test_mismatched_index_rejected(self):
        dataset = build_dataset(bars_4h(300), horizon=6, periods_per_year=PPY_4H)
        with pytest.raises(ValueError, match="share an index"):
            type(dataset)(
                features=dataset.features,
                labels=dataset.labels.iloc[:-1],
                horizon=6,
                periods_per_year=PPY_4H,
            )


class TestScoringHelpers:
    def test_perfect_direction_scores_one(self):
        assert directional_accuracy(np.array([1.0, -1.0]), np.array([0.5, -0.5])) == 1.0

    def test_inverted_direction_scores_zero(self):
        assert directional_accuracy(np.array([1.0, -1.0]), np.array([-0.5, 0.5])) == 0.0

    def test_flat_outcomes_are_excluded(self):
        """There is no direction to get right when the return was exactly zero."""
        accuracy = directional_accuracy(np.array([1.0, 1.0, 1.0]), np.array([0.5, 0.0, 0.5]))
        assert accuracy == 1.0

    def test_nan_when_nothing_is_scorable(self):
        assert np.isnan(directional_accuracy(np.array([1.0]), np.array([0.0])))

    def test_information_coefficient_of_a_perfect_forecast(self):
        actual = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        assert information_coefficient(actual, actual) == pytest.approx(1.0)

    def test_information_coefficient_of_an_inverted_forecast(self):
        actual = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        assert information_coefficient(-actual, actual) == pytest.approx(-1.0)

    def test_information_coefficient_nan_without_variation(self):
        assert np.isnan(information_coefficient(np.ones(10), np.arange(10.0)))

    def test_information_coefficient_needs_a_few_points(self):
        assert np.isnan(information_coefficient(np.array([1.0]), np.array([1.0])))
