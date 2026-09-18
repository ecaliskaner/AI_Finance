from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.data.quality import STALE_RUN_THRESHOLD, check_bars
from ai_finance.data.sources import SyntheticSource
from tests.conftest import bars_frame


class TestCleanData:
    def test_perfect_series_is_clean(self):
        report = check_bars(bars_frame("2024-01-01", 500), "BTCUSDT", "1m")

        assert report.is_clean()
        assert report.errors == []
        assert report.n_bars == 500
        assert report.expected_bars == 500
        assert report.missing_bars == 0
        assert report.coverage == 1.0

    def test_synthetic_source_output_is_clean(self):
        """The offline fixture must not itself trip the quality checks."""
        bars = SyntheticSource(epoch_ms=0, seed=4).fetch_chunk("S", "1m", 0, limit=20_000)
        report = check_bars(bars, "S", "1m")
        assert report.is_clean(), report.errors

    def test_empty_series_reports_nothing_rather_than_crashing(self):
        report = check_bars(bars_frame("2024-01-01", 0), "BTCUSDT", "1m")
        assert report.n_bars == 0
        assert report.is_clean()
        assert "No bars stored" in report.to_text()


class TestGaps:
    def test_detects_a_single_gap_and_counts_it(self):
        bars = pd.concat(
            [bars_frame("2024-01-01 00:00", 10), bars_frame("2024-01-01 00:15", 10)],
            ignore_index=True,
        )
        report = check_bars(bars, "BTCUSDT", "1m")

        assert not report.is_clean()
        assert report.missing_bars == 5
        assert len(report.gaps) == 1
        gap = report.gaps[0]
        assert gap.after == pd.Timestamp("2024-01-01 00:09", tz="UTC")
        assert gap.before == pd.Timestamp("2024-01-01 00:15", tz="UTC")
        assert gap.missing == 5
        assert "5 bars missing" in str(gap)

    def test_detects_multiple_gaps(self):
        bars = pd.concat(
            [
                bars_frame("2024-01-01 00:00", 5),
                bars_frame("2024-01-01 00:10", 5),
                bars_frame("2024-01-01 00:30", 5),
            ],
            ignore_index=True,
        )
        report = check_bars(bars, "BTCUSDT", "1m")

        assert len(report.gaps) == 2
        assert report.missing_bars == 5 + 15
        assert report.coverage < 1.0

    def test_finds_gaps_a_synthetic_outage_created(self):
        """End-to-end: inject outages, prove the report surfaces them."""
        source = SyntheticSource(epoch_ms=0, seed=9, gap_probability=0.05)
        bars = source.fetch_chunk("S", "1m", 0, limit=5000)

        report = check_bars(bars, "S", "1m")

        assert not report.is_clean()
        assert report.missing_bars > 0
        assert sum(g.missing for g in report.gaps) == report.missing_bars
        assert 0.9 < report.coverage < 1.0


class TestStructuralProblems:
    def test_detects_duplicate_timestamps(self):
        bars = bars_frame("2024-01-01", 10)
        doubled = pd.concat([bars, bars.iloc[[4]]], ignore_index=True)

        report = check_bars(doubled, "BTCUSDT", "1m")

        assert report.duplicate_timestamps == 1
        assert not report.is_clean()

    def test_detects_out_of_order_timestamps(self):
        bars = bars_frame("2024-01-01", 10).iloc[[0, 1, 5, 2, 3, 4, 6, 7, 8, 9]]
        report = check_bars(bars.reset_index(drop=True), "BTCUSDT", "1m")

        assert report.out_of_order > 0
        assert not report.is_clean()
        # Deduplicated, the underlying series is still complete.
        assert report.missing_bars == 0

    def test_detects_high_below_low(self):
        bars = bars_frame("2024-01-01", 10)
        bars.loc[3, "high"] = bars.loc[3, "low"] - 1.0

        report = check_bars(bars, "BTCUSDT", "1m")

        assert report.ohlc_violations >= 1
        assert not report.is_clean()

    def test_detects_close_outside_the_high_low_range(self):
        bars = bars_frame("2024-01-01", 10)
        bars.loc[5, "close"] = bars.loc[5, "high"] * 10

        report = check_bars(bars, "BTCUSDT", "1m")
        assert report.ohlc_violations == 1

    def test_detects_nonpositive_prices(self):
        bars = bars_frame("2024-01-01", 10)
        bars.loc[2, "low"] = 0.0

        report = check_bars(bars, "BTCUSDT", "1m")

        assert report.nonpositive_prices == 1
        assert not report.is_clean()

    def test_detects_wrong_close_time(self):
        bars = bars_frame("2024-01-01", 10)
        bars.loc[1, "close_time"] = bars.loc[1, "open_time"]

        report = check_bars(bars, "BTCUSDT", "1m")

        assert report.close_time_mismatches == 1
        assert not report.is_clean()

    def test_float_noise_does_not_trip_ohlc_checks(self):
        bars = bars_frame("2024-01-01", 200)
        bars["high"] = bars[["open", "close"]].max(axis=1) * (1 + 1e-16)

        assert check_bars(bars, "BTCUSDT", "1m").ohlc_violations == 0


class TestWarnings:
    def test_zero_volume_is_a_warning_not_an_error(self):
        bars = bars_frame("2024-01-01", 10)
        bars.loc[4, "volume"] = 0.0

        report = check_bars(bars, "BTCUSDT", "1m")

        assert report.zero_volume_bars == 1
        assert report.is_clean(), "zero volume is legitimate in a quiet market"
        assert any("zero-volume" in w for w in report.warnings)

    def test_detects_a_frozen_feed(self):
        bars = bars_frame("2024-01-01", 60)
        frozen = bars.loc[10, ["open", "high", "low", "close"]]
        for i in range(10, 10 + STALE_RUN_THRESHOLD + 5):
            bars.loc[i, ["open", "high", "low", "close"]] = frozen

        report = check_bars(bars, "BTCUSDT", "1m")

        assert len(report.stale_runs) == 1
        start, length = report.stale_runs[0]
        assert start == pd.Timestamp("2024-01-01 00:10", tz="UTC")
        assert length >= STALE_RUN_THRESHOLD
        assert any("stale run" in w for w in report.warnings)

    def test_short_flat_stretch_is_not_flagged(self):
        bars = bars_frame("2024-01-01", 60)
        frozen = bars.loc[10, ["open", "high", "low", "close"]]
        for i in range(10, 13):
            bars.loc[i, ["open", "high", "low", "close"]] = frozen

        assert check_bars(bars, "BTCUSDT", "1m").stale_runs == []

    def test_detects_a_bad_print(self):
        bars = SyntheticSource(epoch_ms=0, seed=6).fetch_chunk("S", "1m", 0, limit=3000)
        bars.loc[1500, ["open", "high", "low", "close"]] *= 10.0

        report = check_bars(bars, "S", "1m")

        assert report.extreme_returns, "a 10x spike must be flagged"
        flagged_times = {ts for ts, _ in report.extreme_returns}
        assert bars["open_time"].iloc[1500] in flagged_times
        assert any("extreme return" in w for w in report.warnings)

    def test_normal_volatility_is_not_flagged_as_extreme(self):
        bars = SyntheticSource(epoch_ms=0, seed=8).fetch_chunk("S", "1m", 0, limit=20_000)
        report = check_bars(bars, "S", "1m")
        # A few genuine outliers are fine; a flood means the threshold is wrong.
        assert len(report.extreme_returns) < 0.001 * len(bars)

    def test_extreme_returns_need_enough_history_to_judge(self):
        assert check_bars(bars_frame("2024-01-01", 50), "S", "1m").extreme_returns == []


class TestOtherIntervals:
    def test_hourly_series_uses_the_hourly_grid(self):
        bars = bars_frame("2024-01-01", 24, interval="1h")
        report = check_bars(bars, "BTCUSDT", "1h")

        assert report.is_clean()
        assert report.expected_bars == 24

    def test_hourly_gap_detected_in_hours_not_minutes(self):
        bars = pd.concat(
            [
                bars_frame("2024-01-01 00:00", 3, interval="1h"),
                bars_frame("2024-01-01 05:00", 3, interval="1h"),
            ],
            ignore_index=True,
        )
        report = check_bars(bars, "BTCUSDT", "1h")
        assert report.missing_bars == 2

    def test_unknown_interval_rejected(self):
        with pytest.raises(ValueError, match="unknown interval"):
            check_bars(bars_frame("2024-01-01", 5), "BTCUSDT", "7m")


class TestReportText:
    def test_clean_report_says_so(self):
        text = check_bars(bars_frame("2024-01-01", 200), "BTCUSDT", "1m").to_text()
        assert "ERRORS     none" in text
        assert "coverage   100.0000%" in text

    def test_dirty_report_lists_errors_and_biggest_gaps(self):
        bars = pd.concat(
            [bars_frame("2024-01-01 00:00", 5), bars_frame("2024-01-01 02:00", 5)],
            ignore_index=True,
        )
        text = check_bars(bars, "BTCUSDT", "1m").to_text()

        assert "ERRORS" in text
        assert "missing bars" in text
        assert "Largest gaps" in text

    def test_coverage_of_empty_expectation_is_one(self):
        report = check_bars(bars_frame("2024-01-01", 1), "BTCUSDT", "1m")
        assert report.expected_bars == 1
        assert report.coverage == 1.0
        assert np.isfinite(report.coverage)
