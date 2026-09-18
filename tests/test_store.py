from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.data.schema import assert_schema
from ai_finance.data.store import (
    last_bar_time,
    load_bars,
    partition_path,
    resample_bars,
    store_summary,
    write_bars,
)
from tests.conftest import bars_frame


class TestWriteAndLoad:
    def test_round_trip(self, store_root):
        bars = bars_frame("2024-01-01", 100)
        write_bars(bars, "BTCUSDT", root=store_root)

        loaded = load_bars("BTCUSDT", root=store_root)

        assert_schema(loaded)
        pd.testing.assert_frame_equal(loaded, bars)

    def test_symbol_case_is_normalised(self, store_root):
        write_bars(bars_frame("2024-01-01", 10), "btcusdt", root=store_root)
        assert len(load_bars("BTCUSDT", root=store_root)) == 10
        assert len(load_bars("BtcUsdt", root=store_root)) == 10

    def test_splits_across_monthly_partitions(self, store_root):
        # Straddles the January/February boundary.
        bars = bars_frame("2024-01-31 23:50", 30)
        written = write_bars(bars, "BTCUSDT", root=store_root)

        assert [p.name for p in written] == ["2024-01.parquet", "2024-02.parquet"]
        assert partition_path("BTCUSDT", "1m", "2024-01", store_root).exists()
        assert len(load_bars("BTCUSDT", root=store_root)) == 30

    def test_writing_twice_is_idempotent(self, store_root):
        bars = bars_frame("2024-01-01", 50)
        write_bars(bars, "BTCUSDT", root=store_root)
        write_bars(bars, "BTCUSDT", root=store_root)

        pd.testing.assert_frame_equal(load_bars("BTCUSDT", root=store_root), bars)

    def test_overlapping_write_keeps_the_newer_version(self, store_root):
        """What makes a re-run after a partial failure safe."""
        original = bars_frame("2024-01-01", 10)
        write_bars(original, "BTCUSDT", root=store_root)

        corrected = original.iloc[3:7].assign(close=12345.0)
        write_bars(corrected, "BTCUSDT", root=store_root)

        loaded = load_bars("BTCUSDT", root=store_root)
        assert len(loaded) == 10
        assert (loaded["close"].iloc[3:7] == 12345.0).all()
        assert loaded["close"].iloc[0] == original["close"].iloc[0]

    def test_appending_later_bars_extends_the_series(self, store_root):
        write_bars(bars_frame("2024-01-01", 10), "BTCUSDT", root=store_root)
        write_bars(bars_frame("2024-01-01 00:10", 10), "BTCUSDT", root=store_root)

        loaded = load_bars("BTCUSDT", root=store_root)
        assert len(loaded) == 20
        assert loaded["open_time"].is_monotonic_increasing

    def test_writing_empty_frame_is_a_no_op(self, store_root):
        assert write_bars(bars_frame("2024-01-01", 0), "BTCUSDT", root=store_root) == []

    def test_load_from_empty_store_returns_canonical_empty(self, store_root):
        bars = load_bars("NOTHING", root=store_root)
        assert bars.empty
        assert_schema(bars)

    def test_range_filter_is_inclusive(self, store_root):
        write_bars(bars_frame("2024-01-01", 60), "BTCUSDT", root=store_root)

        loaded = load_bars(
            "BTCUSDT", start="2024-01-01 00:10", end="2024-01-01 00:19", root=store_root
        )

        assert len(loaded) == 10
        assert loaded["open_time"].iloc[0] == pd.Timestamp("2024-01-01 00:10", tz="UTC")
        assert loaded["open_time"].iloc[-1] == pd.Timestamp("2024-01-01 00:19", tz="UTC")

    def test_naive_range_bounds_are_treated_as_utc(self, store_root):
        write_bars(bars_frame("2024-01-01", 60), "BTCUSDT", root=store_root)
        naive = load_bars("BTCUSDT", start="2024-01-01 00:30", root=store_root)
        aware = load_bars(
            "BTCUSDT", start=pd.Timestamp("2024-01-01 00:30", tz="UTC"), root=store_root
        )
        pd.testing.assert_frame_equal(naive, aware)

    def test_month_pruning_does_not_lose_data_at_boundaries(self, store_root):
        write_bars(bars_frame("2024-01-31 23:55", 20), "BTCUSDT", root=store_root)

        loaded = load_bars(
            "BTCUSDT", start="2024-01-31 23:58", end="2024-02-01 00:02", root=store_root
        )

        assert len(loaded) == 5
        assert loaded["open_time"].iloc[0] == pd.Timestamp("2024-01-31 23:58", tz="UTC")
        assert loaded["open_time"].iloc[-1] == pd.Timestamp("2024-02-01 00:02", tz="UTC")


class TestLastBarTime:
    def test_none_when_empty(self, store_root):
        assert last_bar_time("BTCUSDT", root=store_root) is None

    def test_returns_newest_open_time(self, store_root):
        write_bars(bars_frame("2024-01-01", 5), "BTCUSDT", root=store_root)
        assert last_bar_time("BTCUSDT", root=store_root) == pd.Timestamp(
            "2024-01-01 00:04", tz="UTC"
        )

    def test_looks_at_the_newest_partition(self, store_root):
        write_bars(bars_frame("2024-01-31 23:59", 3), "BTCUSDT", root=store_root)
        assert last_bar_time("BTCUSDT", root=store_root) == pd.Timestamp(
            "2024-02-01 00:01", tz="UTC"
        )


class TestResample:
    def test_aggregates_each_field_correctly(self):
        bars = bars_frame("2024-01-01", 5)
        out = resample_bars(bars, "5m")

        assert len(out) == 1
        row = out.iloc[0]
        assert row["open_time"] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
        assert row["open"] == bars["open"].iloc[0]
        assert row["high"] == bars["high"].max()
        assert row["low"] == bars["low"].min()
        assert row["close"] == bars["close"].iloc[-1]
        assert row["volume"] == pytest.approx(bars["volume"].sum())
        assert row["trades"] == bars["trades"].sum()
        assert row["close_time"] == pd.Timestamp("2024-01-01 00:04:59.999", tz="UTC")

    def test_buckets_are_anchored_to_the_epoch_not_to_the_first_bar(self):
        """4h candles must land on 00:00/04:00/08:00 UTC, as the exchange labels them."""
        # 02:00 .. 11:59, so the 00:00 and 12:00 buckets are only partly covered.
        bars = bars_frame("2024-01-01 02:00", 10 * 60)
        out = resample_bars(bars, "4h")

        assert list(out["open_time"]) == [
            pd.Timestamp("2024-01-01 04:00", tz="UTC"),
            pd.Timestamp("2024-01-01 08:00", tz="UTC"),
        ]

    def test_drops_incomplete_periods_at_both_ends(self):
        """A 4h bar built from ten minutes of data is not a 4h bar."""
        bars = bars_frame("2024-01-01 03:50", 4 * 60 + 20)  # 03:50 .. 08:09
        out = resample_bars(bars, "4h")

        assert len(out) == 1
        assert out["open_time"].iloc[0] == pd.Timestamp("2024-01-01 04:00", tz="UTC")

    def test_does_not_invent_bars_across_a_gap(self):
        early = bars_frame("2024-01-01 00:00", 60)
        late = bars_frame("2024-01-01 10:00", 60)
        bars = pd.concat([early, late], ignore_index=True)

        out = resample_bars(bars, "1h")

        assert list(out["open_time"]) == [
            pd.Timestamp("2024-01-01 00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 10:00", tz="UTC"),
        ]

    def test_empty_input_gives_canonical_empty(self):
        out = resample_bars(bars_frame("2024-01-01", 0), "4h")
        assert out.empty
        assert_schema(out)

    def test_unknown_interval_rejected(self):
        with pytest.raises(ValueError, match="unknown interval"):
            resample_bars(bars_frame("2024-01-01", 5), "7m")

    def test_load_bars_resamples_on_read(self, store_root):
        write_bars(bars_frame("2024-01-01", 240), "BTCUSDT", root=store_root)

        hourly = load_bars("BTCUSDT", interval="1h", root=store_root)

        assert len(hourly) == 4
        assert_schema(hourly)
        assert hourly["open_time"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")

    def test_load_bars_widens_range_to_whole_periods(self, store_root):
        """Asking for 00:30..03:30 of 1h bars must not build them from partial slices."""
        write_bars(bars_frame("2024-01-01", 300), "BTCUSDT", root=store_root)

        hourly = load_bars(
            "BTCUSDT",
            start="2024-01-01 00:30",
            end="2024-01-01 03:30",
            interval="1h",
            root=store_root,
        )

        # 00:00 is excluded by the final clip; 01:00..03:00 survive intact.
        assert list(hourly["open_time"]) == [
            pd.Timestamp("2024-01-01 01:00", tz="UTC"),
            pd.Timestamp("2024-01-01 02:00", tz="UTC"),
            pd.Timestamp("2024-01-01 03:00", tz="UTC"),
        ]
        assert hourly["volume"].iloc[0] == pytest.approx(
            load_bars("BTCUSDT", "2024-01-01 01:00", "2024-01-01 01:59", root=store_root)[
                "volume"
            ].sum()
        )


class TestStoreSummary:
    def test_empty_store(self, store_root):
        assert store_summary(store_root).empty

    def test_reports_each_symbol(self, store_root):
        write_bars(bars_frame("2024-01-01", 10), "BTCUSDT", root=store_root)
        write_bars(bars_frame("2024-01-01", 20), "ETHUSDT", root=store_root)

        summary = store_summary(store_root).set_index("symbol")

        assert summary.loc["BTCUSDT", "bars"] == 10
        assert summary.loc["ETHUSDT", "bars"] == 20
        assert summary.loc["BTCUSDT", "first"] == pd.Timestamp("2024-01-01", tz="UTC")
