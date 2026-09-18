from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.data.schema import (
    BAR_COLUMNS,
    SchemaError,
    assert_schema,
    empty_bars,
    normalize_bars,
)
from tests.conftest import bars_frame


def test_empty_bars_is_canonical():
    assert_schema(empty_bars())
    assert tuple(empty_bars().columns) == BAR_COLUMNS


def test_normalize_converts_epoch_millis_and_strings():
    raw = pd.DataFrame(
        {
            "open_time": [1_700_000_000_000],
            "open": ["100.5"],
            "high": ["101.0"],
            "low": ["99.0"],
            "close": ["100.0"],
            "volume": ["3.5"],
            "trades": ["7"],
            "close_time": [1_700_000_059_999],
        }
    )
    bars = normalize_bars(raw)
    assert_schema(bars)
    assert bars["open_time"].iloc[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")
    assert bars["open"].iloc[0] == pytest.approx(100.5)
    assert bars["trades"].iloc[0] == 7


def test_normalize_sorts_and_drops_duplicates_keeping_last():
    bars = bars_frame("2024-01-01", 3)
    shuffled = pd.concat([bars.iloc[[2, 0, 1]], bars.iloc[[1]].assign(close=999.0)])
    result = normalize_bars(shuffled)

    assert_schema(result)
    assert len(result) == 3
    # The later occurrence of the duplicated timestamp wins.
    assert result["close"].iloc[1] == 999.0


def test_normalize_rejects_missing_columns():
    with pytest.raises(SchemaError, match="missing required column"):
        normalize_bars(pd.DataFrame({"open_time": [0], "open": [1.0]}))


def test_assert_schema_rejects_naive_timestamps():
    bars = bars_frame("2024-01-01", 2)
    broken = bars.assign(open_time=bars["open_time"].dt.tz_localize(None))
    with pytest.raises(SchemaError, match="timezone-aware"):
        assert_schema(broken)


def test_assert_schema_rejects_unsorted():
    bars = bars_frame("2024-01-01", 3)
    with pytest.raises(SchemaError, match="sorted"):
        assert_schema(bars.iloc[::-1].reset_index(drop=True))


def test_close_time_marks_when_a_bar_became_knowable():
    """The timing convention the whole point-in-time defense rests on."""
    bars = bars_frame("2024-01-01", 1)
    assert bars["open_time"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert bars["close_time"].iloc[0] == pd.Timestamp("2024-01-01 00:00:59.999", tz="UTC")
