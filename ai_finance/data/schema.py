"""The canonical bar schema, and the one function that enforces it.

Every DataFrame of bars anywhere in this project has exactly these columns with
exactly these dtypes. Anything that produces bars — the exchange, the synthetic
source, a resampler, a Parquet file — passes through :func:`normalize_bars`
before it is handed to anything else.

**Timing convention, which matters more than it looks.**

A bar is labelled by ``open_time`` and covers the half-open interval
``[open_time, close_time]``. A 1-minute bar labelled ``12:00:00`` covers
``12:00:00.000``–``12:00:59.999``, so its close price is *not known* until
``12:01:00``. Acting on that bar at ``12:00:00`` is look-ahead bias, and it is
the single most common way a backtest becomes fiction.

``close_time`` is therefore stored rather than recomputed: it is the timestamp at
which the bar became knowable, and the point-in-time layer uses it to decide
what a strategy is allowed to see.
"""

from __future__ import annotations

import pandas as pd

#: Column order for every bars DataFrame in the project.
BAR_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "close_time",
)

_FLOAT_COLUMNS = ("open", "high", "low", "close", "volume")
_TIME_COLUMNS = ("open_time", "close_time")


class SchemaError(ValueError):
    """Raised when a DataFrame cannot be coerced to the bar schema."""


def empty_bars() -> pd.DataFrame:
    """An empty DataFrame with the correct columns and dtypes."""
    return pd.DataFrame(
        {
            "open_time": pd.Series([], dtype="datetime64[ms, UTC]"),
            "open": pd.Series([], dtype="float64"),
            "high": pd.Series([], dtype="float64"),
            "low": pd.Series([], dtype="float64"),
            "close": pd.Series([], dtype="float64"),
            "volume": pd.Series([], dtype="float64"),
            "trades": pd.Series([], dtype="int64"),
            "close_time": pd.Series([], dtype="datetime64[ms, UTC]"),
        }
    )


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``df`` to the canonical bar schema.

    Sorts by ``open_time``, drops duplicate timestamps keeping the last
    occurrence (a later fetch of the same bar is the more authoritative one),
    and resets the index. Does *not* judge whether the data is any good — that
    is :mod:`ai_finance.data.quality`'s job.

    Raises:
        SchemaError: if a required column is missing.
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")

    out = df.loc[:, list(BAR_COLUMNS)].copy()

    for col in _TIME_COLUMNS:
        series = out[col]
        if not isinstance(series.dtype, pd.DatetimeTZDtype):
            # Integers are milliseconds since epoch (what Binance sends).
            if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
                series = pd.to_datetime(series, unit="ms", utc=True)
            else:
                series = pd.to_datetime(series, utc=True)
        out[col] = series.astype("datetime64[ms, UTC]")

    for col in _FLOAT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out["trades"] = pd.to_numeric(out["trades"], errors="coerce").fillna(0).astype("int64")

    out = (
        out.sort_values("open_time", kind="stable")
        .drop_duplicates(subset="open_time", keep="last")
        .reset_index(drop=True)
    )
    return out


def assert_schema(df: pd.DataFrame) -> None:
    """Raise :class:`SchemaError` unless ``df`` is already canonical.

    Used in tests and at module boundaries. Cheap enough to call freely.
    """
    if tuple(df.columns) != BAR_COLUMNS:
        raise SchemaError(f"expected columns {BAR_COLUMNS}, got {tuple(df.columns)}")
    for col in _TIME_COLUMNS:
        if not isinstance(df[col].dtype, pd.DatetimeTZDtype):
            raise SchemaError(f"{col} must be timezone-aware datetime, got {df[col].dtype}")
        if str(df[col].dt.tz) != "UTC":
            raise SchemaError(f"{col} must be UTC, got {df[col].dt.tz}")
    for col in _FLOAT_COLUMNS:
        if df[col].dtype != "float64":
            raise SchemaError(f"{col} must be float64, got {df[col].dtype}")
    if df["trades"].dtype != "int64":
        raise SchemaError(f"trades must be int64, got {df['trades'].dtype}")
    if not df["open_time"].is_monotonic_increasing:
        raise SchemaError("open_time must be sorted ascending")
    if df["open_time"].duplicated().any():
        raise SchemaError("open_time must be unique")
