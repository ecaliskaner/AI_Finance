"""Feature construction.

Everything here is computed vectorised over a whole bars frame, which is fast
and convenient and also exactly how look-ahead bias gets into a model. So one
rule governs this module:

    **The value of any feature at row i must depend only on rows 0..i.**

Concretely that means rolling windows that end at the current row, `ewm` rather
than a centred filter, and never a negative shift. The rule is not enforced by
review — :func:`ai_finance.features.pipeline.assert_point_in_time` proves it by
recomputing the features on a truncated frame and checking that nothing changed.
That test is what makes it safe to precompute a feature matrix and hand the
backtest row ``i`` at bar ``i``, rather than recomputing indicators inside the
event loop and hoping the two implementations agree.

**What is missing, and why.** `docs/PLAN.md` Phase 3 also lists order-book
imbalance and funding rates. Neither is derivable from OHLCV bars, which is all
the store holds, so neither is here. They are a data-collection task, not a
feature-engineering one, and pretending otherwise would put two empty columns in
the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Lookbacks, in bars, for the multi-horizon return features.
RETURN_LOOKBACKS = (1, 4, 12, 24)

#: Windows for realised volatility.
VOL_WINDOWS = (12, 48)

#: Window for the channel-position and volume features.
CHANNEL_WINDOW = 20

#: Moving averages whose ratio expresses trend.
MA_FAST, MA_SLOW = 20, 50

#: RSI period.
RSI_PERIOD = 14


def build_features(bars: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """Compute the feature matrix for ``bars``.

    Args:
        bars: canonical bars, as returned by
            :func:`ai_finance.data.store.load_bars`.
        periods_per_year: bars per year at this cadence, for annualising
            volatility.

    Returns:
        A frame indexed by ``close_time`` — the moment each row became knowable
        — with one column per feature. Early rows contain ``NaN`` where a window
        is not yet full; callers drop them rather than filling, because a
        volatility estimate from four observations is not a volatility estimate.
    """
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    volume = bars["volume"].to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars["close_time"], name="close_time")

    close_series = pd.Series(close, index=index)
    log_close = pd.Series(np.log(np.where(close > 0, close, np.nan)), index=index)
    log_returns = log_close.diff()

    features: dict[str, pd.Series] = {}

    # --- momentum over several horizons -----------------------------------
    for lookback in RETURN_LOOKBACKS:
        features[f"ret_{lookback}"] = log_close.diff(lookback)

    # --- volatility, and the ratio that flags a regime change -------------
    annualise = float(np.sqrt(periods_per_year))
    for window in VOL_WINDOWS:
        features[f"vol_{window}"] = log_returns.rolling(window).std(ddof=1) * annualise
    short, long = VOL_WINDOWS[0], VOL_WINDOWS[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        features["vol_ratio"] = features[f"vol_{short}"] / features[f"vol_{long}"]

    # --- trend -------------------------------------------------------------
    fast = close_series.rolling(MA_FAST).mean()
    slow = close_series.rolling(MA_SLOW).mean()
    features["ma_ratio"] = fast / slow - 1.0

    # --- position within the recent range ---------------------------------
    channel_high = pd.Series(high, index=index).rolling(CHANNEL_WINDOW).max()
    channel_low = pd.Series(low, index=index).rolling(CHANNEL_WINDOW).min()
    features["dist_from_high"] = close_series / channel_high - 1.0
    features["dist_from_low"] = close_series / channel_low - 1.0

    # --- RSI, as an exponentially weighted mean, which is Wilder smoothing --
    delta = close_series.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1.0 / RSI_PERIOD, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = gain / loss
    features["rsi"] = (100.0 - 100.0 / (1.0 + strength)).where(loss > 0, 100.0)

    # --- activity ----------------------------------------------------------
    volume_series = pd.Series(volume, index=index)
    volume_mean = volume_series.rolling(CHANNEL_WINDOW).mean()
    volume_std = volume_series.rolling(CHANNEL_WINDOW).std(ddof=1)
    features["volume_z"] = (volume_series - volume_mean) / volume_std.replace(0.0, np.nan)
    features["range_pct"] = (pd.Series(high - low, index=index) / close_series).replace(
        [np.inf, -np.inf], np.nan
    )

    # --- time of day and week, encoded cyclically --------------------------
    # Sine and cosine rather than a raw hour number, so that 23:00 and 00:00 are
    # adjacent to the model instead of maximally far apart.
    hour = index.hour + index.minute / 60.0
    features["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24.0), index=index)
    features["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24.0), index=index)
    dayofweek = index.dayofweek.to_numpy(dtype=float)
    features["dow_sin"] = pd.Series(np.sin(2 * np.pi * dayofweek / 7.0), index=index)
    features["dow_cos"] = pd.Series(np.cos(2 * np.pi * dayofweek / 7.0), index=index)

    frame = pd.DataFrame(features, index=index)
    return frame.replace([np.inf, -np.inf], np.nan)


def feature_names() -> list[str]:
    """Column order produced by :func:`build_features`, for assertions."""
    names = [f"ret_{lookback}" for lookback in RETURN_LOOKBACKS]
    names += [f"vol_{window}" for window in VOL_WINDOWS]
    names += [
        "vol_ratio",
        "ma_ratio",
        "dist_from_high",
        "dist_from_low",
        "rsi",
        "volume_z",
        "range_pct",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    return names


def warmup_rows() -> int:
    """Count of leading rows that contain ``NaN`` for at least one feature.

    Row ``warmup_rows()`` is the first complete one. The windows differ in when
    they fill: a ``diff(n)`` needs ``n`` prior rows, a ``rolling(n)`` needs
    ``n - 1``, and a rolling window over *returns* needs one more than that
    because the first return is itself undefined.
    """
    return max(
        max(RETURN_LOOKBACKS),  # diff(n) is valid from row n
        max(VOL_WINDOWS),  # rolling(n) over returns is valid from row n
        MA_SLOW - 1,  # rolling(n) over prices is valid from row n-1
        CHANNEL_WINDOW - 1,
    )


def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """The label: simple return over the next ``horizon`` bars.

    This is deliberately forward-looking — it is the thing being predicted — and
    so it is the single most dangerous series in the project. It must never
    appear as a feature, and the last ``horizon`` rows have no label at all
    because the future they refer to has not happened yet.

    Measured close-to-close. The backtest fills at the *next* bar's open, so the
    model's target and the strategy's realised return differ by one bar of
    slippage. That gap is real and is exactly what the cost model exists to
    charge for.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    close = pd.Series(
        bars["close"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(bars["close_time"], name="close_time"),
    )
    return close.shift(-horizon) / close - 1.0
