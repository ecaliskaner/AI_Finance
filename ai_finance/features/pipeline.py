"""Assembling features and labels into something a model can be fitted to.

Two jobs, and the second matters more than the first:

1. Join the feature matrix to the forward-return label, drop rows that are not
   usable, and keep a record of which bar each row came from.
2. **Prove** that the features are point-in-time correct, by recomputing them on
   truncated data and checking that nothing changed.

Job 2 is the licence for everything downstream. Features are built vectorised
over the whole history because that is fast and readable, and a vectorised
feature is also the easiest place in a quant codebase to leak the future. The
proof is cheap, it is a test rather than a code review, and without it the
convenience is not worth taking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ai_finance.features.technical import build_features, forward_return

log = logging.getLogger(__name__)


class PointInTimeError(AssertionError):
    """Raised when a feature's value depends on data that came after it."""


@dataclass(frozen=True)
class Dataset:
    """Aligned features and labels, with the bars they came from."""

    features: pd.DataFrame
    labels: pd.Series
    horizon: int
    periods_per_year: float

    def __post_init__(self) -> None:
        if not self.features.index.equals(self.labels.index):
            raise ValueError("features and labels must share an index")

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        """``close_time`` of each row: when that row became knowable."""
        return self.features.index

    def __len__(self) -> int:
        return len(self.features)

    def slice(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Feature and label arrays for the given row positions."""
        return (
            self.features.to_numpy(dtype=float)[positions],
            self.labels.to_numpy(dtype=float)[positions],
        )


def build_dataset(
    bars: pd.DataFrame,
    *,
    horizon: int,
    periods_per_year: float,
    drop_incomplete: bool = True,
) -> Dataset:
    """Build features and labels from ``bars``.

    Args:
        bars: canonical bars.
        horizon: label horizon in bars. The model predicts the return over this
            many bars ahead.
        periods_per_year: bars per year, for annualising volatility.
        drop_incomplete: drop rows with any missing feature or a missing label.
            The missing labels are the last ``horizon`` rows, whose future has
            not happened; the missing features are the warm-up rows at the
            start.

    Returns:
        A :class:`Dataset`. Its index is ``close_time``, so a row can be matched
        to the bar a strategy is looking at without relying on positions, which
        shift whenever a window is sliced.
    """
    features = build_features(bars, periods_per_year)
    labels = forward_return(bars, horizon)

    if drop_incomplete:
        # A single degenerate column empties the whole dataset, because a row is
        # usable only if every feature is. That is the right policy — imputing a
        # missing volatility invents data — but the failure is otherwise silent
        # and looks like "not enough history".
        always_missing = [name for name in features.columns if features[name].isna().all()]
        if always_missing:
            log.warning(
                "feature(s) %s are missing for every row and will drop the entire "
                "dataset; this usually means a constant input (for example a series "
                "with unchanging volume, which makes volume_z undefined)",
                always_missing,
            )

        usable = features.notna().all(axis=1) & labels.notna()
        features = features.loc[usable]
        labels = labels.loc[usable]

    return Dataset(
        features=features,
        labels=labels,
        horizon=horizon,
        periods_per_year=periods_per_year,
    )


def assert_point_in_time(
    bars: pd.DataFrame,
    periods_per_year: float,
    *,
    cut_points: tuple[float, ...] = (0.4, 0.6, 0.8),
) -> None:
    """Verify that no feature's value depends on data that came after it.

    Recomputes the features on progressively truncated copies of ``bars`` and
    compares every overlapping row against the full-history version. If any
    value differs, that feature saw the future.

    This is the guarantee that makes precomputing a feature matrix safe. Without
    it, handing the backtest row ``i`` at bar ``i`` would be an act of faith.

    Raises:
        PointInTimeError: naming the first offending column and timestamp.
    """
    full = build_features(bars, periods_per_year)

    for fraction in cut_points:
        cut = int(len(bars) * fraction)
        if cut < 2:
            continue
        truncated = build_features(bars.iloc[:cut], periods_per_year)
        overlap = truncated.index

        for column in full.columns:
            left = full.loc[overlap, column].to_numpy(dtype=float)
            right = truncated[column].to_numpy(dtype=float)
            mismatch = ~(np.isclose(left, right, rtol=1e-12, atol=1e-15, equal_nan=True))
            if mismatch.any():
                first = int(np.flatnonzero(mismatch)[0])
                raise PointInTimeError(
                    f"feature {column!r} changed when future bars were removed: "
                    f"at {overlap[first]} it is {left[first]!r} with the full series "
                    f"but {right[first]!r} with only the first {cut} bars. "
                    "That value depends on data from the future."
                )


def directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Fraction of non-flat outcomes whose direction was predicted correctly.

    Bars where the realised return was exactly zero are excluded; there is no
    direction to get right. ``nan`` when nothing is left to score.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    scorable = (actuals != 0.0) & np.isfinite(predictions) & np.isfinite(actuals)
    if not scorable.any():
        return float("nan")
    return float((np.sign(predictions[scorable]) == np.sign(actuals[scorable])).mean())


def information_coefficient(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Correlation between predicted and realised returns.

    The standard quant measure of signal quality. An IC of 0.02 to 0.05 is a real,
    tradeable edge; anything above 0.15 on financial data almost always means a
    leak rather than a discovery.
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    usable = np.isfinite(predictions) & np.isfinite(actuals)
    if usable.sum() < 3:
        return float("nan")
    left, right = predictions[usable], actuals[usable]
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
