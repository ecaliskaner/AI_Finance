"""Purged, embargoed splits for supervised learning on time series.

Standard cross-validation is wrong here twice over. Shuffling folds trains on
the future to predict the past. But even a naive walk-forward split leaks, for a
subtler reason: a training row at time ``t`` is labelled with the return from
``t`` to ``t + horizon``. If ``t + horizon`` falls inside the test window, that
training row's *label* is partly made of the very data the model is about to be
scored on.

So two things are removed from the end of every training window:

- **Purge** — the last ``horizon`` rows, whose labels reach into the test period.
- **Embargo** — a further margin, because serial correlation means bars adjacent
  to the boundary carry much the same information even when their label windows
  do not literally overlap.

Both shrink the training set. That is the cost of a number you can believe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedSplit:
    """One train/test division, with the leaky rows already removed."""

    index: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    purged: int
    train_end_before_purge: int

    @property
    def n_train(self) -> int:
        return len(self.train_positions)

    @property
    def n_test(self) -> int:
        return len(self.test_positions)


def purged_splits(
    n_rows: int,
    *,
    train_size: int,
    test_size: int,
    horizon: int,
    embargo: int = 0,
    anchored: bool = False,
    max_splits: int | None = None,
) -> list[PurgedSplit]:
    """Rolling train/test splits with overlapping-label rows removed.

    Args:
        n_rows: rows in the dataset.
        train_size: rows in each training window *before* purging.
        test_size: rows in each test window, and the step between splits.
        horizon: label horizon in rows. Rows whose label reaches into the test
            window are dropped from training.
        embargo: extra rows dropped beyond the purge.
        anchored: expand the training window from row zero instead of rolling.
        max_splits: stop after this many.

    Returns:
        Splits in chronological order. A split is skipped entirely if purging
        leaves it with no training rows — a silently tiny training set is worse
        than a missing fold.
    """
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be positive")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")

    splits: list[PurgedSplit] = []
    start = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > n_rows:
            break

        # A row at position j is labelled with the return to j + horizon. Keep
        # it only if that label closes before the test window opens.
        cutoff = train_end - horizon - embargo
        train_start = 0 if anchored else start
        train_positions = np.arange(train_start, max(train_start, cutoff))

        if len(train_positions) > 0:
            splits.append(
                PurgedSplit(
                    index=len(splits),
                    train_positions=train_positions,
                    test_positions=np.arange(train_end, test_end),
                    purged=train_end - max(train_start, cutoff),
                    train_end_before_purge=train_end,
                )
            )
            if max_splits is not None and len(splits) >= max_splits:
                break
        start += test_size

    return splits
