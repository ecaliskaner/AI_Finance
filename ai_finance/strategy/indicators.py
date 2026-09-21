"""Indicator primitives.

Pure functions over numpy arrays, each returning the value **for the most recent
bar** rather than a whole series. That shape matches the event loop: a strategy
is asked what it wants to hold right now, so it needs one number, not a column.

Every function returns ``nan`` when there is not enough history, and strategies
are expected to check. Returning a value computed from a half-full window would
silently make early bars behave differently from later ones, which is the kind
of thing that shows up as an unexplained divergence in paper trading months
later.

Complexity: these recompute over their window on every call, which is O(window)
per bar rather than O(1). At a 4-hour cadence that is ~11,000 bars over five
years and the cost is irrelevant. It would matter on 1-minute bars, and the fix
then is incremental state, not a different indicator.
"""

from __future__ import annotations

import numpy as np


def sma(values: np.ndarray, window: int) -> float:
    """Simple moving average of the last ``window`` values."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(values) < window:
        return float("nan")
    return float(np.mean(values[-window:]))


def rsi(values: np.ndarray, window: int = 14) -> float:
    """Wilder's Relative Strength Index, in ``[0, 100]``.

    Wilder's smoothing is recursive, so the result depends on how much history
    it was seeded with. Pass several times ``window`` values (5x is plenty) for
    it to converge; with exactly ``window + 1`` it degenerates to a simple
    average of gains and losses, which is a different indicator wearing the same
    name.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(values) < window + 1:
        return float("nan")

    deltas = np.diff(values)
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)

    # Seed with a simple average, then smooth the rest Wilder's way.
    avg_gain = float(np.mean(gains[:window]))
    avg_loss = float(np.mean(losses[:window]))
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window

    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def realized_volatility(values: np.ndarray, window: int, periods_per_year: float) -> float:
    """Annualised standard deviation of log returns over ``window`` bars."""
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(values) < window + 1:
        return float("nan")

    tail = values[-(window + 1) :]
    if np.any(tail <= 0):
        return float("nan")
    log_returns = np.diff(np.log(tail))
    return float(np.std(log_returns, ddof=1) * np.sqrt(periods_per_year))


def donchian(high: np.ndarray, low: np.ndarray, window: int) -> tuple[float, float]:
    """Highest high and lowest low over the ``window`` bars **before** the last one.

    The current bar is excluded on purpose. A breakout strategy asks "is today's
    close above the range that came before it?" — including today's own high in
    that range makes the test nearly impossible to pass and quietly changes the
    strategy into something else.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(high) < window + 1 or len(low) < window + 1:
        return float("nan"), float("nan")
    return float(np.max(high[-(window + 1) : -1])), float(np.min(low[-(window + 1) : -1]))


def momentum(values: np.ndarray, window: int) -> float:
    """Simple return over the last ``window`` bars."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(values) < window + 1:
        return float("nan")
    earlier = values[-(window + 1)]
    if earlier <= 0:
        return float("nan")
    return float(values[-1] / earlier - 1.0)
