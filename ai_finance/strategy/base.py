"""What a strategy is, and what it is allowed to see.

:class:`MarketState` is the primary defense against look-ahead bias. It is
handed to a strategy once per bar and exposes **only** the bars that had already
closed at that moment. A strategy physically cannot read tomorrow's price,
because the arrays it is given are sliced at the current index before it ever
sees them.

That is a deliberate design choice over the convenient alternative of passing
the whole DataFrame and trusting the strategy to only look backwards. Trust is
not a defense: every vectorised backtest that ever produced a spectacular
equity curve and then lost money live was written by someone who intended to
only look backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Bar:
    """One completed bar."""

    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int


class BarWindow:
    """Read-only view of every bar up to and including the current one.

    Slices are numpy views, so this is cheap to construct per bar and does not
    copy the underlying data. Every accessor cuts at ``index``, which is what
    makes future data unreachable rather than merely discouraged.
    """

    __slots__ = ("_arrays", "_index")

    def __init__(self, arrays: dict[str, np.ndarray], index: int) -> None:
        self._arrays = arrays
        self._index = index

    def __len__(self) -> int:
        return self._index + 1

    def _view(self, field: str) -> np.ndarray:
        return self._arrays[field][: self._index + 1]

    @property
    def open(self) -> np.ndarray:
        return self._view("open")

    @property
    def high(self) -> np.ndarray:
        return self._view("high")

    @property
    def low(self) -> np.ndarray:
        return self._view("low")

    @property
    def close(self) -> np.ndarray:
        return self._view("close")

    @property
    def volume(self) -> np.ndarray:
        return self._view("volume")

    def last(self, field: str, n: int) -> np.ndarray:
        """The most recent ``n`` values of ``field``, oldest first.

        Returns fewer than ``n`` values early in the series, so a strategy must
        check the length before computing an indicator that needs a full window.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        return self._view(field)[-n:]


@dataclass(frozen=True)
class MarketState:
    """Everything a strategy may know at one point in time."""

    symbol: str
    index: int
    bar: Bar
    history: BarWindow
    equity: float
    position_weight: float
    bar_seconds: float

    @property
    def timestamp(self) -> pd.Timestamp:
        """When this information became available: the current bar's close time."""
        return self.bar.close_time

    @property
    def periods_per_year(self) -> float:
        """Bars per year at this cadence, for annualising volatility.

        365 days, because crypto does not close. Supplied by the engine so a
        strategy never has to be told its own cadence out of band — the same
        code then works on 4-hour and daily bars without a constructor change.
        """
        return 365.0 * 24.0 * 3600.0 / self.bar_seconds


@dataclass(frozen=True)
class Signal:
    """What the strategy wants. Not yet an order.

    Args:
        symbol: what to trade.
        target_weight: desired position as a fraction of equity, in ``[-1, 1]``.
            This is a *target*, not a delta — emitting 1.0 twice does not double
            the position. That makes strategies idempotent and much easier to
            reason about than buy/sell instructions.
        confidence: in ``[0, 1]``. The risk layer may scale size by this.
        reason: why. Mandatory, logged on every signal, so any trade the system
            takes can be explained after the fact.
    """

    symbol: str
    target_weight: float
    confidence: float = 1.0
    reason: str = ""

    def __post_init__(self) -> None:
        if not -1.0 <= self.target_weight <= 1.0:
            raise ValueError(f"target_weight must be in [-1, 1], got {self.target_weight}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@runtime_checkable
class Strategy(Protocol):
    """Anything that can look at a bar and say what it wants to hold."""

    name: str

    def on_bar(self, state: MarketState) -> Signal | None:
        """Decide a target weight from ``state``.

        Returning ``None`` means "no change" — the previous target stands. That
        is different from returning a signal with ``target_weight=0``, which
        means "go flat".
        """
        ...
