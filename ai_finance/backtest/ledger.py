"""Trade accounting.

A *trade* is one continuous stretch of non-zero exposure: it opens when the
position leaves zero and closes when it returns. Rebalances in between are part
of the same trade.

Profit is computed from cash flows rather than from prices. Over a complete
round trip the signed cash flows sum to exactly the negative of the profit, fees
and spread included, with no separate cost adjustment to get wrong::

    buy 1 @ 100, fee 0.10   ->  cash out  +100.10
    sell 1 @ 110, fee 0.11  ->  cash out  -109.89
                                          --------
                               total       -9.79   ->  profit +9.79
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ai_finance.execution.base import Fill


@dataclass(frozen=True)
class Trade:
    """A completed round trip."""

    symbol: str
    direction: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    entry_notional: float
    pnl: float
    fees: float
    price_concession: float
    n_fills: int
    bars_held: int

    @property
    def return_pct(self) -> float:
        """Profit as a fraction of the notional put at risk."""
        if self.entry_notional == 0:
            return 0.0
        return self.pnl / self.entry_notional

    @property
    def total_cost(self) -> float:
        return self.fees + self.price_concession

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class _OpenTrade:
    symbol: str
    direction: str
    open_time: pd.Timestamp
    open_index: int
    entry_notional: float
    cash_flow: float = 0.0
    fees: float = 0.0
    price_concession: float = 0.0
    n_fills: int = 0


class TradeLedger:
    """Turns a stream of fills into completed trades."""

    def __init__(self) -> None:
        self.closed: list[Trade] = []
        self._units = 0.0
        self._max_abs_units = 0.0
        self._open: _OpenTrade | None = None

    @property
    def has_open_position(self) -> bool:
        return self._open is not None

    @property
    def units(self) -> float:
        return self._units

    def record(self, fill: Fill, bar_index: int) -> None:
        """Absorb a fill, splitting it if it flips the position through zero."""
        delta = fill.delta_units
        if delta == 0.0:
            return

        flipping = self._units != 0.0 and (self._units > 0.0) != (delta > 0.0)
        if flipping and abs(delta) > abs(self._units):
            # Part of this fill closes the existing position; the rest opens a
            # new one in the other direction. Costs split by traded size.
            closing = math.copysign(abs(self._units), delta)
            opening = delta - closing
            share = abs(closing) / abs(delta)
            self._absorb(fill, closing, share, bar_index)
            self._absorb(fill, opening, 1.0 - share, bar_index)
        else:
            self._absorb(fill, delta, 1.0, bar_index)

    def _absorb(self, fill: Fill, units: float, cost_share: float, bar_index: int) -> None:
        if units == 0.0:
            return

        if self._open is None:
            self._open = _OpenTrade(
                symbol=fill.symbol,
                direction="long" if units > 0 else "short",
                open_time=fill.timestamp,
                open_index=bar_index,
                entry_notional=abs(units) * fill.fill_price,
            )

        fee = fill.fee * cost_share
        self._open.cash_flow += units * fill.fill_price + fee
        self._open.fees += fee
        self._open.price_concession += fill.price_concession * cost_share
        self._open.n_fills += 1

        self._units += units
        self._max_abs_units = max(self._max_abs_units, abs(self._units))

        if self._is_flat():
            self._close(fill.timestamp, bar_index)

    def _is_flat(self) -> bool:
        tolerance = 1e-12 * max(1.0, self._max_abs_units)
        return abs(self._units) <= tolerance

    def _close(self, timestamp: pd.Timestamp, bar_index: int) -> None:
        assert self._open is not None
        trade = self._open
        self.closed.append(
            Trade(
                symbol=trade.symbol,
                direction=trade.direction,
                open_time=trade.open_time,
                close_time=timestamp,
                entry_notional=trade.entry_notional,
                pnl=-trade.cash_flow,
                fees=trade.fees,
                price_concession=trade.price_concession,
                n_fills=trade.n_fills,
                bars_held=bar_index - trade.open_index,
            )
        )
        self._open = None
        self._units = 0.0
        self._max_abs_units = 0.0


@dataclass
class Portfolio:
    """Cash and one asset position. Single-symbol by design for now.

    Multi-asset portfolios need a correlation-aware risk layer to be meaningful,
    which is a Phase 3 concern. One symbol keeps the accounting verifiable.
    """

    cash: float
    units: float = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def weight(self, price: float) -> float:
        equity = self.equity(price)
        if equity == 0:
            return 0.0
        return (self.units * price) / equity

    def apply(self, fill: Fill) -> None:
        self.cash -= fill.cash_flow
        self.units += fill.delta_units
