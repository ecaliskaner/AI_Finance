"""Orders and fills: the shared vocabulary of all three execution modes.

The same types flow through the backtest, paper and live adapters. That is what
lets one strategy implementation run in all three, and what makes a divergence
between backtest and live a *measurable* discrepancy in the cost model rather
than an unexplainable mystery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Order:
    """A request to change the position by ``delta_units``."""

    symbol: str
    delta_units: float
    reference_price: float
    timestamp: pd.Timestamp
    reason: str = ""

    @property
    def side(self) -> str:
        return "buy" if self.delta_units > 0 else "sell"

    @property
    def reference_notional(self) -> float:
        return abs(self.delta_units) * self.reference_price


@dataclass(frozen=True)
class Fill:
    """What actually happened, with the cost decomposed.

    Keeping fee and price concession separate is what makes a cost post-mortem
    possible: "we lost to fees" and "we lost to slippage" call for different
    fixes.
    """

    timestamp: pd.Timestamp
    symbol: str
    delta_units: float
    reference_price: float
    fill_price: float
    fee: float
    reason: str = ""

    @property
    def notional(self) -> float:
        """Traded value at the fill price."""
        return abs(self.delta_units) * self.fill_price

    @property
    def price_concession(self) -> float:
        """Cost of not filling at the reference price: spread plus slippage."""
        return abs(self.delta_units) * abs(self.fill_price - self.reference_price)

    @property
    def total_cost(self) -> float:
        return self.fee + self.price_concession

    @property
    def cash_flow(self) -> float:
        """Signed cash leaving the account. Positive when buying.

        Summing this over a complete round trip gives the negative of the
        trade's profit, which is how the trade ledger stays exact.
        """
        return self.delta_units * self.fill_price + self.fee


class ExecutionAdapter(Protocol):
    """Turns an :class:`Order` into a :class:`Fill`, or into nothing."""

    name: str

    def submit(self, order: Order) -> Fill | None:
        """Execute ``order``. ``None`` means it was not filled."""
        ...
