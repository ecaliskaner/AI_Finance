"""Simulated execution against historical bars."""

from __future__ import annotations

from ai_finance.backtest.costs import CostModel
from ai_finance.execution.base import Fill, Order


class SimulatedExecution:
    """Fills every order at the reference price plus the modelled cost.

    Deliberately optimistic in exactly one way: it always fills, in full,
    immediately. It does not model a limit order that never gets hit, or a book
    too thin to absorb the size. At $5k notional in BTC/USDT that is a fair
    approximation — the book is millions deep. It would not be for a thin
    altcoin, and it is the first assumption to revisit if paper trading diverges
    from the backtest.
    """

    name = "backtest"

    def __init__(self, costs: CostModel) -> None:
        self.costs = costs

    def submit(self, order: Order) -> Fill | None:
        if order.delta_units == 0.0:
            return None
        fill_price = self.costs.fill_price(order.reference_price, order.delta_units)
        fee = self.costs.fee(abs(order.delta_units) * fill_price)
        return Fill(
            timestamp=order.timestamp,
            symbol=order.symbol,
            delta_units=order.delta_units,
            reference_price=order.reference_price,
            fill_price=fill_price,
            fee=fee,
            reason=order.reason,
        )
