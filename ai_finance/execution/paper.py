"""Paper execution: live prices, simulated money.

This deliberately reuses :class:`~ai_finance.execution.backtest.SimulatedExecution`
rather than reimplementing the fill. Identical inputs must produce an identical
fill in both modes, because that is the entire premise of the reconciliation in
`docs/PLAN.md` Phase 4: when paper trading and the backtest disagree, the
disagreement has to be attributable to the *inputs* — a price that differed, a
bar that arrived late, an order the risk layer blocked — and never to two
implementations of the same arithmetic drifting apart.

So the only real difference between backtest and paper is where the reference
price comes from: history in one case, the live market in the other.
"""

from __future__ import annotations

import logging

from ai_finance.backtest.costs import CostModel
from ai_finance.execution.backtest import SimulatedExecution
from ai_finance.execution.base import Fill, Order

log = logging.getLogger(__name__)


class PaperExecution:
    """Simulated fills at live prices. Moves no real money, ever."""

    name = "paper"

    def __init__(self, costs: CostModel) -> None:
        self.costs = costs
        self._simulator = SimulatedExecution(costs)

    def submit(self, order: Order) -> Fill | None:
        fill = self._simulator.submit(order)
        if fill is not None:
            log.info(
                "PAPER %s %.8f %s at %.2f (reference %.2f, fee %.4f) — no real money moved",
                order.side,
                abs(fill.delta_units),
                order.symbol,
                fill.fill_price,
                fill.reference_price,
                fill.fee,
            )
        return fill
