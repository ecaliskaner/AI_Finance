"""The cost model.

The whole project turns on one number: it costs about 20 basis points to buy and
sell once. A typical 1-minute BTC move is about 7. That arithmetic is why this
system decides on a 4-hour horizon rather than a 1-minute one, and it only stays
honest if every simulated fill pays in full.

So costs are an explicit input to every backtest, never a correction applied
afterwards, and three separate things are charged:

- **Fee** — the exchange's cut, per side, on notional.
- **Half-spread** — you buy at the ask and sell at the bid, never at the mid.
- **Slippage** — everything else that makes a real fill worse than the quote.

Defaults are deliberately a little pessimistic (22 bps round trip against the
20 bps the plan assumes). A strategy that only works at optimistic costs does
not work.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """What it costs to trade. All rates are fractions of notional.

    Args:
        fee_rate: exchange fee **per side**. Binance spot standard tier is
            0.001 (10 bps); 0.00075 with the BNB discount.
        half_spread_bps: half the bid-ask spread, paid on every crossing. BTC/USDT
            sits around 1 bp full spread, so 0.5 bp per side.
        slippage_bps: per-side allowance for everything else. At $5k order sizes
            in BTC/USDT the book is deep enough that this is small; for thin
            altcoins it is not.
    """

    fee_rate: float = 0.001
    half_spread_bps: float = 0.5
    slippage_bps: float = 0.5

    def __post_init__(self) -> None:
        if self.fee_rate < 0 or self.half_spread_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost components cannot be negative")

    @classmethod
    def free(cls) -> CostModel:
        """A zero-cost model. **For engine tests only.**

        Used to prove the engine reproduces a known price return exactly. Any
        research result produced with this is meaningless.
        """
        return cls(fee_rate=0.0, half_spread_bps=0.0, slippage_bps=0.0)

    @property
    def edge(self) -> float:
        """Price concession per side, as a fraction: half-spread plus slippage."""
        return (self.half_spread_bps + self.slippage_bps) * BPS

    @property
    def per_side_cost(self) -> float:
        """Total cost of one trade, as a fraction of notional."""
        return self.fee_rate + self.edge

    @property
    def round_trip_cost(self) -> float:
        """Total cost of buying and selling once, as a fraction of notional.

        The number to compare against the size of the move you are predicting.
        """
        return 2.0 * self.per_side_cost

    def fill_price(self, reference_price: float, delta_units: float) -> float:
        """Price actually paid, worse than ``reference_price`` in both directions."""
        if delta_units > 0:
            return reference_price * (1.0 + self.edge)
        if delta_units < 0:
            return reference_price * (1.0 - self.edge)
        return reference_price

    def fee(self, notional: float) -> float:
        """Exchange fee on an absolute notional."""
        return abs(notional) * self.fee_rate

    def describe(self) -> str:
        return (
            f"fee {self.fee_rate * 100:.4f}%/side, "
            f"half-spread {self.half_spread_bps:.2f}bp, "
            f"slippage {self.slippage_bps:.2f}bp "
            f"-> {self.round_trip_cost * 10_000:.1f}bp round trip"
        )
