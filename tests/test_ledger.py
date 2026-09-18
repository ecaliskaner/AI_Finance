from __future__ import annotations

import pandas as pd
import pytest

from ai_finance.backtest.ledger import Portfolio, TradeLedger
from ai_finance.execution.base import Fill

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def fill(units, price, fee=0.0, *, ref=None, minutes=0):
    return Fill(
        timestamp=T0 + pd.Timedelta(minutes=minutes),
        symbol="BTCUSDT",
        delta_units=units,
        reference_price=ref if ref is not None else price,
        fill_price=price,
        fee=fee,
    )


class TestPortfolio:
    def test_equity_and_weight(self):
        p = Portfolio(cash=500.0, units=5.0)
        assert p.equity(100.0) == 1000.0
        assert p.weight(100.0) == pytest.approx(0.5)

    def test_weight_of_empty_account(self):
        assert Portfolio(cash=0.0).weight(100.0) == 0.0

    def test_applying_a_buy_moves_cash_into_units(self):
        p = Portfolio(cash=1000.0)
        p.apply(fill(5.0, 100.0, fee=0.5))
        assert p.cash == pytest.approx(1000.0 - 500.0 - 0.5)
        assert p.units == 5.0

    def test_applying_a_sell_returns_cash_minus_fee(self):
        p = Portfolio(cash=0.0, units=5.0)
        p.apply(fill(-5.0, 110.0, fee=0.55))
        assert p.cash == pytest.approx(550.0 - 0.55)
        assert p.units == 0.0


class TestTradeLedger:
    def test_round_trip_pnl_is_net_of_all_costs(self):
        """The worked example from the module docstring."""
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0, fee=0.10), 0)
        ledger.record(fill(-1.0, 110.0, fee=0.11, minutes=5), 5)

        assert len(ledger.closed) == 1
        trade = ledger.closed[0]
        assert trade.pnl == pytest.approx(9.79)
        assert trade.fees == pytest.approx(0.21)
        assert trade.direction == "long"
        assert trade.entry_notional == pytest.approx(100.0)
        assert trade.return_pct == pytest.approx(0.0979)
        assert trade.bars_held == 5
        assert trade.n_fills == 2
        assert trade.is_win

    def test_losing_trade(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0, fee=0.1), 0)
        ledger.record(fill(-1.0, 90.0, fee=0.09), 1)

        trade = ledger.closed[0]
        assert trade.pnl == pytest.approx(-10.19)
        assert not trade.is_win

    def test_short_trade_profits_when_price_falls(self):
        ledger = TradeLedger()
        ledger.record(fill(-1.0, 100.0, fee=0.1), 0)
        ledger.record(fill(1.0, 90.0, fee=0.09), 1)

        trade = ledger.closed[0]
        assert trade.direction == "short"
        assert trade.pnl == pytest.approx(10.0 - 0.19)

    def test_position_open_until_it_returns_to_zero(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0), 0)
        assert ledger.has_open_position
        assert ledger.closed == []

        ledger.record(fill(-1.0, 100.0), 1)
        assert not ledger.has_open_position
        assert len(ledger.closed) == 1

    def test_partial_exits_stay_within_one_trade(self):
        ledger = TradeLedger()
        ledger.record(fill(2.0, 100.0, fee=0.2), 0)
        ledger.record(fill(-1.0, 110.0, fee=0.11), 1)
        assert ledger.closed == []  # still holding one unit

        ledger.record(fill(-1.0, 120.0, fee=0.12), 2)

        assert len(ledger.closed) == 1
        trade = ledger.closed[0]
        # -200 spent, +110 and +120 received, 0.43 of fees.
        assert trade.pnl == pytest.approx(30.0 - 0.43)
        assert trade.n_fills == 3

    def test_adding_to_a_position_stays_one_trade(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0), 0)
        ledger.record(fill(1.0, 100.0), 1)
        ledger.record(fill(-2.0, 110.0), 2)

        assert len(ledger.closed) == 1
        assert ledger.closed[0].pnl == pytest.approx(20.0)

    def test_flipping_long_to_short_splits_into_two_trades(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0, fee=0.1), 0)
        # Sell 3: one unit closes the long, two open a short.
        ledger.record(fill(-3.0, 110.0, fee=0.33), 1)

        assert len(ledger.closed) == 1
        closed = ledger.closed[0]
        assert closed.direction == "long"
        # Fee on the closing leg is one third of 0.33.
        assert closed.pnl == pytest.approx(10.0 - 0.1 - 0.11)
        # The short is now open for two units.
        assert ledger.has_open_position
        assert ledger.units == pytest.approx(-2.0)

    def test_fees_split_proportionally_across_a_flip(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0, fee=0.0), 0)
        ledger.record(fill(-4.0, 100.0, fee=4.0), 1)
        ledger.record(fill(3.0, 100.0, fee=3.0), 2)

        # Closing leg took 1/4 of the 4.0 fee; the short leg took 3/4 plus its
        # own 3.0 on exit.
        assert ledger.closed[0].fees == pytest.approx(1.0)
        assert ledger.closed[1].fees == pytest.approx(3.0 + 3.0)

    def test_exact_flip_to_flat_closes_without_opening(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0), 0)
        ledger.record(fill(-1.0, 100.0), 1)
        assert len(ledger.closed) == 1
        assert not ledger.has_open_position

    def test_zero_size_fill_is_ignored(self):
        ledger = TradeLedger()
        ledger.record(fill(0.0, 100.0), 0)
        assert not ledger.has_open_position
        assert ledger.closed == []

    def test_tiny_residual_counts_as_flat(self):
        """Float arithmetic must not leave a trade open forever."""
        ledger = TradeLedger()
        ledger.record(fill(1.0, 100.0), 0)
        ledger.record(fill(-1.0 + 1e-17, 100.0), 1)
        assert not ledger.has_open_position
        assert len(ledger.closed) == 1

    def test_trade_return_of_zero_notional_is_zero(self):
        ledger = TradeLedger()
        ledger.record(fill(1.0, 0.0), 0)
        ledger.record(fill(-1.0, 0.0), 1)
        assert ledger.closed[0].return_pct == 0.0
