"""The event loop. The most important code in the project.

Bars are replayed one at a time. The strategy is shown bar ``i`` only after it
has closed, and any resulting order fills at the **open of bar i+1**.

That one-bar delay is the single most consequential decision in this file. The
convenient alternative — deciding on bar ``i``'s close and filling at that same
close — is what most tutorial backtests do, and it is look-ahead bias: you
cannot observe a bar's closing price and simultaneously trade at it. In
practice you learn the close when the bar ends and your order reaches the book
moments later, which the next bar's open approximates honestly.

The cost of getting this wrong is not subtle. A strategy given one bar of
foresight on minute data can look extraordinary and be worth nothing.

Order of operations per bar:

1. Fill the order decided on the previous bar, at this bar's **open**.
2. Mark equity at this bar's **close**.
3. Let the risk engine observe that equity and trip any halts.
4. Show the strategy this bar and take its target for the next one.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.ledger import Portfolio, TradeLedger
from ai_finance.backtest.metrics import BacktestResult
from ai_finance.data.schema import assert_schema
from ai_finance.execution.backtest import SimulatedExecution
from ai_finance.execution.base import Order
from ai_finance.risk.engine import RiskEngine, RiskLimits
from ai_finance.strategy.base import Bar, BarWindow, MarketState, Strategy

log = logging.getLogger(__name__)

_FIELDS = ("open", "high", "low", "close", "volume")


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    *,
    symbol: str = "",
    initial_equity: float = 10_000.0,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """Replay ``bars`` through ``strategy`` and report what happened.

    Args:
        bars: canonical bars, as returned by
            :func:`ai_finance.data.store.load_bars`. At least two are needed,
            since a decision on the last bar has nothing to fill into.
        strategy: anything implementing :class:`~ai_finance.strategy.base.Strategy`.
        symbol: label for reporting. Defaults to the strategy's view of it.
        initial_equity: starting cash.
        costs: cost model. Defaults to :class:`CostModel`'s pessimistic defaults.
        limits: risk limits. Defaults to the real ones in ``docs/RISK.md``, which
            cap a single position at 25% of equity — so a buy-and-hold strategy
            run with defaults will *not* reproduce the asset's return, and should
            not.
        liquidate_at_end: close any open position at the final close. Off by
            default, because a real strategy still holds what it holds; the
            final position stays marked to market and only the entry cost has
            been paid.

    Returns:
        A :class:`~ai_finance.backtest.metrics.BacktestResult`.
    """
    assert_schema(bars)
    if len(bars) < 2:
        raise ValueError(
            f"need at least 2 bars (a decision on the last bar has nothing to "
            f"fill into), got {len(bars)}"
        )
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")

    costs = costs if costs is not None else CostModel()
    limits = limits if limits is not None else RiskLimits()
    symbol = symbol or "UNKNOWN"

    arrays = {name: bars[name].to_numpy(dtype=np.float64) for name in _FIELDS}
    trades_array = bars["trades"].to_numpy(dtype=np.int64)
    open_time = bars["open_time"].to_numpy()
    close_time = bars["close_time"].to_numpy()
    n = len(bars)

    portfolio = Portfolio(cash=initial_equity)
    risk = RiskEngine(limits=limits)
    execution = SimulatedExecution(costs)
    ledger = TradeLedger()

    equity_values: list[float] = []
    weight_values: list[float] = []
    fills = []
    risk_adjustments: list[str] = []
    rejected: list[str] = []

    pending_weight: float | None = None
    pending_reason = ""

    def rebalance(target_weight: float, reference_price: float, timestamp, reason: str, index: int):
        """Move the position toward ``target_weight`` at ``reference_price``."""
        if not math.isfinite(reference_price) or reference_price <= 0:
            log.warning(
                "skipping rebalance at %s: bad reference price %r", timestamp, reference_price
            )
            return

        equity = portfolio.equity(reference_price)
        if equity <= 0:
            rejected.append("account equity is zero or negative; no further trading")
            return

        target_units = target_weight * equity / reference_price
        delta = target_units - portfolio.units

        # You cannot invest 100% of equity *and* pay the fee out of the same
        # money. Clamp a buy to what the cash balance can actually settle.
        if delta > 0:
            affordable = portfolio.cash / (
                costs.fill_price(reference_price, 1.0) * (1.0 + costs.fee_rate)
            )
            if delta > affordable:
                risk_adjustments.append("buy reduced to available cash (fees are not free)")
                delta = max(0.0, affordable)

        if delta == 0.0:
            return

        notional = abs(delta) * reference_price
        permitted, why = risk.permits_order(notional, timestamp)
        if not permitted:
            rejected.append(why)
            return

        fill = execution.submit(
            Order(
                symbol=symbol,
                delta_units=delta,
                reference_price=reference_price,
                timestamp=timestamp,
                reason=reason,
            )
        )
        if fill is None:
            return

        risk.record_order(timestamp)
        portfolio.apply(fill)
        ledger.record(fill, index)
        fills.append(fill)

    for i in range(n):
        if pending_weight is not None:
            rebalance(
                pending_weight,
                float(arrays["open"][i]),
                pd.Timestamp(open_time[i]),
                pending_reason,
                i,
            )
            pending_weight = None

        close_price = float(arrays["close"][i])
        equity = portfolio.equity(close_price)
        equity_values.append(equity)
        weight_values.append(portfolio.weight(close_price))

        bar_close_time = pd.Timestamp(close_time[i])
        risk.observe(equity, bar_close_time)

        if i == n - 1:
            # Nothing left to fill into, so no decision is taken.
            break

        state = MarketState(
            symbol=symbol,
            index=i,
            bar=Bar(
                open_time=pd.Timestamp(open_time[i]),
                close_time=bar_close_time,
                open=float(arrays["open"][i]),
                high=float(arrays["high"][i]),
                low=float(arrays["low"][i]),
                close=close_price,
                volume=float(arrays["volume"][i]),
                trades=int(trades_array[i]),
            ),
            history=BarWindow(arrays, i),
            equity=equity,
            position_weight=portfolio.weight(close_price),
        )

        # The strategy is consulted even while halted, so its internal state
        # (moving averages and the like) has no hole in it when trading resumes.
        signal = strategy.on_bar(state)

        if risk.is_halted(bar_close_time):
            pending_weight = 0.0 if portfolio.units != 0.0 else None
            pending_reason = f"risk halt: {risk.halt_reason}"
        elif signal is not None:
            decision = risk.apply(signal, bar_close_time)
            pending_weight = decision.target_weight
            pending_reason = signal.reason
            risk_adjustments.extend(decision.adjustments)

    if liquidate_at_end and portfolio.units != 0.0:
        last_close = float(arrays["close"][n - 1])
        rebalance(0.0, last_close, pd.Timestamp(close_time[n - 1]), "liquidate at end", n - 1)
        equity_values[-1] = portfolio.equity(last_close)
        weight_values[-1] = portfolio.weight(last_close)

    index = pd.DatetimeIndex(close_time[: len(equity_values)], name="close_time")
    first_open = float(arrays["open"][0])

    return BacktestResult(
        symbol=symbol,
        strategy_name=getattr(strategy, "name", type(strategy).__name__),
        costs=costs,
        initial_equity=initial_equity,
        equity_curve=pd.Series(equity_values, index=index, name="equity"),
        weight_curve=pd.Series(weight_values, index=index, name="weight"),
        benchmark_curve=pd.Series(
            initial_equity * arrays["close"][: len(equity_values)] / first_open,
            index=index,
            name="benchmark",
        ),
        trades=ledger.closed,
        fills=fills,
        risk_adjustments=risk_adjustments,
        rejected_orders=rejected,
        halt_reason=risk.halt_reason if risk.halted_permanently else "",
        liquidated_at_end=liquidate_at_end,
    )
