"""Backtest results and the metrics computed from them.

Two rules, both from ``docs/PLAN.md``:

1. **The benchmark is always reported.** A strategy returning 40% in a year BTC
   returned 90% destroyed value, and a report that shows only the 40% is
   misleading by omission.
2. **Costs are always reported.** Total fees paid, and the drag as a fraction of
   starting capital, next to the returns they were subtracted from.

Sharpe is computed from **daily** returns, not from bar returns. Annualising the
standard deviation of 1-minute returns produces a number that is dominated by
microstructure noise and is not comparable to any published figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.ledger import Trade
from ai_finance.execution.base import Fill

#: Crypto trades every day, so a year is 365 periods rather than 252.
DAYS_PER_YEAR = 365

#: Annualising a return over a very short window produces nonsense.
MIN_DAYS_TO_ANNUALISE = 7


@dataclass
class BacktestResult:
    """Everything one backtest produced. Immutable in practice."""

    symbol: str
    strategy_name: str
    costs: CostModel
    initial_equity: float
    equity_curve: pd.Series
    weight_curve: pd.Series
    benchmark_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    risk_adjustments: list[str] = field(default_factory=list)
    rejected_orders: list[str] = field(default_factory=list)
    halt_reason: str = ""
    liquidated_at_end: bool = False

    # ---------------- returns ----------------

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0

    @property
    def span_days(self) -> float:
        delta = self.equity_curve.index[-1] - self.equity_curve.index[0]
        return delta.total_seconds() / 86_400

    @property
    def annualized_return(self) -> float:
        """CAGR. ``nan`` when the window is too short for the number to mean anything."""
        if self.span_days < MIN_DAYS_TO_ANNUALISE or self.final_equity <= 0:
            return float("nan")
        years = self.span_days / DAYS_PER_YEAR
        return (self.final_equity / self.initial_equity) ** (1.0 / years) - 1.0

    @property
    def daily_returns(self) -> pd.Series:
        return _daily_returns(self.equity_curve)

    #: Daily return standard deviations below this are treated as zero.
    #:
    #: A perfectly steady equity curve has a variance of pure float noise
    #: (~1e-18), which divides into a Sharpe of 1e15. That number is not a
    #: brilliant strategy, it is a degenerate curve, and printing it invites
    #: exactly the wrong conclusion. Real daily volatility is ~1e-2.
    MIN_MEANINGFUL_VOLATILITY = 1e-12

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe from daily returns, zero risk-free rate.

        ``nan`` when the curve has no meaningful variance to divide by.
        """
        return _sharpe_of(self.equity_curve, self.MIN_MEANINGFUL_VOLATILITY)

    @property
    def benchmark_sharpe(self) -> float:
        """The benchmark's own Sharpe, on the same days.

        Without this there is no risk-adjusted comparison, only a return
        comparison — and a strategy can "beat" a benchmark that fell 86% while
        still losing 30% of the account. Both numbers are needed to say anything
        useful.
        """
        return _sharpe_of(self.benchmark_curve, self.MIN_MEANINGFUL_VOLATILITY)

    @property
    def benchmark_max_drawdown(self) -> float:
        return _max_drawdown_of(self.benchmark_curve)

    @property
    def beat_benchmark_risk_adjusted(self) -> bool:
        """Higher Sharpe than buy-and-hold. The Phase 2 gate's actual question.

        A strategy that merely loses less than a falling market has not found an
        edge; it has found cash. Risk-adjusted comparison is what separates the
        two.
        """
        mine, theirs = self.sharpe, self.benchmark_sharpe
        if not (np.isfinite(mine) and np.isfinite(theirs)):
            return False
        return mine > theirs

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough fall, as a negative fraction."""
        return _max_drawdown_of(self.equity_curve)

    # ---------------- benchmark ----------------

    @property
    def benchmark_return(self) -> float:
        """Buy-and-hold over the same window, **cost-free**.

        Cost-free on purpose: it makes the bar deliberately hard to clear, and
        any strategy that cannot beat holding the asset is not worth running.

        Measured from the **first bar's open**, which is the earliest price an
        investor could have paid. Dividing by the curve's own first value would
        silently measure from the first bar's *close* instead and quietly drop
        bar zero's move from the comparison.
        """
        return float(self.benchmark_curve.iloc[-1]) / self.initial_equity - 1.0

    @property
    def excess_return(self) -> float:
        """The only number that says whether the work was worth doing."""
        return self.total_return - self.benchmark_return

    #: Excess returns inside this band count as matching the benchmark rather
    #: than beating or losing to it. One basis point is noise, not a result.
    MATCHED_TOLERANCE = 1e-4

    @property
    def beat_benchmark(self) -> bool:
        return self.excess_return > self.MATCHED_TOLERANCE

    @property
    def benchmark_verdict(self) -> str:
        if abs(self.excess_return) <= self.MATCHED_TOLERANCE:
            return "MATCHED"
        return "BEAT" if self.beat_benchmark else "LOST TO"

    # ---------------- trades ----------------

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.is_win]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if not t.is_win]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return float("nan")
        return len(self.wins) / len(self.trades)

    @property
    def avg_win(self) -> float:
        return float(np.mean([t.pnl for t in self.wins])) if self.wins else 0.0

    @property
    def avg_loss(self) -> float:
        return float(np.mean([t.pnl for t in self.losses])) if self.losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit over gross loss. Above 1 means the wins outweigh the losses."""
        gross_loss = -sum(t.pnl for t in self.losses)
        if gross_loss == 0:
            return float("inf") if self.wins else float("nan")
        return sum(t.pnl for t in self.wins) / gross_loss

    # ---------------- costs ----------------

    @property
    def total_fees(self) -> float:
        return sum(f.fee for f in self.fills)

    @property
    def total_price_concession(self) -> float:
        """Everything lost to spread and slippage rather than to fees."""
        return sum(f.price_concession for f in self.fills)

    @property
    def total_costs(self) -> float:
        return self.total_fees + self.total_price_concession

    @property
    def cost_drag(self) -> float:
        """Total costs as a fraction of starting capital.

        The number that kills high-frequency retail strategies. Compare it to
        :attr:`total_return` — if it is the larger of the two, the strategy is
        paying the exchange to take risk on its behalf.
        """
        return self.total_costs / self.initial_equity

    @property
    def turnover(self) -> float:
        """Total notional traded, as a multiple of starting capital."""
        return sum(f.notional for f in self.fills) / self.initial_equity

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def exposure(self) -> float:
        """Fraction of bars spent holding a non-zero position."""
        if self.weight_curve.empty:
            return 0.0
        return float((self.weight_curve.abs() > 1e-12).mean())

    # ---------------- reporting ----------------

    def to_text(self) -> str:
        """The report. Benchmark and costs always visible."""
        lines = [
            f"Backtest: {self.strategy_name} on {self.symbol}",
            "=" * 62,
            f"window       {self.equity_curve.index[0]:%Y-%m-%d %H:%M} .. "
            f"{self.equity_curve.index[-1]:%Y-%m-%d %H:%M} UTC "
            f"({self.span_days:.1f} days, {len(self.equity_curve):,} bars)",
            f"costs        {self.costs.describe()}",
            "",
            f"equity       {self.initial_equity:,.2f} -> {self.final_equity:,.2f}",
            f"return       {self.total_return * 100:+.2f}%",
            f"benchmark    {self.benchmark_return * 100:+.2f}%   (buy and hold, cost-free)",
            f"excess       {self.excess_return * 100:+.2f}%   "
            f"<- {self.benchmark_verdict} the benchmark",
            "",
            f"annualized   {_pct(self.annualized_return)}",
            f"sharpe       {_num(self.sharpe)}   vs benchmark {_num(self.benchmark_sharpe)}"
            f"   <- {'BEAT' if self.beat_benchmark_risk_adjusted else 'did NOT beat'}"
            f" risk-adjusted",
            f"max dd       {self.max_drawdown * 100:.2f}%"
            f"   vs benchmark {self.benchmark_max_drawdown * 100:.2f}%",
            f"exposure     {self.exposure * 100:.1f}% of bars in a position",
            "",
            f"trades       {self.n_trades:,}  ({self.n_fills:,} fills)",
            f"win rate     {_rate(self.win_rate)}",
            f"avg win      {self.avg_win:+,.2f}",
            f"avg loss     {self.avg_loss:+,.2f}",
            f"profit fctr  {_num(self.profit_factor)}",
            "",
            f"fees paid    {self.total_fees:,.2f}",
            f"spread+slip  {self.total_price_concession:,.2f}",
            f"total cost   {self.total_costs:,.2f}  "
            f"({self.cost_drag * 100:.2f}% of starting capital)",
            f"turnover     {self.turnover:.1f}x starting capital",
        ]

        if self.halt_reason:
            lines += ["", f"RISK HALT    {self.halt_reason}"]
        if self.risk_adjustments:
            lines += ["", f"risk adjusted {len(self.risk_adjustments):,} signal(s), e.g.:"]
            lines += [f"  - {a}" for a in _first_distinct(self.risk_adjustments, 3)]
        if self.rejected_orders:
            lines += ["", f"rejected {len(self.rejected_orders):,} order(s), e.g.:"]
            lines += [f"  - {r}" for r in _first_distinct(self.rejected_orders, 3)]

        return "\n".join(lines)


def _daily_returns(curve: pd.Series) -> pd.Series:
    daily = curve.resample("1D").last().dropna()
    return daily.pct_change().dropna()


def _sharpe_of(curve: pd.Series, min_volatility: float) -> float:
    returns = _daily_returns(curve)
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if not np.isfinite(std) or std < min_volatility:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(DAYS_PER_YEAR))


def _max_drawdown_of(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1.0).min())


def _pct(value: float) -> str:
    """A signed percentage, for quantities that can go either way."""
    return "n/a" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def _rate(value: float) -> str:
    """An unsigned percentage, for quantities bounded in [0, 1]."""
    return "n/a" if not np.isfinite(value) else f"{value * 100:.1f}%"


def _num(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2f}"


def _first_distinct(items: list[str], limit: int) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
        if len(seen) == limit:
            break
    return seen
