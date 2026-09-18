from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.ledger import Trade
from ai_finance.backtest.metrics import BacktestResult
from ai_finance.execution.base import Fill

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def daily_index(n):
    return pd.DatetimeIndex([T0 + pd.Timedelta(days=i) for i in range(n)], name="close_time")


def result_from(equity, *, weights=None, benchmark=None, trades=None, fills=None, initial=None):
    index = daily_index(len(equity))
    initial = initial if initial is not None else equity[0]
    return BacktestResult(
        symbol="BTCUSDT",
        strategy_name="test",
        costs=CostModel(),
        initial_equity=initial,
        equity_curve=pd.Series(equity, index=index, dtype=float),
        weight_curve=pd.Series(
            weights if weights is not None else [1.0] * len(equity), index=index, dtype=float
        ),
        benchmark_curve=pd.Series(
            benchmark if benchmark is not None else equity, index=index, dtype=float
        ),
        trades=trades or [],
        fills=fills or [],
    )


def trade(pnl, notional=100.0):
    return Trade(
        symbol="BTCUSDT",
        direction="long",
        open_time=T0,
        close_time=T0 + pd.Timedelta(days=1),
        entry_notional=notional,
        pnl=pnl,
        fees=0.1,
        price_concession=0.05,
        n_fills=2,
        bars_held=1,
    )


class TestReturns:
    def test_total_return(self):
        assert result_from([100.0, 110.0]).total_return == pytest.approx(0.10)

    def test_annualized_return_over_a_year(self):
        equity = [100.0] * 365 + [120.0]
        assert result_from(equity).annualized_return == pytest.approx(0.20, rel=1e-3)

    def test_annualizing_a_short_window_is_refused(self):
        """A 3-day backtest annualized would read like a fortune."""
        result = result_from([100.0, 100.0, 130.0])
        assert np.isnan(result.annualized_return)
        assert "n/a" in result.to_text()

    def test_max_drawdown_is_measured_from_the_peak(self):
        result = result_from([100.0, 150.0, 75.0, 120.0])
        assert result.max_drawdown == pytest.approx(-0.50)

    def test_flat_curve_has_no_drawdown(self):
        assert result_from([100.0] * 10).max_drawdown == 0.0

    def test_a_perfectly_steady_curve_has_no_sharpe_rather_than_a_huge_one(self):
        """Float noise in a zero-variance curve would otherwise read as 1e15."""
        equity = [100.0 * (1.01**i) for i in range(30)]
        result = result_from(equity)

        assert np.isnan(result.sharpe)
        assert "sharpe       n/a" in result.to_text()

    def test_sharpe_of_a_noisy_curve_is_finite(self):
        rng = np.random.default_rng(0)
        equity = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, 200))
        result = result_from(list(equity))
        assert np.isfinite(result.sharpe)
        assert 0 < result.sharpe < 10

    def test_sharpe_needs_at_least_two_daily_returns(self):
        assert np.isnan(result_from([100.0, 101.0]).sharpe)


class TestBenchmark:
    def test_benchmark_measured_against_starting_equity(self):
        """Regression: it must not be measured from the curve's own first value."""
        # Benchmark bought at an open of 100 while bar 0 closed at 105.
        benchmark = [105.0, 110.0, 120.0]
        result = result_from([100.0, 100.0, 100.0], benchmark=benchmark, initial=100.0)

        assert result.benchmark_return == pytest.approx(0.20)

    def test_excess_return_is_strategy_minus_benchmark(self):
        result = result_from([100.0, 140.0], benchmark=[100.0, 120.0], initial=100.0)
        assert result.total_return == pytest.approx(0.40)
        assert result.benchmark_return == pytest.approx(0.20)
        assert result.excess_return == pytest.approx(0.20)
        assert result.beat_benchmark
        assert result.benchmark_verdict == "BEAT"

    def test_losing_to_the_benchmark_is_reported_as_such(self):
        result = result_from([100.0, 140.0], benchmark=[100.0, 190.0], initial=100.0)
        assert not result.beat_benchmark
        assert result.benchmark_verdict == "LOST TO"
        assert "LOST TO the benchmark" in result.to_text()

    def test_a_basis_point_of_excess_counts_as_matching(self):
        result = result_from([100.0, 120.001], benchmark=[100.0, 120.0], initial=100.0)
        assert result.benchmark_verdict == "MATCHED"
        assert not result.beat_benchmark


class TestTradeStatistics:
    def test_win_rate_and_averages(self):
        trades = [trade(10.0), trade(20.0), trade(-5.0), trade(-25.0)]
        result = result_from([100.0, 100.0], trades=trades)

        assert result.n_trades == 4
        assert result.win_rate == pytest.approx(0.5)
        assert result.avg_win == pytest.approx(15.0)
        assert result.avg_loss == pytest.approx(-15.0)
        assert result.profit_factor == pytest.approx(1.0)

    def test_profit_factor_above_one_means_wins_outweigh_losses(self):
        result = result_from([100.0, 100.0], trades=[trade(30.0), trade(-10.0)])
        assert result.profit_factor == pytest.approx(3.0)

    def test_a_breakeven_trade_counts_as_a_loss(self):
        """Breakeven after costs is not a win, and pretending otherwise flatters."""
        result = result_from([100.0, 100.0], trades=[trade(0.0)])
        assert result.win_rate == 0.0

    def test_no_trades_gives_undefined_statistics_not_zero(self):
        result = result_from([100.0, 100.0])
        assert result.n_trades == 0
        assert np.isnan(result.win_rate)
        assert np.isnan(result.profit_factor)

    def test_all_wins_gives_infinite_profit_factor(self):
        result = result_from([100.0, 100.0], trades=[trade(5.0)])
        assert result.profit_factor == float("inf")
        assert "n/a" in result.to_text() or "inf" in result.to_text()


class TestCosts:
    def test_cost_decomposition_and_drag(self):
        fills = [
            Fill(T0, "BTCUSDT", 1.0, 100.0, 100.1, fee=0.1),
            Fill(T0, "BTCUSDT", -1.0, 110.0, 109.9, fee=0.11),
        ]
        result = result_from([1000.0, 1000.0], fills=fills, initial=1000.0)

        assert result.total_fees == pytest.approx(0.21)
        assert result.total_price_concession == pytest.approx(0.1 + 0.1)
        assert result.total_costs == pytest.approx(0.41)
        assert result.cost_drag == pytest.approx(0.00041)
        assert result.n_fills == 2

    def test_turnover_counts_traded_notional_against_capital(self):
        fills = [Fill(T0, "BTCUSDT", 10.0, 100.0, 100.0, fee=0.0)] * 3
        result = result_from([1000.0, 1000.0], fills=fills, initial=1000.0)
        assert result.turnover == pytest.approx(3.0)

    def test_exposure_counts_bars_holding_a_position(self):
        result = result_from([100.0] * 4, weights=[0.0, 0.5, 0.5, 0.0])
        assert result.exposure == pytest.approx(0.5)

    def test_zero_exposure_when_never_invested(self):
        assert result_from([100.0] * 4, weights=[0.0] * 4).exposure == 0.0


class TestReport:
    def test_report_shows_the_numbers_that_matter(self):
        text = result_from(
            [1000.0, 1100.0], benchmark=[1000.0, 1050.0], trades=[trade(100.0)], initial=1000.0
        ).to_text()

        for expected in ("return", "benchmark", "excess", "max dd", "total cost", "turnover"):
            assert expected in text

    def test_report_surfaces_a_risk_halt(self):
        result = result_from([100.0, 90.0])
        result.halt_reason = "max drawdown 16.0% >= 15.0%"
        assert "RISK HALT" in result.to_text()

    def test_report_lists_distinct_risk_adjustments_only(self):
        result = result_from([100.0, 100.0])
        result.risk_adjustments = ["capped at 0.25"] * 50 + ["shorting disabled"]
        text = result.to_text()

        assert "risk adjusted 51 signal(s)" in text
        assert text.count("capped at 0.25") == 1
