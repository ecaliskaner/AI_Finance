"""Walk-forward validation.

Fit on the past, test on the future, roll forward, repeat. Report only what
happened on data the parameters had never seen.

This exists because the obvious alternative is fatally flawed. Picking the
moving-average lengths that worked best over 2019-2024 and then reporting how
well they did over 2019-2024 measures nothing except your ability to read a
chart you have already seen. Standard k-fold cross-validation has the same
problem in a less obvious costume: shuffling time series into folds trains the
model on Thursday to predict Wednesday.

So each window here does the whole job honestly:

1. **Train.** Grid-search parameters on a window of history.
2. **Embargo.** Skip a gap, so the first test bar does not share information
   with the last training bar.
3. **Test.** Run the single chosen parameter set on the next unseen window.
4. **Roll.** Advance by one test window and do it again.

The out-of-sample equity curve is the test windows chained together. It is the
only curve worth looking at. Every training run is also written to the
experiment registry, because the number of parameter sets tried is exactly what
determines how impressive the winner has to be to mean anything.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.backtest.ledger import Trade
from ai_finance.backtest.metrics import BacktestResult
from ai_finance.execution.base import Fill
from ai_finance.research.registry import Registry
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import (
    PARAM_GRIDS,
    STRATEGY_FACTORIES,
    warmup_bars_for,
)

log = logging.getLogger(__name__)

SELECTION_METRICS = ("sharpe", "excess_return", "total_return")


@contextlib.contextmanager
def _quiet_risk_engine() -> Iterator[None]:
    """Silence per-run risk-halt warnings for the duration of a search.

    A grid search runs hundreds of backtests, and a halt inside a training
    variant that is about to be discarded is not news — at one warning each it
    buries everything worth reading. Nothing is lost: the counts surface as
    :attr:`WalkForwardResult.halted_windows`.
    """
    risk_logger = logging.getLogger("ai_finance.risk.engine")
    previous = risk_logger.level
    risk_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        risk_logger.setLevel(previous)


@dataclass(frozen=True)
class Split:
    """One train/test division, in bar indices."""

    index: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive
    warmup: int

    @property
    def test_run_start(self) -> int:
        """Where the test backtest actually starts, including warm-up bars.

        The strategy needs history before it can produce a signal. Those bars
        come from before the test window and are *not* measured — they exist so
        the strategy is not blind for the first stretch of every test period,
        which is what a live system would actually have.
        """
        return self.test_start - self.warmup


def make_splits(
    n_bars: int,
    *,
    train_bars: int,
    test_bars: int,
    warmup_bars: int = 0,
    embargo_bars: int = 0,
    anchored: bool = False,
    max_splits: int | None = None,
) -> list[Split]:
    """Divide ``n_bars`` into rolling train/test windows.

    Args:
        train_bars: length of each training window.
        test_bars: length of each test window, and the step between splits.
        warmup_bars: history a strategy needs before its first signal.
        embargo_bars: gap between the end of training and the start of testing.
            Guards against a training bar and a test bar sharing information —
            which for an overlapping-return label they literally do.
        anchored: expand the training window from the start instead of rolling
            a fixed-length one. Uses more data; adapts to regime change more
            slowly.
        max_splits: stop after this many.

    Returns:
        Splits in chronological order. Empty if the data cannot accommodate even
        one window.
    """
    if train_bars < 1 or test_bars < 1:
        raise ValueError("train_bars and test_bars must be positive")
    if warmup_bars < 0 or embargo_bars < 0:
        raise ValueError("warmup_bars and embargo_bars cannot be negative")
    if train_bars <= warmup_bars:
        raise ValueError(
            f"train_bars ({train_bars}) must exceed warmup_bars ({warmup_bars}), "
            "or the strategy is blind for the whole training window"
        )

    splits: list[Split] = []
    start = 0
    while True:
        train_end = start + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars
        if test_end > n_bars:
            break
        splits.append(
            Split(
                index=len(splits),
                train_start=0 if anchored else start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                warmup=warmup_bars,
            )
        )
        if max_splits is not None and len(splits) >= max_splits:
            break
        start += test_bars
    return splits


@dataclass(frozen=True)
class SplitOutcome:
    """What one window produced."""

    split: Split
    chosen_params: dict[str, Any]
    train_score: float
    variants_tried: int
    test_result: BacktestResult
    oos_return: float

    @property
    def test_window(self) -> str:
        index = self.test_result.equity_curve.index
        return f"{index[0]:%Y-%m-%d} .. {index[-1]:%Y-%m-%d}"


@dataclass
class WalkForwardResult:
    """Out-of-sample performance, chained across every test window."""

    strategy_name: str
    symbol: str
    interval: str
    outcomes: list[SplitOutcome]
    oos: BacktestResult
    variants_tested: int
    selection_metric: str
    params_by_split: list[dict[str, Any]] = field(default_factory=list)
    undecided_splits: int = 0

    @property
    def n_splits(self) -> int:
        return len(self.outcomes)

    @property
    def param_turnover(self) -> float:
        """Fraction of windows where the winning parameters changed.

        A diagnostic for overfitting the training window. If the best moving
        average is 20/50 one quarter and 40/200 the next, the search is fitting
        noise, and neither answer should be trusted out of sample.
        """
        if len(self.params_by_split) < 2:
            return 0.0
        changes = sum(
            1
            for previous, current in itertools.pairwise(self.params_by_split)
            if previous != current
        )
        return changes / (len(self.params_by_split) - 1)

    @property
    def windows_beating_benchmark(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.test_result.excess_return > 0)

    @property
    def halted_windows(self) -> int:
        """Test windows where the drawdown limit stopped trading permanently."""
        return sum(1 for outcome in self.outcomes if outcome.test_result.halt_reason)

    def to_text(self) -> str:
        lines = [
            f"Walk-forward: {self.strategy_name} on {self.symbol} ({self.interval})",
            "=" * 62,
            f"splits        {self.n_splits}  "
            f"({self.variants_tested:,} parameter runs, selected by {self.selection_metric})",
            f"param churn   {self.param_turnover * 100:.0f}% of windows changed parameters",
            f"windows won   {self.windows_beating_benchmark}/{self.n_splits} beat buy-and-hold",
            f"risk halts    {self.halted_windows}/{self.n_splits} windows hit the drawdown limit",
            *(
                [
                    f"UNDECIDED     {self.undecided_splits}/{self.n_splits} windows could not "
                    f"score any variant — parameters there are arbitrary"
                ]
                if self.undecided_splits
                else []
            ),
            "",
            "OUT OF SAMPLE (the only numbers that count)",
            "-" * 62,
        ]
        lines += self.oos.to_text().splitlines()[2:]
        return "\n".join(lines)


def run_walk_forward(
    bars: pd.DataFrame,
    strategy_name: str,
    *,
    param_grid: dict[str, list] | None = None,
    train_bars: int,
    test_bars: int,
    embargo_bars: int = 0,
    anchored: bool = False,
    max_splits: int | None = None,
    symbol: str = "",
    interval: str = "",
    initial_equity: float = 10_000.0,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    selection_metric: str = "sharpe",
    registry: Registry | None = None,
) -> WalkForwardResult:
    """Grid-search on each training window, evaluate on the next unseen one.

    Returns:
        A :class:`WalkForwardResult` whose ``oos`` attribute is the chained
        out-of-sample curve. Training performance is deliberately not summarised
        anywhere: it is a selection artefact, not a result.

    Raises:
        ValueError: if the strategy is unknown, the metric is unknown, or the
            data is too short for even one split.
    """
    if strategy_name not in STRATEGY_FACTORIES:
        raise ValueError(
            f"unknown strategy {strategy_name!r}; expected one of {sorted(STRATEGY_FACTORIES)}"
        )
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"unknown selection_metric {selection_metric!r}; expected one of {SELECTION_METRICS}"
        )

    factory = STRATEGY_FACTORIES[strategy_name]
    grid = param_grid if param_grid is not None else PARAM_GRIDS[strategy_name]
    combos = _valid_combinations(factory, grid)
    if not combos:
        raise ValueError(f"no valid parameter combinations for {strategy_name}")

    costs = costs if costs is not None else CostModel()
    limits = limits if limits is not None else RiskLimits()
    warmup = max(warmup_bars_for(strategy_name, combo) for combo in combos)

    splits = make_splits(
        len(bars),
        train_bars=train_bars,
        test_bars=test_bars,
        warmup_bars=warmup,
        embargo_bars=embargo_bars,
        anchored=anchored,
        max_splits=max_splits,
    )
    if not splits:
        raise ValueError(
            f"{len(bars):,} bars is too short for train={train_bars} + "
            f"embargo={embargo_bars} + test={test_bars}"
        )

    outcomes: list[SplitOutcome] = []
    variants_tested = 0
    undecided_splits = 0
    # Each test window starts from the equity the previous one ended with, so
    # position sizes — and therefore the fees they incur — are the ones the
    # account would really have paid. Restarting every window at the original
    # capital would record costs against a balance that no longer exists.
    running_equity = initial_equity

    with _quiet_risk_engine():
        for split in splits:
            train_slice = bars.iloc[split.train_start : split.train_end]
            # Start on the first combination rather than on "nothing chosen".
            # Every variant can legitimately score -inf — a training window
            # shorter than two days has no daily returns, so every Sharpe is
            # nan — and the right response is an arbitrary-but-deterministic
            # pick plus a warning, not a crash.
            best_params: dict[str, Any] = combos[0]
            best_score = -np.inf

            for combo in combos:
                result = run_backtest(
                    train_slice,
                    factory(**combo),
                    symbol=symbol,
                    initial_equity=initial_equity,
                    costs=costs,
                    limits=limits,
                )
                variants_tested += 1
                if registry is not None:
                    registry.record_result(
                        result,
                        params=combo,
                        interval=interval,
                        kind="train",
                        note=f"split {split.index}",
                    )
                score = _score(result, selection_metric)
                if score > best_score:
                    best_score, best_params = score, combo

            if not np.isfinite(best_score):
                undecided_splits += 1
                log.warning(
                    "split %d: no variant produced a finite %s (window too short?); "
                    "falling back to %s",
                    split.index,
                    selection_metric,
                    best_params,
                )
            test_slice = bars.iloc[split.test_run_start : split.test_end]
            test_result = run_backtest(
                test_slice,
                factory(**best_params),
                symbol=symbol,
                initial_equity=running_equity,
                costs=costs,
                limits=limits,
            )
            if registry is not None:
                registry.record_result(
                    test_result,
                    params=best_params,
                    interval=interval,
                    kind="test",
                    note=f"split {split.index} out-of-sample",
                )

            measured_from = bars["close_time"].iloc[split.test_start]
            measured = _restrict(test_result, measured_from, running_equity)
            outcomes.append(
                SplitOutcome(
                    split=split,
                    chosen_params=best_params,
                    train_score=best_score,
                    variants_tried=len(combos),
                    test_result=measured,
                    oos_return=measured.total_return,
                )
            )
            running_equity = measured.final_equity

    return WalkForwardResult(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        outcomes=outcomes,
        oos=_chain(outcomes, strategy_name, symbol, costs, initial_equity),
        variants_tested=variants_tested,
        selection_metric=selection_metric,
        params_by_split=[outcome.chosen_params for outcome in outcomes],
        undecided_splits=undecided_splits,
    )


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _valid_combinations(factory, grid: dict[str, list]) -> list[dict[str, Any]]:
    """Expand a grid, dropping combinations the strategy rejects.

    ``fast >= slow`` is not a crossover, and the constructor says so. Silently
    skipping those beats either crashing or, worse, testing a nonsense variant
    and counting it as a trial.
    """
    names = sorted(grid)
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[name] for name in names)):
        combo = dict(zip(names, values, strict=True))
        try:
            factory(**combo)
        except (ValueError, TypeError) as exc:
            log.debug("skipping invalid combination %s: %s", combo, exc)
            continue
        combos.append(combo)
    return combos


def _score(result: BacktestResult, metric: str) -> float:
    value = getattr(result, metric)
    # A window with no trades has an undefined Sharpe. Treat it as the worst
    # possible choice rather than letting nan win a comparison by accident.
    return float(value) if np.isfinite(value) else -np.inf


def _restrict(
    result: BacktestResult, measured_from: pd.Timestamp, starting_equity: float
) -> BacktestResult:
    """Drop the warm-up prefix, rebasing the window to ``starting_equity``.

    The warm-up bars let the strategy build up its indicators, and the position
    it holds entering the test window is real — but the returns it earned
    getting there are not out-of-sample and must not be counted.
    """
    equity = result.equity_curve.loc[measured_from:]
    benchmark = result.benchmark_curve.loc[measured_from:]
    scale = starting_equity / float(equity.iloc[0])
    benchmark_scale = starting_equity / float(benchmark.iloc[0])

    return BacktestResult(
        symbol=result.symbol,
        strategy_name=result.strategy_name,
        costs=result.costs,
        initial_equity=starting_equity,
        equity_curve=equity * scale,
        weight_curve=result.weight_curve.loc[measured_from:],
        benchmark_curve=benchmark * benchmark_scale,
        trades=[t for t in result.trades if t.close_time >= measured_from],
        fills=[f for f in result.fills if f.timestamp >= measured_from],
        risk_adjustments=result.risk_adjustments,
        rejected_orders=result.rejected_orders,
        halt_reason=result.halt_reason,
    )


def _chain(
    outcomes: list[SplitOutcome],
    strategy_name: str,
    symbol: str,
    costs: CostModel,
    initial_equity: float,
) -> BacktestResult:
    """Stitch the test windows into one continuous out-of-sample curve."""
    equity_parts: list[pd.Series] = []
    benchmark_parts: list[pd.Series] = []
    weight_parts: list[pd.Series] = []
    trades: list[Trade] = []
    fills: list[Fill] = []

    running_benchmark = initial_equity

    for outcome in outcomes:
        result = outcome.test_result
        # Equity windows already start where the previous one ended, so they
        # concatenate as they are. The benchmark is a pure price series and is
        # chained by ratio.
        benchmark = result.benchmark_curve / result.initial_equity * running_benchmark
        equity_parts.append(result.equity_curve)
        benchmark_parts.append(benchmark)
        weight_parts.append(result.weight_curve)
        trades.extend(result.trades)
        fills.extend(result.fills)
        running_benchmark = float(benchmark.iloc[-1])

    return BacktestResult(
        symbol=symbol,
        strategy_name=f"{strategy_name} (walk-forward)",
        costs=costs,
        initial_equity=initial_equity,
        equity_curve=pd.concat(equity_parts),
        weight_curve=pd.concat(weight_parts),
        benchmark_curve=pd.concat(benchmark_parts),
        trades=trades,
        fills=fills,
    )
