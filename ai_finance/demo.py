"""A guided end-to-end run of the whole pipeline.

Answers "does this thing actually work?" in one command, with no exchange
account, no API key and no network. It generates a synthetic price series,
stores it, checks its quality, runs the cost demonstration, walks the baselines
forward, and fits a model — narrating each step.

**What the demo can and cannot tell you.** It exercises every component and
shows the shape of the output, so it answers "is the machinery sound?". It says
nothing whatever about whether Bitcoin is predictable, because the prices are a
random walk generated on the spot. Any strategy that appeared to work here would
be a bug, not a discovery. The real question needs real data, and the last
section prints exactly how to run it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import click
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.data.fetch import fetch_history
from ai_finance.data.quality import check_bars
from ai_finance.data.sources import SyntheticSource
from ai_finance.data.store import load_bars
from ai_finance.research.ml_walkforward import run_ml_walk_forward
from ai_finance.research.registry import Registry, expected_max_sharpe
from ai_finance.research.walkforward import run_walk_forward
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import STRATEGY_FACTORIES, RandomStrategy

DEMO_SYMBOL = "DEMO"
DEMO_START = "2021-01-01"
DEMO_END = "2023-12-31 23:59"


@dataclass
class DemoResult:
    """What the demo produced, so the caller can report or chart it."""

    bars: pd.DataFrame
    interval: str
    walkforward: list
    model: object | None
    elapsed: float


def _step(number: int, title: str) -> None:
    click.echo("")
    click.secho(f"[{number}] {title}", fg="cyan", bold=True)
    click.secho("-" * 62, fg="cyan")


def run_demo(
    *,
    interval: str = "4h",
    seed: int = 7,
    initial_equity: float = 5_000.0,
    quick: bool = False,
    registry: Registry | None = None,
    start: str = DEMO_START,
    end: str = DEMO_END,
    train_bars: int = 600,
    test_bars: int = 200,
) -> DemoResult:
    """Run every stage of the pipeline and narrate it.

    Args:
        start, end: the period to generate. Shorter ranges make the demo quick
            enough to run in a test; the default covers three years, which is
            long enough for the walk-forward to have something to say.
    """
    started = time.monotonic()
    costs = CostModel()
    limits = RiskLimits(max_position_weight=0.25)

    click.secho("=" * 62, fg="cyan")
    click.secho("AI_Finance end-to-end demo", fg="cyan", bold=True)
    click.secho("=" * 62, fg="cyan")
    click.echo(
        "Synthetic prices, generated here. Every number below is real output\n"
        "from the real pipeline — but the market it is trading does not exist,\n"
        "so nothing here says whether Bitcoin is predictable."
    )

    # ---------------------------------------------------------------- 1
    _step(1, "Fetch and store bars")
    source = SyntheticSource(
        epoch_ms=int(pd.Timestamp(start, tz="UTC").timestamp() * 1000),
        seed=seed,
        annual_vol=0.55,
    )
    result = fetch_history(source, DEMO_SYMBOL, start, end, limit=1000)
    click.echo(result.summary())

    bars = load_bars(DEMO_SYMBOL, interval=interval)
    click.echo(f"Resampled on read to {len(bars):,} {interval} bars.")

    # ---------------------------------------------------------------- 2
    _step(2, "Check the data before trusting it")
    report = check_bars(bars, DEMO_SYMBOL, interval)
    click.echo(f"coverage {report.coverage * 100:.2f}%, errors: {report.errors or 'none'}")
    click.echo("Bad data makes beautiful backtests, so this runs before anything else.")

    # ---------------------------------------------------------------- 3
    _step(3, "What trading frequency costs")
    click.echo("One strategy with no edge, one price series. Only the cadence changes.\n")
    click.echo(f"{'cadence':>8} {'fills':>8} {'cost drag':>11} {'return':>9}   final equity")
    open_limits = RiskLimits(
        max_position_weight=0.25,
        min_order_notional=0.0,
        max_drawdown=1e9,
        max_daily_loss=1e9,
        max_orders_per_hour=10**9,
    )
    for cadence in ("1m", "1h", "4h", "1d"):
        series = load_bars(DEMO_SYMBOL, interval=cadence)
        run = run_backtest(
            series,
            RandomStrategy(seed=1, every_n_bars=1),
            symbol=DEMO_SYMBOL,
            initial_equity=initial_equity,
            costs=costs,
            limits=open_limits,
        )
        click.echo(
            f"{cadence:>8} {run.n_fills:>8,} {run.cost_drag * 100:>10.1f}%"
            f" {run.total_return * 100:>8.1f}%   ${run.final_equity:>10,.2f}"
        )
    click.echo("\nSame signal quality throughout. The fee schedule does the rest.")

    # ---------------------------------------------------------------- 4
    _step(4, "Walk the classic baselines forward")
    click.echo("Fit parameters on the past, measure on the next unseen window, repeat.\n")
    names = ["ma-crossover", "breakout"] if quick else sorted(STRATEGY_FACTORIES)
    walk_results = []
    for name in names:
        if len(bars) < train_bars + test_bars + 6:
            click.secho("  not enough bars for a walk-forward window; skipping", fg="yellow")
            break
        walk = run_walk_forward(
            bars,
            name,
            train_bars=train_bars,
            test_bars=test_bars,
            embargo_bars=6,
            symbol=DEMO_SYMBOL,
            interval=interval,
            initial_equity=initial_equity,
            costs=costs,
            limits=limits,
            registry=registry,
        )
        walk_results.append(walk)
        oos = walk.oos
        click.echo(
            f"  {name:<20} sharpe {_num(oos.sharpe):>6} vs {_num(oos.benchmark_sharpe):>6}"
            f"   return {oos.total_return * 100:+7.1f}%   churn {walk.param_turnover * 100:3.0f}%"
        )

    if walk_results:
        trials = sum(w.variants_tested for w in walk_results)
        days = max(len(w.oos.daily_returns) for w in walk_results)
        floor = expected_max_sharpe(trials, days) if days >= 2 else 0.0
        click.echo(
            f"\n{trials:,} parameter runs over {days:,} days. The best of that many\n"
            f"worthless strategies would show a Sharpe of {floor:.2f} by luck alone —\n"
            "so that, not zero, is the bar."
        )
        click.echo("'churn' is how often the winning parameters changed. High churn is")
        click.echo("what fitting noise looks like.")

    # ---------------------------------------------------------------- 5
    _step(5, "Fit a model")
    model_result = None
    try:
        model_result = run_ml_walk_forward(
            bars,
            horizon=6,
            model_kind="ridge",
            train_size=train_bars,
            test_size=test_bars,
            embargo=12,
            symbol=DEMO_SYMBOL,
            interval=interval,
            initial_equity=initial_equity,
            costs=costs,
            limits=limits,
            registry=registry,
        )
    except ValueError as exc:
        click.secho(f"  skipped: {exc}", fg="yellow")
    else:
        click.echo("Point-in-time check passed: no feature depends on future data.")
        click.echo(
            f"  accuracy    {model_result.accuracy * 100:.2f}%  z={model_result.accuracy_z:.2f}"
            f"   ({'above' if model_result.accuracy_z > 2 else 'NOT above'} chance)"
        )
        click.echo(f"  info coef   {model_result.ic:.3f}")
        click.echo(
            f"  sharpe      {_num(model_result.oos.sharpe)} vs benchmark "
            f"{_num(model_result.oos.benchmark_sharpe)}"
        )
        click.echo(f"  cost drag   {model_result.oos.cost_drag * 100:.1f}% of starting capital")
        click.echo(
            f"  threshold   {model_result.policy.threshold:.2%} predicted move required "
            "before it will trade"
        )

    # ---------------------------------------------------------------- 6
    elapsed = time.monotonic() - started
    _step(6, "What this did and did not show")
    click.echo(
        "Shown: every stage runs, the accounting reconciles, the cost model\n"
        "bites, and the validation refuses to call noise an edge.\n"
    )
    click.secho(
        "NOT shown: whether any of this makes money. These prices are\n"
        "a random walk with no edge to find, so a strategy that looked good\n"
        "here would be a bug. For a real answer, run it on real data:",
        fg="yellow",
    )
    click.echo(
        "\n  aifin fetch --symbol BTCUSDT --start 2017-08-17\n"
        "  aifin quality --symbol BTCUSDT\n"
        "  aifin walkforward --symbol BTCUSDT --interval 4h\n"
        "  aifin train --symbol BTCUSDT --interval 4h --model all\n"
    )
    click.echo(
        "That needs outbound access to api.binance.com, which this sandbox blocks\n"
        "by policy — so it has to run on your own machine."
    )
    click.secho(f"\nDemo finished in {elapsed:.0f}s.", fg="green")

    return DemoResult(
        bars=bars,
        interval=interval,
        walkforward=walk_results,
        model=model_result,
        elapsed=elapsed,
    )


def _num(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"
