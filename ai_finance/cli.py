"""Command-line entry point.

aifin fetch --symbol BTCUSDT --start 2024-01-01
aifin quality --symbol BTCUSDT
aifin info
aifin show --symbol BTCUSDT --interval 4h --tail 5
"""

from __future__ import annotations

import logging
import sys

import click
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.backtest.engine import run_backtest
from ai_finance.config import BASE_INTERVAL, INTERVAL_MS, bars_dir
from ai_finance.data.fetch import fetch_history
from ai_finance.data.quality import check_bars
from ai_finance.data.sources import BinanceSource, SyntheticSource
from ai_finance.data.store import load_bars, store_summary
from ai_finance.research.registry import Registry, expected_max_sharpe
from ai_finance.research.walkforward import SELECTION_METRICS, run_walk_forward
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import (
    STRATEGY_FACTORIES,
    AlwaysFlat,
    BuyAndHold,
    RandomStrategy,
)

SOURCES = ("binance", "synthetic")

STRATEGIES = {
    "buy-and-hold": lambda seed: BuyAndHold(),
    "always-flat": lambda seed: AlwaysFlat(),
    "random": lambda seed: RandomStrategy(seed=seed, every_n_bars=1),
}


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def main(verbose: bool) -> None:
    """AI_Finance data tooling.

    Phase 0: fetch bars, store them, and check whether they are trustworthy.
    """
    _configure_logging(verbose)


@main.command()
@click.option("--symbol", "symbols", multiple=True, default=("BTCUSDT",), help="Repeatable.")
@click.option("--start", default="2017-08-17", show_default=True, help="First bar wanted, UTC.")
@click.option("--end", default=None, help="Last bar wanted, UTC. Omit for 'up to now'.")
@click.option(
    "--interval",
    default=BASE_INTERVAL,
    type=click.Choice(sorted(INTERVAL_MS)),
    show_default=True,
    help="Only 1m should be stored; coarser intervals are derived on read.",
)
@click.option(
    "--source",
    default="binance",
    type=click.Choice(SOURCES),
    show_default=True,
    help="'synthetic' generates deterministic fake bars for offline testing.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Continue from the newest stored bar. --no-resume re-fetches the range.",
)
@click.option("--seed", default=0, show_default=True, help="Synthetic source seed.")
def fetch(
    symbols: tuple[str, ...],
    start: str,
    end: str | None,
    interval: str,
    source: str,
    resume: bool,
    seed: int,
) -> None:
    """Download bars and merge them into the store.

    Safe to re-run: writes are idempotent and resume from the last stored bar.
    """
    if interval != BASE_INTERVAL:
        click.secho(
            f"warning: storing {interval} bars. Convention is to store only "
            f"{BASE_INTERVAL} and derive the rest on read.",
            fg="yellow",
            err=True,
        )

    for symbol in symbols:
        if source == "binance":
            bar_source = BinanceSource()
        else:
            bar_source = SyntheticSource(
                epoch_ms=int(pd.Timestamp(start, tz="UTC").timestamp() * 1000), seed=seed
            )

        def progress(request: int, cursor: pd.Timestamp, count: int, _symbol: str = symbol) -> None:
            if request % 25 == 0:
                click.echo(f"  {_symbol}: request {request}, through {cursor:%Y-%m-%d %H:%M}")

        result = fetch_history(
            bar_source,
            symbol,
            start=start,
            end=end,
            interval=interval,
            resume=resume,
            on_progress=progress,
        )
        if result.resumed_from is not None:
            click.echo(f"  {symbol}: resumed after {result.resumed_from:%Y-%m-%d %H:%M} UTC")
        click.secho(result.summary(), fg="green")


@main.command()
@click.option("--symbol", "symbols", multiple=True, default=("BTCUSDT",), help="Repeatable.")
@click.option(
    "--interval", default=BASE_INTERVAL, type=click.Choice(sorted(INTERVAL_MS)), show_default=True
)
@click.option("--start", default=None, help="Restrict the check to a range, UTC.")
@click.option("--end", default=None)
def quality(symbols: tuple[str, ...], interval: str, start: str | None, end: str | None) -> None:
    """Report gaps, duplicates, bad prints and frozen feeds.

    Exits non-zero if any symbol has errors, so it can gate a pipeline.
    """
    failures = 0
    for symbol in symbols:
        bars = load_bars(symbol, start, end, interval)
        report = check_bars(bars, symbol, interval)
        click.echo(report.to_text())
        click.echo("")
        if not report.is_clean():
            failures += 1

    if failures:
        click.secho(f"{failures} symbol(s) have data quality errors.", fg="red", err=True)
        sys.exit(1)
    click.secho("All checked series are clean.", fg="green")


@main.command()
def info() -> None:
    """What is currently in the store."""
    summary = store_summary()
    if summary.empty:
        click.echo(f"Store is empty ({bars_dir()}). Run 'aifin fetch' first.")
        return
    click.echo(f"Store: {bars_dir()}\n")
    click.echo(summary.to_string(index=False))


@main.command()
@click.option("--symbol", default="BTCUSDT", show_default=True)
@click.option(
    "--interval", default=BASE_INTERVAL, type=click.Choice(sorted(INTERVAL_MS)), show_default=True
)
@click.option("--start", default=None, help="UTC.")
@click.option("--end", default=None, help="UTC.")
@click.option("--tail", default=10, show_default=True, help="Rows to print from the end.")
def show(symbol: str, interval: str, start: str | None, end: str | None, tail: int) -> None:
    """Print the last few bars. For eyeballing that the data looks sane."""
    bars = load_bars(symbol, start, end, interval)
    if bars.empty:
        click.echo("No bars for that request.")
        return
    click.echo(f"{symbol} {interval}: {len(bars):,} bars")
    click.echo(bars.tail(tail).to_string(index=False))


@main.command()
@click.option("--symbol", default="BTCUSDT", show_default=True)
@click.option(
    "--strategy",
    "strategy_names",
    multiple=True,
    default=("all",),
    help="Repeatable, or 'all' for every baseline.",
)
@click.option("--interval", default="4h", type=click.Choice(sorted(INTERVAL_MS)), show_default=True)
@click.option("--start", default=None, help="UTC.")
@click.option("--end", default=None, help="UTC.")
@click.option("--train-bars", default=1000, show_default=True, help="Bars per training window.")
@click.option("--test-bars", default=250, show_default=True, help="Bars per test window.")
@click.option(
    "--embargo-bars",
    default=10,
    show_default=True,
    help="Gap between training and testing, to stop the two sharing information.",
)
@click.option("--anchored", is_flag=True, help="Expand the training window instead of rolling it.")
@click.option(
    "--select-by",
    default="sharpe",
    type=click.Choice(SELECTION_METRICS),
    show_default=True,
    help="Metric used to pick parameters on each training window.",
)
@click.option("--initial-equity", default=10_000.0, show_default=True)
@click.option("--fee", default=0.001, show_default=True, help="Exchange fee per side.")
@click.option("--max-position", default=0.25, show_default=True, help="Risk cap on one position.")
@click.option("--registry/--no-registry", default=True, show_default=True)
def walkforward(
    symbol: str,
    strategy_names: tuple[str, ...],
    interval: str,
    start: str | None,
    end: str | None,
    train_bars: int,
    test_bars: int,
    embargo_bars: int,
    anchored: bool,
    select_by: str,
    initial_equity: float,
    fee: float,
    max_position: float,
    registry: bool,
) -> None:
    """Fit parameters on the past, measure on the future, roll forward.

    Only out-of-sample results are printed. Training performance is a selection
    artefact, not a finding, so it is deliberately not summarised anywhere.
    """
    names = sorted(STRATEGY_FACTORIES) if "all" in strategy_names else list(strategy_names)
    unknown = [n for n in names if n not in STRATEGY_FACTORIES]
    if unknown:
        raise click.ClickException(
            f"unknown strategy {unknown}; expected from {sorted(STRATEGY_FACTORIES)}"
        )

    bars = load_bars(symbol, start, end, interval)
    if bars.empty:
        raise click.ClickException(f"no {interval} bars for {symbol}. Run 'aifin fetch' first.")

    quality = check_bars(bars, symbol, interval)
    if not quality.is_clean():
        click.secho("warning: data has quality errors; see 'aifin quality'.", fg="yellow", err=True)

    log_book = Registry() if registry else None
    costs = CostModel(fee_rate=fee)
    limits = RiskLimits(max_position_weight=max_position)

    results = []
    for name in names:
        try:
            result = run_walk_forward(
                bars,
                name,
                train_bars=train_bars,
                test_bars=test_bars,
                embargo_bars=embargo_bars,
                anchored=anchored,
                symbol=symbol,
                interval=interval,
                initial_equity=initial_equity,
                costs=costs,
                limits=limits,
                selection_metric=select_by,
                registry=log_book,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        results.append(result)
        click.echo(result.to_text())
        click.echo("")

    _print_verdict(results, log_book)


def _print_verdict(results, log_book: Registry | None) -> None:
    """The Phase 2 gate: did anything beat buy-and-hold risk-adjusted, after costs?

    Three hurdles, all of which a real edge must clear:

    1. A **positive** Sharpe. Losing less than a falling market is not an edge,
       it is cash, and cash is free.
    2. Better than buy-and-hold on the same days.
    3. Above the noise floor implied by how many parameter sets were tried. This
       is the hurdle that makes the experiment registry worth keeping: with
       enough attempts, something always looks good.
    """
    trials = sum(r.variants_tested for r in results) if results else 0
    observations = max((len(r.oos.daily_returns) for r in results), default=0)
    floor = expected_max_sharpe(trials, observations) if observations >= 2 and trials else 0.0

    click.echo("=" * 62)
    click.echo("VERDICT")
    click.echo("=" * 62)

    passed = []
    for result in results:
        verdict, reason = _judge(result.oos, floor)
        if verdict == "PASS":
            passed.append(result)
        oos = result.oos
        click.echo(
            f"  [{verdict}] {result.strategy_name:<20} "
            f"sharpe {_fmt(oos.sharpe):>7} vs {_fmt(oos.benchmark_sharpe):>7}   "
            f"return {oos.total_return * 100:+7.1f}%   churn {result.param_turnover * 100:3.0f}%"
        )
        if reason:
            click.echo(f"         {reason}")

    click.echo("")
    if floor:
        click.echo(
            f"Noise floor: {trials:,} parameter runs over {observations:,} days. The best of "
            f"that\nmany worthless strategies would show a Sharpe of {floor:.2f} by luck alone, "
            f"so\nthat is the bar, not zero."
        )
        click.echo("")

    if passed:
        click.secho(
            f"{len(passed)} baseline(s) cleared all three hurdles out of sample.",
            fg="green",
        )
        click.echo("Worth investigating further — and worth re-running on a different period.")
    else:
        click.secho("No baseline found an edge on this data.", fg="yellow")
        click.echo(
            "That is a documented result, not a failure. It says these classic rules\n"
            "carry no edge here after costs, which is worth knowing before spending\n"
            "a month on machine learning."
        )

    if log_book is not None:
        click.echo("")
        click.echo(f"Recorded to {log_book.path}")


def _judge(oos, floor: float) -> tuple[str, str]:
    """Grade one out-of-sample result. Returns ``(verdict, reason_if_failed)``."""
    if oos.sharpe != oos.sharpe:  # nan
        return "fail", "no Sharpe: too few days, or the curve never moved"
    if oos.sharpe <= 0:
        return "fail", "negative Sharpe — it lost money risk-adjusted, benchmark aside"
    if oos.benchmark_sharpe == oos.benchmark_sharpe and oos.sharpe <= oos.benchmark_sharpe:
        return "fail", "did not beat buy-and-hold on the same days"
    if oos.sharpe <= floor:
        return "fail", f"below the {floor:.2f} noise floor for this many parameter runs"
    return "PASS", ""


def _fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.2f}"


@main.command()
@click.option("--symbol", default="BTCUSDT", show_default=True)
@click.option(
    "--strategy",
    "strategy_name",
    default="buy-and-hold",
    type=click.Choice(sorted(STRATEGIES)),
    show_default=True,
    help="Phase 1 ships reference strategies only; real baselines arrive in Phase 2.",
)
@click.option(
    "--interval",
    default="4h",
    type=click.Choice(sorted(INTERVAL_MS)),
    show_default=True,
    help="Decision cadence. See PLAN.md section 1 for why this is not 1m.",
)
@click.option("--start", default=None, help="UTC.")
@click.option("--end", default=None, help="UTC.")
@click.option("--initial-equity", default=10_000.0, show_default=True)
@click.option("--fee", default=0.001, show_default=True, help="Exchange fee per side.")
@click.option("--spread-bps", default=0.5, show_default=True, help="Half-spread, basis points.")
@click.option("--slippage-bps", default=0.5, show_default=True, help="Slippage, basis points.")
@click.option(
    "--unconstrained",
    is_flag=True,
    help="Disable the risk limits. For engine calibration only \u2014 never for research.",
)
@click.option("--liquidate", is_flag=True, help="Close any open position at the final bar.")
@click.option("--seed", default=0, show_default=True, help="Seed for the random strategy.")
def backtest(
    symbol: str,
    strategy_name: str,
    interval: str,
    start: str | None,
    end: str | None,
    initial_equity: float,
    fee: float,
    spread_bps: float,
    slippage_bps: float,
    unconstrained: bool,
    liquidate: bool,
    seed: int,
) -> None:
    """Replay a strategy over stored bars and report what it would have done.

    The benchmark and the cost bill are always printed. A result without both is
    not interpretable.
    """
    bars = load_bars(symbol, start, end, interval)
    if len(bars) < 2:
        raise click.ClickException(
            f"need at least 2 {interval} bars for {symbol}; found {len(bars)}. "
            "Run 'aifin fetch' first."
        )

    report = check_bars(bars, symbol, interval)
    if not report.is_clean():
        click.secho(
            "warning: this series has data quality errors, so the result below is "
            "built on data you have not vouched for:",
            fg="yellow",
            err=True,
        )
        for problem in report.errors:
            click.secho(f"  - {problem}", fg="yellow", err=True)
        click.echo("")

    costs = CostModel(fee_rate=fee, half_spread_bps=spread_bps, slippage_bps=slippage_bps)
    limits = RiskLimits.unconstrained() if unconstrained else RiskLimits()
    if unconstrained:
        click.secho(
            "risk limits disabled: this measures the engine, not a strategy you could run.",
            fg="yellow",
            err=True,
        )

    result = run_backtest(
        bars,
        STRATEGIES[strategy_name](seed),
        symbol=symbol,
        initial_equity=initial_equity,
        costs=costs,
        limits=limits,
        liquidate_at_end=liquidate,
    )
    click.echo(result.to_text())


if __name__ == "__main__":
    main()
