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
from ai_finance.risk.engine import RiskLimits
from ai_finance.strategy.baselines import AlwaysFlat, BuyAndHold, RandomStrategy

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
