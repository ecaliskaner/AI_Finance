"""Data quality checks.

Bad data produces beautiful backtests. A missing hour during a crash, a bad
print at ten times the real price, a frozen feed repeating the same candle —
each one makes a strategy look better than it is, because the losses that
actually happened are simply absent from the file.

So the store is never trusted implicitly. :func:`check_bars` runs before any
research, and its report is either empty or explained.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ai_finance.config import interval_ms

#: A return this many robust sigmas from the median is flagged for inspection.
#: Not an error: crypto genuinely moves. It marks bars worth a human glance.
EXTREME_RETURN_SIGMAS = 12.0

#: Consecutive identical OHLC bars beyond this length suggest a frozen feed
#: rather than a quiet market.
STALE_RUN_THRESHOLD = 10


@dataclass(frozen=True)
class Gap:
    """A stretch of missing bars."""

    after: pd.Timestamp
    before: pd.Timestamp
    missing: int

    def __str__(self) -> str:
        return (
            f"{self.missing:>7,} bars missing between "
            f"{self.after:%Y-%m-%d %H:%M} and {self.before:%Y-%m-%d %H:%M}"
        )


@dataclass(frozen=True)
class QualityReport:
    """What we know about a stored series. Empty findings mean it is usable."""

    symbol: str
    interval: str
    n_bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    expected_bars: int
    missing_bars: int
    gaps: list[Gap] = field(default_factory=list)
    duplicate_timestamps: int = 0
    out_of_order: int = 0
    ohlc_violations: int = 0
    nonpositive_prices: int = 0
    close_time_mismatches: int = 0
    zero_volume_bars: int = 0
    stale_runs: list[tuple[pd.Timestamp, int]] = field(default_factory=list)
    extreme_returns: list[tuple[pd.Timestamp, float]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of expected bars actually present, in ``[0, 1]``."""
        if self.expected_bars == 0:
            return 1.0
        return self.n_bars / self.expected_bars

    @property
    def errors(self) -> list[str]:
        """Findings that make the data unfit for research until resolved."""
        problems: list[str] = []
        if self.missing_bars:
            problems.append(f"{self.missing_bars:,} missing bars in {len(self.gaps)} gap(s)")
        if self.duplicate_timestamps:
            problems.append(f"{self.duplicate_timestamps:,} duplicate timestamps")
        if self.out_of_order:
            problems.append(f"{self.out_of_order:,} out-of-order timestamps")
        if self.ohlc_violations:
            problems.append(f"{self.ohlc_violations:,} bars violating OHLC invariants")
        if self.nonpositive_prices:
            problems.append(f"{self.nonpositive_prices:,} bars with non-positive prices")
        if self.close_time_mismatches:
            problems.append(f"{self.close_time_mismatches:,} bars with a wrong close_time")
        return problems

    @property
    def warnings(self) -> list[str]:
        """Findings worth a look that are not necessarily wrong."""
        notes: list[str] = []
        if self.zero_volume_bars:
            notes.append(f"{self.zero_volume_bars:,} zero-volume bars")
        if self.stale_runs:
            longest = max(length for _, length in self.stale_runs)
            notes.append(
                f"{len(self.stale_runs)} stale run(s) of identical OHLC, longest {longest} bars"
            )
        if self.extreme_returns:
            worst = max(abs(r) for _, r in self.extreme_returns)
            notes.append(
                f"{len(self.extreme_returns)} extreme return(s), largest {worst * 100:.1f}%"
            )
        return notes

    def is_clean(self) -> bool:
        """True when nothing in :attr:`errors` was found."""
        return not self.errors

    def to_text(self, max_items: int = 10) -> str:
        """Human-readable report, as printed by ``aifin quality``."""
        lines = [f"Data quality: {self.symbol} {self.interval}", "=" * 46]
        if self.n_bars == 0:
            lines.append("No bars stored.")
            return "\n".join(lines)

        lines += [
            f"bars       {self.n_bars:,}",
            f"range      {self.first:%Y-%m-%d %H:%M} .. {self.last:%Y-%m-%d %H:%M} UTC",
            f"expected   {self.expected_bars:,}",
            f"coverage   {self.coverage * 100:.4f}%",
            "",
        ]

        if self.errors:
            lines.append("ERRORS")
            lines += [f"  - {problem}" for problem in self.errors]
        else:
            lines.append("ERRORS     none")
        lines.append("")

        if self.warnings:
            lines.append("WARNINGS")
            lines += [f"  - {note}" for note in self.warnings]
        else:
            lines.append("WARNINGS   none")

        if self.gaps:
            lines += ["", f"Largest gaps (of {len(self.gaps)}):"]
            biggest = sorted(self.gaps, key=lambda g: g.missing, reverse=True)[:max_items]
            lines += [f"  {gap}" for gap in biggest]

        if self.extreme_returns:
            lines += ["", f"Largest returns flagged (of {len(self.extreme_returns)}):"]
            biggest = sorted(self.extreme_returns, key=lambda r: abs(r[1]), reverse=True)
            lines += [
                f"  {ts:%Y-%m-%d %H:%M}  {ret * 100:+.2f}%" for ts, ret in biggest[:max_items]
            ]

        return "\n".join(lines)


def check_bars(bars: pd.DataFrame, symbol: str, interval: str) -> QualityReport:
    """Inspect ``bars`` and return findings.

    Expects the canonical schema but tolerates unsorted or duplicated input, so
    it can be pointed at raw source output as well as at the store.
    """
    step_ms = interval_ms(interval)

    if bars.empty:
        return QualityReport(
            symbol=symbol.upper(),
            interval=interval,
            n_bars=0,
            first=None,
            last=None,
            expected_bars=0,
            missing_bars=0,
        )

    open_time = bars["open_time"]
    duplicates = int(open_time.duplicated().sum())
    out_of_order = 0 if open_time.is_monotonic_increasing else int((open_time.diff() < pd.Timedelta(0)).sum())

    ordered = bars.sort_values("open_time", kind="stable").drop_duplicates(
        subset="open_time", keep="last"
    )
    first = ordered["open_time"].iloc[0]
    last = ordered["open_time"].iloc[-1]

    span_ms = int((last - first).total_seconds() * 1000)
    expected = span_ms // step_ms + 1
    present = len(ordered)

    step = pd.Timedelta(milliseconds=step_ms)
    deltas = ordered["open_time"].diff()
    gap_positions = np.flatnonzero((deltas > step).to_numpy())
    gaps = [
        Gap(
            after=ordered["open_time"].iloc[pos - 1],
            before=ordered["open_time"].iloc[pos],
            missing=int(deltas.iloc[pos] / step) - 1,
        )
        for pos in gap_positions
    ]

    high = ordered["high"].to_numpy()
    low = ordered["low"].to_numpy()
    open_ = ordered["open"].to_numpy()
    close = ordered["close"].to_numpy()

    prices = np.vstack([open_, high, low, close])
    nonpositive = int(np.count_nonzero(~(prices > 0).all(axis=0)))
    violations = int(
        np.count_nonzero(
            (high < low)
            | (high < np.maximum(open_, close) - _tolerance(high))
            | (low > np.minimum(open_, close) + _tolerance(low))
        )
    )

    expected_close = ordered["open_time"] + pd.Timedelta(milliseconds=step_ms - 1)
    close_mismatches = int((ordered["close_time"] != expected_close).sum())

    zero_volume = int((ordered["volume"] <= 0).sum())

    return QualityReport(
        symbol=symbol.upper(),
        interval=interval,
        n_bars=present,
        first=first,
        last=last,
        expected_bars=int(expected),
        missing_bars=max(0, int(expected) - present),
        gaps=gaps,
        duplicate_timestamps=duplicates,
        out_of_order=out_of_order,
        ohlc_violations=violations,
        nonpositive_prices=nonpositive,
        close_time_mismatches=close_mismatches,
        zero_volume_bars=zero_volume,
        stale_runs=_stale_runs(ordered),
        extreme_returns=_extreme_returns(ordered),
    )


def _tolerance(values: np.ndarray) -> np.ndarray:
    """Float-comparison slack, scaled to price magnitude."""
    return np.abs(values) * 1e-9


def _stale_runs(bars: pd.DataFrame) -> list[tuple[pd.Timestamp, int]]:
    """Runs of consecutive bars with identical OHLC, i.e. a frozen feed."""
    if len(bars) < STALE_RUN_THRESHOLD:
        return []
    same = (
        bars[["open", "high", "low", "close"]].diff().abs().sum(axis=1).to_numpy() == 0.0
    )
    runs: list[tuple[pd.Timestamp, int]] = []
    start_pos: int | None = None
    for pos, is_same in enumerate(same):
        if is_same:
            if start_pos is None:
                start_pos = pos - 1
        elif start_pos is not None:
            length = pos - start_pos
            if length >= STALE_RUN_THRESHOLD:
                runs.append((bars["open_time"].iloc[start_pos], length))
            start_pos = None
    if start_pos is not None:
        length = len(same) - start_pos
        if length >= STALE_RUN_THRESHOLD:
            runs.append((bars["open_time"].iloc[start_pos], length))
    return runs


def _extreme_returns(bars: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """Bar-to-bar returns far from the median, measured in MAD-sigmas.

    A median-absolute-deviation scale is used rather than a standard deviation
    because the outliers we are hunting would themselves inflate a standard
    deviation and hide each other.
    """
    if len(bars) < 100:
        return []
    close = bars["close"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(close) / close[:-1]
    finite = returns[np.isfinite(returns)]
    if finite.size == 0:
        return []
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad == 0.0:
        return []
    sigma = mad * 1.4826  # MAD -> standard-deviation equivalent for a normal
    threshold = EXTREME_RETURN_SIGMAS * sigma
    flagged = np.flatnonzero(np.abs(returns - median) > threshold)
    times = bars["open_time"].to_numpy()[1:]
    return [(pd.Timestamp(times[i]), float(returns[i])) for i in flagged]
