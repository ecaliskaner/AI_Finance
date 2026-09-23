"""Fetch orchestration: pagination, resumption, and writing to the store.

The exchange hands out at most 1000 bars per request, so five years of minute
data is about 2,600 requests. This module walks that cursor, writes as it goes,
and can be re-run to pick up only what is new.

Re-runnability is the point. A cron job that fails halfway through leaves the
store consistent, and the next run continues from the last stored bar. Nothing
here needs to succeed in one shot.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_finance.config import BASE_INTERVAL, interval_ms
from ai_finance.data.sources import MAX_LIMIT, BarSource
from ai_finance.data.store import last_bar_time, write_bars

log = logging.getLogger(__name__)

#: Safety valve. At 1000 bars a request this is ~20 years of minute data.
DEFAULT_MAX_REQUESTS = 20_000

#: Bars to accumulate before writing to disk.
#:
#: Writing every 1000-bar chunk means re-reading and re-writing a monthly
#: partition ~44 times per month, which is quadratic in the month's size and
#: dominated a five-year backfill. Batching makes it linear. The cost is that an
#: interrupted run may have to re-fetch up to this many bars, which is a few
#: seconds of network time.
DEFAULT_FLUSH_BARS = 100_000


@dataclass(frozen=True)
class FetchResult:
    """What one :func:`fetch_history` call did."""

    symbol: str
    interval: str
    requests: int
    bars_written: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    resumed_from: pd.Timestamp | None

    def summary(self) -> str:
        if self.bars_written == 0:
            return f"{self.symbol} {self.interval}: already up to date ({self.requests} request(s))"
        return (
            f"{self.symbol} {self.interval}: wrote {self.bars_written:,} bars "
            f"({self.first:%Y-%m-%d %H:%M} .. {self.last:%Y-%m-%d %H:%M} UTC) "
            f"in {self.requests} request(s)"
        )


def fetch_history(
    source: BarSource,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    interval: str = BASE_INTERVAL,
    *,
    root: Path | None = None,
    resume: bool = True,
    limit: int = MAX_LIMIT,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    flush_bars: int = DEFAULT_FLUSH_BARS,
    on_progress: Callable[[int, pd.Timestamp, int], None] | None = None,
) -> FetchResult:
    """Download ``symbol`` bars from ``start`` and merge them into the store.

    Args:
        source: where bars come from. Any :class:`~ai_finance.data.sources.BarSource`.
        symbol: exchange symbol, e.g. ``BTCUSDT``.
        start: first bar wanted, UTC. Ignored for the portion already stored
            when ``resume`` is true.
        end: last bar wanted, UTC. ``None`` means "up to now".
        interval: bar interval. Only ``1m`` should ever be stored.
        root: store root, for tests.
        resume: continue from the newest stored bar instead of re-downloading.
            Turn this off to force a re-fetch — writes are idempotent, so
            re-fetching a range is safe and simply overwrites it.
        limit: bars per request, capped by the exchange at 1000.
        max_requests: refuse to loop forever.
        flush_bars: bars to buffer before writing. Buffered bars are always
            flushed on the way out, including when an exception propagates, so
            an interrupted run keeps whatever it managed to download.
        on_progress: called as ``(request_number, cursor, bars_in_chunk)``.

    Returns:
        A :class:`FetchResult` describing what was written.
    """
    step = interval_ms(interval)
    symbol = symbol.upper()

    resumed_from: pd.Timestamp | None = None
    cursor_ts = _to_utc(start)
    if resume:
        stored = last_bar_time(symbol, interval, root)
        if stored is not None and stored >= cursor_ts:
            resumed_from = stored
            # Start one bar after the last stored one.
            cursor_ts = stored + pd.Timedelta(milliseconds=step)

    end_ts = _to_utc(end) if end is not None else None
    end_ms = _epoch_ms(end_ts) if end_ts is not None else None

    if flush_bars < 1:
        raise ValueError("flush_bars must be positive")

    cursor_ms = _epoch_ms(cursor_ts)
    total_written = 0
    requests = 0
    first_written: pd.Timestamp | None = None
    last_written: pd.Timestamp | None = None
    buffer: list[pd.DataFrame] = []
    buffered = 0
    exhausted_requests = True

    def flush() -> None:
        nonlocal buffer, buffered
        if not buffer:
            return
        write_bars(pd.concat(buffer, ignore_index=True), symbol, interval, root)
        buffer = []
        buffered = 0

    try:
        while requests < max_requests:
            if end_ms is not None and cursor_ms > end_ms:
                exhausted_requests = False
                break

            chunk = source.fetch_chunk(symbol, interval, cursor_ms, end_ms, limit)
            requests += 1

            if chunk.empty:
                # No more closed bars available. This is the normal exit.
                exhausted_requests = False
                break

            buffer.append(chunk)
            buffered += len(chunk)
            total_written += len(chunk)
            if buffered >= flush_bars:
                flush()

            chunk_last = chunk["open_time"].iloc[-1]
            if first_written is None:
                first_written = chunk["open_time"].iloc[0]
            last_written = chunk_last

            if on_progress is not None:
                on_progress(requests, chunk_last, len(chunk))

            next_cursor_ms = _epoch_ms(chunk_last) + step
            if next_cursor_ms <= cursor_ms:
                # A source that will not advance would spin forever. Stop instead.
                log.warning("source did not advance past %s for %s; stopping", chunk_last, symbol)
                exhausted_requests = False
                break
            cursor_ms = next_cursor_ms
    finally:
        # Persist whatever we have, even on the way out of an exception.
        flush()

    if exhausted_requests:
        log.warning(
            "hit max_requests=%d for %s; re-run to continue from %s",
            max_requests,
            symbol,
            last_written,
        )

    return FetchResult(
        symbol=symbol,
        interval=interval,
        requests=requests,
        bars_written=total_written,
        first=first_written,
        last=last_written,
        resumed_from=resumed_from,
    )


def _to_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def _epoch_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)
