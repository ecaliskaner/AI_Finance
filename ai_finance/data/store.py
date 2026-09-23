"""The bar store: Parquet on disk, and the one function everything reads through.

Layout::

    <data_dir>/bars/<SYMBOL>/<interval>/<YYYY-MM>.parquet

Monthly partitions keep files small enough to rewrite cheaply during an
incremental fetch, while keeping the file count manageable over years of
1-minute data.

Two rules this module exists to enforce:

1. **Raw data is immutable in spirit.** Only 1-minute bars are ever written.
   Coarser intervals are derived on read by :func:`resample_bars`, so there is
   exactly one source of truth and no way for two stored intervals to disagree.
2. **Everything reads through :func:`load_bars`.** One entry point means one
   place where the schema, sorting, deduplication and timezone handling are
   guaranteed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from ai_finance.config import (
    BASE_INTERVAL,
    INTERVAL_PANDAS_FREQ,
    bars_dir,
    interval_ms,
)
from ai_finance.data.schema import empty_bars, normalize_bars

log = logging.getLogger(__name__)

_AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trades": "sum",
}


def symbol_dir(symbol: str, interval: str = BASE_INTERVAL, root: Path | None = None) -> Path:
    """Directory holding one symbol's partitions for one interval."""
    base = root if root is not None else bars_dir()
    return base / symbol.upper() / interval


def partition_path(
    symbol: str, interval: str, period: str, root: Path | None = None
) -> Path:
    """Path of a single ``YYYY-MM`` partition file."""
    return symbol_dir(symbol, interval, root) / f"{period}.parquet"


def write_bars(
    bars: pd.DataFrame,
    symbol: str,
    interval: str = BASE_INTERVAL,
    root: Path | None = None,
) -> list[Path]:
    """Merge ``bars`` into the store, one file per calendar month.

    Idempotent: writing the same bars twice leaves the store unchanged, and
    overlapping writes keep the newly supplied version of a duplicated
    timestamp. That is what makes an incremental fetch safe to re-run, and it is
    what lets a cron job recover from a partial failure by simply running again.

    Returns:
        The partition files that were written, in chronological order.
    """
    bars = normalize_bars(bars)
    if bars.empty:
        return []

    directory = symbol_dir(symbol, interval, root)
    directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    periods = bars["open_time"].dt.strftime("%Y-%m")
    for period, chunk in bars.groupby(periods, sort=True):
        path = partition_path(symbol, interval, str(period), root)
        if path.exists():
            merged = normalize_bars(pd.concat([_read_partition(path), chunk], ignore_index=True))
        else:
            merged = normalize_bars(chunk)
        merged.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
        written.append(path)

    log.info("wrote %d bars for %s %s across %d file(s)", len(bars), symbol, interval, len(written))
    return written


def load_bars(
    symbol: str,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    interval: str = BASE_INTERVAL,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load bars for ``symbol``, optionally resampled and range-filtered.

    This is the only supported way to read bars. Args ``start`` and ``end`` are
    inclusive and interpreted as UTC.

    Intervals coarser than ``1m`` are derived from stored 1-minute bars. Periods
    that the stored range does not fully cover are dropped, so the last bar is
    never a partial one — see :func:`resample_bars`.
    """
    interval_ms(interval)  # validate
    base = _load_base(symbol, start, end, interval, root)

    if interval == BASE_INTERVAL:
        result = base
    else:
        result = resample_bars(base, interval)

    return _clip(result, start, end)


def last_bar_time(
    symbol: str, interval: str = BASE_INTERVAL, root: Path | None = None
) -> pd.Timestamp | None:
    """``open_time`` of the newest stored bar, or ``None`` if the store is empty.

    Used to resume an incremental fetch without re-downloading history.
    """
    paths = _partitions(symbol, interval, root)
    if not paths:
        return None
    newest = _read_partition(paths[-1])
    if newest.empty:
        return None
    return newest["open_time"].iloc[-1]


def resample_bars(bars: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate 1-minute bars up to ``interval``.

    Buckets are anchored to the Unix epoch and labelled by their **start**, so a
    4-hour bar labelled ``12:00`` covers ``12:00:00``–``15:59:59.999``. That
    matches how exchanges label their own coarse candles, which keeps stored and
    exchange-reported bars comparable.

    Incomplete periods at both ends are dropped. A 4-hour bar built from ten
    minutes of data is not a 4-hour bar, and letting one through would put a
    half-formed bar at the end of every live run — exactly where a strategy is
    about to act on it.
    """
    freq = INTERVAL_PANDAS_FREQ.get(interval)
    if freq is None:
        raise ValueError(f"cannot resample to unknown interval {interval!r}")
    if bars.empty:
        return empty_bars()

    step = interval_ms(interval)
    indexed = bars.set_index("open_time").sort_index()

    # `origin` only applies to tick-like frequencies; pandas ignores it (and
    # warns) for calendar offsets such as 1D. Daily buckets already begin at
    # midnight UTC, which is what epoch anchoring means for them, so the
    # guarantee holds either way — but asking for it would emit a warning that
    # we would then have to teach ourselves to ignore.
    options: dict[str, object] = {"label": "left", "closed": "left"}
    if isinstance(to_offset(freq), Tick):
        options["origin"] = "epoch"

    grouped = (
        indexed.resample(freq, **options).agg(_AGGREGATION).dropna(subset=["open"])
    )
    grouped = grouped.reset_index()
    grouped["close_time"] = grouped["open_time"] + pd.Timedelta(milliseconds=step - 1)

    # Keep only periods fully covered by the input range.
    first_known = bars["open_time"].iloc[0]
    last_known = bars["close_time"].iloc[-1]
    covered = (grouped["open_time"] >= first_known) & (grouped["close_time"] <= last_known)

    return normalize_bars(grouped.loc[covered])


def store_summary(root: Path | None = None) -> pd.DataFrame:
    """One row per (symbol, interval) held on disk. Backs ``aifin info``."""
    base = root if root is not None else bars_dir()
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(
            columns=["symbol", "interval", "bars", "first", "last", "files", "megabytes"]
        )

    for symbol_path in sorted(p for p in base.iterdir() if p.is_dir()):
        for interval_path in sorted(p for p in symbol_path.iterdir() if p.is_dir()):
            paths = sorted(interval_path.glob("*.parquet"))
            if not paths:
                continue
            frames = [_read_partition(p) for p in paths]
            combined = normalize_bars(pd.concat(frames, ignore_index=True))
            rows.append(
                {
                    "symbol": symbol_path.name,
                    "interval": interval_path.name,
                    "bars": len(combined),
                    "first": combined["open_time"].iloc[0] if not combined.empty else pd.NaT,
                    "last": combined["open_time"].iloc[-1] if not combined.empty else pd.NaT,
                    "files": len(paths),
                    "megabytes": round(sum(p.stat().st_size for p in paths) / 1e6, 2),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _partitions(symbol: str, interval: str, root: Path | None) -> list[Path]:
    directory = symbol_dir(symbol, interval, root)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def _read_partition(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def _load_base(
    symbol: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
    interval: str,
    root: Path | None,
) -> pd.DataFrame:
    """Read stored 1-minute partitions, skipping months outside the range.

    When resampling, the range is widened to whole target periods so an edge
    bucket is not silently built from a truncated slice of minutes.
    """
    paths = _partitions(symbol, BASE_INTERVAL, root)
    if not paths:
        return empty_bars()

    lower = _to_utc(start)
    upper = _to_utc(end)
    if interval != BASE_INTERVAL:
        step = pd.Timedelta(milliseconds=interval_ms(interval))
        if lower is not None:
            lower = lower.floor(step)
        if upper is not None:
            upper = upper.ceil(step)

    selected = [p for p in paths if _partition_overlaps(p, lower, upper)]
    if not selected:
        return empty_bars()

    frames = [_read_partition(p) for p in selected]
    combined = normalize_bars(pd.concat(frames, ignore_index=True))
    return _clip(combined, lower, upper)


def _partition_overlaps(
    path: Path, lower: pd.Timestamp | None, upper: pd.Timestamp | None
) -> bool:
    """Cheap month-level pruning from the filename, before reading anything."""
    try:
        month_start = pd.Timestamp(path.stem, tz="UTC")
    except ValueError:
        return True  # unrecognised name: read it rather than silently skip data
    month_end = month_start + pd.offsets.MonthBegin(1)
    if upper is not None and month_start > upper:
        return False
    return not (lower is not None and month_end <= lower)


def _clip(
    bars: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    lower = _to_utc(start)
    upper = _to_utc(end)
    if lower is not None:
        bars = bars.loc[bars["open_time"] >= lower]
    if upper is not None:
        bars = bars.loc[bars["open_time"] <= upper]
    return bars.reset_index(drop=True)


def _to_utc(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
