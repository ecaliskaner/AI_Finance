"""Project-wide configuration: paths and interval definitions.

Deliberately small. Anything that looks like a strategy parameter belongs with
the strategy, not here.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Supported bar intervals, mapped to their length in milliseconds.
#:
#: 1m is the only interval fetched from the exchange. Everything coarser is
#: derived from it by resampling (see :func:`ai_finance.data.store.resample_bars`),
#: so there is one source of truth on disk and no risk of two intervals
#: disagreeing.
INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

#: The interval we store raw. Never change this without re-fetching everything.
BASE_INTERVAL = "1m"

#: pandas offset aliases for resampling, keyed by our interval names.
INTERVAL_PANDAS_FREQ: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
}


def interval_ms(interval: str) -> int:
    """Length of ``interval`` in milliseconds, or raise for an unknown one."""
    try:
        return INTERVAL_MS[interval]
    except KeyError:
        raise ValueError(
            f"unknown interval {interval!r}; expected one of {sorted(INTERVAL_MS)}"
        ) from None


def data_dir() -> Path:
    """Root of the bar store. Override with the ``AIFIN_DATA_DIR`` env var."""
    return Path(os.environ.get("AIFIN_DATA_DIR", "data")).expanduser()


def bars_dir() -> Path:
    """Where raw bars live: ``<data_dir>/bars``."""
    return data_dir() / "bars"
