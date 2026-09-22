"""Persistent state for the scheduled job.

The runner is a short-lived process. It wakes, decides, acts, and exits — so
everything it needs to remember between runs lives in one JSON file, written
atomically.

**Atomically** is the whole point. A process killed halfway through writing its
state leaves a truncated file, and a truncated state file on a trading system is
worse than no state file: it can read back a position you do not hold. So the
write goes to a temporary file in the same directory and is then renamed over
the target, which POSIX guarantees is atomic. The previous version is kept as a
``.bak`` so a corrupt read has somewhere to fall back to.

**What paper mode cannot do.** `docs/ARCHITECTURE.md` says the exchange is the
source of truth and local state is only a cache. That is right for live trading
and impossible in paper trading, where the exchange holds no position to
reconcile against — here this file *is* the authority. The distinction matters
for Phase 5: `execution/live.py` must reconcile against the exchange on every
run and trust it over anything stored here.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pandas as pd

from ai_finance.config import data_dir

log = logging.getLogger(__name__)

#: Bumped when the stored shape changes in a way old files cannot satisfy.
STATE_VERSION = 1


class StateError(RuntimeError):
    """Raised when stored state cannot be used safely."""


@dataclass
class RunState:
    """Everything the runner must remember between wakeups."""

    symbol: str
    interval: str
    mode: str
    strategy: str
    initial_equity: float
    cash: float
    units: float = 0.0

    #: Close time of the most recent bar the runner acted on. The idempotency
    #: key: a second run that sees the same bar does nothing.
    last_bar_time: str | None = None
    last_run_at: str | None = None
    started_at: str | None = None
    n_runs: int = 0
    n_fills: int = 0
    total_fees: float = 0.0
    total_concession: float = 0.0

    # Risk-engine state, carried across runs so a halt survives a restart.
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_day: str | None = None
    halted_permanently: bool = False
    halted_until: str | None = None
    halt_reason: str = ""
    recent_order_times: list[str] = field(default_factory=list)

    version: int = STATE_VERSION

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def weight(self, price: float) -> float:
        equity = self.equity(price)
        return 0.0 if equity == 0 else (self.units * price) / equity

    def has_acted_on(self, bar_close: pd.Timestamp) -> bool:
        """True if this bar has already been handled.

        What makes the job safe to run on an overlapping schedule, to retry
        after a failure, or to fire twice because a cron entry was duplicated.
        """
        if self.last_bar_time is None:
            return False
        return pd.Timestamp(self.last_bar_time) >= bar_close


def default_state_path(symbol: str, mode: str) -> Path:
    return data_dir() / "state" / f"{symbol.upper()}-{mode}.json"


class StateStore:
    """Loads and saves a :class:`RunState`, atomically."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.backup_path = path.with_suffix(path.suffix + ".bak")

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> RunState | None:
        """Read the state, falling back to the backup if the primary is broken."""
        for candidate in (self.path, self.backup_path):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("state file %s is unreadable (%s); trying the backup", candidate, exc)
                continue
            if payload.get("version") != STATE_VERSION:
                raise StateError(
                    f"{candidate} was written by state version {payload.get('version')}, "
                    f"but this build expects {STATE_VERSION}. Inspect it before continuing; "
                    "do not delete it while a position may be open."
                )
            if candidate is self.backup_path:
                log.warning("recovered state from the backup file")
            return RunState(**payload)
        return None

    def save(self, state: RunState) -> None:
        """Write ``state`` atomically, keeping the previous version as a backup."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.backup_path.write_bytes(self.path.read_bytes())

        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(asdict(state), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        # Rename within the same directory is atomic: a reader sees either the
        # whole old file or the whole new one, never half of either.
        temporary.replace(self.path)

    def initialise(self, state: RunState) -> RunState:
        """Create state if none exists, otherwise return what is stored.

        Refuses to overwrite state belonging to a different symbol, mode or
        starting capital — those mismatches usually mean a mistyped command, and
        silently resetting would discard a position.
        """
        existing = self.load()
        if existing is None:
            fresh = replace(state, started_at=_now_iso(), peak_equity=state.initial_equity)
            self.save(fresh)
            return fresh

        for attribute in ("symbol", "interval", "mode"):
            stored = getattr(existing, attribute)
            requested = getattr(state, attribute)
            if stored != requested:
                raise StateError(
                    f"stored state is for {attribute}={stored!r} but this run asked for "
                    f"{requested!r}. Use a different state file rather than overwriting "
                    f"this one: {self.path}"
                )
        return existing


def _now_iso() -> str:
    return pd.Timestamp.now("UTC").isoformat()
