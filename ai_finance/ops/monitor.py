"""Heartbeats and missed-run detection.

The dangerous failure for a scheduled job is not a crash — a crash is loud. It
is the run that silently never happened: the cron entry someone edited, the
container that stopped being restarted, the timer that was masked by a package
upgrade. Nothing appears in a log that nobody is reading, and the position sits
there unmanaged.

So each run writes a heartbeat, and a separate cheap check asks whether the
newest one is older than the cadence allows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ai_finance.config import data_dir

#: How many cadence periods may elapse before a heartbeat counts as stale. Two
#: allows for one genuinely skipped run without crying wolf; three would let a
#: whole trading day pass unnoticed at a 4-hour cadence.
STALE_AFTER_PERIODS = 2.0


def heartbeat_path(symbol: str, mode: str, root: Path | None = None) -> Path:
    base = root if root is not None else data_dir()
    return base / "state" / f"{symbol.upper()}-{mode}.heartbeat.json"


@dataclass(frozen=True)
class Health:
    """Whether the runner is alive and acting."""

    symbol: str
    mode: str
    last_run: pd.Timestamp | None
    age_seconds: float | None
    cadence_seconds: float
    action: str = ""
    note: str = ""

    @property
    def is_stale(self) -> bool:
        if self.age_seconds is None:
            return True
        return self.age_seconds > self.cadence_seconds * STALE_AFTER_PERIODS

    @property
    def status(self) -> str:
        if self.last_run is None:
            return "NEVER RUN"
        return "STALE" if self.is_stale else "OK"

    def to_text(self) -> str:
        lines = [f"{self.symbol} ({self.mode}): {self.status}"]
        if self.last_run is None:
            lines.append("  no heartbeat found — has the scheduled job ever fired?")
            return "\n".join(lines)
        lines.append(f"  last run   {self.last_run:%Y-%m-%d %H:%M:%S} UTC")
        lines.append(f"  age        {self.age_seconds / 60:.0f} minutes")
        lines.append(f"  cadence    {self.cadence_seconds / 60:.0f} minutes")
        if self.action:
            lines.append(f"  last action {self.action}")
        if self.is_stale:
            lines.append(
                "  STALE: the newest heartbeat is older than the schedule allows. "
                "The job is not running."
            )
        return "\n".join(lines)


def write_heartbeat(
    symbol: str,
    mode: str,
    *,
    action: str,
    note: str = "",
    root: Path | None = None,
    now: pd.Timestamp | None = None,
) -> Path:
    """Record that a run completed."""
    path = heartbeat_path(symbol, mode, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol.upper(),
        "mode": mode,
        "at": (now if now is not None else pd.Timestamp.now("UTC")).isoformat(),
        "action": action,
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def check_health(
    symbol: str,
    mode: str,
    cadence_seconds: float,
    *,
    root: Path | None = None,
    now: pd.Timestamp | None = None,
) -> Health:
    """Report whether the runner has checked in recently enough."""
    path = heartbeat_path(symbol, mode, root)
    moment = now if now is not None else pd.Timestamp.now("UTC")

    if not path.exists():
        return Health(symbol.upper(), mode, None, None, cadence_seconds)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        last = pd.Timestamp(payload["at"])
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return Health(
            symbol.upper(), mode, None, None, cadence_seconds, note="heartbeat unreadable"
        )

    return Health(
        symbol=symbol.upper(),
        mode=mode,
        last_run=last,
        age_seconds=(moment - last).total_seconds(),
        cadence_seconds=cadence_seconds,
        action=payload.get("action", ""),
        note=payload.get("note", ""),
    )
