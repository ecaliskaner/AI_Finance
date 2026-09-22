"""The risk engine: the only place a position size is finally decided.

It sits between the strategy and execution in **all** modes, backtest included.
That placement is deliberate. A backtest that ignores the limits the live system
enforces is measuring a strategy you will never actually run, and its results
are therefore about a different strategy than the one whose returns you are
about to believe.

Limits are the ones in ``docs/RISK.md``. Two of them halt trading rather than
resize it:

- **Daily loss** halts for 24 hours, then resumes on its own.
- **Max drawdown** halts permanently and demands a human. The asymmetry is the
  point: a bad day is noise, a 15% drawdown is information, and the friction
  exists precisely for the moment you would most want to override it.

**Position caps bind at order time, not continuously.** A position entered at
25% of equity drifts above 25% when the asset rallies, and it is left alone. The
alternative — forcing a rebalance whenever drift breaches the cap — would sell
into strength and pay a fee for the privilege, on a schedule set by volatility
rather than by any view. What the cap prevents is *deciding* to take a larger
position, which is the actual risk. Strategies that rebalance regularly hold the
drift down on their own as a side effect.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import date

import pandas as pd

from ai_finance.strategy.base import Signal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits, applied to every order with no override flag.

    Defaults are the initial values from ``docs/RISK.md``.
    """

    max_position_weight: float = 0.25
    max_total_exposure: float = 1.0
    allow_short: bool = False
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    min_order_notional: float = 20.0
    max_orders_per_hour: int = 20
    scale_by_confidence: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in [0, 1]")
        if self.max_total_exposure < 0:
            raise ValueError("max_total_exposure must be non-negative")
        if self.max_daily_loss <= 0 or self.max_drawdown <= 0:
            raise ValueError("loss limits must be positive")
        if self.min_order_notional < 0:
            raise ValueError("min_order_notional must be non-negative")
        if self.max_orders_per_hour < 1:
            raise ValueError("max_orders_per_hour must be at least 1")

    @classmethod
    def unconstrained(cls) -> RiskLimits:
        """Limits that never bind. **For engine tests only.**

        The gate tests need to prove the *engine* reproduces a known return, so
        they must not have the risk layer capping position size at 25%. Nothing
        that informs a capital decision should ever use this.
        """
        return cls(
            max_position_weight=1.0,
            max_total_exposure=1.0,
            allow_short=True,
            max_daily_loss=1e9,
            max_drawdown=1e9,
            min_order_notional=0.0,
            max_orders_per_hour=1_000_000,
        )

    def allowing_short(self) -> RiskLimits:
        return replace(self, allow_short=True)


@dataclass
class RiskDecision:
    """The outcome of applying the limits to one signal."""

    target_weight: float
    adjustments: tuple[str, ...] = ()

    @property
    def was_adjusted(self) -> bool:
        return bool(self.adjustments)


@dataclass
class RiskEngine:
    """Stateful: tracks peak equity, the day's starting equity, and order rate."""

    limits: RiskLimits = field(default_factory=RiskLimits)

    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_day: object = None
    halted_permanently: bool = False
    halted_until: pd.Timestamp | None = None
    halt_reason: str = ""
    _order_times: deque[pd.Timestamp] = field(default_factory=deque, repr=False)

    def observe(self, equity: float, timestamp: pd.Timestamp) -> None:
        """Update state from the latest mark, and trip halts if breached.

        Called once per bar, before the strategy is consulted.
        """
        if self.peak_equity == 0.0:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        day = timestamp.date()
        if self.current_day != day:
            self.current_day = day
            self.day_start_equity = equity

        if self.peak_equity > 0:
            drawdown = 1.0 - equity / self.peak_equity
            if drawdown >= self.limits.max_drawdown and not self.halted_permanently:
                self.halted_permanently = True
                self.halt_reason = (
                    f"max drawdown {drawdown * 100:.1f}% >= "
                    f"{self.limits.max_drawdown * 100:.1f}% — manual restart required"
                )
                log.warning("risk halt (permanent): %s", self.halt_reason)
                return

        if self.day_start_equity > 0:
            daily_loss = 1.0 - equity / self.day_start_equity
            if daily_loss >= self.limits.max_daily_loss and not self.is_halted(timestamp):
                self.halted_until = timestamp + pd.Timedelta(hours=24)
                self.halt_reason = (
                    f"daily loss {daily_loss * 100:.1f}% >= "
                    f"{self.limits.max_daily_loss * 100:.1f}% — halted for 24h"
                )
                log.warning("risk halt (24h): %s", self.halt_reason)

    def is_halted(self, timestamp: pd.Timestamp) -> bool:
        if self.halted_permanently:
            return True
        return self.halted_until is not None and timestamp < self.halted_until

    def apply(self, signal: Signal, timestamp: pd.Timestamp) -> RiskDecision:
        """Clamp ``signal`` to the limits. A halt forces a flat target."""
        if self.is_halted(timestamp):
            return RiskDecision(0.0, (f"halted: {self.halt_reason}",))

        weight = signal.target_weight
        adjustments: list[str] = []

        if self.limits.scale_by_confidence and signal.confidence < 1.0:
            weight *= signal.confidence
            adjustments.append(f"scaled by confidence {signal.confidence:.2f}")

        if weight < 0 and not self.limits.allow_short:
            adjustments.append("shorting disabled; forced flat")
            weight = 0.0

        cap = min(self.limits.max_position_weight, self.limits.max_total_exposure)
        if abs(weight) > cap:
            adjustments.append(f"capped at {cap:.2f} (was {weight:+.2f})")
            weight = cap if weight > 0 else -cap

        return RiskDecision(weight, tuple(adjustments))

    def permits_order(self, notional: float, timestamp: pd.Timestamp) -> tuple[bool, str]:
        """Whether an order of ``notional`` may be sent right now."""
        if notional < self.limits.min_order_notional:
            return False, (
                f"notional {notional:.2f} below minimum "
                f"{self.limits.min_order_notional:.2f}; fees would dominate"
            )

        cutoff = timestamp - pd.Timedelta(hours=1)
        while self._order_times and self._order_times[0] < cutoff:
            self._order_times.popleft()
        if len(self._order_times) >= self.limits.max_orders_per_hour:
            return False, f"order rate limit {self.limits.max_orders_per_hour}/hour reached"

        return True, ""

    def record_order(self, timestamp: pd.Timestamp) -> None:
        self._order_times.append(timestamp)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, object]:
        """Serialisable state, so a halt survives a process restart.

        A drawdown halt that evaporated when the scheduled job next started
        would be worse than having no halt at all: it would look like a working
        safety mechanism while quietly resetting itself every few hours.
        """
        return {
            "peak_equity": self.peak_equity,
            "day_start_equity": self.day_start_equity,
            "current_day": self.current_day.isoformat() if self.current_day else None,
            "halted_permanently": self.halted_permanently,
            "halted_until": self.halted_until.isoformat() if self.halted_until else None,
            "halt_reason": self.halt_reason,
            "recent_order_times": [t.isoformat() for t in self._order_times],
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Load state produced by :meth:`snapshot`."""
        self.peak_equity = float(snapshot.get("peak_equity") or 0.0)
        self.day_start_equity = float(snapshot.get("day_start_equity") or 0.0)
        day = snapshot.get("current_day")
        self.current_day = date.fromisoformat(str(day)) if day else None
        self.halted_permanently = bool(snapshot.get("halted_permanently"))
        until = snapshot.get("halted_until")
        self.halted_until = pd.Timestamp(str(until)) if until else None
        self.halt_reason = str(snapshot.get("halt_reason") or "")
        self._order_times = deque(
            pd.Timestamp(str(t)) for t in (snapshot.get("recent_order_times") or [])
        )
