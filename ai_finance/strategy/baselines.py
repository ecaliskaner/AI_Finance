"""Reference strategies.

Three of these exist to test the engine rather than to make money:

- :class:`BuyAndHold` — the benchmark, and the engine's calibration target. Run
  with zero costs it must reproduce the asset's price return exactly.
- :class:`RandomStrategy` — has no edge by construction, so whatever it loses
  should be the cost model and nothing else.
- :class:`AlwaysFlat` — must return exactly zero and pay nothing.

The real baselines a strategy has to beat (moving-average crossover, RSI mean
reversion, breakout) arrive in Phase 2, once walk-forward validation exists to
evaluate them honestly.
"""

from __future__ import annotations

import numpy as np

from ai_finance.strategy.base import MarketState, Signal


class BuyAndHold:
    """Buy on the first bar, then never trade again."""

    name = "buy-and-hold"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
        self._entered = False

    def on_bar(self, state: MarketState) -> Signal | None:
        if self._entered:
            return None
        self._entered = True
        return Signal(
            symbol=state.symbol,
            target_weight=self.weight,
            reason="buy and hold: initial entry",
        )


class AlwaysFlat:
    """Never holds anything. The do-nothing control."""

    name = "always-flat"

    def on_bar(self, state: MarketState) -> Signal | None:
        return None


class RandomStrategy:
    """Flips a coin every ``every_n_bars`` and goes long or flat.

    Has no edge by construction. Its purpose is to isolate the cost model: run
    it and the losses should be explained by fees and spread, not by anything
    the strategy did.
    """

    name = "random"

    def __init__(self, *, seed: int = 0, every_n_bars: int = 1, long_probability: float = 0.5):
        if every_n_bars < 1:
            raise ValueError("every_n_bars must be at least 1")
        if not 0.0 <= long_probability <= 1.0:
            raise ValueError("long_probability must be in [0, 1]")
        self._rng = np.random.default_rng(seed)
        self._every = every_n_bars
        self._p = long_probability

    def on_bar(self, state: MarketState) -> Signal | None:
        if state.index % self._every != 0:
            return None
        go_long = self._rng.random() < self._p
        return Signal(
            symbol=state.symbol,
            target_weight=1.0 if go_long else 0.0,
            reason="random: coin flip",
        )


class TargetWeightSchedule:
    """Holds a fixed weight from a given bar index. Used to test fill timing."""

    name = "scheduled-weight"

    def __init__(self, schedule: dict[int, float]) -> None:
        self.schedule = schedule

    def on_bar(self, state: MarketState) -> Signal | None:
        if state.index not in self.schedule:
            return None
        return Signal(
            symbol=state.symbol,
            target_weight=self.schedule[state.index],
            reason=f"scheduled weight at bar {state.index}",
        )
