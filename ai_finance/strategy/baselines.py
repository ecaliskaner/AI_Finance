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

from ai_finance.strategy import indicators
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


class _TargetTracker:
    """Emits a signal only when the desired weight actually changes.

    The engine treats ``None`` as "no change", so a strategy that re-states the
    same target every bar would generate a small rebalancing order every bar as
    the position drifts with price. That is a real design choice, not a detail:
    continuously rebalancing to a target costs fees on a schedule set by
    volatility, which is exactly the churn ``docs/PLAN.md`` §1 is about. These
    baselines take a position and let it ride, which also makes them directly
    comparable to buy-and-hold.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold
        self._target: float | None = None

    def emit(self, symbol: str, weight: float, reason: str) -> Signal | None:
        if self._target is not None and abs(weight - self._target) <= self._threshold:
            return None
        self._target = weight
        return Signal(symbol=symbol, target_weight=weight, reason=reason)


class MovingAverageCrossover(_TargetTracker):
    """Long while the fast average is above the slow one, flat otherwise.

    The oldest trend-following rule there is, and the first thing any new
    strategy has to beat.
    """

    name = "ma-crossover"

    def __init__(self, fast: int = 20, slow: int = 50, weight: float = 1.0) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.weight = weight

    def on_bar(self, state: MarketState) -> Signal | None:
        closes = state.history.last("close", self.slow)
        if len(closes) < self.slow:
            return None

        fast_ma = indicators.sma(closes, self.fast)
        slow_ma = indicators.sma(closes, self.slow)
        if not (np.isfinite(fast_ma) and np.isfinite(slow_ma)):
            return None

        above = fast_ma > slow_ma
        return self.emit(
            state.symbol,
            self.weight if above else 0.0,
            f"MA{self.fast}={fast_ma:.2f} {'>' if above else '<='} MA{self.slow}={slow_ma:.2f}",
        )


class RSIMeanReversion(_TargetTracker):
    """Buy oversold, sell back into strength.

    The counterpart to trend following: it profits when trend following does
    not, which is why both belong in the baseline set. A strategy that only
    beats one of them has beaten a regime, not the market.
    """

    name = "rsi-mean-reversion"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        exit_level: float = 50.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0 < oversold < exit_level < 100:
            raise ValueError("need 0 < oversold < exit_level < 100")
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level
        self.weight = weight
        #  Wilder smoothing is recursive; feed it well past the period so the
        #  value does not depend on where the backtest happened to start.
        self._history = period * 5

    def on_bar(self, state: MarketState) -> Signal | None:
        closes = state.history.last("close", self._history)
        value = indicators.rsi(closes, self.period)
        if not np.isfinite(value):
            return None

        if value < self.oversold:
            return self.emit(state.symbol, self.weight, f"RSI {value:.1f} < {self.oversold}")
        if value > self.exit_level:
            return self.emit(state.symbol, 0.0, f"RSI {value:.1f} > {self.exit_level}")
        return None


class Breakout(_TargetTracker):
    """Donchian channel: long on a new high, out on a new low."""

    name = "breakout"

    def __init__(self, lookback: int = 20, exit_lookback: int = 10, weight: float = 1.0) -> None:
        super().__init__()
        if lookback < 2 or exit_lookback < 2:
            raise ValueError("lookbacks must be at least 2")
        self.lookback = lookback
        self.exit_lookback = exit_lookback
        self.weight = weight

    def on_bar(self, state: MarketState) -> Signal | None:
        needed = max(self.lookback, self.exit_lookback) + 1
        highs = state.history.last("high", needed)
        lows = state.history.last("low", needed)
        if len(highs) < needed:
            return None

        upper, _ = indicators.donchian(highs, lows, self.lookback)
        _, lower = indicators.donchian(highs, lows, self.exit_lookback)
        close = state.bar.close

        if close > upper:
            return self.emit(state.symbol, self.weight, f"close {close:.2f} broke {upper:.2f}")
        if close < lower:
            return self.emit(state.symbol, 0.0, f"close {close:.2f} broke down {lower:.2f}")
        return None


class VolatilityScaledTrend(_TargetTracker):
    """Long when the trend is up, sized so that risk stays roughly constant.

    Position size is ``target_vol / realized_vol``, so a calm market gets a
    bigger position than a turbulent one. Constant *dollar* sizing takes far
    more risk in turbulent markets than calm ones, which is backwards.

    ``rebalance_threshold`` is a no-trade band. Without one, a continuously
    varying target means a small trade every single bar, and the fees would eat
    the strategy for reasons that have nothing to do with whether the signal
    works.
    """

    name = "vol-scaled-trend"

    def __init__(
        self,
        lookback: int = 50,
        vol_window: int = 50,
        target_vol: float = 0.20,
        max_weight: float = 1.0,
        rebalance_threshold: float = 0.05,
    ) -> None:
        super().__init__(threshold=rebalance_threshold)
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_weight = max_weight

    def on_bar(self, state: MarketState) -> Signal | None:
        needed = max(self.lookback, self.vol_window) + 1
        closes = state.history.last("close", needed)
        if len(closes) < needed:
            return None

        trend = indicators.momentum(closes, self.lookback)
        vol = indicators.realized_volatility(closes, self.vol_window, state.periods_per_year)
        if not (np.isfinite(trend) and np.isfinite(vol)) or vol <= 0:
            return None

        if trend <= 0:
            return self.emit(state.symbol, 0.0, f"{self.lookback}-bar momentum {trend:+.2%} <= 0")

        weight = min(self.max_weight, self.target_vol / vol)
        return self.emit(
            state.symbol,
            weight,
            f"momentum {trend:+.2%}, vol {vol:.1%} -> weight {weight:.2f}",
        )


#: Parameter grids for walk-forward selection. Deliberately small: every extra
#: combination is another lottery ticket in the multiple-testing count that
#: :mod:`ai_finance.research.registry` keeps.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "ma-crossover": {"fast": [10, 20, 40], "slow": [50, 100, 200]},
    "rsi-mean-reversion": {"period": [7, 14, 21], "oversold": [20.0, 30.0]},
    "breakout": {"lookback": [20, 55], "exit_lookback": [10, 20]},
    "vol-scaled-trend": {"lookback": [20, 50, 100], "target_vol": [0.15, 0.30]},
}

#: Every strategy the research tooling knows how to build, by name.
STRATEGY_FACTORIES = {
    "ma-crossover": MovingAverageCrossover,
    "rsi-mean-reversion": RSIMeanReversion,
    "breakout": Breakout,
    "vol-scaled-trend": VolatilityScaledTrend,
}


def warmup_bars_for(name: str, params: dict) -> int:
    """How many bars a strategy needs before it can produce its first signal.

    Walk-forward runs feed this many extra bars in front of each window, so a
    strategy is not blind for the first stretch of every test period.
    """
    if name == "ma-crossover":
        return int(params.get("slow", 50)) + 1
    if name == "rsi-mean-reversion":
        return int(params.get("period", 14)) * 5 + 1
    if name == "breakout":
        return max(int(params.get("lookback", 20)), int(params.get("exit_lookback", 10))) + 2
    if name == "vol-scaled-trend":
        return max(int(params.get("lookback", 50)), int(params.get("vol_window", 50))) + 2
    raise ValueError(f"unknown strategy {name!r}")
