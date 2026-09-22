"""The scheduled job: one wake-up, one decision, one exit.

This is what `docs/PLAN.md` §1.5 buys by dropping the 24/7 requirement. There is
no event loop, no WebSocket, no long-lived process holding a position in memory.
A timer fires, this function runs for a few seconds, and the machine goes back to
doing nothing. If it crashes there is nothing to recover, because there was
nothing running.

Order of operations, mirroring the backtest exactly:

1. Sync new 1-minute bars into the store (incremental — the first run backfills,
   later ones fetch a few hundred bars).
2. Load the decision interval, resampled on read. Incomplete periods are already
   dropped, so the newest bar is genuinely closed.
3. If that bar has already been acted on, stop. This is what makes the job safe
   to run on an overlapping schedule or retry after a failure.
4. Replay the visible history through the strategy and take its current target.
   Replaying rather than recomputing indicators live is what guarantees the live
   signal equals the one the backtest produced.
5. Apply the risk limits, restored from disk so a halt survives a restart.
6. Fill at the current price, which is where the next bar opens — the same
   convention the backtest uses.

**Known limitation.** During replay, ``MarketState.equity`` and
``position_weight`` carry today's values at every historical bar rather than the
values of that moment. None of the shipped baselines read them, so the replayed
signal is identical to the backtest's. A strategy that *did* read them would
diverge, and would need its position state threaded through the replay before it
could be trusted live.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_finance.backtest.costs import CostModel
from ai_finance.config import bars_dir, data_dir, interval_ms
from ai_finance.data.fetch import fetch_history
from ai_finance.data.sources import BarSource, BinanceSource
from ai_finance.data.store import load_bars
from ai_finance.execution.base import Fill, Order
from ai_finance.execution.paper import PaperExecution
from ai_finance.ops.alerts import Notifier, NullNotifier, halt_reason, is_halted
from ai_finance.ops.monitor import write_heartbeat
from ai_finance.ops.state import RunState, StateStore, default_state_path
from ai_finance.risk.engine import RiskEngine, RiskLimits
from ai_finance.strategy.base import Bar, BarWindow, MarketState
from ai_finance.strategy.baselines import STRATEGY_FACTORIES, warmup_bars_for

log = logging.getLogger(__name__)

SUPPORTED_MODES = ("paper",)

#: Extra bars fetched beyond the strategy's warm-up, so an indicator is never
#: computed from the bare minimum.
HISTORY_MARGIN_BARS = 60


@dataclass(frozen=True)
class RunOutcome:
    """What one wake-up did."""

    action: str
    symbol: str
    mode: str
    state: RunState
    bar_time: pd.Timestamp | None = None
    price: float | None = None
    target_weight: float | None = None
    fill: Fill | None = None
    equity: float = 0.0
    message: str = ""

    @property
    def traded(self) -> bool:
        return self.fill is not None

    def to_text(self) -> str:
        lines = [f"{self.symbol} [{self.mode}] {self.action}"]
        if self.bar_time is not None:
            lines.append(f"  bar       {self.bar_time:%Y-%m-%d %H:%M} UTC")
        if self.price is not None:
            lines.append(f"  price     {self.price:,.2f}")
        if self.target_weight is not None:
            lines.append(f"  target    {self.target_weight:.2%} of equity")
        lines.append(f"  position  {self.state.units:.8f} units")
        lines.append(f"  equity    {self.equity:,.2f}")
        if self.fill is not None:
            lines.append(
                f"  FILLED    {'buy' if self.fill.delta_units > 0 else 'sell'} "
                f"{abs(self.fill.delta_units):.8f} at {self.fill.fill_price:,.2f} "
                f"(cost {self.fill.total_cost:,.2f})"
            )
        if self.message:
            lines.append(f"  {self.message}")
        return "\n".join(lines)


def run_once(
    *,
    symbol: str,
    interval: str,
    strategy_name: str,
    mode: str = "paper",
    strategy_params: dict | None = None,
    source: BarSource | None = None,
    costs: CostModel | None = None,
    limits: RiskLimits | None = None,
    initial_equity: float = 10_000.0,
    root: Path | None = None,
    state_path: Path | None = None,
    notifier: Notifier | None = None,
    now: pd.Timestamp | None = None,
    sync: bool = True,
) -> RunOutcome:
    """Wake up, decide, act, persist, exit.

    Args:
        mode: only ``"paper"`` is implemented. ``"live"`` raises — there is no
            code path to a real order before Phase 5, which is a stronger
            guarantee than a configuration flag.
        source: where live bars come from. Defaults to Binance.
        sync: fetch new bars before deciding. Turn it off to re-run a decision
            against data already stored.

    Raises:
        NotImplementedError: for ``mode="live"``.
        ValueError: for an unknown mode or strategy.
    """
    if mode == "live":
        raise NotImplementedError(
            "live trading is not implemented. execution/live.py is the last module "
            "written (docs/PLAN.md §2), so before Phase 5 there is no code path to a "
            "real order at all. Run with --mode paper."
        )
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {SUPPORTED_MODES}")
    if strategy_name not in STRATEGY_FACTORIES:
        raise ValueError(
            f"unknown strategy {strategy_name!r}; expected one of {sorted(STRATEGY_FACTORIES)}"
        )

    symbol = symbol.upper()
    params = dict(strategy_params or {})
    costs = costs if costs is not None else CostModel()
    limits = limits if limits is not None else RiskLimits()
    notifier = notifier if notifier is not None else NullNotifier()
    moment = now if now is not None else pd.Timestamp.now("UTC")
    root = root if root is not None else data_dir()

    store = StateStore(state_path or default_state_path(symbol, mode))
    state = store.initialise(
        RunState(
            symbol=symbol,
            interval=interval,
            mode=mode,
            strategy=strategy_name,
            initial_equity=initial_equity,
            cash=initial_equity,
        )
    )

    warmup = warmup_bars_for(strategy_name, params)
    if sync:
        try:
            _sync_bars(source, symbol, interval, warmup, state, moment)
        except Exception as exc:
            # Could not reach fresh data. Do not fall back to whatever is in the
            # store: acting on stale prices is how a system keeps trading
            # confidently through an exchange outage. Hold, and say so loudly.
            log.error("data sync failed for %s: %s", symbol, exc)
            return _finish(
                store,
                state,
                notifier,
                RunOutcome(
                    action="sync-failed",
                    symbol=symbol,
                    mode=mode,
                    state=state,
                    message=(
                        f"could not fetch fresh bars ({exc}). Holding the current "
                        "position rather than acting on stale prices."
                    ),
                ),
                root,
            )

    bars = load_bars(symbol, interval=interval, root=bars_dir())
    if len(bars) < warmup + 2:
        return _finish(
            store,
            state,
            notifier,
            RunOutcome(
                action="insufficient-data",
                symbol=symbol,
                mode=mode,
                state=state,
                message=(
                    f"{len(bars)} {interval} bars stored, need {warmup + 2} for "
                    f"{strategy_name}. Back-fill more history."
                ),
            ),
            root,
            quiet=True,
        )

    bar_close = pd.Timestamp(bars["close_time"].iloc[-1])
    price = _latest_price(symbol, moment)
    equity = state.equity(price)

    if state.has_acted_on(bar_close):
        return _finish(
            store,
            state,
            notifier,
            RunOutcome(
                action="already-done",
                symbol=symbol,
                mode=mode,
                state=state,
                bar_time=bar_close,
                price=price,
                equity=equity,
                message="this bar was already handled; nothing to do",
            ),
            root,
            quiet=True,
        )

    engine = RiskEngine(limits=limits)
    engine.restore(
        {
            "peak_equity": state.peak_equity,
            "day_start_equity": state.day_start_equity,
            "current_day": state.current_day,
            "halted_permanently": state.halted_permanently,
            "halted_until": state.halted_until,
            "halt_reason": state.halt_reason,
            "recent_order_times": state.recent_order_times,
        }
    )
    engine.observe(equity, moment)

    target, reason = _current_target(bars, strategy_name, params, symbol, equity, state, price)

    manual_halt = is_halted(root)
    if manual_halt:
        target, reason = 0.0, f"kill switch engaged: {halt_reason(root) or 'no reason given'}"
    elif engine.is_halted(moment):
        target, reason = 0.0, f"risk halt: {engine.halt_reason}"
    elif target is None:
        return _finish(
            store,
            _persist_risk(state, engine, moment, bar_close),
            notifier,
            RunOutcome(
                action="no-signal",
                symbol=symbol,
                mode=mode,
                state=state,
                bar_time=bar_close,
                price=price,
                equity=equity,
                message="strategy has no opinion yet",
            ),
            root,
            quiet=True,
        )
    else:
        decision = engine.apply(_signal(symbol, target, reason), moment)
        target, reason = decision.target_weight, "; ".join([reason, *decision.adjustments])

    fill, note = _rebalance(
        symbol=symbol,
        target_weight=target,
        price=price,
        state=state,
        engine=engine,
        costs=costs,
        moment=moment,
        reason=reason,
        block_new_positions=manual_halt,
    )

    if fill is not None:
        state.cash -= fill.cash_flow
        state.units += fill.delta_units
        state.n_fills += 1
        state.total_fees += fill.fee
        state.total_concession += fill.price_concession
        engine.record_order(moment)

    equity = state.equity(price)
    state = _persist_risk(state, engine, moment, bar_close)

    return _finish(
        store,
        state,
        notifier,
        RunOutcome(
            action="traded" if fill is not None else "held",
            symbol=symbol,
            mode=mode,
            state=state,
            bar_time=bar_close,
            price=price,
            target_weight=target,
            fill=fill,
            equity=equity,
            message=note or reason,
        ),
        root,
    )


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _signal(symbol: str, weight: float, reason: str):
    from ai_finance.strategy.base import Signal

    return Signal(symbol=symbol, target_weight=weight, reason=reason)


def _sync_bars(
    source: BarSource | None,
    symbol: str,
    interval: str,
    warmup: int,
    state: RunState,
    moment: pd.Timestamp,
) -> None:
    """Bring the 1-minute store up to date, back-filling on the first run."""
    feed = source if source is not None else BinanceSource()
    needed_bars = warmup + HISTORY_MARGIN_BARS
    span = pd.Timedelta(milliseconds=interval_ms(interval) * needed_bars)
    start = moment - span

    if state.last_bar_time is not None:
        # Already warm: only the newest bars are missing, but keep a generous
        # overlap so a gap from a skipped run gets filled rather than stepped over.
        start = min(start, pd.Timestamp(state.last_bar_time) - span)

    result = fetch_history(feed, symbol, start=start, interval="1m", root=bars_dir())
    log.info("sync: %s", result.summary())


def _latest_price(symbol: str, moment: pd.Timestamp) -> float:
    """Most recent traded price, from the newest stored 1-minute bar.

    Up to a minute stale, which is the right order of magnitude for a job that
    wakes every four hours. The backtest fills at the next bar's open; running
    promptly after a bar closes, the current price *is* that open.
    """
    recent = load_bars(symbol, start=moment - pd.Timedelta(days=2), interval="1m", root=bars_dir())
    if recent.empty:
        raise ValueError(f"no recent 1-minute bars for {symbol}; cannot price a fill")
    return float(recent["close"].iloc[-1])


def _current_target(
    bars: pd.DataFrame,
    strategy_name: str,
    params: dict,
    symbol: str,
    equity: float,
    state: RunState,
    price: float,
) -> tuple[float | None, str]:
    """Replay history through a fresh strategy and return its standing target.

    The strategy emits only when its target *changes*, so the last bar usually
    produces nothing. What matters is the target accumulated across the replay —
    exactly what the backtest's pending-weight variable holds.
    """
    strategy = STRATEGY_FACTORIES[strategy_name](**params)
    arrays = {
        name: bars[name].to_numpy(dtype=np.float64)
        for name in ("open", "high", "low", "close", "volume")
    }
    trades = bars["trades"].to_numpy(dtype=np.int64)
    open_time = bars["open_time"].to_numpy()
    close_time = bars["close_time"].to_numpy()
    bar_seconds = (bars["close_time"].iloc[0] - bars["open_time"].iloc[0]).total_seconds() + 0.001

    target: float | None = None
    reason = ""
    weight = state.weight(price)

    for i in range(len(bars)):
        market = MarketState(
            symbol=symbol,
            index=i,
            bar=Bar(
                open_time=pd.Timestamp(open_time[i]),
                close_time=pd.Timestamp(close_time[i]),
                open=float(arrays["open"][i]),
                high=float(arrays["high"][i]),
                low=float(arrays["low"][i]),
                close=float(arrays["close"][i]),
                volume=float(arrays["volume"][i]),
                trades=int(trades[i]),
            ),
            history=BarWindow(arrays, i),
            equity=equity,
            position_weight=weight,
            bar_seconds=bar_seconds,
        )
        signal = strategy.on_bar(market)
        if signal is not None:
            target, reason = signal.target_weight, signal.reason

    return target, reason


def _rebalance(
    *,
    symbol: str,
    target_weight: float,
    price: float,
    state: RunState,
    engine: RiskEngine,
    costs: CostModel,
    moment: pd.Timestamp,
    reason: str,
    block_new_positions: bool,
) -> tuple[Fill | None, str]:
    """Move the position toward ``target_weight`` at ``price``."""
    equity = state.equity(price)
    if equity <= 0:
        return None, "equity is zero or negative; no further trading"

    target_units = target_weight * equity / price
    delta = target_units - state.units

    if block_new_positions and abs(target_units) > abs(state.units):
        return None, "kill switch engaged: will reduce a position but never add to one"

    if delta > 0:
        affordable = state.cash / (costs.fill_price(price, 1.0) * (1.0 + costs.fee_rate))
        delta = min(delta, max(0.0, affordable))

    if delta == 0.0:
        return None, reason

    notional = abs(delta) * price
    permitted, why = engine.permits_order(notional, moment)
    if not permitted:
        return None, why

    fill = PaperExecution(costs).submit(
        Order(
            symbol=symbol,
            delta_units=delta,
            reference_price=price,
            timestamp=moment,
            reason=reason,
        )
    )
    return fill, reason


def _persist_risk(
    state: RunState, engine: RiskEngine, moment: pd.Timestamp, bar_close: pd.Timestamp
) -> RunState:
    snapshot = engine.snapshot()
    return replace(
        state,
        last_bar_time=bar_close.isoformat(),
        last_run_at=moment.isoformat(),
        n_runs=state.n_runs + 1,
        peak_equity=float(snapshot["peak_equity"]),
        day_start_equity=float(snapshot["day_start_equity"]),
        current_day=snapshot["current_day"],
        halted_permanently=bool(snapshot["halted_permanently"]),
        halted_until=snapshot["halted_until"],
        halt_reason=str(snapshot["halt_reason"]),
        recent_order_times=list(snapshot["recent_order_times"]),
    )


def _finish(
    store: StateStore,
    state: RunState,
    notifier: Notifier,
    outcome: RunOutcome,
    root: Path,
    *,
    quiet: bool = False,
) -> RunOutcome:
    """Persist, heartbeat, notify. Always the last thing a run does."""
    outcome = replace(outcome, state=state)
    store.save(state)
    write_heartbeat(
        outcome.symbol, outcome.mode, action=outcome.action, note=outcome.message, root=root
    )
    if not quiet:
        notifier.send(outcome.to_text())
    return outcome
