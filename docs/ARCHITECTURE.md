# Architecture

## Core principle

**One strategy code path, three execution modes.**

```
                      ┌──────────────────┐
                      │  Strategy code   │   ← written once
                      │  (the only place │
                      │   decisions are  │
                      │   made)          │
                      └────────┬─────────┘
                               │  emits Signal objects
                      ┌────────▼─────────┐
                      │   Risk engine    │   ← can veto or shrink any order
                      └────────┬─────────┘
                               │  emits Order objects
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  ┌───────────┐         ┌───────────┐          ┌───────────┐
  │ Backtest  │         │   Paper   │          │   Live    │
  │  adapter  │         │  adapter  │          │  adapter  │
  │ (historic │         │(live data,│          │(live data,│
  │  bars)    │         │ fake fill)│          │ real fill)│
  └───────────┘         └───────────┘          └───────────┘
```

The strategy cannot tell which adapter it's running under. This is what makes
backtest results meaningful: if backtest and live diverge, it is the *cost
model* that is wrong, and that is a finding, not a mystery.

The risk engine sits between strategy and execution in all three modes,
including backtest. A strategy that would be blocked live must also be blocked
in the backtest, or the backtest is measuring a strategy you will never run.

**`paper` is the default mode.** `live` requires an explicit flag and separate
credentials, and `execution/live.py` is the last module written — so before
Phase 5 there is no code path to a real order at all. See PLAN.md §2.

## Data flow

```
Binance API ──► raw Parquet (immutable) ──► feature pipeline ──► model/strategy
                      │                            │
                      ▼                            ▼
              data quality report          feature store (cached,
              (gaps, dupes, outliers)       always rebuildable)
```

Raw data is never edited. Features are always recomputed from raw. When a
feature bug is found months later, everything downstream can be rebuilt.

## Module layout

```
ai_finance/
├── data/
│   ├── fetch.py          # Binance REST/WS clients, rate limit handling
│   ├── store.py          # Parquet read/write, load_bars() — the one entry point
│   └── quality.py        # gap detection, outlier flags, integrity report
├── features/
│   ├── technical.py      # returns, volatility, volume, RSI etc.
│   ├── microstructure.py # spread, order book imbalance, funding rate
│   └── pipeline.py       # feature assembly, caching, point-in-time correctness
├── strategy/
│   ├── base.py           # Strategy interface: on_bar(state) -> Signal
│   ├── indicators.py     # SMA, RSI, Donchian, realised vol — scalar per bar
│   ├── baselines.py      # MA cross, RSI, breakout — the things to beat
│   └── ml.py             # model prediction → Signal, via explicit policy layer
├── risk/
│   ├── engine.py         # veto/resize logic, all limits enforced here
│   └── sizing.py         # position sizing rules
├── execution/
│   ├── base.py           # ExecutionAdapter interface
│   ├── backtest.py       # simulated fills against historical bars
│   ├── paper.py          # live data, simulated fills
│   └── live.py           # real orders (Phase 5, last thing written)
├── backtest/
│   ├── engine.py         # the event loop
│   ├── costs.py          # fee + spread + slippage model
│   ├── ledger.py         # portfolio + trade accounting from cash flows
│   └── metrics.py        # returns, Sharpe, drawdown, fees paid, vs benchmark
├── research/
│   ├── walkforward.py    # rolling fit/test splits, purge + embargo
│   └── registry.py       # experiment log — every run recorded
├── report.py             # standalone HTML: equity, drawdown, costs
├── demo.py               # guided end-to-end run, no network needed
└── ops/
    ├── runner.py         # the scheduled job: sync, decide, act, persist, exit
    ├── state.py          # crash-safe position state, atomic writes
    ├── alerts.py         # Telegram notifications, file-based kill switch
    └── monitor.py        # heartbeats and missed-run detection
```

Built so far (Phases 0–4): all of `data/`, all of `strategy/`, `risk/engine.py`,
`execution/base.py`, `execution/backtest.py`, `execution/paper.py`, all of
`backtest/`, all of `research/`, all of `ops/` (with `runner.py` in place of the
sketch's `state.py`-only listing), and `features/` (as `technical.py` and
`pipeline.py` rather than the three modules sketched above — order-book and
funding features need data the store does not hold). Still to come:
`risk/sizing.py` and `execution/live.py`, both Phase 5.

## Tech stack

| Layer | Choice | Why this one |
|---|---|---|
| Language | Python 3.11+ | Where the entire quant ecosystem lives |
| Deps | `uv` | Much faster than pip/poetry, simpler to learn |
| Exchange API | `ccxt` | One interface across exchanges; swapping venues later is cheap |
| Data | `pandas` + `pyarrow` | pandas for familiarity; Parquet for compact columnar storage |
| Storage | Parquet files, partitioned by symbol/month | No database to run; DuckDB can query the files directly if needed |
| Backtest | **Custom, ~300 lines** | Vectorized libraries (vectorbt, backtrader) make look-ahead bias easy and invisible. Writing the loop yourself is the single best way to actually understand what a backtest claims |
| ML | `scikit-learn` only | Ridge first for interpretability, then `HistGradientBoostingRegressor` — the same histogram-based algorithm LightGBM popularised, with one fewer dependency. No deep learning — see PLAN.md Phase 3 |
| Validation | Custom walk-forward | `sklearn`'s standard CV leaks across time and will flatter every model |
| Scheduling | `cron` or a systemd timer | A 4-hour-horizon strategy needs no persistent process. Each run is short-lived and stateless, which is far less to get wrong than an always-on async service |
| Deploy | Docker on a small VPS | Reproducible, restartable, ~$5–10/month |
| Alerts | Telegram bot | Free, reliable, works from a phone, and supports the kill switch |
| Logs | `structlog` → JSON lines | Machine-readable, so post-mortems are greppable |

Deliberately excluded for now: Kubernetes, message queues, a web dashboard, a
database server, microservices, WebSocket streaming, async. A short-lived Python
process on one small VPS handles this workload. Every component added is a
component that can fail at 3am while holding a position.

## Key interfaces

```python
@dataclass(frozen=True)
class Signal:
    """What the strategy wants. Not yet an order."""

    symbol: str
    target_weight: float  # -1.0 to 1.0, fraction of equity
    confidence: float  # drives sizing in the risk layer
    reason: str  # human-readable, logged on every signal


class Strategy(Protocol):
    def on_bar(self, state: MarketState) -> Signal | None: ...


class ExecutionAdapter(Protocol):
    def submit(self, order: Order) -> Fill | None: ...
    def positions(self) -> dict[str, Position]: ...
    def equity(self) -> float: ...
```

`MarketState` exposes only data available at that timestamp. This is enforced in
the type, not by convention — it's the primary defense against look-ahead bias,
and convention is not a defense.

`reason` being mandatory is intentional. Every trade the system takes must be
explainable after the fact, including at Phase 6 if an RL policy is ever added.

## Operational requirements (from Phase 4)

Short because the system is a scheduled job rather than a persistent service.
Most of what would otherwise be on this list is handled by the process simply
not existing between runs.

- **The exchange is the source of truth.** Every run begins by reading actual
  positions from the exchange. Local state is a cache, never an authority. This
  dissolves most of the crash-recovery problem: a process that died mid-run just
  leaves the next run some reconciling to do.
- **Idempotent orders.** Client order IDs prevent a retry after a network
  timeout from becoming a duplicate position.
- **Missed-run detection.** The dangerous failure here is a scheduled run that
  silently didn't happen. Each run writes a heartbeat; a separate check alerts if
  the newest one is older than the cadence allows.
- **Kill switch.** A Telegram command that flattens all positions and halts
  trading, reachable from a phone, tested regularly.
- **Daily reconciliation.** Compare what the backtest would have done on the same
  bars against what actually happened. Every mismatch gets investigated.
- **Mode is always explicit.** `backtest` | `paper` | `live`, defaulting to
  `paper`.
