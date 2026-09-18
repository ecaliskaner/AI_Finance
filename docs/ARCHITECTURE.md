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
│   └── metrics.py        # returns, Sharpe, drawdown, fees paid, vs benchmark
├── research/
│   ├── walkforward.py    # rolling fit/test splits, purge + embargo
│   └── registry.py       # experiment log — every run recorded
└── ops/
    ├── state.py          # crash-safe persistence of positions and orders
    ├── alerts.py         # Telegram notifications, kill switch listener
    └── monitor.py        # health checks, daily P&L summary
```

## Tech stack

| Layer | Choice | Why this one |
|---|---|---|
| Language | Python 3.11+ | Where the entire quant ecosystem lives |
| Deps | `uv` | Much faster than pip/poetry, simpler to learn |
| Exchange API | `ccxt` | One interface across exchanges; swapping venues later is cheap |
| Data | `pandas` + `pyarrow` | pandas for familiarity; Parquet for compact columnar storage |
| Storage | Parquet files, partitioned by symbol/month | No database to run; DuckDB can query the files directly if needed |
| Backtest | **Custom, ~300 lines** | Vectorized libraries (vectorbt, backtrader) make look-ahead bias easy and invisible. Writing the loop yourself is the single best way to actually understand what a backtest claims |
| ML | `scikit-learn` → `lightgbm` | Linear first for interpretability, then gradient boosting. No deep learning — see PLAN §3 |
| Validation | Custom walk-forward | `sklearn`'s standard CV leaks across time and will flatter every model |
| Scheduling | `asyncio` event loop | One process, WebSocket-driven, no cron |
| Deploy | Docker on a small VPS | Reproducible, restartable, ~$5–10/month |
| Alerts | Telegram bot | Free, reliable, works from a phone, and supports the kill switch |
| Logs | `structlog` → JSON lines | Machine-readable, so post-mortems are greppable |

Deliberately excluded for now: Kubernetes, message queues, a web dashboard, a
database server, microservices. One Python process on one VPS handles this
workload. Every component added is a component that can fail at 3am while
holding a position.

## Key interfaces

```python
@dataclass(frozen=True)
class Signal:
    """What the strategy wants. Not yet an order."""
    symbol: str
    target_weight: float      # -1.0 to 1.0, fraction of equity
    confidence: float         # drives sizing in the risk layer
    reason: str               # human-readable, logged on every signal

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

- **Crash-safe state.** The process can be killed at any moment and restart
  knowing its true position. Reconcile against the exchange on every startup,
  and trust the exchange over local state.
- **Idempotent orders.** Client order IDs prevent a retry after a network
  timeout from becoming a duplicate position.
- **Heartbeat.** If no bar arrives for N minutes, alert. Silent failure while
  holding a position is the worst failure mode.
- **Kill switch.** A Telegram command that flattens all positions and halts
  trading, reachable from a phone, tested regularly.
- **Daily reconciliation.** Compare what the backtest would have done on the same
  bars against what actually happened. Every mismatch gets investigated.
