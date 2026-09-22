# AI_Finance

An automated crypto trading system, built to be evaluated honestly before it is
ever trusted with real money.

**Current status: Phase 4 built — the paper trader runs on a timer against live
prices with simulated money. No capital at risk, and no code path that could
put any at risk.**

The system trades mock money by default. Real capital requires an explicit flag
and credentials that don't exist yet.

## What this is

An algorithmic trading system for crypto (Binance first), with a research
pipeline for finding and validating signals, and a hard risk layer that can
veto or halt trading. US equities are a possible later extension, not the
starting point.

## See it work in one command

```bash
uv venv && uv pip install -e ".[dev]"

aifin demo --report demo.html
```

Generates a price series, stores it, checks its quality, demonstrates what
trading frequency costs, walks four baselines forward, fits a model, and writes
a standalone HTML report — in about 90 seconds, with no exchange account, no API
key and no network.

It shows that the machinery works. It says **nothing** about whether Bitcoin is
predictable, because the prices are a random walk generated on the spot: any
strategy that appeared to work there would be a bug, not a discovery. For a real
answer, see *Running it on real data* below.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev]"

# Offline: generate deterministic fake bars and exercise the whole pipeline.
aifin fetch --source synthetic --symbol SYNTH --start 2024-01-01 --end 2024-06-30
aifin quality --symbol SYNTH
aifin show --symbol SYNTH --interval 4h --tail 5

# Real data. Safe to re-run: idempotent, and resumes from the last stored bar.
aifin fetch --symbol BTCUSDT --symbol ETHUSDT --start 2017-08-17
aifin quality --symbol BTCUSDT --symbol ETHUSDT
aifin info

# Backtest. The benchmark and the cost bill are always printed.
aifin backtest --symbol SYNTH --interval 4h --strategy buy-and-hold

# The project's central claim, one command each.
aifin backtest --symbol SYNTH --interval 1m --strategy random
aifin backtest --symbol SYNTH --interval 1d --strategy random

# Walk-forward every baseline: fit on the past, measure on the future.
# Prints out-of-sample results only, plus the multiple-testing noise floor.
aifin walkforward --symbol SYNTH --interval 4h

# Fit a model on purged history and predict the next window. Aborts if any
# feature turns out to depend on data from the future.
aifin train --symbol SYNTH --interval 4h --model all

# A standalone HTML report: equity against buy-and-hold, drawdown, cost
# breakdown. No CDN, no network — it opens anywhere.
aifin report --symbol SYNTH --interval 4h -o report.html

pytest && ruff check .
```

## Paper trading on real prices

Real market data, simulated money, on a timer:

```bash
aifin run --symbol BTCUSDT --interval 4h --strategy ma-crossover --equity 10000
aifin status --symbol BTCUSDT
aifin health --symbol BTCUSDT --interval 4h   # is the job actually running?
aifin halt --reason "stepping away"           # kill switch; a file, so it always works
```

`--mode live` raises a `NotImplementedError` rather than trading: `execution/live.py`
is the last module that will be written, so before Phase 5 there is no code path
to a real order at all. Scheduling, alerts and the 60-day gate are in
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## Running it on real data

Everything above works on generated prices. The actual question — does any of
this work on Bitcoin — needs a real backfill:

```bash
aifin fetch --symbol BTCUSDT --symbol ETHUSDT --start 2017-08-17
aifin quality --symbol BTCUSDT --symbol ETHUSDT
aifin walkforward --symbol BTCUSDT --interval 4h
aifin train --symbol BTCUSDT --interval 4h --model all
aifin report --symbol BTCUSDT --interval 4h -o btc.html
```

The fetch needs outbound access to `api.binance.com`, so it has to run somewhere
that allows it — your laptop, or the VPS from Phase 4. Expect the baselines and
the models to fail there too: that is the base rate, and finding out cheaply is
the point of the preceding four phases.

Only 1-minute bars are ever stored. Coarser intervals are derived on read, so
there is one source of truth on disk and no way for two stored intervals to
disagree.

> **Note on fetching real data.** `aifin fetch` needs outbound access to
> `api.binance.com`. Some networks — including Anthropic's sandboxed session
> environment, where this was built — block it by policy, and Binance itself
> geo-blocks some regions. The pipeline is fully unit-tested against an
> injectable HTTP transport and exercised end to end with `--source synthetic`,
> so run the real backfill from a machine with access (your laptop, or the VPS
> from Phase 4) and it will work unchanged.

## What this is not (yet)

- Not a minute-by-minute trading bot, by design. Decisions are made on a
  4-hour-to-daily horizon. See [`docs/PLAN.md`](docs/PLAN.md) §1 for the cost
  arithmetic that settles this, and §1.5 for how much engineering it saves.
- Not a reinforcement-learning agent. RL is deferred to Phase 6, for reasons in
  [`docs/PLAN.md`](docs/PLAN.md) §8.
- Not connected to a live account. Live keys arrive at Phase 5, after 60+ days
  of paper trading. See [`docs/PLAN.md`](docs/PLAN.md) §2.

## Documents

| Doc | What it covers |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | The roadmap, the cost math, and the go/no-go gate between each phase |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, module layout, tech stack |
| [`docs/RISK.md`](docs/RISK.md) | Risk limits, kill switches, key handling, operational safety |
| [`docs/DEPLOY.md`](docs/DEPLOY.md) | Putting the paper trader on a timer, and what paper mode cannot tell you |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Finance and ML terms used throughout, in plain language |

## The one number that matters most

At $5,000 capital and Binance's standard 0.1% per-side spot fee, **every
round-trip trade costs 0.2% of the traded notional.**

| Round trips per day | Annual fee cost, as % of a $5,000 account |
|---:|---:|
| 10 | ~730% |
| 1 | ~73% |
| 0.3 (~2/week) | ~21% |

Trading frequency is a budget you spend, not a feature you add. The system is
designed around that constraint from day one — which is also why it runs as a
scheduled job a few times a day rather than a 24/7 service.

## Where things stand

| Phase | Status |
|---|---|
| 0 — Foundations and data | **Done.** Gate verified: 2,628,001 bars over 5 years, 100% coverage, zero quality errors, reproducible and idempotent. |
| 1 — Backtest engine with honest costs | **Done.** Gate verified as exact identities, not tolerances: zero-cost buy-and-hold reproduces the price return to `1e-12`, and every cost is explained to the same precision. |
| 2 — Baselines and validation framework | **Done.** Four classic baselines, walk-forward validation with embargo, and an experiment registry that computes the noise floor. All four baselines correctly find nothing on a zero-drift random walk. |
| 3 — Features and supervised models | **Done.** 17 point-in-time-verified features, purged cross-validation, and a policy layer that refuses to trade a predicted move smaller than the round trip. Validated in both directions: finds a planted AR(1) signal (z = 7.8), finds nothing in a random walk (z = 1.7). |
| 4 — Live paper trading | **Built.** 553 tests. Scheduled runner, crash-safe state, kill switch, missed-run detection. `--mode live` raises. The 60-day clock is yours to run — see [`docs/DEPLOY.md`](docs/DEPLOY.md). |
| 5 — Small real capital | Not started. Needs the Phase 4 gate and every box in [`docs/RISK.md`](docs/RISK.md) §8. |

## Three things this repo will not let you fool yourself about

**Trading frequency.** See the table above, and §1.6 of the plan for the same
claim measured rather than argued.

**How many times you rolled the dice.** Every backtest is logged. With 200
parameter runs over 1,000 days, the *best* of 200 worthless strategies would be
expected to show a Sharpe of 1.67 by luck alone — so that, not zero, is the bar
`aifin walkforward` holds results to.

**Whether a feature saw the future.** Features are built vectorised over the
whole history, which is fast and is also the easiest place in a quant codebase
to leak. So the leak is not argued about, it is tested: the features are
recomputed on truncated data and any value that moved names itself. `aifin
train` runs that check before fitting anything and aborts the run if it fails.

## Start here

Read [`docs/PLAN.md`](docs/PLAN.md), then §1.6 for the fee-drag table. Phase 3
is next: features and supervised models, framed as prediction plus an explicit
cost-aware position rule rather than an end-to-end agent.
