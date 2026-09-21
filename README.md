# AI_Finance

An automated crypto trading system, built to be evaluated honestly before it is
ever trusted with real money.

**Current status: Phase 2 complete — data pipeline, backtest engine, baselines
and walk-forward validation built and verified. No capital at risk.**

The system trades mock money by default. Real capital requires an explicit flag
and credentials that don't exist yet.

## What this is

An algorithmic trading system for crypto (Binance first), with a research
pipeline for finding and validating signals, and a hard risk layer that can
veto or halt trading. US equities are a possible later extension, not the
starting point.

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

pytest && ruff check .
```

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
| 2 — Baselines and validation framework | **Done.** 348 tests. Four classic baselines, walk-forward validation with embargo, and an experiment registry that computes the noise floor. All four baselines correctly find nothing on a zero-drift random walk. |
| 3 — Features and supervised models | Next |
| 4 — Live paper trading | |
| 5 — Small real capital | |

## Two things this repo will not let you fool yourself about

**Trading frequency.** See the table above, and §1.6 of the plan for the same
claim measured rather than argued.

**How many times you rolled the dice.** Every backtest is logged. With 200
parameter runs over 1,000 days, the *best* of 200 worthless strategies would be
expected to show a Sharpe of 1.67 by luck alone — so that, not zero, is the bar
`aifin walkforward` holds results to.

## Start here

Read [`docs/PLAN.md`](docs/PLAN.md), then §1.6 for the fee-drag table. Phase 3
is next: features and supervised models, framed as prediction plus an explicit
cost-aware position rule rather than an end-to-end agent.
