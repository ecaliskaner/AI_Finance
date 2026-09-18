# AI_Finance

An automated crypto trading system, built to be evaluated honestly before it is
ever trusted with real money.

**Current status: planning. No code yet. No capital at risk.**

The system trades mock money by default. Real capital requires an explicit flag
and credentials that don't exist yet.

## What this is

A 24/7 algorithmic trading system for crypto (Binance first), with a research
pipeline for finding and validating signals, and a hard risk layer that can
veto or halt trading. US equities are a possible later extension, not the
starting point.

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

## Start here

Read [`docs/PLAN.md`](docs/PLAN.md). Phase 0 is the next piece of work.
