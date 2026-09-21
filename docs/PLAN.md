# AI_Finance — Project Plan

**Goal:** an AI system that manages capital autonomously and makes money, in
crypto markets (Binance first), extending to US equities later.

**Constraints:** ~$5,000 maximum capital at risk, comfortable with some coding,
new to Python and ML.

**Explicitly not requirements** (clarified by the owner): 1-minute trading and
24/7 operation. They were means, not ends. The end is profit, and the system
must earn trust on mock money before any real capital is committed.

This document is deliberately blunt about what will and will not work at this
account size. The plan is designed so that the first six months produce a real
answer to "does this have an edge?" rather than a system that looks profitable
in a backtest and bleeds money live.

---

## 1. The arithmetic that shapes everything

Before any design decisions, three numbers.

### 1.1 What a trade costs

On Binance spot at standard tier, the taker fee is **0.1% per side**. Buy and
sell, and you have paid **0.2% (20 basis points) of notional** — before spread
or slippage.

Reducible, but not by much:

| Lever | Effect |
|---|---|
| Hold BNB for fee discount | 0.1% → 0.075% per side (0.15% round trip) |
| Post limit orders (maker) instead of market orders | Maker fee is lower, but you risk not filling — a real cost when the signal decays |
| Zero-fee promo pairs (e.g. BTC/FDUSD historically) | Can approach 0%, but promos end and liquidity shifts |

Plan for **20 bps round trip**, and treat anything better as upside.

### 1.2 What a 1-minute move is worth

BTC's annualized volatility runs roughly 50%. There are 525,600 minutes in a
year, so per-minute volatility is `0.50 / √525,600 ≈ 0.069%` — about **7 basis
points**.

Now compare the cost of trading to the size of the move you're trying to catch:

| Holding period | Typical move (1σ) | Round-trip cost as a multiple of that move |
|---|---:|---:|
| 1 minute | 7 bps | **2.9σ** |
| 5 minutes | 15 bps | 1.3σ |
| 15 minutes | 27 bps | 0.75σ |
| 1 hour | 53 bps | 0.37σ |
| 4 hours | 107 bps | 0.19σ |
| 1 day | 262 bps | 0.08σ |

**To profit on a 1-minute holding period you must correctly predict a ~3-sigma
move, repeatedly, net of fees.** Firms that do this have servers in the same
building as the exchange's matching engine, custom network hardware, and
microsecond latency. Competing there from a laptop or a VPS is not a difficulty
problem, it's a category error.

### 1.3 What frequency does to the account

Fees compound against you relentlessly:

| Round trips / day | Annual fee drag on a $5,000 account (full notional each trade) |
|---:|---:|
| 10 | ~730% |
| 3 | ~219% |
| 1 | ~73% |
| ~2 per week | ~21% |

A strategy trading ten times a day needs to gross **730% per year** just to
break even. This is why the overwhelming majority of retail high-frequency bots
lose money: not because the signal is wrong, but because the fee schedule is a
tax on activity and they pay it hundreds of times.

### 1.4 What this means for the design

Since 1-minute trading and 24/7 operation are not requirements, the design can
follow the arithmetic instead of fighting it.

| Original idea | What we build | Why |
|---|---|---|
| Buy and sell on a 1-minute spectrum | Decide on a **4-hour to daily** horizon | Round-trip cost drops from 2.9σ of the move to under 0.2σ. The same signal quality that loses money at 1 minute makes money at 4 hours |
| Run 24/7 | A **scheduled job** waking a few times a day | The biggest simplification available here — see §1.5 |
| Reward the machine each time it gains money (RL) | Supervised prediction + an explicit, cost-aware position rule | RL needs millions of samples; you have ~500k minute bars, and they're non-stationary. See §8 |
| Analyze news with AI | LLM layer as a **context and veto** signal, not a trade trigger | News is priced in seconds. You cannot beat that; you can use it to size down and stay out of trouble |
| — | **Mock money by default**, live behind an explicit gate | See §2 |

Nothing about the goal changes. Only the timescale — and the timescale was the
thing making the goal unreachable.

**Trading frequency is a budget.** Every phase below reports its fee drag
alongside its returns. A strategy that doesn't beat buy-and-hold BTC *after
costs* is not a strategy.

### 1.5 What dropping 1-minute and 24/7 actually saves

Worth stating plainly, because it is the largest cost reduction in the project
and it isn't a financial one.

A system that trades every minute, continuously, needs persistent WebSocket
feeds with reconnect logic, an async event loop, a crash-safe state machine that
can resume mid-position, sub-second order lifecycle handling, a dead man's
switch at minute granularity, and real 24/7 uptime monitoring. That is the hard
two thirds of the engineering. It is the part most likely to fail at 3am while
holding a position, and it is genuinely difficult to write correctly while also
learning Python.

A system that decides every 4 hours needs a scheduled job that wakes up, fetches
recent bars, computes features, asks the model, places at most a few orders, and
exits. If it crashes, the next run picks up from the exchange's own record of
your position. There is no mid-position state to corrupt, because the process
isn't running most of the time.

Same strategy, same research pipeline, same expected returns — roughly a third
of the code, and the third that's removed is the dangerous third.

### 1.6 The same argument, measured

Once the backtest engine existed (Phase 1) the claim above stopped being
arithmetic and became something testable. Below: **one** no-edge strategy,
**one** price series, one year, a 25% position cap, default costs. The only
thing that changes between rows is how often it is allowed to trade.

| Cadence | Fills | Turnover | Cost drag | Return | Final equity from $5,000 |
|---|---:|---:|---:|---:|---:|
| 1 minute | 393,444 | 909× | 100.0% | −100.00% | **$0.00** |
| 15 minutes | 26,217 | 925× | 101.7% | −99.06% | $46.89 |
| 1 hour | 6,459 | 652× | 71.7% | −67.19% | $1,640.72 |
| 4 hours | 1,643 | 234× | 25.8% | −23.29% | $3,835.50 |
| 1 day | 279 | 44× | 4.8% | −6.51% | $4,674.73 |

The 1-minute account is annihilated inside a year. Not because the strategy is
bad — it is the *same* strategy in every row — but because it paid the fee 393,444
times.

Two caveats, stated plainly. This runs on synthetic bars, so the exact return
column is specific to that price path; and a real strategy would trade on a
signal rather than every bar. What does *not* depend on either is the cost drag,
which follows only from the trade count and the fee schedule. That column is the
finding, and it is why this project decides on a 4-hour-to-daily horizon.

Reproduce it with `aifin backtest --interval 1m --strategy random` against your
own data.

---

## 2. Operating principle: mock money by default

The system has to prove itself before it is believed. That is not just Phase 4;
it is a property built into the software.

### 2.1 Live trading is opt-in, not a setting you can forget

- Execution mode is explicit on every run: `backtest`, `paper`, or `live`.
- **`paper` is the default.** Running the system with no mode specified trades
  fake money.
- `live` requires an explicit flag *and* a separate set of API credentials that
  do not exist on disk until Phase 5.
- The live adapter is the **last** module written, not the first. Until then
  there is no code path to a real order at all — a stronger guarantee than any
  configuration check.

### 2.2 Two kinds of mock money, and they catch different lies

These get treated as one thing. They aren't, and skipping the second is where
backtested strategies go to die.

| | **Backtest** | **Paper trading** |
|---|---|---|
| Data | Historical, replayed fast | Live, in real time |
| Speed | 5 years in seconds | 5 years takes 5 years |
| Catches | Bad strategy logic, cost blindness, regime fragility | Look-ahead bias, data feed gaps, wrong fill assumptions, crashes, API errors |
| Blind to | Fills, latency, live data quality, everything operational | Almost nothing — this is the honest one |

A backtest tells you whether the *idea* is any good. Paper trading tells you
whether the *system* is any good. Both are needed, in that order, and a strategy
that survives the first but not the second is the common case rather than the
exception.

### 2.3 What "believing it" should require

Promotion from paper to real money is a checklist, not a feeling. The full list
is [`RISK.md`](RISK.md) §8; its core:

- 60+ consecutive days of paper trading
- Live results inside the backtest's confidence interval — **not** better.
  Dramatically better means something is wrong, and finding out what is cheaper
  now than later
- Zero unexplained divergences between what the backtest would have done and
  what paper actually did
- Every hard risk limit tested by deliberately triggering it
- First real capital is $200–500, not $5,000

And the criterion that matters most: **you can explain why it makes money.** A
system that is profitable in paper for reasons you cannot articulate will
surprise you eventually, and the surprise will not be in your favour.

---

## 3. Phase plan

Each phase has a concrete deliverable and a **gate** — a measurable criterion
that must be met before moving on. The gates are the point. They are what stop
this from becoming six months of work on something that was never going to
function.

### Phase 0 — Foundations and data (weeks 1–2)

Build the ability to ask questions before trying to answer them.

- Python 3.11+ project, dependency management with `uv`, linting and formatting
  configured, tests runnable from one command.
- Historical data pipeline: pull BTC/USDT and ETH/USDT 1-minute OHLCV from
  Binance back as far as available (2017+), store as partitioned Parquet.
- Data quality layer: detect and log gaps, exchange outages, duplicate bars, and
  obviously bad prints. Bad data silently produces beautiful fake backtests.
- A single `load_bars(symbol, start, end, timeframe)` function everything else
  uses. One source of truth.

**Gate:** you can reload 5+ years of clean minute bars reproducibly, and the
data quality report is empty or explained.

> **Status: done.** Implemented in `ai_finance/data/`, 116 tests passing. Gate
> verified against 2,628,001 bars spanning five years: 100% coverage, zero
> quality errors, byte-identical on re-run, and a no-op second fetch. The real
> Binance backfill must be run from a network that allows `api.binance.com` —
> see the note in the README.

### Phase 1 — Backtest engine with honest costs (weeks 2–4)

The most important code in the project. If this is wrong, everything downstream
is fiction.

- **Event-driven**, not vectorized. Bars are fed one at a time; the strategy
  sees only what existed at that moment. Vectorized backtesting libraries make
  look-ahead bias almost effortless to introduce and nearly invisible.
- Explicit cost model: fee per side, spread crossing, and a slippage assumption.
  Costs are a first-class input, not an afterthought.
- The **benchmark is buy-and-hold BTC**, and it is reported on every single
  backtest result, always. A strategy that returns 40% in a year BTC returned
  90% has destroyed value.
- Metrics: total and annualized return, Sharpe, max drawdown, win rate, average
  win/loss, number of trades, **total fees paid**, and net-of-fee return.

**Gate:** feed the engine a buy-and-hold strategy and a random strategy. Buy-
and-hold reproduces BTC's actual return within a few bps. Random loses roughly
its fee bill and nothing more. If either is off, the engine is broken.

> **Status: done.** Implemented in `ai_finance/backtest/`, `strategy/`, `risk/`
> and `execution/`; 237 tests passing. The gate is asserted as exact identities
> rather than tolerances, because a loose bound would hide precisely the bug it
> is meant to catch:
>
> - Zero-cost buy-and-hold reproduces the price return to `rel=1e-12`.
> - With costs, the final equity differs by exactly
>   `(1 + edge) × (1 + fee)` — the discrepancy is fully explained, not merely
>   small.
> - Every fill's cost equals the cost model's prediction to `rel=1e-12`.
> - Trading a pinned price with zero costs leaves equity untouched to `1e-12`,
>   and with costs on, loses *exactly* the cost bill. Any accounting leak shows
>   up here.
> - A strategy that buys after a green bar on an alternating 100/110 series
>   loses exactly `(100/110)²`, because it can only ever buy the next open. One
>   bar of foresight would have turned that into a profit, so this is the
>   look-ahead detector.

### Phase 2 — Baselines and the validation framework (weeks 4–6)

Find out whether *anything simple* works before reaching for ML.

- Implement classic baselines: moving-average crossover, RSI mean reversion,
  breakout/momentum, volatility-scaled trend following.
- Build **walk-forward validation**: fit or tune on a window, test on the
  following unseen window, roll forward, repeat. Report the out-of-sample
  results only.
- Build the **p-hacking counter**: log every strategy variant ever tested. If
  you try 200 parameter combinations, roughly 10 will look great at the 5%
  significance level purely by chance. The log makes that visible instead of
  letting you fool yourself.

**Gate:** at least one baseline beats buy-and-hold on a risk-adjusted basis
out-of-sample, after costs — or you have documented that none do. Both outcomes
are informative and both allow proceeding. What is not allowed is proceeding
without knowing.

> **Status: tooling done, verdict pending real data.** Implemented in
> `ai_finance/research/` and `strategy/baselines.py`; 348 tests passing.
>
> A baseline passes only by clearing three hurdles: a **positive** Sharpe
> (losing less than a falling market is cash, not an edge), a higher Sharpe than
> buy-and-hold on the same days, and a Sharpe above the noise floor implied by
> how many parameter sets were tried.
>
> Run against the synthetic series — which is a zero-drift random walk, so
> nothing *should* work — all four baselines fail, which is the correct answer
> and the best available check that the tooling is not manufacturing edges:
>
> ```
>   [fail] breakout             sharpe -0.82 vs -0.64   return -22.2%   churn 42%
>   [fail] ma-crossover         sharpe -0.65 vs -0.64   return -22.7%   churn 45%
>   [fail] rsi-mean-reversion   sharpe -0.59 vs -0.64   return  -9.4%   churn 34%
>   [fail] vol-scaled-trend     sharpe -1.41 vs -0.64   return -42.1%   churn 58%
>   Noise floor: 975 parameter runs over 1,625 days -> Sharpe 1.54 from luck alone.
> ```
>
> Note the churn column: the winning parameters changed in 34–58% of windows.
> A rule whose best setting moves that much between quarters is fitting noise,
> and would be worth distrusting even if it had scored well.
>
> **The gate is not closed.** This says the machinery is sound; it says nothing
> about real markets. Closing it needs a Binance backfill, which cannot run from
> this sandbox (see the README). Run `aifin walkforward --symbol BTCUSDT` once
> you have the data — and expect these rules to fail there too, which is the
> base rate and the reason Phase 3 exists.

### Phase 3 — Features and supervised models (weeks 6–10)

Now ML, in its most tractable form.

- **Frame it as prediction, not decision.** The model predicts forward return
  over a fixed horizon (start with 4 hours). A separate, deterministic policy
  layer converts prediction into position size, applying the cost threshold:
  don't trade unless expected move comfortably exceeds 20 bps. Separating these
  two makes the system debuggable in a way an end-to-end agent never is.
- Features: returns over multiple lookbacks, realized volatility, volume
  patterns, order book imbalance, funding rates, time-of-day and day-of-week
  effects, cross-asset (BTC↔ETH) relationships.
- Models in this order: linear/ridge regression → gradient boosting (LightGBM).
  **No deep learning.** On this much data with this signal-to-noise ratio, a
  neural net's main accomplishment is memorizing the training set.
- Purged, embargoed cross-validation. Standard k-fold leaks information across
  adjacent time periods and will flatter every model you build.

**Gate:** out-of-sample directional accuracy meaningfully above 50%, *and* the
resulting strategy beats both buy-and-hold and the best Phase 2 baseline after
costs. If the model wins on accuracy but loses on returns, costs are eating the
edge — go back and lengthen the horizon.

### Phase 4 — Live paper trading (weeks 8–14, overlaps Phase 3)

The same code path as live, with fake money. This is where most backtested
strategies quietly die, and it is much better to find that out here.

- A **scheduled job** (cron or a systemd timer) firing on the strategy's cadence
  — every 4 hours to start. It fetches recent bars, runs the strategy, places
  orders, and exits. No persistent process, no WebSocket feed.
- **Identical strategy code** to the backtest. Only the execution adapter
  differs. If backtest and paper diverge, that divergence is a bug report about
  your cost model, and it is valuable.
- Position state read from the exchange at the start of every run, never trusted
  from local memory. This removes an entire category of bug, and it is only
  affordable because the runs are infrequent.
- Operational stack, now much smaller: structured logs, Telegram alerts, a daily
  P&L summary, and a kill switch.
- Reconciliation: every day, compare what the backtest *would have done* on the
  same bars to what paper trading actually did. Investigate every mismatch.

**Gate:** **60 consecutive days** of paper trading with no unexplained
divergence from backtest expectations, no scheduled run silently missed, and
performance within the backtest's confidence interval. Sixty days is short for
statistical significance but long enough to expose operational failures, which
are the ones that actually cost money.

### Phase 5 — Small real capital (month 4+)

- Start at **$200–500**, not $5,000. The first live deployment is an operational
  test, not an investment.
- Hard limits enforced in code, not discipline: max position size, max daily
  loss, max drawdown auto-halt. See [`RISK.md`](RISK.md).
- API keys: **trade permission only, withdrawal permanently disabled, IP
  whitelisted.** Non-negotiable.
- **Keep the paper instance running in parallel.** It is your control group:
  when live underperforms paper, the gap is execution cost, and you want it
  measured rather than guessed at.
- Scale up only on evidence: a defined ladder tied to live track record, not to
  how confident you feel after a good week.

**Gate:** 90 days live at small size, performing in line with paper. Only then
consider increasing capital.

### Phase 6 — Extensions (month 6+, optional)

Only after Phase 5 is stable. Each is a project in itself:

- **News / LLM layer** (§7)
- **Reinforcement learning** (§8)
- **US equities** — a better fit than they first looked. With 24/7 operation off
  the requirements list, a daily-horizon strategy doesn't care that the market
  closes overnight, and the PDT rule stops binding: it counts positions opened
  *and closed* in the same session, so overnight holds aren't day trades. Note
  this does constrain the horizon — a 4-hour intraday round trip still counts as
  a day trade, so equities means holding at least overnight. Needs a different
  execution adapter (Alpaca or IBKR). The research pipeline transfers intact;
  only the execution assumptions change.

---

## 4. Data strategy

| Data | Source | Cost | Phase |
|---|---|---|---|
| Crypto OHLCV, 1m, historical | Binance public API / data dumps | Free | 0 |
| Crypto live bars + order book | Binance WebSocket | Free | 4 |
| Funding rates, open interest | Binance futures API | Free | 3 |
| US equities bars | Alpaca free tier | Free | 6 |
| News headlines | RSS, CryptoPanic, exchange announcements | Free–cheap | 6 |

Store raw data immutably. Derived features are always recomputed from raw, never
edited in place. When you discover a bug in feature code six months from now —
and you will — this is what lets you rebuild everything correctly.

---

## 5. Validation methodology

This section is the difference between a system that works and one that only
appears to.

1. **Walk-forward, always.** Fit on the past, test on the future, roll forward.
   Never evaluate on data the model has seen.
2. **Purge and embargo.** When predicting 4-hour-forward returns, drop training
   samples within 4 hours of any test sample. Otherwise the label leaks.
3. **Hold out a final test set and do not touch it.** Reserve the most recent
   6–12 months. Look at it once, at the end, before going live. Every look costs
   you a piece of its validity.
4. **Log every experiment.** Strategy, parameters, date range, result. The count
   of experiments determines how impressive a result needs to be to mean
   anything.

   Implemented in `ai_finance/research/registry.py`, which turns the count into
   a number. Given `N` parameter runs over `T` days, the *best* of `N` strategies
   with no edge at all would be expected to show this Sharpe from luck alone:

   | Parameter runs | Expected best Sharpe (1,000 days) |
   |---:|---:|
   | 1 | 0.00 |
   | 20 | 1.15 |
   | 100 | 1.53 |
   | 200 | 1.67 |
   | 1,000 | 1.97 |

   So a Sharpe of 1.5 after 5 trials is interesting; the same 1.5 after 500 is
   the null hypothesis behaving exactly as expected. `aifin walkforward` prints
   this floor and refuses to pass anything below it.
5. **Test regime robustness.** Separate results for 2021 bull, 2022 bear, and
   2023–24 chop. A strategy that only works in one regime is a bet on that
   regime returning, and should be described that way.
6. **Sensitivity check.** Re-run the winning strategy with fees at 30 bps
   instead of 20. If the edge vanishes, it was never an edge — it was a cost
   assumption.

---

## 6. Common failure modes

Checked explicitly at each gate:

- **Look-ahead bias** — using the bar's close to decide a trade executed at that
  same close. The single most common backtest bug.
- **Survivorship bias** — backtesting on today's top coins. The ones that went
  to zero are missing from your dataset and they were not obviously doomed at
  the time.
- **Overfitting** — 200 variants tried, the best one reported, no mention of the
  other 199.
- **Ignoring funding and borrow costs** — perpetual futures charge funding every
  8 hours; it swamps small edges.
- **Ignoring downtime** — exchanges halt, APIs rate-limit, VPSs reboot. If the
  strategy assumes continuous presence, model its absence.
- **Position sizing as an afterthought** — a correct signal with wrong sizing
  still blows up.
- **Regime change** — the market that generated your training data may not be
  the market you trade in.

---

## 7. The news / LLM layer (Phase 6)

Worth doing, but not as the trade trigger.

**Why not a trigger:** major news is reflected in price within seconds, by
systems reading structured feeds you don't have access to. By the time an LLM
has read a headline and returned a completion, the move has happened. You would
be systematically buying the top of every news spike.

**Where it genuinely helps:**

- **Regime context** — a daily or hourly summary feeding a single "risk on /
  neutral / risk off" feature into the model.
- **Event veto** — stay flat around scheduled high-volatility events (FOMC, CPI
  releases, major unlocks). Avoiding bad trades is worth as much as finding good
  ones and is far more reliable.
- **Tail risk detection** — exchange insolvency rumors, hacks, regulatory
  action. Here latency matters less, because the move plays out over hours or
  days. This is the highest-value use.
- **Research assistant** — summarizing what happened during a drawdown so you
  can tell bad luck from a broken model.

Design it as one more feature column subject to the same validation as every
other feature. If it doesn't improve out-of-sample results, it doesn't ship.

---

## 8. Why reinforcement learning is deferred

Your instinct — reward the machine when it makes money — is exactly how RL
works, and it's the right intuition. It is also, in finance specifically, where
a very large number of projects go to die. The reasons are concrete:

- **Sample efficiency.** RL algorithms typically need millions of episodes. Eight
  years of BTC minute bars is ~4 million bars, which sounds like a lot until you
  account for the fact that adjacent bars are almost perfectly correlated. The
  effective sample size is orders of magnitude smaller.
- **Non-stationarity.** RL assumes a stable environment. Markets change
  regime, and an agent trained on 2021 has learned the dynamics of a market that
  no longer exists.
- **Signal-to-noise.** A genuinely excellent trading signal is right maybe 53%
  of the time. RL's credit assignment — figuring out which action caused which
  reward — is extremely difficult when 47% of correct actions are punished by
  noise.
- **Reward hacking.** Agents reliably find degenerate solutions: exploiting
  simulator bugs, taking enormous leverage that happens to have worked in the
  training window, or learning to trade constantly if fees are modeled even
  slightly too low.
- **Opacity.** When a supervised model loses money you can inspect its
  prediction and its features. When an RL policy loses money you have a policy
  network and a shrug.

**The path back to it:** everything in Phases 0–4 is a prerequisite for RL
anyway — the simulator, the cost model, the feature pipeline, the validation
framework. If you reach Phase 5 with a working supervised system, you will have
built exactly the environment an RL agent would need, and you will be able to
tell whether it beats the simpler thing. Approach RL as the challenger, with
supervised as the champion to beat. Not the other way around.

---

## 9. Effort and timeline

Assuming part-time work and that Python and ML are being learned alongside:

| Phase | Calendar | What dominates the time |
|---|---|---|
| 0 | 1–2 weeks | Python tooling, API handling, data hygiene |
| 1 | 2 weeks | Getting the backtester provably correct |
| 2 | 2 weeks | Validation discipline — the concepts, not the code |
| 3 | 4 weeks | Feature engineering and resisting overfitting |
| 4 | 6–8 weeks | Mostly waiting, deliberately |
| 5 | 12 weeks | Mostly waiting, deliberately |

The build effort in Phases 0–4 is meaningfully smaller than it would be for a
continuous minute-level system (§1.5); the calendar is dominated by deliberate
waiting, not by coding.

**Roughly six months before real money, most of it spent waiting on purpose.**
That waiting is not wasted time; it is the experiment running. Compressing it is
the single most expensive mistake available here.

### Expected outcome, stated plainly

The probable result is a system that does **not** beat buy-and-hold BTC after
costs. That is the base rate, and a plan that doesn't acknowledge it is selling
something. What this project reliably produces is a rigorous, working
quantitative research pipeline, real ML and Python skill, and a genuine answer
to the question rather than a guess. If an edge does turn up, this is the
machinery that will let you distinguish it from luck — which is the only way it
would ever be safe to scale.

---

## 10. Open items

1. **Exchange account** — Binance availability and KYC vary by jurisdiction.
   Confirm what you can actually open before Phase 0 finishes, since it
   determines the fee tier the cost model assumes.
2. **Tax treatment** — every trade is potentially a taxable event, and a bot
   generates thousands of them. Check how your jurisdiction treats crypto
   trading gains and what reporting is required; it may influence target holding
   periods. Worth a conversation with an accountant before Phase 5, not after.
3. **Hosting** — a $5–10/month VPS is more than sufficient for a job that runs a
   few times a day. Needed by Phase 4. An always-on home machine would also work
   for paper trading, though a VPS is worth it before live.
4. **Capital ladder** — define the specific rule for scaling from $500 upward
   before going live, so the decision isn't made mid-winning-streak.

---

## 11. Next step

**Phase 0.** Project scaffolding, the Binance data pipeline, and the data
quality report. Nothing here touches an exchange account or requires a key with
any permissions.
