# Glossary

Terms used across these documents, in plain language.

## Market mechanics

**Basis point (bp)** — one hundredth of a percent. 20 bps = 0.2%. Trading costs
are quoted this way because they're small numbers that matter enormously in
aggregate.

**Spread** — the gap between the highest price a buyer will pay (bid) and the
lowest a seller will accept (ask). Crossing it is a cost you pay on every market
order. On BTC/USDT it's ~1 bp; on small altcoins it can be 50 bps.

**Slippage** — the difference between the price you expected and the price you
got. Grows with order size and market turbulence.

**Maker vs taker** — a *maker* posts a limit order that sits in the book adding
liquidity; a *taker* removes liquidity with a market order. Exchanges charge
takers more. Maker orders are cheaper but may never fill.

**Notional** — the total value of a trade. Buying $1,000 of BTC is $1,000
notional, regardless of your account size.

**Round trip** — buying and later selling (or vice versa). You pay fees twice,
which is why the cost is doubled in all the math.

**OHLCV** — Open, High, Low, Close, Volume. The five numbers summarizing a time
period. "1-minute OHLCV" means one such row per minute.

**Order book** — the live list of all resting buy and sell orders. Its shape
carries short-term predictive information.

**Funding rate** — on perpetual futures, a periodic payment between longs and
shorts that keeps the contract near spot price. Charged every 8 hours; easily
large enough to erase a small edge.

**Perpetual future (perp)** — a derivative tracking an asset with no expiry
date. Where most crypto leverage lives. Not used in this project's early phases.

**PDT (Pattern Day Trader) rule** — a US regulation: with under $25,000 in a
margin account you're limited to 3 day trades per 5 business days. A *day trade*
is a position opened and closed in the same session, so overnight holds don't
count. Does not apply to crypto. For equities at this account size it rules out
intraday trading but leaves multi-day holds open.

## Strategy and performance

**Alpha** — return above what the benchmark gives you. Beating BTC when BTC
rose 90% requires more than a 90% return.

**Benchmark** — what you'd have earned doing nothing clever. Here: buy-and-hold
BTC. Reported on every backtest, always.

**Sharpe ratio** — return divided by volatility. Roughly: how much return per
unit of stomach-churn. Above 1 is decent, above 2 is excellent, above 3 in a
backtest usually means a bug.

**Max drawdown** — the largest peak-to-trough fall in account value. The number
that determines whether you can actually live with the strategy.

**Long / short / flat** — betting price rises / betting it falls / holding no
position. Flat is a position, and often the right one.

**Position sizing** — how much to bet on a given signal. Frequently matters
more than the signal itself.

**Kelly criterion** — the mathematically optimal bet size for a known edge.
Dangerous in practice because your edge estimate is always optimistic. Use a
fraction of it, if at all.

**Mean reversion** — the bet that prices return toward an average after moving
away.

**Momentum / trend following** — the opposite bet: what's been moving keeps
moving. Both work, in different regimes, which is why regime matters.

**Regime** — a period where the market behaves in a consistent way (bull, bear,
choppy). Strategies are usually regime-specific even when they don't look it.

## Machine learning

**Supervised learning** — learning from labeled examples: here, "given these
features, what was the return over the next 4 hours?" The approach used in
Phase 3.

**Reinforcement learning (RL)** — learning from rewards rather than labels. The
agent acts, receives feedback, and adjusts. Intuitive for trading, and deferred
to Phase 6 for the reasons in PLAN §8.

**Feature** — an input to the model. "Return over the last hour" is a feature.

**Label / target** — what the model is trying to predict. Here, forward return.

**Overfitting** — learning patterns specific to the training data that don't
generalize. The central failure mode of ML in finance, because financial data is
mostly noise and noise is very learnable.

**Look-ahead bias** — accidentally using information that wasn't available at
decision time. Produces spectacular backtests and immediate live losses. The
architecture is built specifically to prevent it.

**Survivorship bias** — testing only on assets that still exist today. The ones
that went to zero are missing, and they weren't obviously doomed at the time.

**Walk-forward validation** — fit on a past window, test on the next unseen
window, roll forward, repeat. Respects the direction of time, unlike standard
cross-validation.

**Purging and embargo** — when predicting 4-hour-forward returns, training
samples within 4 hours of a test sample share overlapping outcomes and leak
information. Purging removes them.

**Out-of-sample** — evaluated on data the model never saw during fitting. The
only results that mean anything.

**Signal-to-noise ratio** — how much of the data is real pattern versus
randomness. In finance it is extraordinarily low, which is why so many
techniques that work elsewhere fail here.

**p-hacking** — trying many variants and reporting the best. With 200
strategies tested, about 10 will look significant at the 5% level by pure
chance. The experiment registry exists to keep this honest.

**Gradient boosting / LightGBM** — an ensemble of small decision trees, each
correcting the last. Strong on tabular data, far more data-efficient than neural
networks, and the right default here.

## Engineering

**Event-driven backtest** — bars are replayed one at a time, and the strategy
sees only what existed at that moment. Slower than vectorized backtesting, much
harder to accidentally cheat in.

**Vectorized backtest** — computing all signals at once over an entire array.
Fast, and makes look-ahead bias nearly invisible. Avoided here.

**Paper trading** — running against live market data, in real time, with
simulated money. Distinct from a backtest, which replays history fast. They
catch different failures; see PLAN §2.2.

**Idempotent** — an operation that produces the same result if repeated. Matters
for order submission: a retry after a network timeout must not become a second
position.

**Reconciliation** — comparing what your system thinks it holds against what the
exchange says. The exchange is always right.

**Kill switch** — a manual or automatic command that flattens positions and
halts trading.

**Dead man's switch** — an alert triggered by *absence* of a heartbeat. Catches
the silent failures, which are the dangerous ones.
