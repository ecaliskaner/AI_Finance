# Running the paper trader

Phase 4 puts the system on a timer against live prices with simulated money. No
exchange credentials are involved, because paper trading does not place orders —
nothing in this document can move real funds, and `--mode live` raises rather
than trading.

## 1. Get the history

The strategy needs a warm-up window before it can produce a signal, and the
runner will say so rather than guess:

```bash
aifin fetch --symbol BTCUSDT --start 2017-08-17
aifin quality --symbol BTCUSDT
```

This needs outbound access to `api.binance.com`. Binance geo-blocks some
regions, and some networks block it by policy; if `aifin fetch` fails, that is
where to look first.

## 2. Decide what you are trading before you start

Pick the strategy from the walk-forward results, not from a hunch:

```bash
aifin walkforward --symbol BTCUSDT --interval 4h
```

If nothing clears the three hurdles, the honest thing to paper trade is the one
that came closest — and to expect it to lose. A paper run whose purpose is to
confirm the *machinery* works is worth doing even when the strategy is not.

## 3. Run it once by hand

```bash
aifin run --symbol BTCUSDT --interval 4h --strategy ma-crossover --equity 10000
aifin status --symbol BTCUSDT
```

The first run back-fills, decides, and writes state. Run it a second time
immediately: it should say `already-done`. That is the property that makes the
schedule safe.

## 4. Put it on a timer

Match the timer to your decision cadence. Running *more* often than the cadence
is harmless — a bar already acted on is skipped — and gives you a free retry
when a run fails.

### cron

```cron
# Every hour, on the hour. A 4-hour strategy only acts on 4-hour boundaries;
# the other runs cost one API call and exit.
0 * * * * cd /opt/ai_finance && /opt/ai_finance/.venv/bin/aifin run \
  --symbol BTCUSDT --interval 4h --strategy ma-crossover >> logs/run.log 2>&1

# Twice a day, shout if the runner has stopped checking in.
0 6,18 * * * cd /opt/ai_finance && /opt/ai_finance/.venv/bin/aifin health \
  --symbol BTCUSDT --interval 4h || echo "AI_Finance runner is stale"
```

### systemd

`/etc/systemd/system/aifin.service`:

```ini
[Unit]
Description=AI_Finance paper trading run
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/opt/ai_finance
EnvironmentFile=/opt/ai_finance/.env
ExecStart=/opt/ai_finance/.venv/bin/aifin run --symbol BTCUSDT --interval 4h --strategy ma-crossover
```

`/etc/systemd/system/aifin.timer`:

```ini
[Unit]
Description=Run AI_Finance hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

`Persistent=true` matters: after the machine has been off, the timer fires once
on boot to catch up rather than silently skipping the missed window.

```bash
systemctl enable --now aifin.timer
systemctl list-timers aifin.timer
```

## 5. Alerts

Optional, and the system runs correctly without them — a notification failure
must never take the trading system down with it.

```bash
# In .env (which is gitignored)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Unset, notifications go to the log instead. Quiet runs are not announced: four
messages a day saying "nothing happened" trains you to ignore the one that
matters.

## 6. Stopping it

```bash
aifin halt --reason "stepping away for a week"
aifin resume
```

`halt` writes a file. Deliberately: it needs no network, no credentials and no
third-party API, so it still works on the day the ones that do are down. While
it exists the runner will reduce a position but never add to one.

## 7. Watching it

```bash
aifin status --symbol BTCUSDT          # position, cash, costs paid
aifin health --symbol BTCUSDT --interval 4h   # is the job actually running?
aifin report --symbol BTCUSDT --interval 4h -o report.html
```

`health` exits non-zero when stale, so a monitoring cron can act on it. The
dangerous failure for a scheduled job is not a crash — a crash is loud. It is
the run that silently never happened.

## What paper mode cannot tell you

- **Fills are simulated.** The cost model charges a fee and a spread, but every
  order fills instantly and in full. A limit order that never gets hit, or a
  book too thin to absorb the size, is not modelled. At $5k in BTC/USDT that is
  a fair approximation; for a thin altcoin it is not.
- **There is no exchange position to reconcile against.** `docs/ARCHITECTURE.md`
  says the exchange is the source of truth and local state is a cache. That is
  right for live trading and impossible here — in paper mode the state file
  *is* the authority. Phase 5's live adapter has to reconcile on every run and
  trust the exchange over anything stored locally, which is a genuinely
  different and harder problem than what this phase solved.
- **Nothing about your own behaviour.** Watching a paper drawdown is not the
  same experience as watching a real one, and the pre-live checklist in
  `RISK.md` exists because the difference is where most of the money is lost.

## The gate

`docs/PLAN.md` Phase 4 asks for **60 consecutive days** with no unexplained
divergence from backtest expectations and no scheduled run silently missed.
Sixty days is short for statistical significance and long enough to expose
operational failures, which are the ones that actually cost money.

Reconcile as you go: each day, compare what the backtest *would have done* on
the same bars against what the runner actually did. Every mismatch is either a
bug or a lesson about the cost model, and both are worth more than the P&L.
