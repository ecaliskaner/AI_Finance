# Risk Management

Everything here is enforced **in code**, in the risk engine, in all three
execution modes. Limits that depend on discipline are not limits.

## 1. Hard limits

Configured in one file, applied to every order, no exceptions and no override
flag:

| Limit | Initial value | Rationale |
|---|---|---|
| Max position size | 25% of equity per asset | A single wrong call cannot be decisive |
| Max total exposure | 100% of equity | **No leverage.** Not in Phase 5, not later, until there is a multi-year live record |
| Max daily loss | 3% of equity | Halts trading for 24h when breached |
| Max drawdown | 15% from peak equity | Halts trading entirely; requires manual restart and a written post-mortem |
| Max orders per hour | 20 | Catches a runaway loop before the fee bill does |
| Min order size | $20 notional | Below this, fees make the trade pointless |
| Max slippage tolerance | 0.5% | Order cancels rather than filling into a dislocated book |

The drawdown halt requiring manual restart is deliberate. It forces a human to
look at what happened before the system is allowed to keep losing money.

## 2. Position sizing

Start with **fixed fractional**: each position is a set fraction of equity,
scaled by model confidence and inversely by recent volatility.

```
position_fraction = base_fraction × confidence × (target_vol / realized_vol)
```

capped by the hard limits above.

Volatility scaling matters more than it looks. Constant dollar sizing means you
take far more risk in turbulent markets than calm ones, which is precisely
backwards.

**On Kelly sizing:** full Kelly is the mathematically optimal growth rate *if
your edge estimate is exactly right*. Yours will not be. Kelly is brutally
sensitive to overestimated edge, and overestimating edge is the default outcome
of a backtest. If Kelly is used at all, use a quarter of it.

## 3. Kill switches

Three layers, all tested before Phase 5:

1. **Automatic** — drawdown or daily loss limit breached → flatten and halt.
2. **Manual remote** — a Telegram command reachable from a phone → flatten and
   halt. Test it monthly. An untested kill switch is decoration.
3. **Dead man's switch** — each scheduled run writes a heartbeat; if none
   arrives within one full cadence period, alert loudly. Exchange-side
   stop-losses act as a backstop so a skipped run never leaves a position
   unmanaged.

## 4. API key security

Non-negotiable, from the first live key onward:

- **Withdrawal permission disabled.** Permanently. A compromised trade-only key
  costs you some bad trades; a compromised withdrawal key costs you everything.
- **IP whitelist** restricted to the VPS address.
- Keys in environment variables or a secrets file. **Never in the repo.**
  `.env` in `.gitignore` from the first commit, before any key exists.
- Separate keys for paper and live.
- Rotate after any suspected exposure, and after anyone else touches the server.
- Enable 2FA on the exchange account itself.

A pre-commit hook that scans for key-shaped strings is cheap insurance against
the most common way people lose crypto to a public repo.

## 5. Counterparty risk

Funds on an exchange are that exchange's liability, not your property. FTX,
Mt. Gox, Celsius, QuadrigaCX — this risk is not theoretical.

- Keep on the exchange only what's needed for trading.
- With $5,000 total, holding ~$1,000 on-exchange and the rest in self-custody is
  reasonable even though it constrains position size. That constraint is the
  price of not being a creditor in a bankruptcy.
- Prefer the largest, most-regulated venue available to you.

## 6. Operational risk

| Risk | Mitigation |
|---|---|
| VPS dies mid-position | Exchange-side stop-loss as backstop; the next scheduled run reconciles against the exchange |
| Scheduled run silently skipped | Heartbeat written per run; alert when the newest is older than one cadence |
| Live mode entered by accident | `paper` is the default; `live` needs an explicit flag *and* credentials that don't exist until Phase 5 |
| Exchange API outage | Detect, alert, halt new orders; never assume a silent API means flat |
| Rate limiting | Respect documented limits with headroom; exponential backoff |
| Bad data (bad print, gap) | Sanity-check every bar before it reaches the strategy; reject implausible prices |
| Duplicate orders after a timeout | Client order IDs, idempotent submission |
| Code bug in a new deploy | Deploy to paper first for 48 hours, always. Paper keeps running in parallel with live as a control group |
| Flash crash | Slippage tolerance; limit orders rather than market orders |

## 7. The risks that aren't technical

- **Overconfidence after a winning streak.** Three good weeks is noise. The
  capital ladder (PLAN §10.4) must be written down *before* going live,
  precisely so the scaling decision isn't made while feeling clever.
- **Loss aversion mid-drawdown.** Disabling a limit because it's about to
  trigger is the specific move that turns a 15% loss into a total one. The
  manual-restart requirement exists to add friction at exactly that moment.
- **Scope creep.** "It'd work better with leverage / more pairs / higher
  frequency" usually means the current version isn't working and the response is
  to add risk rather than diagnose. Fix the edge, not the exposure.
- **Sunk cost.** If six months of work produces no edge, the correct action is
  to stop trading it and keep the pipeline. That is a successful outcome of the
  experiment, not a failure of it.

## 8. Pre-live checklist

Every item verified before the first real order:

- [ ] 60+ days of paper trading with no unexplained divergence
- [ ] All hard limits tested by forcing each one to trigger
- [ ] Kill switch tested from a phone, end to end
- [ ] Crash recovery tested — run killed mid-execution, next run reconciles correctly
- [ ] Missed-run alerting tested by deliberately skipping a scheduled run
- [ ] API key confirmed withdrawal-disabled and IP-whitelisted
- [ ] `.env` confirmed absent from git history, not just from the working tree
- [ ] Capital ladder written down and dated
- [ ] Tax treatment understood; trade logging sufficient for reporting
- [ ] Starting capital is $200–500, not $5,000
- [ ] Paper instance configured to keep running alongside live, as a control
- [ ] Post-mortem template exists, so the first drawdown produces analysis
- [ ] **You can explain, in a paragraph, why the strategy makes money.** If you
      can't, the evidence isn't understanding yet — it's a winning streak
