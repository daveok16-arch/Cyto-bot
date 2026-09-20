# AGENTS.md — repository notes

## Environment gotcha (important)

The Python site-packages can be **wiped between sessions** (observed twice: only
`pip`, `fastapi`, `uvicorn`, `pydantic` survived). `data/` and git history have
always survived. If imports fail, reinstall:

```bash
pip install numpy pandas scikit-learn xgboost scipy fastapi uvicorn pytest
# only if you need the LSTM/Transformer studies:
pip install tensorflow-cpu
```

## What this repo is

A rigorous evaluation harness for short-horizon XAUUSD forecasting, plus a
prediction bot. The consistent finding across every study is that the signal is
far too small to pay transaction or binary-option costs, and the code is built to
demonstrate that honestly rather than to produce an attractive backtest.

Three stages, each committed:

1. `src/experiment.py` — 1-minute OHLCV direction study (base rate, mean reversion,
   persistence, XGBoost, LSTM, Transformer).
2. `src/experiment_flow.py` — 5m/15m with real order flow from Dukascopy ticks,
   with a price-only vs order-flow ablation.
3. `src/bot.py` + `src/api.py` — the prediction bot (direction + probability +
   trade gate), served over HTTP.
4. `src/scalp_bot.py` + `src/run_scalp.py` — advanced scalping bot
   (direction + magnitude + cost gate) with an execution-aware backtest.

## Core measured facts (do not re-derive; they took a long time)

- Dukascopy ticks: 16,015,637 ticks, 55 complete trading days, 2026-07-06 to
  2026-09-18. Cached per hour under `data/ticks/hours/`.
- Dukascopy path: **0-indexed months**, and XAUUSD prices scale by **1e-3**
  (verified against `GC=F`: ~4358 spot vs ~4408 futures basis).
- Median spread: **$0.68** on gold near $4369 = **1.56 bp**.
- Typical 1-second move: **$0.095**. The spread is ~7x the 1-second move.
- Round-trip cost floor (spread + fee + slippage): **1.70 bp**.
- Median 1-minute move: **1.69 bp** — *equal to* the cost floor.
- Median 5-minute move: **3.95 bp** — only 2.3x the cost floor.
- Order flow adds ~0.6pp of directional accuracy (grows with horizon) but the
  Brier improvement is **not** statistically distinguishable from zero
  (Diebold-Mariano p=0.56 at 5m, p=0.21 at 15m).
- XGBoost is significantly *worse* than a constant base rate at every horizon.
  If a result shows otherwise, suspect leakage.

## Method requirements (non-negotiable)

- Features must be causal: a feature at bar `t` uses only data <= close of `t`.
- Walk-forward splits with an embargo; never random K-fold on time series.
- Proper scoring rules (Brier, log loss), not accuracy. Accuracy rewards
  confidence and hides overconfidence.
- Always report a naive baseline (persistence / base rate) alongside.
- Align series by index, not position — `load_dataset` drops warm-up NaN rows.
- Charge costs on every taker leg; a maker *earns* the spread instead.

## The honest conclusion

Directional scalping of XAUUSD at 1m/5m does not clear costs. The diagnostic
that matters: a **perfect** direction oracle trading every bar still loses money,
because the typical move (1.69 bp at 1m) is smaller than the round trip (1.70 bp).
Selectivity via a magnitude gate is the only structural fix, and it moves the
result from *significantly negative* to *indistinguishable from zero* — never
reliably positive.

## Volatility findings (fifth stage)

Two traps, both caught, both worth remembering:

1. **The rolling-window persistence artifact.** A rolling std of iid noise shows
   autocorrelation at lag k equal to the window overlap `(W-k)/W`. Measured ACF
   matched overlap almost exactly (0.956 vs 0.933 at lag 1; 0.019 vs 0.000 at lag
   15). Non-overlapping blocks showed **-0.005**. So "vol is 96% persistent" was
   the window talking, not the market. There is a test for this
   (`test_rolling_window_acf_tracks_overlap_fraction`).

2. **The horizon-scaling bug.** `trailing_realized_vol` returns a ONE-bar vol. A
   variance swap at a 15-bar horizon needs it scaled by `sqrt(15)`. Unscaled, the
   backtest showed realized²=3181 vs strike²=213, a ratio of 14.9 — i.e. exactly
   the horizon. Scaling fixed it to 0.994. Getting this wrong makes a variance
   swap look like it prints money.

What the sample actually shows:
- Squared returns **do** cluster within the day (Ljung-Box(10) p=4.7e-12).
- Day-to-day realized vol does **not** persist (autocorr -0.08 over 55 days).
- The dominant predictable component is **intraday seasonality**, a 4.0x range
  from the quietest hour (20 UTC) to the busiest (12 UTC).

Forecastability (out-of-sample, overlap-free subsample):
| model | QLIKE | vs seasonality |
|---|---|---|
| constant | 0.447 | worse (p=0.001) |
| persistence | 22.8 | far worse |
| HAR | 0.448 | worse (p=0.001) |
| seasonality | 0.274 | — |
| xgboost | 0.260 | **not significant (p=0.693)** |

XGBoost beats a constant, but does **not** significantly beat the seasonality
clock. On the monetization controls, a pure clock captures **82.8%** of the
perfect-forecast score while XGBoost captures 85.2% — ML adds 2.4pp.

**The critical caveat:** a positive variance-swap PnL here is NOT evidence of a
tradeable edge. That backtest uses a *naive* strike (recent realized vol) instead
of a market price. A real options market would already price in intraday
seasonality, leaving only the variance risk premium — which this dataset cannot
measure because it contains no option quotes.

Do not add complexity expecting profitability. The measured structure is a clock,
and clocks are public information.

## Stage 6: the variance risk premium — the first genuine positive result

Every earlier stage tried to beat the market using price data. This one measures
whether the market *pays* you to bear risk. It uses implied volatility (the price
of vol), which needs option data:

- **VIX** — 30-day implied vol on the S&P 500, daily since **1990** (9,276 points)
- **DVOL** — Deribit's BTC implied vol index, ~1,000 daily points (cross-check)

### The premium is real

Implied vol minus subsequently realized vol, 1990–2026:

| | |
|---|---|
| mean premium | **+4.06 vol points** |
| median | +4.69 |
| fraction positive | **81.7%** |
| mean VIX vs mean realized | 19.44 vs 15.39 |

### But the naive t-stat is inflated — and I corrected it

21-day realized vol computed daily shares 20 of 21 days with its neighbour, so the
9,200 rows are **not** independent. The naive t-stat was inflated by roughly
`sqrt(21)`:

| | naive (overlapping) | corrected (non-overlapping) |
|---|---|---|
| VRP level | t = **+43.73** | t = **+6.52** (n=419) |
| short-variance P&L | t = **+5.48** | t = **−1.69** (n=419) |

**This is the crux.** The *premium* survives correction and stays highly
significant. The *harvest* does not — it goes from looking like a Sharpe 0.91
winner to statistically indistinguishable from zero.

There's a test asserting overlap inflates the t-stat, so this correction can't be
silently dropped.

### The tail is the story

```
worst single period   -49.75 (as a fraction of variance notional)
average gain           +0.1245
ratio                 400x
skew                  -13.96
```

One crisis erases ~400 periods of ordinary income. The five worst periods are all
February–March 2020 (COVID). You win 81.7% of the time and lose catastrophically
in the tail — which is exactly what a risk premium *should* look like.

### By decade

| decade | mean | t |
|---|---|---|
| 1990s | +0.4306 | +43.62 |
| 2000s | +0.0715 | +2.44 |
| 2010s | +0.1770 | +8.54 |
| 2020s | **−0.3347** | −3.01 |

And removing crises: excluding 2008+2020 leaves mean +0.2774 (t=+32.61).

### The 2020s negative is COVID, not a regime change

| window | n | mean | t |
|---|---|---|---|
| 2020s all | 1666 | −0.3347 | −3.01 |
| 2020s ex-2020 | 1413 | **+0.2312** | +9.15 |
| 2021 onwards | 1413 | +0.2312 | +9.15 |

Strip out the COVID crash and the premium returns. But note that by 2021–2026 the
naive t-stats are again overlapping (t=+9.15 is not the honest number).

### Verdict

**This is the first positive result in the project that is not an artifact.** The
variance risk premium exists, is stable across three decades, and is independent
of any forecasting skill — you are paid to bear risk, not to out-predict anyone.

**But it is not free money:**
- returns are severely negatively skewed every way I measured it
- the losses cluster in crashes, precisely when margin calls arrive
- the naive harvest is *not* significant once overlap is corrected
- a real implementation needs option quotes (for actual strikes), delta-hedging
  costs, margin, and the ability to survive the −400x-period drawdown
- it is a *short-volatility* business: you are selling insurance, and the
  catastrophic scenario is the one you must plan for

## Why this project took six stages to find one honest positive

Every earlier stage produced a number that *looked* like an edge, and every time a
control showed it was an artifact:

| Stage | Apparent edge | What the control showed |
|---|---|---|
| 1m direction | 52.4% accuracy | base-rate skew (47.6% up-drift) |
| 5m/15m order flow | +0.6pp accuracy | Brier delta p=0.56/0.21 |
| Binary bot | positive claimed EV | realized win rate below breakeven |
| Scalping | best 5m cell +0.08bp | p=0.97, 110 trades |
| Volatility | ACF 0.956 persistence | window measuring itself |
| **VRP** | **t=+43.7** | **overlap; corrected t=+6.52 for the level, −1.69 for the harvest** |

The pattern is the lesson: in this domain, a plausible-looking number is the
default outcome, and the work is in falsifying it.

## Stage 7: can the premium actually be captured? (real option chains)

Stage 6 established the premium exists *at the index level*. That is necessary,
not sufficient — capturing it means trading options, which carry their own
spreads. The historical study had **no cost model at all**, so this stage
supplies one from real quotes:

- CBOE SPX chain: **29,518 quotes**, 58 expiries, live bid/ask
- Deribit BTC: 848 two-sided option quotes

A snapshot cannot be a backtest (no outcome to score), but it settles the cost
question definitively.

### The cost is not the obstacle

| quantity | value |
|---|---|
| historical VRP (stage 6) | +4.063 vol points |
| median straddle round-trip cost | **0.086 vol points** |
| **premium / cost ratio** | **47x** |

Median SPX straddle spread: **0.67% of mid**. Delta-hedging the underlying costs
~0.25 bp/day, negligible over a 21-day holding period.

**The cost objection that killed the scalping study does NOT kill this one.** In
scalping, the round trip (1.70 bp) exceeded the typical move (1.69 bp). Here the
premium exceeds the cost 47-fold.

### The skew is the real price of the tail

Put IV minus call IV at matched 5% OTM strikes, 21-day expiry:

| | |
|---|---|
| OTM put IV | **16.73%** |
| OTM call IV | **9.51%** |
| skew | **+7.22 vol points** |

Downside protection is priced *rich* — and that is exactly the cost of insuring
the short-vol tail. You cannot collect the +4 vol point premium without paying
for the crash protection if you want to hedge it.

### Defined risk is surprisingly cheap

| | credit | tail |
|---|---|---|
| naked ATM put | 75.40 | unlimited to zero |
| put spread (5% wide) | 59.50 | capped at 320.50 |

The spread retains **79% of the premium** while capping the loss at 320 points.

### BTC is a different market

Deribit BTC: median spread **3.88% of mid**, IV 40.04% → implied round-trip cost
**~1.55 vol points**, i.e. **~18x more expensive than SPX**. The crypto VRP is
larger (+13 vol points) but thin liquidity eats much more of it.

### A bug worth recording

The CBOE `delta` field returned an **in-the-money put as "25-delta"**, which
inverted the skew sign (I first measured −3.47, then +7.22 after fixing).
Selection now uses **moneyness**, not the delta field, and there's a test for it.

### Verdict

**Cost is NOT the obstacle. Tail risk is.** The remaining obstacles are all
unmeasured here:

1. The premium is earned *by holding through crises* — the −49.75 vs +0.1245 tail.
2. VIX is a 30-day constant-maturity index; tradeable straddles have discrete
   expiries, so the realized window will not match.
3. Quoted spreads are best-case; size, adverse selection and weekend gaps cost more.
4. Margin: a naked short-vol book can be called precisely when the premium is best.

That is the honest status: one measured barrier removed, one structural barrier
standing.