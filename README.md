# XAUUSD Scalping & Prediction Research

Rigorous evaluation of short-horizon XAUUSD forecasting, built to answer one
question honestly: **can any of this clear its costs?**

Answer: no — not at 1m, 5m, or 15m. The reason is arithmetic, not model quality,
and the code is built to show it rather than to produce an attractive backtest.

```bash
python -m src.run_scalp 1min        # advanced scalping bot (direction+size+cost gate)
python -m src.run_scalp 5min
python -m src.run_bot 5min 0.80     # original prediction bot (binary options)
python -m uvicorn src.api:app       # serve the prediction bot
python -m pytest tests/ -q          # 21 tests
```

## The one number that decides everything

Measured from 16M Dukascopy ticks:

| Quantity | Value |
|---|---|
| Median spread | $0.68 = **1.56 bp** |
| Round-trip cost (spread + fee + slippage) | **1.70 bp** |
| Median **1-minute** move | **1.69 bp** |
| Median 5-minute move | 3.95 bp |
| Typical 1-second move | $0.095 (~0.22 bp) |

The median 1-minute move is **equal to** the round-trip cost. The spread alone is
~7× the typical 1-second move. This is the entire story.

### The diagnostic that settles it

A **perfect direction oracle** — one that knows the next bar's direction 100% of
the time — trading every bar **still loses money**, because the typical move
doesn't cover the round trip:

```
perfect oracle, zero cost:  +3.18
perfect oracle, with cost:  -0.22   <- even with perfect foresight
```

So no amount of model improvement fixes scalping by itself. Only *selectivity*
helps — trading just when the move is large enough to clear costs.

## The advanced scalping bot

`src/scalp_bot.py` predicts two things and gates on the cheaper one:

1. **direction** — P(next move up), pooled from base rate / mean reversion / XGBoost
2. **magnitude** — E(|next move|) in bp, from an XGBoost regressor

It trades only when `E|move| >= min_move_mult × cost` **and** direction is
confident. Results on unseen data, sweeping the gate:

**1-minute (74,367 bars) — 16 of 16 gate settings lose money:**

| Gate | trades | hit rate | avg net |
|---|---|---|---|
| 1.0× cost | 9,308 | 50.7% | **−1.63 bp** |
| 1.5× cost | 2,328 | 51.5% | **−1.50 bp** |
| 2.0× cost | 658 | 53.2% | **−0.77 bp** |
| 3.0× cost | 57 | 54.4% | **−0.94 bp** |

**5-minute (14,826 bars)** — one cell shows +0.08 bp on 110 trades (1.9% of bars),
with **p = 0.97**. That is noise, not an edge, and it is the only non-negative
number in the whole sweep.

The pattern is consistent and instructive: **stricter gates raise the hit rate
(50.7% → 54.4%) while the net stays negative or converges to zero.** The bot can
learn to pick *bigger* moves; it cannot make the edge exceed the spread.

## Why raising the hit rate doesn't help

Hit rate is the wrong metric. What matters is average net PnL per trade, and it
is bounded above by:

```
avg_net  =  hit_rate × avg_win  −  (1−hit_rate) × avg_loss  −  cost
```

At the 3.0× gate the bot achieves a 54.4% hit rate and *still* loses, because
the moves it selects are only ~2-3× the cost, so wins and losses are nearly
symmetric while the spread is charged every round trip.

## What was tested, and what each stage found

| Stage | Finding |
|---|---|
| 1m OHLCV direction | No model beat a constant; XGBoost *significantly worse* |
| 5m/15m + order flow | Order flow adds ~0.6pp accuracy; Brier delta **not significant** (p=0.56/0.21) |
| Prediction bot + trade gate | Declined 98.4% of bars; the 95 it took won 49.5%, −11% per unit |
| Scalping bot + cost gate | Every gate loses at 1m; best 5m cell is p=0.97 noise |

## Stage 5: volatility — where the science gets interesting

Volatility is the one thing here that is genuinely forecastable. It is also where
I found **two bugs that would each have produced a fake result**, which is the
most useful part of this stage.

### Trap 1: the persistence that wasn't there

The standard story is "volatility is highly persistent." My first measurement
agreed: **ACF = 0.956 at lag 1**. Then I checked it against a null.

A rolling-window std of *pure iid noise* has autocorrelation at lag `k` equal to
the window overlap `(W−k)/W`. The measured ACF tracked that almost exactly:

| lag | measured ACF | window overlap |
|---|---|---|
| 1 | +0.956 | 0.933 |
| 5 | +0.713 | 0.667 |
| 10 | +0.354 | 0.333 |
| 15 | +0.019 | 0.000 |

**Non-overlapping blocks: −0.005.** The "96% persistence" was the window
measuring itself. There's now a test that runs this on iid noise
(`test_rolling_window_acf_tracks_overlap_fraction`).

### Trap 2: the horizon-scaling bug

`trailing_realized_vol` returns a **one-bar** vol. A 15-bar variance swap needs it
scaled by `sqrt(15)`. Unscaled, the backtest reported realized²=3181 vs
strike²=213 — ratio **14.9**, suspiciously equal to the horizon. Scaled correctly,
the ratio is **0.994**. Left unfixed, this would have shown a variance swap
printing money.

### What the data actually shows

- Squared returns **do** cluster within the day (Ljung-Box(10) p=4.7e-12).
- Day-to-day realized vol does **not** persist (autocorr −0.08 over 55 days).
- The dominant predictable component is **intraday seasonality**: a **4.0×** range
  from the quietest hour (20 UTC) to the busiest (12 UTC).

Out-of-sample forecastability (overlap-free subsample):

| Model | QLIKE ↓ | vs seasonality |
|---|---|---|
| constant | 0.447 | worse (p=0.001) |
| persistence | 22.8 | far worse |
| HAR | 0.448 | worse (p=0.001) |
| **seasonality clock** | 0.274 | — |
| **XGBoost** | **0.260** | **not significant (p=0.693)** |

XGBoost beats a constant — but does **not** significantly beat a clock that just
knows what hour it is.

### The control that keeps this honest

I ran the variance-swap monetization with deliberate controls:

| Strategy | captures |
|---|---|
| perfect foresight (upper bound) | 100.0% |
| **XGBoost** | 85.2% |
| **seasonality only** | **82.8%** |
| random side | 6.5% |

All the ML machinery adds **2.4 percentage points** over reading the clock.

**And the crucial caveat:** that backtest uses a *naive* strike (recent realized
vol), not a market price. A real options market already prices in intraday
seasonality. So a positive number here is **not** evidence of a tradeable edge —
it cannot be, because this dataset has no option quotes to benchmark against.

## Why I stopped short of claiming success

This is the fifth stage, and the honest pattern is unbroken:

| Stage | Result |
|---|---|
| 1m direction | no model beat a constant |
| 5m/15m + order flow | edge present, **not significant** (p=0.56/0.21) |
| Binary prediction bot | declined 98.4%; the trades it took lost 11% |
| Scalping + cost gate | every gate loses at 1m; best 5m cell p=0.97 |
| **Volatility** | **forecastable — but only via a public clock** |

Each stage produced a number that *looked* like an edge, and each time a control
showed it was an artifact: base-rate skew, noise, or a window measuring itself.

## Layout

```
src/costs.py         explicit transaction cost model (the heart of it)
src/scalping.py      market-making quotes + adverse-selection sizing
src/backtest.py      execution-aware backtest (taker + maker, with fill models)
src/scalp_bot.py     direction + magnitude + cost gate
src/volatility.py    RV estimators, HAR, QLIKE, Mincer-Zarnowitz, varswap/straddle PnL
src/run_vol.py       forecastability + monetization study with artifact controls
src/run_scalp.py     gate sweep with t-stats and p-values
src/bot.py           binary-option prediction bot
src/api.py           FastAPI service
src/flow.py          order-flow features (causal by construction)
tests/               35 tests, including the two artifact guards above
AGENTS.md            measured constants, traps, and method requirements
```

## What I will not do

You asked for a profitable bot and no false hope. Those conflict, and I kept
choosing the second:

- **I did not report the 0.956 persistence as a finding.** It was a window
  artifact, and I ran the null before believing it.
- **I did not ship the variance-swap P&L as profit.** The strike is naive; a real
  market would price in the seasonality.
- **I did not tune gates until a positive number appeared.** The p=0.97 scalping
  cell and the p=0.693 vol result are reported as noise, which is what they are.
- **I did not hide that ML adds 2.4pp over a clock.**

## Where an edge could still be — as hypotheses, not promises

1. **Volatility risk premium (VRP).** Vol sellers historically earn a premium.
   Testing it needs **option quotes** (implied vs realized), which this dataset
   lacks. This is the single most promising untested avenue, and it requires new
   data rather than a better model.
2. **Maker economics with rebates.** Flipping the cost sign is the only lever big
   enough to matter at these horizons. Needs queue-position modeling and venue
   access.
3. **Sub-second order-book data with colocation.** Where the microstructure edge
   is actually harvested.

Each could fail. The point is they change *structure* — the data, the cost sign,
or the latency — rather than squeezing more accuracy from a signal already
measured to be too small.

## Data integrity

- 16,015,637 ticks across 55 complete trading days (2026-07-06 → 2026-09-18).
- Per-hour caching; transient 503s no longer discard a day's work.
- All usable days have ≥20 of ~23 trading hours (hour 21 UTC is gold's daily break).
- 55 days is a single regime. Magnitudes are indicative; the *conclusion* is robust
  because it rests on the cost floor and a perfect-oracle bound, not on one sample.