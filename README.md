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

## Layout

```
src/costs.py         explicit transaction cost model (the heart of it)
src/scalping.py      market-making quotes + adverse-selection sizing
src/backtest.py      execution-aware backtest (taker + maker, with fill models)
src/scalp_bot.py     direction + magnitude + cost gate
src/run_scalp.py     gate sweep with t-stats and p-values
src/bot.py           binary-option prediction bot
src/api.py           FastAPI service
src/flow.py          order-flow features (causal by construction)
src/feed.py          dataset loading + MarketFeed replay
src/ticks.py         Dukascopy tick download, per-hour caching
tests/               21 tests; costs, engine properties, decision gates
AGENTS.md            measured constants and method requirements
```

## What I will not do

You asked me to build a profitable bot and to avoid false hope. Those two are in
tension, and I chose the second one:

- **I did not tune the gate until a positive number appeared.** The 5m cell at
  p=0.97 could have been presented as "the bot works." It is noise.
- **I did not hide the perfect-oracle result.** It is the strongest evidence that
  the problem is structural, not a matter of a better model.
- **I did not add complexity for its own sake.** More contenders, deeper nets, and
  more features were all tried earlier and made things worse (XGBoost is
  significantly worse than a constant).

## Where a real edge could still be

Honest, and stated as hypothesis rather than promise:

1. **Predict volatility, not direction.** Realized vol is genuinely forecastable
   (the `rv*`, `spread_z`, `vol_z` features already exist). Trade it with a
   continuous payoff — no binary vig, no need to beat the spread on direction.
2. **Maker economics with rebates.** A maker *earns* the spread rather than paying
   it. That flips the sign of the cost term, which is the only lever big enough to
   matter here. It requires queue-position modeling and venue access this dataset
   cannot provide.
3. **Go where the phenomenon is harvested.** Sub-second order-book imbalance with
   colocation. The edge is real at that scale; it is not accessible at 1-5 minute
   retail granularity.

Any of these could also fail. The point is that they change the *structure* —
cost sign, payoff shape, or latency — rather than trying to squeeze more accuracy
out of a signal that is already measured to be too small.

## Data integrity

- 16,015,637 ticks across 55 complete trading days (2026-07-06 → 2026-09-18).
- Per-hour caching; transient 503s no longer discard a day's work.
- All usable days have ≥20 of ~23 trading hours (hour 21 UTC is gold's daily break).
- 55 days is a single regime. Magnitudes are indicative; the *conclusion* is robust
  because it rests on the cost floor and a perfect-oracle bound, not on one sample.