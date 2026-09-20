# Summary: seven stages, one honest positive

A complete record of what was built, what was measured, and what is actually
proven. Every figure below is pulled from `results/*.json`, not from memory.

**Bottom line.** Directional trading of XAUUSD at 1m/5m/15m does not clear costs,
and no model architecture fixes that — a *perfect* direction oracle still loses
money. The one genuine positive is the **volatility risk premium**, which is
independent of forecasting skill: you are paid to bear risk. Its cost barrier is
small (47×), its tail risk is the real obstacle, and that tail is not yet
measured end-to-end.

---

## Stage-by-stage results

### 1. 1-minute candle direction (OHLCV)

Data: Yahoo `GC=F`, 25,792 bars. Features causal; expanding walk-forward with
embargo.

| | |
|---|---|
| lag-1 autocorrelation | −0.0338 (Ljung-Box p=1.2e−13) |
| period up-rate | 0.479 (a down-drift) |
| base-rate accuracy | 0.5245 |
| best pool accuracy | 0.5242 |

**The trap:** apparent 52% accuracy was just *always predicting down* during a
down-drifting window. Against a constant base rate, XGBoost was significantly
**worse** (ΔE[Brier]=+0.002, t=+4.86, p=0.000).

### 2. 5m / 15m with real order flow

Data: 16,015,637 Dukascopy spot ticks, 55 trading days.

| arm | best Brier | accuracy |
|---|---|---|
| price only | 0.25020 | 0.4992 |
| + order flow | 0.25008 | 0.5062 |

Order flow added ~0.7pp of accuracy, but the Brier improvement was **not
significant** (Diebold-Mariano p=0.558 at 5m, p=0.214 at 15m). XGBoost again
significantly worse than a constant.

### 3. The binary-option prediction bot

Fit/pool/calibrate/decide, served over HTTP. On 5,931 unseen bars it **declined
98.4%**, trading 95.

| | |
|---|---|
| win rate on those 95 | 0.4947 |
| breakeven at 80% payout | 0.5556 |
| realized EV per unit | **−0.1095** |

Extreme confidence was *anti*-predictive. The gate is the product.

### 4. Scalping with an explicit cost model

The decisive diagnostic:

```
median 1-minute move   1.69 bp
round-trip cost        1.70 bp
perfect oracle, with cost   -0.22   <- loses with perfect foresight
```

| gate sweep | result |
|---|---|
| 1-minute, 16 gate settings | **all negative** |
| 5-minute best cell | +0.08 bp, n=110, **p=0.968** |

Tightening the gate raised hit rate 50.7% → 54.4% while net PnL stayed negative.
Selectivity helps; it does not cross zero.

### 5. Volatility — two bugs caught

**Bug 1 (persistence artifact):** ACF of a rolling std of *iid noise* equals its
window overlap. Measured ACF matched overlap almost exactly (0.956 vs 0.933 at
lag 1; 0.019 vs 0.000 at lag 15). Non-overlapping blocks: **−0.0054**. The "96%
persistence" was the window measuring itself.

**Bug 2 (horizon scaling):** an unscaled 1-bar vol against a 15-bar target gave a
ratio of 14.9 — exactly the horizon. Scaled: 0.994.

| model | QLIKE ↓ |
|---|---|
| constant | 0.4600 |
| HAR | 0.4552 |
| **seasonality clock** | **0.2898** |
| XGBoost | 0.2794 |

XGBoost did **not** significantly beat a clock (p=0.693), and captured 85.2% of
the perfect score vs 82.8% for the clock alone.

### 6. The variance risk premium — the positive result

Data: CBOE VIX daily 1990–2026 (9,276 points) + S&P 500 realized vol; Deribit
DVOL/BTC as cross-check.

| | |
|---|---|
| mean VRP | **+4.063 vol points** |
| fraction positive | 81.7% |
| naive t | +43.73 |
| **overlap-corrected t** | **+6.52** (n=419) |

**The correction that matters.** 21-day realized vol computed daily overlaps its
neighbour 20/21, so the rows are not independent:

| | naive | corrected |
|---|---|---|
| VRP level | t=+43.73 | **t=+6.52** |
| short-variance harvest | t=+5.48 | **t=−1.69** |

The **premium survives** correction. The **harvest does not** — not statistically
distinguishable from zero.

The tail: worst period −49.75 vs average gain +0.1245 (**400×**), skew −13.96, all
five worst periods in Feb–Mar 2020. Decades: 1990s +0.43, 2000s +0.07, 2010s
+0.18, 2020s −0.33 — but **ex-2020 the 2020s are +0.23**, so that is COVID, not
regime decay.

### 7. Can it be captured? (real option chains)

Data: CBOE SPX 29,518 quotes / 58 expiries; Deribit BTC 848 two-sided quotes.

| | |
|---|---|
| historical VRP | +4.063 vol points |
| median straddle round-trip cost | **0.086 vol points** |
| **premium / cost** | **47×** |

Median SPX straddle spread 0.67% of mid; delta-hedging ~0.25 bp/day. **The cost
objection that killed scalping does not apply here.**

The skew is the real price of the tail:

| | |
|---|---|
| OTM put IV (5% below) | 16.73% |
| OTM call IV (5% above) | 9.51% |
| **skew** | **+7.22 vol points** |

A 5%-wide put spread retains **79% of premium** (59.50 credit) with the tail
capped at 320.5. BTC is ~**18× worse** to trade (3.88% spread, ~1.555 vol pts).

---

## The pattern

Every stage produced a number that *looked* like an edge. Every time, a control
showed it was an artifact, a cost, or a tail.

| Stage | Apparent edge | What the control showed |
|---|---|---|
| 1m direction | 52.4% accuracy | base-rate skew (down-drift) |
| 5m/15m order flow | +0.6pp accuracy | p=0.558 / 0.214 |
| Binary bot | positive claimed EV | realized win rate below breakeven |
| Scalping | +0.08 bp | p=0.968 on 110 trades |
| Volatility | ACF 0.956 | window measuring itself |
| VRP | t=+43.7 | overlap → +6.52 level, −1.69 harvest |
| Capture | 47× premium/cost | skew +7.22 is what the tail costs |

Bugs caught and fixed along the way, each of which would have produced a false
result: a window-overlap "persistence", a horizon-scaling error, a double-indexed
mask, an unreliable CBOE delta field that inverted the skew sign.

## What is proven vs unproven

**Proven (measured, with controls):**
- Direction at 1m/5m/15m does not clear costs; a perfect oracle loses.
- Order flow adds accuracy but not a statistically significant probabilistic edge.
- The variance risk premium is real: +4.06 vol points, t=+6.52 corrected, stable
  across three decades.
- Option round-trip cost is ~47× smaller than that premium on SPX.
- Crash protection is priced rich (+7.22 skew) — that is the tail's price.

**Unproven (and could still sink it):**
- The premium is earned by *holding through crises*; the −400×-period tail is not
  modeled end-to-end.
- VIX is a 30-day constant-maturity index; tradeable straddles have discrete
  expiries, so realized windows will not match.
- Quoted spreads are best-case; size, adverse selection and weekend gaps cost more.
- Margin: a naked short-vol book gets called precisely when the premium is best.

**Not claimed:** that any of this is profitable. Stage 7 measures cost from a live
snapshot with no outcome to score, so it cannot be a backtest.

## The decisive missing dataset

**Historical option chains** (ORATS, CBOE DataShop — paid). Stage 6 measures the
premium; stage 7 measures the cost; only historical chains turn those into a P&L
backtest with real strikes and expiries. Everything achievable with free data is
done.

## Reproduce

```bash
pip install numpy pandas scikit-learn xgboost scipy fastapi uvicorn pytest

python -m src.experiment 1          # stage 1: 1m OHLCV
python -m src.analyze
python -m src.download_ticks 2026-07-05 2026-09-18   # ~1.5h, resumable
python -m src.experiment_flow 5min  # stage 2 (also 15min)
python -m src.analyze_flow 5min
python -m src.run_bot 5min 0.80     # stage 3
python -m src.run_scalp 1min        # stage 4 (also 5min)
python -m src.run_vol 15min         # stage 5
python -m src.run_vrp               # stage 6
python -m src.run_capture           # stage 7
python -m pytest tests/ -q          # 56 tests
```

## Layout

```
src/bot.py, api.py          prediction bot + HTTP service
src/scalp_bot.py, run_scalp.py, scalping.py, backtest.py, costs.py
src/volatility.py, run_vol.py
src/implied.py, run_vrp.py
src/options.py, run_capture.py
src/flow.py, feed.py, tick*/data/features  (data + features)
tests/                     56 tests, incl. guards for every artifact found
AGENTS.md                  constants, traps, method requirements
```

## The one thing I would not do

Present any of this as a profitable system. Across seven stages, the honest yield
is: five negative results, one real risk premium with a measured-but-small cost
barrier and an unmeasured tail, and a method that is now good at falsifying
attractive-looking numbers.