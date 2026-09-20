# XAUUSD Prediction Bot

A prediction bot that forecasts the direction of the next XAUUSD bar and only
recommends a trade when the probability clears the binary-option breakeven.

```bash
python -m src.run_bot 5min 0.80     # backtest: fit, replay, report
python -m uvicorn src.api:app       # serve it: GET /health, POST /predict
python -m pytest tests/ -q          # 10 tests on the decision gate
```

## What it does

Given recent bars, the bot answers one question: **P(next bar closes up)** — and
then refuses to recommend a trade unless that probability beats the payout
breakeven with margin.

The payout sets the bar. A binary paying 80% on a win needs `1/(1+0.80) = 55.6%`
just to break even; 85% needs 54.1%. The bot enforces that arithmetic instead of
trusting the model's enthusiasm.

```
action: "UP" | "DOWN" | "NO_TRADE"
edge:   the win-rate margin over breakeven
reason: why it acted, or why it declined
```

## What it declined to do

Run against held-out data (5,931 unseen bars), the bot **declined 98.4% of bars**,
trading only 95. On those 95:

| | |
|---|---|
| win rate | 49.5% |
| breakeven at 80% payout | 55.6% |
| realized EV per unit staked | **−0.11** |

This is the most useful thing the bot does. When it got confident enough to
trade, it was *worse* than a coin flip. Extreme confidence in this model is
anti-predictive, and the gate is what stops that from becoming a losing position.

For comparison, its accuracy across all bars was 51.2% — the same phantom edge
seen at every stage: slightly better than chance, and far short of the ~56%
needed to pay the vig.

## How it works

Narrow on purpose. Three contenders, pooled in log-odds, then calibrated:

| Piece | Role |
|---|---|
| `base_rate` | constant training up-rate — the anchor and the honesty check |
| `mean_rev` | bets against the last move (small measured mean reversion) |
| `xgboost` | gradient-boosted trees on 27 order-flow + price features |
| pool | log-odds weighted by each contender's holdout Brier skill |
| calibration | isotonic, fit on a 30% holdout tail never used for weighting |

Features come from tick data via `src/flow.py`: signed-volume imbalance,
trade-count intensity, effective spread, price impact, queue imbalance, realized
volatility and its term structure. **Training and serving call the same feature
function**, which is what prevents training/serving skew.

Data: Dukascopy spot XAUUSD ticks, 16M ticks over 55 trading days
(Jul 6 – Sep 18 2026), cached per hour so downloads are resumable.

## Layout

```
src/bot.py         PredictionBot: fit, predict_proba, decide  <- the product
src/feed.py        dataset loading + MarketFeed replay
src/run_bot.py     end-to-end backtest CLI
src/api.py         FastAPI service (/health, /predict)
src/flow.py        order-flow features, causal by construction
src/contenders.py  model implementations
src/aggregate.py   log-odds pooling
src/scoring.py     Brier, log loss, isotonic calibration
src/ticks.py       tick download with per-hour caching
tests/test_bot.py  10 tests, mostly on the decision gate
```

## API

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"bars":[{...}, ...]}'   # >=65 bars, oldest first
```

Returns `{"decision": "NO_TRADE", "p_up": 0.5167, "breakeven": 0.5556,
"edge": -0.0389, "reason": "..."}`. Sending fewer than 65 bars returns 422;
NaN features (insufficient warm-up) return 422 with an explanation.

## Honest status

**This bot is a correct instrument, not a profitable strategy.** It forecasts
close to the noise floor, and on the rare occasions it clears its own bar it has
lost money. That is not a bug to tune away — three studies across 1m, 5m, and 15m
all landed in the same place: the signal is ~0.6pp, the vig is ~5.6pp.

Where it *is* useful today:

- **As a gate.** It refuses to trade, which is the correct action 98% of the time.
- **As a harness.** Swap the label to realized volatility (genuinely predictable)
  or add contenders, and the same fit/pool/calibrate/decide pipeline applies.
- **As a control.** Any future model should be measured against this one.

The next honest experiment is re-pointing it at volatility. Direction at this
horizon is a losing game; volatility is not.
