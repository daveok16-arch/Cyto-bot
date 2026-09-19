# XAUUSD 1-Minute Direction Forecasting Experiment

Does a contention of modern models forecast the next XAUUSD 1-minute candle
direction well enough to beat a binary-option payout? Short answer: **no**, and
the experiment shows exactly where the claim breaks.

## Data

- Source: Yahoo Finance `GC=F` (COMEX front-month gold futures), the practical
  free proxy for spot XAUUSD. 1-minute granularity is capped at ~30 days.
- 25,792 bars, 2026-08-23 to 2026-09-18. 25,690 usable rows after feature
  warm-up.
- Note: this is futures, not spot. A real deployment should use broker spot
  ticks; the statistical conclusion does not depend on the difference.

## Method

- **Features are causal**: every feature at bar `t` uses only data at or before
  the close of `t`. Label is the direction of bar `t+1`.
- **Walk-forward, expanding window**: 6 blocks, 50/50 train/test split growing
  forward, with a 5-bar embargo so no label window crosses the boundary.
- **Pooling** is done in log-odds with weights from each contender's *prior*
  Brier score, and isotonic calibration fit only on earlier blocks.

## Contenders

| Contender | Family |
|---|---|
| `base_rate` | constant training-period up-rate (the honesty check) |
| `mean_rev` | bets against the last move (motivated by measured autocorrelation) |
| `persistence` | naive: next bar repeats the last bar's direction |
| `xgboost` | gradient-boosted trees on 29 tabular features |
| `lstm` | Keras LSTM over a 60-bar sequence window |
| `transformer` | small Keras multi-head attention over the same window |
| pools | equal- and Brier-weighted log-odds, with/without isotonic calibration |

## Result 1 — the market is close to a martingale

Lag-1 return autocorrelation is **-0.0338** (Ljung-Box(10) = 83.2, p ≈ 1e-13):
a *tiny* mean-reverting tendency, on the order of 0.1% of variance. Combined
with a **47.6% up-rate** (a down-drift in this window), this is the only
exploitable structure visible — and it is far too small to trade.

## Result 2 — no model beats the base rate

Out-of-sample, n = 23,516:

| Contender | Accuracy | p vs 50% | Brier | mean p |
|---|---|---|---|---|
| base_rate | 0.5245 | 0.0000 | 0.24943 | 0.4803 |
| mean_rev | 0.5245 | 0.0000 | 0.24946 | 0.4804 |
| xgboost | 0.5225 | 0.0000 | 0.25139 | 0.4799 |
| lstm | 0.5181 | 0.0000 | 0.24984 | 0.4778 |
| transformer | 0.5045 | 0.1729 | 0.25047 | 0.4944 |
| persistence | 0.4983 | 0.5974 | 0.25267 | 0.4976 |
| pool (best) | 0.5242 | — | 0.24924 | 0.4848 |

The apparent "52% accuracy" is an illusion: the whole test period skewed down
(47.6% up), so *always predicting down* scores 52.4%. Against a constant
base-rate forecast, every learned model is **statistically worse** (XGBoost
ΔBrier = +0.002, t = +4.86, p = 0.000). The Transformer is indistinguishable
from a coin flip (p = 0.17). Deep sequence models did not find signal; they found
noise and fit it.

## Result 3 — the claimed edge does not survive contact with reality

Acting on the model's own probabilities, with realistic payouts:

| Payout | Breakeven | Realized EV/unit (best) |
|---|---|---|
| 0.80 | 0.5556 | negative for every contender |
| 0.85 | 0.5405 | negative for every contender |
| 0.90 | 0.5263 | ~0 for the strongest, p ≈ 0.5–0.8 (noise) |

Meanwhile the pooled model *claimed* positive expected value at payouts 0.80–0.90
(+1.1% to +1.9% per unit) based on its own probabilities — while its realized win
rate on those same selected bars was 30–50%, **below breakeven at every payout**.
That gap between claimed and realized EV *is* miscalibration, and it is precisely
how a bot like this talks you into losing money confidently.

## Verdict

For directional 1-minute XAUUSD binary options, the expected value is negative
and no combination of these models changes that. The structural reasons:

1. **The payout sets a high bar.** At 80% payout you need 55.6% just to break
   even; at 85%, 54.1%.
2. **The horizon offers almost no signal.** Lag-1 autocorrelation is ~ -0.03 and
   the exploitable edge is on the order of 0.1% of variance.
3. **The vig is larger than any edge.** Even a genuine 53% model loses 4.6% per
   unit at 80% payout. The model must beat both the market *and* the fee.
4. **Deep models overfit this noise.** XGBoost, LSTM and Transformer all
   underperform a constant in honest walk-forward evaluation; their good-looking
   backtests are leakage or hindsight.
5. **Calibration fails where it matters.** The model is confident precisely where
   it is wrong, which is the most dangerous failure mode for a prediction bot.

## What would be worth building instead

- **Forecast volatility, not direction.** 1-minute realized volatility is
  genuinely predictable; direction is not.
- **Ditch the binary wrapper.** Its fixed payout structure is what makes an
  already-hard problem unwinnable. Any real edge is better expressed through an
  instrument with a continuous payoff and no built-in vig.
- **Move up the horizon and to order-flow data.** Signal plausibly exists at
  seconds-to-minutes scales in order-book imbalance, not in candle patterns, and
  it requires depth-of-book and maker rebates to monetize.

## Reproduce

```bash
python -m src.experiment 1   # walk-forward run -> results/report_h1.json, preds_h1.npz
python -m src.analyze        # significance tests, realized-EV paper sim
```

Layout: `src/data.py` (ingest), `src/features.py` (causal features + splits),
`src/contenders.py` (model interface), `src/aggregate.py` (log-odds pool),
`src/scoring.py` (proper scoring rules), `src/experiment.py`, `src/analyze.py`.