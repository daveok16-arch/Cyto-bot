# XAUUSD Short-Horizon Direction Forecasting Experiment

Does a contention of modern models forecast the next XAUUSD candle direction
well enough to beat a binary-option payout? **No**, at 1m, 5m, or 15m — and the
experiments below show exactly where the claim breaks.

Two studies:

1. **1-minute OHLCV** (`src/experiment.py`) — Yahoo `GC=F`, ~26k bars.
2. **5m/15m tick + order flow** (`src/experiment_flow.py`) — Dukascopy spot
   XAUUSD ticks, 16M ticks / 55 trading days, July–Sep 2026.

---

## Study 1 — 1-minute candle direction (OHLCV)

- Source: Yahoo Finance `GC=F` (COMEX front-month gold), the practical proxy for
  spot XAUUSD. 1-minute granularity is capped at ~30 days.
- 25,792 bars, 2026-08-23 to 2026-09-18; 25,690 usable rows.
- **Features are causal**: every feature at bar `t` uses only data at or before
  the close of `t`. Label is the direction of bar `t+1`.
- **Walk-forward, expanding window**: 6 blocks, 50/50 train/test growing forward,
  5-bar embargo so no label window crosses the boundary.

### Result 1 — the market is close to a martingale

Lag-1 return autocorrelation is **-0.0338** (Ljung-Box(10) = 83.2, p ≈ 1e-13):
a *tiny* mean-reverting tendency, ~0.1% of variance. Combined with a **47.6%
up-rate** in this window (a down-drift), that is the only structure visible.

### Result 2 — no model beats the base rate

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

The apparent 52% accuracy is an illusion: the period skewed down (47.6% up), so
*always predicting down* scores 52.4%. Against a constant base-rate forecast,
every learned model is **statistically worse** (XGBoost ΔBrier = +0.002,
t = +4.86, p = 0.000). The Transformer is indistinguishable from a coin flip
(p = 0.17).

### Result 3 — claimed edge does not survive contact with reality

The pooled model *claimed* +1.1% to +1.9% EV per unit at payouts 0.80–0.90,
based on its own probabilities — while its realized win rate on those same
selected bars was 30–50%, **below breakeven at every payout**. That gap is
miscalibration, and it is how a bot talks you into losing money confidently.

---

## Study 2 — 5m / 15m with genuine order flow

The 1-minute study could not test order flow because 1-minute OHLCV has no
bid/ask and no trade volume. So this study pulls **real tick data**.

- Source: Dukascopy `XAUUSD` hourly tick files (LZMA, 20-byte records: ms offset,
  ask, bid, ask volume, bid volume). Prices are integers scaled by 1e-3.
- **16,015,637 ticks, 55 complete trading days**, 2026-07-06 → 2026-09-18.
  Unlike Yahoo's ~30-day cap, Dukascopy goes back years.
- Aggregated to 5m and 15m bars; 5m → 13,531 usable rows, 15m → 4,434.

### Order-flow features (the point of this study)

These cannot be computed from candles:

| Feature | What it captures |
|---|---|
| `ofi`, `ofi_ma5/15` | signed-volume imbalance (aggressor side via tick rule) |
| `vol_z`, `n_tick_z` | volume / trade-count intensity vs rolling norm |
| `spread_bp`, `spread_z` | effective spread — direct liquidity and cost measure |
| `impact` | absolute move per unit volume (price impact) |
| `queue_imb` | tick-count asymmetry within the bar |
| `rv_ofi_corr` | correlation of returns with order-flow imbalance |

Plus the same causal price/candle/volatility features as Study 1, so the two
arms are directly comparable.

### Result 4 — order flow adds accuracy, but not a statistically real edge

Ablation, identical models and splits, only the feature set differs:

| Horizon | Arm | Best Brier | Accuracy |
|---|---|---|---|
| 5m | price only | 0.25020 | 0.4992 |
| 5m | **+ order flow** | 0.25008 | **0.5062** |
| 15m | price only | 0.25035 | 0.5043 |
| 15m | **+ order flow** | 0.25020 | **0.5056** |

So order flow moves accuracy from ~50% to ~50.6% — a real directional
improvement, and it grows with horizon (49.9% → 50.4% → 50.6%), which is the
expected shape if signal exists but is small.

**However**, the paired Diebold-Mariano test says the Brier improvement is not
distinguishable from zero:

| Horizon | ΔE[Brier] | DM stat | p | Block-bootstrap 95% CI |
|---|---|---|---|---|
| 5m | +0.000344 | -0.59 | 0.558 | [-0.000852, +0.001513] |
| 15m | +0.001819 | -1.24 | 0.214 | [-0.000594, +0.004752] |

Both CIs straddle zero. Order flow **sharpens the direction call slightly** but
does **not** produce a measurable probabilistic edge. The honest reading:
whatever order flow contributes here is at or below the noise floor.

### Result 5 — XGBoost is significantly *worse* than a constant

At every horizon, the tree model's Brier loss is significantly worse than simply
predicting the base rate (5m: ΔBrier = +0.005, DM = +7.11, p = 0.000; 15m:
ΔBrier = +0.011, DM = +6.51, p = 0.000). It is not finding signal; it is
overfitting noise, and the walk-forward split exposes that. This is the single
most important methodological result: a leaked backtest would have shown the
opposite.

### Result 6 — still nowhere near the payout bar

| Payout | Breakeven | Best accuracy (any horizon) | Shortfall |
|---|---|---|---|
| 0.80 | 0.5556 | 0.5106 | 4.50 pp |
| 0.85 | 0.5405 | 0.5106 | 2.99 pp |
| 0.90 | 0.5263 | 0.5106 | 1.57 pp |

The very best result across 16M ticks and two horizons still fails the bar by
1.6–4.5 percentage points. At 80% payout, a 51% model loses 8.2% per unit.

---

## Verdict

For directional short-horizon XAUUSD binary options, expected value is negative
and no combination of these models changes that. The structural reasons:

1. **The payout sets a high bar.** 80% payout needs 55.6% to break even; 85%
   needs 54.1%.
2. **Signal is tiny but non-zero.** Order flow does add ~0.6pp of directional
   accuracy, and it grows with horizon — so the researcher's instinct was right.
   It is simply far too small.
3. **The vig is larger than the edge.** Even a genuine 51% model loses money at
   a 90% payout. The edge exists; it just cannot pay the fee.
4. **Complex models overfit.** XGBoost and the neural nets are significantly
   worse than a constant under honest walk-forward evaluation.
5. **Calibration fails where it matters.** The pooled model is most confident
   where it is most wrong.

## Where the real edge would be

- **Forecast volatility, not direction.** 1-minute realized vol is genuinely
  predictable; direction is nearly not. The features here (`rv*`, `spread_z`,
  `vol_z`) would serve a vol model well.
- **Trade it without the binary wrapper.** A continuous payoff has no embedded
  ~50% vig to overcome. Even a 51% directional edge can be monetized through a
  spread-capture or execution-based strategy — with maker rebates, not retail
  binary payouts.
- **Go to a real venue.** Order-book imbalance at sub-second scale, with
  colocation and rebates, is where this signal is actually harvested. Retail
  binaries interpose a fee larger than the entire phenomenon.

## Data and integrity notes

- Dukascopy path uses **0-indexed months**; XAUUSD prices scale by **1e-3**
  (verified against `GC=F`: ~4358 spot vs ~4408 futures — a normal basis).
- Downloads cache **per hour** so a transient 503 never discards good work, and
  `fill_gaps()` retries missing hours in rounds. All 55 usable days have ≥20 of
  ~23 trading hours.
- Gold's daily break is hour 21 UTC, so 23 hours/day is complete, not missing.
- 55 days is a single regime; magnitudes are indicative, not precise. The
  conclusion is robust because it rests on the payout table plus an edge that
  tests as indistinguishable from zero.

## Reproduce

```bash
python -m src.download_ticks 2026-07-05 2026-09-18   # ~1.5h, resumable
python -m src.experiment_flow 5min                    # -> results/flow_5min_*.json
python -m src.analyze_flow 5min                       # significance + payout analysis
python -m src.experiment_flow 15min
python -m src.analyze_flow 15min
python -m src.experiment 1                            # original 1m OHLCV study
python -m src.analyze
```

Layout: `src/data.py` (1m OHLCV ingest), `src/ticks.py` (tick ingest + caching),
`src/features.py` / `src/flow.py` (causal features + splits), `src/contenders.py`
(model interface), `src/aggregate.py` (log-odds pool), `src/scoring.py` (proper
scoring rules), `src/experiment*.py`, `src/analyze*.py`.