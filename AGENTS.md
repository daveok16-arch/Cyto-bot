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

Do not add complexity expecting profitability. The next honest avenue is
predicting **volatility** rather than direction, and trading it without a
binary-option payout.