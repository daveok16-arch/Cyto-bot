"""Volatility forecasting study: forecastability first, then monetization.

An earlier version of this file assumed the standard story -- "volatility is
highly persistent" -- and measured persistence of 0.956 at lag 1. That number was
an artifact. The autocorrelation of a rolling-window vol estimate tracks the
window's own overlap: at lag k, overlap is (W-k)/W, and the observed
autocorrelation matched it almost exactly (0.956 vs 0.933 overlap at lag 1,
0.354 vs 0.333 at lag 10, 0.019 vs 0.000 at lag 15). Non-overlapping blocks show
correlation -0.005.

What this sample actually shows, tested properly:

  * squared returns DO cluster within the day (Ljung-Box(10) p=4.7e-12)
  * day-to-day realized vol does NOT persist (autocorr -0.08 over 55 days)
  * the dominant predictable component is intraday SEASONALITY, a ~4x range
    between the quietest hour (20 UTC) and the busiest (12 UTC)

So the forecastable thing here is largely a clock, not a memory. That distinction
matters enormously for monetization: intraday seasonality is public information
and is very likely already in option premia, whereas a genuine memory effect
would be tradeable.

Usage: python -m src.run_vol 15min
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import flow, volatility as V
from .feed import load_bars
from .volatility import HARModel, VolXGB, log_rmse, mincer_zarnowitz, qlike

RES = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------- diagnostics
def persistence_diagnostics(returns: pd.Series, window: int = 15) -> dict:
    """Separate genuine persistence from the rolling-window overlap artifact."""
    W = window
    rv = returns.rolling(W).std().dropna()
    acf = {k: float(rv.autocorr(k)) for k in (1, 2, 5, 10, 15)}
    overlap = {k: (W - k) / W for k in (1, 2, 5, 10, 15)}

    rr = returns.dropna().to_numpy()
    nb = len(rr) // W
    blocks = rr[: nb * W].reshape(nb, W).std(axis=1)
    lag1_block = float(np.corrcoef(blocks[:-1], blocks[1:])[0, 1])

    sq = returns.dropna() ** 2
    N = len(sq)
    lb = N * (N + 2) * sum(sq.autocorr(k) ** 2 / (N - k) for k in range(1, 11))
    return {
        "acf_rolling": acf,
        "window_overlap": overlap,
        "acf_nonoverlap_lag1": lag1_block,
        "ljung_box_sq_stat": float(lb),
        "ljung_box_sq_p": float(1 - stats.chi2.cdf(lb, 10)),
    }


# ---------------------------------------------------------------- study
def main(bar: str = "15min", horizon: int = 15, window: int = 15):
    bars = load_bars(bar).set_index("ts")
    close = bars["close"].to_numpy(float)
    ret = V.log_returns(close)
    rets = pd.Series(ret, index=bars.index)

    print(f"=== volatility study: {bar}, forward horizon = {horizon} bars ===")
    diag = persistence_diagnostics(rets, window)
    print("persistence diagnostics:")
    print("  rolling-window ACF vs overlap fraction (if these track, the "
          "'persistence' is the window):")
    for k in (1, 2, 5, 10, 15):
        print(f"    lag {k:>2}: acf={diag['acf_rolling'][k]:+.4f}  "
              f"overlap={diag['window_overlap'][k]:.3f}")
    print(f"  NON-overlapping block corr (lag1) = {diag['acf_nonoverlap_lag1']:+.4f}")
    print(f"  Ljung-Box(10) on squared returns: stat={diag['ljung_box_sq_stat']:.1f} "
          f"p={diag['ljung_box_sq_p']:.3g}  -> squared returns cluster")

    sea = rets.groupby(rets.index.hour).std()
    print(f"\nintraday seasonality: quietest hour {sea.idxmin()} "
          f"({sea.min()*1e5:.1f}e-5), busiest {sea.idxmax()} ({sea.max()*1e5:.1f}e-5), "
          f"ratio {sea.max()/sea.min():.1f}x")

    y = V.forward_realized_vol(ret, horizon)
    feat = flow.order_flow_features(bars)
    feat["trail_rv"] = V.trailing_realized_vol(ret, window)
    feat["parkinson"] = V.parkinson_rv(bars["high"].to_numpy(float),
                                       bars["low"].to_numpy(float))
    feat["hour"] = bars.index.hour.to_numpy().astype(float)

    X = feat.drop(columns=["_ret"])
    valid = X.notna().all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    ylog = np.log(np.clip(y, 1e-9, None))
    n = len(X)
    print(f"\nrows={n}")

    Xm = X.to_numpy(np.float32)
    trv = X["trail_rv"].to_numpy()
    Xh = V.har_design(pd.Series(np.log(np.clip(trv, 1e-9, None)), index=X.index),
                      lags=(1, 4, 16))

    folds = []
    for i in range(5):
        cut = int(n * (0.4 + 0.1 * i))
        te = np.arange(cut, min(cut + int(n / 7), n))
        if len(te) >= 100 and cut >= 500:
            folds.append((np.arange(0, cut), te))

    keys = ("constant", "seasonality", "persistence", "har", "xgboost")
    preds = {k: np.full(n, np.nan) for k in keys}
    hrs = X["hour"].to_numpy()
    for tr, te in folds:
        preds["constant"][te] = np.exp(ylog[tr].mean())
        seas = pd.Series(ylog[tr]).groupby(hrs[tr]).mean()
        preds["seasonality"][te] = np.exp(
            pd.Series(hrs[te]).map(seas).fillna(ylog[tr].mean()).to_numpy())
        preds["persistence"][te] = np.exp(np.log(np.clip(trv[te], 1e-9, None)))
        preds["har"][te] = np.exp(HARModel().fit(Xh.iloc[tr], ylog[tr])
                                  .predict(Xh.iloc[te]))
        preds["xgboost"][te] = np.exp(VolXGB().fit(Xm[tr], ylog[tr])
                                      .predict(Xm[te]))

    done = np.all([np.isfinite(preds[k]) for k in keys], axis=0)
    yt = y[done]
    idx_done = np.where(done)[0]
    print(f"out-of-sample rows={int(done.sum())}")

    print("\n-- FORECASTABILITY (QLIKE, logRMSE: lower is better) --")
    res = {}
    for k in keys:
        p = preds[k][done]
        r = {"qlike": qlike(yt, p), "log_rmse": log_rmse(yt, p),
             "mz": mincer_zarnowitz(yt, p)}
        res[k] = r
        print(f"  {k:<12} QLIKE={r['qlike']:.5f} logRMSE={r['log_rmse']:.4f} "
              f"MZ_R2={r['mz']['r2']:.4f} beta={r['mz']['beta']:+.3f}")

    # overlap-free comparison: targets overlap by construction, so t-stats on the
    # full sample are inflated. Subsample every `horizon` bars.
    step = horizon
    nos = np.arange(0, int(done.sum()), step)
    yt_n = yt[nos]
    print(f"\n-- overlap-free subsample (every {step}th row, n={len(nos)}) --")
    base = preds["seasonality"][done][nos]
    l0 = (np.log(yt_n) - np.log(base)) ** 2
    for k in keys:
        p = preds[k][done][nos]
        d = (np.log(yt_n) - np.log(p)) ** 2 - l0
        t = float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else 0.0
        pv = float(2 * (1 - stats.norm.cdf(abs(t))))
        print(f"  {k:<12} QLIKE={qlike(yt_n, p):.5f} vs seasonality: "
              f"t={t:+.2f} p={pv:.3f}")

    # ---- monetization ----
    print("\n-- MONETIZATION: variance swap (pays PATH variance) --")
    # the strike must be at the SAME horizon as the target; a per-bar vol needs
    # scaling by sqrt(horizon) or the comparison is off by exactly that factor
    strike_h = V.to_horizon_scale(trv, horizon)[done]
    rv2, k2 = yt ** 2, strike_h ** 2
    print(f"  realized^2 mean={rv2.mean():.1f} vs strike^2 mean={k2.mean():.1f} "
          f"(ratio {rv2.mean()/k2.mean():.3f}) <- should be near 1 if scaled right")
    for label, side in (("fade (short vol)", -np.ones(len(yt))),
                        ("forecast-direction", np.sign(preds["xgboost"][done] - strike_h))):
        pnl = side * (rv2 - k2)
        pn = pnl[nos]
        t = float(pn.mean() / pn.std() * np.sqrt(len(pn))) if pn.std() > 0 else 0.0
        pv = float(2 * (1 - stats.norm.cdf(abs(t))))
        tag = "significant" if pv < 0.05 else "not significant"
        print(f"  {label:<20} n={len(pn)} mean={pn.mean():+.3f} bp^2 "
              f"t={t:+.2f} p={pv:.3f} ({tag})")

    # Is any edge just the variance risk premium? Sellers earn it, buyers pay it.
    print("  -- variance risk premium check --")
    prem_ratio = float(np.sqrt(rv2.mean() / k2.mean()))
    print(f"     realized vol / strike vol = {prem_ratio:.4f} "
          f"({(prem_ratio-1)*100:+.2f}%)")

    # --- CONTROLS: is 'forecast-direction' a real edge or an artifact? ---
    # Any forecast correlated with the outcome will show positive PnL here,
    # because the payoff sign(forecast-strike)*(realized^2-strike^2) rewards
    # correctly signed deviations. A perfect forecaster earns |deviation| by
    # construction. The controls below show what "no information" earns.
    print("  -- controls (if these are also positive, the result is an artifact) --")
    dev = np.abs(rv2 - k2)[nos]
    rng = np.random.default_rng(0)
    controls = {
        "perfect (upper bound)": np.sign(yt - strike_h)[nos],
        "random side": rng.choice([-1.0, 1.0], size=len(nos)),
        "constant short vol": -np.ones(len(nos)),
    }
    # seasonality-only forecast as the comparison that matters
    seas_side = np.sign(preds["seasonality"][done] - strike_h)[nos]
    controls["seasonality-only"] = seas_side
    controls["xgboost"] = np.sign(preds["xgboost"][done] - strike_h)[nos]

    print(f"     mean |realized^2 - strike^2| = {dev.mean():.1f} bp^2 (perfect score)")
    for name, side_n in controls.items():
        pn = side_n * (rv2 - k2)[nos]
        t = float(pn.mean() / pn.std() * np.sqrt(len(pn))) if pn.std() > 0 else 0.0
        frac = pn.mean() / dev.mean() if dev.mean() > 0 else 0.0
        print(f"     {name:<24} mean={pn.mean():>+9.1f} t={t:+.2f} "
              f"captures {frac*100:>5.1f}% of |dev|")
    print("     -> a positive number is NOT sufficient: it only shows the forecast")
    print("        agrees with realized vol, which any weak signal does. The")
    print("        benchmark that matters is a MARKET strike (unavailable here),")
    print("        which would already price in whatever the model knows.")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"vol_{bar}_h{horizon}.json").write_text(json.dumps({
        "bar": bar, "horizon": horizon, "rows": int(done.sum()),
        "diagnostics": diag,
        "seasonality": {str(k): float(v) for k, v in sea.items()},
        "forecastability": res,
        "overlap_free_n": int(len(nos)),
    }, indent=2, default=float))
    print(f"\nwrote {RES / f'vol_{bar}_h{horizon}.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "15min")