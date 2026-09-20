"""Multi-horizon direction forecasting on XAUUSD tick data with order-flow
features, plus an ablation isolating whether order flow adds anything over
plain price/candle features.

Usage:
  python -m src.experiment_flow 5min
  python -m src.experiment_flow 15min
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import flow, ticks
from .aggregate import LogOddsPooler
from .contenders import BaseRate, MeanReversion, Persistence, XGBHead
from .scoring import (accuracy, brier, brier_decomposition, log_loss,
                      reliability_curve)

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"

# features that genuinely require tick/quote data
OF_COLS = ["ofi", "ofi_ma5", "ofi_ma15", "vol_z", "n_tick_z", "spread_bp",
           "spread_z", "impact", "queue_imb", "rv_ofi_corr"]


def cached_days(min_hours: int = 20) -> list[dt.date]:
    """Days with enough cached hours to be usable (gold trades ~23h/day; the
    daily break is hour 21 UTC)."""
    days = set()
    for f in ticks.HOURS.glob("XAUUSD_*.csv.gz"):
        if f.stat().st_size == 0:
            continue
        s = f.stem.split("_")[1][:8]
        days.add(dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])))
    return sorted(d for d in days if ticks.cached_count(d) >= min_hours)


def load_bars(bar: str) -> pd.DataFrame:
    days = cached_days()
    if not days:
        raise SystemExit("no cached tick days; run src.download_ticks first")
    frames = []
    for d in days:
        x = ticks.load_day(d)
        if len(x):
            frames.append(x)
    t = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    print(f"ticks={len(t):,} days={len(days)} "
          f"span={t.ts.min()} -> {t.ts.max()}")
    return flow.to_bars(t, bar)


def evaluate(X: pd.DataFrame, y: pd.Series, horizon_bars: int, label: str,
             n_splits: int = 6):
    cols = list(X.columns)
    sign_col = cols.index("ret1")
    valid = X.notna().all(axis=1) & y.notna()
    Xv, yv = X[valid].to_numpy(np.float32), y[valid].to_numpy(int)
    idx = X.index[valid]
    n = len(yv)
    if n < 2000:
        raise SystemExit(f"only {n} rows for {label}; need more tick days")

    splits = list(flow.walk_forward(n, n_splits=n_splits))
    contenders = [BaseRate(), MeanReversion(sign_col), XGBHead(),
                  Persistence(sign_col)]
    names = [c.name for c in contenders]

    blocks = {}
    for bi, (tr, te) in enumerate(splits):
        P = np.zeros((len(te), len(contenders)))
        for j, c in enumerate(contenders):
            c.fit(Xv[tr], yv[tr])
            P[:, j] = c.predict_proba(Xv[te])
        blocks[bi] = (P, yv[te])

    ordered = [blocks[i] for i in range(len(splits))]
    Pall = np.vstack([b[0] for b in ordered])
    yall = np.concatenate([b[1] for b in ordered])

    out = {"label": label, "n": int(n), "up_rate": float(yv.mean()),
           "horizon_bars": horizon_bars, "contenders": {}, "pool": {},
           "span": [str(idx.min()), str(idx.max())]}

    for j, nm in enumerate(names):
        p = Pall[:, j]
        out["contenders"][nm] = {
            "brier": brier(yall, p), "logloss": log_loss(yall, p),
            "accuracy": accuracy(yall, p), "mean_pred": float(p.mean()),
        }

    best = None
    for mode in ("equal", "brier"):
        for cal in (False, True):
            key = f"{mode}{'_cal' if cal else ''}"
            p, w = LogOddsPooler(mode=mode, calibrate=cal).run(ordered)
            rec = {"brier": brier(yall, p), "logloss": log_loss(yall, p),
                   "accuracy": accuracy(yall, p), "mean_pred": float(p.mean()),
                   "weights": dict(zip(names, w[-1])),
                   "decomp": brier_decomposition(yall, p)}
            out["pool"][key] = rec
            if best is None or rec["brier"] < best[1]:
                best = (key, rec["brier"], p)

    out["best_pool"] = best[0]
    return out, Pall, yall, names


def main(bar: str = "5min"):
    horizon_bars = {"1min": 1, "5min": 1, "15min": 1}.get(bar, 1)
    bars = load_bars(bar)
    feat = flow.order_flow_features(bars)
    y = flow.make_labels(bars["close"].set_axis(feat.index), horizon=horizon_bars)
    X_full = feat.drop(columns=["_ret"])

    # ---- ablation: price/candle features only vs + order flow ----
    price_only = X_full.drop(columns=[c for c in OF_COLS if c in X_full.columns])
    of_extra = X_full[[c for c in OF_COLS if c in X_full.columns]]

    results = {}
    armed = {}
    for label, X in (("price_only", price_only), ("with_orderflow", X_full)):
        print(f"\n=== {bar} horizon={horizon_bars} features={label} "
              f"({X.shape[1]} cols) ===")
        rec, Pall, yall, names = evaluate(X, y, horizon_bars, label)
        results[label] = rec
        armed[label] = (Pall, yall, names)
        for nm, v in rec["contenders"].items():
            print(f"  {nm:<12} brier={v['brier']:.5f} acc={v['accuracy']:.4f} "
                  f"meanp={v['mean_pred']:.4f}")
        for k, v in rec["pool"].items():
            print(f"  pool[{k:10s}] brier={v['brier']:.5f} acc={v['accuracy']:.4f}")

    # ---- the decisive comparison ----
    print("\n=== ABLATION: does order flow add information? ===")
    a = results["price_only"]["pool"][results["price_only"]["best_pool"]]
    b = results["with_orderflow"]["pool"][results["with_orderflow"]["best_pool"]]
    db = b["brier"] - a["brier"]
    print(f"price_only   best brier={a['brier']:.5f} acc={a['accuracy']:.4f}")
    print(f"+orderflow   best brier={b['brier']:.5f} acc={b['accuracy']:.4f}")
    print(f"delta brier = {db:+.6f} ({'order flow HELPS' if db < 0 else 'order flow does NOT help'})")

    RES.mkdir(parents=True, exist_ok=True)
    stem = f"flow_{bar}"
    for label, (Pall, yall, names) in armed.items():
        np.savez_compressed(RES / f"{stem}_{label}.npz", Pall=Pall, y=yall,
                            names=np.array(names))
    (RES / f"{stem}_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RES / f'{stem}_report.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5min")