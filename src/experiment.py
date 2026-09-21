"""Walk-forward experiment: can any contender, or their pool, forecast the
direction of the next XAUUSD 1-minute candle well enough to beat a binary-option
payout? Run: python -m src.experiment
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from . import data, features
from .aggregate import LogOddsPooler
from .contenders import (BaseRate, LSTMHead, MeanReversion, Persistence, TransformerHead,
                         XGBHead)
from .scoring import (accuracy, brier, brier_decomposition, log_loss,
                      reliability_curve)

warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"
PAYOUTS = [0.70, 0.75, 0.80, 0.85, 0.90]


def breakeven(payout: float) -> float:
    return 1.0 / (1.0 + payout)


def ev_per_unit(p: float, payout: float) -> float:
    """Expected value of risking 1 unit on a binary paying `payout` on a win."""
    return p * (1 + payout) - 1.0


def main(horizon: int = 1, n_splits: int = 6) -> dict:
    df = data.load()
    feat = features.build_features(df)
    y = features.make_labels(df["close"], horizon)
    X = feat.drop(columns=["_ret"])

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    Xv, yv = X.to_numpy(np.float32), y.to_numpy(int)
    print(f"bars={len(yv)} features={X.shape[1]} up_rate={yv.mean():.4f} "
          f"span={X.index.min()} -> {X.index.max()}")

    # martingale / autocorrelation check on 1-minute returns
    from scipy import stats

    r = feat["_ret"].dropna().to_numpy()
    ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
    # Ljung-Box on the first few lags
    lb_stat = len(r) * (len(r) + 2) * sum(
        (np.corrcoef(r[:-k], r[k:])[0, 1] ** 2) / (len(r) - k) for k in range(1, 11)
    )
    lb_p = 1 - stats.chi2.cdf(lb_stat, 10)
    print(f"autocorr(lag1)={ac1:+.5f}  Ljung-Box(10)={lb_stat:.1f} p={lb_p:.4g}")

    splits = list(features.walk_forward(len(yv), n_splits=n_splits))
    contenders = [BaseRate(), MeanReversion(), XGBHead(), LSTMHead(), TransformerHead(), Persistence()]
    names = [c.name for c in contenders]

    # per-block out-of-sample predictions
    blocks = {i: [] for i in range(len(splits))}
    for bi, (tr, te) in enumerate(splits):
        P = np.zeros((len(te), len(contenders)))
        for j, c in enumerate(contenders):
            c.fit(Xv[tr], yv[tr])
            P[:, j] = c.predict_proba(Xv[te])
        blocks[bi] = (P, yv[te])
        print(f"  block {bi}: train={len(tr)} test={len(te)}")

    ordered = [blocks[i] for i in range(len(splits))]
    Pall = np.vstack([b[0] for b in ordered])
    yall = np.concatenate([b[1] for b in ordered])

    # --- individual contenders, on the same pooled out-of-sample set ---
    report = {"n": int(len(yall)), "up_rate": float(yall.mean()),
              "horizon": horizon, "contenders": {}, "pool": {}, "payouts": {}}

    for j, name in enumerate(names):
        p = Pall[:, j]
        report["contenders"][name] = {
            "brier": brier(yall, p), "logloss": log_loss(yall, p),
            "accuracy": accuracy(yall, p),
            "mean_pred": float(p.mean()),
            "decomp": brier_decomposition(yall, p),
            "reliability": reliability_curve(yall, p),
        }
        print(f"{name:12s} brier={report['contenders'][name]['brier']:.5f} "
              f"acc={report['contenders'][name]['accuracy']:.4f} "
              f"meanp={p.mean():.4f}")

    # --- pooled, both weighting modes, with and without calibration ---
    for mode in ("equal", "brier"):
        for cal in (False, True):
            key = f"{mode}{'_cal' if cal else ''}"
            p, w = LogOddsPooler(mode=mode, calibrate=cal).run(ordered)
            report["pool"][key] = {
                "brier": brier(yall, p), "logloss": log_loss(yall, p),
                "accuracy": accuracy(yall, p), "mean_pred": float(p.mean()),
                "final_weights": dict(zip(names, w[-1])),
                "decomp": brier_decomposition(yall, p),
                "reliability": reliability_curve(yall, p),
            }
            print(f"pool[{key:10s}] brier={report['pool'][key]['brier']:.5f} "
                  f"acc={report['pool'][key]['accuracy']:.4f} "
                  f"meanp={p.mean():.4f}")

    # --- the question that actually matters: does it clear the vig? ---
    best_key = min(report["pool"], key=lambda k: report["pool"][k]["brier"])
    p_best = LogOddsPooler(mode=best_key.split("_")[0],
                           calibrate="cal" in best_key).run(ordered)[0]
    for payout in PAYOUTS:
        be = breakeven(payout)
        # trade only when the model's probability implies positive EV at this payout
        trade = np.array([ev_per_unit(q, payout) > 0 for q in p_best])
        n_trades = int(trade.sum())
        if n_trades:
            wins = yall[trade].mean()
            ev = np.mean([ev_per_unit(q, payout) for q in p_best[trade]])
        else:
            wins, ev = float("nan"), float("nan")
        report["payouts"][f"{payout:.2f}"] = {
            "breakeven_rate": be, "n_selected": n_trades,
            "selected_win_rate": float(wins), "realized_ev_per_unit": float(ev),
        }
        print(f"payout={payout:.2f} breakeven={be:.4f} selected={n_trades} "
              f"win_rate={wins:.4f} ev={ev:+.5f}")

    RES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RES / f"preds_h{horizon}.npz", Pall=Pall, y=yall,
                        names=np.array(names), blocks=np.array([len(b[1]) for b in ordered]))
    (RES / f"report_h{horizon}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {RES / f'report_h{horizon}.json'} and preds_h{horizon}.npz")
    return report


if __name__ == "__main__":
    import sys

    h = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(horizon=h)