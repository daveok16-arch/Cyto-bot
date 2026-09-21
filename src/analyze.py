"""Statistical analysis of the out-of-sample forecasts.

Answers three questions the raw report cannot:
1. Is any contender's edge distinguishable from a coin flip (binomial test)?
2. Does the model's *claimed* edge survive realized P&L, or is it just confidence?
3. Does the naive/simple baseline beat the deep models?
Run: python -m src.analyze
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .scoring import accuracy, brier, log_loss

RES = Path(__file__).resolve().parent.parent / "results"
PAYOUTS = [0.70, 0.75, 0.80, 0.85, 0.90]


def binom_p(wins: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test vs a null rate."""
    return float(stats.binomtest(wins, n, p0).pvalue)


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    for hz in (1,):
        f = RES / f"preds_h{hz}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        P, y, names = d["Pall"], d["y"], list(d["names"])
        n = len(y)
        base = y.mean()
        print(f"\n=== horizon {hz}m | n={n} | up_rate={base:.4f} "
              f"| tendency={'DOWN' if base < 0.5 else 'UP'} {(1-base if base<.5 else base)*100:.1f}% ===\n")

        print(f"{'contender':<12}{'acc':>8}{'vs50% p':>10}{'brier':>10}"
              f"{'logloss':>10}{'mean_p':>9}{'acc_ci95':>18}")
        for j, nm in enumerate(names):
            p = P[:, j]
            pred = (p >= 0.5).astype(int)
            acc = accuracy(y, p)
            wins = int((pred == y).sum())
            pv = binom_p(wins, n, 0.5)
            lo, hi = wilson_ci(wins, n)
            print(f"{nm:<12}{acc:>8.4f}{pv:>10.4f}{brier(y,p):>10.5f}"
                  f"{log_loss(y,p):>10.5f}{p.mean():>9.4f}"
                  f"{f'[{lo:.4f},{hi:.4f}]':>18}")

        # Does the best contender beat the naive base-rate rule?
        print("\n-- edge over a constant base-rate forecast --")
        for j, nm in enumerate(names):
            p = P[:, j]
            pbase = np.full(n, np.clip(base, 0.01, 0.99))
            # Diebold-Mariano style: paired loss difference, HAC-free t-test
            dloss = (p - y) ** 2 - (pbase - y) ** 2
            t, pv = stats.ttest_1samp(dloss, 0)
            print(f"{nm:<12} dE[Brier]={dloss.mean():+.6f} t={t:+.2f} p={pv:.3f}"
                  f"  {'better' if dloss.mean()<0 else 'worse'}")

        # Realized P&L: act on every bar the model says is +EV, and also on
        # pure directional signal, with no assumption that p is well calibrated.
        print("\n-- trading the signal (act whenever the model leans up or down) --")
        for j, nm in enumerate(names):
            p = P[:, j]
            lean = np.where(p >= 0.5, 1, 0)
            for payout in (0.80, 0.85, 0.90):
                for thr in (0.0, 0.02, 0.05):
                    sel = np.abs(p - 0.5) > thr
                    if sel.sum() == 0:
                        continue
                    wins = (lean[sel] == y[sel])
                    wr = wins.mean()
                    be = 1 / (1 + payout)
                    realized = wr * (1 + payout) - 1
                    pv = binom_p(int(wins.sum()), int(sel.sum()), be)
                    print(f"{nm:<12} pay={payout:.2f} thr={thr:.2f} "
                          f"n={sel.sum():>5} win={wr:.4f} breakeven={be:.4f} "
                          f"realized_EV={realized:+.4f} p={pv:.3f}")

        # Claimed vs realized EV for the best pooled model
        rep = json.loads((RES / f"report_h{hz}.json").read_text())
        print("\n-- model's CLAIMED edge (from report) --")
        for k, v in rep["payouts"].items():
            print(f"payout={k} selected={v['n_selected']:>4} "
                  f"win_rate={v['selected_win_rate']:.4f} "
                  f"claimed_EV/unit={v['realized_ev_per_unit']:+.5f} "
                  f"(this is E[EV] under the model's own probabilities, not realized P&L)")


if __name__ == "__main__":
    main()