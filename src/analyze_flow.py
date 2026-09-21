"""Significance analysis for the order-flow ablation and per-contender edge.

The headline ablation delta is small, so the question 'does order flow help?'
must be answered with a test, not with the sign of a difference. We use a paired
Diebold-Mariano style test on the per-bar squared-error loss and a block
bootstrap CI (blocks preserve serial dependence in the loss series).

Usage: python -m src.analyze_flow 5min
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path(__file__).resolve().parent.parent / "results"


def block_bootstrap_ci(d: np.ndarray, block: int = 50, n_boot: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """CI for the mean of a serially-dependent series via moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = max(1, n // block)
    starts = rng.integers(0, n - block, size=(n_boot, nb))
    means = np.empty(n_boot)
    for i, s in enumerate(starts):
        idx = (s[:, None] + np.arange(block)[None, :]).ravel()
        means[i] = d[idx % n].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(stem: str, label: str):
    f = RES / f"{stem}_{label}.npz"
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return d["Pall"], d["y"], list(d["names"])


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano with a Newey-West HAC variance (lag ~ n^(1/3))."""
    d = loss_a - loss_b
    n = len(d)
    lag = int(round(n ** (1 / 3)))
    dbar = d.mean()
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for k in range(1, lag + 1):
        gk = np.dot(dc[k:], dc[:-k]) / n
        var += 2 * (1 - k / (lag + 1)) * gk
    se = np.sqrt(max(var, 1e-30) / n)
    stat = dbar / se
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def main(bar: str = "5min"):
    stem = f"flow_{bar}"
    rep = json.loads((RES / f"{stem}_report.json").read_text())
    a = load(stem, "price_only")
    b = load(stem, "with_orderflow")
    if a is None or b is None:
        raise SystemExit(f"missing npz for {stem}; run src.experiment_flow first")
    Pa, ya, names_a = a
    Pb, yb, names_b = b
    assert np.array_equal(ya, yb), "label sets differ between ablation arms"
    y = ya

    print(f"\n=== {bar} order-flow ablation (n={len(y)}) ===")
    print(f"tail-average up-rate: {y.mean():.4f}")

    # --- per-contender edge vs a constant base-rate forecast ---
    print("\n-- contender edge vs constant base-rate (paired DM test) --")
    pbase = np.full(len(y), np.clip(y.mean(), 0.01, 0.99))
    l_base = (pbase - y) ** 2
    for j, nm in enumerate(names_a):
        p = Pa[:, j]
        l = (p - y) ** 2
        stat, pv = dm_test(l, l_base)
        print(f"{nm:<12} dE[Brier]={l.mean()-l_base.mean():+.6f} "
              f"DM={stat:+.2f} p={pv:.3f}")

    # --- the ablation itself, on the same contender so it is like-for-like ---
    pa = Pa[:, names_a.index("xgboost")]
    pb = Pb[:, names_b.index("xgboost")]
    la = (pa - y) ** 2
    lb = (pb - y) ** 2
    stat, pv = dm_test(la, lb)
    print("\n-- order flow vs price-only (XGBoost, paired) --")
    print(f"delta E[Brier] = {lb.mean()-la.mean():+.6f}")
    print(f"DM statistic  = {stat:+.2f}  p = {pv:.3f}")
    lo, hi = block_bootstrap_ci(lb - la)
    print(f"block-bootstrap 95% CI for delta = [{lo:+.6f}, {hi:+.6f}]")
    print("verdict:", "significant" if (lo > 0 or hi < 0) else
          "NOT distinguishable from zero (order flow adds no measurable edge)")

    # --- pooled-arm comparison reported from the JSON ---
    ka = rep["price_only"]["best_pool"]
    kb = rep["with_orderflow"]["best_pool"]
    print(f"\nbest pool   price_only   brier={rep['price_only']['pool'][ka]['brier']:.5f} "
          f"acc={rep['price_only']['pool'][ka]['accuracy']:.4f} ({ka})")
    print(f"best pool   +orderflow   brier={rep['with_orderflow']['pool'][kb]['brier']:.5f} "
          f"acc={rep['with_orderflow']['pool'][kb]['accuracy']:.4f} ({kb})")

    # --- what you would need to clear a binary payout ---
    print("\n-- payout requirement --")
    for payout in (0.80, 0.85, 0.90):
        be = 1 / (1 + payout)
        best_acc = max(rep["price_only"]["contenders"]["xgboost"]["accuracy"],
                       rep["with_orderflow"]["contenders"]["xgboost"]["accuracy"])
        print(f"payout={payout:.2f} breakeven={be:.4f} best_acc={best_acc:.4f} "
              f"{'CLEARS' if best_acc > be else 'FAILS'} by {abs(best_acc-be)*100:.2f} pp")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5min")