"""Scoring and calibration for probabilistic prediction.

Accuracy is deliberately secondary here: a forecast is a probability, so it is
judged with proper scoring rules (Brier, log loss) and decomposed into the
Murphy reliability/resolution/uncertainty terms.
"""
from __future__ import annotations

import numpy as np


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p), 1e-12, 1 - 1e-12)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(y, p) -> float:
    return float(np.mean((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)))


def brier_decomposition(y, p, n_bins: int = 10):
    """Murphy: Brier = reliability - resolution + uncertainty (lower is better;
    reliability is the miscalibration penalty, resolution the discriminating power)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        pk, yk, nk = p[m].mean(), y[m].mean(), m.sum()
        rel += nk / n * (pk - yk) ** 2
        res += nk / n * (yk - y.mean()) ** 2
    unc = y.mean() * (1 - y.mean())
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier_check": rel - res + unc}


def reliability_curve(y, p, n_bins: int = 10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                     "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return rows


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


class IsotonicCalibrator:
    """Fit on out-of-sample (model, outcome) pairs; monotone map to calibrated
    probabilities."""

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression

        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
        self.iso_.fit(p, y)
        return self

    def transform(self, p):
        return self.iso_.predict(p)