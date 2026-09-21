"""Realized volatility forecasting.

This is a different regime from direction prediction, for a structural reason:

  * directional returns are (close to) a martingale -- the best forecast of the
    next return sign is ~50/50, which is why every earlier stage failed
  * volatility is *persistent* -- today's vol strongly predicts tomorrow's, which
    is one of the most robust empirical facts in finance

So this module attacks a problem that is actually forecastable. The honest
baselines matter most: HAR (Corsi 2009) is the literature standard, and any ML
model that cannot beat it has found nothing.

Critically, forecastability is not the same as profitability. Two distinct
objects are computed and must not be confused:

  * PATH volatility  -- sqrt(sum of squared returns) -- what a variance swap pays
  * NET move         -- |S_T - S_0|                -- what a straddle pays

A single path can wander a long way and end up back where it started, so path
vol can be large while the net move is small. Path vol is far more predictable
than the net move, and that asymmetry decides which product is even tradable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BP = 1e4


def log_returns(close: np.ndarray) -> np.ndarray:
    r = np.full(len(close), np.nan)
    r[1:] = np.log(close[1:] / close[:-1])
    return r


def trailing_realized_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Causal per-bar vol estimate: std of returns up to and including t.

    Units: this is the volatility of ONE bar. To compare against a forward
    estimate over `horizon` bars, it must be scaled by sqrt(horizon) -- see
    `to_horizon_scale`. Without that, a variance-swap comparison is off by
    exactly the horizon factor.
    """
    s = pd.Series(returns)
    return (np.sqrt(s.pow(2).rolling(window).mean()) * BP).to_numpy()


def to_horizon_scale(per_bar_vol: np.ndarray, horizon: int) -> np.ndarray:
    """Convert a one-bar vol into the expected vol over `horizon` bars.

    Under roughly independent returns, variance adds, so vol scales by
    sqrt(horizon). Skipping this makes a variance swap appear wildly profitable.
    """
    return np.asarray(per_bar_vol, dtype=float) * np.sqrt(horizon)


def forward_realized_vol(returns: np.ndarray, horizon: int) -> np.ndarray:
    """PATH vol over the next `horizon` bars. NaN for the final `horizon` bars."""
    n = len(returns)
    sq = np.nan_to_num(returns ** 2, nan=0.0)
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    out = np.full(n, np.nan)
    idx = np.arange(n - horizon)
    out[: n - horizon] = np.sqrt(cs[idx + 1 + horizon] - cs[idx + 1]) * BP
    return out


def forward_abs_move(close: np.ndarray, horizon: int) -> np.ndarray:
    """NET move over the next `horizon` bars, in bp. NaN for the final bars."""
    n = len(close)
    out = np.full(n, np.nan)
    out[: n - horizon] = np.abs(close[horizon:] / close[: n - horizon] - 1.0) * BP
    return out


def parkinson_rv(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """High-low range estimator; more efficient than close-to-close."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2))) * BP


def har_design(log_rv: pd.Series, lags=(1, 12, 72)) -> pd.DataFrame:
    """HAR: daily / weekly / monthly averages of log realized vol."""
    X = pd.DataFrame(index=log_rv.index)
    X["d"] = log_rv
    for L in lags:
        X[f"m{L}"] = log_rv.rolling(L).mean()
    return X


class HARModel:
    """HAR baseline via OLS. The benchmark ML must beat to mean anything."""

    name = "har"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        m = X.notna().all(axis=1)
        A = np.column_stack([np.ones(m.sum()), X[m].to_numpy(float)])
        self.coef_, *_ = np.linalg.lstsq(A, np.asarray(y)[m], rcond=None)
        self.cols_ = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        A = np.column_stack([np.ones(len(X)), X[self.cols_].to_numpy(float)])
        return A @ self.coef_


class VolXGB:
    """Gradient-boosted vol forecaster on the full feature set."""

    name = "xgboost"

    def __init__(self, **kw):
        import xgboost as xgb

        self.m_ = xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_lambda=1.0, n_jobs=4, tree_method="hist", **kw)

    def fit(self, X, y):
        m = np.isfinite(np.asarray(y))
        self.m_.fit(np.asarray(X)[m], np.asarray(y)[m], verbose=False)
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X))


class NaiveVol:
    """Forecast = last observed trailing vol. The laziest honest model."""

    name = "naive"

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X)
        col = 0 if X.ndim == 2 else 0
        return X[:, col] if X.ndim == 2 else X


# ------------------------------------------------------------- evaluation
def qlike(true_rv: np.ndarray, pred_rv: np.ndarray) -> float:
    """QLIKE loss on variance: robust to noise in the realized-vol proxy.
    Lower is better; 0 is perfect."""
    t = np.asarray(true_rv, float) ** 2
    p = np.clip(np.asarray(pred_rv, float) ** 2, 1e-12, None)
    m = np.isfinite(t) & np.isfinite(p) & (t > 0)
    ratio = t[m] / p[m]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def log_rmse(true_rv: np.ndarray, pred_rv: np.ndarray) -> float:
    t = np.log(np.clip(np.asarray(true_rv, float), 1e-9, None))
    p = np.log(np.clip(np.asarray(pred_rv, float), 1e-9, None))
    m = np.isfinite(t) & np.isfinite(p)
    return float(np.sqrt(np.mean((t[m] - p[m]) ** 2)))


def mincer_zarnowitz(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Regress actual on predicted: R^2 is the honest 'how much is explained'.

    A well-calibrated forecast has intercept ~0 and slope ~1.
    """
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) < 10:
        return {"alpha": float("nan"), "beta": float("nan"), "r2": float("nan"),
                "n": int(len(a))}
    A = np.column_stack([np.ones(len(a)), p])
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((a - pred) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return {"alpha": float(coef[0]), "beta": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "n": int(len(a))}


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return 1.0 - float(np.sum((a - p) ** 2)) / ss_tot if ss_tot > 0 else float("nan")


def variance_swap_pnl(realized_rv_bp: np.ndarray, strike_bp: np.ndarray,
                      side: np.ndarray, notional: float = 1.0) -> np.ndarray:
    """PnL of a variance swap per unit notional (in bp^2).

    Buyer (side=+1) receives realized variance and pays the strike variance.
    """
    rv2 = np.asarray(realized_rv_bp, float) ** 2
    k2 = np.asarray(strike_bp, float) ** 2
    return notional * side * (rv2 - k2)


def straddle_pnl(net_move_bp: np.ndarray, premium_bp: np.ndarray,
                 side: np.ndarray) -> np.ndarray:
    """PnL of an at-the-money straddle (bp).

    Buyer (side=+1) pays the premium and receives the absolute net move.
    """
    return side * (np.asarray(net_move_bp, float) - np.asarray(premium_bp, float))