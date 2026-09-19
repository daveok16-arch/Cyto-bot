"""Causal feature engineering and walk-forward splitting.

Every feature at row t uses only information available at or before the close of
bar t. The label is the direction of the *next* bar. Splits are strictly
forward-in-time with an embargo so no label window crosses the train/test
boundary. This is what keeps the backtest honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    prev_c = c.shift(1)
    ret1 = np.log(c / prev_c)
    f = pd.DataFrame(index=df.index)

    for lag in (1, 2, 3, 5, 10, 15, 30, 60):
        f[f"ret{lag}"] = np.log(c / c.shift(lag))

    # candle shape
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["close_pos"] = (c - l) / rng
    f["range_pct"] = rng / c

    # volatility (past-only windows)
    for n in (5, 15, 30, 60):
        f[f"vol{n}"] = ret1.rolling(n).std()
        f[f"mom{n}"] = c / c.shift(n) - 1
    f["vol_ratio"] = f["vol5"] / f["vol60"].replace(0, np.nan)

    # trend position
    for n in (15, 60):
        ma = c.rolling(n).mean()
        sd = c.rolling(n).std().replace(0, np.nan)
        f[f"zscore{n}"] = (c - ma) / sd

    f["rsi14"] = rsi(c, 14)
    f["rsi7"] = rsi(c, 7)

    # time-of-day context (gold trades ~23h; cyclical encoding)
    mins = df.index.hour * 60 + df.index.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    f["dow"] = df.index.dayofweek

    f["_ret"] = ret1
    return f


def make_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """1 if close moves up over the next `horizon` bars, else 0."""
    fwd = close.shift(-horizon) / close - 1
    return (fwd > 0).astype(int).rename("y")


def walk_forward(n: int, n_splits: int = 6, min_train: float = 0.5, embargo: int = 5):
    """Expanding-window forward splits with an embargo of `embargo` bars."""
    sizes = np.arange(min_train, 1.0, (1.0 - min_train) / n_splits)
    for frac in sizes:
        cut = int(n * frac)
        if cut + embargo >= n:
            break
        train = np.arange(0, cut)
        test = np.arange(cut + embargo, min(cut + int(n / n_splits), n))
        if len(test) < 50:
            continue
        yield train, test