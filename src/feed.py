"""Data feed for the bot: historical bars for fitting and a live-style window.

Narrow by design. Two jobs:
  * `load_dataset()` -> (X, y, timestamps) from cached ticks, for fitting.
  * `MarketFeed.latest_bars()` -> the most recent bars, for serving one decision.

The same feature code (`src.flow`) is used for both, which is what prevents
training/serving skew.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import flow, ticks

MIN_HOURS = 20  # gold trades ~23h/day; hour 21 UTC is the daily break


def cached_days(min_hours: int = MIN_HOURS) -> list[dt.date]:
    days = set()
    for f in ticks.HOURS.glob("XAUUSD_*.csv.gz"):
        if f.stat().st_size == 0:
            continue
        s = f.stem.split("_")[1][:8]
        days.add(dt.date(int(s[:4]), int(s[4:6]), int(s[6:8])))
    return sorted(d for d in days if ticks.cached_count(d) >= min_hours)


def load_bars(bar: str = "5min") -> pd.DataFrame:
    days = cached_days()
    if not days:
        raise SystemExit("no cached ticks; run: python -m src.download_ticks")
    frames = []
    for d in days:
        x = ticks.load_day(d)
        if len(x):
            frames.append(x)
    t = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    return flow.to_bars(t, bar)


def load_dataset(bar: str = "5min", order_flow: bool = True, horizon: int = 1):
    """Returns X (DataFrame), y (Series), timestamps (Index)."""
    bars = load_bars(bar)
    feat = flow.order_flow_features(bars)
    y = flow.make_labels(bars["close"].set_axis(feat.index), horizon=horizon)
    X = feat.drop(columns=["_ret"])
    if not order_flow:
        of = ["ofi", "ofi_ma5", "ofi_ma15", "vol_z", "n_tick_z", "spread_bp",
              "spread_z", "impact", "queue_imb", "rv_ofi_corr"]
        X = X.drop(columns=[c for c in of if c in X.columns])
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid], X.index[valid]


class MarketFeed:
    """Replays historical bars as if they were live, one bar at a time."""

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.ts = X.index
        self.i = 0
        self.sign_col = list(X.columns).index("ret1")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def latest(self):
        """The current bar's features, plus the realized outcome if known."""
        row = self.X.iloc[self.i : self.i + 1].to_numpy(np.float32)
        outcome = int(self.y.iloc[self.i]) if self.i < len(self.y) else -1
        return row, outcome, str(self.ts[self.i])

    def step(self):
        self.i += 1

    def rewind(self):
        self.i = 0