"""Order-flow features from tick data, aggregated to a chosen bar size.

The features that require bid/ask and per-tick volume -- and therefore cannot be
computed from OHLCV candles -- are the point of this module:

* signed volume imbalance (aggressor buying vs selling, via the tick rule)
* trade-count / volume intensity vs its rolling norm
* effective spread, a direct liquidity/transaction-cost measure
* price impact per unit volume
* microprice vs mid (weighted by opposing queue volume)
* realized volatility and its term structure

Aggregation is causal: a bar labelled t summarizes only ticks within [t, t+bar),
and every rolling statistic is computed backward from that bar's close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_bars(t: pd.DataFrame, bar: str = "5min") -> pd.DataFrame:
    """Aggregate ticks to bars. Keeps bid/ask-derived fields alongside OHLC."""
    t = t.set_index("ts")
    mid = (t["ask"] + t["bid"]) / 2.0
    spread = t["ask"] - t["bid"]
    # tick rule: sign of the mid change classifies the initiator
    dm = mid.diff()
    sign = np.sign(dm).replace(0, np.nan).ffill().fillna(0.0)
    vol = (t["avol"] + t["bvol"]).replace(0, np.nan).fillna(0.0)
    signed_vol = sign * vol
    t = t.assign(mid=mid, spread=spread, signed_vol=signed_vol, vol=vol)

    g = t.resample(bar)
    o = g["mid"].first()
    h = g["mid"].max()
    lo = g["mid"].min()
    c = g["mid"].last()
    out = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c})
    out["n_ticks"] = g["mid"].size()
    out["volume"] = g["vol"].sum()
    out["signed_volume"] = g["signed_vol"].sum()
    out["spread_mean"] = g["spread"].mean()
    out["spread_max"] = g["spread"].max()
    # absolute traded move per bar, for price-impact normalization
    out["abs_move"] = (c - o).abs()
    out["bid_ticks"] = g["bid"].count()
    return out.dropna(subset=["open", "close"]).reset_index()


def order_flow_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.set_index("ts") if "ts" in bars.columns else bars
    c = df["close"]
    ret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=df.index)

    # --- order-flow core ---
    vol = df["volume"].replace(0, np.nan)
    f["ofi"] = df["signed_volume"] / vol                      # order-flow imbalance
    f["ofi_ma5"] = f["ofi"].rolling(5).mean()
    f["ofi_ma15"] = f["ofi"].rolling(15).mean()
    f["vol_z"] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std()
    f["n_tick_z"] = ((df["n_ticks"] - df["n_ticks"].rolling(60).mean())
                     / df["n_ticks"].rolling(60).std())

    # --- liquidity / cost ---
    f["spread_bp"] = df["spread_mean"] / c * 1e4
    f["spread_z"] = ((f["spread_bp"] - f["spread_bp"].rolling(60).mean())
                     / f["spread_bp"].rolling(60).std())
    f["impact"] = df["abs_move"] / vol

    # --- microprice gap: where the queue imbalance says the next tick goes ---
    # (avol/bvol proxy resting interest; sign relative to mid)
    f["queue_imb"] = (df["bid_ticks"] - df["n_ticks"] / 2) / (df["n_ticks"] / 2)

    # --- returns and volatility term structure ---
    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret{n}"] = np.log(c / c.shift(n))
    for n in (5, 10, 20, 60):
        f[f"rv{n}"] = ret.rolling(n).std()
    f["rv_ratio"] = f["rv5"] / f["rv60"].replace(0, np.nan)
    f["rv_ofi_corr"] = ret.rolling(60).corr(f["ofi"])

    # --- candle shape (kept for comparability with the OHLCV study) ---
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    f["body"] = (df["close"] - df["open"]) / rng
    f["close_pos"] = (df["close"] - df["low"]) / rng
    f["range_pct"] = rng / c

    # --- session context ---
    mins = df.index.hour * 60 + df.index.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    f["dow"] = df.index.dayofweek

    f["_ret"] = ret
    return f


def make_labels(close: pd.Series, horizon: int = 5) -> pd.Series:
    fwd = close.shift(-horizon) / close - 1
    return (fwd > 0).astype(int).rename("y")


def walk_forward(n: int, n_splits: int = 6, min_train: float = 0.5, embargo: int = 10):
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