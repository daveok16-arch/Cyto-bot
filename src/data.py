"""Load and cache XAUUSD 1-minute bars.

Source: Yahoo Finance GC=F (COMEX front-month gold futures), the practical free
proxy for spot XAUUSD. Yahoo caps 1-minute history at ~30 days, fetched in
<=7-day windows.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "xauusd_gcf_1m.csv"

OHLC = ["open", "high", "low", "close"]


def fetch(max_windows: int = 6, symbol: str = "GC=F") -> pd.DataFrame:
    import yfinance as yf

    end = dt.date.today()
    frames = []
    for i in range(max_windows):
        e = end - dt.timedelta(days=7 * i)
        s = e - dt.timedelta(days=7)
        x = yf.download(
            symbol, start=s, end=e, interval="1m", progress=False, auto_adjust=False
        )
        if len(x):
            frames.append(x)
    if not frames:
        raise RuntimeError("no data returned")
    df = pd.concat(frames).sort_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="last")]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH)
    return df


def load() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return fetch()
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df[OHLC].astype(float)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index.name = "time"
    return df.sort_index()