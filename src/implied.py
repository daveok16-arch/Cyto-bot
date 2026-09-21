"""Implied volatility data: CBOE VIX and Deribit DVOL.

The variance risk premium is the one hypothesis in this project with a structural
reason to be profitable, and it requires IMPLIED volatility -- the price the
market charges for volatility -- which no price-only dataset can supply.

  * VIX   -- 30-day implied vol on the S&P 500, daily since 1990 (~9,000 points).
             The primary dataset: long enough to span multiple regimes.
  * DVOL  -- Deribit's 30-day implied vol index for BTC, ~1,000 daily points.
             An independent cross-check on a different asset class.

Both are fetched over HTTP and cached to disk. Nothing here is simulated.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout: int = 45) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read()


def fetch_vix(cache: bool = True) -> pd.DataFrame:
    """CBOE VIX daily history. Columns: date, open, high, low, close."""
    path = DATA / "vix_history.csv"
    if cache and path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    raw = _get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_spx(cache: bool = True) -> pd.DataFrame:
    """S&P 500 daily closes from Yahoo, for realized vol on the same calendar."""
    path = DATA / "spx_daily.csv"
    if cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    out = []
    now = int(time.time())
    # Yahoo caps range lookback; step 10y windows from 1990
    for start in range(int(pd.Timestamp("1985-01-01").timestamp()), now, 10 * 365 * 86400):
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
               f"?period1={start}&period2={start + 10 * 365 * 86400}&interval=1d")
        try:
            j = json.loads(_get(url))
            r = j["chart"]["result"][0]
            ts = r.get("timestamp", [])
            cl = r["indicators"]["quote"][0]["close"]
            for t, c in zip(ts, cl):
                if c is not None:
                    out.append((pd.Timestamp(t, unit="s").normalize(), float(c)))
        except Exception:
            continue
    df = pd.DataFrame(out, columns=["date", "close"]).drop_duplicates("date")
    df = df.sort_values("date").reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def fetch_dvol(days: int = 1000, cache: bool = True) -> pd.DataFrame:
    """Deribit DVOL for BTC. Chrome-style prices: [ts, open, high, low, close]."""
    path = DATA / "dvol_btc.csv"
    if cache and path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    now = int(time.time() * 1000)
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data"
           f"?currency=BTC&start_timestamp={now - days*86400*1000}"
           f"&end_timestamp={now}&resolution=1D")
    j = json.loads(_get(url))["result"]["data"]
    df = pd.DataFrame(j, columns=["ts", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df[["date", "open", "high", "low", "close"]].sort_values("date")
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df.reset_index(drop=True)


def fetch_btc(cache: bool = True) -> pd.DataFrame:
    path = DATA / "btc_daily.csv"
    if cache and path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
           "?range=5y&interval=1d")
    j = json.loads(_get(url))
    r = j["chart"]["result"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(r["timestamp"], unit="s").normalize(),
        "close": r["indicators"]["quote"][0]["close"],
    }).dropna().reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------ realized vol
def realized_vol_from_close(close: pd.Series, window: int = 21,
                            forward: bool = True) -> pd.Series:
    """Annualized realized vol over `window` trading days.

    forward=True gives the vol realized AFTER each date, which is what an
    implied-vol quote at that date is trying to predict. This orientation is the
    whole point of the study and getting it backwards would invert the result.
    """
    r = np.log(close / close.shift(1))
    if forward:
        return r.rolling(window).std().shift(-window) * np.sqrt(252) * 100
    return r.rolling(window).std() * np.sqrt(252) * 100


def align_iv_rv(iv: pd.DataFrame, px: pd.DataFrame, iv_col: str = "close",
                window: int = 21) -> pd.DataFrame:
    """Join implied vol to the realized vol it is forecasting."""
    a = iv[["date", iv_col]].rename(columns={iv_col: "iv"})
    b = px[["date", "close"]].copy()
    b["rv_fwd"] = realized_vol_from_close(b["close"], window, forward=True)
    b["rv_trail"] = realized_vol_from_close(b["close"], window, forward=False)
    m = a.merge(b, on="date", how="inner").dropna()
    return m.sort_values("date").reset_index(drop=True)