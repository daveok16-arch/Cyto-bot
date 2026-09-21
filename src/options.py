"""Real option chain ingestion and analysis.

The VRP test showed the premium exists at the index level. That is not the same
as being able to capture it: the moment you trade options you pay option bid-ask
spreads, and those are far wider (in vol terms) than the underlying's spread.

This module pulls REAL chains so the cost side is measured rather than assumed:

  * CBOE SPX chain -- ~30,000 quotes with bid/ask, IV, delta, vega, open interest
  * Deribit BTC options -- full book summary with bid/ask by strike

A live chain is a snapshot, so it cannot be a historical backtest. What it CAN do
is calibrate realistic transaction costs, which is exactly the missing piece --
the historical test had no cost model at all.

OSI symbol format: ROOT + YYMMDD + C/P + strike*1000
e.g. SPX261016C00200000 -> SPX, 2026-10-16, Call, strike 200.000
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}
OSI = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def _get(url: str, timeout: int = 45) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout).read()


def parse_osi(sym: str) -> tuple[str, pd.Timestamp, str, float] | None:
    m = OSI.match(sym)
    if not m:
        return None
    root, ymd, cp, strike = m.groups()
    return (root, pd.Timestamp("20" + ymd), "call" if cp == "C" else "put",
            int(strike) / 1000.0)


def fetch_spx_chain(cache: bool = True) -> tuple[pd.DataFrame, float, str]:
    """Returns (chain, underlying_price, as_of). Snapshot, not history."""
    path = DATA / "spx_chain.json"
    if cache and path.exists() and (time.time() - path.stat().st_mtime) < 3600:
        j = json.loads(path.read_text())
    else:
        raw = _get("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
        j = json.loads(raw)
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    rows = []
    for o in j["data"]["options"]:
        p = parse_osi(o["option"])
        if p is None:
            continue
        root, expiry, kind, strike = p
        rows.append({
            "symbol": o["option"], "expiry": expiry, "kind": kind,
            "strike": strike, "bid": o.get("bid"), "ask": o.get("ask"),
            "iv": o.get("iv"), "delta": o.get("delta"), "vega": o.get("vega"),
            "open_interest": o.get("open_interest"), "volume": o.get("volume"),
        })
    df = pd.DataFrame(rows)
    spot = float(j["data"]["current_price"])
    return df, spot, j.get("timestamp", "")


def fetch_deribit_chain(cache: bool = True) -> pd.DataFrame:
    """Deribit BTC option book summary: real bid/ask by strike."""
    path = DATA / "deribit_chain.json"
    if cache and path.exists() and (time.time() - path.stat().st_mtime) < 1800:
        j = json.loads(path.read_text())
    else:
        raw = _get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
                   "?currency=BTC&kind=option")
        j = json.loads(raw)
        DATA.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    rows = []
    for o in j["result"]:
        name = o["instrument_name"]           # BTC-26MAR27-40000-C
        parts = name.split("-")
        if len(parts) != 4:
            continue
        rows.append({
            "symbol": name, "expiry": pd.Timestamp(parts[1]),
            "strike": float(parts[2]),
            "kind": "call" if parts[3] == "C" else "put",
            "bid": o.get("bid_price"), "ask": o.get("ask_price"),
            "mid": o.get("mark_price"), "iv": o.get("mark_iv"),
            "open_interest": o.get("open_interest"), "volume": o.get("volume"),
            "underlying": o.get("underlying_price"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------- chain analysis
def atm_straddle(chain: pd.DataFrame, spot: float, expiry: pd.Timestamp,
                 max_spread_pct: float | None = None) -> dict | None:
    """Price the at-the-money straddle from real quotes.

    Returns bid/ask/mid for the straddle (call + put at the nearest strike) and
    the spread as a fraction of mid -- the actual cost of putting the position on.
    """
    q = chain[chain["expiry"] == expiry].copy()
    q = q.dropna(subset=["bid", "ask"])
    q = q[(q["ask"] > 0) & (q["bid"] >= 0)]
    if q.empty:
        return None
    strikes = sorted(q["strike"].unique())
    k = min(strikes, key=lambda s: abs(s - spot))
    c = q[(q["strike"] == k) & (q["kind"] == "call")]
    p = q[(q["strike"] == k) & (q["kind"] == "put")]
    if c.empty or p.empty:
        return None
    c, p = c.iloc[0], p.iloc[0]
    bid = float(c["bid"]) + float(p["bid"])
    ask = float(c["ask"]) + float(p["ask"])
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    spread_pct = (ask - bid) / mid
    if max_spread_pct is not None and spread_pct > max_spread_pct:
        return None
    return {"strike": float(k), "expiry": expiry, "call_bid": float(c["bid"]),
            "call_ask": float(c["ask"]), "put_bid": float(p["bid"]),
            "put_ask": float(p["ask"]), "straddle_bid": bid, "straddle_ask": ask,
            "mid": mid, "spread_pct": spread_pct,
            "iv_call": float(c["iv"]) if pd.notna(c["iv"]) else np.nan,
            "iv_put": float(p["iv"]) if pd.notna(p["iv"]) else np.nan}


def straddle_theoretical(spot: float, iv: float, days: float) -> float:
    """Standard ATM straddle approximation: S * sigma * sqrt(2T/pi).

    Used only to sanity-check that the empirical quotes are in a sensible range,
    never as a substitute for a real quote.
    """
    T = days / 365.0
    return spot * iv * np.sqrt(2 * T / np.pi)


def put_spread_quote(chain: pd.DataFrame, spot: float, expiry: pd.Timestamp,
                     width_pct: float = 0.05) -> dict | None:
    """Defined-risk alternative: sell ATM put, buy a further OTM put.

    Caps the loss at the strike width, at the cost of two spreads and giving up
    the tail premium. Compares directly against the naked put.
    """
    q = chain[(chain["expiry"] == expiry) & (chain["kind"] == "put")].copy()
    q = q.dropna(subset=["bid", "ask"])
    q = q[(q["ask"] > 0) & (q["bid"] >= 0)].sort_values("strike")
    if len(q) < 5:
        return None
    k_short = min(q["strike"], key=lambda s: abs(s - spot))
    k_long_target = k_short * (1 - width_pct)
    k_long = min(q["strike"], key=lambda s: abs(s - k_long_target))
    if k_long == k_short:
        return None
    s = q[q["strike"] == k_short].iloc[0]
    l = q[q["strike"] == k_long].iloc[0]
    # sell the short put (receive bid), buy the long put (pay ask)
    credit = float(s["bid"]) - float(l["ask"])
    cost_if_pinned = k_short - k_long
    return {"k_short": float(k_short), "k_long": float(k_long),
            "credit": credit, "max_loss": cost_if_pinned - credit,
            "short_bid": float(s["bid"]), "long_ask": float(l["ask"]),
            "width": float(k_short - k_long)}


def skew(chain: pd.DataFrame, expiry: pd.Timestamp, spot: float,
         otm_pct: float = 0.05) -> dict | None:
    """Put IV minus call IV at matched out-of-the-money strikes.

    Selection is by MONEYNESS, not by the chain's delta field: the delta column
    in the CBOE feed proved unreliable (it returned an in-the-money put as
    '25-delta', which inverted the skew). A ~5% OTM strike is the standard
    25-delta region for a 30-day option and is robust to feed quirks.

    Steep positive skew is why crash protection is expensive and why the short-vol
    tail is costly to insure.
    """
    q = chain[(chain["expiry"] == expiry)].dropna(subset=["iv"])
    q = q[q["iv"] > 0]
    if q.empty:
        return None
    k_put = min(q["strike"].unique(), key=lambda s: abs(s - spot * (1 - otm_pct)))
    k_call = min(q["strike"].unique(), key=lambda s: abs(s - spot * (1 + otm_pct)))
    put = q[(q["kind"] == "put") & (q["strike"] == k_put)]
    call = q[(q["kind"] == "call") & (q["strike"] == k_call)]
    if put.empty or call.empty:
        return None
    pr, cr = put.iloc[0], call.iloc[0]
    return {"put_iv": float(pr["iv"]), "call_iv": float(cr["iv"]),
            "skew": float(pr["iv"] - cr["iv"]),
            "put_strike": float(pr["strike"]), "call_strike": float(cr["strike"])}


def liquid_expiries(chain: pd.DataFrame, spot: float,
                    min_strikes: int = 20) -> list[pd.Timestamp]:
    out = []
    for exp, g in chain.groupby("expiry"):
        near = g[(g["strike"] > spot * 0.9) & (g["strike"] < spot * 1.1)]
        if len(near) >= min_strikes:
            out.append(exp)
    return sorted(out)