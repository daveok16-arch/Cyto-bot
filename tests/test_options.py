"""Tests for option chain parsing and capture analysis.

The guards here are about not trusting fields blindly: the CBOE `delta` column
returned an in-the-money put as "25-delta", which inverted the skew sign until
selection was switched to moneyness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import options as O
from src.run_capture import bs_straddle, implied_vol_straddle


# ------------------------------------------------------------- OSI parsing
def test_parse_osi_call():
    root, exp, kind, strike = O.parse_osi("SPX261016C00200000")
    assert root == "SPX" and kind == "call" and strike == pytest.approx(200.0)
    assert exp == pd.Timestamp("2026-10-16")


def test_parse_osi_put():
    root, exp, kind, strike = O.parse_osi("SPX261016P00765000")
    assert kind == "put" and strike == pytest.approx(765.0)


def test_parse_osi_rejects_garbage():
    assert O.parse_osi("NOTASYMBOL") is None
    assert O.parse_osi("SPX261016X00200000") is None


# ------------------------------------------------------------- BS helpers
def test_bs_straddle_positive_and_increases_with_vol():
    a = bs_straddle(100.0, 100.0, 0.10, 30 / 365)
    b = bs_straddle(100.0, 100.0, 0.30, 30 / 365)
    assert a > 0 and b > a


def test_implied_vol_round_trips():
    """Inverting a BS straddle price must recover the vol that produced it."""
    for sigma in (0.08, 0.15, 0.40):
        px = bs_straddle(5000.0, 5000.0, sigma, 30 / 365)
        got = implied_vol_straddle(px, 5000.0, 5000.0, 30 / 365)
        assert got == pytest.approx(sigma, abs=1e-3)


def test_implied_vol_handles_expiry_edge():
    assert np.isnan(implied_vol_straddle(10.0, 100.0, 100.0, T=0.0))


def test_bs_straddle_at_expiry_is_intrinsic():
    assert bs_straddle(105.0, 100.0, 0.2, 0.0) == pytest.approx(5.0)


# ------------------------------------------------------------- chain logic
def _fake_chain(spot=100.0, expiry="2026-10-16"):
    exp = pd.Timestamp(expiry)
    strikes = np.arange(spot * 0.7, spot * 1.3, spot * 0.01)
    rows = []
    for k in strikes:
        for kind in ("call", "put"):
            # classic equity skew: puts richer below spot
            moneyness = (k - spot) / spot
            iv = 0.20 + (0.05 + 0.6 * max(0, -moneyness)) if kind == "put" else 0.20 - 0.4 * max(0, moneyness)
            mid = abs(spot - k) + spot * iv * 0.05
            rows.append({"symbol": "", "expiry": exp, "kind": kind,
                         "strike": float(k), "bid": mid * 0.99,
                         "ask": mid * 1.01, "iv": iv, "delta": np.nan,
                         "vega": 1.0, "open_interest": 100.0, "volume": 10.0})
    return pd.DataFrame(rows)


def test_atm_straddle_picks_nearest_strike():
    ch = _fake_chain(spot=100.0)
    s = O.atm_straddle(ch, 100.0, pd.Timestamp("2026-10-16"))
    assert s is not None
    assert abs(s["strike"] - 100.0) <= 1.0
    assert s["straddle_ask"] > s["straddle_bid"]


def test_skew_selects_by_moneyness_not_delta():
    """The bug: trusting the delta field inverted the sign. With moneyness
    selection, an equity-style skew must come out positive (puts richer)."""
    ch = _fake_chain(spot=100.0)
    sk = O.skew(ch, pd.Timestamp("2026-10-16"), 100.0, otm_pct=0.05)
    assert sk is not None
    assert sk["skew"] > 0, "puts should be richer than calls below spot"
    assert sk["put_strike"] < 100.0 < sk["call_strike"]


def test_liquid_expiries_filters_thin_ones():
    ch = _fake_chain(spot=100.0)
    exps = O.liquid_expiries(ch, 100.0, min_strikes=20)
    assert pd.Timestamp("2026-10-16") in exps
    # a chain with almost no strikes should be excluded
    thin = ch[ch["strike"].between(99, 101)]
    assert O.liquid_expiries(thin, 100.0, min_strikes=20) == []


def test_put_spread_credit_is_less_than_width():
    ch = _fake_chain(spot=100.0)
    ps = O.put_spread_quote(ch, 100.0, pd.Timestamp("2026-10-16"), width_pct=0.05)
    assert ps is not None
    assert ps["k_long"] < ps["k_short"]
    assert ps["credit"] < ps["width"], "a spread must not pay more than its width"
    assert ps["max_loss"] > 0


def test_straddle_theoretical_is_scaled_correctly():
    """S*sigma*sqrt(2T/pi): a 20% vol 1-month straddle on 100 is ~4.6."""
    th = O.straddle_theoretical(100.0, 0.20, 30)
    assert 4.0 < th < 5.2


def test_straddle_theoretical_scales_with_time():
    a = O.straddle_theoretical(100.0, 0.2, 10)
    b = O.straddle_theoretical(100.0, 0.2, 40)
    assert b > a
    assert b / a == pytest.approx(np.sqrt(4), rel=0.02)