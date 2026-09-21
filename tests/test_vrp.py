"""Tests for implied-volatility data and the variance risk premium.

The critical guard here is the overlap correction: 21-day realized vol computed
daily shares 20 of 21 days between neighbours, so naive t-stats are inflated by
roughly sqrt(21). A test asserts that overlapping windows really do inflate the
t-stat, so the correction in run_vrp.py cannot be silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import implied


def test_realized_vol_forward_looks_ahead_and_trailing_does_not():
    """forward=True must use FUTURE returns -- that is what IV forecasts."""
    px = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200))))
    fwd = implied.realized_vol_from_close(px, window=21, forward=True)
    trail = implied.realized_vol_from_close(px, window=21, forward=False)
    assert np.isnan(fwd.iloc[-21:]).all()
    assert np.isfinite(trail.iloc[-1])
    assert np.isnan(trail.iloc[:20]).all()


def test_forward_vol_is_causal_in_the_opposite_direction():
    """Changing a past return must not alter a forward window that excludes it."""
    rng = np.random.default_rng(1)
    px = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    fwd = implied.realized_vol_from_close(px, 21, forward=True)
    px2 = px.copy()
    px2.iloc[150:] *= 1.05           # only affects future windows
    fwd2 = implied.realized_vol_from_close(px2, 21, forward=True)
    assert np.allclose(fwd.iloc[:125], fwd2.iloc[:125], equal_nan=True)


def test_align_iv_rv_produces_finite_rows():
    dates = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(2)
    px = pd.DataFrame({"date": dates,
                       "close": 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))})
    iv = pd.DataFrame({"date": dates, "close": 18 + rng.normal(0, 2, 400)})
    m = implied.align_iv_rv(iv, px, "close", window=21)
    assert len(m) > 300
    assert m[["iv", "rv_fwd", "rv_trail"]].notna().all().all()
    assert list(m.columns) == ["date", "iv", "close", "rv_fwd", "rv_trail"]


def test_overlapping_windows_inflate_the_t_stat():
    """The reason run_vrp.py subsamples. Long-window targets overlap heavily, so
    treating every row as independent roughly sqrt(window) inflates significance."""
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(0, 1, n)
    # a genuine but small signal
    y = 0.05 * x + rng.normal(0, 1, n)

    def tstat(a, b):
        d = a - b
        return d.mean() / d.std() * np.sqrt(len(d))

    # overlapping: each row's target is a 21-step average of the next 21 rows
    def fwd_avg(v, w=21):
        out = np.full(len(v), np.nan)
        for i in range(len(v) - w):
            out[i] = v[i + 1: i + 1 + w].mean()
        return out

    yf = fwd_avg(y)
    xf = fwd_avg(x)
    m = np.isfinite(yf) & np.isfinite(xf)
    t_overlap = abs(tstat(yf[m], xf[m]))

    # non-overlapping: sample every 21st row
    idx = np.arange(0, m.sum(), 21)
    t_non = abs(tstat(yf[m][idx], xf[m][idx]))
    assert t_overlap > t_non, "overlap should inflate the t-stat"


def test_vrp_series_shifts_implied_forward():
    """The premium must pair a quote with the realized vol that FOLLOWS it."""
    m = pd.DataFrame({
        "iv": [20.0, 20.0, 20.0, 20.0],
        "rv_fwd": [10.0, 10.0, 10.0, 10.0],
        "rv_trail": [10.0, 10.0, 10.0, 10.0],
    })
    from src.run_vrp import vrp_series
    v = vrp_series(m, lag=1)
    # with lag=1 the first row is dropped and the rest are iv_prev - rv = +10
    assert len(v) == 3
    assert np.allclose(v, 10.0)


def test_short_vol_pnl_sign_is_positive_when_implied_exceeds_realized():
    from src.run_vrp import short_vol_pnl
    m = pd.DataFrame({"iv": [20.0, 20.0, 20.0], "rv_fwd": [10.0, 10.0, 10.0],
                      "rv_trail": [10.0, 10.0, 10.0]})
    p = short_vol_pnl(m, "always", lag=1)
    assert (p > 0).all()


def test_short_vol_pnl_is_negative_when_realized_exceeds_implied():
    """A vol spike reverses the sign -- the seller's loss mode."""
    from src.run_vrp import short_vol_pnl
    m = pd.DataFrame({"iv": [20.0, 20.0, 20.0], "rv_fwd": [60.0, 60.0, 60.0],
                      "rv_trail": [10.0, 10.0, 10.0]})
    p = short_vol_pnl(m, "always", lag=1)
    assert (p < 0).all()


def test_describe_returns_reports_negative_skew_for_tail_risk():
    from src.run_vrp import describe_returns
    rng = np.random.default_rng(4)
    benign = rng.normal(0.1, 1, 1000)
    benign[::100] = -20          # occasional crashes
    d = describe_returns(benign, "test")
    assert d["skew"] < 0
    assert d["max_drawdown_units"] < 0
    assert 0 < d["hit_rate"] < 1