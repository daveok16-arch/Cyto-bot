"""Tests for volatility forecasting.

Two things are worth guarding here:
  1. the horizon scaling, because getting it wrong makes a variance swap look
     absurdly profitable (the bug this suite was written after fixing)
  2. the overlap artifact, because a rolling-window vol estimate appears almost
     perfectly persistent purely as a function of its own window
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import volatility as V


def test_forward_rv_is_nan_at_the_tail():
    r = np.zeros(50)
    r[1:] = 0.001
    f = V.forward_realized_vol(r, horizon=5)
    assert np.isnan(f[-5:]).all()
    assert np.isfinite(f[:45]).all()


def test_forward_abs_move_is_nan_at_the_tail():
    c = np.linspace(100, 110, 50)
    a = V.forward_abs_move(c, horizon=5)
    assert np.isnan(a[-5:]).all()
    assert (a[:45] > 0).all()


def test_trailing_rv_is_causal():
    """A change at index k must not alter trailing vol before k."""
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.001, 200)
    before = V.trailing_realized_vol(r, 10)
    r2 = r.copy()
    r2[150:] += 0.05                      # shock only in the far tail
    after = V.trailing_realized_vol(r2, 10)
    assert np.allclose(before[:150], after[:150], equal_nan=True)


def test_horizon_scaling_is_sqrt_h():
    """The bug this guards: an unscaled per-bar vol against a horizon target is
    off by exactly sqrt(h), which makes a variance swap appear to print money."""
    vol = np.array([2.0, 4.0])
    assert np.allclose(V.to_horizon_scale(vol, 1), vol)
    assert V.to_horizon_scale(vol, 4)[0] == pytest.approx(4.0)
    assert V.to_horizon_scale(vol, 9)[1] == pytest.approx(12.0)


def test_variance_adds_under_scaling():
    """Consistency: variance of the scaled series should equal h * variance."""
    vol = np.array([1.5])
    h = 9
    scaled = V.to_horizon_scale(vol, h)
    assert scaled[0] ** 2 == pytest.approx(h * vol[0] ** 2)


def test_rolling_window_acf_tracks_overlap_fraction():
    """The artifact, demonstrated on pure noise: a rolling std of iid noise shows
    autocorrelation at lag k equal to the window overlap, not real persistence."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 1, 20000))   # iid: no persistence at all
    W = 15
    rv = r.rolling(W).std().dropna()
    for k in (1, 5, 10):
        acf = rv.autocorr(k)
        overlap = (W - k) / W
        assert abs(acf - overlap) < 0.05, f"lag {k}: acf={acf:.3f} overlap={overlap:.3f}"


def test_nonoverlapping_blocks_reveal_no_persistence_in_noise():
    """Once the window overlap is removed, iid noise shows ~zero correlation."""
    rng = np.random.default_rng(2)
    W = 15
    r = rng.normal(0, 1, W * 500)
    blocks = r.reshape(500, W).std(axis=1)
    c = np.corrcoef(blocks[:-1], blocks[1:])[0, 1]
    assert abs(c) < 0.1


def test_qlike_is_zero_for_a_perfect_forecast():
    rv = np.array([1.0, 2.0, 3.0, 4.0])
    assert V.qlike(rv, rv) == pytest.approx(0.0, abs=1e-9)


def test_qlike_penalises_under_and_over_forecast():
    rv = np.array([2.0, 2.0, 2.0, 2.0])
    assert V.qlike(rv, rv * 2) > 0
    assert V.qlike(rv, rv / 2) > 0


def test_mincer_zarnowitz_flags_a_calibrated_forecast():
    rng = np.random.default_rng(3)
    true = rng.normal(10, 2, 2000)
    pred = true + rng.normal(0, 0.5, 2000)
    mz = V.mincer_zarnowitz(true, pred)
    assert mz["beta"] == pytest.approx(1.0, abs=0.1)
    assert mz["alpha"] == pytest.approx(0.0, abs=1.0)
    assert mz["r2"] > 0.9


def test_variance_swap_buyer_wins_when_realized_exceeds_strike():
    realized = np.array([10.0])
    strike = np.array([5.0])
    pnl = V.variance_swap_pnl(realized, strike, side=np.array([1.0]))
    assert pnl[0] == pytest.approx(100.0 - 25.0)


def test_straddle_buyer_wins_only_if_move_exceeds_premium():
    assert V.straddle_pnl(np.array([10.0]), np.array([8.0]),
                          np.array([1.0]))[0] == pytest.approx(2.0)
    assert V.straddle_pnl(np.array([5.0]), np.array([8.0]),
                          np.array([1.0]))[0] == pytest.approx(-3.0)


def test_parkinson_rv_is_positive_and_scales_with_range():
    hi = np.array([101.0, 102.0])
    lo = np.array([100.0, 100.0])
    p = V.parkinson_rv(hi, lo)
    assert np.all(p > 0)
    assert p[1] > p[0]


def test_har_design_has_expected_columns():
    rv = pd.Series(np.log(np.abs(np.random.default_rng(4).normal(1, 0.1, 100))))
    X = V.har_design(rv, lags=(1, 5, 20))
    assert list(X.columns) == ["d", "m1", "m5", "m20"]
    assert len(X) == len(rv)