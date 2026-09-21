"""Tests for the prediction bot's core contract.

The decision gate is the part that protects capital, so it is tested directly:
the bot must refuse to trade whenever the model's probability fails to clear the
breakeven plus margin. Everything else is secondary.

Run: python -m pytest tests/ -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bot import BotConfig, PredictionBot, breakeven


# ---------------------------------------------------------------- the gate
def test_breakeven_matches_payout_arithmetic():
    assert breakeven(0.80) == pytest.approx(0.5556, abs=1e-4)
    assert breakeven(0.85) == pytest.approx(0.5405, abs=1e-4)
    assert breakeven(0.90) == pytest.approx(0.5263, abs=1e-4)


def _fitted_bot(**kw):
    """Fit on synthetic data where the sign of ret1 carries a real edge, so the
    bot's plumbing is exercised rather than its accuracy."""
    rng = np.random.default_rng(0)
    n = 4000
    sign = rng.choice([-1.0, 1.0], size=n)          # ret1 column
    prob_up = np.where(sign > 0, 0.62, 0.38)
    y = (rng.random(n) < prob_up).astype(int)
    X = np.zeros((n, 5), dtype=np.float32)
    X[:, 0] = sign                                    # ret1
    X[:, 1] = rng.normal(size=n)
    X[:, 2] = rng.normal(size=n)
    X[:, 3] = rng.normal(size=n)
    X[:, 4] = rng.normal(size=n)
    bot = PredictionBot(BotConfig(**kw))
    bot.fit(X[: int(n * 0.7)], y[: int(n * 0.7)], sign_col=0)
    return bot, X, y


def test_declines_when_edge_is_below_threshold():
    bot, _, _ = _fitted_bot(payout=0.80, min_edge=0.10)  # unreachably high bar
    d = bot.decide(0.60)
    assert d.action == "NO_TRADE"
    assert d.edge < 0.10
    assert "does not clear" in d.reason


def test_trades_up_when_probability_clears_the_bar():
    bot, _, _ = _fitted_bot(payout=0.80, min_edge=0.005)
    d = bot.decide(0.60)
    assert d.action == "UP"
    assert d.p_up == pytest.approx(0.60)
    assert d.edge == pytest.approx(0.60 - breakeven(0.80), abs=1e-9)


def test_trades_down_on_a_confident_low_probability():
    bot, _, _ = _fitted_bot(payout=0.80, min_edge=0.005)
    d = bot.decide(0.40)
    assert d.action == "DOWN"
    assert d.edge == pytest.approx(0.60 - breakeven(0.80), abs=1e-9)


def test_coin_flip_never_trades():
    """At p=0.5 both sides are at -breakeven, so a correct bot always declines."""
    for payout in (0.70, 0.80, 0.85, 0.90):
        bot, _, _ = _fitted_bot(payout=payout, min_edge=0.0)
        assert bot.decide(0.5).action == "NO_TRADE"


def test_higher_payout_makes_the_bot_stricter():
    """A better payout lowers breakeven, so the same probability can flip from
    NO_TRADE to a trade."""
    p = 0.545
    strict = _fitted_bot(payout=0.80, min_edge=0.0)[0].decide(p)
    loose = _fitted_bot(payout=0.90, min_edge=0.0)[0].decide(p)
    assert strict.action == "NO_TRADE"   # 0.545 < 0.5556
    assert loose.action == "UP"          # 0.545 > 0.5263


# ---------------------------------------------------------------- plumbing
def test_predict_proba_is_bounded_and_sized():
    bot, X, _ = _fitted_bot()
    p = bot.predict_proba(X[-200:])
    assert p.shape == (200,)
    assert np.all((p > 0) & (p < 1))


def test_decide_many_matches_individual_decisions():
    bot, X, _ = _fitted_bot()
    Xt = X[-300:]
    many = bot.decide_many(Xt)
    single = [bot.decide(float(pi)) for pi in bot.predict_proba(Xt)]
    assert [d.action for d in many] == [d.action for d in single]


def test_unfitted_bot_refuses_to_predict():
    with pytest.raises(RuntimeError):
        PredictionBot().predict_proba(np.zeros((5, 3), dtype=np.float32))


def test_fit_rejects_tiny_samples():
    with pytest.raises(ValueError):
        PredictionBot().fit(np.zeros((10, 3), dtype=np.float32), np.zeros(10, int))