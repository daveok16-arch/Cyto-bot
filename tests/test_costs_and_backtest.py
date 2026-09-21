"""Tests for the cost model and backtest engine.

These guard the assumptions that decide whether a scalping result is real:
costs are charged, a zero-edge strategy loses, and a real edge is detectable.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest import backtest_taker
from src.costs import CostModel, required_hit_rate
from src.scalp_bot import ScalpBot, ScalpConfig


def test_round_trip_includes_spread_and_both_legs_of_fees():
    c = CostModel(spread_bp=1.56, commission_bp=0.02, slippage_bp=0.05)
    # spread once + (fee+slip) twice
    assert c.round_trip_bp == pytest.approx(1.56 + 2 * (0.02 + 0.05))
    assert c.round_trip_return() == pytest.approx(c.round_trip_bp / 1e4)


def test_net_deducts_exactly_one_round_trip():
    c = CostModel()
    gross = np.array([0.0, 0.001, -0.001])
    net = c.net(gross)
    assert np.allclose(net, gross - c.round_trip_return())


def test_required_hit_rate_rises_with_costs():
    no_cost = required_hit_rate(payoff_ratio=1.0, cost_return=0.0)
    with_cost = required_hit_rate(payoff_ratio=1.0, cost_return=0.0017)
    assert no_cost == pytest.approx(0.5, abs=1e-9)
    assert with_cost > no_cost


# ------------------------------------------------------- engine behaviour
def _random_walk(n=8000, seed=0, sigma=2e-4):
    rng = np.random.default_rng(seed)
    return 4300 * np.exp(np.cumsum(rng.normal(0, sigma, n)))


def test_zero_edge_strategy_loses_after_costs():
    """The central property: random signals must not appear profitable."""
    prices = _random_walk()
    prob = np.random.default_rng(1).random(len(prices))
    r = backtest_taker(prices, prob, threshold=0.55, costs=CostModel())
    assert r.n_trades > 1000
    assert r.net_return < 0
    # net should equal gross minus the cost charged per trade
    expected = r.gross_return - r.n_trades * CostModel().round_trip_return()
    assert r.net_return == pytest.approx(expected, rel=1e-9)


def test_perfect_oracle_can_be_profitable_when_costs_are_zero():
    """Guards against an engine that cannot detect a real edge."""
    rng = np.random.default_rng(2)
    n = 8000
    rets = rng.normal(0, 2e-4, n)
    prices = 4300 * np.exp(np.cumsum(rets))
    nxt = np.concatenate([rets[1:], [0.0]])
    prob = np.where(nxt > 0, 0.99, 0.01)
    r = backtest_taker(prices, prob, threshold=0.55,
                       costs=CostModel(spread_bp=0, commission_bp=0, slippage_bp=0))
    assert r.net_return > 0
    assert r.hit_rate > 0.99


def test_costs_can_turn_a_real_edge_negative():
    """The finding that drives the whole scalping analysis: a genuine edge is
    not sufficient when the typical move is smaller than the round trip."""
    rng = np.random.default_rng(3)
    n = 8000
    rets = rng.normal(0, 1.5e-4, n)          # small moves
    prices = 4300 * np.exp(np.cumsum(rets))
    nxt = np.concatenate([rets[1:], [0.0]])
    prob = np.where(nxt > 0, 0.99, 0.01)      # perfect direction
    free = backtest_taker(prices, prob, 0.55, CostModel(0, 0, 0))
    paid = backtest_taker(prices, prob, 0.55, CostModel())
    assert free.net_return > 0
    assert paid.net_return < 0


def test_no_trades_is_reported_not_crashed():
    prices = _random_walk(500)
    prob = np.full(500, 0.5)
    r = backtest_taker(prices, prob, threshold=0.90, costs=CostModel())
    assert r.n_trades == 0
    assert "no trades" in r.note


# ------------------------------------------------------- scalping gate
def test_magnitude_gate_blocks_small_expected_moves():
    cfg = ScalpConfig(cost=CostModel(), min_move_mult=2.0)
    bot = ScalpBot(cfg)
    cost = cfg.cost.round_trip_bp
    # expected move below 2x cost -> must decline even with perfect direction
    d = bot.decide(0.99, exp_move_bp=cost * 1.5)
    assert d.action == "NO_TRADE"
    assert "required" in d.reason


def test_magnitude_gate_allows_large_expected_moves():
    cfg = ScalpConfig(cost=CostModel(), min_move_mult=1.5, min_dir_edge=0.0)
    bot = ScalpBot(cfg)
    cost = cfg.cost.round_trip_bp
    d = bot.decide(0.70, exp_move_bp=cost * 3.0)
    assert d.action == "UP"
    assert d.edge_bp == pytest.approx(cost * 2.0, abs=1e-9)


def test_direction_gate_blocks_coin_flip_even_with_big_move():
    cfg = ScalpConfig(cost=CostModel(), min_move_mult=1.0, min_dir_edge=0.05)
    bot = ScalpBot(cfg)
    d = bot.decide(0.51, exp_move_bp=100.0)
    assert d.action == "NO_TRADE"
    assert "direction too weak" in d.reason


def test_low_probability_routes_to_down():
    cfg = ScalpConfig(cost=CostModel(), min_move_mult=1.0)
    bot = ScalpBot(cfg)
    d = bot.decide(0.20, exp_move_bp=100.0)
    assert d.action == "DOWN"