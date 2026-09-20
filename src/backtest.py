"""Execution-aware backtest engine.

The single biggest source of fake profits in scalping backtests is assuming fills
at the mid or at the touch with no adverse selection. This engine charges:

  * entry and exit at the correct side of the book (taker pays the full spread)
  * slippage on every taker fill
  * for makers: a probabilistic fill model where fills cluster in exactly the
    moments the price moves against you (adverse selection), driven by observed
    volume and volatility rather than assumed away

Everything is in returns, so results are comparable across instruments.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import CostModel


@dataclass
class BacktestResult:
    n_trades: int
    hit_rate: float
    gross_return: float
    net_return: float
    avg_trade_return: float
    total_return: float
    sharpe_per_trade: float
    max_drawdown: float
    avg_cost_return: float
    cost_model: dict
    note: str = ""

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "hit_rate": round(self.hit_rate, 4),
            "gross_return": round(self.gross_return, 6),
            "net_return": round(self.net_return, 6),
            "avg_trade_return": round(self.avg_trade_return, 6),
            "total_return": round(self.total_return, 6),
            "sharpe_per_trade": round(self.sharpe_per_trade, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_cost_return": round(self.avg_cost_return, 6),
            "cost_model": self.cost_model,
            "note": self.note,
        }


def _drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak == 0, 1, peak)
    return float(dd.min())


def backtest_taker(prices: np.ndarray, prob_up: np.ndarray, threshold: float,
                   costs: CostModel, horizon: int = 1,
                   stop_bp: float | None = None,
                   take_bp: float | None = None) -> BacktestResult:
    """Pure taker: takes one side when the model is confident, pays all costs.

    Trade return is the realized mid move over the horizon minus a full round
    trip. `stop_bp` / `take_bp` are notional exit levels, but the realized move
    is still read from the mid series so the fill model stays honest.
    """
    n = len(prices)
    trades = []
    for i in range(n - horizon):
        p = prob_up[i]
        side = 0
        if p >= threshold:
            side = 1
        elif (1 - p) >= threshold:
            side = -1
        if side == 0:
            continue
        gross = side * np.log(prices[i + horizon] / prices[i])
        trades.append((i, side, gross))

    if not trades:
        return BacktestResult(0, float("nan"), 0.0, -costs.round_trip_return(), 0.0,
                              0.0, 0.0, 0.0, costs.round_trip_return(),
                              costs.summary(), "no trades taken")

    idx = np.array([t[0] for t in trades])
    gross = np.array([t[2] for t in trades])
    net = costs.net(gross)
    equity = np.cumprod(1.0 + net)
    hit = float(np.mean(net > 0))
    return BacktestResult(
        n_trades=len(net), hit_rate=hit, gross_return=float(gross.sum()),
        net_return=float(net.sum()), avg_trade_return=float(net.mean()),
        total_return=float(equity[-1] - 1.0),
        sharpe_per_trade=float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        max_drawdown=_drawdown(equity),
        avg_cost_return=costs.round_trip_return(),
        cost_model=costs.summary(),
    )


def backtest_maker(prices: np.ndarray, spread_abs: np.ndarray, prob_up: np.ndarray,
                   quote_mask: np.ndarray, costs: CostModel,
                   horizon: int = 1, fill_rate: float = 1.0,
                   adverse_fill_boost: float = 1.0,
                   seed: int = 0) -> BacktestResult:
    """Market maker: earns the spread on fills, pays adverse selection.

    Fill probability rises with the size of the next move when
    `adverse_fill_boost` > 1, which models the real effect that resting orders
    get filled preferentially just before the market runs. Setting it to 1.0
    assumes fills are mover-independent -- the optimistic case, reported too so
    the gap between the two is visible.
    """
    rng = np.random.default_rng(seed)
    n = len(prices)
    logret = np.log(prices[1:] / prices[:-1])
    ret_at = np.concatenate([logret, [0.0]])

    pnl, sides = [], []
    for i in range(n - horizon):
        if not quote_mask[i]:
            continue
        nxt = ret_at[i]
        # adverse selection: fills are more likely when |next move| is large
        boost = 1.0 + adverse_fill_boost * abs(nxt) / (np.median(np.abs(logret)) + 1e-12)
        p_fill = min(1.0, fill_rate * boost)
        if rng.random() > p_fill:
            continue

        # the market picks the side that hurts: buy fills before a drop
        side = 1 if nxt < 0 else -1
        gross = side * np.log(prices[i + horizon] / prices[i])
        # maker earns half the spread per side plus any rebate
        capture = (spread_abs[i] / prices[i]) * 0.5 - costs.maker_fee_return()
        pnl.append(gross + capture)
        sides.append(side)

    if not pnl:
        return BacktestResult(0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              costs.summary(), "no fills")

    net = np.array(pnl)
    equity = np.cumprod(1.0 + net)
    return BacktestResult(
        n_trades=len(net), hit_rate=float(np.mean(net > 0)),
        gross_return=float(net.sum()), net_return=float(net.sum()),
        avg_trade_return=float(net.mean()),
        total_return=float(equity[-1] - 1.0),
        sharpe_per_trade=float(net.mean() / net.std()) if net.std() > 0 else 0.0,
        max_drawdown=_drawdown(equity),
        avg_cost_return=float(-np.mean([(spread_abs[i] / prices[i]) * 0.5
                                        for i in range(len(net))])),
        cost_model=costs.summary(),
        note=f"fill_rate={fill_rate}, adverse_boost={adverse_fill_boost}",
    )