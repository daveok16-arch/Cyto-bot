"""Transaction cost model.

Scalping lives or dies on this module, so it is explicit rather than a fudge
factor. Costs are expressed in price units and converted to returns, so they are
comparable to the moves the model is trying to predict.

Measured on the cached data: median spread $0.68 on gold near $4369 (1.56 bp),
while the typical 1-second move is $0.095. The spread is ~7x the move. Any
strategy must overcome that, and this module makes sure the backtest actually
charges it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """All costs in basis points of notional, per side unless noted."""

    spread_bp: float = 1.56      # measured median, one full spread to cross and exit
    commission_bp: float = 0.02  # per side; ~$0 retail ECN is optimistic, keep a floor
    slippage_bp: float = 0.05    # per side; market orders do not fill at the touch

    # A round trip crosses the spread once in each direction plus commissions and
    # slippage on both legs.
    @property
    def round_trip_bp(self) -> float:
        return self.spread_bp + 2 * (self.commission_bp + self.slippage_bp)

    def round_trip_return(self) -> float:
        return self.round_trip_bp / 1e4

    def net(self, gross_return: np.ndarray) -> np.ndarray:
        """Deduct a full round trip from each realized trade return."""
        return np.asarray(gross_return, dtype=float) - self.round_trip_return()

    def breakeven_move_bp(self) -> float:
        return self.round_trip_bp

    def maker_fee_return(self) -> float:
        """Per-side maker cost as a return. Negative values are rebates."""
        return self.commission_bp / 1e4

    def summary(self) -> dict:
        return {
            "spread_bp": self.spread_bp,
            "commission_bp_per_side": self.commission_bp,
            "slippage_bp_per_side": self.slippage_bp,
            "round_trip_bp": round(self.round_trip_bp, 4),
            "round_trip_return": self.round_trip_return(),
        }


def infer_spread_bp(mid: float, spread_abs: float) -> float:
    return float(spread_abs / mid * 1e4)


def required_hit_rate(payoff_ratio: float, cost_return: float) -> float:
    """Win rate needed to break even given a win/loss payoff ratio and costs.

    payoff_ratio = average_win / average_loss (both gross, in return units).
    """
    return (1.0 + cost_return) / (1.0 + payoff_ratio)