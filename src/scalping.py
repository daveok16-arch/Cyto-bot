"""Scalping strategy: market making with adverse-selection control.

Diversification away from directional prediction. A market maker does not need to
forecast direction at all -- it earns the spread on two-sided quotes and loses
only when its resting order is filled right before the price moves against it
(adverse selection). The entire question is whether the spread it earns exceeds
the loss from being picked off.

This makes the economics honest and checkable:
  * revenue  = spread captured per filled quote
  * loss     = adverse price move after the fill
  * gate     = only quote when predicted adverse move < spread captured

The bot quotes at the touch (no queue-position advantage is assumed), and the
cost model charges the correct side: a maker EARNS the spread, it does not pay it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class QuoteDecision:
    ts: str
    action: str            # "QUOTE_BOTH" | "QUOTE_ONE" | "PULL"
    adverse_bp: float      # predicted adverse move, bp
    capture_bp: float      # spread captured if filled, bp
    edge_bp: float         # capture - adverse
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScalpConfig:
    min_edge_bp: float = 0.0      # require capture to exceed adverse by this much
    capture_frac: float = 0.5     # fraction of the spread actually captured
    maker_fee_bp: float = 0.0     # per side; many venues pay a rebate (negative)
    horizon: int = 1              # bars ahead to predict the adverse move


class MarketMaker:
    """Sizes expected adverse selection against the spread it would capture."""

    def __init__(self, cfg: ScalpConfig | None = None):
        self.cfg = cfg or ScalpConfig()

    def quote(self, mid: float, spread_abs: float, adverse_move_bp: float,
              ts: str = "") -> QuoteDecision:
        capture = (spread_abs / mid * 1e4) * self.cfg.capture_frac
        net_capture = capture - self.cfg.maker_fee_bp
        edge = net_capture - adverse_move_bp
        if edge >= self.cfg.min_edge_bp and adverse_move_bp < net_capture:
            action = "QUOTE_BOTH"
            reason = (f"capture {net_capture:.3f}bp > adverse {adverse_move_bp:.3f}bp "
                      f"(edge {edge:+.3f}bp)")
        else:
            action = "PULL"
            reason = (f"adverse {adverse_move_bp:.3f}bp >= capture {net_capture:.3f}bp; "
                      f"quoting would be picked off")
        return QuoteDecision(ts=ts, action=action, adverse_bp=float(adverse_move_bp),
                             capture_bp=float(net_capture), edge_bp=float(edge),
                             reason=reason)

    def quote_many(self, mid, spread_abs, adverse_bp, ts=None) -> list[QuoteDecision]:
        n = len(mid)
        ts = ["" for _ in range(n)] if ts is None else list(ts)
        return [self.quote(float(m), float(s), float(a), str(t))
                for m, s, a, t in zip(mid, spread_abs, adverse_bp, ts)]


def fill_and_pnl(prices: np.ndarray, sides: np.ndarray, fills: np.ndarray,
                 horizon: int = 1) -> np.ndarray:
    """Realized PnL per bar for a maker with fills at that bar's price.

    A filled buy earns (future price - fill price); a filled sell earns the
    reverse. `sides` is +1 for a buy fill, -1 for a sell fill.
    """
    n = len(prices)
    out = np.zeros(n)
    for i in range(n):
        if not fills[i]:
            continue
        j = min(i + horizon, n - 1)
        move = prices[j] - prices[i]
        out[i] = sides[i] * move
    return out