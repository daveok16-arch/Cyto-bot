"""The prediction bot: a calibrated probability plus a trade/no-trade decision.

Scope is deliberately narrow. The bot does one thing: given the current market
state, output P(next bar up), and only recommend a trade when that probability
clears the binary-option breakeven with margin. Given the measured signal is
~0.6pp, the honest default is that it usually declines to trade -- and the
decision layer is what enforces that rather than letting the model talk.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .aggregate import LogOddsPooler, softmax_weights
from .contenders import BaseRate, MeanReversion, XGBHead
from .scoring import IsotonicCalibrator, brier, logit, sigmoid


def breakeven(payout: float) -> float:
    """Win rate needed to break even on a binary paying `payout` on a win."""
    return 1.0 / (1.0 + payout)


@dataclass
class Decision:
    ts: str
    p_up: float
    breakeven: float
    edge: float
    action: str          # "UP" | "DOWN" | "NO_TRADE"
    reason: str
    payout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BotConfig:
    payout: float = 0.80
    min_edge: float = 0.005      # required margin above breakeven before acting
    bar: str = "5min"
    horizon: int = 1
    weights_mode: str = "brier"  # "brier" | "equal"
    calibrate: bool = True
    oos_frac: float = 0.3        # tail of training data held out for calibration


class PredictionBot:
    """Fit on historical bars, then produce a calibrated P(up) and a decision."""

    def __init__(self, config: BotConfig | None = None):
        self.cfg = config or BotConfig()
        self.names_: list[str] = []
        self.weights_: np.ndarray | None = None
        self.calibrator_: IsotonicCalibrator | None = None
        self.contenders_: list = []
        self.base_rate_: float = 0.5
        self.fitted_: bool = False
        self.metrics_: dict = {}

    # ---------------------------------------------------------------- training
    def fit(self, X: np.ndarray, y: np.ndarray, sign_col: int = 0) -> "PredictionBot":
        """Train contenders, learn pool weights, and fit calibration on a held-out
        tail. The tail is used ONLY for calibration, never for weighting."""
        n = len(y)
        if n < 500:
            raise ValueError(f"need >=500 rows to fit, got {n}")
        cut = int(n * (1 - self.cfg.oos_frac))
        Xtr, ytr = X[:cut], y[:cut]
        Xho, yho = X[cut:], y[cut:]

        self.contenders_ = [BaseRate(), MeanReversion(sign_col), XGBHead()]
        self.names_ = [c.name for c in self.contenders_]
        for c in self.contenders_:
            c.fit(Xtr, ytr)

        # holdout predictions -> per-contender skill -> weights
        Ph = np.column_stack([c.predict_proba(Xho) for c in self.contenders_])
        briers = [brier(yho, Ph[:, j]) for j in range(len(self.names_))]
        self.weights_ = (np.full(len(self.names_), 1.0 / len(self.names_))
                         if self.cfg.weights_mode == "equal"
                         else softmax_weights(briers))

        self.base_rate_ = float(np.clip(ytr.mean(), 0.01, 0.99))
        pooled_ho = sigmoid(logit(Ph) @ self.weights_)
        if self.cfg.calibrate:
            self.calibrator_ = IsotonicCalibrator().fit(pooled_ho, yho)
        else:
            self.calibrator_ = None

        self.metrics_ = {
            "n_train": int(cut), "n_holdout": int(n - cut),
            "up_rate_train": float(ytr.mean()),
            "contender_brier": dict(zip(self.names_, [float(b) for b in briers])),
            "weights": dict(zip(self.names_, self.weights_.round(4).tolist())),
        }
        self.fitted_ = True
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("bot is not fitted")
        P = np.column_stack([c.predict_proba(X) for c in self.contenders_])
        pooled = sigmoid(logit(P) @ self.weights_)
        if self.calibrator_ is not None:
            pooled = self.calibrator_.transform(pooled)
        return np.clip(pooled, 1e-6, 1 - 1e-6)

    def decide(self, p_up: float, ts: str = "") -> Decision:
        """Act only on a probability that clears breakeven plus margin."""
        be = breakeven(self.cfg.payout)
        thr = be + self.cfg.min_edge
        if p_up >= thr:
            action, edge = "UP", p_up - be
            reason = f"p={p_up:.4f} >= threshold {thr:.4f}"
        elif (1 - p_up) >= thr:
            action, edge = "DOWN", (1 - p_up) - be
            reason = f"p_down={1-p_up:.4f} >= threshold {thr:.4f}"
        else:
            action, edge = "NO_TRADE", max(p_up, 1 - p_up) - be
            reason = (f"best side {max(p_up,1-p_up):.4f} < threshold {thr:.4f}; "
                      f"edge {edge:+.4f} does not clear the vig")
        return Decision(ts=ts, p_up=float(p_up), breakeven=be, edge=float(edge),
                        action=action, reason=reason, payout=self.cfg.payout)

    def decide_many(self, X: np.ndarray, timestamps=None) -> list[Decision]:
        p = self.predict_proba(X)
        ts = ["" for _ in range(len(p))] if timestamps is None else list(timestamps)
        return [self.decide(float(pi), str(ti)) for pi, ti in zip(p, ts)]