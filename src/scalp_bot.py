"""Advanced scalping bot: direction + magnitude + cost gate.

The naive scalp bot predicts direction and trades every time it is confident.
That cannot work here, and the reason is arithmetic rather than model quality:

    the typical 1-bar move is ~1.59 bp
    the round-trip cost is    1.70 bp

A perfect direction oracle still loses money trading every bar. The only way a
scalp is profitable is to trade selectively -- when the expected move is large
enough to clear costs. So this bot predicts two things:

  1. direction  P(next move up)
  2. magnitude  E(|next move|) as a multiple of the cost floor

and only trades when (2) clears the cost with margin AND (1) is confident. The
gate is the edge; direction is nearly free by comparison.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .contenders import BaseRate, MeanReversion, XGBHead
from .costs import CostModel
from .scoring import IsotonicCalibrator, brier, logit, sigmoid


@dataclass
class ScalpConfig:
    cost: CostModel = None
    min_move_mult: float = 1.5      # required E[|move|] / cost floor
    min_dir_edge: float = 0.0       # required |p-0.5|
    horizon: int = 1
    oos_frac: float = 0.3

    def __post_init__(self):
        if self.cost is None:
            self.cost = CostModel()


@dataclass
class ScalpDecision:
    ts: str
    action: str            # UP | DOWN | NO_TRADE
    p_up: float
    exp_move_bp: float     # predicted |move|, bp
    cost_bp: float
    edge_bp: float         # exp_move - cost
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


class MagnitudeModel:
    """Predicts log |next move| in bp, i.e. how big the move is likely to be.

    Trained on the same features as direction but with a regression target, and
    used only to decide whether a scalp is worth taking.
    """

    name = "magnitude"

    def __init__(self):
        import xgboost as xgb

        self.m_ = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_lambda=1.0, n_jobs=4, tree_method="hist",
        )

    def fit(self, X, y_mag_bp):
        self.m_.fit(X, np.log(np.clip(y_mag_bp, 1e-4, None)), verbose=False)
        return self

    def predict_bp(self, X) -> np.ndarray:
        return np.exp(self.m_.predict(X))


class ScalpBot:
    """Direction + magnitude + cost gate."""

    def __init__(self, cfg: ScalpConfig | None = None):
        self.cfg = cfg or ScalpConfig()
        self.dir_contenders_: list = []
        self.names_: list[str] = []
        self.weights_: np.ndarray | None = None
        self.calibrator_: IsotonicCalibrator | None = None
        self.mag_ = MagnitudeModel()
        self.fitted_ = False
        self.metrics_: dict = {}

    def fit(self, X, y_dir, y_mag_bp, sign_col: int = 0):
        n = len(y_dir)
        cut = int(n * (1 - self.cfg.oos_frac))
        Xtr, ytr = X[:cut], y_dir[:cut]
        Xho, yho = X[cut:], y_dir[cut:]

        self.dir_contenders_ = [BaseRate(), MeanReversion(sign_col), XGBHead()]
        self.names_ = [c.name for c in self.dir_contenders_]
        for c in self.dir_contenders_:
            c.fit(Xtr, ytr)

        Ph = np.column_stack([c.predict_proba(Xho) for c in self.dir_contenders_])
        briers = [brier(yho, Ph[:, j]) for j in range(len(self.names_))]
        z = -(np.array(briers) - min(briers)) / 0.02
        w = np.exp(z)
        self.weights_ = w / w.sum()

        pooled = sigmoid(logit(Ph) @ self.weights_)
        self.calibrator_ = IsotonicCalibrator().fit(pooled, yho)

        self.mag_.fit(Xtr, y_mag_bp[:cut])
        self.metrics_ = {
            "n_train": int(cut), "n_holdout": int(n - cut),
            "weights": dict(zip(self.names_, self.weights_.round(4).tolist())),
            "contender_brier": dict(zip(self.names_, [round(float(b), 6) for b in briers])),
            "cost_floor_bp": self.cfg.cost.round_trip_bp,
        }
        self.fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        P = np.column_stack([c.predict_proba(X) for c in self.dir_contenders_])
        pooled = sigmoid(logit(P) @ self.weights_)
        return np.clip(self.calibrator_.transform(pooled), 1e-6, 1 - 1e-6)

    def expected_move_bp(self, X) -> np.ndarray:
        return self.mag_.predict_bp(X)

    def decide(self, p_up: float, exp_move_bp: float, ts: str = "") -> ScalpDecision:
        cost = self.cfg.cost.round_trip_bp
        need = cost * self.cfg.min_move_mult
        edge = exp_move_bp - cost
        dir_edge = abs(p_up - 0.5)

        if exp_move_bp < need:
            return ScalpDecision(ts, "NO_TRADE", p_up, exp_move_bp, cost, edge,
                                 f"E|move| {exp_move_bp:.2f}bp < required "
                                 f"{need:.2f}bp ({self.cfg.min_move_mult}x cost)")
        if dir_edge < self.cfg.min_dir_edge:
            return ScalpDecision(ts, "NO_TRADE", p_up, exp_move_bp, cost, edge,
                                 f"direction too weak |p-.5|={dir_edge:.3f}")
        action = "UP" if p_up > 0.5 else "DOWN"
        return ScalpDecision(ts, action, p_up, exp_move_bp, cost, edge,
                             f"E|move| {exp_move_bp:.2f}bp clears cost {cost:.2f}bp, "
                             f"p={p_up:.3f}")

    def decide_many(self, X, timestamps=None) -> list[ScalpDecision]:
        p = self.predict_proba(X)
        m = self.expected_move_bp(X)
        ts = ["" for _ in range(len(p))] if timestamps is None else list(timestamps)
        return [self.decide(float(pi), float(mi), str(ti))
                for pi, mi, ti in zip(p, m, ts)]