"""Aggregation of contenders into one calibrated probability.

Probabilities are combined in log-odds space. Weights and the final isotonic
calibration are always fit on *earlier* out-of-sample predictions than the block
they are applied to, so nothing peeks at the future.
"""
from __future__ import annotations

import numpy as np

from .scoring import IsotonicCalibrator, brier, logit, sigmoid


def softmax_weights(briers, tau: float = 0.02) -> np.ndarray:
    """More accurate contenders get exponentially more weight."""
    b = np.asarray(briers, dtype=float)
    z = -(b - b.min()) / tau
    w = np.exp(z)
    return w / w.sum()


def equal_weights(k: int) -> np.ndarray:
    return np.full(k, 1.0 / k)


class LogOddsPooler:
    """Block-wise rolling pool.

    For block t the weights come from the average Brier of each contender on
    blocks < t; the isotonic calibration likewise. Weighting mode controls
    whether we trust the pool equally or by recent accuracy.
    """

    def __init__(self, mode: str = "brier", calibrate: bool = True, tau: float = 0.02):
        assert mode in ("equal", "brier")
        self.mode = mode
        self.calibrate = calibrate
        self.tau = tau

    def run(self, blocks: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list]:
        """blocks: list of (P_block [n,k], y_block [n]) in chronological order."""
        preds = np.full(sum(len(y) for _, y in blocks), np.nan)
        weights_used = []
        pos = 0
        k = blocks[0][0].shape[1]

        for i, (P, y) in enumerate(blocks):
            if i == 0:
                w = equal_weights(k)
                cal = None
            else:
                prev_P = np.vstack([b[0] for b in blocks[:i]])
                prev_y = np.concatenate([b[1] for b in blocks[:i]])
                w = (equal_weights(k) if self.mode == "equal"
                     else softmax_weights([brier(prev_y, prev_P[:, j]) for j in range(k)],
                                          self.tau))
                cal = IsotonicCalibrator().fit(
                    sigmoid(logit(prev_P) @ w), prev_y) if self.calibrate else None

            pooled = sigmoid(logit(P) @ w)
            if cal is not None:
                pooled = cal.transform(pooled)
            preds[pos : pos + len(y)] = pooled
            weights_used.append(w.tolist())
            pos += len(y)

        return preds, weights_used