"""Contending forecasters behind one interface.

Each contender predicts P(next-bar close is up). Diversity is the point: a
base-rate anchor, a gradient-boosted tree head on tabular features, and two
sequence models (LSTM and a small Transformer) over a rolling window.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class Contender:
    name = "base"
    needs_sequence = False

    def fit(self, X, y, Xtest=None):
        raise NotImplementedError

    def predict_proba(self, X, Xseq=None) -> np.ndarray:
        raise NotImplementedError


class BaseRate(Contender):
    """Predicts the training-period up-rate. The honesty check."""

    name = "base_rate"

    def fit(self, X, y, Xtest=None):
        self.p_ = float(np.clip(np.mean(y), 0.01, 0.99))
        return self

    def predict_proba(self, X, Xseq=None):
        return np.full(len(X), self.p_)


class XGBHead(Contender):
    name = "xgboost"

    def __init__(self, **kw):
        self.kw = dict(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            reg_lambda=1.0, n_jobs=4, eval_metric="logloss", tree_method="hist",
        )
        self.kw.update(kw)

    def fit(self, X, y, Xtest=None):
        import xgboost as xgb

        self.m_ = xgb.XGBClassifier(**self.kw)
        self.m_.fit(X, y, verbose=False)
        return self

    def predict_proba(self, X, Xseq=None):
        return self.m_.predict_proba(X)[:, 1]


def _seq_windows(X, window):
    """(n, window, n_feat) windows ending at each row; rows with too little history
    are back-filled with their earliest available window."""
    n, f = X.shape
    out = np.empty((n, window, f), dtype=np.float32)
    pad = np.repeat(X[:1], window - 1, axis=0) if n else np.zeros((0, window, f))
    Xp = np.vstack([pad, X])
    for i in range(n):
        out[i] = Xp[i : i + window]
    return out


class Persistence(Contender):
    """Naive: the next bar repeats the last bar's direction. The baseline deep
    models must beat to prove they found signal rather than noise."""

    name = "persistence"

    def __init__(self, sign_col: int = 0):
        self.sign_col = sign_col

    def fit(self, X, y, Xtest=None):
        return self

    def predict_proba(self, X, Xseq=None):
        return np.where(X[:, self.sign_col] > 0, 0.55, 0.45)


class MeanReversion(Contender):
    """Bets against the last move. Motivated by measured negative lag-1
    autocorrelation rather than by candle-pattern folklore."""

    name = "mean_rev"

    def __init__(self, sign_col: int = 0):
        self.sign_col = sign_col

    def fit(self, X, y, Xtest=None):
        # calibrate the conditional up-rate on the training set
        up = (X[:, self.sign_col] > 0)
        self.p_up_ = float(y[up].mean()) if up.any() else 0.5
        self.p_dn_ = float(y[~up].mean()) if (~up).any() else 0.5
        return self

    def predict_proba(self, X, Xseq=None):
        return np.where(X[:, self.sign_col] > 0, self.p_up_, self.p_dn_)


class _KerasSeq(Contender):
    needs_sequence = True
    window = 60

    def _build(self, n_feat):
        raise NotImplementedError

    def fit(self, X, y, Xtest=None):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(7)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        seq = _seq_windows(Xs, self.window)
        self.model_ = self._build(seq.shape[2])
        self.model_.fit(
            seq, y, epochs=6, batch_size=256, verbose=0,
            validation_split=0.1,
        )
        return self

    def predict_proba(self, X, Xseq=None):
        Xs = self.scaler_.transform(X).astype(np.float32)
        seq = _seq_windows(Xs, self.window)
        return self.model_.predict(seq, batch_size=1024, verbose=0).ravel()


class LSTMHead(_KerasSeq):
    name = "lstm"

    def _build(self, n_feat):
        import tensorflow as tf
        from tensorflow.keras import layers as L

        m = tf.keras.Sequential([
            L.Input((self.window, n_feat)),
            L.LSTM(32),
            L.Dropout(0.2),
            L.Dense(16, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ])
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy")
        return m


class TransformerHead(_KerasSeq):
    name = "transformer"

    def _build(self, n_feat):
        import tensorflow as tf
        from tensorflow.keras import layers as L

        d = 32
        inp = L.Input((self.window, n_feat))
        x = L.Dense(d)(inp)
        attn = L.MultiHeadAttention(num_heads=2, key_dim=d // 2)(x, x)
        x = L.LayerNormalization()(L.Add()([x, attn]))
        ff = L.Dense(d * 2, activation="relu")(x)
        ff = L.Dense(d)(ff)
        x = L.LayerNormalization()(L.Add()([x, ff]))
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dropout(0.2)(x)
        out = L.Dense(1, activation="sigmoid")(x)
        m = tf.keras.Model(inp, out)
        m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy")
        return m