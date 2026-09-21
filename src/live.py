"""Live signal generation from a real-time-accessible feed.

Design constraint: the cloud runner has no cached tick data and cannot be
trusted to have any state. So this module is fully self-contained -- it fetches
its own history, builds features, fits the model, and predicts, all in one run.

Critically, it trains and serves on the SAME source (Yahoo GC=F 1-minute bars).
The earlier studies trained on Dukascopy mid-prices and would have suffered a
domain mismatch if served from Yahoo futures bars. Using one source end to end
removes that class of bug.

Feature set is OHLCV-only. Live feeds do not provide bid/ask or aggressor volume,
so order-flow features (OFI, spread, price impact) are unavailable here by
construction. The price-only arm was measured in the 5m/15m study at ~49.9%
accuracy -- i.e. no directional edge. That is what this bot has to work with, and
it is why the honest output is almost always NO_TRADE.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
YAHOO = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
         "?range={range}&interval={interval}")
BP = 1e4


# ------------------------------------------------------------------ fetching
def fetch_ohlcv(symbol: str = "GC=F", rng: str = "5d",
                interval: str = "1m") -> pd.DataFrame:
    """Recent OHLCV bars. Raises on failure rather than returning stale data."""
    url = YAHOO.format(sym=urllib.parse.quote(symbol), range=rng, interval=interval)
    raw = urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=30).read()
    j = json.loads(raw)
    res = j["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q.get("volume"),
    }).dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("ts").reset_index(drop=True)


def resample(df: pd.DataFrame, bar: str = "5min") -> pd.DataFrame:
    d = df.set_index("ts")
    g = d.resample(bar)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna(subset=["open", "close"])
    return out.reset_index()


# ------------------------------------------------------------------ features
FEATURE_COLS = [
    "ret1", "ret2", "ret3", "ret5", "ret10", "ret20",
    "vol5", "vol10", "vol20", "vol60", "rv_ratio",
    "body", "close_pos", "range_pct", "zscore15", "zscore60",
    "tod_sin", "tod_cos", "dow",
]


def price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Causal OHLCV-only features. Identical code for training and live serving."""
    o, h, l, c = (bars[k] for k in ("open", "high", "low", "close"))
    ret = np.log(c / c.shift(1))
    f = pd.DataFrame(index=bars.index)

    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret{n}"] = np.log(c / c.shift(n))
    for n in (5, 10, 20, 60):
        f[f"vol{n}"] = ret.rolling(n).std()
    f["rv_ratio"] = f["vol5"] / f["vol60"].replace(0, np.nan)

    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["close_pos"] = (c - l) / rng
    f["range_pct"] = rng / c

    for n in (15, 60):
        ma = c.rolling(n).mean()
        sd = c.rolling(n).std().replace(0, np.nan)
        f[f"zscore{n}"] = (c - ma) / sd

    # time-of-day features: index may be a DatetimeIndex (training on an indexed
    # frame) or a plain RangeIndex with a 'ts' column (live path)
    if isinstance(bars.index, pd.DatetimeIndex):
        stamps = bars.index
    else:
        stamps = pd.DatetimeIndex(pd.to_datetime(bars["ts"]))
    mins = stamps.hour * 60 + stamps.minute
    f["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    f["dow"] = stamps.dayofweek
    return f


def make_labels(close: pd.Series, horizon: int = 1) -> pd.Series:
    """1 if close rises over the next `horizon` bars, else 0.

    The final `horizon` bars have no realized outcome and MUST stay NaN. Casting
    them to 0 (the obvious `.astype(int)` shortcut) silently fabricates a
    negative label for the most recent bar -- the exact bar being predicted.
    """
    fwd = close.shift(-horizon) / close - 1
    y = pd.Series(np.where(fwd.isna(), np.nan, (fwd > 0).astype(float)),
                  index=close.index)
    return y


# ------------------------------------------------------------------ training
def train_model(bars: pd.DataFrame, horizon: int = 1, oos_frac: float = 0.25):
    """Fit a pooled, calibrated direction model on recent bars.

    Same pooler as src/bot.py but restricted to features the live feed supports.
    Returns (model_bundle, metrics).
    """
    from .bot import BotConfig, PredictionBot

    f = price_features(bars)
    y = make_labels(bars["close"], horizon)
    X = f[FEATURE_COLS]
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]
    if len(X) < 500:
        raise RuntimeError(f"only {len(X)} usable bars; need >=500")
    Xv, yv = X.to_numpy(np.float32), y.to_numpy(int)
    bot = PredictionBot(BotConfig(
        payout=0.80, min_edge=0.005, bar="5min", calibrate=True,
        oos_frac=oos_frac)).fit(Xv, yv, sign_col=FEATURE_COLS.index("ret1"))

    # honest in-sample-vs-holdout read for the message
    from .scoring import accuracy, brier
    hold = int(len(X) * (1 - oos_frac))
    p_ho = bot.predict_proba(Xv[hold:])
    metrics = {
        "n_bars": int(len(X)),
        "n_train": int(hold),
        "n_holdout": int(len(X) - hold),
        "holdout_brier": float(brier(yv[hold:], p_ho)),
        "holdout_accuracy": float(accuracy(yv[hold:], p_ho)),
        "up_rate": float(yv.mean()),
        "weights": bot.metrics_["weights"],
        "contender_brier": bot.metrics_["contender_brier"],
    }
    return bot, metrics, X.index


def predict_latest(bot, bars: pd.DataFrame) -> dict:
    """Score the most recent complete bar."""
    f = price_features(bars)
    X = f[FEATURE_COLS]
    last = X.iloc[[-1]]
    if not np.isfinite(last.to_numpy(np.float32)).all():
        raise RuntimeError("latest bar has NaN features (insufficient history)")
    p_up = float(bot.predict_proba(last.to_numpy(np.float32))[0])
    return {"p_up": p_up, "ts": str(bars["ts"].iloc[-1]),
            "close": float(bars["close"].iloc[-1])}


# ------------------------------------------------------------------ pipeline
def run_signal(symbol: str = "GC=F", bar: str = "5min", horizon: int = 1,
               payout: float = 0.80, history_range: str = "5d",
               min_holdout_accuracy: float = 0.54) -> dict:
    """Full self-contained run: fetch -> resample -> train -> predict -> decide.

    `min_holdout_accuracy` is a skill gate, not a tuning knob. The price-only arm
    measured ~49.9% accuracy in the 5m/15m study, so a model scoring near 0.50 on
    its own holdout has no demonstrated skill and must not emit a directional
    call -- doing so would dress noise up as a signal. The gate is deliberately
    low (0.54) and is documented rather than hidden, but a model that clears it
    here is still not a validated edge.
    """
    raw = fetch_ohlcv(symbol, history_range, "1m")
    bars = resample(raw, bar)
    bot, metrics, _ = train_model(bars, horizon=horizon)
    pred = predict_latest(bot, bars)

    skill_ok = metrics["holdout_accuracy"] >= min_holdout_accuracy

    from .bot import BotConfig, PredictionBot
    decider = PredictionBot(BotConfig(payout=payout, min_edge=0.005))
    # reuse the fitted pooling/calibration without refitting
    decider.contenders_, decider.weights_ = bot.contenders_, bot.weights_
    decider.calibrator_, decider.fitted_ = bot.calibrator_, True
    d = decider.decide(pred["p_up"], pred["ts"])

    action, reason = d.action, d.reason
    if action != "NO_TRADE" and not skill_ok:
        action = "NO_TRADE"
        reason = (f"model holdout accuracy {metrics['holdout_accuracy']:.4f} is "
                  f"below the {min_holdout_accuracy:.2f} skill gate; the "
                  f"directional read is indistinguishable from noise")

    return {
        "symbol": symbol, "bar": bar, "as_of": pred["ts"],
        "last_close": pred["close"], "p_up": pred["p_up"],
        "decision": action, "breakeven": d.breakeven, "edge": d.edge,
        "reason": reason, "metrics": metrics,
        "skill_gate": {"threshold": min_holdout_accuracy, "passed": bool(skill_ok),
                       "holdout_accuracy": metrics["holdout_accuracy"]},
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }