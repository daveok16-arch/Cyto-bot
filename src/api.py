"""HTTP API for the prediction bot.

One model in memory, two useful endpoints:
  GET  /health   -> liveness plus what the bot was fit on
  POST /predict  -> given recent bars, return P(up) and a trade decision

Run: python -m src.api       (then POST to http://127.0.0.1:8000/predict)
"""
from __future__ import annotations

import datetime as dt
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .bot import BotConfig, PredictionBot
from .feed import load_dataset

STATE: dict = {}


class Bar(BaseModel):
    ts: str = Field(..., description="ISO timestamp of the bar")
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    signed_volume: float = 0.0
    spread_mean: float = 0.0
    spread_max: float = 0.0
    n_ticks: float = 0.0
    bid_ticks: float = 0.0


class PredictRequest(BaseModel):
    bars: list[Bar] = Field(..., min_length=65,
                            description=">=65 recent bars, oldest first")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fit once at startup on cached history; serving then uses the same features
    X, y, ts = load_dataset(bar=STATE.get("bar", "5min"), order_flow=True)
    bot = PredictionBot(BotConfig(payout=STATE.get("payout", 0.80)))
    Xv, yv = X.to_numpy(np.float32), y.to_numpy(int)
    cut = int(len(y) * 0.6)
    bot.fit(Xv[:cut], yv[:cut], sign_col=0)
    STATE["bot"] = bot
    STATE["features"] = list(X.columns)
    STATE["sign_col"] = list(X.columns).index("ret1")
    STATE["fitted_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    STATE["n_train"] = cut
    yield
    STATE.clear()


app = FastAPI(title="XAUUSD Prediction Bot", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    bot: PredictionBot = STATE.get("bot")
    if bot is None:
        raise HTTPException(503, "bot not ready")
    return {
        "status": "ok",
        "fitted_at": STATE["fitted_at"],
        "n_train_bars": STATE["n_train"],
        "features": STATE["features"],
        "weights": bot.metrics_["weights"],
        "contender_brier": bot.metrics_["contender_brier"],
        "up_rate_train": bot.metrics_["up_rate_train"],
        "payout": bot.cfg.payout,
        "min_edge": bot.cfg.min_edge,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Score the last bar and return a decision.

    Uses the same `flow.order_flow_features` path as training, so the request
    only has to supply bars -- no feature logic is reimplemented here.
    """
    bot: PredictionBot = STATE.get("bot")
    if bot is None:
        raise HTTPException(503, "bot not ready")

    import pandas as pd
    from . import flow

    df = pd.DataFrame([b.model_dump() for b in req.bars])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.set_index("ts").sort_index()
    if len(df) < 65:
        raise HTTPException(400, "need at least 65 bars for feature warm-up")

    # derived fields the feature builder expects, so the request stays minimal
    df["abs_move"] = (df["close"] - df["open"]).abs()

    feat = flow.order_flow_features(df)
    missing = [c for c in STATE["features"] if c not in feat.columns]
    if missing:
        raise HTTPException(400, f"could not compute features: {missing}")

    X = feat[STATE["features"]].to_numpy(np.float32)
    last = X[-1:]
    if not np.isfinite(last).all():
        raise HTTPException(
            422, "latest bar has NaN features -- send more history for warm-up")

    p_up = float(bot.predict_proba(last)[0])
    d = bot.decide(p_up, str(df.index[-1]))
    return {
        "decision": d.action,
        "p_up": round(d.p_up, 6),
        "breakeven": round(d.breakeven, 6),
        "edge": round(d.edge, 6),
        "reason": d.reason,
        "ts": d.ts,
    }