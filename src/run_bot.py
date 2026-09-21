"""Run the prediction bot end to end: fit on history, replay forward, report.

This is the whole product in one command. It answers, plainly:
  * how often the bot was right when it chose to trade
  * what those trades would have returned at a real payout
  * how often it correctly declined to trade

Usage:
  python -m src.run_bot                  # 5min, 80% payout
  python -m src.run_bot 5min 0.80
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .bot import BotConfig, PredictionBot, breakeven
from .feed import MarketFeed, load_dataset
from .scoring import accuracy, brier, log_loss

RES = Path(__file__).resolve().parent.parent / "results"


def main(bar: str = "5min", payout: float = 0.80, order_flow: bool = True):
    X, y, ts = load_dataset(bar=bar, order_flow=order_flow, horizon=1)
    print(f"data: {len(y)} bars of {bar} "
          f"({ts.min()} -> {ts.max()}), up-rate={y.mean():.4f}, "
          f"features={X.shape[1]}")

    # single chronological split: fit on the past, trade the unseen future
    n = len(y)
    cut = int(n * 0.6)
    Xv, yv = X.to_numpy(np.float32), y.to_numpy(int)

    cfg = BotConfig(payout=payout, bar=bar, min_edge=0.005)
    bot = PredictionBot(cfg).fit(Xv[:cut], yv[:cut], sign_col=0)

    feed = MarketFeed(X.iloc[cut:], y.iloc[cut:])
    decisions = bot.decide_many(feed.X.to_numpy(np.float32), feed.ts)
    p_all = bot.predict_proba(feed.X.to_numpy(np.float32))
    y_test = yv[cut:]

    # ---- accuracy on everything the bot saw ----
    print(f"\nbot weights: {bot.metrics_['weights']}")
    print(f"fit: {bot.metrics_['n_train']} bars, calibration holdout "
          f"{bot.metrics_['n_holdout']} bars")
    print(f"out-of-sample: {len(y_test)} bars, up-rate={y_test.mean():.4f}")
    print(f"  brier={brier(y_test, p_all):.5f}  "
          f"logloss={log_loss(y_test, p_all):.5f}  "
          f"acc(p>=.5)={accuracy(y_test, p_all):.4f}")

    # ---- trading behaviour ----
    actions = np.array([d.action for d in decisions])
    traded = actions != "NO_TRADE"
    n_trade = int(traded.sum())
    be = breakeven(payout)

    print(f"\n-- decisions at payout {payout:.0%} (breakeven {be:.4f}, "
          f"min edge {cfg.min_edge:.3f}) --")
    print(f"traded {n_trade}/{len(decisions)} bars "
          f"({100*n_trade/len(decisions):.1f}%), declined "
          f"{len(decisions)-n_trade}")

    if n_trade:
        side_up = actions[traded] == "UP"
        wins = np.where(side_up, y_test[traded] == 1, y_test[traded] == 0)
        wr = wins.mean()
        realized_ev = wr * (1 + payout) - 1
        print(f"  win rate {wr:.4f} vs breakeven {be:.4f}  "
              f"({'ABOVE' if wr > be else 'BELOW'})")
        print(f"  realized EV per unit staked = {realized_ev:+.4f}")
        from .analyze import binom_p
        wins_n = int(wins.sum())
        print(f"  binomial p vs breakeven = {binom_p(wins_n, n_trade, be):.4f}")
    else:
        print("  the bot declined every bar")

    # ---- how often it was right to decline ----
    no = ~traded
    if no.any():
        print(f"\n-- declined bars --")
        print(f"  model accuracy on declined bars: "
              f"{accuracy(y_test[no], p_all[no]):.4f} "
              f"(a coin would be ~0.5000)")

    RES.mkdir(parents=True, exist_ok=True)
    stem = f"bot_{bar}_{int(payout*100)}"
    (RES / f"{stem}.json").write_text(json.dumps({
        "bar": bar, "payout": payout, "n_oos": int(len(y_test)),
        "n_traded": n_trade, "breakeven": be,
        "brier": brier(y_test, p_all),
        "metrics": bot.metrics_,
        "sample_decisions": [d.to_dict() for d in decisions[:5]],
    }, indent=2))
    print(f"\nwrote {RES / f'{stem}.json'}")


if __name__ == "__main__":
    bar = sys.argv[1] if len(sys.argv) > 1 else "5min"
    pay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.80
    main(bar, pay)