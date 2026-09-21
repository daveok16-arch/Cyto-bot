"""Evaluate the scalping bot honestly, across horizons and gates.

Reports, for each configuration: how often it traded, the realized hit rate, the
gross and net returns, and whether it cleared costs. Uses the true realized move
to compute PnL -- no assumed fills beyond the cost model.

Usage:
  python -m src.run_scalp 5min
  python -m src.run_scalp 1min
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import flow
from .backtest import backtest_taker
from .costs import CostModel
from .feed import load_bars, load_dataset
from .scalp_bot import ScalpBot, ScalpConfig

RES = Path(__file__).resolve().parent.parent / "results"


def realized_move_bp(close: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed and absolute forward move in bp for each bar.

    Returns NaN for the final `horizon` bars, which have no realized outcome.
    Callers must mask those out; treating them as zero would invent a perfect
    prediction of 'no move'.
    """
    n = len(close)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = close[horizon:] / close[: n - horizon] - 1.0
    return fwd * 1e4, np.abs(fwd) * 1e4


def main(bar: str = "5min", horizon: int = 1):
    X, y, ts = load_dataset(bar=bar, order_flow=True, horizon=horizon)
    bars = load_bars(bar).set_index("ts")
    # load_dataset filters rows (warm-up NaNs), so align the close series to the
    # surviving index rather than assuming positional correspondence
    close = bars.loc[X.index, "close"].to_numpy(float)

    signed_bp, abs_bp = realized_move_bp(close, horizon)
    # target for the magnitude model must be forward-looking but only used as a
    # training label on past data, exactly like the direction label
    mag_target = np.nan_to_num(abs_bp, nan=np.nanmedian(abs_bp))

    cost = CostModel()
    n = len(y)
    cut = int(n * 0.6)

    print(f"=== {bar} horizon={horizon} ===")
    print(f"bars={n} up-rate={y.mean():.4f}")
    print(f"cost floor: {cost.round_trip_bp:.3f} bp round trip "
          f"(spread {cost.spread_bp} + fees/slip)")
    med_move = float(np.nanmedian(abs_bp[:cut]))
    print(f"median |move| in-sample: {med_move:.3f} bp "
          f"-> cost floor is {cost.round_trip_bp/med_move:.2f}x the median move")
    print(f"bars where |move| > cost: {np.nanmean(abs_bp[:cut] > cost.round_trip_bp)*100:.1f}%")

    bot = ScalpBot(ScalpConfig(cost=cost, horizon=horizon))
    Xv = X.to_numpy(np.float32)
    bot.fit(Xv[:cut], y.to_numpy(int)[:cut], mag_target[:cut],
            sign_col=list(X.columns).index("ret1"))

    print(f"\nweights: {bot.metrics_['weights']}")

    Xte = Xv[cut:]
    yte = y.to_numpy(int)[cut:]
    ts_te = np.array([str(t) for t in X.index[cut:]])
    move_te = signed_bp[cut:]
    abs_te = abs_bp[cut:]

    out = {}
    # sweep the magnitude gate -- this is the parameter that decides everything
    for mult in (1.0, 1.25, 1.5, 2.0, 3.0):
        for dir_edge in (0.0, 0.02, 0.05):
            cfg = ScalpConfig(cost=cost, min_move_mult=mult, min_dir_edge=dir_edge,
                              horizon=horizon)
            b = ScalpBot(cfg)
            b.dir_contenders_, b.names_, b.weights_ = (
                bot.dir_contenders_, bot.names_, bot.weights_)
            b.calibrator_, b.mag_, b.fitted_ = bot.calibrator_, bot.mag_, True

            dec = b.decide_many(Xte, ts_te)
            actions = np.array([d.action for d in dec])
            # drop bars with no realized outcome (the final `horizon` bars)
            valid = ~np.isnan(move_te)
            traded = (actions != "NO_TRADE") & valid
            k = int(traded.sum())

            key = f"mult{mult}_dir{dir_edge}"
            if k == 0:
                out[key] = {"n_trades": 0, "note": "declined every bar"}
                print(f"  mult={mult:<4} dirgate={dir_edge:<5} declined every bar")
                continue

            side = np.where(actions[traded] == "UP", 1, -1)
            gross = side * move_te[traded]            # bp
            net = gross - cost.round_trip_bp
            hit = float(np.mean(gross > 0))
            # t-stat on net per-trade PnL: is the mean distinguishable from zero?
            tstat = (float(net.mean() / net.std() * np.sqrt(len(net)))
                     if net.std() > 0 else 0.0)
            from scipy import stats
            pval = float(2 * (1 - stats.norm.cdf(abs(tstat))))
            total_net_bp = float(net.sum() * 1e-4)
            eq = np.cumprod(1 + net * 1e-4)
            out[key] = {
                "n_trades": k,
                "trade_pct": round(100 * k / valid.sum(), 2),
                "hit_rate": round(hit, 4),
                "avg_gross_bp": round(float(gross.mean()), 4),
                "avg_net_bp": round(float(net.mean()), 4),
                "t_stat": round(tstat, 3),
                "p_value": round(pval, 4),
                "total_net_return": round(total_net_bp, 4),
                "sharpe_per_trade": round(float(net.mean() / net.std()), 4)
                if net.std() > 0 else 0.0,
                "max_drawdown": round(float(((eq - np.maximum.accumulate(eq))
                                             / np.maximum.accumulate(eq)).min()), 4),
            }
            print(f"  mult={mult:<4} dirgate={dir_edge:<5} "
                  f"n={k:>5} ({100*k/valid.sum():>5.1f}%) "
                  f"hit={hit:.4f} avg_net={net.mean():+.3f}bp "
                  f"t={tstat:+.2f} p={pval:.3f} total={total_net_bp:+.3f}")

    candidates = [v for v in out.values() if v.get("n_trades", 0) >= 30]
    best = max(candidates, key=lambda v: v.get("avg_net_bp", -1e9), default=None)
    print("\nbest by avg net per trade (>=30 trades):", best)
    if best and best["avg_net_bp"] > 0 and best["p_value"] < 0.05:
        print("-> positive AND significant. Verify on data outside this window "
              "before believing it.")
    elif best and best["avg_net_bp"] > 0:
        print(f"-> positive but NOT significant (p={best['p_value']}); "
              f"consistent with noise, not a demonstrated edge")
    else:
        print("-> does NOT clear costs at any gate setting")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"scalp_{bar}_h{horizon}.json").write_text(json.dumps(
        {"bar": bar, "horizon": horizon, "cost": cost.summary(),
         "median_move_bp": med_move, "results": out,
         "metrics": bot.metrics_}, indent=2))
    print(f"wrote {RES / f'scalp_{bar}_h{horizon}.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "5min",
         int(sys.argv[2]) if len(sys.argv) > 2 else 1)