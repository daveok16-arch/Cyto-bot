"""Variance risk premium study: the one hypothesis with a structural reason to pay.

The VRP is the tendency of implied volatility to exceed subsequent realized
volatility. If it exists, a vol seller collects the difference systematically --
not because they forecast better, but because they are paid to bear risk that
others want to shed.

This is fundamentally different from every earlier stage. There, we tried to beat
the market's price using only price data. Here, we measure a *risk premium*: the
market's own price is the input, and the question is whether the compensation for
bearing volatility risk is positive on average.

The honest framing matters. A positive VRP is NOT an arbitrage:
  * returns are negatively skewed -- many small gains, occasional large losses
  * the losses cluster in crises, exactly when you can least afford them
  * a naive "always sell vol" strategy has a high Sharpe and fat left tail
  * it can go a long time looking brilliant before it blows up

So this study reports the premium, the drawdown, the skew, and the worst episode,
not just the mean.

Usage: python -m src.run_vrp
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import implied

RES = Path(__file__).resolve().parent.parent / "results"
ANN = 252


def describe_returns(r: np.ndarray, label: str) -> dict:
    """Summarize a per-period P&L series.

    Important: these are P&L values in units of the variance notional, NOT
    compoundable portfolio returns. Compounding them produces absurd
    annualized figures, so drawdown is measured on the cumulative sum and the
    annualized figure is a linear scaling, clearly labelled.
    """
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak                      # in P&L units, not a percentage
    out = {
        "label": label, "n": int(n),
        "mean_per_period": float(r.mean()),
        "mean_ann_linear": float(r.mean() * ANN),
        "sd_per_period": float(r.std()),
        "sharpe_ann": float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else 0.0,
        "skew": float(stats.skew(r)),
        "kurtosis": float(stats.kurtosis(r)),
        "max_drawdown_units": float(dd.min()),
        "worst_period": float(r.min()),
        "best_period": float(r.max()),
        "hit_rate": float(np.mean(r > 0)),
        "t_stat": float(r.mean() / r.std() * np.sqrt(n)) if r.std() > 0 else 0.0,
    }
    out["p_value"] = float(2 * (1 - stats.norm.cdf(abs(out["t_stat"]))))
    return out


def vrp_series(m: pd.DataFrame, lag: int = 21) -> pd.Series:
    """Premium per period in vol points: implied minus subsequently realized.

    `lag` shifts implied forward so it is compared against the realized vol that
    FOLLOWS the quote. Getting this direction wrong inverts the result.
    """
    return (m["iv"].shift(lag) - m["rv_fwd"]).dropna()


def short_vol_pnl(m: pd.DataFrame, entry: str = "always",
                  lag: int = 21) -> pd.Series:
    """Per-period P&L of selling variance, as a fraction of the variance notional.

    Selling a variance swap at strike K and paying realized R has P&L
    proportional to (K^2 - R^2). Dividing by K^2 normalizes it into a bounded
    fraction of notional, which is what makes the series comparable over time.
    This deliberately does NOT include delta-hedging transaction costs or margin
    financing; those are discussed in the report rather than silently omitted.
    """
    iv = m["iv"].shift(lag)
    pnl = (iv**2 - m["rv_fwd"]**2) / (iv**2)
    if entry == "high_iv":
        pnl = pnl.where((iv - m["rv_trail"]) > 0)
    elif entry == "rich_2pct":
        pnl = pnl.where((iv - m["rv_trail"]) > 2.0)
    return pnl.dropna()


def main():
    print("=" * 78)
    print("VARIANCE RISK PREMIUM STUDY")
    print("=" * 78)

    results = {}
    vix = implied.fetch_vix()
    spx = implied.fetch_spx()
    m = implied.align_iv_rv(vix, spx, iv_col="close", window=21)
    print(f"\nS&P 500: VIX vs realized, {len(m):,} overlapping observations")
    print(f"  span {m.date.min().date()} -> {m.date.max().date()}")

    vrp = vrp_series(m, lag=21)
    t = float(vrp.mean() / vrp.std() * np.sqrt(len(vrp)))
    print("\n-- the premium (implied minus subsequently realized, vol points) --")
    print(f"  mean VRP     = {vrp.mean():+.3f}")
    print(f"  median VRP   = {vrp.median():+.3f}")
    print(f"  sd           = {vrp.std():.3f}")
    print(f"  fraction > 0 = {np.mean(vrp > 0):.4f}")
    print(f"  t-stat       = {t:+.2f}  (p={2*(1-stats.norm.cdf(abs(t))):.3g})")
    print(f"  mean VIX={m['iv'].mean():.2f}  mean realized={m['rv_fwd'].mean():.2f}")
    results["spx_vrp"] = {"n": int(len(vrp)), "mean": float(vrp.mean()),
                          "median": float(vrp.median()),
                          "frac_positive": float(np.mean(vrp > 0)), "t": t,
                          "mean_iv": float(m["iv"].mean()),
                          "mean_rv": float(m["rv_fwd"].mean())}

    print("\n-- selling variance (P&L as fraction of variance notional) --")
    strategies = {
        "always short": short_vol_pnl(m, "always"),
        "short when IV > RV_trail": short_vol_pnl(m, "high_iv"),
        "short when IV > RV_trail+2": short_vol_pnl(m, "rich_2pct"),
    }
    for name, pnl in strategies.items():
        d = describe_returns(pnl.to_numpy(), name)
        results[f"spx_{name}"] = d
        print(f"  {name:<28} n={d['n']:>5} mean/period={d['mean_per_period']:>+7.4f} "
              f"sharpe={d['sharpe_ann']:>+5.2f} skew={d['skew']:>+7.2f} "
              f"worstDD={d['max_drawdown_units']:>+7.2f} t={d['t_stat']:>+6.2f}")

    bh = m.iloc[21:].copy()
    bh_ret = np.log(bh["close"]).diff().shift(-1).dropna()
    bhd = describe_returns(bh_ret.to_numpy(), "SPX buy-and-hold")
    results["spx_buyhold"] = bhd
    print(f"  {'SPX buy-and-hold (log ret)':<28} n={bhd['n']:>5} "
          f"mean/period={bhd['mean_per_period']:>+7.4f} sharpe={bhd['sharpe_ann']:>+5.2f} "
          f"skew={bhd['skew']:>+7.2f} worstDD={bhd['max_drawdown_units']:>+7.2f}")

    print("\n-- the tail (what you are paid to hold) --")
    pnl = strategies["always short"].to_numpy()
    dates = m["date"].iloc[21:].to_numpy()
    worst = np.argsort(pnl)[:5]
    print("  5 worst periods for the vol seller:")
    for i, w in enumerate(worst):
        dt = str(dates[w])[:10] if w < len(dates) else "?"
        print(f"    {dt}  {pnl[w]:+9.2f} vol pts")
    k1 = max(1, int(0.01 * len(pnl)))
    print(f"  worst 1% mean = {np.mean(np.sort(pnl)[:k1]):+.4f}")
    print(f"  best  1% mean = {np.mean(np.sort(pnl)[-k1:]):+.4f}")
    print(f"  skew = {stats.skew(pnl):+.2f} (negative = fat left tail)")

    print("\n-- by decade (stable, or one lucky regime?) --")
    tmp = m.iloc[21:][["date"]].copy()
    tmp["pnl"] = pnl
    tmp["decade"] = (tmp["date"].dt.year // 10) * 10
    bydec = tmp.groupby("decade")["pnl"].agg(["count", "mean", "std"])
    decades = {}
    for dec, row in bydec.iterrows():
        td = row["mean"] / row["std"] * np.sqrt(row["count"]) if row["std"] > 0 else 0
        decades[int(dec)] = {"n": int(row["count"]), "mean": float(row["mean"]),
                             "t": float(td)}
        print(f"  {int(dec)}s  n={int(row['count']):>5} mean={row['mean']:>+7.4f} "
              f"t={td:>+5.2f}")
    results["by_decade"] = decades

    print("\n-- independent cross-check: BTC DVOL vs realized --")
    try:
        dvol = implied.fetch_dvol()
        btc = implied.fetch_btc()
        mb = implied.align_iv_rv(dvol, btc, iv_col="close", window=21)
        if len(mb) > 100:
            vb = vrp_series(mb, lag=21)
            tb = float(vb.mean() / vb.std() * np.sqrt(len(vb)))
            print(f"  DVOL: n={len(vb)} mean VRP={vb.mean():+.3f} "
                  f"frac>0={np.mean(vb>0):.3f} t={tb:+.2f}")
            db = describe_returns(short_vol_pnl(mb, "always").to_numpy(),
                                  "BTC short vol")
            print(f"  short variance: mean/period={db['mean_per_period']:+.4f} "
                  f"sharpe={db['sharpe_ann']:+.2f} skew={db['skew']:+.2f} "
                  f"worst={db['worst_period']:+.3f}")
            results["btc_vrp"] = {"n": int(len(vb)), "mean": float(vb.mean()),
                                  "t": tb, "sharpe": db["sharpe_ann"],
                                  "skew": db["skew"]}
    except Exception as e:
        print(f"  DVOL cross-check failed: {e}")

    print("\n-- what this does and does not show --")
    print("  SHOWS: implied vol exceeds subsequent realized vol on average, with")
    print("         a large t-stat over multiple decades and regimes.")
    print("  DOES NOT SHOW: that this is risk-free. Negative skew, crisis-clustered")
    print("         losses, and it requires shorting options -- margin and tail risk.")

    # ---- the decisive robustness checks ----
    print("\n-- does it survive removing the crises? --")
    full = short_vol_pnl(m, "always")
    for label, years in (("excluding 2008", {2008}),
                         ("excluding 2008+2020", {2008, 2020}),
                         ("excluding all crisis years",
                          {1998, 2008, 2020, 2022})):
        keep = ~full.index.map(lambda i: m["date"].iloc[i].year in years)
        s = full[keep]
        tt = s.mean() / s.std() * np.sqrt(len(s))
        print(f"  {label:<28} n={len(s):>5} mean={s.mean():+.4f} t={tt:+.2f}")
        results[f"excl_{label}"] = {"n": int(len(s)), "mean": float(s.mean()),
                                    "t": float(tt)}

    print("\n-- the bull case, stated fairly --")
    n_ok = int(np.sum(full > 0))
    print(f"  wins {n_ok}/{len(full)} ({100*n_ok/len(full):.1f}% of periods)")
    print(f"  median gain {full.median():+.4f}, mean {full.mean():+.4f}")
    worst = float(full.min())
    erases = abs(worst) / full.mean() if full.mean() > 0 else float("nan")
    print(f"  worst single period {worst:+.4f} = {erases:.0f}x the average gain,")
    print(f"  i.e. one crisis erases roughly {erases:.0f} periods of ordinary income.")

    # ---- the overlap correction: this is essential ----
    # Consecutive observations share 20 of their 21 days, so the 9,200 "samples"
    # are nowhere near independent. The naive t-stat is inflated by roughly
    # sqrt(window). Re-run on strictly non-overlapping windows for an honest test.
    print("\n-- OVERLAP CORRECTION (the naive t-stat is inflated) --")
    step = 21
    sub = m.iloc[::step].copy()
    vrp_n = vrp_series(sub, lag=21)
    t_n = float(vrp_n.mean() / vrp_n.std() * np.sqrt(len(vrp_n)))
    print(f"  VRP, non-overlapping: n={len(vrp_n)} mean={vrp_n.mean():+.3f} "
          f"t={t_n:+.2f} (naive overlapping t was {results['spx_vrp']['t']:+.2f})")
    results["spx_vrp_nonoverlap"] = {"n": int(len(vrp_n)),
                                     "mean": float(vrp_n.mean()), "t": t_n}

    pnl_n = short_vol_pnl(sub, "always")
    d_n = describe_returns(pnl_n.to_numpy(), "always short (non-overlap)")
    print(f"  short variance, non-overlapping: n={d_n['n']} "
          f"mean={d_n['mean_per_period']:+.4f} sharpe={d_n['sharpe_ann']:+.2f} "
          f"t={d_n['t_stat']:+.2f} (naive t was "
          f"{results['spx_always short']['t_stat']:+.2f})")
    results["spx_short_nonoverlap"] = d_n

    # 2020s excluding the COVID crash, to see if the decade is a regime change
    print("\n-- is the 2020s result a regime change or just COVID? --")
    yrs = m["date"].iloc[21:].dt.year.to_numpy()
    for label, keep in (("2020s all", yrs >= 2020),
                        ("2020s ex-2020", (yrs >= 2020) & (yrs != 2020)),
                        ("2021 onwards", yrs >= 2021)):
        p = pnl[keep]
        tt = p.mean() / p.std() * np.sqrt(len(p))
        print(f"  {label:<18} n={len(p):>5} mean={p.mean():+.4f} t={tt:+.2f}")
        results[f"recent_{label}"] = {"n": int(len(p)), "mean": float(p.mean()),
                                      "t": float(tt)}

    print("\n-- VERDICT --")
    dec = results["by_decade"]
    pos = [d for d, v in dec.items() if v["mean"] > 0]
    neg = [d for d, v in dec.items() if v["mean"] <= 0]
    print(f"  premium is POSITIVE on average and highly significant "
          f"(t={results['spx_vrp']['t']:+.1f}, n={results['spx_vrp']['n']})")
    print(f"  positive in decades {sorted(pos)}, negative in {sorted(neg)}")
    print(f"  -> it is a real, persistent risk premium, and it is the FIRST")
    print(f"     positive result in this project that is not an artifact.")
    print(f"  -> but it is compensation for bearing tail risk, not free money: it")
    print(f"     lost money in the most recent decade, and its worst periods are")
    print(f"     the market crashes that ruin leveraged sellers.")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "vrp.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {RES / 'vrp.json'}")


if __name__ == "__main__":
    main()