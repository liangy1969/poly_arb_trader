"""Maker policy table from a maker_tape.py parquet (data/nowcast/tape_<day>.parquet): P&L per fill at 6/30/60 s under
naive / quiet-only / pull / 1c-tick-region policies, market-clustered t, plus a beta-neutral column (per market the
long-YES and short-YES fill means are averaged with equal weight, removing the day's drift).
Usage: maker_policy.py tape.parquet [tape2.parquet ...]   (several days are pooled; t is over markets)"""
import sys

import numpy as np
import pandas as pd

R = pd.concat([pd.read_parquet(p) for p in sys.argv[1:]], ignore_index=True)
W = R[(R.tte_s >= 60) & (R.tte_s <= 300) & R.at_touch & (R.spread_c > 0) & (R.spread_c <= 10)].copy()


def clus(h, col):
    e = h.groupby("ticker")[col].mean()
    return e.mean(), e.mean() / (e.std(ddof=1) / np.sqrt(len(e))), len(e)


def clus_sym(h, col):
    e = h.groupby(["ticker", "taker_yes"])[col].mean().unstack().dropna().mean(axis=1)
    return e.mean(), e.mean() / (e.std(ddof=1) / np.sqrt(len(e))), len(e)


one_c = (W.mid0 >= .1) & (W.mid0 <= .9)
pol = {"naive (all touch fills)": W, "quiet-only (|g|<=0.25c)": W[W.g.abs() <= 0.25], "pull policy (drop g<-0.25c)": W[~(W.g < -0.25)],
       "1c-tick region only (0.10-0.90)": W[one_c], "1c region + pull policy": W[one_c & ~(W.g < -0.25)], "1c region + quiet-only": W[one_c & (W.g.abs() <= 0.25)]}
print("markets %d | touch fills in window %d | flag share: pull %.1f%% quiet %.1f%% fav %.1f%% unflagged %.1f%%" % (
    W.ticker.nunique(), len(W), *(100 * x for x in ((W.g < -0.25).mean(), (W.g.abs() <= 0.25).mean(), (W.g > 0.25).mean(), W.g.isna().mean()))))
print("%-34s | %-26s | %-18s | %-18s | %s" % ("policy", "P&L/fill @6s  (t, mkts)", "@30s (t)", "@60s (t)", "beta-neutral @6s / @30s (t)"))
for k, h in pol.items():
    a6, t6, n = clus(h, "pnl6"); a30, t30, _ = clus(h, "pnl30"); a60, t60, _ = clus(h, "pnl60"); s6, st6, _ = clus_sym(h, "pnl6"); s30, st30, _ = clus_sym(h, "pnl30")
    print("%-34s | %+6.2fc (t %+5.2f, n=%3d) | %+6.2fc (t %+5.2f) | %+6.2fc (t %+5.2f) | %+6.2fc (t %+5.2f) / %+6.2fc (t %+5.2f)" % (k, a6, t6, n, a30, t30, a60, t60, s6, st6, s30, st30))
print("\nby nowcast flag (all touch fills): P&L @6s / @30s and share")
for lab, m in (("g < -0.25c (against)", W.g < -0.25), ("|g| <= 0.25c (quiet)", W.g.abs() <= 0.25), ("g > 0.25c (for)", W.g > 0.25)):
    h = W[m]
    print("  %-22s %5.1f%% | %+6.2fc / %+6.2fc" % (lab, 100 * len(h) / len(W), h.pnl6.mean(), h.pnl30.mean()))
