"""Cross-day aggregation of maker_queue_sim.py runs (written with --tag): per variant, per-fill P&L at 6/30 s and
settlement, market-clustered t over ALL markets across days (per-market means; zero-fill markets enter the
per-market settlement total as 0), plus a per-day table.
Usage: maker_days.py TAG1 [TAG2 ...] -- DAY1 [DAY2 ...]     (reads data/nowcast/qsim_<day>_<tag>.parquet and _mkts)"""
import os
import sys

import numpy as np
import pandas as pd

args = sys.argv[1:]
sep = args.index("--")
tags, days = args[:sep], args[sep + 1:]


def clus(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.mean(), (v.mean() / (v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 3 and v.std() > 0 else 0.0), len(v)


print("%-22s | %-30s | %-22s | %-24s | %-30s | %s" % ("variant (days pooled)", "P&L/fill @6s (t, mkts w/ fills)", "@30s (t)", "settle/contract (t)", "per-MARKET settle P&L (t, mkts)", "fills/mkt | sweep%"))
for tag in tags:
    F, M = [], []
    for d in days:
        pf, pm = f"data/nowcast/qsim_{d}_{tag}.parquet", f"data/nowcast/qsim_{d}_{tag}_mkts.parquet"
        if os.path.exists(pf):
            f = pd.read_parquet(pf); f["day"] = d; F.append(f)
        if os.path.exists(pm):
            m = pd.read_parquet(pm); m["day"] = d; M.append(m)
    if not F:
        print("%-22s | no data" % tag); continue
    F = pd.concat(F); M = pd.concat(M) if M else None
    g = F.groupby(["day", "ticker"])
    a6, t6, n6 = clus(g["pnl6"].mean()); a30, t30, _ = clus(g["pnl30"].mean()); aS, tS, _ = clus(g["pnl_settle"].mean())
    if M is not None:
        pm, tpm, npm = clus(M["pnl_settle_tot"]); fpm = M["n"].mean()
    else:
        pm, tpm, npm, fpm = float("nan"), 0.0, 0, float("nan")
    sweep = 100 * F.kind.str.contains("sweep").mean()
    print("%-22s | %+6.2fc (t %+5.2f, n=%3d) | %+6.2fc (t %+5.2f) | %+6.2fc (t %+5.2f) | %+7.2fc (t %+5.2f, n=%3d) | %5.1f | %4.0f%%" % (tag, a6, t6, n6, a30, t30, aS, tS, pm, tpm, npm, fpm, sweep))
print("\nper day, P&L/fill @6s (market-clustered t) | per-market settle P&L:")
print("%-22s | " % "variant" + " | ".join("%-24s" % d[5:] for d in days))
for tag in tags:
    cells = []
    for d in days:
        pf, pm = f"data/nowcast/qsim_{d}_{tag}.parquet", f"data/nowcast/qsim_{d}_{tag}_mkts.parquet"
        if not os.path.exists(pf):
            cells.append("%-24s" % "--"); continue
        f = pd.read_parquet(pf); a6, t6, _ = clus(f.groupby("ticker")["pnl6"].mean())
        pmv = clus(pd.read_parquet(pm)["pnl_settle_tot"])[0] if os.path.exists(pm) else float("nan")
        cells.append("%+5.2fc t%+4.1f | %+6.1fc" % (a6, t6, pmv))
    print("%-22s | " % tag + " | ".join("%-24s" % c for c in cells))
