"""Entry-gap BAND filter from an analyze_online trades.csv: take the entries of one delta (default 0.01) and split them
by the gap at entry (|fair - mid| in cents at the moment the threshold was crossed), so 'enter at >= 1c but not if the
gap is already >= 2c' is the [1, 2) band. Event-normalized net/trade + event-clustered t on PRE / FRESH / OOS.
Usage: cells_band.py trades.csv [model_prefix] [delta] [min_n]"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["model", "delta", "ticker", "date", "gap", "net", "side_yes", "mid0"])
flt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
dl = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
min_n = int(sys.argv[4]) if len(sys.argv) > 4 else 30
if flt:
    df = df[df.model.str.startswith(flt)]
df = df[np.isclose(df.delta, dl)]
df["slice"] = np.where(df.date < "2026-09-01", "PRE", np.where(df.date < "2026-09-05", "FRESH", "OOS"))
g = 100 * df.gap.abs()
edges = [dl * 100, 1.5, 2.0, 3.0, 5.0, 100.0] if dl < 0.015 else [dl * 100, 3.0, 5.0, 100.0]
labels = ["[%g,%g)" % (edges[i], edges[i + 1]) if edges[i + 1] < 100 else ">=%g" % edges[i] for i in range(len(edges) - 1)]
df["band"] = pd.cut(g, edges, labels=labels, right=False, include_lowest=True)


def clus(h):
    e = h.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t, len(h), len(e)


def row(lab, h):
    cells = []
    for sl in ("PRE", "FRESH", "OOS"):
        hs = h[h["slice"] == sl]
        cells.append("%+6.2fc t%+5.2f n=%5d ev=%3d" % clus(hs) if len(hs) >= min_n else "    -- n=%d" % len(hs))
    print("  %-14s | %s | %s | %s | yes %2.0f%%" % (lab, cells[0], cells[1], cells[2], 100 * h.side_yes.mean() if len(h) else 0))


for m, gm in df.groupby("model"):
    print("\n%s  delta %.4f  (%d entries; gap at entry: median %.2fc p75 %.2fc p90 %.2fc)" % (m, dl, len(gm), np.median(100 * gm.gap.abs()), np.percentile(100 * gm.gap.abs(), 75), np.percentile(100 * gm.gap.abs(), 90)))
    print("  %-14s | %-30s | %-30s | %-30s |" % ("entry gap", "PRE", "FRESH", "OOS"))
    row("all", gm)
    for b in labels:
        h = gm[gm.band == b]
        if len(h) >= min_n:
            row(b, h)
    if dl < 0.015:
        row("band [1,2)", gm[(g.loc[gm.index] >= 1.0) & (g.loc[gm.index] < 2.0)])
        row("band [1,3)", gm[(g.loc[gm.index] >= 1.0) & (g.loc[gm.index] < 3.0)])
