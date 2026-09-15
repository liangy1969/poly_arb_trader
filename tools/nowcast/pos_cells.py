"""Grid summary + positive-cell census for an analyze_online gate-grid trades.csv.
Usage: pos_cells.py trades.csv [min_n]"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["model", "delta", "ticker", "date", "net", "side_yes"])
min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "2026-09-01"     # first slice = dates before SPLIT, second = from SPLIT on


def clus(g):
    e = g.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t, len(g), len(e), 100 * g["side_yes"].mean()


fmt = lambda c: "%+6.2fc (%+5.2f) %6d/%3d %2.0f%%" % c
rows = []
print("%-18s %-6s | %-36s | %-36s | %s" % ("cell", "delta", "PRE evnorm (t) n/ev yes%", "FRESH evnorm (t) n/ev yes%", "ALL evnorm (t) n/ev"))
for (m, dl), g in df.groupby(["model", "delta"]):
    pre, fr = g[g["date"] < SPLIT], g[g["date"] >= SPLIT]
    a = clus(pre) if len(pre) >= min_n else None
    b = clus(fr) if len(fr) >= min_n else None
    c = clus(g)
    print("%-18s %-6.3f | %-36s | %-36s | %+6.2fc (%+5.2f) %6d/%3d" % (m, dl, fmt(a) if a else "-- n=%d" % len(pre), fmt(b) if b else "-- n=%d" % len(fr), c[0], c[1], c[2], c[3]))
    if a and b:
        rows.append((m, dl, a, b, c))
both = [r for r in rows if r[2][0] > 0 and r[3][0] > 0]
neg = [r for r in rows if r[2][0] < 0 and r[3][0] < 0]
print("\ncells with n>=%d on both slices: %d | positive both: %d | negative both: %d | |t|>2 both same sign: %d"
      % (min_n, len(rows), len(both), len(neg), sum(1 for r in rows if abs(r[2][1]) > 2 and abs(r[3][1]) > 2 and r[2][0] * r[3][0] > 0)))
print("== positive on BOTH slices ==")
for m, dl, a, b, c in sorted(both, key=lambda r: -r[4][1]):
    print("%-18s %-6.3f | %s | %s | ALL %+6.2fc (t %+5.2f) n=%d ev=%d" % (m, dl, fmt(a), fmt(b), c[0], c[1], c[2], c[3]))
