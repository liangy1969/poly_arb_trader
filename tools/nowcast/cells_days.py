"""Per (model x delta): event-normalized net/trade BY DAY plus the pooled total with the event-clustered t,
from an analyze_online trades.csv. Usage: cells_days.py trades.csv [min_n]"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["model", "delta", "ticker", "date", "net", "side_yes"])
min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def clus(g):
    e = g.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t, len(e)


days = sorted(df["date"].unique())
for (m, dl), g in df.groupby(["model", "delta"]):
    ev, t, ne = clus(g)
    print("%-22s delta %.4f | ALL %+6.2fc (t %+5.2f) n=%5d ev=%3d yes=%2.0f%% | per day:" % (m, dl, ev, t, len(g), ne, 100 * g["side_yes"].mean()))
    for d in days:
        h = g[g["date"] == d]
        if len(h) >= min_n:
            e2, t2, n2 = clus(h)
            print("      %s %+6.2fc (t %+5.2f) n=%4d ev=%3d" % (d, e2, t2, len(h), n2))
        else:
            print("      %s   -- n=%d" % (d, len(h)))
