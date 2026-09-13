"""Per (model x delta): event-normalized net/trade (event-clustered t) by ENTRY PRICE bucket and by ENTRY TTE bucket,
on the three slices PRE 08-27..31 | FRESH 09-01..04 | OOS 09-05..08, plus the market's closing move (close_dmid,
cents, signed toward the trade) per bucket. From an analyze_online trades.csv with --close-eps.
Usage: cells_buckets.py trades.csv [model_filter] [min_n]"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["model", "delta", "ticker", "date", "tte", "mid0", "net", "side_yes", "close_dmid", "close_dfair"])
flt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
min_n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
if flt:
    df = df[df.model.str.startswith(flt)]
df["slice"] = np.where(df.date < "2026-09-01", "PRE", np.where(df.date < "2026-09-05", "FRESH", "OOS"))
sgn = np.where(df.side_yes.astype(bool), 1.0, -1.0)
df["mkt_move"] = 100 * sgn * df.close_dmid          # market move at reconvergence, signed toward the trade (as closure_cells2)
df["pb"] = pd.cut(df.mid0, [0, 0.10, 0.30, 0.70, 0.90, 1.0], labels=["<0.10", "0.10-0.30", "0.30-0.70", "0.70-0.90", ">0.90"], include_lowest=True)
df["tb"] = pd.cut(df.tte, [0, 60, 120, 180, 240, 300, 1e9], labels=["<60", "60-120", "120-180", "180-240", "240-300", ">300"])


def clus(g):
    e = g.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t, len(g), len(e)


def table(g, col, title):
    print("  %s" % title)
    print("  %-11s | %-30s | %-30s | %-30s | mkt close move (c) PRE/FRESH/OOS" % ("bucket", "PRE", "FRESH", "OOS"))
    for b in g[col].cat.categories:
        h = g[g[col] == b]
        if len(h) < min_n:
            continue
        cells = []; moves = []
        for sl in ("PRE", "FRESH", "OOS"):
            hs = h[h["slice"] == sl]
            if len(hs) >= min_n:
                ev, t, n, ne = clus(hs); cells.append("%+6.2fc t%+5.2f n=%5d ev=%3d" % (ev, t, n, ne)); moves.append("%+5.2f" % hs.mkt_move.mean())
            else:
                cells.append("    -- n=%d" % len(hs)); moves.append("   --")
        print("  %-11s | %s | %s | %s | %s" % (b, cells[0], cells[1], cells[2], " / ".join(moves)))


for (m, dl), g in df.groupby(["model", "delta"]):
    print("\n%s  delta %.4f  (trades %d, yes %.0f%%)" % (m, dl, len(g), 100 * g.side_yes.mean()))
    table(g, "pb", "by ENTRY PRICE (YES mid at entry)")
    table(g, "tb", "by ENTRY TTE (s)")
