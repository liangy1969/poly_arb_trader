"""Slice summary of an analyze_online trades.csv for the exceedance-classifier runs: per model label (base = settlement
exit, .h<sec> = fixed-horizon taker exit) x delta (= |P(up)-P(down)| cut), event-normalized net/trade (mean of per-ticker
means, cents) with the event-clustered t, trades and events, on VAL 08-27..31 (selection) | TEST 09-01..04 | OOS 09-05..17.
Also the YES share and the mean entry price. Usage: exceed_slices.py trades.csv [min_trades]"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], usecols=["model", "delta", "ticker", "date", "net", "side_yes", "mid0", "cost", "exit_kind", "tte"])
min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
df["slice"] = np.where(df.date < "2026-09-01", "VAL", np.where(df.date < "2026-09-05", "TEST", "OOS"))


def clus(h):
    e = h.groupby("ticker")["net"].mean()
    t = e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0
    return 100 * e.mean(), t, len(h), len(e)


print("event-normalized net/trade (c) with event-clustered t | n trades / events;  %d trades total, days %s..%s" % (len(df), df.date.min(), df.date.max()))
print("%-26s %-6s | %-30s | %-30s | %-30s | %s" % ("model", "delta", "VAL 08-27..31", "TEST 09-01..04", "OOS 09-05..17", "yes% | px | settle-fallback%"))
for (m, dl), g in df.groupby(["model", "delta"]):
    cells = []
    for sl in ("VAL", "TEST", "OOS"):
        h = g[g["slice"] == sl]
        cells.append("%+6.2fc t%+5.2f %5d/%3d" % clus(h) if len(h) >= min_n else "    -- n=%d" % len(h))
    fb = 100 * (g.exit_kind == "settle").mean() if "h" in m.split(".")[-1] else float("nan")
    print("%-26s %-6g | %s | %s | %s | %3.0f%% | %.3f | %s" % (m, dl, cells[0], cells[1], cells[2], 100 * g.side_yes.mean(), g.cost.mean(), ("%.0f%%" % fb) if np.isfinite(fb) else "-"))
