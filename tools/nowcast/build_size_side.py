"""Side-file with RAW Kalshi displayed-size history keyed to the v10 rows:
midmove_v10_sz/<day>.sz.npz  with  szb/sza = log1p(top-of-book bid/ask size) at [now, 0.2, 0.4, 0.6, 1, 2, 5, 10 s]
(the v10 bl lag grid). The v10 per-ticker 200 ms grid is deterministic (arange(ts0+1000, tsN-5200, 200)), so rows are
matched by exact grid timestamp — no need to replicate the row-selection logic. NaN where a row's ts is off-grid.
Usage: build_size_side.py [DAY ...]  (default: every day in midmove_v10)"""
import glob, os, sys

import numpy as np
import pandas as pd

SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
OUT = os.path.join(SP, "midmove_v10_sz"); os.makedirs(OUT, exist_ok=True)
GRID = 200
LB = np.array([0, 1, 2, 3, 5, 10, 25, 50])   # ticks of 200 ms; 0 = now

days = sys.argv[1:] or [os.path.basename(p)[:10] for p in sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz")))]
for day in days:
    p = os.path.join(SP, "midmove_v10", day + ".npz")
    op = os.path.join(OUT, day + ".sz.npz")
    if os.path.exists(op):
        continue
    fn = "data/samples/%s.csv.gz" % day
    if not os.path.exists(fn):
        fn = "data/samples/%s.csv" % day
    if not os.path.exists(fn):
        print(day, "sampler MISSING", flush=True); continue
    z = np.load(p)
    names = [str(x) for x in z["names"]]
    tk, ts = z["tk"], z["ts"].astype(np.int64)
    df = pd.read_csv(fn, usecols=["ts_ms", "ticker", "ybid_sz", "yask_sz"], dtype=str, engine="c", on_bad_lines="skip")
    for c in ("ts_ms", "ybid_sz", "yask_sz"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["ts_ms"])]
    df["ts_ms"] = df["ts_ms"].astype(np.int64)
    n = len(ts)
    szb = np.full((n, len(LB)), np.nan, np.float32); sza = np.full((n, len(LB)), np.nan, np.float32)
    matched = 0
    for ti, name in enumerate(names):
        rows = np.where(tk == ti)[0]
        if len(rows) == 0:
            continue
        g = df[df.ticker == name].sort_values("ts_ms")
        if len(g) < 100:
            continue
        t_ = g["ts_ms"].to_numpy()
        grid = np.arange(t_[0] + 1000, t_[-1] - 5200, GRID)
        gi = np.searchsorted(t_, grid, side="right") - 1
        bs = np.log1p(np.maximum(g["ybid_sz"].to_numpy()[gi], 0.0)).astype(np.float32)
        as_ = np.log1p(np.maximum(g["yask_sz"].to_numpy()[gi], 0.0)).astype(np.float32)
        gi0 = (ts[rows] - grid[0]) // GRID
        ok = (gi0 >= LB.max()) & (gi0 < len(grid)) & ((ts[rows] - grid[0]) % GRID == 0)
        r_, g_ = rows[ok], gi0[ok].astype(np.int64)
        li = g_[:, None] - LB[None, :]
        szb[r_] = bs[li]; sza[r_] = as_[li]
        matched += len(r_)
    np.savez_compressed(op, szb=szb, sza=sza, lb=LB)
    print("%s rows %d matched %d (%.1f%%)" % (day, n, matched, 100 * matched / max(n, 1)), flush=True)
