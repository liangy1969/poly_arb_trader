"""Cross-asset side-file: ETH perp mid (from the lake ETHUSDT trade stream, last-price 200 ms bars with a 50 ms
causality lag, same convention as build_lake) at the v10 rows' own lag grid. Writes midmove_v10_xa/<day>.xa.npz with
eh = eth price at [now, LAGS...] (float32, NaN where no print within 60 s). Join is by GRID TIMESTAMP, market-free.
Usage: [V10_OUT=midmove_v10] build_xasset_side.py DAY [DAY ...]  (default: all v10 days)"""
import glob, os, sys

import numpy as np
import pandas as pd

SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
V10 = os.path.join(SP, os.environ.get("V10_OUT", "midmove_v10"))
OUT = V10 + "_xa"; os.makedirs(OUT, exist_ok=True)
BASE = r"E:\crypto\data\parquet"
SYM = os.environ.get("XA_SYMBOL", "ETHUSDT")
LAGS = np.array([0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30])   # ticks of 200 ms; 0 = now
LAG_MS = 50

days = sys.argv[1:] or [os.path.basename(p)[:10] for p in sorted(glob.glob(os.path.join(V10, "*.npz")))]
for day in days:
    op = os.path.join(OUT, day + ".xa.npz")
    if os.path.exists(op):
        continue
    p = os.path.join(V10, day + ".npz")
    if not os.path.exists(p):
        continue
    fs = sorted(glob.glob(os.path.join(BASE, "stream=trade", "venue=binance", "market=USDT_PERP", "symbol=" + SYM, "date=" + day, "*.parquet")))
    fs += sorted(glob.glob(os.path.join(BASE, "stream=trade", "venue=binance", "market=USDT_PERP", "symbol=" + SYM, "date=" + (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "*.parquet")))
    if not fs:
        print(day, "no lake trades"); continue
    P = pd.concat([pd.read_parquet(f, columns=["exch_ts_ns", "price"]) for f in fs]).sort_values("exch_ts_ns")
    pts = P.exch_ts_ns.to_numpy() // 1_000_000
    ppx = P.price.to_numpy()
    z = np.load(p)
    ts = z["ts"].astype(np.int64)
    out = np.full((len(ts), len(LAGS)), np.nan, np.float32)
    for li, L in enumerate(LAGS):
        q = ts - L * 200 - LAG_MS
        j = np.searchsorted(pts, q, side="right") - 1
        jc = np.clip(j, 0, len(pts) - 1)
        ok = (j >= 0) & ((q - pts[jc]) <= 60_000)
        out[ok, li] = ppx[jc][ok]
    np.savez_compressed(op, eh=out, lags=LAGS)
    print("%s: %d rows, cov %.1f%%" % (day, len(ts), 100 * np.isfinite(out).all(1).mean()), flush=True)
