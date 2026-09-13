"""Extra future-mid targets for the midmove_v10 rows, from the 50 ms sampler: the mid on the 200 ms grid at
+3 s (mid15 = 15 ticks) and, as a consistency check against fut[:, 2, 0], at +5 s (mid25). A grid mid at time T is
the last sampler row at/before T (as build_v10 does), valid if that row is <= 400 ms old and the book is sane.
Writes <SP>/midmove_v10_fut/<day>.fut.npz : mid3s, mid5s (float32, NaN when invalid).
Usage: build_fut.py DAY [DAY ...]"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
V10 = os.path.join(SP, "midmove_v10")
OUT = os.path.join(SP, "midmove_v10_fut")
HOR = {"mid3s": 3000, "mid5s": 5000}


def build_day(day):
    z = np.load(os.path.join(V10, day + ".npz"))
    ts = z["ts"].astype(np.int64); tk = z["tk"]; names = [str(x) for x in z["names"]]
    fn = glob.glob(f"E:/poly/crypto_trader/data/samples/{day}.csv*")[0]
    S = pd.read_csv(fn, usecols=["ts_ms", "ticker", "ybid", "yask"], dtype=str, on_bad_lines="skip")
    for c in ("ts_ms", "ybid", "yask"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S = S.dropna().sort_values(["ticker", "ts_ms"])
    out = {k: np.full(len(ts), np.nan, np.float32) for k in HOR}
    for t_idx, name in enumerate(names):
        rows = np.where(tk == t_idx)[0]
        if len(rows) == 0:
            continue
        B = S[S.ticker == name]
        if len(B) == 0:
            continue
        bts = B.ts_ms.to_numpy(); yb = B.ybid.to_numpy(); ya = B.yask.to_numpy()
        sane = (yb > 0) & (ya > yb) & (ya <= 1.0) & (ya - yb <= 0.10)
        for k, h in HOR.items():
            j = np.searchsorted(bts, ts[rows] + h, side="right") - 1
            ok = (j >= 0)
            jj = np.clip(j, 0, len(bts) - 1)
            ok &= (ts[rows] + h - bts[jj] <= 400) & sane[jj]
            v = np.where(ok, 0.5 * (yb[jj] + ya[jj]), np.nan)
            out[k][rows] = v
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, day + ".fut.npz"), **out)
    fut5 = z["fut"].reshape(len(ts), 4, 3)[:, 2, 0]
    both = np.isfinite(out["mid5s"]) & np.isfinite(fut5)
    print("%s: rows %d | mid3s valid %.1f%% | mid5s valid %.1f%% | 5s check vs fut: max|d| %.4f mean|d| %.5f on %d rows" % (
        day, len(ts), 100 * np.isfinite(out["mid3s"]).mean(), 100 * np.isfinite(out["mid5s"]).mean(),
        np.abs(out["mid5s"][both] - fut5[both]).max() if both.any() else np.nan, np.abs(out["mid5s"][both] - fut5[both]).mean() if both.any() else np.nan, both.sum()), flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        try:
            build_day(d)
        except Exception as e:  # noqa: BLE001
            print("%s: FAILED %s" % (d, e), flush=True)
