"""v10 = v9 (full history, futures, settlement, market ids) + book snapshot (mx),
lagged book (bl) and strike geometry (kz). One dataset for the outcome study.
Keys: mh ph sh (300 lags fp16), mx (11), bl (35), kz (6), ctx (6: mid0 tte
sigma spread bid ask), fut (12), tk, ts, settle, trig, names.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.special import ndtr

os.chdir("e:/poly/crypto_trader")
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
OUT = os.path.join(SP, os.environ.get("V10_OUT", "midmove_v10")); os.makedirs(OUT, exist_ok=True)
META_PATH = os.environ.get("V10_META", "data/samples/meta_cache.json")
SAMPLES_DIR = os.environ.get("V10_SAMPLES", "data/samples")
TKPREFIX = os.environ.get("V10_PREFIX", "KXBTC")
SIG_FLOOR = float(os.environ.get("V10_SIG_FLOOR", "1.25"))   # $ floor on the 1 s perp-move std (BTC 1.25; scale by price for other coins)
META = json.load(open(META_PATH))
GRID, MAXL = 200, 300
FUT = np.array([1, 5, 25, 150]); LB = np.array([1, 2, 3, 5, 10, 25, 50])
LAGS300 = np.arange(1, MAXL + 1)
COLS = ["ts_ms", "ticker", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz",
        "perp_bid", "perp_ask", "perp_bid_sz", "perp_ask_sz", "cb_bid", "cb_ask"]

for day in sys.argv[1:]:
    fn = "%s/%s.csv.gz" % (SAMPLES_DIR, day)
    if not os.path.exists(fn):
        print("skip %s" % day, flush=True); continue
    df = pd.read_csv(fn, usecols=COLS, dtype=str, engine="c", on_bad_lines="skip")
    for c in COLS:
        if c != "ticker":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[np.isfinite(df["ts_ms"])]
    df["ts_ms"] = df["ts_ms"].astype(np.int64)
    K_ = {k: [] for k in ("mh", "ph", "sh", "mx", "bl", "kz", "ctx", "fut", "tk", "ts", "settle", "trig")}
    names = []
    for tk, g in df.groupby("ticker"):
        if len(g) < 4000 or not str(tk).startswith(TKPREFIX):
            continue
        meta = META.get(str(tk)) or {}
        K = meta.get("strike")
        if K is None:
            continue
        K = float(K)
        g = g.sort_values("ts_ms"); ts = g["ts_ms"].to_numpy()
        grid = np.arange(ts[0] + 1000, ts[-1] - 5200, GRID)
        gi = np.searchsorted(ts, grid, side="right") - 1
        ok0 = (gi >= 0) & (grid - ts[np.clip(gi, 0, None)] < 400)
        col = lambda c: g[c].to_numpy()[gi]
        yb, ya, ybs, yas = col("ybid"), col("yask"), col("ybid_sz"), col("yask_sz")
        pb, pa, pbs, pas = col("perp_bid"), col("perp_ask"), col("perp_bid_sz"), col("perp_ask_sz")
        cb, ca = col("cb_bid"), col("cb_ask")
        tte = col("tte_ms") / 1000.0
        mid = 0.5 * (yb + ya); perp = 0.5 * (pb + pa); cbm = 0.5 * (cb + ca); spr = ya - yb
        sane = (yb > 0.01) & (ya < 0.99) & (ya > yb) & (spr <= 0.06) & np.isfinite(mid)
        ok0 &= sane & (pb > 0) & (cb > 0) & np.isfinite(perp) & np.isfinite(cbm)
        n = len(grid)
        if n < MAXL + 160 + 50:
            continue
        c1 = perp - np.roll(perp, 5)
        sig = pd.Series(c1).rolling(600, min_periods=200).std().to_numpy()
        sig = np.maximum(np.where(np.isfinite(sig), sig, SIG_FLOOR), SIG_FLOOR)
        dm = np.diff(mid, prepend=mid[0]); chg = dm != 0
        idx = np.arange(n)
        last_chg = np.maximum.accumulate(np.where(chg, idx, 0))
        tsl = np.minimum(idx - last_chg, 300).astype(np.float32)
        cs_ = np.cumsum(chg)
        n5 = cs_ - np.concatenate([np.zeros(25), cs_[:-25]])
        n30 = cs_ - np.concatenate([np.zeros(150), cs_[:-150]])
        lastdir = np.sign(dm[last_chg])
        ssum = np.maximum(ybs + yas, 1e-6); imb = (ybs - yas) / ssum
        micro = (yb * yas + ya * ybs) / ssum - mid
        psum = np.maximum(pbs + pas, 1e-6); pimb = (pbs - pas) / psum
        cs = np.concatenate([[0], np.cumsum(ok0)])
        run301 = cs[MAXL + 1:] - cs[:-(MAXL + 1)]
        okw = np.zeros(n, bool); okw[MAXL:] = run301 == MAXL + 1
        for h in (1, 5, 25):
            okw &= np.roll(ok0, -h)
        okw[:MAXL + 5] = False; okw[-160:] = False
        okw &= (tte > 15) & (tte < 880)
        motion = np.abs(c1) >= 1.0 * sig
        backg = (idx % 5) == 0
        sel = np.where(okw & (motion | backg))[0]
        if len(sel) == 0:
            continue
        li = sel[:, None] - LAGS300[None, :]; bi = sel[:, None] - LB[None, :]
        K_["mh"].append(mid[li].astype(np.float16))
        K_["ph"].append((perp[li] - perp[sel][:, None]).astype(np.float16))
        K_["sh"].append(spr[li].astype(np.float16))
        K_["mx"].append(np.column_stack([np.log1p(ybs[sel]), np.log1p(yas[sel]), imb[sel], micro[sel],
                                         np.log1p(pbs[sel]), np.log1p(pas[sel]), pimb[sel], tsl[sel],
                                         n5[sel], n30[sel], lastdir[sel]]).astype(np.float32))
        K_["bl"].append(np.concatenate([imb[bi], micro[bi], pimb[bi], spr[bi],
                                        (mid + micro)[bi] - mid[sel][:, None]], 1).astype(np.float32))
        sc = 150.0 * np.sqrt(np.maximum(tte[sel], 1.0) / 900.0)
        dp_, dc_ = perp[sel] - K, cbm[sel] - K
        K_["kz"].append(np.column_stack([dp_, dc_, dp_ / sc, dc_ / sc, ndtr(dc_ / sc) - mid[sel],
                                         ndtr(dp_ / sc) - mid[sel]]).astype(np.float32))
        K_["ctx"].append(np.column_stack([mid[sel], tte[sel], sig[sel], spr[sel], yb[sel], ya[sel]]).astype(np.float32))
        fi = sel[:, None] + FUT[None, :]; fok = sane[fi]
        fut = np.stack([np.where(fok, mid[fi], np.nan), np.where(fok, yb[fi], np.nan),
                        np.where(fok, ya[fi], np.nan)], 2)
        K_["fut"].append(fut.reshape(len(sel), -1).astype(np.float32))
        K_["tk"].append(np.full(len(sel), len(names), np.int32)); names.append(str(tk))
        K_["ts"].append(grid[sel].astype(np.int64))
        r = meta.get("result")
        K_["settle"].append(np.full(len(sel), 1.0 if r == "yes" else 0.0 if r == "no" else np.nan, np.float32))
        K_["trig"].append(np.abs(c1[sel]) >= 2.0 * sig[sel])
    if not K_["mh"]:
        print("%s: no data" % day, flush=True); continue
    np.savez_compressed(os.path.join(OUT, day + ".npz"), names=np.array(names),
                        **{k: (np.vstack(v) if v[0].ndim > 1 else np.concatenate(v)) for k, v in K_.items()})
    print("%s: %d rows, %d markets" % (day, sum(len(x) for x in K_["trig"]), len(names)), flush=True)
