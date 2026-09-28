"""How do Kalshi touch levels disappear: consumed by trades or cancelled? For every sampler row where the best bid DROPS
(or the best ask RISES) — the old touch level is gone — sum the tape volume printed AT the old price on that side
(taker NO hits the YES bid, taker YES lifts the YES ask) within [prev row - PRE ms, drop row + POST ms] (the tape is on
the venue clock, the sampler on ours: median lag ~33 ms, tail longer), and compare with the size displayed at the old
touch on the previous row. Classes: CANCELLED (no print at the price), PARTIAL (printed < 50% of displayed), TRADED
(>= 50%). By region (tails < .10 / > .90, core) and by the displayed size.
Usage: level_disappear.py DAY [DAY ...] [--pre 500] [--post 300]"""
import argparse
import glob

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("days", nargs="+"); ap.add_argument("--pre", type=int, default=500); ap.add_argument("--post", type=int, default=300)
a = ap.parse_args()
rows = []
for day in a.days:
    T = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"data/samples/trades-{day}.csv*"))], ignore_index=True).drop_duplicates("trade_id")
    T["ts"] = pd.to_datetime(T.created_time, utc=True, format="ISO8601", errors="coerce").astype("int64") // 1_000_000
    T = T[T.yes_price.notna() & T["count"].notna()]
    T["hit_bid"] = ~T.taker_side.astype(str).str.startswith("y")
    S = pd.read_csv(glob.glob(f"data/samples/{day}.csv*")[0], usecols=["ts_ms", "ticker", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"], dtype=str, on_bad_lines="skip")
    for c in ("ts_ms", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S = S.dropna(); S = S[(S.yask > S.ybid) & (S.tte_ms > 60000) & (S.tte_ms <= 300000)].sort_values(["ticker", "ts_ms"])
    Tg = {tk: g.sort_values("ts") for tk, g in T.groupby("ticker")}
    for tk, B in S.groupby("ticker"):
        tp = Tg.get(tk)
        if tp is None:
            continue
        pts, ppx, pcnt, phb = tp.ts.to_numpy(), tp.yes_price.to_numpy(), tp["count"].to_numpy(), tp.hit_bid.to_numpy()
        ts, yb, ya, ybs, yas = (B[c].to_numpy() for c in ("ts_ms", "ybid", "yask", "ybid_sz", "yask_sz"))
        for i in range(1, len(ts)):
            for side, cur, prev, sz in (("bid", yb[i], yb[i - 1], ybs[i - 1]), ("ask", ya[i], ya[i - 1], yas[i - 1])):
                gone = (cur < prev - 5e-5) if side == "bid" else (cur > prev + 5e-5)
                if not gone:
                    continue
                lo, hi = np.searchsorted(pts, ts[i - 1] - a.pre), np.searchsorted(pts, ts[i] + a.post, side="right")
                m = np.abs(ppx[lo:hi] - prev) < 5e-5
                m &= phb[lo:hi] if side == "bid" else ~phb[lo:hi]
                vol = pcnt[lo:hi][m].sum()
                # beyond-level prints (a sweep through the level) also mean it was consumed
                # a sweep THROUGH the level = a print beyond it on that side stamped before our book showed the drop
                hb = np.searchsorted(pts, ts[i], side="right")
                beyond = ((ppx[lo:hb] < prev - 5e-5) & phb[lo:hb]).any() if side == "bid" else ((ppx[lo:hb] > prev + 5e-5) & ~phb[lo:hb]).any()
                mid = 0.5 * (yb[i - 1] + ya[i - 1])
                rows.append((day, side, "tails" if (mid < 0.10 or mid > 0.90) else "core", sz, vol, beyond, abs(cur - prev)))
R = pd.DataFrame(rows, columns=["day", "side", "region", "size", "vol", "beyond", "jump"])
R["frac"] = R.vol / R["size"].clip(lower=1e-9)
R["cls"] = np.where(R.beyond | (R.frac >= 0.5), "TRADED", np.where(R.vol > 0, "PARTIAL", "CANCELLED"))
print("touch-level disappearances, days %s, window [prev row -%d ms, drop +%d ms], tte 60-300 s" % (",".join(a.days), a.pre, a.post))
for reg, g in R.groupby("region"):
    c = g.cls.value_counts(normalize=True) * 100
    print("\n%s: %d disappearances (%.0f per market-minute-ish)" % (reg, len(g), 0))
    print("   CANCELLED (no print at the price) %4.1f%% | PARTIAL (<50%% printed) %4.1f%% | TRADED (>=50%% or swept through) %4.1f%%" % (c.get("CANCELLED", 0), c.get("PARTIAL", 0), c.get("TRADED", 0)))
    g = g.copy(); g["szb"] = pd.cut(g["size"], [0, 10, 100, 1000, 1e12], labels=["<=10", "10-100", "100-1000", ">1000"])
    t = g.groupby("szb", observed=True).cls.value_counts(normalize=True).unstack(fill_value=0) * 100
    t["n"] = g.groupby("szb", observed=True).size()
    print("   by displayed size at the old touch:"); print(t.round(1).to_string())
    print("   median displayed size %.0f; median printed at the level %.0f" % (g["size"].median(), g.vol.median()))
for w in ((200, 100), (1500, 800)):
    pass
