"""Model-free 'panic fade' backtest on the 50 ms sampler days (the one archetype the public 4,904-strategy Turbine
backtest found profitable on KXBTC15M): when the Kalshi YES mid has moved by >= X cents against its value L seconds
earlier, take the OTHER side as a taker at the touch (fee 0.07*p*(1-p) per contract, buy at the ask / sell at the bid
via the complement), one entry per market, hold to settlement (meta_cache outcome). Also the 'from window open'
variant (move measured from the mid at tte = 840 s). Event-normalized net/trade with event-clustered t, by day range.
Usage: panic_fade.py DAY [DAY ...]   env X=3,5,8,10,15 L=30,60,120 TTE=60,600 (entry window, s to expiry)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

XS = [float(x) for x in os.environ.get("X", "3,5,8,10,15").split(",")]
LS = [int(x) for x in os.environ.get("L", "30,60,120,open").replace("open", "-1").split(",")]
TTE_LO, TTE_HI = (float(x) for x in os.environ.get("TTE", "60,600").split(","))
FEE = 0.07
meta = json.load(open("data/samples/meta_cache.json"))
rows = []
for day in sys.argv[1:]:
    fn = sorted(glob.glob(f"data/samples/{day}.csv*"))
    if not fn:
        continue
    S = pd.read_csv(fn[0], usecols=["ts_ms", "ticker", "tte_ms", "ybid", "yask"], dtype=str, on_bad_lines="skip")
    for c in ("ts_ms", "tte_ms", "ybid", "yask"):
        S[c] = pd.to_numeric(S[c], errors="coerce")
    S = S.dropna(); S = S[(S.yask > S.ybid) & (S.yask - S.ybid <= 0.10)]
    S = S[(S.ts_ms // 1000) != (S.ts_ms // 1000).shift()]            # ~1 row per second
    for tk, B in S.groupby("ticker"):
        m = meta.get(tk) or {}
        res = m.get("result") if isinstance(m, dict) else None
        if res not in ("yes", "no"):
            continue
        settle = 1.0 if res == "yes" else 0.0
        B = B.sort_values("ts_ms"); ts = B.ts_ms.to_numpy(); tte = B.tte_ms.to_numpy() / 1000.0
        yb, ya = B.ybid.to_numpy(), B.yask.to_numpy(); mid = 0.5 * (yb + ya)
        open_idx = np.searchsorted(-tte, -840.0)            # first row with tte <= 840 s (window open + ~1 min)
        mid_open = mid[min(open_idx, len(mid) - 1)]
        for L in LS:
            if L > 0:
                j = np.searchsorted(ts, ts - 1000 * L, side="right") - 1
                ok = j >= 0; ref = np.where(ok, mid[np.clip(j, 0, len(mid) - 1)], np.nan)
            else:
                ref = np.full(len(mid), mid_open)
            move = 100 * (mid - ref)
            for X in XS:
                inwin = (tte >= TTE_LO) & (tte <= TTE_HI)
                hit = np.where(inwin & np.isfinite(move) & (np.abs(move) >= X))[0]
                if len(hit) == 0:
                    continue
                i = hit[0]
                if move[i] > 0:          # price jumped up -> fade: buy NO = sell YES at the bid
                    px = yb[i]; fee = FEE * px * (1 - px); net = (px - settle) - fee
                    side = "NO"
                else:                    # price dropped -> buy YES at the ask
                    px = ya[i]; fee = FEE * px * (1 - px); net = (settle - px) - fee
                    side = "YES"
                rows.append(dict(day=day, ticker=tk, L=L, X=X, tte=tte[i], px=px, side=side, move=move[i], net=100 * net))
R = pd.DataFrame(rows)


def clus(g):
    e = g.groupby("ticker")["net"].mean()
    return e.mean(), (e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 3 and e.std() > 0 else 0.0), len(e)


days = sorted(R.day.unique()) if len(R) else []
sl = {"all": R, "Jul25-Aug26": R[R.day <= "2026-08-26"], "Aug27-Sep04": R[(R.day > "2026-08-26") & (R.day <= "2026-09-04")], "Sep05-15": R[R.day > "2026-09-04"]}
print("panic fade, taker at the touch, hold to settlement, fee 0.07 p(1-p); %d days %s..%s | entry tte %g-%g s" % (len(days), days[0] if days else "-", days[-1] if days else "-", TTE_LO, TTE_HI))
print("%-6s %-5s | %-28s | %-28s | %-28s | %-28s" % ("L", "X", "ALL", "Jul25-Aug26", "Aug27-Sep04", "Sep05-15"))
for L in LS:
    for X in XS:
        cells = []
        for k in ("all", "Jul25-Aug26", "Aug27-Sep04", "Sep05-15"):
            h = sl[k]; h = h[(h.L == L) & (h.X == X)]
            cells.append("%+6.2fc t%+5.2f n=%4d" % clus(h) if len(h) >= 20 else "   -- n=%d" % len(h))
        print("%-6s %-5g | %s" % ("open" if L < 0 else L, X, " | ".join(cells)))
R.to_parquet("data/nowcast/panic_fade.parquet")
