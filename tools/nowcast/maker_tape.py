"""Phase 0 analysis (DESIGN_MARKET_MAKING.md): maker economics from the Kalshi trade tape.
Inputs: trades-YYYY-MM-DD.csv (kalshi_trades_poller.py), the same day's 50ms sampler
(data/samples/YYYY-MM-DD.csv[.gz]) and, optionally, a harness --dump-fair CSV of the nowcast.
For every print: the book at the print time (bid/ask/sizes), which side of the book it hit
(taker YES buy hits the ask -> the resting ASK maker was filled; taker NO buy hits the bid ->
resting BID maker filled), the maker's realized P&L at horizons (mid 1s / 6s / 30s after the print
minus the fill price, signed for the maker), the queue ahead (displayed size at that level), and
the nowcast flag at the print (fair - mid, signed toward the maker's side).
Outputs: taker arrival rate at the touch by side/price/tte; maker P&L per fill = spread capture
(touch - mid) - adverse move; split by nowcast flag (informed vs uninformed proxy); sizes.
Usage: maker_tape.py DAY [DUMP.csv]   (e.g. 2026-09-11)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

day = sys.argv[1]
dump_path = sys.argv[2] if len(sys.argv) > 2 else None
tp = f"data/samples/trades-{day}.csv"
sp = [p for p in glob.glob(f"data/samples/{day}.csv*")]
assert os.path.exists(tp) and sp, "need trades-%s.csv and the sampler day" % day
T = pd.read_csv(tp)
T["created_time"] = pd.to_datetime(T["created_time"], utc=True, errors="coerce")
T = T[T.created_time.notna() & T.yes_price.notna() & T["count"].notna()].copy()
T["ts_ms"] = (T.created_time.astype("int64") // 1_000_000)
S = pd.read_csv(sp[0], usecols=["ts_ms", "ticker", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"], dtype=str, on_bad_lines="skip")
S["ticker"] = S["ticker"].astype(str)
for c in ("ts_ms", "tte_ms", "ybid", "yask", "ybid_sz", "yask_sz"):
    S[c] = pd.to_numeric(S[c], errors="coerce")
S = S.dropna().sort_values(["ticker", "ts_ms"])
meta = json.load(open("data/samples/meta_cache.json"))
D = None
if dump_path and os.path.exists(dump_path):
    D = pd.read_csv(dump_path, usecols=["ticker", "ts_ms", "fair", "mid"]); D["ticker"] = D["ticker"].astype(str)
rows = []
for tk, g in T.groupby("ticker"):
    B = S[S.ticker == tk]
    if len(B) == 0:
        continue
    ts = B.ts_ms.to_numpy(); yb, ya, ybs, yas, tte = (B[c].to_numpy() for c in ("ybid", "yask", "ybid_sz", "yask_sz", "tte_ms"))
    Dk = D[D.ticker == tk].sort_values("ts_ms") if D is not None else None
    dts = Dk.ts_ms.to_numpy() if Dk is not None else None
    for _, r in g.iterrows():
        k = int(np.searchsorted(ts, r.ts_ms, side="right")) - 1     # last book row at/before the print
        if k < 0 or r.ts_ms - ts[k] > 400:
            continue
        mid0 = 0.5 * (yb[k] + ya[k])
        taker_yes = str(r.taker_side).startswith("y")
        # maker side: taker YES buy lifts the ASK (maker sold YES at ~ask); taker NO buy hits the BID (maker bought YES at ~bid)
        fill_px = r.yes_price
        if taker_yes:
            maker_sign, touch, queue = -1.0, ya[k], yas[k]   # maker is short YES at fill_px
        else:
            maker_sign, touch, queue = +1.0, yb[k], ybs[k]   # maker is long YES at fill_px
        at_touch = abs(fill_px - touch) < 0.0005
        def mid_after(sec):
            j = int(np.searchsorted(ts, r.ts_ms + 1000 * sec, side="left"))
            j = min(j, len(ts) - 1)
            return 0.5 * (yb[j] + ya[j])
        pnl = {h: 100 * maker_sign * (mid_after(h) - fill_px) for h in (1, 6, 30)}
        g_ = np.nan
        if dts is not None and len(dts):
            jd = int(np.searchsorted(dts, r.ts_ms, side="right")) - 1
            if jd >= 0 and r.ts_ms - dts[jd] <= 400:
                g_ = 100 * maker_sign * (Dk.fair.iloc[jd] - Dk.mid.iloc[jd])   # + = nowcast says the mid moves in the MAKER's favour
        rows.append({"ticker": tk, "ts_ms": r.ts_ms, "tte_s": tte[k] / 1000.0, "mid0": mid0, "px": fill_px, "count": r["count"],
                     "taker_yes": taker_yes, "at_touch": at_touch, "spread_c": 100 * (ya[k] - yb[k]), "queue": queue,
                     "capture_c": 100 * maker_sign * (fill_px - mid0), "pnl1": pnl[1], "pnl6": pnl[6], "pnl30": pnl[30], "g": g_})
R = pd.DataFrame(rows)
print("prints %d (of %d in the tape) | markets %d | at-touch %.0f%% | median size %.2f | in trade window (60-300s): %d" % (
    len(R), len(T), R.ticker.nunique(), 100 * R.at_touch.mean(), R["count"].median(), ((R.tte_s >= 60) & (R.tte_s <= 300)).sum()))
W = R[(R.tte_s >= 60) & (R.tte_s <= 300) & R.at_touch]
print("\nMAKER at the touch, trade window: per fill (cents, maker-signed): capture = touch - mid at the print")
for lab, m in (("all", np.ones(len(W), bool)), ("maker sold YES (taker bought)", ~W.taker_yes.astype(bool) == False), ("maker bought YES (taker sold)", ~W.taker_yes.astype(bool))):
    h = W[m]
    if len(h) < 20:
        continue
    print("  %-30s n %6d | capture %+5.2fc | P&L @1s %+5.2fc @6s %+5.2fc @30s %+5.2fc | adverse@6s %+5.2fc | queue median %.0f" % (
        lab, len(h), h.capture_c.mean(), h.pnl1.mean(), h.pnl6.mean(), h.pnl30.mean(), (h.pnl6 - h.capture_c).mean(), h.queue.median()))
if W.g.notna().any():
    print("\nby NOWCAST flag at the print (g = fair - mid signed toward the maker; g < -0.25c = model says the maker is about to be run over):")
    for lab, m in (("g < -0.25c (pull)", W.g < -0.25), ("-0.25..0.25 (quiet)", W.g.abs() <= 0.25), ("g > 0.25c (favourable)", W.g > 0.25)):
        h = W[m]
        if len(h) < 20:
            continue
        print("  %-24s n %6d (%4.1f%%) | capture %+5.2fc | P&L @6s %+5.2fc @30s %+5.2fc" % (lab, len(h), 100 * len(h) / max(len(W), 1), h.capture_c.mean(), h.pnl6.mean(), h.pnl30.mean()))
print("\narrival rate at the touch by tte bucket (prints per market-minute) and by price region:")
for lo, hi in ((60, 120), (120, 180), (180, 240), (240, 300)):
    h = W[(W.tte_s >= lo) & (W.tte_s < hi)]
    print("  tte %3d-%3d: %6d prints | %.2f per market-minute | mean size %.1f" % (lo, hi, len(h), len(h) / max(W.ticker.nunique(), 1), h["count"].mean()))
for lab, m in (("mid < 0.10", W.mid0 < 0.10), ("0.10-0.90", (W.mid0 >= 0.10) & (W.mid0 <= 0.90)), ("mid > 0.90", W.mid0 > 0.90)):
    h = W[m]
    if len(h):
        print("  %-10s: %6d prints | capture %+5.2fc | P&L@6s %+5.2fc" % (lab, len(h), h.capture_c.mean(), h.pnl6.mean()))
R.to_parquet(f"data/nowcast/tape_{day}.parquet")
print("\nsaved data/nowcast/tape_%s.parquet" % day)
