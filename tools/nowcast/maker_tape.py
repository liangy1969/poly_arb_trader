"""Phase 0 analysis (DESIGN_MARKET_MAKING.md): maker economics from the Kalshi trade tape.
Inputs: trades-YYYY-MM-DD.csv* (kalshi_trades_poller.py; .csv/.csv.gz/.part.csv pieces are concatenated and
deduped on trade_id), the same day's 50ms sampler (data/samples/YYYY-MM-DD.csv[.gz]) and, optionally, a harness
--dump-fair CSV of the nowcast. For every print: the book at the print time (bid/ask/sizes), which side of the
book it hit (taker YES buy lifts the ASK -> the resting ASK maker was filled, short YES at the print price;
taker NO buy hits the BID -> the resting BID maker is long YES), the maker's mark-to-mid P&L at horizons
(1 s / 6 s / 30 s / 60 s after the print, and at settlement from meta_cache), the queue ahead (displayed size at
that level) and the nowcast flag at the print (fair - mid, signed toward the maker's side).
Outputs: taker arrival rate at the touch by tte and price region; maker P&L per fill = spread capture (touch - mid)
- adverse move, split by the nowcast flag (informed vs uninformed proxy); sizes. Vectorized per market.
Usage: maker_tape.py DAY [DUMP.csv]   (e.g. 2026-09-11)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

day = sys.argv[1]
dump_path = sys.argv[2] if len(sys.argv) > 2 else None
tps = sorted(glob.glob(f"data/samples/trades-{day}.csv*"))
sp = [p for p in glob.glob(f"data/samples/{day}.csv*")]
assert tps and sp, "need trades-%s.csv* and the sampler day" % day
T = pd.concat([pd.read_csv(f) for f in tps], ignore_index=True).drop_duplicates("trade_id")
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
    D = D.drop_duplicates(["ticker", "ts_ms"], keep="first").sort_values(["ticker", "ts_ms"])
HORIZ = (1, 6, 30, 60)
parts = []
for tk, g in T.groupby("ticker"):
    B = S[S.ticker == tk]
    if len(B) == 0:
        continue
    ts = B.ts_ms.to_numpy(); yb, ya, ybs, yas, tte = (B[c].to_numpy() for c in ("ybid", "yask", "ybid_sz", "yask_sz", "tte_ms"))
    pts = g.ts_ms.to_numpy()
    k = np.searchsorted(ts, pts, side="right") - 1                        # last book row at/before the print
    ok = (k >= 0) & (pts - ts[np.clip(k, 0, len(ts) - 1)] <= 400)
    if not ok.any():
        continue
    g = g[ok]; k = k[ok]; pts = pts[ok]
    mid0 = 0.5 * (yb[k] + ya[k])
    taker_yes = g.taker_side.astype(str).str.startswith("y").to_numpy()
    px = g.yes_price.to_numpy()
    sign = np.where(taker_yes, -1.0, 1.0)                                  # maker short YES on a taker YES buy
    touch = np.where(taker_yes, ya[k], yb[k]); queue = np.where(taker_yes, yas[k], ybs[k])
    at_touch = np.abs(px - touch) < 0.0005
    inside = (px > yb[k] + 0.0005) & (px < ya[k] - 0.0005)                 # print strictly inside the sampled spread (hidden/improved maker)
    rec = {"ticker": tk, "ts_ms": pts, "tte_s": tte[k] / 1000.0, "mid0": mid0, "px": px, "count": g["count"].to_numpy(),
           "taker_yes": taker_yes, "at_touch": at_touch, "inside": inside, "spread_c": 100 * (ya[k] - yb[k]), "queue": queue,
           "capture_c": 100 * sign * (mid0 - px)}
    for h in HORIZ:
        j = np.minimum(np.searchsorted(ts, pts + 1000 * h, side="left"), len(ts) - 1)
        rec["pnl%d" % h] = 100 * sign * (0.5 * (yb[j] + ya[j]) - px)
    m = meta.get(tk) or {}
    outcome = m.get("result") if isinstance(m, dict) else None
    if outcome in ("yes", "no"):
        rec["pnl_settle"] = 100 * sign * ((1.0 if outcome == "yes" else 0.0) - px)
    else:
        rec["pnl_settle"] = np.full(len(pts), np.nan)
    gg = np.full(len(pts), np.nan)
    if D is not None:
        Dk = D[D.ticker == tk]
        if len(Dk):
            dts = Dk.ts_ms.to_numpy(); jd = np.searchsorted(dts, pts, side="right") - 1
            okd = (jd >= 0) & (pts - dts[np.clip(jd, 0, len(dts) - 1)] <= 400)
            gg[okd] = 100 * sign[okd] * (Dk.fair.to_numpy()[jd[okd]] - Dk.mid.to_numpy()[jd[okd]])   # + = nowcast says the mid moves in the MAKER's favour
    rec["g"] = gg
    parts.append(pd.DataFrame(rec))
R = pd.concat(parts, ignore_index=True)
nraw = len(R); R = R[(R.spread_c > 0) & (R.spread_c <= 10)].copy()          # drop crossed / garbage book rows (sampler glitches)
print("book rows dropped as crossed or >10c wide: %d of %d" % (nraw - len(R), nraw))
print("prints %d (of %d in the tape) | markets %d | at-touch %.0f%% inside-spread %.0f%% | median size %.2f | in trade window (60-300s): %d" % (
    len(R), len(T), R.ticker.nunique(), 100 * R.at_touch.mean(), 100 * R.inside.mean(), R["count"].median(), ((R.tte_s >= 60) & (R.tte_s <= 300)).sum()))


def show(h, lab):
    w = (lambda c: (h[c] * h["count"]).sum() / h["count"].sum())           # size-weighted
    print("  %-30s n %7d | capture %+5.2fc | P&L @1s %+5.2fc @6s %+5.2fc @30s %+5.2fc @60s %+5.2fc settle %+5.2fc | adverse@6s %+5.2fc | size-wtd @6s %+5.2fc | queue med %.0f" % (
        lab, len(h), h.capture_c.mean(), h.pnl1.mean(), h.pnl6.mean(), h.pnl30.mean(), h.pnl60.mean(), h.pnl_settle.mean(), (h.pnl6 - h.capture_c).mean(), w("pnl6"), h.queue.median()))


W = R[(R.tte_s >= 60) & (R.tte_s <= 300) & R.at_touch]
print("\nMAKER at the touch, trade window 60-300s: per fill (cents, maker-signed): capture = touch - mid at the print; P&L = mid later - fill")
show(W, "all")
show(W[~W.taker_yes], "maker bought YES (taker sold)")
show(W[W.taker_yes], "maker sold YES (taker bought)")
for lab, m in (("mid < 0.10", W.mid0 < 0.10), ("0.10-0.30", (W.mid0 >= 0.10) & (W.mid0 < 0.30)), ("0.30-0.70", (W.mid0 >= 0.30) & (W.mid0 <= 0.70)), ("0.70-0.90", (W.mid0 > 0.70) & (W.mid0 <= 0.90)), ("mid > 0.90", W.mid0 > 0.90)):
    if m.sum() >= 50:
        show(W[m], lab)
for lab, m in (("size < 1", W["count"] < 1), ("1-10", (W["count"] >= 1) & (W["count"] < 10)), ("10-100", (W["count"] >= 10) & (W["count"] < 100)), ("size >= 100", W["count"] >= 100)):
    if m.sum() >= 50:
        show(W[m], "print " + lab)
if W.g.notna().any():
    print("\nby NOWCAST flag at the print (g = fair - mid signed toward the maker; g < -0.25c = model says the maker is about to be run over):")
    for lab, m in (("g < -0.25c (pull)", W.g < -0.25), ("-0.25..0.25 (quiet)", W.g.abs() <= 0.25), ("g > 0.25c (favourable)", W.g > 0.25)):
        h = W[m]
        if len(h) >= 20:
            print("  %-24s n %7d (%4.1f%%) | capture %+5.2fc | P&L @1s %+5.2fc @6s %+5.2fc @30s %+5.2fc @60s %+5.2fc settle %+5.2fc" % (
                lab, len(h), 100 * len(h) / max(len(W), 1), h.capture_c.mean(), h.pnl1.mean(), h.pnl6.mean(), h.pnl30.mean(), h.pnl60.mean(), h.pnl_settle.mean()))
    q = W[W.g.abs() <= 0.25]
    print("  POLICY quiet-only (quote only when |g|<=0.25c): fills kept %.0f%% | P&L/fill @6s %+5.2fc @30s %+5.2fc @60s %+5.2fc (naive %+5.2fc / %+5.2fc / %+5.2fc)" % (
        100 * len(q) / max(len(W), 1), q.pnl6.mean(), q.pnl30.mean(), q.pnl60.mean(), W.pnl6.mean(), W.pnl30.mean(), W.pnl60.mean()))
print("\narrival rate at the touch by tte bucket (prints per market-minute of sampler coverage) and mean size:")
Sw = S[(S.tte_ms >= 60000) & (S.tte_ms <= 300000)]
for lo, hi in ((60, 120), (120, 180), (180, 240), (240, 300)):
    h = W[(W.tte_s >= lo) & (W.tte_s < hi)]
    cov_min = ((Sw.tte_ms >= lo * 1000) & (Sw.tte_ms < hi * 1000)).sum() * 0.05 / 60.0
    print("  tte %3d-%3d: %7d prints | %.1f per market-minute | mean size %.1f | contracts per market-minute %.0f" % (lo, hi, len(h), len(h) / max(cov_min, 1e-9), h["count"].mean(), h["count"].sum() / max(cov_min, 1e-9)))
os.makedirs("data/nowcast", exist_ok=True)
R.to_parquet(f"data/nowcast/tape_{day}.parquet")
print("\nsaved data/nowcast/tape_%s.parquet" % day)
