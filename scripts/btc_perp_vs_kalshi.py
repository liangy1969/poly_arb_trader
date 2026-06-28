#!/usr/bin/env python3
"""Binance perp (via the TOKYO relay) vs Kalshi BTC quote P(up) lead-lag.

For each perp >=bps/win move (t0 on the local recv clock), show the perp trajectory
(bps) and the active KXBTC15M P(up) trajectory (cents, signed by move dir), on the
SAME recv clock. The earlier SG-feed result was synchronous (quote moved with the
perp => no seam). With the Tokyo feed ~77ms fresher, the perp t0 shifts earlier, so a
lagging quote should now ramp AFTER t0. The offset where the quote reaches its final
value, minus 0, is the capturable lead.

  py scripts/btc_perp_vs_kalshi.py --dir data/leadlag-btc-tokyo --bps 2
"""
import argparse, glob, json
import numpy as np
import pandas as pd


def load_perp(dir):
    rows = []
    for f in glob.glob(f"{dir}/stream=book/venue=binance/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            if not b["bids"] or not b["asks"]:
                continue
            rows.append((e["ts_ns"], (b["bids"][0][0] + b["asks"][0][0]) / 2))
    rows.sort()
    s = pd.Series([m for _, m in rows], index=pd.to_datetime([t for t, _ in rows], utc=True))
    return s[~s.index.duplicated(keep="last")]


def load_kalshi_books(dir):
    per = {}
    for f in glob.glob(f"{dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            inst = b.get("instrument", "")
            if "KXBTC15M" not in inst or not inst.endswith(".YES") or not b["bids"] or not b["asks"]:
                continue
            per.setdefault(inst, []).append((e["ts_ns"], (b["bids"][0][0] + b["asks"][0][0]) / 2))
    out = {}
    for k, v in per.items():
        v.sort()
        out[k] = (np.array([t for t, _ in v]), np.array([m for _, m in v]))
    return out


def analyze(perp, books, pts, pmid, ii, mag, sgn, bps, win_ms, refractory_ms, hrs):
    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= bps
        if on and not prev and (last is None or ii[k] - last >= refractory_ms * 1e6):
            events.append((ii[k], sgn[k])); last = ii[k]
        prev = on
    print(f"\n===== >={bps}bps/{win_ms}ms =====")
    print(f"span={hrs:.2f}h  perp moves: {len(events)} ({len(events)/max(hrs,0.01):.0f}/h)")
    if len(events) < 3:
        print("  too few events"); return

    offs = [-200, -150, -100, -50, 0, 50, 100, 150, 200, 300, 400, 500]
    ptraj = {o: [] for o in offs}; qtraj = {o: [] for o in offs}; used = 0
    for t0, d in events:
        # active KXBTC15M market = most book updates within +-1s of t0, P(up) still tradeable
        best = None; bestn = 0
        for inst, (bt, bm) in books.items():
            j0 = np.searchsorted(bt, t0 - 3e8) - 1
            if j0 < 0 or not (0.05 <= bm[j0] <= 0.95):
                continue
            n = np.searchsorted(bt, t0 + 5e8) - np.searchsorted(bt, t0 - 3e8)
            if n > bestn:
                bestn = n; best = (bt, bm, j0)
        if best is None:
            continue
        bt, bm, j0 = best
        p0i = np.searchsorted(pts, t0 - 3e8) - 1
        if p0i < 0:
            continue
        base = bm[j0]; pbase = np.log(pmid[p0i]); used += 1
        for o in offs:
            j = np.searchsorted(bt, t0 + o * 1e6) - 1
            if j >= 0:
                qtraj[o].append(d * (bm[j] - base) * 100)
            pj = np.searchsorted(pts, t0 + o * 1e6) - 1
            if pj >= 0:
                ptraj[o].append(d * (np.log(pmid[pj]) - pbase) * 1e4)

    print(f"matched to an active Kalshi market: {used}")
    pf = np.mean(ptraj[500]) or 1; qf = np.mean(qtraj[500]) or 1
    print(f"  {'offset':>7}  {'PERP(bps)':>9} {'%fin':>5}   {'Kalshi P(up)':>12} {'%fin':>5}")
    for o in offs:
        if ptraj[o] and qtraj[o]:
            pm_ = np.mean(ptraj[o]); qm_ = np.mean(qtraj[o])
            print(f"   {o:+5d}ms  {pm_:+9.2f} {100*pm_/pf:>4.0f}%   {qm_:+10.2f}c {100*qm_/qf:>4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-btc-tokyo")
    ap.add_argument("--bps", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    perp = load_perp(a.dir).resample("50ms").last().ffill().dropna()
    books = load_kalshi_books(a.dir)
    hrs = (perp.index[-1] - perp.index[0]).total_seconds() / 3600
    print(f"loaded perp={len(perp):,} grid pts, {len(books)} kalshi markets, span={hrs:.2f}h")

    win = max(1, a.win_ms // 50)
    mv = (np.log(perp) - np.log(perp).shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(perp.index.view("int64"))
    pts = np.asarray(perp.index.view("int64")); pmid = perp.to_numpy()
    for bps in a.bps:
        analyze(perp, books, pts, pmid, ii, mag, sgn, bps, a.win_ms, a.refractory_ms, hrs)
    print("\n  perp leads => perp hits ~100% by t0 while Kalshi P(up) still ramps after t0")


if __name__ == "__main__":
    main()
