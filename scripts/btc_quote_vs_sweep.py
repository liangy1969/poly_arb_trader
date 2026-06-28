#!/usr/bin/env python3
"""Does the Kalshi QUOTE (book mid) move before or after the SWEEP (trade)?
For each perp move, on the swept market: trajectory of the book mid (signed by move
dir) vs t0, plus the main-sweep time. Both Kalshi-direct (~33ms), so tunnel-free.

  quote moves BEFORE sweep  => MM reprices first; sweep hits already-fair book (no stale-quote arb)
  quote moves AT/AFTER sweep => the sweep itself moves the book (front-runnable)
"""
import argparse, glob, json
import numpy as np
import pandas as pd
from btc_trade_tape import load_perp, load_trades


def load_kalshi_book():
    per = {}
    for f in glob.glob("data/leadlag-collect/stream=book/venue=kalshi/**/events.jsonl", recursive=True):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=float, default=2.0)
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--refractory-ms", type=int, default=1000)
    a = ap.parse_args()
    perp = load_perp().resample("50ms").last().ffill().dropna()
    tts, tpx, tqty, tside, texch = load_trades()
    tinst = []
    for f in glob.glob("data/leadlag-collect/stream=trade/venue=kalshi/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                t = json.loads(ln)["payload"]["Trade"]
            except Exception:
                continue
            if "KXBTC15M" in t.get("instrument", ""):
                tinst.append(t["instrument"])
    tinst = np.array(tinst, dtype=object)
    book = load_kalshi_book()

    win = max(1, a.win_ms // 50)
    mv = (np.log(perp) - np.log(perp).shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(perp.index.view("int64"))
    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            events.append((ii[k], sgn[k])); last = ii[k]
        prev = on

    pts = np.asarray(perp.index.view("int64")); pmid = perp.to_numpy()
    offs = [-200, -100, -50, 0, 50, 100, 200, 300, 400, 500]
    qtraj = {o: [] for o in offs}; ptraj = {o: [] for o in offs}; sweeps = []
    used = 0
    for t0, d in events:
        m = (tts >= t0 - 3e8) & (tts < t0 + 5e8)
        if m.sum() == 0:
            continue
        insts, q = tinst[m], tqty[m]
        swept = max(set(insts), key=lambda x: q[insts == x].sum())
        if swept not in book:
            continue
        bt, bm = book[swept]
        j0 = np.searchsorted(bt, t0 - 3e8) - 1
        p0i = np.searchsorted(pts, t0 - 3e8) - 1
        if j0 < 0 or p0i < 0:
            continue
        base, pbase = bm[j0], np.log(pmid[p0i])
        used += 1
        for o in offs:
            j = np.searchsorted(bt, t0 + o * 1e6) - 1
            if j >= 0:
                qtraj[o].append(d * (bm[j] - base) * 100)
            pj = np.searchsorted(pts, t0 + o * 1e6) - 1
            if pj >= 0:
                ptraj[o].append(d * (np.log(pmid[pj]) - pbase) * 1e4)  # bps
        st = tts[m][insts == swept]
        sweeps.append((st.min() - t0) / 1e6)

    print(f">={a.bps}bps perp moves with a swept market: {used}\n")
    qf = np.mean(qtraj[500]); pf = np.mean(ptraj[500])
    print(f"  {'offset':>7}  {'PERP(bps)':>9} {'perp%':>6}   {'QUOTE(c)':>8} {'quote%':>6}")
    for o in offs:
        if qtraj[o] and ptraj[o]:
            pm_, qm_ = np.mean(ptraj[o]), np.mean(qtraj[o])
            print(f"   {o:+5d}ms  {pm_:+9.2f} {100*pm_/pf:>5.0f}%   {qm_:+8.2f} {100*qm_/qf:>5.0f}%")
    if sweeps:
        s = np.array(sweeps)
        print(f"\n  main sweep first-trade: median {np.median(s):+.0f}ms rel t0  (recall quote is Kalshi-direct, t0 is +~133ms tunnel)")


if __name__ == "__main__":
    main()
