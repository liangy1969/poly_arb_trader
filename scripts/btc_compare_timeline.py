#!/usr/bin/env python3
"""Aggregated perp-vs-Kalshi timeline comparison. Normalizes each to % of its final
move and prints them side by side as bars so the lead is visible, plus the lag in ms
(how much later the quote reaches the same %done). Shows ALL matched moves and the
CLEAN tradeable subset (moderate bps, mid-P, mid-TTE) which strips the violent/pinned
and near-settle-overshoot cases.
  py scripts/btc_compare_timeline.py --dir data/leadlag-btc-tokyo --bps 2
"""
import argparse
import numpy as np
from btc_perp_vs_kalshi import load_perp, load_kalshi_books
from btc_event_timeline import settle_ns


def interp_time(m, final, lvl, offs):
    target = final * lvl / 100.0
    prev_o = prev_v = None
    for o in offs:
        v = m[o]
        if prev_v is not None and prev_v < target <= v and v != prev_v:
            return (prev_o + (target - prev_v) / (v - prev_v) * (o - prev_o)) * 50
        prev_o, prev_v = o, v
    return None


def compare(name, raw, books, ii, pmid, offs, bps_hi, p_lo, p_hi, tte_lo, tte_hi):
    pacc = {o: [] for o in offs}; qacc = {o: [] for o in offs}; used = 0
    for t0, d, mg in raw:
        if mg > bps_hi:
            continue
        best = None; bestn = 0
        for inst, (bt, bm) in books.items():
            j0 = np.searchsorted(bt, t0 - 3e8) - 1
            if j0 < 0 or not (p_lo <= bm[j0] <= p_hi):
                continue
            n = np.searchsorted(bt, t0 + 5e8) - np.searchsorted(bt, t0 - 3e8)
            if n > bestn:
                bestn = n; best = (bt, bm, j0, inst)
        if best is None:
            continue
        bt, bm, j0, inst = best
        sN = settle_ns(inst); tte = (sN - t0) / 6e10 if sN else 999
        if not (tte_lo <= tte <= tte_hi):
            continue
        p0i = np.searchsorted(ii, t0 - 3e8) - 1
        if p0i < 0:
            continue
        pbase = np.log(pmid[p0i]); base = bm[j0]; used += 1
        for o in offs:
            pj = np.searchsorted(ii, t0 + o * 50 * 1e6) - 1
            if pj >= 0:
                pacc[o].append(d * (np.log(pmid[pj]) - pbase) * 1e4)
            j = np.searchsorted(bt, t0 + o * 50 * 1e6) - 1
            if j >= 0:
                qacc[o].append(d * (bm[j] - base) * 100)
    pm = {o: (np.mean(pacc[o]) if pacc[o] else 0.0) for o in offs}
    qm = {o: (np.mean(qacc[o]) if qacc[o] else 0.0) for o in offs}
    qsd = {o: (np.std(qacc[o]) / np.sqrt(len(qacc[o])) if qacc[o] else 0.0) for o in offs}
    print(f"\n### {name}: {used} events   (perp trigger ~{pm[max(o for o in offs if o<=4)]:+.1f}bps)")
    print(f"  post-trigger Kalshi P(up) move (cents, signed by perp dir):")
    print(f"  {'offset':>8}  {'P(up) move':>10}  {'+/-se':>6}")
    for o in offs:
        if o < 0:
            continue
        print(f"  {o*50:+7d}ms  {qm[o]:+9.2f}c  {qsd[o]:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-btc-tokyo")
    ap.add_argument("--bps", type=float, default=2.0)
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    perp = load_perp(a.dir).resample("50ms").last().ffill().dropna()
    books = load_kalshi_books(a.dir)
    win = max(1, a.win_ms // 50)
    mv = (np.log(perp) - np.log(perp).shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(perp.index.view("int64")); pmid = perp.to_numpy()
    raw = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            raw.append((ii[k], sgn[k], mag[k])); last = ii[k]
        prev = on
    hrs = (perp.index[-1] - perp.index[0]).total_seconds() / 3600
    print(f"span={hrs:.2f}h  raw >={a.bps}bps moves: {len(raw)}")
    offs = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20]  # 0..+1000ms @50ms grid
    compare("ALL matched", raw, books, ii, pmid, offs, 99.0, 0.05, 0.95, 0, 999)
    compare("CLEAN (bps<=8, P 0.2-0.8, TTE 4-14min)", raw, books, ii, pmid, offs, 8.0, 0.2, 0.8, 4, 14)


if __name__ == "__main__":
    main()
