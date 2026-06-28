#!/usr/bin/env python3
"""Post-trigger Kalshi P(up) move (cents, baselined at t0) bucketed by P(up) at the
trigger (pre-move resting level) and by time-to-settle. Shows where the response is
biggest. Violent moves (>max-bps) excluded since they pin the book.
  py scripts/btc_buckets.py --dir data/leadlag-btc-tokyo
"""
import argparse
import numpy as np
from btc_perp_vs_kalshi import load_perp, load_kalshi_books
from btc_event_timeline import settle_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-btc-tokyo")
    ap.add_argument("--bps", type=float, default=2.0)
    ap.add_argument("--max-bps", type=float, default=8.0)
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

    offs = [1, 2, 4, 10, 20]  # +50,+100,+200,+500,+1000ms
    rows = []
    for t0, d, mg in raw:
        if mg > a.max_bps:
            continue
        best = None; bestn = 0
        for inst, (bt, bm) in books.items():
            j0 = np.searchsorted(bt, t0 - 3e8) - 1
            if j0 < 0 or not (0.05 <= bm[j0] <= 0.95):
                continue
            n = np.searchsorted(bt, t0 + 5e8) - np.searchsorted(bt, t0 - 3e8)
            if n > bestn:
                bestn = n; best = (bt, bm, j0, inst)
        if best is None:
            continue
        bt, bm, j0, inst = best
        sN = settle_ns(inst); tte = (sN - t0) / 6e10 if sN else 999
        jt0 = np.searchsorted(bt, t0) - 1
        if jt0 < 0:
            continue
        base = bm[jt0]; P0 = bm[j0]
        inc = {}
        for o in offs:
            j = np.searchsorted(bt, t0 + o * 50 * 1e6) - 1
            inc[o] = d * (bm[j] - base) * 100 if j >= 0 else np.nan
        rows.append((P0, tte, mg, inc))

    print(f"matched events (bps {a.bps}-{a.max_bps}): {len(rows)}")
    print("post-trigger P(up) move in cents (baselined at t0, signed by perp dir)\n")

    def show(title, key, edges, fmt):
        print(f"=== by {title} ===")
        print(f"  {'bucket':>11} {'n':>3}   " + "  ".join(f"{o*50:+5d}" for o in offs) + "  ms")
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sub = [r for r in rows if lo <= key(r) < hi]
            lab = fmt(lo, hi)
            if not sub:
                print(f"  {lab:>11} {0:>3}")
                continue
            vals = [np.nanmean([r[3][o] for r in sub]) for o in offs]
            print(f"  {lab:>11} {len(sub):>3}   " + "  ".join(f"{v:+5.1f}" for v in vals))
        print()

    show("P(up) at trigger (pre-move level)", lambda r: r[0],
         [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95], lambda lo, hi: f"{lo:.2f}-{hi:.2f}")
    show("time-to-settle", lambda r: r[1],
         [0, 2, 4, 6, 9, 12, 15], lambda lo, hi: f"{lo:.0f}-{hi:.0f}m")


if __name__ == "__main__":
    main()
