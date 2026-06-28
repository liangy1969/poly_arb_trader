#!/usr/bin/env python3
"""When does the Kalshi sweep hit, relative to my (SG-tunnel-delayed ~117ms) perp
detection t0? Since my perp feed is ~117ms late, a sweep at t0-117ms means the
sweeper acted at the TRUE move instant; earlier than that means they beat even a
zero-latency view of my feed. Bounds whether the race is winnable.

  py scripts/btc_sweep_timing.py --bps 3
"""
import argparse
import numpy as np
import pandas as pd
from btc_trade_tape import load_perp, load_trades

TUNNEL_MS = 117  # measured SG SOCKS tunnel add vs Kalshi-direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--pre-ms", type=int, default=600)
    ap.add_argument("--post-ms", type=int, default=600)
    ap.add_argument("--refractory-ms", type=int, default=1000)
    a = ap.parse_args()
    perp = load_perp().resample("50ms").last().ffill().dropna()
    tts, tpx, tqty, tside, texch = load_trades()

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

    print(f">={a.bps}bps/{a.win_ms}ms perp moves: {len(events)};  t0 = my perp detection (~{TUNNEL_MS}ms after true move)\n")
    print(f"  {'dir':>4} {'first_trade':>11} {'main_sweep':>10} {'sweep_vol':>9} {'levels':>6}  (ms relative to t0)")
    firsts, sweeps = [], []
    for t0, d in events:
        lo, hi = t0 - a.pre_ms * 1e6, t0 + a.post_ms * 1e6
        m = (tts >= lo) & (tts < hi)
        if m.sum() == 0:
            continue
        wt, wex, wpx, wq, wsd = tts[m], texch[m], tpx[m], tqty[m], tside[m]
        want = "Buy" if d > 0 else "Sell"
        ft = (wt.min() - t0) / 1e6
        # main sweep = exch_ts cluster (>=2 levels) with most aligned volume
        best = None
        for et in set(wex):
            sub = wex == et
            if sub.sum() >= 2 and len(set(np.round(wpx[sub], 4))) >= 2:
                vol = wq[sub & (wsd == want)].sum()
                if best is None or vol > best[0]:
                    best = (vol, (wt[sub].min() - t0) / 1e6, len(set(np.round(wpx[sub], 4))))
        firsts.append(ft)
        sd = f"{best[1]:+.0f}" if best else "  -"
        sv = f"{best[0]:.0f}" if best else "-"
        lv = f"{best[2]}" if best else "-"
        print(f"  {'UP' if d>0 else 'DN':>4} {ft:+11.0f} {sd:>10} {sv:>9} {lv:>6}")
        if best:
            sweeps.append(best[1])

    def stats(name, xs):
        if not xs:
            print(f"\n{name}: none"); return
        x = np.array(xs)
        print(f"\n{name} (n={len(x)}): median={np.median(x):+.0f}ms  p10={np.percentile(x,10):+.0f}  p90={np.percentile(x,90):+.0f}  "
              f"% before t0={100*np.mean(x<0):.0f}%")
        print(f"   => vs TRUE move (t0-{TUNNEL_MS}ms): median={np.median(x)+TUNNEL_MS:+.0f}ms after true move")
    stats("first trade", firsts)
    stats("main sweep", sweeps)


if __name__ == "__main__":
    main()
