#!/usr/bin/env python3
"""Actual ATM-strike P(yes) CENT move around >=bps/win ES moves, split by time-to-
settlement. Picks the strike nearest P=0.5 at t0, FIXES it, and tracks its P(yes) as
ES moves + Kalshi reprices. Tests the 'near-settle pays in cents' claim directly
(no slope estimate)."""
import argparse
import numpy as np
import pandas as pd
from sp_event_study import load_future, load_strikes_by_event
from sp_timeline import settle_ns
from sp_implied_leadlag import _series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--series", default="KXINXU")
    ap.add_argument("--grid", default="100ms")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--fwd-ms", type=int, default=1000)
    ap.add_argument("--split-min", type=float, default=30.0)
    ap.add_argument("--atm-band", type=float, default=0.25, help="require |P(yes)-0.5| <= this at t0 (0.15 = tight ATM)")
    ap.add_argument("--date", default="", help="restrict to one date partition")
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step)); O = round(a.fwd_ms / step)

    es = load_future(a.dir, a.future, a.date).resample(a.grid).last().ffill()
    by_ev = load_strikes_by_event(a.dir, a.series, a.date)
    near, far = [], []
    for E, strikes in by_ev.items():
        sE = settle_ns(E)
        if sE is None or len(strikes) < 4:
            continue
        ks = sorted(strikes)
        P = pd.DataFrame({k: _series(strikes[k]).resample(a.grid).last() for k in ks}).sort_index().ffill()
        idx = P.index
        e = es.reindex(idx).ffill()
        mv = (np.log(e) - np.log(e).shift(win))
        mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
        Pv = P.to_numpy(); ii = np.asarray(idx.view("int64"))
        last = None; prev = False
        for k in range(len(mag)):
            on = mag[k] >= a.bps
            if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
                last = ii[k]
                if k + O < len(Pv):
                    row = Pv[k]
                    if not np.all(np.isnan(row)):
                        ki = int(np.nanargmin(np.abs(row - 0.5)))
                        if abs(row[ki] - 0.5) <= a.atm_band:
                            tl = [sgn[k] * (Pv[k + o][ki] - Pv[k][ki]) * 100 for o in range(O + 1)]
                            tte = (sE - ii[k]) / 6e10
                            (near if tte <= a.split_min else far).append((tte, row[ki], tl))
            prev = on

    def show(name, rows):
        if not rows:
            print(f"\n{name}: 0 events"); return
        A = np.array([r[2] for r in rows], float)
        ttes = [r[0] for r in rows]
        print(f"\n{name}: {len(rows)} events  (median TTE {np.median(ttes):.0f} min)")
        print("   offset   ATM P(yes) move (cents)")
        for o in [0, 1, 2, 3, 4, 5, 6, 8, 10]:
            if o <= O:
                print(f"   +{int(o*step):4d}ms   {np.nanmean(A[:, o]):+.1f}c")

    print(f"{a.future} vs {a.series}: >={a.bps}bps/{a.win_ms}ms, ATM-strike P(yes) cent move")
    show(f"NEAR settle (<={a.split_min:.0f}min)", near)
    show(f"FAR settle (>{a.split_min:.0f}min)", far)


if __name__ == "__main__":
    main()
