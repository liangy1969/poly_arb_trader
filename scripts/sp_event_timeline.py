#!/usr/bin/env python3
"""Per-event timeline (not averaged): every RTH 3bps ES move today, each shown with
the future (ES) and Kalshi-implied-level paths in S&P points, cumulative from a
pre-move baseline so you see ES move first and Kalshi follow, event by event.
  py scripts/sp_event_timeline.py --date 2026-06-26
"""
import argparse
import numpy as np
import pandas as pd
from sp_event_study import load_future, load_strikes_by_event
from sp_implied_leadlag import implied_level
from sp_timeline import settle_ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--series", default="KXINXU")
    ap.add_argument("--grid", default="100ms")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--date", default="2026-06-26")
    ap.add_argument("--rth-utc", default="13:30:00", help="RTH open (UTC) — events before this dropped")
    ap.add_argument("--refractory-ms", type=int, default=1500)
    ap.add_argument("--max-tte-min", type=float, default=0, help="keep only moves within this many min of settlement")
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step))

    es = load_future(a.dir, a.future, a.date).resample(a.grid).last().ffill()
    by_ev = load_strikes_by_event(a.dir, a.series, a.date)
    L = pd.concat([implied_level(s, a.grid) for s in by_ev.values() if len(s) >= 4]).sort_index()
    L = L[~L.index.duplicated(keep="last")]
    j = pd.concat({"es": es, "L": L}, axis=1).sort_index().ffill().dropna()
    esv = j["es"].to_numpy(); Lv = j["L"].to_numpy()
    ii = np.asarray(j.index.view("int64"))
    loges = np.log(j["es"]); mv = (loges - loges.shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    rth = pd.Timestamp(f"{a.date}T{a.rth_utc}Z").value
    settles = sorted(s for s in (settle_ns(e) for e in by_ev) if s)

    def tte_min(t):
        nxt = [s for s in settles if s > t]
        return (min(nxt) - t) / 6e10 if nxt else 1e9

    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and ii[k] >= rth and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            tte = tte_min(ii[k])
            if a.max_tte_min == 0 or tte <= a.max_tte_min:
                events.append((k, sgn[k], tte)); last = ii[k]
        prev = on

    offs = [-3, -1, 0, 2, 4, 6, 8, 10, 14]  # -300..+1400ms @100ms
    hdr = "  ".join(f"{int(o*step):+5d}" for o in offs)
    print(f"RTH 3bps ES moves today ({a.date}, after {a.rth_utc} UTC): {len(events)}\n")
    print(f"  (cumulative move in S&P points from t0-300ms baseline; offsets ms rel t0)")
    print(f"  {'':>12} {hdr}")
    for k, d, tte in events:
        b = k - 3
        if b < 0 or k + offs[-1] >= len(esv):
            continue
        ts = pd.Timestamp(ii[k], tz="UTC").strftime("%H:%M:%S")
        e = [d * (esv[k + o] - esv[b]) for o in offs]
        l = [d * (Lv[k + o] - Lv[b]) for o in offs]
        print(f"  {ts} {'UP' if d>0 else 'DN'} {mag[k]:.1f}bps  TTE={tte:.0f}min")
        print(f"  {'ES (pt)':>12} " + "  ".join(f"{x:+5.2f}" for x in e))
        print(f"  {'Kalshi(pt)':>12} " + "  ".join(f"{x:+5.2f}" for x in l))
        print()


if __name__ == "__main__":
    main()
