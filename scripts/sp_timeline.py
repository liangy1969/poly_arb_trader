#!/usr/bin/env python3
"""Clean event-study timeline: ES and Kalshi implied-level, both as cumulative move
(bps) from a pre-move baseline, averaged over all >=bps/win moves. Shows ES rising
first and Kalshi lagging/catching up, as an ASCII chart.

  py scripts/sp_timeline.py --win-ms 200 --bps 3
"""
import argparse, datetime as dt, re
import numpy as np
import pandas as pd
from sp_event_study import load_future, load_strikes_by_event
from sp_implied_leadlag import implied_level

_MON = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def settle_ns(event):
    """KXINXU-26JUN25H1400 -> settle ts (ns). H<HHMM> is ET (EDT, UTC-4)."""
    m = re.search(r"(\d{2})([A-Z]{3})(\d{2})H(\d{2})(\d{2})", event)
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    et = dt.datetime(2000 + int(yy), _MON[mon], int(dd), int(hh), int(mm),
                     tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    return int(et.timestamp() * 1e9)


def bar(x, lo, hi, ch, width=34):
    if np.isnan(x):
        return ""
    n = int(round((x - lo) / (hi - lo) * width))
    return ch * max(0, min(width, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--series", default="KXINXU")
    ap.add_argument("--grid", default="100ms")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--base-ms", type=int, default=500)   # baseline before t0
    ap.add_argument("--fwd-ms", type=int, default=1200)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    ap.add_argument("--max-tte-min", type=float, default=0, help="keep only moves within this many minutes of settlement (0=all)")
    ap.add_argument("--min-tte-min", type=float, default=0, help="keep only moves at least this many minutes from settlement")
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step))

    es = load_future(a.dir, a.future).resample(a.grid).last().ffill()
    by_ev = load_strikes_by_event(a.dir, a.series)
    L = pd.concat([implied_level(s, a.grid) for s in by_ev.values() if len(s) >= 4]).sort_index()
    L = L[~L.index.duplicated(keep="last")]
    j = pd.concat({"es": es, "L": L}, axis=1).sort_index().ffill().dropna()

    loges = np.log(j["es"]).to_numpy()
    Lv = j["L"].to_numpy()
    mv = (np.log(j["es"]) - np.log(j["es"].shift(win)))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = j.index.view("int64")

    settles = sorted(s for s in (settle_ns(e) for e in by_ev) if s)
    def tte_min(t):
        nxt = [s for s in settles if s > t]
        return (min(nxt) - t) / 6e10 if nxt else 1e9

    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            tte = tte_min(ii[k])
            if (a.max_tte_min == 0 or tte <= a.max_tte_min) and tte >= a.min_tte_min:
                events.append((k, sgn[k]))
            last = ii[k]
        prev = on
    b = round(a.base_ms / step)
    offs = list(range(-b, round(a.fwd_ms / step) + 1))
    E = {o: [] for o in offs}; K = {o: [] for o in offs}
    for pos, s in events:
        if pos - b < 0 or pos + offs[-1] >= len(Lv):
            continue
        e0 = loges[pos - b]; l0 = Lv[pos - b]; lb = Lv[pos - b]
        for o in offs:
            E[o].append(s * (loges[pos + o] - e0) * 1e4)
            K[o].append(s * (Lv[pos + o] - l0) / lb * 1e4)
    em = {o: np.mean(E[o]) for o in offs}
    km = {o: np.mean(K[o]) for o in offs}
    hi = max(max(em.values()), max(km.values())) * 1.05
    lo = min(0, min(km.values()))
    print(f"{a.future} vs {a.series}:  >={a.bps}bps within {a.win_ms}ms  ->  {len(events)} events")
    print(f"cumulative move from baseline (t0-{a.base_ms}ms), bps.  E=ES  K=Kalshi level\n")
    print(f"  {'t (rel t0)':>11} {'ES':>6} {'Kalshi':>7}   {'ES (E) / Kalshi (K)':<36}")
    for o in offs:
        mark = "  <- t0 (ES move done)" if o == 0 else ""
        be = bar(em[o], lo, hi, "E"); bk = bar(km[o], lo, hi, "K")
        print(f"  {o*step:+7.0f}ms {em[o]:6.2f} {km[o]:7.2f}   E:{be}")
        print(f"  {'':>11} {'':>6} {'':>7}   K:{bk}{mark}")


if __name__ == "__main__":
    main()
