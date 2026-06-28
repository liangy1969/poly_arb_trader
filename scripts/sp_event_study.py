#!/usr/bin/env python3
"""Event study on the top-pct 500ms future moves: how does the Kalshi implied level
react around each big move? Confirms the lead AND measures the capturable edge.

Trigger: |500ms future move| >= the pct-th percentile (rising edge, refractory).
At t0 (move just completed) we align, in the move's direction:
  - the future path (should already have moved by t0)
  - the Kalshi implied-level path (if the future leads, it ramps up AFTER t0)
The level move from t0 -> t0+~700ms is the edge you capture by acting at t0.

  py scripts/sp_event_study.py --future databento.ES --series KXINXU --pct 99.9
"""
import argparse, glob, json, re
import numpy as np
import pandas as pd
from sp_implied_leadlag import implied_level, load_future, _dglob


def load_strikes_by_event(dir, series, date=""):
    d = {}
    for f in _dglob(dir, "kalshi", date):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            inst = b.get("instrument", "")
            if series not in inst or not inst.endswith(".YES") or not b["bids"] or not b["asks"]:
                continue
            m = re.search(rf"({series}-[0-9A-Z]+)-T(\d+\.?\d*)", inst)
            if not m:
                continue
            d.setdefault(m.group(1), {}).setdefault(float(m.group(2)), []).append(
                (e["ts_ns"], (b["bids"][0][0] + b["asks"][0][0]) / 2))
    return d


def atm_slope(strikes_by_event):
    """median |dP / d$| near P=0.5 across events (for $ -> cents translation)."""
    sl = []
    for strikes in strikes_by_event.values():
        ks = sorted(strikes)
        last = {k: strikes[k][-1][1] for k in ks}
        for i in range(len(ks) - 1):
            if last[ks[i]] >= 0.4 and last[ks[i + 1]] <= 0.6 and ks[i + 1] > ks[i]:
                sl.append(abs(last[ks[i + 1]] - last[ks[i]]) / (ks[i + 1] - ks[i]))
    return float(np.median(sl)) if sl else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--series", default="KXINXU")
    ap.add_argument("--grid", default="100ms")
    ap.add_argument("--win-ms", type=int, default=500)
    ap.add_argument("--pct", type=float, default=99.9)
    ap.add_argument("--bps", type=float, default=0.0, help="absolute move threshold in bps (overrides --pct)")
    ap.add_argument("--refractory-ms", type=int, default=2000)
    ap.add_argument("--date", default="", help="restrict to one date partition, e.g. 2026-06-26")
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step))

    es = load_future(a.dir, a.future, a.date).resample(a.grid).last().ffill()
    by_ev = load_strikes_by_event(a.dir, a.series, a.date)
    Ls = [implied_level(s, a.grid) for s in by_ev.values() if len(s) >= 4]
    L = pd.concat(Ls).sort_index()
    L = L[~L.index.duplicated(keep="last")]
    j = pd.concat({"es": es, "L": L}, axis=1).sort_index().ffill().dropna()
    slope = atm_slope(by_ev)

    loges = np.log(j["es"])
    mv = (loges - loges.shift(win))
    mag = (mv.abs() * 1e4).to_numpy()
    sign = np.sign(mv).to_numpy()
    thr = a.bps if a.bps > 0 else np.nanpercentile(mag, a.pct)
    ii = j.index.view("int64")
    Lv = j["L"].to_numpy(); Ev = loges.to_numpy()

    events = []
    last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= thr
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            events.append((k, sign[k])); last = ii[k]
        prev = on
    hrs = (j.index[-1] - j.index[0]).total_seconds() / 3600
    print(f"{a.future} vs {a.series}  span={hrs:.2f}h  grid={a.grid}  ATM slope={slope:.4f} P/$ (1$ move ~ {slope*100:.1f}c)")
    print(f"top {100-a.pct:.2f}% move threshold = {thr:.2f} bps  ->  {len(events)} distinct events ({len(events)/hrs:.0f}/h)\n")
    if len(events) < 4:
        print("too few events"); return

    offs = list(range(-round(a.win_ms / step) - 2, round(1500 / step) + 1))
    print(f"  {'offset':>7}  {'fut_move(bps)':>13}  {'Kalshi_L($)':>11}  {'~edge(c)':>8}")
    for o in offs:
        e_, l_ = [], []
        for pos, s in events:
            k = pos + o
            if 0 <= k < len(Lv):
                e_.append(s * (Ev[k] - Ev[pos]) * 1e4)
                l_.append(s * (Lv[k] - Lv[pos]))
        if e_:
            print(f"  {o*step:+6.0f}ms  {np.mean(e_):+13.2f}  {np.mean(l_):+11.3f}  {np.mean(l_)*slope*100:+8.1f}")


if __name__ == "__main__":
    main()
