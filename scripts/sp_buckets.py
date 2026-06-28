#!/usr/bin/env python3
"""Per current-P(yes)-bucket economics of the ES-lead trade. On each >=bps/win ES
move, EVERY strike's P(yes) shifts; we record each strike's cent move (edge) and
spread bucketed by its P(yes) at t0, then net against the Kalshi quadratic taker fee
(7*P*(1-P) cents/side). Finds which price bucket actually pays."""
import argparse, glob, json, re
import numpy as np
import pandas as pd
from sp_event_study import load_future
from sp_timeline import settle_ns
from sp_implied_leadlag import _series


def load_full(dir, series):
    d = {}
    for f in glob.glob(f"{dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True):
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
            bid, ask = b["bids"][0][0], b["asks"][0][0]
            d.setdefault(m.group(1), {}).setdefault(float(m.group(2)), []).append(
                (e["ts_ns"], (bid + ask) / 2, ask - bid))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--series", default="KXINXU")
    ap.add_argument("--grid", default="100ms")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--horizon-ms", type=int, default=600)
    ap.add_argument("--max-tte-min", type=float, default=0)
    ap.add_argument("--min-tte-min", type=float, default=0)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step)); H = round(a.horizon_ms / step)

    es = load_future(a.dir, a.future).resample(a.grid).last().ffill()
    by_ev = load_full(a.dir, a.series)
    obs = []  # (P_t0, edge_c, spread_c)
    nev = 0
    for E, strikes in by_ev.items():
        sE = settle_ns(E)
        if sE is None or len(strikes) < 4:
            continue
        ks = sorted(strikes)
        mid = pd.DataFrame({k: _series([(t, m) for t, m, _ in strikes[k]]).resample(a.grid).last() for k in ks}).sort_index().ffill()
        spr = pd.DataFrame({k: _series([(t, s) for t, _, s in strikes[k]]).resample(a.grid).last() for k in ks}).sort_index().ffill()
        idx = mid.index
        e = es.reindex(idx).ffill()
        mv = (np.log(e) - np.log(e).shift(win))
        mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
        Mv = mid.to_numpy(); Sv = spr.to_numpy(); ii = np.asarray(idx.view("int64"))
        last = None; prev = False
        for kk in range(len(mag)):
            on = mag[kk] >= a.bps
            if on and not prev and (last is None or ii[kk] - last >= a.refractory_ms * 1e6):
                last = ii[kk]
                tte = (sE - ii[kk]) / 6e10
                ok = (a.max_tte_min == 0 or tte <= a.max_tte_min) and tte >= a.min_tte_min
                if ok and kk + H < len(Mv):
                    nev += 1
                    for ci in range(len(ks)):
                        p0 = Mv[kk][ci]
                        if np.isnan(p0) or p0 <= 0.02 or p0 >= 0.98:
                            continue
                        edge = sgn[kk] * (Mv[kk + H][ci] - p0) * 100
                        if not np.isnan(edge):
                            obs.append((p0, edge, Sv[kk][ci] * 100))
            prev = on
    obs = np.array(obs)
    tag = f"TTE<={a.max_tte_min}" if a.max_tte_min else (f"TTE>={a.min_tte_min}" if a.min_tte_min else "all TTE")
    print(f"{a.future} vs {a.series}: >={a.bps}bps/{a.win_ms}ms, edge@+{a.horizon_ms}ms, {tag}")
    print(f"{nev} move-events -> {len(obs)} strike-observations\n")
    print(f"  {'P(yes)':>9} {'n':>5} {'edge(c)':>8} {'spread(c)':>9} {'fee/side':>8} {'net_taker':>9} {'net_maker':>9}")
    for lo in np.arange(0.05, 0.95, 0.1):
        hi = lo + 0.1
        m = (obs[:, 0] >= lo) & (obs[:, 0] < hi)
        if m.sum() < 5:
            continue
        P = (lo + hi) / 2
        fee = 7 * P * (1 - P)              # cents/side (taker quadratic fee)
        edge = obs[m, 1].mean(); spread = obs[m, 2].mean()
        net_t = edge - spread - 2 * fee    # taker entry + taker exit
        net_m = edge - fee                 # taker entry + maker exit (spreads ~cancel)
        print(f"  {lo:.2f}-{hi:.2f} {m.sum():>5} {edge:>8.2f} {spread:>9.2f} {fee:>8.2f} {net_t:>+9.2f} {net_m:>+9.2f}")


if __name__ == "__main__":
    main()
