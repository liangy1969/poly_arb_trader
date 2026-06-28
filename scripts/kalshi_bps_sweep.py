#!/usr/bin/env python3
"""Sweep the perp-move trigger threshold (bps) and show the PRE-bucket breakdown of
the forward Kalshi response for each. Loads + grids ONCE, then sweeps thresholds.

  py scripts/kalshi_bps_sweep.py --dir data/kalshi-arb-collect --grid 50ms --trig-s 0.2
"""
import argparse, glob
import numpy as np
import pandas as pd
from kalshi_arb_study import load_books, grid

HZ = [1.0, 2.0, 3.0, 5.0, 10.0]
PRE_BINS = [("pre<=0", -1e9, 0), ("0-1", 0, 1), ("1-3", 1, 3), ("3-10", 3, 10), (">10", 10, 1e9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/kalshi-arb-collect")
    ap.add_argument("--grid", default="50ms")
    ap.add_argument("--trig-s", type=float, default=0.2)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--bps", default="2,3,5,8,12")
    a = ap.parse_args()
    thresholds = [float(x) for x in a.bps.split(",")]

    btc = load_books(glob.glob(f"{a.dir}/stream=book/venue=binance/**/events.jsonl", recursive=True),
                     lambda s: s.startswith("binance."))
    kal = load_books(glob.glob(f"{a.dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True),
                     lambda s: s.endswith(".YES"))
    span_h = (btc["ts_ns"].max() - btc["ts_ns"].min()) / 3.6e12
    print(f"perp={len(btc):,} kalshi={len(kal):,} windows={kal['inst'].nunique()} span={span_h:.1f}h")

    bg = grid(btc, a.grid)
    step = pd.Timedelta(a.grid).total_seconds()
    kt = max(1, round(a.trig_s / step))
    hz_k = [max(1, round(h / step)) for h in HZ]
    cd = max(1, round(a.cooldown_s / step))

    # pre-build per-window aligned arrays + the perp-return (bps) array — ONCE.
    wins = []
    for inst, seg in kal.groupby("inst"):
        pg = grid(seg, a.grid)
        al = pd.concat({"mid": bg.reindex(pg.index), "pup": pg}, axis=1, sort=True).ffill().dropna()
        if len(al) < kt + max(hz_k) + 5:
            continue
        mid = al["mid"].to_numpy(); pup = al["pup"].to_numpy(); n = len(al)
        b = np.full(n, np.nan)
        b[kt:] = (np.log(mid[kt:]) - np.log(mid[:-kt])) * 1e4
        wins.append((pup, b))

    for thr in thresholds:
        rows, n_edges = [], 0
        for pup, b in wins:
            n = len(pup)
            edges = [i for i in range(kt + 1, n - max(hz_k))
                     if abs(b[i]) >= thr and abs(b[i - 1]) < thr]
            n_edges += len(edges)
            last = -10**9
            for i in edges:
                if i - last < cd:
                    continue
                last = i
                d = 1.0 if b[i] > 0 else -1.0
                rec = {"p0": pup[i] * 100, "dir": d, "pre": d * (pup[i] - pup[i - kt]) * 100}
                for h, hk in zip(HZ, hz_k):
                    rec[f"r{h}"] = d * (pup[i + hk] - pup[i]) * 100
                rows.append(rec)
        if not rows:
            print(f"\n=== bps>={thr:g} : {n_edges} edges -> 0 events ===")
            continue
        df = pd.DataFrame(rows)
        nlag = int((df["pre"] <= 0).sum()); nev = len(df)
        hdr = " ".join(f"{'+'+format(h,'g')+'s':>6}" for h in HZ)
        print(f"\n=== bps>={thr:g} : {n_edges} edges -> {nev} events "
              f"(LAG pre<=0: {100*nlag/nev:.0f}%) ===")
        print(f"    {'pre_bin':>8} {'n':>4} {'pre_c':>6}  {hdr}  {'dh@1s':>9}")
        for lab, lo, hi in PRE_BINS:
            s = df[(df["pre"] > lo) & (df["pre"] <= hi)]
            if not len(s):
                print(f"    {lab:>8} {0:>4}"); continue
            vals = " ".join(f"{s[f'r{h}'].mean():>+6.2f}" for h in HZ)
            nz = s["r1.0"][s["r1.0"] != 0]
            dh = f"{(nz>0).mean()*100:>3.0f}%(n{len(nz)})" if len(nz) else "--"
            sm = " *" if len(s) < 8 else ""
            print(f"    {lab:>8} {len(s):>4} {s['pre'].mean():>+6.2f}  {vals}  {dh:>9}{sm}")


if __name__ == "__main__":
    main()
