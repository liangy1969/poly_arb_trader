#!/usr/bin/env python3
"""How long does the Kalshi P(up) reaction ('pre' move) take? Event-aligned impulse
response: average cumulative perp move (bps) and token move (cents), signed in the
perp direction, vs offset from the trigger — on FULL-RES windows only (median Kalshi
spacing < 20ms; the 50ms-downsampled portion is excluded), at a 10ms grid.

  py scripts/kalshi_impulse.py --dir data/kalshi-arb-collect --bps 2
"""
import argparse, glob
import numpy as np
import pandas as pd
from kalshi_arb_study import load_books, grid

GRID = "10ms"
TRIG_S = 0.2
OFFS_MS = [-300, -200, -150, -100, -50, 0, 50, 100, 150, 200, 300, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/kalshi-arb-collect")
    ap.add_argument("--bps", type=float, default=2.0)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    a = ap.parse_args()

    btc = load_books(glob.glob(f"{a.dir}/stream=book/venue=binance/**/events.jsonl", recursive=True),
                     lambda s: s.startswith("binance."))
    kal = load_books(glob.glob(f"{a.dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True),
                     lambda s: s.endswith(".YES"))
    bg = grid(btc, GRID)
    step = pd.Timedelta(GRID).total_seconds()
    kt = round(TRIG_S / step)
    cd = round(a.cooldown_s / step)
    offs = [round(o / 1000 / step) for o in OFFS_MS]
    omin, omax = min(offs), max(offs)

    n_win_full = 0
    perp = {o: [] for o in offs}   # cumulative perp move (bps) from window start (i-kt)
    tok = {o: [] for o in offs}    # cumulative token move (cents) from window start
    n_ev = 0
    for inst, seg in kal.groupby("inst"):
        sp = np.median(np.diff(np.sort(seg["ts_ns"].to_numpy())) / 1e6) if len(seg) > 5 else 1e9
        if sp > 20:        # exclude the 50ms-downsampled portion → full-res only
            continue
        n_win_full += 1
        pg = grid(seg, GRID)
        al = pd.concat({"mid": bg.reindex(pg.index), "pup": pg}, axis=1, sort=True).ffill().dropna()
        mid = al["mid"].to_numpy(); pup = al["pup"].to_numpy(); n = len(al)
        if n < kt - omin + omax + 5:
            continue
        b = np.full(n, np.nan)
        b[kt:] = (np.log(mid[kt:]) - np.log(mid[:-kt])) * 1e4
        edges = [i for i in range(kt - omin, n - omax)
                 if abs(b[i]) >= a.bps and abs(b[i - 1]) < a.bps]
        last = -10**9
        for i in edges:
            if i - last < cd:
                continue
            last = i
            d = 1.0 if b[i] > 0 else -1.0
            base_mid, base_pup = mid[i - kt], pup[i - kt]
            for o in offs:
                perp[o].append(d * (np.log(mid[i + o] / base_mid)) * 1e4)
                tok[o].append(d * (pup[i + o] - base_pup) * 100)
            n_ev += 1

    print(f"full-res windows={n_win_full}  events(bps>={a.bps:g})={n_ev}  grid={GRID}  trig=200ms")
    print("(t=0 is trigger detection; window = [-200ms, 0]; baseline = P(up) at -200ms)\n")
    print(f"  {'offset':>8} {'perp_bps':>9} {'token_c':>8} {'tok %final':>11}")
    final = np.mean(tok[offs[OFFS_MS.index(300)]])  # token move at +300ms = 'final'
    for o, oms in zip(offs, OFFS_MS):
        pb = np.mean(perp[o]); tc = np.mean(tok[o])
        pct = 100 * tc / final if abs(final) > 1e-9 else float("nan")
        print(f"  {oms:>+6}ms {pb:>+9.2f} {tc:>+8.2f} {pct:>10.0f}%")
    # rise time: first offset where token reaches 50% / 90% of its +300ms move
    series = [(oms, np.mean(tok[o]) / final * 100 if abs(final) > 1e-9 else 0) for o, oms in zip(offs, OFFS_MS)]
    def first_reach(p):
        for oms, pc in series:
            if pc >= p:
                return oms
        return None
    print(f"\n  token reaches 50% of its move by {first_reach(50)}ms, 90% by {first_reach(90)}ms "
          f"(0ms = trigger; negative = within the trigger window)")


if __name__ == "__main__":
    main()
