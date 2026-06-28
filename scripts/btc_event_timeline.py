#!/usr/bin/env python3
"""Per-event (not averaged) timeline of each perp >=bps move: perp path (bps) + the
active Kalshi market's P(up) path (cents), signed by move dir, with time-to-settle.
Event by event, so you see perp lead and quote follow per move (and near-settle vs not).
  py scripts/btc_event_timeline.py --dir data/leadlag-btc-tokyo --bps 3
"""
import argparse, re
import numpy as np
import pandas as pd
from btc_perp_vs_kalshi import load_perp, load_kalshi_books

MON = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
       "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def settle_ns(inst):
    m = re.search(r"KXBTC15M-(\d{2})([A-Z]{3})(\d{2})(\d{2})(\d{2})", inst)
    if not m:
        return None
    yy, mon, dd, hh, mm = m.groups()
    try:
        return pd.Timestamp(year=2000 + int(yy), month=MON[mon], day=int(dd),
                            hour=int(hh), minute=int(mm), tz="America/New_York").value
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-btc-tokyo")
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    perp = load_perp(a.dir).resample("50ms").last().ffill().dropna()
    books = load_kalshi_books(a.dir)
    win = max(1, a.win_ms // 50)
    mv = (np.log(perp) - np.log(perp).shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(perp.index.view("int64")); pmid = perp.to_numpy()
    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            events.append((ii[k], sgn[k], mag[k])); last = ii[k]
        prev = on

    offs = [-100, -50, 0, 50, 100, 150, 200, 300, 500]
    hdr = "  ".join(f"{o:+5d}" for o in offs)
    print(f"perp >={a.bps}bps/{a.win_ms}ms moves: {len(events)}  (cumulative from t0-100ms; offsets ms rel t0)\n")
    print(f"  {'':>20} {hdr}")
    for t0, d, m in events:
        best = None; bestn = 0
        for inst, (bt, bm) in books.items():
            j0 = np.searchsorted(bt, t0 - 3e8) - 1
            if j0 < 0 or not (0.05 <= bm[j0] <= 0.95):
                continue
            n = np.searchsorted(bt, t0 + 5e8) - np.searchsorted(bt, t0 - 3e8)
            if n > bestn:
                bestn = n; best = (inst, bt, bm, j0)
        ts = pd.Timestamp(t0, tz="UTC").strftime("%H:%M:%S")
        tag = f"{ts} {'UP' if d > 0 else 'DN'} {m:.1f}bps"
        if best is None:
            print(f"  {tag}  (no ATM market)\n"); continue
        inst, bt, bm, j0 = best
        p0i = np.searchsorted(ii, t0 - 3e8) - 1
        pbase = np.log(pmid[p0i]); base = bm[j0]
        sN = settle_ns(inst); tte = (sN - t0) / 6e10 if sN else float("nan")
        short = inst.split("KXBTC15M-")[1].split(".")[0]
        prow = []; qrow = []
        for o in offs:
            pj = np.searchsorted(ii, t0 + o * 1e6) - 1
            prow.append(d * (np.log(pmid[pj]) - pbase) * 1e4 if pj >= 0 else float("nan"))
            j = np.searchsorted(bt, t0 + o * 1e6) - 1
            qrow.append(d * (bm[j] - base) * 100 if j >= 0 else float("nan"))
        print(f"  {tag}  {short} P0={bm[j0]:.2f} TTE={tte:.0f}min")
        print(f"  {'perp(bps)':>20} " + "  ".join(f"{x:+5.1f}" for x in prow))
        print(f"  {'Kalshi(c)':>20} " + "  ".join(f"{x:+5.1f}" for x in qrow))
        print()


if __name__ == "__main__":
    main()
