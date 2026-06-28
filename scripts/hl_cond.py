#!/usr/bin/env python3
"""3bps-conditioned comparison of BTC sources (data/hl_compare). Detect Binance-perp
>=bps/200ms moves; show each source's response trajectory (bps, signed by move dir)
relative to t0, plus the cross-correlation restricted to move windows. Answers: on
the big moves we'd actually trade, does HL perp track Binance perp, and when?
"""
import argparse
import numpy as np
import pandas as pd

DIR = "data/hl_compare"


def load(s, grid):
    df = pd.read_csv(f"{DIR}/{s}.csv", header=None, names=["ts", "mid"])
    ser = pd.Series(df["mid"].values, index=pd.to_datetime(df["ts"].values))
    return ser[~ser.index.duplicated(keep="last")].sort_index().resample(grid).last().ffill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="50ms")
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--refractory-ms", type=int, default=1500)
    a = ap.parse_args()
    step = pd.Timedelta(a.grid).total_seconds() * 1000
    win = max(1, round(a.win_ms / step))
    srcs = ["bp", "hl", "cb", "bus"]
    S = {k: load(k, a.grid) for k in srcs}
    hrs = (S["bp"].index[-1] - S["bp"].index[0]).total_seconds() / 3600

    bp = S["bp"]; lr = np.log(bp); mv = (lr - lr.shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(bp.index.view("int64"))
    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            events.append((ii[k], sgn[k])); last = ii[k]
        prev = on
    print(f"span={hrs:.2f}h  bp >={a.bps}bps/{a.win_ms}ms moves: {len(events)} ({len(events)/hrs:.0f}/h)\n")
    if len(events) < 3:
        print("too few events yet"); return

    offs = list(range(-6, 13))  # -300..+600ms @50ms
    resp = {s: {o: [] for o in offs} for s in srcs}
    for s in srcs:
        ser = S[s]; t = np.asarray(ser.index.view("int64")); m = ser.to_numpy()
        for t0, d in events:
            j0 = np.searchsorted(t, t0 - 3e8) - 1
            if j0 < 0:
                continue
            base = np.log(m[j0])
            for o in offs:
                jj = np.searchsorted(t, t0 + o * step * 1e6) - 1
                if jj >= 0:
                    resp[s][o].append(d * (np.log(m[jj]) - base) * 1e4)
    print("  response to bp move (bps, signed by dir; t0 = bp's tunneled detection)")
    print(f"  {'offset':>7} {'bp':>7} {'hl':>7} {'cb':>7} {'bus':>7}")
    for o in offs:
        row = "  ".join(f"{np.mean(resp[s][o]):+6.2f}" if resp[s][o] else "   -  " for s in srcs)
        print(f"  {o*step:+6.0f}ms {row}")

    # conditioned correlation vs bp, restricted to +/-400ms of a move
    ev = np.array([t for t, _ in events])
    print("\n  conditioned corr vs bp (move windows only), peak lag:")
    for s in ("hl", "cb", "bus"):
        j = pd.concat({"bp": S["bp"], "x": S[s]}, axis=1).dropna()
        idx = np.asarray(j.index.view("int64"))
        keep = np.zeros(len(idx), bool)
        pos = np.searchsorted(ev, idx)
        for k in (pos - 1, pos):
            kk = np.clip(k, 0, len(ev) - 1)
            keep |= np.abs(idx - ev[kk]) <= 4e8
        rb = np.log(j["bp"]).diff().to_numpy(); rx = np.log(j["x"]).diff().to_numpy()
        best = (None, -9)
        for L in range(-12, 13):
            y = np.roll(rx, -L)
            mm = keep & ~(np.isnan(rb) | np.isnan(y))
            if mm.sum() > 30 and rb[mm].std() > 0 and y[mm].std() > 0:
                c = float(np.corrcoef(rb[mm], y[mm])[0, 1])
                if c > best[1]:
                    best = (L * step, c)
        print(f"    {s} vs bp: peak corr {best[1]:+.3f} at lag {best[0]:+.0f}ms  (lag<0 => {s} leads bp)")


if __name__ == "__main__":
    main()
