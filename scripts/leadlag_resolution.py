#!/usr/bin/env python3
"""Show the perp↔P(up) cross-correlation CURVE (not just the peak) at several grid
resolutions, so the lead-lag is reported with its actual uncertainty.

  py scripts/leadlag_resolution.py --dir data/kalshi-arb-collect
"""
import argparse, glob
import numpy as np
import pandas as pd
from kalshi_arb_study import load_books, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/kalshi-arb-collect")
    a = ap.parse_args()
    btc = load_books(glob.glob(f"{a.dir}/stream=book/venue=binance/**/events.jsonl", recursive=True),
                     lambda s: s.startswith("binance."))
    kal = load_books(glob.glob(f"{a.dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True),
                     lambda s: s.endswith(".YES"))
    span_h = (btc["ts_ns"].max() - btc["ts_ns"].min()) / 3.6e12
    # native inter-update spacing (the real resolution floor of each feed)
    for name, df in [("perp", btc), ("kalshi", kal)]:
        d = np.diff(np.sort(df["ts_ns"].to_numpy())) / 1e6
        print(f"{name}: n={len(df):,}  inter-update ms: p50={np.median(d):.1f} p90={np.quantile(d,.9):.1f}")
    print(f"span={span_h:.1f}h\n")

    for g in ["100ms", "50ms", "20ms", "10ms"]:
        step_ms = pd.Timedelta(g).total_seconds() * 1000
        bg = grid(btc, g)
        pg = grid(kal, g)
        j = pd.concat({"mid": bg, "pup": pg}, axis=1, sort=True).ffill().dropna()
        br = np.log(j["mid"]).diff()
        dp = j["pup"].diff()
        nz = (dp != 0).mean() * 100  # % of grid cells where P(up) actually changed
        span = int(round(300 / step_ms))  # scan ±300ms
        cc = [(L, br.corr(dp.shift(-L))) for L in range(-span, span + 1)]
        cc = [(L, c) for L, c in cc if not np.isnan(c)]
        pk = max(cc, key=lambda t: t[1])
        # curve near zero (±5 steps)
        near = {L: c for L, c in cc if abs(L) <= 5}
        curve = "  ".join(f"{int(L*step_ms):+d}:{near[L]:+.3f}" for L in sorted(near))
        print(f"grid={g:>5} (P(up) changes in {nz:4.1f}% of cells) | "
              f"PEAK corr={pk[1]:+.3f} @ {pk[0]*step_ms:+.0f}ms")
        print(f"   curve(ms:corr): {curve}")


if __name__ == "__main__":
    main()
