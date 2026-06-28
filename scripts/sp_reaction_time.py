#!/usr/bin/env python3
"""Is Kalshi's ~600ms reaction event-driven (MM watches ES) or cadence-driven (MM
refreshes on its own clock)? Test: does the time from an ES move to the next Kalshi
near-money quote change DEPEND on the ES move size?

  constant across move sizes  => cadence-driven  => stale quote rests, tradeable
  shrinks for big moves       => event-driven    => MM pulls fast, not tradeable
"""
import argparse, glob, json, re
import numpy as np
import pandas as pd


def load_es(dir, inst):
    ts, mid = [], []
    for f in glob.glob(f"{dir}/stream=book/venue=databento/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            if b.get("instrument") != inst or not b["bids"] or not b["asks"]:
                continue
            ts.append(e["ts_ns"]); mid.append((b["bids"][0][0] + b["asks"][0][0]) / 2)
    o = np.argsort(ts)
    return np.array(ts)[o], np.array(mid)[o]


def load_kalshi_changes(dir, event):
    """ts (ns) of every near-money (last mid in 0.2-0.8) strike mid-change."""
    per = {}
    for f in glob.glob(f"{dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            inst = b.get("instrument", "")
            if event not in inst or not inst.endswith(".YES") or not b["bids"] or not b["asks"]:
                continue
            m = re.search(r"-T(\d+\.?\d*)", inst)
            if not m:
                continue
            mid = (b["bids"][0][0] + b["asks"][0][0]) / 2
            per.setdefault(m.group(1), []).append((e["ts_ns"], mid))
    chg = []
    for k, v in per.items():
        v.sort()
        for i in range(1, len(v)):
            if v[i][1] != v[i - 1][1] and 0.2 <= v[i][1] <= 0.8:
                chg.append(v[i][0])
    return np.array(sorted(chg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-cme-collect")
    ap.add_argument("--future", default="databento.ES")
    ap.add_argument("--event", default="KXINXU-26JUN25H1500")
    ap.add_argument("--win-ms", type=int, default=500)
    a = ap.parse_args()
    ets, emid = load_es(a.dir, a.future)
    K = load_kalshi_changes(a.dir, a.event)
    print(f"{a.future} vs {a.event}: ES ticks={len(ets):,}  Kalshi near-money mid-changes={len(K):,}")
    d = np.diff(K) / 1e6
    print(f"baseline Kalshi near-money change interval: mean={d.mean():.0f}ms  median={np.median(d):.0f}ms  (rate {1000/max(d.mean(),1e-9):.1f}/s)\n")

    # sample ES move on a 100ms clock: trailing win-ms return in bps
    s = pd.Series(emid, index=pd.to_datetime(ets, utc=True))
    s = s[~s.index.duplicated(keep="last")].sort_index().resample("100ms").last().ffill().dropna()
    win = max(1, a.win_ms // 100)
    mv = (np.log(s) - np.log(s.shift(win))).abs() * 1e4
    t0s = np.asarray(s.index.view("int64"))
    mvv = mv.to_numpy()

    # for each sample, time to next Kalshi change
    pos = np.searchsorted(K, t0s, side="right")
    react = np.where(pos < len(K), (K[np.minimum(pos, len(K) - 1)] - t0s) / 1e6, np.nan)

    buckets = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 99)]
    print(f"  {'ES 500ms move':>16} {'n':>6} {'median react(ms)':>16} {'mean react(ms)':>15}")
    for lo, hi in buckets:
        m = (mvv >= lo) & (mvv < hi) & ~np.isnan(react)
        if m.sum() > 5:
            r = react[m]
            print(f"  {f'{lo}-{hi} bps':>16} {m.sum():>6} {np.median(r):>16.0f} {np.mean(r):>15.0f}")


if __name__ == "__main__":
    main()
