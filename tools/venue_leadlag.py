#!/usr/bin/env python3
"""Cross-venue lead-lag from the venue-latency probe (`data/venue_latency.csv`).

Method
------
1. Per series (venue x market x symbol): mid = (bid+ask)/2, then LOG mid so
   BTC (~$79k) and ETH (~$4k) are on one scale.
2. Resample every series onto ONE uniform grid by last-value (step `--step-ms`),
   so the ~20x difference in native update rate cannot decide the answer.
3. Delta series: d(t) = 1e4 * (logmid(t) - logmid(t - `--delta-ms`))  [bps].
4. Cross-correlate pairs over lags: xcorr(L) = corr(d_a[t], d_b[t+L]).
   The peak L is the lead: **L > 0 means A LEADS B by L ms**.

Clocks
------
Uses `exch_ts` by default, never `recv_ts`: recv ordering is OUR NETWORK
GEOMETRY, not event order (memory: venue-latency-geometry). `--clock recv` is
offered only to contrast the two — it measures deliverability from this box.

Two hard caveats, printed with the results:
  * cross-exchange NTP skew is ~40ms, so |lead| below that is NOT interpretable
    on exch_ts;
  * overlapping deltas (delta_ms > step_ms) induce MA autocorrelation, which
    inflates naive significance — the effective sample is ~n*step/delta.

Binance SPOT is dropped automatically on the exch clock: its `@bookTicker`
payload carries no exchange timestamp (the futures one does).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import os
import sys

import numpy as np


def load(path, clock, t0_ns, t1_ns):
    """Stream the CSV -> {series: (ts_ns asc, logmid)}. Pure-python parse keeps
    memory flat; the file is ~1GB and pandas would copy it several times."""
    ts_i = 0 if clock == "recv" else 1
    acc = collections.defaultdict(lambda: ([], []))
    n = kept = bad = 0
    with open(path) as fh:
        next(fh)
        for line in fh:
            n += 1
            p = line.rstrip("\n").split(",")
            if len(p) != 7:
                continue
            t = p[ts_i]
            if not t:
                bad += 1  # e.g. binance spot on the exch clock
                continue
            t = int(t)
            if t < t0_ns or t > t1_ns:
                continue
            try:
                bid = float(p[5]); ask = float(p[6])
            except ValueError:
                continue
            if not (bid > 0 and ask > 0 and ask >= bid):
                continue
            k = (p[2], p[3], p[4])
            a = acc[k]
            a[0].append(t)
            a[1].append(0.5 * (bid + ask))
            kept += 1
    out = {}
    for k, (t, m) in acc.items():
        t = np.asarray(t, dtype=np.int64)
        m = np.log(np.asarray(m, dtype=np.float64))
        o = np.argsort(t, kind="stable")  # venues can emit slightly out of order
        out[k] = (t[o], m[o])
    return out, n, kept, bad


def grid_resample(ts, val, grid):
    """Last observed value at each grid point; NaN before the first sample."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, len(val) - 1)], np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/venue_latency.csv")
    ap.add_argument("--clock", choices=["exch", "recv"], default="exch")
    ap.add_argument("--step-ms", type=int, default=100, help="resample grid")
    ap.add_argument("--delta-ms", type=int, default=1000, help="price-change horizon")
    ap.add_argument("--max-lag-ms", type=int, default=1000)
    ap.add_argument("--from", dest="t_from", default="", help="UTC HH:MM (inclusive)")
    ap.add_argument("--to", dest="t_to", default="", help="UTC HH:MM (exclusive)")
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--symbol", default="", help="restrict to one base, e.g. BTC")
    ap.add_argument("--tail-pct", type=float, default=0.0,
                    help="keep only ticks whose CONDITIONING |delta| is at or above this "
                         "percentile (e.g. 99). 0 = use every tick.")
    ap.add_argument("--tail-ref", choices=["median", "max", "pair"], default="median",
                    help="what defines a big-move tick. 'median'/'max' = cross-sectional "
                         "stat over ALL series at t (symmetric wrt any pair -- the default, "
                         "because conditioning on one side's own |delta| manufactures a lead "
                         "for that side). 'pair' = union of the two sides' own tails.")
    a = ap.parse_args()

    t0_ns, t1_ns = -(2**62), 2**62
    if a.t_from or a.t_to:
        with open(a.csv) as fh:
            next(fh)
            day = int(next(fh).split(",", 1)[0]) // 86_400_000_000_000
        base = day * 86_400
        if a.t_from:
            h, m = a.t_from.split(":")
            t0_ns = (base + int(h) * 3600 + int(m) * 60) * 10**9
        if a.t_to:
            h, m = a.t_to.split(":")
            t1_ns = (base + int(h) * 3600 + int(m) * 60) * 10**9

    S, n, kept, bad = load(a.csv, a.clock, t0_ns, t1_ns)
    if a.symbol:
        S = {k: v for k, v in S.items() if k[2].upper().startswith(a.symbol.upper())}
    S = {k: v for k, v in S.items() if len(v[0]) >= a.min_rows}
    if len(S) < 2:
        sys.exit("need >=2 series with enough rows; got %d" % len(S))

    lo = max(v[0][0] for v in S.values())
    hi = min(v[0][-1] for v in S.values())
    step = a.step_ms * 10**6
    grid = np.arange(lo, hi, step, dtype=np.int64)
    f = lambda x: dt.datetime.utcfromtimestamp(x / 1e9).strftime("%H:%M:%S")
    print("venue lead-lag  |  clock=%s  step=%dms  delta=%dms  span %s..%s (%.1f min)"
          % (a.clock, a.step_ms, a.delta_ms, f(lo), f(hi), (hi - lo) / 6e10))
    print("parsed %s rows, used %s%s\n"
          % (format(n, ","), format(kept, ","),
             ", dropped %s with no %s_ts" % (format(bad, ","), a.clock) if bad else ""))

    k_off = max(1, a.delta_ms // a.step_ms)
    D, names = {}, []
    print("%-9s %-5s %-14s %9s %10s %9s" % ("venue", "mkt", "symbol", "updates", "upd/s", "sd(bps)"))
    for k in sorted(S):
        ts, lm = S[k]
        g = grid_resample(ts, lm, grid)
        d = np.full_like(g, np.nan)
        d[k_off:] = 1e4 * (g[k_off:] - g[:-k_off])
        D[k] = d
        names.append(k)
        fin = np.isfinite(d)
        print("%-9s %-5s %-14s %9s %10.1f %9.3f"
              % (k + (format(len(ts), ","), len(ts) / ((hi - lo) / 1e9), np.nanstd(d))))

    # ---- conditioning mask: restrict to the biggest cross-venue moves ----
    # Selection is on a series that is NOT either side of the pair, so the
    # filter cannot itself create a lead. Conditioning on |dA| would guarantee
    # A looks like the leader (A is extreme at t by construction, B merely
    # correlates), which is the trap this avoids.
    keep = None
    if a.tail_pct > 0:
        M = np.vstack([np.abs(D[k]) for k in names])
        with np.errstate(invalid="ignore"):
            ref = np.nanmedian(M, axis=0) if a.tail_ref == "median" else np.nanmax(M, axis=0)
        if a.tail_ref == "pair":
            ref = None  # computed per pair below
        else:
            thr = np.nanpercentile(ref, a.tail_pct)
            keep = np.isfinite(ref) & (ref >= thr)
            print("\nTAIL FILTER: cross-sectional %s |delta| >= p%.4g  (= %.3f bps)"
                  % (a.tail_ref, a.tail_pct, thr))
            print("  %s of %s grid ticks kept (%.3f%%)"
                  % (format(int(keep.sum()), ","), format(len(keep), ","),
                     100 * keep.mean()))

    lags = np.arange(-a.max_lag_ms, a.max_lag_ms + 1, a.step_ms) // a.step_ms
    print("\nPAIRWISE  xcorr(L) = corr(dA[t], dB[t+L]);  L>0 => A LEADS B")
    print("%-24s %-24s %9s %9s %9s %9s" % ("A", "B", "peak L", "corr@peak", "corr@0", "n@peak"))
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = D[names[i]], D[names[j]]
            kp = keep
            if a.tail_pct > 0 and a.tail_ref == "pair":
                tx = np.nanpercentile(np.abs(x), a.tail_pct)
                ty = np.nanpercentile(np.abs(y), a.tail_pct)
                kp = (np.abs(x) >= tx) | (np.abs(y) >= ty)
            best = (None, -2.0)
            c0 = np.nan
            npk = 0
            for L in lags:
                if L >= 0:
                    xa, yb = x[: len(x) - L if L else None], y[L:]
                    ka = kp[: len(x) - L if L else None] if kp is not None else None
                else:
                    xa, yb = x[-L:], y[: len(y) + L]
                    ka = kp[-L:] if kp is not None else None
                m = np.isfinite(xa) & np.isfinite(yb)
                if ka is not None:
                    # mask anchored at the A-side index t, i.e. the tick where
                    # the big move happened -- NOT re-selected per lag.
                    m &= ka
                if m.sum() < 100:
                    continue
                sa, sb = xa[m].std(), yb[m].std()
                if sa == 0 or sb == 0:
                    continue
                c = float(np.corrcoef(xa[m], yb[m])[0, 1])
                if L == 0:
                    c0 = c
                if c > best[1]:
                    best = (int(L) * a.step_ms, c)
                    npk = int(m.sum())
            lbl = lambda k: "%s.%s.%s" % k
            rows.append((lbl(names[i]), lbl(names[j]), best[0], best[1], c0))
            print("%-24s %-24s %+9s %9.4f %9.4f %9s"
                  % (lbl(names[i]), lbl(names[j]),
                     "n/a" if best[0] is None else "%dms" % best[0], best[1], c0,
                     format(npk, ",")))

    eff = int(((hi - lo) / 1e9) / (a.delta_ms / 1000.0))
    print("\nCAVEATS")
    print("  * exch clocks differ across venues by ~40ms (NTP skew, 2026-08-18 study):")
    print("    a |lead| below ~40ms is NOT interpretable as a real lead.")
    if a.delta_ms > a.step_ms:
        print("  * deltas OVERLAP (%dms delta on a %dms grid) -> MA(%d) autocorrelation."
              % (a.delta_ms, a.step_ms, k_off))
        print("    Effective independent samples ~%s, not %s: naive p-values are far"
              % (format(eff, ","), format(len(grid), ",")))
        print("    too small. Re-run with --step-ms %d for non-overlapping deltas."
              % a.delta_ms)
    print("  * a conflated feed (fewer, batched updates) is SMOOTHER after")
    print("    last-value resampling and can show a spurious lag; compare the")
    print("    upd/s column before trusting any pair.")


if __name__ == "__main__":
    main()
