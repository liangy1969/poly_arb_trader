#!/usr/bin/env python3
"""Cross-venue lead-lag from the TRADER's own 50ms sampler.

Complements `venue_leadlag.py` (which reads the standalone probe). Two things
make this view different and worth having:

  * it covers the BRTI SETTLEMENT CONSTITUENTS -- coinbase, kraken, bitstamp,
    gemini -- which the probe never carried. KXBTC15M settles on a consolidated
    BRTI mid, so which of THOSE leads is the question that touches settlement,
    not just signal;
  * the sampler is already a synchronised 50ms grid stamped by one box, so
    there is no resampling step and no proxy asymmetry.

Cost: 50ms resolution, so it cannot resolve leads finer than one grid step.
The probe (10ms) is the tool for that.

Every series is a last-known carry, so `*_age_ms` says how stale each quote
is; `--max-age-ms` drops rows where a venue's quote is older than that, which
matters because a stale carry attenuates correlation and fakes a lag.

Convention, as in venue_leadlag.py: xcorr(L) = corr(dA[t], dB[t+L]);
**L > 0 means A LEADS B**.
"""
from __future__ import annotations

import argparse
import gzip

import numpy as np

# (name, bid_idx, ask_idx, age_idx) into the extracted column set
COLS = [
    ("perp", 1, 2, None),
    ("cb", 3, 4, 5),
    ("bspot", 6, 7, 8),
    ("kraken", 9, 10, 11),
    ("bitstamp", 12, 13, 14),
    ("gemini", 15, 16, 17),
    ("okx", 18, 19, 20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/sampler_venues_0826.csv.gz")
    ap.add_argument("--delta-ms", type=int, default=1000)
    ap.add_argument("--max-lag-ms", type=int, default=1000)
    ap.add_argument("--step-ms", type=int, default=50, help="sampler grid")
    ap.add_argument("--max-age-ms", type=int, default=2000,
                    help="drop a venue's quote when staler than this (-1 = keep all)")
    a = ap.parse_args()

    op = gzip.open if a.csv.endswith(".gz") else open
    rows = []
    with op(a.csv, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < 21:
                continue
            rows.append(p)
    n = len(rows)
    print("rows: %s" % format(n, ","))
    ts = np.array([int(r[0]) for r in rows], dtype=np.int64)

    def col(i):
        return np.array([float(r[i]) if r[i] not in ("", "NaN", "nan") else np.nan
                         for r in rows])

    mids, ages, names = {}, {}, []
    print("\n%-10s %10s %9s %11s %11s" % ("venue", "valid", "valid%", "med age", "p99 age"))
    for nm, bi, ai, gi in COLS:
        b, k = col(bi), col(ai)
        m = 0.5 * (b + k)
        m[~((b > 0) & (k > 0) & (k >= b))] = np.nan
        g = col(gi) if gi is not None else np.zeros(n)
        v = np.isfinite(m)
        if v.sum() < 1000:
            print("%-10s %10s %8.1f%%   (skipped: too few)" % (nm, format(int(v.sum()), ","),
                                                               100 * v.mean()))
            continue
        gv = g[v & np.isfinite(g) & (g >= 0)]
        print("%-10s %10s %8.1f%% %10.0fms %10.0fms"
              % (nm, format(int(v.sum()), ","), 100 * v.mean(),
                 np.median(gv) if len(gv) else -1,
                 np.percentile(gv, 99) if len(gv) else -1))
        mids[nm], ages[nm], _ = m, g, names.append(nm)

    # staleness gate: a carried quote older than max-age is not a fresh
    # observation and would attenuate the correlation toward a false lag.
    if a.max_age_ms >= 0:
        for nm in names:
            g = ages[nm]
            stale = np.isfinite(g) & (g > a.max_age_ms)
            mids[nm] = np.where(stale, np.nan, mids[nm])
        print("\nstaleness gate: quotes older than %dms dropped" % a.max_age_ms)

    k_off = max(1, a.delta_ms // a.step_ms)
    D = {}
    for nm in names:
        lm = np.log(mids[nm])
        d = np.full(n, np.nan)
        d[k_off:] = 1e4 * (lm[k_off:] - lm[:-k_off])
        D[nm] = d

    lags = np.arange(-a.max_lag_ms, a.max_lag_ms + 1, a.step_ms) // a.step_ms
    print("\nPAIRWISE  xcorr(L) = corr(dA[t], dB[t+L]);  L>0 => A LEADS B"
          "   [delta %dms, grid %dms]" % (a.delta_ms, a.step_ms))
    print("%-10s %-10s %9s %10s %10s %11s" % ("A", "B", "peak L", "corr@peak", "corr@0", "n"))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = D[names[i]], D[names[j]]
            best, c0, nb = (None, -2.0), np.nan, 0
            for L in lags:
                if L >= 0:
                    xa, yb = x[: n - L if L else None], y[L:]
                else:
                    xa, yb = x[-L:], y[: n + L]
                m = np.isfinite(xa) & np.isfinite(yb)
                if m.sum() < 500:
                    continue
                if xa[m].std() == 0 or yb[m].std() == 0:
                    continue
                c = float(np.corrcoef(xa[m], yb[m])[0, 1])
                if L == 0:
                    c0, nb = c, int(m.sum())
                if c > best[1]:
                    best = (int(L) * a.step_ms, c)
            print("%-10s %-10s %+9s %10.4f %10.4f %11s"
                  % (names[i], names[j],
                     "n/a" if best[0] is None else "%dms" % best[0],
                     best[1], c0, format(nb, ",")))
    print("\nNOTE 50ms grid: leads finer than one step are unresolvable here —")
    print("     use tools/venue_leadlag.py on probe data for 10ms resolution.")


if __name__ == "__main__":
    main()
