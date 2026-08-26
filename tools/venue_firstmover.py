#!/usr/bin/env python3
"""Per-EVENT first-mover census across venues (`data/venue_latency.csv`).

Correlation answers "who leads ON AVERAGE". This answers the different
question: on any GIVEN large move, which feed printed it first, and how often
does each feed win? A venue with no average edge can still be first sometimes,
and that is what a latency-sensitive consumer would exploit.

Method
------
1. Common last-value grid over all series (step `--step-ms`), log-mid.
2. Events: ticks where the cross-sectional MEDIAN |move| over `--horizon-ms`
   exceeds its `--pct` percentile. Median (not any single venue) so the event
   definition is symmetric — conditioning on one venue's own move would hand
   that venue the win by construction. Events are de-overlapped: once one
   fires, the next `--horizon-ms` is skipped.
3. For each event, the consensus move is the median signed move across feeds.
   For each feed, find the FIRST grid tick at which it has covered
   `--cross-frac` of that consensus move, measured from the pre-event price.
4. Rank feeds by that crossing time. Report win share, mean rank, and the
   median lead of each feed over the field.

Uses recv_ts by default: it is clock-free (our box stamps every series), so
"who arrived first" is answered without trusting any venue's clock.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt

import numpy as np

from venue_leadlag import grid_resample, load  # same parsing/resampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/venue_latency.csv")
    ap.add_argument("--clock", choices=["exch", "recv"], default="recv")
    ap.add_argument("--step-ms", type=int, default=10)
    ap.add_argument("--horizon-ms", type=int, default=200, help="move window defining an event")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--cross-frac", type=float, default=0.5, help="fraction of the move to call it")
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--min-rows", type=int, default=500)
    ap.add_argument("--max-premove", type=float, default=0.10,
                    help="skip an event unless EVERY feed moved less than this "
                         "fraction of the consensus move in the horizon BEFORE the "
                         "window opens. Guarantees the window contains the move; "
                         "0 disables (and reintroduces the mis-centring bias).")
    a = ap.parse_args()

    S, n, kept, bad = load(a.csv, a.clock, -(2**62), 2**62)
    S = {k: v for k, v in S.items() if k[2].upper().startswith(a.symbol.upper())}
    S = {k: v for k, v in S.items() if len(v[0]) >= a.min_rows}
    names = sorted(S)
    lo = max(v[0][0] for v in S.values())
    hi = min(v[0][-1] for v in S.values())
    step = a.step_ms * 10**6
    grid = np.arange(lo, hi, step, dtype=np.int64)
    G = np.vstack([grid_resample(*S[k], grid) for k in names])  # log-mid, (nfeed, ngrid)

    h = max(1, a.horizon_ms // a.step_ms)
    mv = np.full_like(G, np.nan)
    mv[:, h:] = 1e4 * (G[:, h:] - G[:, :-h])  # signed move over the horizon, bps
    with np.errstate(invalid="ignore"):
        cons = np.nanmedian(mv, axis=0)  # consensus signed move
    absc = np.abs(cons)
    thr = np.nanpercentile(absc, a.pct)

    f = lambda x: dt.datetime.utcfromtimestamp(x / 1e9).strftime("%H:%M:%S")
    print("first-mover census | clock=%s step=%dms horizon=%dms cross=%.0f%% | %s..%s"
          % (a.clock, a.step_ms, a.horizon_ms, 100 * a.cross_frac, f(lo), f(hi)))
    print("feeds: %s" % ", ".join("%s.%s.%s" % k for k in names))
    print("event threshold: |consensus move| >= p%.4g = %.3f bps\n" % (a.pct, thr))

    cand = np.where(np.isfinite(absc) & (absc >= thr))[0]
    events, last = [], -(10**9)
    for t in cand:
        if t - last >= h:  # de-overlap
            events.append(t)
            last = t
    wins = collections.Counter()          # OUTRIGHT (sole earliest) only
    shared = collections.defaultdict(float)  # ties split evenly
    ties = collections.Counter()
    ranks = collections.defaultdict(list)
    cross_t = collections.defaultdict(list)
    best_of_gain = collections.defaultdict(list)
    drop_by = collections.Counter()
    dropped = premoved = 0
    used = 0
    for t in events:
        t0 = t - h
        target = cons[t] * a.cross_frac

        # --- WINDOW-CENTRING GUARD ---------------------------------------
        # `seg` is measured from each feed's OWN price at t0, so seg[0] == 0
        # always and a feed can never "cross at 0ms". The real hazard is the
        # opposite: a feed that moved BEFORE t0 has that move absorbed into
        # its own baseline, so it looks SLOW or never crosses at all. Measured
        # on BTC, the p90 pre-window move was 0.89 of the consensus for
        # binance perp and 0.98-0.99 for okx perp/spot — i.e. on the worst
        # decile the move had already happened. Require every feed to be quiet
        # before t0 so the window provably contains the move being ranked.
        if a.max_premove > 0:
            if t0 - h < 0:
                continue
            pre = []
            for i in range(len(names)):
                if not (np.isfinite(G[i, t0]) and np.isfinite(G[i, t0 - h])):
                    pre = None
                    break
                pre.append(abs(1e4 * (G[i, t0] - G[i, t0 - h]) / cons[t]))
            if pre is None or max(pre) > a.max_premove:
                premoved += 1
                continue

        first = {}
        for i, k in enumerate(names):
            base = G[i, t0]
            if not np.isfinite(base):
                continue
            seg = 1e4 * (G[i, t0 : t + h + 1] - base)
            ok = seg >= target if cons[t] > 0 else seg <= target
            w = np.flatnonzero(np.isfinite(seg) & ok)
            if len(w):
                first[k] = int(w[0]) * a.step_ms
        if len(first) < len(names):
            # Dropping is necessary (a partial field cannot be ranked) but it
            # is NOT random: on BTC, binance SPOT caused 156 of 249 drops
            # because it need not cover half a perp-inclusive consensus move.
            # Attribute every drop so the selection bias is visible, never
            # silent.
            dropped += 1
            for k in names:
                if k not in first:
                    drop_by[k] += 1
            continue
        used += 1
        order = sorted(first, key=lambda k: first[k])
        # TIES ARE COMMON (~35% of events at a 10ms grid) and must NOT go to
        # whichever key sorts first -- Python's sort is stable, so `order[0]`
        # silently handed every tie to the alphabetically-first feed and
        # inflated its win rate by ~2x. Count outright wins separately from
        # ties, and split tied credit evenly.
        mn = min(first.values())
        winners = [k for k, v in first.items() if v == mn]
        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            ties[len(winners)] += 1
        for k in winners:
            shared[k] += 1.0 / len(winners)
        # competition ranking: tied feeds get the same rank
        r = 0
        for i, k in enumerate(order):
            if i and first[k] > first[order[i - 1]]:
                r = i
            ranks[k].append(r + 1)
        med = np.median(list(first.values()))
        for k, v in first.items():
            cross_t[k].append(v - med)
            # how much earlier the field's first print is than THIS feed
            best_of_gain[k].append(v - first[order[0]])

    print("events: %s detected | %s skipped as pre-moved (>%.0f%% of the move "
          "already made before the window) | %s dropped (a feed never crossed) "
          "| %s USED"
          % (format(len(events), ","), format(premoved, ","), 100 * a.max_premove,
             format(dropped, ","), format(used, ",")))
    if dropped:
        # Drops are NOT random -- surface who caused them so the selection
        # bias is visible rather than silent.
        print("  drops attributed to: %s"
              % ", ".join("%s.%s.%s x%d" % (k + (v,)) for k, v in drop_by.most_common() if v))
    print()
    if not used:
        return
    nt = sum(ties.values())
    print("ties for earliest tick: %s of %s events (%.1f%%) -- split evenly, "
          "NOT given to the first-sorted feed\n"
          % (format(nt, ","), format(used, ","), 100 * nt / used))
    print("%-24s %9s %9s %9s %11s %11s"
          % ("feed", "outright%", "shared%", "mean rank", "med vs field", "p10 vs field"))
    for k in sorted(names, key=lambda k: -shared[k]):
        d = np.array(cross_t[k])
        print("%-24s %8.1f%% %8.1f%% %9.2f %10.0fms %10.0fms"
              % ("%s.%s.%s" % k, 100 * wins[k] / used, 100 * shared[k] / used,
                 float(np.mean(ranks[k])), float(np.median(d)), float(np.percentile(d, 10))))
    print("\n'med vs field' = median (this feed's crossing time - median crossing time).")
    print("Negative = habitually early. 'p10 vs field' = how early on its BEST")
    print("decile, i.e. the occasions this feed is genuinely first.")

    # ---- what does consuming ALL feeds buy over the single best one? ----
    # This is the actionable number: a consumer that acts on whichever feed
    # prints first is early by this much versus one wired to a single venue.
    best_single = max(names, key=lambda k: shared[k])
    gain = np.array(best_of_gain[best_single])
    print("\nBEST-OF-ALL vs the single best feed (%s.%s.%s):" % best_single)
    print("  it is NOT first on %.0f%% of events" % (100 * (1 - shared[best_single] / used)))
    print("  taking whichever feed prints first is earlier by:")
    print("     median %.0fms   mean %.1fms   p75 %.0fms   p90 %.0fms   max %.0fms"
          % (np.median(gain), gain.mean(), np.percentile(gain, 75),
             np.percentile(gain, 90), gain.max()))


if __name__ == "__main__":
    main()
