"""Backtest the hidden-driver agent's verdicts against what prices actually did.

For every verdict in hidden_links.jsonl at least --age days old (default 3), using the
snapshot store:
  - mid of each leg at verdict time and now (or at +--horizon days if given)
  - predicted direction per leg from the verdict (fair vs market at the time: fair below
    market = the agent called the leg RICH = predicted the mid falls, and vice versa)
  - score = did the mid move toward the agent's fair value? weighted by the predicted gap.
Reports per-verdict lines and the aggregate: hit rate on direction, cents captured toward
fair vs cents predicted, and the same stats restricted to predicted gaps >= 1c (the calls
that would matter). This is a MARK-TO-MARKET test; final settlement scoring comes free
later since verdicts carry market ids.
Usage: hidden_eval.py [--age 3] [--horizon 7]
"""
import bisect
import json
import os
import sys
import time

import pandas as pd

OUT = "data/xmarket"


def main():
    age_d = 3.0
    if "--age" in sys.argv:
        age_d = float(sys.argv[sys.argv.index("--age") + 1])
    hor_d = None
    if "--horizon" in sys.argv:
        hor_d = float(sys.argv[sys.argv.index("--horizon") + 1])
    S = pd.read_csv(os.path.join(OUT, "snapshots.csv"))
    S = S[(S.yes_bid > 0) & (S.yes_ask > 0) & (S.yes_ask < 1)].copy()
    S["mid"] = 0.5 * (S.yes_bid + S.yes_ask)
    series = {}
    for (v, i), g in S.groupby(["venue", "id"]):
        g = g.sort_values("ts")
        series[(v, str(i))] = (g.ts.to_numpy(), g.mid.to_numpy())

    def mid_at(key, t):
        s = series.get(key)
        if s is None:
            return None
        j = bisect.bisect_right(s[0], t) - 1
        if j < 0 or t - s[0][j] > 6 * 3600:
            return None
        return float(s[1][j])

    now = time.time()
    rows = []
    for line in open(os.path.join(OUT, "hidden_links.jsonl"), encoding="utf-8"):
        try:
            j = json.loads(line)
        except Exception:
            continue
        v = j.get("verdict")
        if not v or now - j["t"] < age_d * 86400:
            continue
        t_eval = j["t"] + hor_d * 86400 if hor_d else now
        for leg, d in (("A", j["a"]), ("B", j["b"])):
            fair = v.get("fair_a" if leg == "A" else "fair_b")
            mkt = v.get("mkt_a" if leg == "A" else "mkt_b")
            if fair is None or mkt is None:
                continue
            key = (d["venue"], str(d["id"]))
            m1 = mid_at(key, min(t_eval, now))
            if m1 is None:
                continue
            gap = fair - mkt                       # predicted move (toward fair)
            realized = m1 - mkt
            rows.append({"t": j["t"], "leg": leg, "id": str(d["id"])[:30], "fair": fair, "mkt": mkt,
                         "now": m1, "gap_c": 100 * gap, "real_c": 100 * realized,
                         "hit": (gap == 0 and abs(realized) < 0.005) or (gap * realized > 0),
                         "conf": v.get("confidence", 0)})
    if not rows:
        print("nothing scoreable yet (verdicts younger than %.0f days or no snapshot coverage)" % age_d)
        return
    D = pd.DataFrame(rows)
    print("%-10s %-3s %-30s %6s %6s %6s %7s %7s %s" % ("date", "leg", "id", "mkt", "fair", "now", "pred_c", "real_c", "hit"))
    for r in D.itertuples():
        print("%-10s %-3s %-30s %6.3f %6.3f %6.3f %+7.1f %+7.1f %s" % (
            time.strftime("%m-%d", time.gmtime(r.t)), r.leg, r.id, r.mkt, r.fair, r.now, r.gap_c, r.real_c, "Y" if r.hit else "n"))
    print("\nall %d legs: direction hit %.0f%% | predicted %.1fc total, captured toward fair %.1fc" % (
        len(D), 100 * D.hit.mean(), D.gap_c.abs().sum(), (D.gap_c.apply(lambda x: 1 if x > 0 else -1) * D.real_c).sum()))
    big = D[D.gap_c.abs() >= 1.0]
    if len(big):
        print("legs with >=1c predicted gap: %d | hit %.0f%% | captured %.1fc of %.1fc predicted" % (
            len(big), 100 * big.hit.mean(), (big.gap_c.apply(lambda x: 1 if x > 0 else -1) * big.real_c).sum(), big.gap_c.abs().sum()))


if __name__ == "__main__":
    main()
