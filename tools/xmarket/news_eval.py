"""The honesty layer: score the news agent's calls against what markets actually did.

For every call in news_calls.jsonl older than --horizon minutes, find the named market's
mid at the call time and at call time + horizon from snapshots.csv, and score:
  hit = sign(mid_change) == called direction and |change| >= --min-move-c.
Reports precision by confidence bucket and by source feed, plus the base rate of same-size
moves across ALL snapshot markets in the same windows (the control — an agent that names
random markets scores the base rate). Nothing here trades; this is the gate any trading
use of the agent must pass.
Usage: news_eval.py [--horizon 60] [--min-move-c 1.0]
"""
import bisect
import csv
import json
import os
import sys
from collections import defaultdict

OUT = "data/xmarket"


def main():
    horizon_s = 3600
    if "--horizon" in sys.argv:
        horizon_s = int(60 * float(sys.argv[sys.argv.index("--horizon") + 1]))
    min_move = 0.01
    if "--min-move-c" in sys.argv:
        min_move = float(sys.argv[sys.argv.index("--min-move-c") + 1]) / 100.0
    # snapshot series per market
    series = defaultdict(list)
    with open(os.path.join(OUT, "snapshots.csv")) as f:
        for row in csv.DictReader(f):
            b, a = float(row["yes_bid"]), float(row["yes_ask"])
            if 0 < b < 1 and 0 < a < 1:
                series[(row["venue"], row["id"])].append((int(row["ts"]), 0.5 * (b + a)))
    for k in series:
        series[k].sort()

    def mid_at(key, t):
        s = series.get(key)
        if not s:
            return None
        i = bisect.bisect_right([x[0] for x in s], t) - 1
        if i < 0 or t - s[i][0] > 1800:
            return None
        return s[i][1]

    import time
    now = time.time()
    scored, hits, moves = 0, 0, []
    ctrl_moves = 0
    ctrl_total = 0
    for line in open(os.path.join(OUT, "news_calls.jsonl"), encoding="utf-8"):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if not j.get("calls") or now - j["t"] < horizon_s + 600:
            continue
        for c in j["calls"]:
            key = (c.get("venue"), c.get("id"))
            m0 = mid_at(key, j["t"])
            m1 = mid_at(key, j["t"] + horizon_s)
            if m0 is None or m1 is None:
                continue
            d = m1 - m0
            scored += 1
            ok = abs(d) >= min_move and ((d > 0) == (c.get("dir") == "up"))
            hits += ok
            moves.append((abs(d), ok, c.get("conf", 0)))
    # control: base rate of a >= min_move move in a random market over the same horizon
    for key, s in series.items():
        for i in range(0, len(s) - 1, 5):
            t0, m0 = s[i]
            m1 = mid_at(key, t0 + horizon_s)
            if m1 is None:
                continue
            ctrl_total += 1
            # directional coin-flip control: half credit for any qualifying move
            if abs(m1 - m0) >= min_move:
                ctrl_moves += 1
    if scored == 0:
        print("nothing scoreable yet (need calls older than the horizon AND snapshot coverage)")
        return
    base = 0.5 * ctrl_moves / max(ctrl_total, 1)
    print("scored calls: %d | hit (right direction, move >= %.1fc): %d (%.1f%%) | random-call control: %.1f%%" % (
        scored, 100 * min_move, hits, 100 * hits / scored, 100 * base))
    hi = [x for x in moves if x[2] >= 0.7]
    if hi:
        print("high-confidence (>=0.7): %d calls, %.1f%% hit" % (len(hi), 100 * sum(1 for x in hi if x[1]) / len(hi)))


if __name__ == "__main__":
    main()
