"""Replay a SHADOW QUOTER against the recorded Kalshi book and the real trade tape. No orders.

Inputs
  book    data/xmarket/books/*.jsonl  (book_recorder.py: WS orderbook snapshot + every delta)
  trades  data/xmarket/flow/kalshi_<ticker>.csv  (maker_flow_probe.py collect: REST tape with
          taker side, price, size)

Kalshi book model: the YES side lists bids to buy YES; the NO side lists bids to buy NO, so the
best YES ask = 1 - best NO bid. Prices move in 0.1c ticks.

Policy simulated
  The quoter keeps a YES bid one tick ABOVE the best YES bid and a YES ask one tick BELOW the best
  YES ask, whenever that leaves at least --min-spread between its own two quotes (it never
  crosses). Being strictly better than everyone, it is first in line: every taker trade on a side
  fills it first, up to --size contracts per trade, then it requotes. It stops quoting the side
  that would push |inventory| past --max-pos.
  This is optimistic about speed (it requotes the instant the book changes) and assumes nobody
  else races to the same improved price; it is pessimistic in that it never earns anything from
  joining existing levels. Kalshi makers pay no fee.

Outputs per day and in total: fills, contracts, average markout per contract after 15 minutes and
after 2 hours (fill price vs the later mid; positive = good), the final mark-to-market P&L
including inventory, and the peak inventory.
Usage: maker_replay.py [--ticker KXBTCVSGOLD-26] [--size 10] [--max-pos 50] [--min-spread 0.005]
"""
import bisect
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

OUT = "data/xmarket"
TICK = 0.001


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def book_timeline(ticker):
    """Rebuild the top of book after every recorded event -> arrays ts, best_bid, best_ask."""
    yes, no = {}, {}
    ts_l, bid_l, ask_l = [], [], []
    def records(path):
        # early recordings split each message across two lines (Kalshi messages end in a newline);
        # accumulate lines until the buffer parses
        buf = ""
        for line in open(path, encoding="utf-8"):
            buf += line
            try:
                yield json.loads(buf)
                buf = ""
            except Exception:
                if len(buf) > 2_000_000:
                    buf = ""
    for f in sorted(glob.glob(os.path.join(OUT, "books", "*.jsonl"))):
        for rec in records(f):
            m = rec.get("m", {})
            typ, msg = m.get("type"), m.get("msg", {})
            if msg.get("market_ticker") != ticker:
                continue
            if typ == "orderbook_snapshot":
                yes = {float(p): float(q) for p, q in msg.get("yes_dollars_fp", [])}
                no = {float(p): float(q) for p, q in msg.get("no_dollars_fp", [])}
                t = rec["rt"]
            elif typ == "orderbook_delta":
                side = yes if msg.get("side") == "yes" else no
                p, d = float(msg["price_dollars"]), float(msg["delta_fp"])
                side[p] = side.get(p, 0.0) + d
                if side[p] <= 1e-9:
                    side.pop(p, None)
                t = msg.get("ts_ms") or rec["rt"]
            else:
                continue
            if yes and no:
                ts_l.append(t / 1000.0)
                bid_l.append(max(yes))
                ask_l.append(1.0 - max(no))
    return np.array(ts_l), np.array(bid_l), np.array(ask_l)


def main():
    ticker = arg("--ticker", "KXBTCVSGOLD-26")
    size = arg("--size", 10.0, float)
    max_pos = arg("--max-pos", 50.0, float)
    min_spread = arg("--min-spread", 0.005, float)
    ts, bb, ba = book_timeline(ticker)
    if len(ts) < 100:
        print("not enough book data yet (%d events) — the recorder needs to run longer" % len(ts))
        return
    mid = 0.5 * (bb + ba)
    T = pd.read_csv(os.path.join(OUT, "flow", "kalshi_%s.csv" % ticker)).sort_values("ts")
    T = T[(T.ts >= ts[0]) & (T.ts <= ts[-1])]
    print("%s: book %d events over %.1f days (%s .. %s) | trades in span %d" % (
        ticker, len(ts), (ts[-1] - ts[0]) / 86400, time.strftime("%m-%d %H:%M", time.gmtime(ts[0])),
        time.strftime("%m-%d %H:%M", time.gmtime(ts[-1])), len(T)))

    def at(arr, t):
        i = bisect.bisect_right(ts, t) - 1
        return arr[i] if i >= 0 else np.nan

    pos, cash, peak = 0.0, 0.0, 0.0
    fills = []
    for r in T.itertuples():
        b, a = at(bb, r.ts), at(ba, r.ts)
        if not (0 < b < a < 1):
            continue
        my_bid, my_ask = b + TICK, a - TICK
        if my_ask - my_bid < min_spread - 1e-12:
            continue
        if r.taker_side == "yes":            # taker buys YES -> hits our ask (we sell YES)
            if pos <= -max_pos:
                continue
            q = min(size, r.count, pos + max_pos)
            px, sgn = my_ask, -1
        else:                                # taker sells YES -> hits our bid (we buy YES)
            if pos >= max_pos:
                continue
            q = min(size, r.count, max_pos - pos)
            px, sgn = my_bid, 1
        if q <= 0:
            continue
        pos += sgn * q
        cash -= sgn * q * px
        peak = max(peak, abs(pos))
        m15, m2h = at(mid, r.ts + 900), at(mid, r.ts + 7200)
        fills.append({"day": time.strftime("%m-%d", time.gmtime(r.ts)), "q": q,
                      "mk15": sgn * (m15 - px) if r.ts + 900 <= ts[-1] else np.nan,
                      "mk2h": sgn * (m2h - px) if r.ts + 7200 <= ts[-1] else np.nan})
    if not fills:
        print("no simulated fills")
        return
    F = pd.DataFrame(fills)
    wavg = lambda d, c: (d[c] * d.q)[d[c].notna()].sum() / d.q[d[c].notna()].sum() if d[c].notna().any() else np.nan
    print("\n%-6s %6s %9s %10s %10s" % ("day", "fills", "contracts", "mk15m_c", "mk2h_c"))
    for d, g in F.groupby("day"):
        print("%-6s %6d %9.0f %+10.2f %+10.2f" % (d, len(g), g.q.sum(), 100 * wavg(g, "mk15"), 100 * wavg(g, "mk2h")))
    final_mid = mid[-1]
    pnl = cash + pos * final_mid
    days = (ts[-1] - ts[0]) / 86400
    print("\nTOTAL: %d fills, %.0f contracts | markout 15m %+.2fc, 2h %+.2fc per contract" % (
        len(F), F.q.sum(), 100 * wavg(F, "mk15"), 100 * wavg(F, "mk2h")))
    print("mark-to-market P&L incl. inventory: %+.2f$ (%+.2f$/day) | end inventory %+.0f | peak |inventory| %.0f" % (
        pnl, pnl / max(days, 1e-9), pos, peak))


if __name__ == "__main__":
    main()
