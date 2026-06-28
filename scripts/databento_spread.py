#!/usr/bin/env python3
"""Measure live bid-ask spread of BTC.v.0 (full) vs MBT.v.0 (micro) CME futures.
  $env:DATABENTO_API_KEY=...; py -u scripts/databento_spread.py [seconds]
"""
import os, sys, time, threading, collections, statistics
import databento as db

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 25
c = db.Live(key=os.environ["DATABENTO_API_KEY"])
SYMS = os.environ.get("DBN_SYMS", "BTC.v.0,MBT.v.0").split(",")
c.subscribe(dataset="GLBX.MDP3", schema="mbp-1", stype_in="continuous", symbols=SYMS)

spreads = collections.defaultdict(list)
mids = collections.defaultdict(list)
label = {}

def run():
    for r in c:
        cn = type(r).__name__
        if "SymbolMapping" in cn:
            iid = getattr(r, "instrument_id", None)
            label[iid] = getattr(r, "stype_in_symbol", None) or getattr(r, "stype_out_symbol", None)
            continue
        l = getattr(r, "levels", None)
        if l and l[0].bid_px < 9e18 and l[0].ask_px < 9e18:
            iid = getattr(r, "instrument_id", None)
            bid, ask = l[0].bid_px / 1e9, l[0].ask_px / 1e9
            if ask > bid:
                spreads[iid].append(ask - bid)
                mids[iid].append((bid + ask) / 2)

threading.Thread(target=run, daemon=True).start()
time.sleep(SECS)

print(f"\nspread over {SECS}s (tick = $5.00):")
print(f"  {'symbol':>10} {'n':>5} {'spr_med$':>9} {'spr_mean$':>9} {'spr_min$':>8} {'ticks_med':>9} {'spr_bps':>8} {'mid':>9}")
for iid in sorted(spreads, key=lambda k: -len(spreads[k])):
    s, m = spreads[iid], mids[iid]
    sym = label.get(iid, str(iid))
    med = statistics.median(s)
    mid = statistics.median(m)
    print(f"  {sym:>10} {len(s):>5} {med:>9.2f} {statistics.mean(s):>9.2f} {min(s):>8.2f} "
          f"{med/5:>9.1f} {med/mid*1e4:>8.1f} {mid:>9.0f}")
