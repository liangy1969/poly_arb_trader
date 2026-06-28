#!/usr/bin/env python3
"""E2E validation of the Databento BTC reference feed before building the Rust collector.

Confirms: API key works + entitled, GLBX.MDP3 / BTC.v.0 (volume-roll continuous front) /
mbp-1 streams live, BBO is sane, and reports Databento's exchange->gateway latency
(ts_recv - ts_event) plus the local-receive network jitter.

  $env:DATABENTO_API_KEY="db-..."; py scripts/databento_validate.py
"""
import os, sys, time, statistics

try:
    import databento as db
except ImportError:
    print("pip install databento"); sys.exit(1)

KEY = os.environ.get("DATABENTO_API_KEY")
if not KEY:
    print("set DATABENTO_API_KEY env var"); sys.exit(1)

DATASET = "GLBX.MDP3"
SCHEMA = os.environ.get("DBN_SCHEMA", "mbp-1")
STYPE = os.environ.get("DBN_STYPE", "continuous")
SYMS = os.environ.get("DBN_SYMS", "BTC.v.0").split(",")
SCALE = getattr(db, "FIXED_PRICE_SCALE", 1_000_000_000)
UNDEF = getattr(db, "UNDEF_PRICE", 9223372036854775807)
N, TIMEOUT_S = 40, 45

client = db.Live(key=KEY)
client.subscribe(dataset=DATASET, schema=SCHEMA, stype_in=STYPE, symbols=SYMS)
print(f"subscribed: {DATASET} {SCHEMA} stype_in={STYPE} {SYMS}")
print(f"waiting for live data (CME crypto = weekday hours, daily 16:00-17:00 CT break)...\n")

n = 0
gw_lat = []   # ts_recv - ts_event  (exchange -> Databento gateway, ns)
loc_gap = []  # local_recv - ts_recv (gateway -> here; offset-dominated, use the SPREAD)
sym_map = {}
t0 = time.time()
try:
    for rec in client:
        now_ns = time.time_ns()
        cls = type(rec).__name__
        # symbol mapping for the continuous resolution
        if "SymbolMapping" in cls:
            iid = getattr(rec, "instrument_id", None)
            out = getattr(rec, "stype_out_symbol", None) or getattr(rec, "raw_symbol", None)
            if iid is not None:
                sym_map[iid] = out
            continue
        if cls in ("SystemMsg", "ErrorMsg"):
            msg = getattr(rec, "msg", "")
            if cls == "ErrorMsg":
                print(f"  ERROR from gateway: {msg}")
            continue
        # mbp-1 data record
        levels = getattr(rec, "levels", None)
        if not levels:
            continue
        l = levels[0]
        if l.bid_px == UNDEF or l.ask_px == UNDEF:
            continue
        bid, ask = l.bid_px / SCALE, l.ask_px / SCALE
        ev = int(getattr(rec, "ts_event", 0) or rec.hd.ts_event)
        rv = int(getattr(rec, "ts_recv", 0))
        if ev and rv:
            gw_lat.append((rv - ev) / 1e6)   # ms
        if rv:
            loc_gap.append((now_ns - rv) / 1e6)  # ms (offset-dominated)
        iid = getattr(rec, "instrument_id", None) or rec.hd.instrument_id
        sym = sym_map.get(iid, iid)
        if n < 6 or n % 10 == 0:
            gl = f"{(rv-ev)/1e6:.3f}ms" if ev and rv else "n/a"
            print(f"  [{sym}] bid={bid:.1f} ask={ask:.1f} mid={(bid+ask)/2:.1f} sz={l.bid_sz}x{l.ask_sz} exch->gw={gl}")
        n += 1
        if n >= N:
            break
        if time.time() - t0 > TIMEOUT_S:
            print("  ...timeout (no/low data — CME daily break, weekend, or entitlement?)")
            break
finally:
    try: client.stop()
    except Exception: pass

print()
if n:
    print(f"PASS: received {n} mbp-1 records for BTC.v.0 -> resolved {sym_map}")
    if gw_lat:
        print(f"  exchange->gateway latency (ts_recv-ts_event): "
              f"min={min(gw_lat):.3f} med={statistics.median(gw_lat):.3f} max={max(gw_lat):.3f} ms")
    if len(loc_gap) > 2:
        sp = statistics.median(loc_gap)
        jit = statistics.pstdev(loc_gap)
        print(f"  gateway->here gap: median={sp:.1f}ms (NOTE: includes home NTP clock offset ~sec; "
              f"network JITTER stdev={jit:.1f}ms is the meaningful part)")
    print("  -> the Rust collector can use this exact dataset/symbol/schema.")
else:
    print("FAIL: no data. Check (a) CME hours, (b) live entitlement for GLBX.MDP3, (c) symbol BTC.v.0.")
