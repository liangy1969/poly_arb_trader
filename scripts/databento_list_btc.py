#!/usr/bin/env python3
"""List the Bitcoin-related instruments available on Databento (CME GLBX.MDP3) so we
can pick a set to collect for the lead-lag-vs-Kalshi study.

  $env:DATABENTO_API_KEY="db-..."; py scripts/databento_list_btc.py
"""
import os, sys, datetime as dt
import databento as db

key = os.environ.get("DATABENTO_API_KEY")
if not key:
    print("set DATABENTO_API_KEY"); sys.exit(1)

h = db.Historical(key)

# 1) which datasets exist (any crypto-specific one?)
try:
    ds = h.metadata.list_datasets()
    print("datasets:", ds)
except Exception as e:
    print("list_datasets ERR:", e)

# 2) resolve CME Bitcoin futures parents -> their contracts
start = (dt.date.today() - dt.timedelta(days=4)).isoformat()
parents = ["BTC.FUT", "MBT.FUT", "BFF.FUT", "MIB.FUT", "BRR.FUT", "BTC.OPT"]
print(f"\nCME GLBX.MDP3 bitcoin parents (active near {start}):")
for p in parents:
    try:
        r = h.symbology.resolve(
            dataset="GLBX.MDP3", symbols=[p],
            stype_in="parent", stype_out="raw_symbol",
            start_date=start,
        )
        res = r.get("result", {}) if isinstance(r, dict) else {}
        entries = res.get(p, [])
        syms = sorted({e.get("s") for e in entries if isinstance(e, dict) and e.get("s")})
        print(f"  {p:10} -> {len(syms):3} contracts: {syms[:10]}{' ...' if len(syms) > 10 else ''}")
    except Exception as e:
        print(f"  {p:10} -> ERR {type(e).__name__}: {str(e)[:90]}")

# 3) instrument definitions for the BTC + MBT parents (asset / description / class)
print("\ndefinitions (asset / class / expiry) for BTC.FUT + MBT.FUT, recent session:")
try:
    end = dt.date.today().isoformat()
    data = h.timeseries.get_range(
        dataset="GLBX.MDP3", schema="definition",
        symbols=["BTC.FUT", "MBT.FUT"], stype_in="parent",
        start=start, end=end, limit=200,
    )
    seen = set()
    for r in data:
        raw = getattr(r, "raw_symbol", None)
        asset = getattr(r, "asset", None)
        cls = getattr(r, "instrument_class", None)
        if raw and raw not in seen:
            seen.add(raw)
            print(f"  {raw:16} asset={asset} class={cls}")
        if len(seen) >= 30:
            break
except Exception as e:
    print("  definition ERR:", type(e).__name__, str(e)[:120])
