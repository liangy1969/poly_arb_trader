#!/usr/bin/env python3
"""Inspect a specific Databento symbol/parent: what instruments + contract specs.
  $env:DATABENTO_API_KEY="db-..."; py scripts/databento_check_sym.py BTCCP1.FUT
"""
import os, sys, datetime as dt
import databento as db

key = os.environ.get("DATABENTO_API_KEY")
sym = sys.argv[1] if len(sys.argv) > 1 else "BTCCP1.FUT"
h = db.Historical(key)
start = (dt.date.today() - dt.timedelta(days=5)).isoformat()
end = dt.date.today().isoformat()

print(f"checking {sym} on GLBX.MDP3, {start}..{end}\n")
try:
    data = h.timeseries.get_range(
        dataset="GLBX.MDP3", schema="definition",
        symbols=[sym], stype_in="parent", start=start, end=end, limit=300,
    )
    seen = {}
    for r in data:
        raw = getattr(r, "raw_symbol", None)
        if not raw or raw in seen:
            continue
        seen[raw] = True
        flds = {f: getattr(r, f, None) for f in
                ("asset", "security_type", "instrument_class", "expiration",
                 "min_price_increment", "unit_of_measure_qty", "underlying",
                 "currency", "display_factor", "group")}
        # expiration is ns since epoch
        exp = flds["expiration"]
        if isinstance(exp, int) and exp > 0:
            try: flds["expiration"] = dt.datetime.utcfromtimestamp(exp / 1e9).strftime("%Y-%m-%d")
            except Exception: pass
        print(f"  {raw:18} {flds}")
    if not seen:
        print("  (no instruments resolved under this parent)")
    else:
        print(f"\n  {len(seen)} instrument(s).")
except Exception as e:
    print("  ERR:", type(e).__name__, str(e)[:200])
