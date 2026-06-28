#!/usr/bin/env python3
"""Dump the full CME instrument definition for a symbol (default BTCM6) and interpret
the contract spec.  $env:DATABENTO_API_KEY=...; py scripts/databento_def_dump.py BTCM6
"""
import os, sys, datetime as dt
import databento as db

h = db.Historical(os.environ["DATABENTO_API_KEY"])
sym = sys.argv[1] if len(sys.argv) > 1 else "BTCM6"
start = (dt.date.today() - dt.timedelta(days=5)).isoformat()
end = dt.date.today().isoformat()

data = h.timeseries.get_range(
    dataset="GLBX.MDP3", schema="definition",
    symbols=[sym], stype_in="raw_symbol", start=start, end=end, limit=5,
)
rec = None
for r in data:
    rec = r
    if getattr(r, "raw_symbol", None) == sym:
        break
if rec is None:
    print(f"{sym}: not found"); sys.exit()

FIELDS = [
    "raw_symbol", "asset", "security_type", "instrument_class", "cfi", "underlying",
    "currency", "settl_currency", "exchange", "group", "channel_id",
    "min_price_increment", "display_factor", "price_display_format",
    "unit_of_measure", "unit_of_measure_qty", "contract_multiplier",
    "main_fraction", "sub_fraction", "min_lot_size_round_lot", "min_trade_vol",
    "activation", "expiration", "high_limit_price", "low_limit_price",
    "settl_price_type", "market_depth", "tick_rule",
]
print(f"== {sym} definition ==")
for f in FIELDS:
    v = getattr(rec, f, None)
    if v is None:
        continue
    if f in ("activation", "expiration") and isinstance(v, int) and v > 0:
        try: v = dt.datetime.utcfromtimestamp(v / 1e9).strftime("%Y-%m-%d %H:%M UTC")
        except Exception: pass
    print(f"  {f:24} = {v}")

# interpret price/size scaling
mpi = getattr(rec, "min_price_increment", 0)
df = getattr(rec, "display_factor", 1) or 1
uom_qty = getattr(rec, "unit_of_measure_qty", 0)
print("\n-- interpreted --")
print(f"  price tick (USD/BTC)   = min_price_increment/1e9 = ${mpi / 1e9:,.4f}")
print(f"  unit_of_measure_qty/1e9 = {uom_qty / 1e9:,.4f}  (contract size in the UoM)")
