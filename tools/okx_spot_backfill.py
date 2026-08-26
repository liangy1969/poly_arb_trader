#!/usr/bin/env python3
"""Per-second OKX SPOT (BTC-USDT) prices from the local parquet lake.

Emits the same `ticker,t,venue,px` schema as `backfill_venues.py` so the
existing multi-venue training scripts consume it unchanged — but reads the
lake instead of the venue REST API, which turns an hours-long paced crawl
into a few minutes and cannot be rate-limited.

`px` is the LAST trade price in each second (matching how kraken/coinbase are
built there), keyed to the event whose [open, close] window contains it.

⚠️ BTC-USDT, not BTC-USD. This series carries a USDT basis that the BRTI
constituents (coinbase/kraken/bitstamp/gemini) do not, so it must not be
pooled into a consolidated settlement mid. It is a model FEATURE only.

Schema note: the lake's `venue` column is `string` in some files and
`dictionary<string>` in others, so a whole-directory read raises
ArrowTypeError. Reading only the two columns actually needed side-steps it.

Usage:
  python tools/okx_spot_backfill.py data/model out/okx_spot [FROM] [TO]
"""
from __future__ import annotations

import calendar
import csv
import glob
import gzip
import os
import sys
import time

import pyarrow.parquet as pq

LAKE = "E:/crypto/data/parquet/stream=trade/venue=okx/market=SPOT/symbol=BTC-USDT"
VENUE = "okx"


def day_seconds(date: str):
    """{unix_sec: last trade price} for one lake date."""
    files = sorted(glob.glob(os.path.join(LAKE, f"date={date}", "*.parquet")))
    out = {}
    for f in files:
        try:
            # ONLY these two columns: avoids the venue string/dictionary clash.
            t = pq.ParquetFile(f).read(columns=["exch_ts_ns", "price"])
        except Exception as e:  # a truncated tail file should not kill the day
            print(f"  ! {os.path.basename(f)}: {type(e).__name__} {e}", flush=True)
            continue
        ts = t.column("exch_ts_ns").to_pylist()
        px = t.column("price").to_pylist()
        for a, p in zip(ts, px):
            if p and p > 0:
                out[a // 1_000_000_000] = p  # later rows overwrite -> LAST in second
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    data_dir, out_dir = sys.argv[1], sys.argv[2]
    d_from = sys.argv[3] if len(sys.argv) > 3 else "2026-05-24"
    d_to = sys.argv[4] if len(sys.argv) > 4 else "2026-12-31"

    meta = list(csv.DictReader(open(os.path.join(data_dir, "events_meta.csv"))))
    evs = []
    for r in meta:
        try:
            o, c = int(r["open_ts"]), int(r["close_ts"])
        except (KeyError, ValueError):
            continue
        evs.append((r["ticker"], o, c))
    evs.sort(key=lambda e: e[1])
    print(f"events in meta: {len(evs):,}")

    dates = sorted(
        os.path.basename(d)[5:]
        for d in glob.glob(os.path.join(LAKE, "date=*"))
        if d_from <= os.path.basename(d)[5:] <= d_to
    )
    print(f"lake dates {dates[0]} .. {dates[-1]} ({len(dates)})")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "venue_prices.csv.gz")
    n_rows = 0
    t0 = time.time()
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ticker", "t", "venue", "px"])
        for di, date in enumerate(dates):
            sec = day_seconds(date)
            if not sec:
                print(f"  {date}: NO DATA", flush=True)
                continue
            lo = calendar.timegm(time.strptime(date, "%Y-%m-%d"))
            hi = lo + 86_400
            # events overlapping this day
            todays = [e for e in evs if e[2] >= lo and e[1] < hi]
            wrote = 0
            for ticker, o, c in todays:
                for t in range(max(o, lo), min(c, hi)):
                    p = sec.get(t)
                    if p is not None:
                        w.writerow([ticker, t, VENUE, f"{p:.2f}"])
                        wrote += 1
            n_rows += wrote
            print(
                f"  {date}: {len(sec):,} secs, {len(todays)} events, {wrote:,} rows "
                f"({di + 1}/{len(dates)}, {time.time() - t0:.0f}s)",
                flush=True,
            )
    print(f"WROTE {path}: {n_rows:,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
