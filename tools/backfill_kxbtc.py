#!/usr/bin/env python3
"""Backfill the KXBTC15M probability-model dataset (offline training, phase 1).

For every settled KXBTC15M market in the lookback window, join:
  - Binance SPOT 1s klines (via the Tokyo SOCKS tunnel on 127.0.0.1:1081 —
    binance geo-blocks US IPs; spot ~= CF settlement source, and any residual
    perp/spot basis is absorbed by the per-event bias b_e anyway)
  - Kalshi tick trades (the market-probability path, last-trade YES price)
  - Kalshi market meta: floor_strike (exact b_e prior), result + expiration_value
    (exact outcome label)

into per-second rows:  ticker,t,tte_s,spot,prob,n_trades,strike,outcome

Run ON THE OHIO BOX (needs the tunnel):  python3 backfill_kxbtc.py [days] [out_dir]
Emits: <out>/events_meta.csv, <out>/rows.csv.gz, progress on stdout.
Dependency-free (urllib + curl subprocess for the SOCKS leg).
"""
import csv
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SOCKS = "127.0.0.1:1081"
PACE_S = 0.08  # ~6 req/s against Kalshi
FFILL_MAX_S = 30  # carry the last trade price at most this far


def kget(path, q=""):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(KALSHI + path + q, timeout=15) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — backoff + retry any transport error
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    return None


def binance_1s(start_ms, end_ms):
    """One 15-min window fits a single 1000-row request."""
    url = (
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1s"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    for attempt in range(4):
        out = subprocess.run(
            ["curl", "-s", "--socks5-hostname", SOCKS, "--max-time", "15", url],
            capture_output=True,
            text=True,
        )
        try:
            rows = json.loads(out.stdout)
            if isinstance(rows, list):
                return {int(r[0]) // 1000: float(r[4]) for r in rows}  # sec -> close
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5 * (attempt + 1))
    return {}


def iso_to_unix(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def settled_markets(days):
    """All settled KXBTC15M markets closing within the lookback, newest first."""
    min_close = int(time.time()) - days * 86400
    cursor, out = None, []
    while True:
        q = f"?series_ticker=KXBTC15M&status=settled&limit=1000&min_close_ts={min_close}"
        if cursor:
            q += f"&cursor={cursor}"
        d = kget("/markets", q)
        ms = d.get("markets", [])
        out += ms
        cursor = d.get("cursor")
        time.sleep(PACE_S)
        if not cursor or not ms:
            return out


def event_trades(ticker):
    """All tick trades for one market: [(unix_s, yes_price)] ascending."""
    cursor, rows = None, []
    for _ in range(30):  # cap ~30k trades
        q = f"?ticker={ticker}&limit=1000" + (f"&cursor={cursor}" if cursor else "")
        d = kget("/markets/trades", q)
        for t in d.get("trades", []):
            yp = t.get("yes_price_dollars") or t.get("yes_price")
            ts = t.get("created_time")
            if yp is not None and ts:
                rows.append((iso_to_unix(ts), float(yp)))
        cursor = d.get("cursor")
        time.sleep(PACE_S)
        if not cursor:
            break
    rows.sort()
    return rows


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/model"
    limit_events = int(os.environ.get("BACKFILL_LIMIT", "0"))  # 0 = all
    os.makedirs(out_dir, exist_ok=True)

    mkts = settled_markets(days)
    print(f"settled markets in {days}d: {len(mkts)}", flush=True)
    if limit_events:
        mkts = mkts[:limit_events]

    meta_f = open(os.path.join(out_dir, "events_meta.csv"), "w", newline="")
    meta = csv.writer(meta_f)
    meta.writerow(["ticker", "open_ts", "close_ts", "strike", "result", "expiration_value", "n_trades"])
    rows_f = gzip.open(os.path.join(out_dir, "rows.csv.gz"), "wt", newline="")
    rows = csv.writer(rows_f)
    rows.writerow(["ticker", "t", "tte_s", "spot", "prob", "n_trades", "strike", "outcome"])

    kept = skipped = 0
    for i, m in enumerate(mkts):
        ticker = m["ticker"]
        try:
            close_ts = iso_to_unix(m["close_time"])
            open_ts = close_ts - 900  # 15-min window
            strike = m.get("floor_strike")
            result = m.get("result", "")
            if strike is None or result not in ("yes", "no"):
                skipped += 1
                continue
            outcome = 1 if result == "yes" else 0

            trades = event_trades(ticker)
            if len(trades) < 50:  # illiquid window: not learnable
                skipped += 1
                continue
            spot = binance_1s(open_ts * 1000, close_ts * 1000 - 1)
            if len(spot) < 600:
                skipped += 1
                continue

            meta.writerow([ticker, open_ts, close_ts, strike, result, m.get("expiration_value", ""), len(trades)])

            # per-second join: last trade price, forward-filled <= FFILL_MAX_S
            ti = 0
            last_px, last_ts = None, None
            for sec in range(open_ts, close_ts):
                while ti < len(trades) and trades[ti][0] <= sec:
                    last_ts, last_px = trades[ti]
                    ti += 1
                if sec not in spot or last_px is None or sec - last_ts > FFILL_MAX_S:
                    continue
                n_in_sec = sum(1 for k in range(ti - 1, -1, -1) if trades[k][0] == sec) if ti else 0
                rows.writerow([ticker, sec, close_ts - sec, spot[sec], last_px, n_in_sec, strike, outcome])
            kept += 1
        except Exception as e:  # noqa: BLE001 — skip bad events, keep the sweep alive
            print(f"  skip {ticker}: {e}", flush=True)
            skipped += 1
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(mkts)}] kept={kept} skipped={skipped}", flush=True)

    meta_f.close()
    rows_f.close()
    print(f"DONE kept={kept} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
