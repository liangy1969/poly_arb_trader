#!/usr/bin/env python3
"""Standalone Kalshi trade-tape poller (Phase 0 of DESIGN_MARKET_MAKING.md).

Polls the PUBLIC trades endpoint for the open markets of a series every --interval seconds and
appends new prints (deduped by trade_id) to data/samples/trades-YYYY-MM-DD.csv (UTC day of the
poll). Runs as its own process: no trader restart, no config change, no auth.

Columns: recv_ms (local), created_time (exchange), ticker, trade_id, yes_price (0-1),
count (contracts, fractional allowed), taker_side (yes|no|?), raw_no_price (0-1).
Usage: kalshi_trades_poller.py [--series KXBTC15M] [--interval 2] [--out data/samples]
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.request

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HDR = {"User-Agent": "poly-arb-trades-poller/1.0", "Accept": "application/json"}


def get(url, timeout=15):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=timeout) as r:
        return json.load(r)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(t):
    yp = num(t.get("yes_price_dollars"))
    if yp is None and t.get("yes_price") is not None:
        yp = num(t.get("yes_price")) / 100.0
    npx = num(t.get("no_price_dollars"))
    if npx is None and t.get("no_price") is not None:
        npx = num(t.get("no_price")) / 100.0
    if yp is None and npx is not None:
        yp = 1.0 - npx
    cnt = num(t.get("count_fp"))
    if cnt is None:
        cnt = num(t.get("count"))
    side = t.get("taker_outcome_side") or t.get("taker_side") or "?"
    return yp, cnt, str(side).lower(), npx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="KXBTC15M")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--out", default="data/samples")
    ap.add_argument("--discover", type=float, default=30.0, help="seconds between market-list refreshes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seen = {}          # ticker -> set(trade_id) (bounded)
    tickers = []
    last_disc = 0.0
    n_rows = 0
    last_log = time.time()
    while True:
        now = time.time()
        if now - last_disc >= a.discover or not tickers:
            try:
                js = get(f"{KALSHI}/markets?series_ticker={a.series}&status=open&limit=50")
                tickers = [m["ticker"] for m in js.get("markets", []) if m.get("ticker")]
                last_disc = now
                for k in list(seen):
                    if k not in tickers:
                        seen.pop(k, None)
            except Exception as e:  # noqa: BLE001
                print(f"{dt.datetime.utcnow().isoformat()}Z discover failed: {e}", file=sys.stderr, flush=True)
        day = dt.datetime.utcnow().strftime("%Y-%m-%d")
        path = os.path.join(a.out, f"trades-{day}.csv")
        new_rows = []
        for tk in tickers:
            try:
                js = get(f"{KALSHI}/markets/trades?ticker={tk}&limit=100")
            except Exception as e:  # noqa: BLE001
                print(f"{dt.datetime.utcnow().isoformat()}Z trades {tk} failed: {e}", file=sys.stderr, flush=True)
                continue
            s = seen.setdefault(tk, set())
            recv_ms = int(time.time() * 1000)
            for t in js.get("trades", []):
                tid = t.get("trade_id") or t.get("id")
                if not tid or tid in s:
                    continue
                s.add(tid)
                yp, cnt, side, npx = parse(t)
                new_rows.append([recv_ms, t.get("created_time", ""), tk, tid,
                                 "" if yp is None else f"{yp:.4f}", "" if cnt is None else f"{cnt:g}", side,
                                 "" if npx is None else f"{npx:.4f}"])
            if len(s) > 20000:
                seen[tk] = set(list(s)[-10000:])
        if new_rows:
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["recv_ms", "created_time", "ticker", "trade_id", "yes_price", "count", "taker_side", "no_price"])
                w.writerows(new_rows)
            n_rows += len(new_rows)
        if time.time() - last_log >= 300:
            print(f"{dt.datetime.utcnow().isoformat()}Z alive: {len(tickers)} markets, {n_rows} rows so far", flush=True)
            last_log = time.time()
        time.sleep(max(0.5, a.interval - (time.time() - now)))


if __name__ == "__main__":
    main()
