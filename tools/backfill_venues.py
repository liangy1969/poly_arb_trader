#!/usr/bin/env python3
"""Multi-venue 1s BTC price backfill for the per-venue KL comparison.

For each event in an existing events_meta.csv (subset: the most recent N days),
fetch a per-second BTC price series for the event's 15-min window from:

  bitstamp   BTC/USD  native 1s OHLC (step=1, start/end; 1 call/event)
  kraken     BTC/USD  public trades since=ts -> last-price per second
  coinbase   BTC-USD  exchange trades, id-paginated: binary-search the id range
                      by time once per event, then page through the window
  binance_p  BTCUSDT perp aggTrades (startTime/endTime) -> 1s   [needs tunnel]

Output: <out>/venue_prices.csv.gz  rows: ticker,t,venue,px   (t = unix sec,
px = last trade price in that second, forward-fillable downstream).

Run ON THE BOX. Usage: python3 backfill_venues.py <data_dir> <out_dir> [days]
Pacing is conservative; a 14-day run (~1300 events x 4 venues) takes hours.
"""
import csv
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request

SOCKS = "127.0.0.1:1081"
PACE = {"bitstamp": 0.35, "kraken": 0.6, "coinbase": 0.25, "binance_p": 0.15}


def http_json(url, headers=None, socks=False, tries=4):
    for a in range(tries):
        try:
            if socks:
                out = subprocess.run(
                    ["curl", "-s", "--socks5-hostname", SOCKS, "--max-time", "20", url],
                    capture_output=True, text=True,
                )
                return json.loads(out.stdout)
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if a == tries - 1:
                raise
            time.sleep(1.5 * (a + 1))
    return None


# ── per-venue fetchers: (open_ts, close_ts) -> {sec: px} ──

def fetch_bitstamp(o, c):
    d = http_json(
        f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=1&limit=1000&start={o}&end={c - 1}"
    )
    out = {}
    for k in d.get("data", {}).get("ohlc", []):
        out[int(k["timestamp"])] = float(k["close"])
    return out


def fetch_kraken(o, c):
    out = {}
    since = o * 1_000_000_000
    for _ in range(20):
        d = http_json(f"https://api.kraken.com/0/public/Trades?pair=XBTUSD&since={since}")
        res = d.get("result", {})
        trades = res.get("XXBTZUSD", [])
        if not trades:
            break
        for tr in trades:
            ts = int(float(tr[2]))
            if ts >= c:
                return out
            if ts >= o:
                out[ts] = float(tr[0])
        since = int(res.get("last", since))
        if int(float(trades[-1][2])) >= c:
            break
        time.sleep(PACE["kraken"])
    return out


_CB_ID_CACHE = {}  # rough (ts -> trade_id) anchors to speed successive searches


def _cb_page(before=None, limit=1000):
    url = f"https://api.exchange.coinbase.com/products/BTC-USD/trades?limit={limit}"
    if before:
        url += f"&after={before}"  # exchange API: 'after' pages BACKWARD in ids
    return http_json(url)


def fetch_coinbase(o, c):
    """Binary-search a trade id near close_ts, then page backward through the
    window. Uses a coarse anchor cache so consecutive events search little."""
    # find any anchor: latest trade
    d = _cb_page()
    if not d:
        return {}
    hi_id, hi_ts = int(d[0]["trade_id"]), int(
        time.mktime(time.strptime(d[0]["time"][:19], "%Y-%m-%dT%H:%M:%S"))
    )
    # nearest cached anchor below close
    anchors = sorted(_CB_ID_CACHE.items())
    lo_id, lo_ts = 1, hi_ts - 10 * 365 * 86400
    for ts_a, id_a in anchors:
        if ts_a <= c:
            lo_ts, lo_id = ts_a, id_a
        if ts_a > c:
            hi_ts, hi_id = ts_a, id_a
            break
    # binary search id whose ts ~ close_ts
    tgt = c
    while hi_id - lo_id > 800:
        mid = (hi_id + lo_id) // 2
        d = _cb_page(before=mid + 1, limit=1)
        if not d:
            break
        ts = int(time.mktime(time.strptime(d[0]["time"][:19], "%Y-%m-%dT%H:%M:%S")))
        _CB_ID_CACHE[ts] = int(d[0]["trade_id"])
        if ts < tgt:
            lo_id, lo_ts = int(d[0]["trade_id"]), ts
        else:
            hi_id, hi_ts = int(d[0]["trade_id"]), ts
        time.sleep(PACE["coinbase"])
    # page backward from hi_id through the window
    out = {}
    cur = hi_id + 1
    for _ in range(60):
        d = _cb_page(before=cur)
        if not d:
            break
        stop = False
        for tr in d:  # newest -> oldest
            ts = int(time.mktime(time.strptime(tr["time"][:19], "%Y-%m-%dT%H:%M:%S")))
            if ts >= c:
                continue
            if ts < o:
                stop = True
                break
            out.setdefault(ts, float(tr["price"]))  # keep newest per second
        cur = int(d[-1]["trade_id"])
        if stop or cur <= 1:
            break
        time.sleep(PACE["coinbase"])
    _CB_ID_CACHE[c] = hi_id
    return out


def fetch_binance_perp(o, c):
    out = {}
    start = o * 1000
    for _ in range(40):
        d = http_json(
            f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&startTime={start}&endTime={c * 1000 - 1}&limit=1000",
            socks=True,
        )
        if not isinstance(d, list) or not d:
            break
        for tr in d:
            out[int(tr["T"]) // 1000] = float(tr["p"])
        if len(d) < 1000:
            break
        start = int(d[-1]["T"]) + 1
        time.sleep(PACE["binance_p"])
    return out


# NOTE: bitstamp dropped — its OHLC API has no 1s step (min 60s), and its
# public trades endpoint can't seek arbitrary historical windows.
FETCHERS = {
    "kraken": fetch_kraken,
    "coinbase": fetch_coinbase,
    "binance_p": fetch_binance_perp,
}


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/model"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "data/venues"
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 14
    os.makedirs(out_dir, exist_ok=True)
    cutoff = time.time() - days * 86400

    events = []
    with open(os.path.join(data_dir, "events_meta.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["open_ts"]) >= cutoff:
                events.append((r["ticker"], int(r["open_ts"]), int(r["close_ts"])))
    events.sort(key=lambda x: x[1])
    print(f"events in last {days}d: {len(events)}", flush=True)

    outf = gzip.open(os.path.join(out_dir, "venue_prices.csv.gz"), "wt", newline="")
    w = csv.writer(outf)
    w.writerow(["ticker", "t", "venue", "px"])
    stats = {v: 0 for v in FETCHERS}
    for i, (tick, o, c) in enumerate(events):
        for v, fn in FETCHERS.items():
            try:
                series = fn(o, c)
                for sec, px in sorted(series.items()):
                    w.writerow([tick, sec, v, px])
                stats[v] += len(series)
            except Exception as e:  # noqa: BLE001
                print(f"  {tick} {v}: {e}", flush=True)
            time.sleep(PACE[v])
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(events)}] rows: {stats}", flush=True)
            outf.flush()
    outf.close()
    print(f"DONE {stats}", flush=True)


if __name__ == "__main__":
    main()
