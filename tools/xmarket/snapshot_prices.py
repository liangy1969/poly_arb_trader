"""Periodic top-of-book snapshot of liquid markets on both venues, for the correlation layer.

Each run appends one row per tracked market to data/xmarket/snapshots.csv:
  ts,venue,id,yes_bid,yes_ask
Tracked = markets from the catalogs with volume >= --min-volume (default 5000), capped at
--max-markets (default 1500) by volume. Kalshi prices come in one paginated sweep of
/markets (fresh quotes); Polymarket from gamma (bestBid/bestAsk refresh on fetch).
Run on a schedule (e.g. every 5 minutes); corr_track.py consumes the file.
Usage: snapshot_prices.py [--min-volume 5000] [--max-markets 1500]
"""
import csv
import json
import os
import sys
import time
import urllib.request

OUT = "data/xmarket"
UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.loads(r.read().decode())


def main():
    min_vol = 5000.0
    if "--min-volume" in sys.argv:
        min_vol = float(sys.argv[sys.argv.index("--min-volume") + 1])
    max_m = 1500
    if "--max-markets" in sys.argv:
        max_m = int(sys.argv[sys.argv.index("--max-markets") + 1])
    track = {}
    for fn in ("kalshi_markets.json", "polymarket_markets.json"):
        for m in json.load(open(os.path.join(OUT, fn))):
            if (m["volume"] or 0) >= min_vol:
                track[(m["venue"], m["id"])] = m["volume"]
    keep = set(sorted(track, key=lambda k: -track[k])[:max_m])
    k_ids = {i for v, i in keep if v == "kalshi"}
    p_ids = {i for v, i in keep if v == "polymarket"}
    ts = int(time.time())
    rows = []
    # kalshi sweep
    cursor = ""
    while True:
        url = "https://api.elections.kalshi.com/trade-api/v2/markets?limit=1000&status=open" + ("&cursor=" + cursor if cursor else "")
        js = get_json(url)
        for m in js.get("markets", []):
            if m["ticker"] in k_ids:
                rows.append((ts, "kalshi", m["ticker"], (m.get("yes_bid") or 0) / 100.0, (m.get("yes_ask") or 0) / 100.0))
        cursor = js.get("cursor") or ""
        if not cursor:
            break
    # polymarket sweep
    offset = 0
    while True:
        js = get_json("https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=500&offset=%d" % offset)
        if not js:
            break
        for m in js:
            cid = m.get("conditionId") or str(m.get("id"))
            if cid in p_ids:
                rows.append((ts, "polymarket", cid, float(m.get("bestBid") or 0), float(m.get("bestAsk") or 0)))
        offset += 500
        if len(js) < 500:
            break
    path = os.path.join(OUT, "snapshots.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "venue", "id", "yes_bid", "yes_ask"])
        w.writerows(rows)
    print("snapshot @%d: %d rows (%d tracked)" % (ts, len(rows), len(keep)))


if __name__ == "__main__":
    main()
