"""Pull the full public market catalogs from Kalshi and Polymarket into local JSON files.

Kalshi: GET /trade-api/v2/markets (status=open, paginated by cursor; public, no auth needed but we
sign anyway since we have keys — unauthenticated works too).
Polymarket: GET https://gamma-api.polymarket.com/markets?active=true&closed=false (paginated by offset).

Writes data/xmarket/kalshi_markets.json and data/xmarket/polymarket_markets.json, each a list of
normalized rows: {venue, id, title, yes_bid, yes_ask, close_ts, volume, extra}.
Usage: fetch_catalogs.py [kalshi|poly|both]
"""
import json
import os
import sys
import time
import urllib.request

OUT = "data/xmarket"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}


def get_json(url):
    for attempt in range(7):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == 6:
                raise
            time.sleep(min(60, 2 ** attempt * 2))


def fetch_kalshi():
    """Enumerate open EVENTS (the human-level catalog; /markets is drowned in millions of
    auto-generated zero-quote shells). Each nested market becomes a row WITHOUT quotes;
    spread_scan fetches live quotes per matched event at scan time."""
    rows, cursor, npages = [], "", 0
    part = os.path.join(OUT, "kalshi_partial.json")
    if os.path.exists(part):
        st = json.load(open(part))
        rows, cursor = st["rows"], st["cursor"]
        print("resuming kalshi from %d rows" % len(rows), flush=True)
    base = "https://api.elections.kalshi.com/trade-api/v2/events?status=open&limit=200&with_nested_markets=true"
    while True:
        js = get_json(base + ("&cursor=" + cursor if cursor else ""))
        for e in js.get("events", []):
            for m in e.get("markets", []):
                rows.append({
                    "venue": "kalshi",
                    "id": m["ticker"],
                    "title": " | ".join(x for x in (e.get("title"), e.get("sub_title"), m.get("yes_sub_title")) if x),
                    "yes_bid": 0.0, "yes_ask": 0.0,
                    "close_ts": m.get("close_time") or "",
                    "volume": m.get("volume") or 0,
                    "extra": {"event": e.get("event_ticker"), "category": e.get("category"),
                              "strike": m.get("floor_strike") or m.get("cap_strike")},
                })
        cursor = js.get("cursor") or ""
        npages += 1
        if npages % 25 == 0:
            print("kalshi: %d markets / %d event pages" % (len(rows), npages), flush=True)
            json.dump({"rows": rows, "cursor": cursor}, open(part, "w"))
        time.sleep(0.25)
        if not cursor:
            break
    json.dump(rows, open(os.path.join(OUT, "kalshi_markets.json"), "w"))
    if os.path.exists(part):
        os.remove(part)
    print("kalshi: wrote %d markets" % len(rows))


def fetch_poly():
    rows, offset = [], 0
    while True:
        try:
            js = get_json("https://gamma-api.polymarket.com/markets?active=true&closed=false&order=volumeNum&ascending=false&limit=100&offset=%d" % offset)
        except Exception as e:
            print("polymarket: pagination stopped at offset %d (%s)" % (offset, e))
            break
        if not js:
            break
        for m in js:
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                outcomes = json.loads(m.get("outcomes") or "[]")
            except Exception:
                prices, outcomes = [], []
            yes_px = None
            if len(outcomes) == 2 and "Yes" in outcomes:
                yes_px = float(prices[outcomes.index("Yes")]) if prices else None
            rows.append({
                "venue": "polymarket",
                "id": m.get("conditionId") or str(m.get("id")),
                "title": m.get("question") or "",
                "yes_bid": float(m.get("bestBid") or 0),
                "yes_ask": float(m.get("bestAsk") or 0),
                "close_ts": m.get("endDate") or "",
                "volume": float(m.get("volumeNum") or 0),
                "extra": {"yes_mid": yes_px, "slug": m.get("slug"), "neg_risk": m.get("negRisk"), "clob": m.get("clobTokenIds")},
            })
        offset += len(js)
        if offset % 2000 < 100:
            print("polymarket: %d markets so far" % len(rows), flush=True)
            json.dump(rows, open(os.path.join(OUT, "polymarket_markets.json"), "w"))
        time.sleep(0.2)
        if len(js) < 100:
            break
    json.dump(rows, open(os.path.join(OUT, "polymarket_markets.json"), "w"))
    print("polymarket: wrote %d" % len(rows))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("kalshi", "both"):
        fetch_kalshi()
    if which in ("poly", "both"):
        fetch_poly()
