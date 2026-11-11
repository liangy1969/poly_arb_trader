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
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def fetch_kalshi():
    rows, cursor = [], ""
    base = "https://api.elections.kalshi.com/trade-api/v2/markets?limit=1000&status=open"
    while True:
        url = base + ("&cursor=" + cursor if cursor else "")
        js = get_json(url)
        for m in js.get("markets", []):
            rows.append({
                "venue": "kalshi",
                "id": m["ticker"],
                "title": (m.get("title") or "") + " | " + (m.get("subtitle") or "") + " | " + (m.get("yes_sub_title") or ""),
                "yes_bid": (m.get("yes_bid") or 0) / 100.0,
                "yes_ask": (m.get("yes_ask") or 0) / 100.0,
                "close_ts": m.get("close_time") or "",
                "volume": m.get("volume") or 0,
                "extra": {"event": m.get("event_ticker"), "strike": m.get("floor_strike") or m.get("cap_strike")},
            })
        cursor = js.get("cursor") or ""
        print("kalshi: %d markets so far" % len(rows), flush=True)
        if not cursor:
            break
    json.dump(rows, open(os.path.join(OUT, "kalshi_markets.json"), "w"))
    print("kalshi: wrote %d" % len(rows))


def fetch_poly():
    rows, offset = [], 0
    while True:
        js = get_json("https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=500&offset=%d" % offset)
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
        offset += 500
        print("polymarket: %d markets so far" % len(rows), flush=True)
        if len(js) < 500:
            break
    json.dump(rows, open(os.path.join(OUT, "polymarket_markets.json"), "w"))
    print("polymarket: wrote %d" % len(rows))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("kalshi", "both"):
        fetch_kalshi()
    if which in ("poly", "both"):
        fetch_poly()
