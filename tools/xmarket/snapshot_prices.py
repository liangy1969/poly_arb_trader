"""Periodic top-of-book snapshot of the markets our monitors actually reference.

The tracked set is the union of: matched pairs (score >= 0.7), classified relations, and any
market the news agent has ever named. Kalshi quotes come from the authenticated host in
batches of 100 tickers (the public host throttles and the /markets sweep drowns in shells);
Polymarket from gamma by condition id in batches of 20.
Appends to data/xmarket/snapshots.csv: ts,venue,id,yes_bid,yes_ask
Run on a schedule; news_eval.py and the correlation layer consume the file.
Usage: snapshot_prices.py
"""
import csv
import json
import os
import time
import urllib.request

OUT = "data/xmarket"
UA = {"User-Agent": "Mozilla/5.0"}


def signed_get(path):
    import base64
    import http.client
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_private_key(open(r"C:/Users/fatli/.poly/kalshi_private.txt", "rb").read(), password=None)
    conn = http.client.HTTPSConnection("external-api.kalshi.com", timeout=30)
    ts = str(int(time.time() * 1000))
    sig = key.sign((ts + "GET" + path.split("?")[0]).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    conn.request("GET", path, headers={"KALSHI-ACCESS-KEY": "e547e0c2-c92f-4d1a-aaef-58d6d589032d",
                                       "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
                                       "User-Agent": "Mozilla/5.0"})
    return json.loads(conn.getresponse().read().decode())


def px(m, side):
    v = m.get(side + "_dollars")
    if v is not None:
        return float(v)
    return (m.get(side) or 0) / 100.0


def tracked():
    k, p = set(), set()
    try:
        for r in json.load(open(OUT + "/pairs.json")):
            if r["score"] >= 0.7:
                k.add(r["k_id"]); p.add(r["p_id"])
    except FileNotFoundError:
        pass
    try:
        for r in json.load(open(OUT + "/relations.json")):
            k.add(r["k_id"]); p.add(r["p_id"])
    except FileNotFoundError:
        pass
    if os.path.exists(OUT + "/news_calls.jsonl"):
        for line in open(OUT + "/news_calls.jsonl", encoding="utf-8"):
            try:
                for c in json.loads(line).get("calls", []):
                    (k if c.get("venue") == "kalshi" else p).add(str(c.get("id")))
            except Exception:
                continue
    return sorted(k), sorted(p)


def main():
    kids, pids = tracked()
    ts = int(time.time())
    rows = []
    for i in range(0, len(kids), 100):
        try:
            js = signed_get("/trade-api/v2/markets?tickers=" + ",".join(kids[i:i + 100]))
            for m in js.get("markets", []):
                rows.append((ts, "kalshi", m["ticker"], px(m, "yes_bid"), px(m, "yes_ask")))
        except Exception as e:
            print("kalshi batch %d: %s" % (i, e))
        time.sleep(0.5)
    for i in range(0, len(pids), 20):
        url = "https://gamma-api.polymarket.com/markets?" + "&".join("condition_ids=" + c for c in pids[i:i + 20])
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                for m in json.loads(r.read().decode()):
                    rows.append((ts, "polymarket", m.get("conditionId"), float(m.get("bestBid") or 0), float(m.get("bestAsk") or 0)))
        except Exception as e:
            print("poly batch %d: %s" % (i, e))
        time.sleep(0.4)
    path = os.path.join(OUT, "snapshots.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts", "venue", "id", "yes_bid", "yes_ask"])
        w.writerows(rows)
    print("snapshot @%d: %d rows (%d kalshi + %d poly tracked)" % (ts, len(rows), len(kids), len(pids)))


if __name__ == "__main__":
    main()
