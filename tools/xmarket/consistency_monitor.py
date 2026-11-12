"""The consistency monitor: check live prices of logically-related cross-venue pairs for
bound violations, the closest thing to riskless structure in this family.

For each relation from relations.json (A = Kalshi, B = Polymarket, prices = YES prob):
  a_implies_b  : riskless when  B_ask + (1 - A_bid) + fees < 1   (buy B-YES + A-NO; the
                 portfolio pays at least $1 in every outcome, $2 when B happens without A)
  b_implies_a  : the mirror (buy A-YES at ask + B-NO at 1 - B_bid)
  exclusive    : riskless when A_ask + B_ask + fees < ... (buying NO on both: pays >= 1 when
                 at most one happens; cost (1-A_bid)+(1-B_bid))  =>  violation when
                 A_bid + B_bid > 1 + fees (sell both sides via NO purchases)
Each run refreshes quotes (Kalshi authenticated batch; Polymarket gamma by condition id),
prints violations and near-misses, and appends every pair's state to
data/xmarket/consistency_log.csv so we also learn how often violations appear and how fast
they close. Nothing trades.
Usage: consistency_monitor.py [--near-c 2.0]
"""
import csv
import json
import os
import sys
import time
import urllib.request

OUT = "data/xmarket"
UA = {"User-Agent": "Mozilla/5.0"}


def kfee(p):
    return 0.07 * max(p, 1e-6) * (1 - min(p, 1 - 1e-6))


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


def kalshi_quotes(tickers):
    q = {}
    for i in range(0, len(tickers), 100):
        try:
            js = signed_get("/trade-api/v2/markets?tickers=" + ",".join(tickers[i:i + 100]))
            for m in js.get("markets", []):
                q[m["ticker"]] = (px(m, "yes_bid"), px(m, "yes_ask"))
        except Exception as e:
            print("kalshi quote batch %d failed: %s" % (i, e))
        time.sleep(0.5)
    return q


def poly_quotes(cids):
    q = {}
    for i in range(0, len(cids), 20):
        url = "https://gamma-api.polymarket.com/markets?" + "&".join("condition_ids=" + c for c in cids[i:i + 20])
        for att in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                    for m in json.loads(r.read().decode()):
                        q[m.get("conditionId")] = (float(m.get("bestBid") or 0), float(m.get("bestAsk") or 0))
                break
            except Exception as e:
                if att == 4:
                    print("poly quote batch %d failed: %s" % (i, e))
                time.sleep(3 * (att + 1))
        time.sleep(1.0)
    return q


def main():
    near = 0.02
    if "--near-c" in sys.argv:
        near = float(sys.argv[sys.argv.index("--near-c") + 1]) / 100.0
    rels = [r for r in json.load(open(OUT + "/relations.json")) if r["rel"] != "none"]
    print("monitoring %d bounded relations" % len(rels))
    kq = kalshi_quotes(sorted({r["k_id"] for r in rels}))
    pq = poly_quotes(sorted({r["p_id"] for r in rels}))
    ts = int(time.time())
    logp = os.path.join(OUT, "consistency_log.csv")
    newf = not os.path.exists(logp)
    nviol = nnear = nchecked = 0
    with open(logp, "a", newline="") as f:
        w = csv.writer(f)
        if newf:
            w.writerow(["ts", "rel", "k_id", "p_id", "k_bid", "k_ask", "p_bid", "p_ask", "edge"])
        for r in rels:
            k = kq.get(r["k_id"])
            p = pq.get(r["p_id"])
            if not k or not p or not (0 < k[0] < 1 and 0 < k[1] < 1 and 0 < p[0] < 1 and 0 < p[1] < 1):
                continue
            nchecked += 1
            kb, ka = k
            pb, pa = p
            if r["rel"] == "a_implies_b":
                # buy B-YES at pa + buy A-NO at (1-kb); payoff >= 1 in all outcomes
                edge = 1.0 - (pa + (1 - kb) + kfee(1 - kb))
            elif r["rel"] == "b_implies_a":
                edge = 1.0 - (ka + (1 - pb) + kfee(ka))
            else:  # exclusive: buy both NO sides; payoff >= 1 unless both happen (impossible)
                edge = 1.0 - ((1 - kb) + (1 - pb) + kfee(1 - kb))
            w.writerow([ts, r["rel"], r["k_id"], r["p_id"], kb, ka, pb, pa, round(edge, 4)])
            tag = None
            if edge > 0:
                nviol += 1
                tag = "VIOLATION"
            elif edge > -near:
                nnear += 1
                tag = "near"
            if tag:
                print("%-9s %+5.1fc %-12s K %.2f/%.2f P %.2f/%.2f | %s <=> %s" % (
                    tag, 100 * edge, r["rel"], kb, ka, pb, pa, r["k_title"][:50], r["p_title"][:50]))
    print("checked %d/%d pairs with live two-sided books | violations %d | near (within %.1fc) %d" % (
        nchecked, len(rels), nviol, 100 * near, nnear))


if __name__ == "__main__":
    main()
