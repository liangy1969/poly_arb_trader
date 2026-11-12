"""Scan matched cross-venue pairs for executable two-leg arbitrage.

For a matched pair (same event, both quoted as YES probability):
  lock A: buy Kalshi YES at k_ask, buy Polymarket NO at (1 - p_bid)
          cost = k_ask + (1 - p_bid) + kalshi_fee(k_ask); profit if cost < 1
  lock B: buy Polymarket YES at p_ask, buy Kalshi NO at (1 - k_bid)
          cost = p_ask + (1 - k_bid) + kalshi_fee(1 - k_bid); profit if cost < 1
Kalshi taker fee = 0.07 * p * (1-p) on the traded price; Polymarket taker fee ~0 on most
markets (fee-less CLOB), spread is the cost there.
Also reports plain price DISAGREEMENT (|k_mid - p_mid|) for pairs with no locked arb —
those are dislocation candidates, not riskless.
Usage: spread_scan.py [--min-edge-c 0.5] [--top 25]
"""
import json
import sys
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def _signed_get(path):
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


def _px(m, side):
    v = m.get(side + "_dollars")                 # 2026-11 field migration: *_dollars strings are canonical
    if v is not None:
        return float(v)
    return (m.get(side) or 0) / 100.0


def refresh_kalshi(pairs, min_score):
    """Fetch live Kalshi quotes for the matched pairs (batched, authenticated host)."""
    want = [r["k_id"] for r in pairs if r["score"] >= min_score]
    quotes = {}
    for i in range(0, len(want), 100):
        try:
            js = _signed_get("/trade-api/v2/markets?tickers=" + ",".join(want[i:i + 100]))
            for m in js.get("markets", []):
                vol = float(m.get("volume_fp") or m.get("volume") or 0)
                quotes[m["ticker"]] = (_px(m, "yes_bid"), _px(m, "yes_ask"), vol)
        except Exception as e:
            print("quote batch failed at %d: %s" % (i, e))
        time.sleep(0.5)
    n = 0
    for r in pairs:
        q = quotes.get(r["k_id"])
        if q:
            r["k_bid"], r["k_ask"], r["k_vol"] = q
            n += 1
    print("refreshed kalshi quotes for %d pairs" % n)
    return pairs


def kfee(p):
    return 0.07 * p * (1 - p)


def main():
    min_edge = 0.005
    if "--min-edge-c" in sys.argv:
        min_edge = float(sys.argv[sys.argv.index("--min-edge-c") + 1]) / 100.0
    top = 25
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    pairs = json.load(open("data/xmarket/pairs.json"))
    min_score = 0.7
    if "--min-score" in sys.argv:
        min_score = float(sys.argv[sys.argv.index("--min-score") + 1])
    pairs = [r for r in pairs if r["score"] >= min_score]
    if "--no-refresh" not in sys.argv:
        pairs = refresh_kalshi(pairs, min_score)
    arbs, dislocs = [], []
    for r in pairs:
        kb, ka, pb, pa = r["k_bid"], r["k_ask"], r["p_bid"], r["p_ask"]
        if not (0 < kb < 1 and 0 < ka < 1 and 0 < pb < 1 and 0 < pa < 1):
            continue
        lock_a = 1.0 - (ka + (1 - pb) + kfee(ka))           # buy K-YES + P-NO
        lock_b = 1.0 - (pa + (1 - kb) + kfee(1 - kb))       # buy P-YES + K-NO
        edge = max(lock_a, lock_b)
        row = dict(r)
        row["edge_c"] = round(100 * edge, 2)
        row["leg"] = "K-YES+P-NO" if lock_a >= lock_b else "P-YES+K-NO"
        if edge >= min_edge:
            arbs.append(row)
        else:
            kmid, pmid = 0.5 * (kb + ka), 0.5 * (pb + pa)
            row["gap_c"] = round(100 * abs(kmid - pmid), 2)
            dislocs.append(row)
    arbs.sort(key=lambda r: -r["edge_c"])
    dislocs.sort(key=lambda r: -r["gap_c"])
    print("=== LOCKED ARBS (edge after Kalshi fee, top-of-book both venues) ===")
    if not arbs:
        print("  none at min edge %.1fc" % (100 * min_edge))
    for r in arbs[:top]:
        print("  %+5.1fc %s  score %.2f  K[%s] %0.2f/%0.2f  P %0.2f/%0.2f  | %s" % (
            r["edge_c"], r["leg"], r["score"], r["k_id"][:28], r["k_bid"], r["k_ask"], r["p_bid"], r["p_ask"], r["k_title"][:70]))
    print("\n=== LARGEST DISAGREEMENTS (no locked arb; dislocation candidates) ===")
    for r in dislocs[:top]:
        print("  gap %5.1fc score %.2f  K[%s] %0.2f/%0.2f  P %0.2f/%0.2f  | %s" % (
            r["gap_c"], r["score"], r["k_id"][:28], r["k_bid"], r["k_ask"], r["p_bid"], r["p_ask"], r["k_title"][:70]))
    json.dump({"arbs": arbs, "dislocations": dislocs[:200]}, open("data/xmarket/scan_result.json", "w"), indent=1)
    print("\nwrote data/xmarket/scan_result.json (%d arbs, %d dislocations tracked)" % (len(arbs), len(dislocs)))


if __name__ == "__main__":
    main()
