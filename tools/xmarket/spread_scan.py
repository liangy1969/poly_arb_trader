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
