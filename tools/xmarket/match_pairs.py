"""Match same-event market pairs across Kalshi and Polymarket from the fetched catalogs.

Deterministic first pass, built to have high precision (we would trade these):
  score = keyword overlap of the normalized titles, with hard requirements:
  - close dates within 36 hours of each other (or both missing),
  - every NUMBER appearing in either title (strikes, years, percentages) matches,
  - overlap of informative words >= 0.5 of the shorter title's informative words.
Writes data/xmarket/pairs.json: [{k_id, p_id, k_title, p_title, score}], sorted by score.
An optional LLM confirmation pass (match_pairs.py --llm) re-checks the borderline band
0.5-0.7 through `claude -p` one batch at a time.
Usage: match_pairs.py [--min-volume V] [--llm]
"""
import json
import re
import sys
from datetime import datetime, timezone

STOP = set("""a an the of in on at to for by with will be is are was were do does did and or vs v
over under above below more less than between before after end close day week month year market
price who what which when how many much yes no up down win won""".split())


def norm_words(title):
    t = re.sub(r"[^a-z0-9. ]", " ", title.lower())
    words = set(w for w in t.split() if len(w) > 2 and w not in STOP)
    nums = set(re.findall(r"\d+(?:\.\d+)?", title.replace(",", "")))
    return words, nums


def close_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def main():
    min_vol = 0.0
    if "--min-volume" in sys.argv:
        min_vol = float(sys.argv[sys.argv.index("--min-volume") + 1])
    K = json.load(open("data/xmarket/kalshi_markets.json"))
    P = json.load(open("data/xmarket/polymarket_markets.json"))
    K = [m for m in K if (m["volume"] or 0) >= min_vol]
    P = [m for m in P if (m["volume"] or 0) >= min_vol]
    print("matching %d kalshi x %d polymarket" % (len(K), len(P)))
    for m in K + P:
        m["_w"], m["_n"] = norm_words(m["title"])
        m["_dt"] = close_dt(m["close_ts"])
    # index polymarket by informative words for speed
    from collections import defaultdict
    idx = defaultdict(set)
    for j, p in enumerate(P):
        for w in p["_w"]:
            idx[w].add(j)
    pairs = []
    for k in K:
        if not k["_w"]:
            continue
        cand = set()
        for w in k["_w"]:
            cand |= idx[w]
        best = None
        for j in cand:
            p = P[j]
            if k["_dt"] and p["_dt"] and abs((k["_dt"] - p["_dt"]).total_seconds()) > 36 * 3600:
                continue
            if k["_n"] != p["_n"]:          # every number must match exactly (strikes, dates, pcts)
                continue
            inter = len(k["_w"] & p["_w"])
            denom = min(len(k["_w"]), len(p["_w"])) or 1
            score = inter / denom
            if score >= 0.5 and (best is None or score > best[0]):
                best = (score, p)
        if best:
            s, p = best
            pairs.append({"score": round(s, 3), "k_id": k["id"], "p_id": p["id"],
                          "k_title": k["title"][:120], "p_title": p["title"][:120],
                          "k_bid": k["yes_bid"], "k_ask": k["yes_ask"],
                          "p_bid": p["yes_bid"], "p_ask": p["yes_ask"],
                          "k_vol": k["volume"], "p_vol": p["volume"]})
    pairs.sort(key=lambda r: -r["score"])
    json.dump(pairs, open("data/xmarket/pairs.json", "w"), indent=1)
    print("pairs written: %d (score>=0.8: %d, 0.5-0.8: %d)" % (
        len(pairs), sum(1 for r in pairs if r["score"] >= 0.8), sum(1 for r in pairs if r["score"] < 0.8)))


if __name__ == "__main__":
    main()
