"""Poll public RSS feeds into a deduplicated headline store (data/xmarket/news.jsonl).

Feeds cover macro, crypto, politics and general wires — the domains Kalshi/Polymarket list
markets on. Each run appends only unseen (hash-deduped) items. No keys needed.
Usage: news_fetch.py  (run on a schedule; each run is one poll)
"""
import hashlib
import json
import os
import re
import time
import urllib.request

FEEDS = {
    "reuters-biz": "https://feeds.reuters.com/reuters/businessNews",
    "ap-top": "https://rsshub.app/apnews/topics/apf-topnews",
    "cnbc-top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "bbc-world": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "politico": "https://rss.politico.com/politics-news.xml",
    "fed-press": "https://www.federalreserve.gov/feeds/press_all.xml",
}
OUT = "data/xmarket"
os.makedirs(OUT, exist_ok=True)
STORE = os.path.join(OUT, "news.jsonl")
UA = {"User-Agent": "Mozilla/5.0"}


def seen_hashes():
    s = set()
    if os.path.exists(STORE):
        for line in open(STORE, encoding="utf-8"):
            try:
                s.add(json.loads(line)["h"])
            except Exception:
                pass
    return s


def parse_items(xml):
    # minimal RSS/Atom title+link extraction, no external deps
    items = []
    for m in re.finditer(r"<item[ >].*?</item>|<entry[ >].*?</entry>", xml, re.S):
        chunk = m.group(0)
        t = re.search(r"<title[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", chunk, re.S)
        if t:
            title = re.sub(r"<[^>]+>", "", t.group(1)).strip()
            if title:
                items.append(title)
    return items


def main():
    seen = seen_hashes()
    added = 0
    with open(STORE, "a", encoding="utf-8") as out:
        for name, url in FEEDS.items():
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
                    xml = r.read().decode("utf-8", errors="replace")
            except Exception as e:
                print("%s: fetch failed (%s)" % (name, e))
                continue
            n = 0
            for title in parse_items(xml):
                h = hashlib.md5(title.lower().encode()).hexdigest()[:16]
                if h in seen:
                    continue
                seen.add(h)
                out.write(json.dumps({"h": h, "src": name, "t": int(time.time()), "title": title}) + "\n")
                n += 1
                added += 1
            print("%s: +%d" % (name, n))
    print("added %d new headlines (store: %s)" % (added, STORE))


if __name__ == "__main__":
    main()
