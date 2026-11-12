"""The news-mapping agent: for fresh headlines, ask an LLM which active markets they move,
which direction, and which OTHER markets should co-move — producing (headline -> market,
direction, confidence) rows plus cross-market linkage hypotheses.

The LLM backend is the local `claude` CLI in headless print mode (claude -p), so it runs on
the user's existing subscription; no API key handling here. The prompt carries a compact
catalog digest (high-volume market titles) so the model maps onto REAL listed markets.

Outputs append to data/xmarket/news_calls.jsonl:
  {"t":..., "h": headline_hash, "headline":..., "calls":[{"venue","id","title","dir","conf"}],
   "links":[{"a_title","b_title","why"}]}
news_eval.py later scores these calls against actual price moves (the honesty layer).
Usage: news_agent.py [--max-headlines 20] [--min-volume 10000]
"""
import json
import os
import subprocess

def claude_bin():
    """Resolve the claude CLI: PATH first, then the newest VS Code extension bundle."""
    import glob as _g
    import os as _os
    import shutil as _sh
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = _sh.which(name)
        if p:
            return p
    pat = _os.path.expanduser(r"~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe")
    cands = sorted(_g.glob(pat))
    if cands:
        return cands[-1]
    raise FileNotFoundError("claude CLI not found on PATH or in VS Code extensions")

import sys
import time

OUT = "data/xmarket"
CALLS = os.path.join(OUT, "news_calls.jsonl")


def catalog_digest(min_vol):
    rows = []
    for fn in ("kalshi_markets.json", "polymarket_markets.json"):
        try:
            for m in json.load(open(os.path.join(OUT, fn))):
                if (m["volume"] or 0) >= min_vol:
                    rows.append("%s|%s|%s" % (m["venue"], m["id"][:40], m["title"][:100].replace("\n", " ")))
        except FileNotFoundError:
            pass
    return rows


def fresh_headlines(limit):
    done = set()
    if os.path.exists(CALLS):
        for line in open(CALLS, encoding="utf-8"):
            try:
                done.add(json.loads(line)["h"])
            except Exception:
                pass
    items = []
    for line in open(os.path.join(OUT, "news.jsonl"), encoding="utf-8"):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j["h"] not in done:
            items.append(j)
    return items[-limit:]


def ask_claude(prompt):
    r = subprocess.run([claude_bin(), "-p", "--output-format", "text"], input=prompt,
                       capture_output=True, text=True, timeout=300, encoding="utf-8")
    return r.stdout.strip()


def main():
    max_h = 20
    if "--max-headlines" in sys.argv:
        max_h = int(sys.argv[sys.argv.index("--max-headlines") + 1])
    min_vol = 10000
    if "--min-volume" in sys.argv:
        min_vol = float(sys.argv[sys.argv.index("--min-volume") + 1])
    cat = catalog_digest(min_vol)
    heads = fresh_headlines(max_h)
    if not heads:
        print("no fresh headlines")
        return
    print("headlines: %d | catalog digest: %d markets" % (len(heads), len(cat)))
    prompt = (
        "You map news headlines onto prediction markets. Below is a catalog of ACTIVE markets "
        "(format venue|id|title) and a list of fresh headlines.\n"
        "For each headline that plausibly moves at least one listed market, output a JSON line:\n"
        '{"h":"<headline hash>","calls":[{"venue":"...","id":"...","dir":"up|down","conf":0-1}],'
        '"links":[{"a_id":"...","b_id":"...","why":"one short clause"}]}\n'
        '"dir" is the direction of the YES price. "links" are pairs of listed markets this headline '
        "connects causally (would co-move). Skip headlines that move nothing. Output ONLY JSON lines.\n\n"
        "CATALOG:\n" + "\n".join(cat[:800]) + "\n\nHEADLINES:\n" +
        "\n".join("%s %s" % (h["h"], h["title"]) for h in heads)
    )
    out = ask_claude(prompt)
    n = 0
    with open(CALLS, "a", encoding="utf-8") as f:
        bym = {h["h"]: h for h in heads}
        for line in out.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            src = bym.get(j.get("h"), {})
            j["t"] = int(time.time())
            j["headline"] = src.get("title", "")
            f.write(json.dumps(j) + "\n")
            n += 1
        # mark headline hashes with no calls as processed (empty record) so they are not re-asked
        answered = set()
        for line in out.splitlines():
            try:
                answered.add(json.loads(line.strip().strip("`")).get("h"))
            except Exception:
                pass
        for h in heads:
            if h["h"] not in answered:
                f.write(json.dumps({"h": h["h"], "t": int(time.time()), "headline": h["title"], "calls": [], "links": []}) + "\n")
    print("agent calls written: %d (plus %d no-op records)" % (n, len(heads) - n))


if __name__ == "__main__":
    main()
