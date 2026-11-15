"""The hidden-driver analyst: for a pair of markets (possibly different venues), run a real
AGENTIC investigation of whether one hidden event decides both, and whether their prices are
mutually consistent given it.

Per pair, the tool first assembles a deterministic dossier:
  - both markets' FULL settlement rules (Kalshi rules_primary via the authenticated API;
    Polymarket description via gamma), live quotes, expiries, volumes
  - price history for both legs from our snapshot store: correlation of changes, the recent
    path, and the current spread vs its baseline
Then it launches the claude CLI headless WITH TOOLS (WebSearch, WebFetch) so the model can
research the current state of the world, and requires a structured verdict:
  - the hidden driver H (or "none"), its researched probability P(H)
  - the latent model: P(A|H), P(A|not H), P(B|H), P(B|not H), each justified
  - implied fair prices vs market prices, per-leg mispricing in cents, the cheap/rich legs
  - rules-difference risks (the de-facto-vs-official trap) and confidence
Verdicts append to data/xmarket/hidden_links.jsonl; convergence can be scored later against
the snapshot store like every other agent output.

Usage:
  hidden_link_agent.py --pair KALSHI_TICKER POLY_CONDITION_ID
  hidden_link_agent.py --kalshi T1 T2          (two Kalshi markets)
  hidden_link_agent.py --from-watch N          (top-N moving pairs from the consistency log)
"""
import json
import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import subprocess
import time
import urllib.request

OUT = "data/xmarket"
UA = {"User-Agent": "Mozilla/5.0"}


def claude_bin():
    import glob as _g
    import shutil as _sh
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = _sh.which(name)
        if p:
            return p
    cands = sorted(_g.glob(os.path.expanduser(r"~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe")))
    if cands:
        return cands[-1]
    raise FileNotFoundError("claude CLI not found")


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


def kalshi_dossier(ticker):
    m = signed_get("/trade-api/v2/markets/" + ticker).get("market", {})
    return {
        "venue": "kalshi", "id": ticker,
        "title": " | ".join(x for x in (m.get("title"), m.get("subtitle"), m.get("yes_sub_title")) if x),
        "rules": (m.get("rules_primary") or "")[:1200],
        "bid": float(m.get("yes_bid_dollars") or 0), "ask": float(m.get("yes_ask_dollars") or 0),
        "close": m.get("close_time"), "volume": float(m.get("volume_fp") or 0),
    }


def poly_dossier(cid):
    url = "https://gamma-api.polymarket.com/markets?condition_ids=" + cid
    js = []
    for att in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                js = json.loads(r.read().decode())
            break
        except Exception:
            time.sleep(4 * (att + 1))
    m = js[0] if js else {}
    return {
        "venue": "polymarket", "id": cid,
        "title": m.get("question") or "",
        "rules": (m.get("description") or "")[:1200],
        "bid": float(m.get("bestBid") or 0), "ask": float(m.get("bestAsk") or 0),
        "close": m.get("endDate"), "volume": float(m.get("volumeNum") or 0),
    }


def history(venue, mid):
    import pandas as pd
    try:
        S = pd.read_csv(os.path.join(OUT, "snapshots.csv"))
    except FileNotFoundError:
        return "no local history"
    s = S[(S.venue == venue) & (S.id == mid) & (S.yes_bid > 0)]
    if len(s) < 4:
        return "no local history"
    s = s.sort_values("ts")
    mids = 0.5 * (s.yes_bid + s.yes_ask)
    pts = [(int(t), round(float(v), 3)) for t, v in zip(s.ts, mids)]
    step = max(1, len(pts) // 24)
    path = ", ".join("%s=%0.3f" % (time.strftime("%m-%d %H:%M", time.gmtime(t)), v) for t, v in pts[::step])
    return "range %.3f..%.3f | path: %s" % (min(v for _, v in pts), max(v for _, v in pts), path)


def analyze(a, b):
    da = kalshi_dossier(a) if not str(a).startswith("0x") else poly_dossier(a)
    db = poly_dossier(b) if str(b).startswith("0x") else kalshi_dossier(b)
    ha, hb = history(da["venue"], da["id"]), history(db["venue"], db["id"])
    dossier = (
        "MARKET A [%s %s]\n  title: %s\n  rules: %s\n  book: bid %.3f / ask %.3f | closes %s | volume %.0f\n  48h path: %s\n\n"
        "MARKET B [%s %s]\n  title: %s\n  rules: %s\n  book: bid %.3f / ask %.3f | closes %s | volume %.0f\n  48h path: %s\n"
        % (da["venue"], da["id"], da["title"], da["rules"], da["bid"], da["ask"], da["close"], da["volume"], ha,
           db["venue"], db["id"], db["title"], db["rules"], db["bid"], db["ask"], db["close"], db["volume"], hb)
    )
    prompt = (
        "You are a prediction-market analyst with web access. Investigate whether these two markets are both "
        "driven by one HIDDEN underlying event, and whether their prices are mutually consistent.\n\n" + dossier +
        "\nDo the following, using WebSearch (and WebFetch for primary sources) to establish the CURRENT state "
        "of the world — at least two searches before concluding:\n"
        "1. Identify the hidden driver H that most decides BOTH markets (or conclude there is none).\n"
        "2. Research H's current probability: recent news, official schedules/filings, expert handicapping.\n"
        "3. Build the latent model: P(H), P(A yes|H), P(A yes|not H), P(B yes|H), P(B yes|not H) — justify each "
        "number in one clause, respecting each market's EXACT settlement rules and dates (rules differences like "
        "de-facto vs official, announce vs complete, date cutoffs are where naive links break).\n"
        "4. Compute implied fair prices fair_a, fair_b. Compare with the books. Identify which leg is cheap/rich "
        "and by how many cents, and what combined position exploits the mismatch if any.\n"
        "5. State the biggest risk that would make the apparent mismatch rational.\n\n"
        "End your reply with EXACTLY one line starting with VERDICT: followed by minified JSON:\n"
        'VERDICT: {"hidden_event":"...","p_h":0.0,"model":{"a_given_h":0.0,"a_given_not_h":0.0,"b_given_h":0.0,'
        '"b_given_not_h":0.0},"fair_a":0.0,"fair_b":0.0,"mkt_a":0.0,"mkt_b":0.0,'
        '"mispricing":[{"leg":"A|B","dir":"cheap|rich","cents":0.0}],"structure":"...","risk":"...","confidence":0.0}'
    )
    r = subprocess.run([claude_bin(), "-p", "--output-format", "text",
                        "--allowedTools", "WebSearch", "WebFetch"],
                       input=prompt, capture_output=True, text=True, timeout=1200, encoding="utf-8")
    txt = r.stdout.strip()
    verdict = None
    for line in reversed(txt.splitlines()):
        if line.strip().startswith("VERDICT:"):
            try:
                verdict = json.loads(line.strip()[8:].strip())
            except Exception:
                pass
            break
    rec = {"t": int(time.time()), "a": {"venue": da["venue"], "id": da["id"], "bid": da["bid"], "ask": da["ask"]},
           "b": {"venue": db["venue"], "id": db["id"], "bid": db["bid"], "ask": db["ask"]},
           "verdict": verdict, "analysis": txt[-6000:]}
    with open(os.path.join(OUT, "hidden_links.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("=" * 100)
    print("A: %s  %.2f/%.2f | %s" % (da["id"], da["bid"], da["ask"], da["title"][:70]))
    print("B: %s  %.2f/%.2f | %s" % (str(db["id"])[:40], db["bid"], db["ask"], db["title"][:70]))
    print("-" * 100)
    print(txt[-3500:])
    return rec


def main():
    if "--pair" in sys.argv:
        i = sys.argv.index("--pair")
        analyze(sys.argv[i + 1], sys.argv[i + 2])
    elif "--kalshi" in sys.argv:
        i = sys.argv.index("--kalshi")
        analyze(sys.argv[i + 1], sys.argv[i + 2])
    elif "--from-watch" in sys.argv:
        import pandas as pd
        n = int(sys.argv[sys.argv.index("--from-watch") + 1])
        D = pd.read_csv(os.path.join(OUT, "consistency_log.csv"))
        rec = D[D.ts > time.time() - 48 * 3600]
        g = rec.groupby(["k_id", "p_id"]).edge.agg(["min", "max"])
        recent = set()
        hl = os.path.join(OUT, "hidden_links.jsonl")
        if os.path.exists(hl):
            for line in open(hl, encoding="utf-8"):
                try:
                    j = json.loads(line)
                    if time.time() - j["t"] < 3 * 86400:
                        recent.add((j["a"]["id"], j["b"]["id"]))
                except Exception:
                    pass
        movers = (g["max"] - g["min"]).sort_values(ascending=False).head(6 * n)
        done = 0
        for (k, p), mv in movers.items():
            if done >= n:
                break
            if (k, p) in recent:
                continue
            # skip resolved/closing markets: require a live two-sided Kalshi book and >24h to close
            try:
                m = signed_get("/trade-api/v2/markets/" + k).get("market", {})
                bid = float(m.get("yes_bid_dollars") or 0)
                ask = float(m.get("yes_ask_dollars") or 0)
                import datetime as _dt
                close = m.get("close_time") or "2000-01-01T00:00:00Z"
                hrs = (_dt.datetime.fromisoformat(close.replace("Z", "+00:00")).timestamp() - time.time()) / 3600
                if m.get("status") != "active" or not (0 < bid < 1 and 0 < ask < 1) or hrs < 24:
                    continue
            except Exception:
                continue
            done += 1
            print("\n### analyzing mover (%.1fc spread movement): %s <-> %s" % (100 * mv, k, p[:30]))
            try:
                analyze(k, p)
            except Exception as e:
                print("failed: %s" % e)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
