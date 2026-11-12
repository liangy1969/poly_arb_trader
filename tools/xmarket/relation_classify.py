"""Classify each verified-RELATED pair's logical structure, so the consistency monitor can
check price bounds.

Relations we can monitor (A = Kalshi market, B = Polymarket market, prices = YES probability):
  a_implies_b   A is a subset of B (if A happens, B must happen)  =>  fair P(A) <= P(B)
  b_implies_a   B is a subset of A                                =>  fair P(B) <= P(A)
  exclusive     A and B cannot both happen                        =>  fair P(A) + P(B) <= 1
  none          related topically but no hard logical bound (correlation fodder only)
Reads data/xmarket/pairs_verified.json (verdict == related), asks `claude -p` in batches,
writes data/xmarket/relations.json.
Usage: relation_classify.py
"""
import json
import subprocess

def claude_bin():
    import glob as _g
    import os as _os
    import shutil as _sh
    for name in ("claude", "claude.cmd", "claude.exe"):
        p = _sh.which(name)
        if p:
            return p
    cands = sorted(_g.glob(_os.path.expanduser(r"~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude.exe")))
    if cands:
        return cands[-1]
    raise FileNotFoundError("claude CLI not found")


OUT = "data/xmarket"


def main():
    rows = [r for r in json.load(open(OUT + "/pairs_verified.json")) if r["verdict"] == "related"]
    print("classifying %d related pairs" % len(rows))
    rels = {}
    B = 40
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        prompt = (
            "For each numbered pair of prediction-market bets A and B, state the LOGICAL relation:\n"
            "a_implies_b  (if A resolves YES then B must resolve YES; A is the stricter/subset bet)\n"
            "b_implies_a  (the reverse)\n"
            "exclusive    (A and B cannot both resolve YES)\n"
            "none         (topically related, no hard logical implication)\n"
            "Judge ONLY from the wording; when in doubt answer none.\n"
            'One JSON line per pair: {"n": <number>, "rel": "a_implies_b|b_implies_a|exclusive|none", "why": "<six words max>"}\n'
            "Output ONLY JSON lines.\n\n"
            + "\n".join("%d. A: %s\n   B: %s" % (i + j + 1, b["k_title"], b["p_title"]) for j, b in enumerate(batch))
        )
        r = subprocess.run([claude_bin(), "-p", "--output-format", "text"], input=prompt,
                           capture_output=True, text=True, timeout=600, encoding="utf-8")
        for line in r.stdout.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
                n = int(j["n"]) - 1
                if 0 <= n < len(rows):
                    rels[n] = (j.get("rel", "none"), j.get("why", ""))
            except Exception:
                continue
        print("batch %d: %d classified" % (i // B + 1, len(rels)), flush=True)
    out = []
    for n, r in enumerate(rows):
        rel, why = rels.get(n, ("none", "unclassified"))
        rr = {k: r[k] for k in ("k_id", "p_id", "k_title", "p_title", "score")}
        rr["rel"], rr["why"] = rel, why
        out.append(rr)
    json.dump(out, open(OUT + "/relations.json", "w"), indent=1)
    from collections import Counter
    print("relations:", dict(Counter(r["rel"] for r in out)))


if __name__ == "__main__":
    main()
