"""LLM verification of matched pairs: are the two titles the SAME bet, or merely related?

Reads data/xmarket/scan_result.json (arbs + dislocations from spread_scan), sends title pairs
to `claude -p` in batches, and writes data/xmarket/pairs_verified.json with a verdict per pair:
  identical — same event, same entity, same threshold/direction; a price gap is real arb material
  related   — logically linked but different bets (subset, finalist-vs-winner, and/or different
              threshold); useful for the correlation/consistency layer, NOT for two-leg arb
  different — the matcher was wrong
Usage: verify_pairs.py
"""
import json
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


OUT = "data/xmarket"


def ask_claude(prompt):
    r = subprocess.run([claude_bin(), "-p", "--output-format", "text"], input=prompt,
                       capture_output=True, text=True, timeout=600, encoding="utf-8")
    return r.stdout


def main():
    scan = json.load(open(OUT + "/scan_result.json"))
    rows = scan["arbs"] + scan["dislocations"]
    seen, uniq = set(), []
    for r in rows:
        k = (r["k_id"], r["p_id"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    print("verifying %d pairs" % len(uniq))
    verdicts = {}
    B = 40
    for i in range(0, len(uniq), B):
        batch = uniq[i:i + B]
        prompt = (
            "For each numbered pair of prediction-market titles, judge whether they are bets on the "
            "IDENTICAL outcome (same event, same entity/person, same threshold and direction), merely "
            "RELATED (logically linked but different bets: subset/superset like 'impeached' vs 'impeached "
            "and removed', finalist vs winner, different threshold or date), or DIFFERENT (about different "
            "entities or events entirely).\n"
            'Answer with one JSON line per pair: {"n": <number>, "v": "identical|related|different", '
            '"why": "<five words max>"}\nOutput ONLY JSON lines.\n\n'
            + "\n".join("%d. A: %s\n   B: %s" % (i + j + 1, b["k_title"], b["p_title"]) for j, b in enumerate(batch))
        )
        out = ask_claude(prompt)
        for line in out.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                j = json.loads(line)
                n = int(j["n"]) - 1
                if 0 <= n < len(uniq):
                    verdicts[(uniq[n]["k_id"], uniq[n]["p_id"])] = (j.get("v", "?"), j.get("why", ""))
            except Exception:
                continue
        print("batch %d: %d verdicts so far" % (i // B + 1, len(verdicts)), flush=True)
    out_rows = []
    for r in uniq:
        v = verdicts.get((r["k_id"], r["p_id"]), ("unverified", ""))
        rr = dict(r)
        rr["verdict"], rr["why"] = v
        out_rows.append(rr)
    json.dump(out_rows, open(OUT + "/pairs_verified.json", "w"), indent=1)
    from collections import Counter
    print("verdicts:", dict(Counter(r["verdict"] for r in out_rows)))
    print("\n=== VERIFIED-IDENTICAL pairs with the largest edges/gaps ===")
    ident = [r for r in out_rows if r["verdict"] == "identical"]
    ident.sort(key=lambda r: -(r.get("edge_c") or r.get("gap_c") or 0))
    for r in ident[:15]:
        print("  %s %5.1fc | K %0.2f/%0.2f  P %0.2f/%0.2f | %s" % (
            "EDGE" if r.get("edge_c", -1) >= 0.5 else "gap ",
            r.get("edge_c") if r.get("edge_c", -1) >= 0.5 else r.get("gap_c", 0),
            r["k_bid"], r["k_ask"], r["p_bid"], r["p_ask"], r["k_title"][:70]))
    rel = [r for r in out_rows if r["verdict"] == "related"]
    print("\nrelated (consistency/correlation fodder): %d pairs -> kept in pairs_verified.json" % len(rel))


if __name__ == "__main__":
    main()
