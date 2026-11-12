"""The correlation analyst agent: deep cross-EVENT correlation analysis, done falsifiably.

Three subcommands (all log to data/xmarket/):

  graph     Ex-ante causal analysis from TITLES alone: the LLM (claude -p) organizes the
            tracked markets into driver groups — a shared driver variable, each market's
            direction and strength of dependence, and which market should react FIRST when
            driver news lands. Written to corr_theses.json as timestamped theses
            {driver, members:[{venue,id,sign,strength}], lead_id, why}. Ex-ante = testable.

  analyze   Data-driven candidates from snapshots.csv: for every tracked pair, count JOINT
            MOVE windows (both legs' mid moved >= 0.25c in the same 15-min window) and the
            sign-agreement rate there (Pearson on sparse books lies; sign agreement on joint
            moves does not). Pairs with >= --min-joint joint moves and agreement far from 50%
            are printed and appended to corr_candidates.json; genuinely surprising ones can
            be fed back to `graph` runs.

  validate  Grade every thesis ONLY on snapshot data AFTER its registration time: realized
            joint-move count, sign agreement vs the predicted sign structure, and lead-lag
            (does the predicted leader's move precede the followers' within 1-4 windows?).
            Prints a per-thesis scorecard; the registry keeps score history.

Usage: corr_agent.py graph|analyze|validate [--min-joint 6]
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = "data/xmarket"
THESES = os.path.join(OUT, "corr_theses.json")


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


def titles_map():
    t = {}
    for fn, venue in (("kalshi_markets.json", "kalshi"), ("polymarket_markets.json", "polymarket")):
        try:
            for m in json.load(open(os.path.join(OUT, fn))):
                t[(venue, m["id"])] = m["title"][:110]
        except FileNotFoundError:
            pass
    return t


def tracked_ids():
    ids = set()
    df = pd.read_csv(os.path.join(OUT, "snapshots.csv"))
    for v, i in df[["venue", "id"]].drop_duplicates().itertuples(index=False):
        ids.add((v, i))
    return ids, df


def mid_panel(df):
    """ts x (venue,id) panel of mids, forward-filled inside the observed span."""
    df = df[(df.yes_bid > 0) & (df.yes_ask > 0) & (df.yes_ask < 1)].copy()
    df["mid"] = 0.5 * (df.yes_bid + df.yes_ask)
    df["key"] = list(zip(df.venue, df.id))
    p = df.pivot_table(index="ts", columns="key", values="mid", aggfunc="last").sort_index()
    return p.ffill(limit=4)


def cmd_graph():
    tm = titles_map()
    ids, _ = tracked_ids()
    rows = ["%s|%s|%s" % (v, str(i)[:44], tm.get((v, i), "?")) for v, i in sorted(ids) if tm.get((v, i))]
    print("graphing %d tracked markets" % len(rows))
    prompt = (
        "You are a cross-market analyst. Below are active prediction markets (venue|id|title).\n"
        "Organize them into DRIVER GROUPS: a group = markets whose prices share one underlying driver "
        "variable (e.g. 'Fed policy path', 'BTC price level', 'Democratic 2028 field', 'SpaceX execution').\n"
        "Only group markets with a REAL economic/causal link, not surface word overlap. 3-15 markets per "
        "group; skip markets with no good group. For each group output one JSON line:\n"
        '{"driver":"<short name>","why":"<one clause>","members":[{"venue":"...","id":"...","sign":"+|-"}],'
        '"lead":{"venue":"...","id":"..."},"lead_why":"<one clause>"}\n'
        "sign = direction the market\'s YES price moves when the driver INCREASES/strengthens. "
        "lead = the single member whose price should react FIRST to driver news (most liquid / most "
        "directly settled on the driver). Output ONLY JSON lines.\n\n" + "\n".join(rows[:700])
    )
    r = subprocess.run([claude_bin(), "-p", "--output-format", "text"], input=prompt,
                       capture_output=True, text=True, timeout=900, encoding="utf-8")
    theses = []
    if os.path.exists(THESES):
        theses = json.load(open(THESES))
    n0 = len(theses)
    for line in r.stdout.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
            if j.get("driver") and j.get("members"):
                j["t"] = int(time.time())
                j["scores"] = []
                theses.append(j)
        except Exception:
            continue
    json.dump(theses, open(THESES, "w"), indent=1)
    print("registered %d new driver theses (total %d)" % (len(theses) - n0, len(theses)))
    for th in theses[n0:]:
        print("  [%s] %d members, lead=%s | %s" % (th["driver"], len(th["members"]), th.get("lead", {}).get("id", "?")[:30], th.get("why", "")[:60]))


def joint_moves(p, a, b, thr=0.0025):
    da, db = p[a].diff(), p[b].diff()
    m = da.notna() & db.notna() & (da.abs() >= thr) & (db.abs() >= thr)
    if m.sum() == 0:
        return 0, 0.5
    agree = (np.sign(da[m]) == np.sign(db[m])).mean()
    return int(m.sum()), float(agree)


def cmd_analyze(min_joint):
    _, df = tracked_ids()
    p = mid_panel(df)
    cols = [c for c in p.columns if p[c].diff().abs().ge(0.0025).sum() >= 3]   # markets that actually move
    print("snapshots: %d | tracked: %d | moving: %d" % (len(p), p.shape[1], len(cols)))
    tm = titles_map()
    out = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            n, agree = joint_moves(p, cols[i], cols[j])
            if n >= min_joint and abs(agree - 0.5) >= 0.3:
                out.append({"a": cols[i], "b": cols[j], "joint": n, "agree": round(agree, 3)})
    out.sort(key=lambda r: -(r["joint"] * abs(r["agree"] - 0.5)))
    for r in out[:20]:
        print("  joint=%2d agree=%.2f | %s <-> %s" % (r["joint"], r["agree"],
              tm.get(tuple(r["a"]), str(r["a"]))[:52], tm.get(tuple(r["b"]), str(r["b"]))[:52]))
    path = os.path.join(OUT, "corr_candidates.json")
    hist = json.load(open(path)) if os.path.exists(path) else []
    hist.append({"t": int(time.time()), "n_snapshots": len(p), "candidates": out[:100]})
    json.dump(hist, open(path, "w"), indent=1)
    print("candidates this pass: %d (appended to corr_candidates.json)" % len(out))


def cmd_validate():
    if not os.path.exists(THESES):
        print("no theses registered")
        return
    theses = json.load(open(THESES))
    _, df = tracked_ids()
    p = mid_panel(df)
    now = int(time.time())
    for th in theses:
        sub = p[p.index > th["t"]]
        mem = [(m["venue"], m["id"]) for m in th["members"] if (m["venue"], m["id"]) in p.columns]
        if len(sub) < 8 or len(mem) < 2:
            print("[%s] not yet scoreable (%d post-thesis snapshots, %d tracked members)" % (th["driver"], len(sub), len(mem)))
            continue
        signs = {(m["venue"], m["id"]): (1 if m.get("sign", "+") == "+" else -1) for m in th["members"]}
        # pairwise sign-agreement among members, adjusted for predicted signs
        ns, ok = 0, 0.0
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                n, agree = joint_moves(sub, mem[i], mem[j])
                if n >= 3:
                    pred_same = signs[mem[i]] == signs[mem[j]]
                    ok += (agree if pred_same else 1 - agree) * n
                    ns += n
        # lead-lag: does the lead's move predict followers' NEXT-window move?
        lead = (th.get("lead", {}).get("venue"), th.get("lead", {}).get("id"))
        lead_score = None
        if lead in p.columns:
            dl = sub[lead].diff()
            fcols = [m for m in mem if m != lead]
            vals = []
            for f in fcols:
                dfol = sub[f].diff().shift(-1)   # follower's next window
                m_ = dl.notna() & dfol.notna() & (dl.abs() >= 0.0025)
                if m_.sum() >= 3:
                    s = signs[f] * signs[lead]
                    vals.append((np.sign(dl[m_]) == s * np.sign(dfol[m_])).mean())
            if vals:
                lead_score = float(np.mean(vals))
        score = {"t": now, "n_joint": ns, "sign_ok": round(ok / ns, 3) if ns else None,
                 "lead_next_agree": round(lead_score, 3) if lead_score is not None else None}
        th.setdefault("scores", []).append(score)
        print("[%s] joint-moves %d | predicted-sign agreement %s | lead->next-window agreement %s" % (
            th["driver"], ns, score["sign_ok"], score["lead_next_agree"]))
    json.dump(theses, open(THESES, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "graph":
        cmd_graph()
    elif cmd == "validate":
        cmd_validate()
    else:
        mj = 6
        if "--min-joint" in sys.argv:
            mj = int(sys.argv[sys.argv.index("--min-joint") + 1])
        cmd_analyze(mj)
