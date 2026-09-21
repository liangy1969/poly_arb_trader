"""Paired comparison of exceedance classifiers by the user's main metric: PRECISION AT THE TOP k% of the score
(realized exceedance rate among the k% highest-scored rows), per head (up / down) and for the direction score
s = P(up) - P(down) (top k% of s: rate of up-exceedance; bottom k%: rate of down-exceedance). Mean over seeds with
the across-seed sd, and the paired per-seed difference B - A with its t. VAL (selection) and TEST (report) columns.
Usage: exceed_cmp.py TAG_A TAG_B [--ks 0.5,1,5] [--seeds 0,1,2] [--h mid5,mid25] [--x 1,2,3]"""
import argparse
import glob
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("tags", nargs="+"); ap.add_argument("--ks", default="0.5,1,5"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--h", default="mid5,mid25"); ap.add_argument("--x", default="1,2,3"); ap.add_argument("--data", default="midmove_v10")
a = ap.parse_args()
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
FUTI = {"mid1": 0, "mid5": 1, "mid25": 2, "mid150": 3, "mid15": 4}
HNAME = {"mid1": "0.2s", "mid5": "1s", "mid15": "3s", "mid25": "5s", "mid150": "30s"}
KS = [float(k) for k in a.ks.split(",")]; SEEDS = [int(s) for s in a.seeds.split(",")]


def rows(lo, hi):
    MID, FUT = [], []
    for p in sorted(glob.glob(os.path.join(SP, a.data, "*.npz"))):
        d = os.path.basename(p)[:10]
        if not (lo <= d <= hi):
            continue
        z = np.load(p); tte = z["ctx"][:, 1]; sel = np.where((tte >= 60) & (tte <= 300))[0]
        fut4 = z["fut"][sel].reshape(len(sel), 4, 3)[:, :, 0]
        m3 = np.load(os.path.join(SP, "midmove_v10_fut", os.path.basename(p).replace(".npz", ".fut.npz")))["mid3s"][sel].astype(np.float32)
        MID.append(z["ctx"][sel, 0]); FUT.append(np.column_stack([fut4, m3]))
    return np.concatenate(MID), np.concatenate(FUT)


VA, TE = rows("2026-08-27", "2026-08-31"), rows("2026-09-01", "2026-09-04")


def prec(score, y, k, top=True):
    q = np.quantile(score, 1 - k / 100) if top else np.quantile(score, k / 100)
    m = score >= q if top else score <= q
    return 100 * y[m].mean()


def metrics(tag, seed, h, x):
    f = "data/nowcast/exceed/%s_%s_x%g_s%d.npz" % (tag, h, x, seed)
    if not os.path.exists(f):
        return None
    z = np.load(f); out = {}
    for sl, (mid, fut), pk, okk in (("VAL", VA, "p_va", "okva"), ("TEST", TE, "p_te", "okte")):
        p, ok = z[pk], z[okk]; mv = 100 * (fut[:, FUTI[h]] - mid); m = ok & np.isfinite(mv)
        p, mv = p[m], mv[m]; yu, yd = mv >= x, mv <= -x; s = p[:, 0] - p[:, 1]
        for k in KS:
            out[(sl, "up", k)] = prec(p[:, 0], yu, k); out[(sl, "dn", k)] = prec(p[:, 1], yd, k)
            out[(sl, "s+", k)] = prec(s, yu, k); out[(sl, "s-", k)] = prec(s, yd, k, top=False)
        out[(sl, "base_up")] = 100 * yu.mean(); out[(sl, "base_dn")] = 100 * yd.mean()
    return out


print("precision (%%) at the top k%% of the score, mean over seeds %s (sd); rows tte 60-300 s; VAL 08-27..31 | TEST 09-01..04" % SEEDS)
for h in a.h.split(","):
    for x in [float(v) for v in a.x.split(",")]:
        M = {t: [m for m in (metrics(t, s, h, x) for s in SEEDS) if m is not None] for t in a.tags}
        if any(len(M[t]) == 0 for t in a.tags):
            print("\n%s X=%gc: missing preds for %s" % (HNAME[h], x, [t for t in a.tags if not M[t]])); continue
        b = M[a.tags[0]][0]
        print("\n=== h %s  X %gc   base up %.1f%% / dn %.1f%% (TEST)   seeds per tag: %s" % (HNAME[h], x, b[("TEST", "base_up")], b[("TEST", "base_dn")], {t: len(M[t]) for t in a.tags}))
        print("%-10s | %-6s | %-34s | %-34s" % ("head@k%", "tag", "VAL   mean (sd)   [diff vs first, t]", "TEST  mean (sd)   [diff vs first, t]"))
        for head in ("up", "dn", "s+", "s-"):
            for k in KS:
                for t in a.tags:
                    cells = []
                    for sl in ("VAL", "TEST"):
                        v = np.array([m[(sl, head, k)] for m in M[t]])
                        txt = "%5.1f (%4.1f)" % (v.mean(), v.std(ddof=1) if len(v) > 1 else 0)
                        if t != a.tags[0]:
                            v0 = np.array([m[(sl, head, k)] for m in M[a.tags[0]]]); n = min(len(v), len(v0)); d = v[:n] - v0[:n]
                            tt = d.mean() / (d.std(ddof=1) / np.sqrt(n)) if n > 1 and d.std() > 0 else 0.0
                            txt += "   [%+5.1f  t%+4.1f]" % (d.mean(), tt)
                        cells.append("%-34s" % txt)
                    print("%-10s | %-6s | %s | %s" % ("%s@%g" % (head, k), t, cells[0], cells[1]))
