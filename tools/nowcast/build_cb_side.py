"""Per-day row alignment v10 vs v11 (match on ticker + ts) and build a coinbase side-file keyed to the v10 rows:
midmove_v10_cb/<day>.cb.npz  with  ch (v11 coinbase lag history) and sig_cb (v11 ctx[:,6]); NaN where the v10 row has
no v11 twin."""
import glob, os, sys
import numpy as np
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
OUT = os.path.join(SP, "midmove_v10_cb"); os.makedirs(OUT, exist_ok=True)
tot10 = tot11 = totm = 0
for p in sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz"))):
    d = os.path.basename(p)
    q = os.path.join(SP, "midmove_v11", d)
    if not os.path.exists(q):
        print(d[:10], "v11 MISSING"); continue
    a, b = np.load(p), np.load(q)
    na, nb = np.array([str(x) for x in a["names"]]), np.array([str(x) for x in b["names"]])
    ka = np.char.add(np.char.add(na[a["tk"]], "|"), a["ts"].astype(np.int64).astype(str))
    kb = np.char.add(np.char.add(nb[b["tk"]], "|"), b["ts"].astype(np.int64).astype(str))
    idx = {k: i for i, k in enumerate(kb.tolist())}
    j = np.array([idx.get(k, -1) for k in ka.tolist()])
    m = int((j >= 0).sum())
    same = len(ka) == len(kb) and bool(np.all(ka == kb))
    tot10 += len(ka); tot11 += len(kb); totm += m
    ch = np.full((len(ka),) + b["ch"].shape[1:], np.nan, np.float32); sg = np.full(len(ka), np.nan, np.float32)
    ok = j >= 0
    ch[ok] = b["ch"][j[ok]]; sg[ok] = b["ctx"][j[ok], 6]
    # sanity: the perp history must agree where matched
    dp = np.nanmax(np.abs(a["ph"][ok][:, :5] - b["ph"][j[ok]][:, :5])) if ok.any() else np.nan
    np.savez_compressed(os.path.join(OUT, d.replace(".npz", ".cb.npz")), ch=ch, sig_cb=sg)
    print("%s v10 %6d v11 %6d  match %6d (%.1f%% of v10)  identical %s  ch %s  max|dph| %.3g  sig_cb med %.2f  ch nan %.1f%%" % (
        d[:10], len(ka), len(kb), m, 100 * m / len(ka), same, b["ch"].shape[1:], dp, np.nanmedian(sg), 100 * np.isnan(ch[ok]).mean()))
print("TOTAL v10 %d v11 %d match %d (%.2f%% of v10)" % (tot10, tot11, totm, 100 * totm / tot10))
