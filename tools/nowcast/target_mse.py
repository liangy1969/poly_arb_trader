"""For a model trained on target y (future mid at its horizon, or settlement):
MSE(y - prediction) vs MSE(y - current market mid), in cents^2, with the
market-clustered t of the per-row difference. Usage: target_mse.py LABEL TARGET pred1.npz [...]"""
import glob, os, sys
import numpy as np
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
FUTI = {"mid1": 0, "mid5": 1, "mid25": 2, "mid150": 3, "mid15": 4}
files = sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz")))
split = {"va": [f for f in files if "2026-08-27" <= os.path.basename(f)[:10] <= "2026-08-31"],
         "te": [f for f in files if os.path.basename(f)[:10] >= "2026-09-01"]}
def rows(paths):
    M, S, K, F, T, M1 = [], [], [], [], [], []
    for di, p in enumerate(paths):
        z = np.load(p); tte = z["ctx"][:, 1]; sel = np.where((tte >= 60) & (tte <= 300))[0]
        M.append(z["ctx"][sel, 0]); S.append(z["settle"][sel]); K.append(di * 10000 + z["tk"][sel])
        fut4 = z["fut"][sel].reshape(len(sel), 4, 3)[:, :, 0]
        fp = os.path.join(SP, "midmove_v10_fut", os.path.basename(p).replace(".npz", ".fut.npz"))
        m3 = np.load(fp)["mid3s"][sel].astype(np.float32) if os.path.exists(fp) else np.full(len(sel), np.nan, np.float32)
        F.append(np.column_stack([fut4, m3])); T.append(z["trig"][sel]); M1.append(z["mh"][sel][:, 0].astype(np.float32))
    return tuple(map(np.concatenate, (M, S, K, F, T, M1)))
def ct(d, mk):
    per = {}
    for kk, v in zip(mk, d):
        per.setdefault(kk, []).append(v)
    e = np.array([np.mean(v) for v in per.values()]); return e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
lab, tgt, preds = sys.argv[1], sys.argv[2], [np.load(f) for f in sys.argv[3:]]
for k, name in (("va", "VAL "), ("te", "TEST")):
    mid, st, mk, fut, tr, mid1 = rows(split[k]); L = np.mean([p["logit_" + k] for p in preds], 0); p = 1 / (1 + np.exp(-L))
    if tgt == "cur":                     # level target: current mid; naive = last-seen (lag-1) mid
        y, mid = mid, mid1
    else:
        y = st if tgt == "settle" else fut[:, FUTI[tgt]]
    out = ["%-11s %-6s %s" % (lab, tgt, name)]
    for sl, m in (("ALL ", np.isfinite(y)), ("TRIG", np.isfinite(y) & tr.astype(bool))):
        em, ek = 1e4 * (y[m] - p[m]) ** 2, 1e4 * (y[m] - mid[m]) ** 2; d = em - ek
        out.append("%s n=%6d  MSE(y-model) %8.3f  MSE(y-market) %8.3f  ratio %.4f  d %+7.4f c2 t %+5.2f"
                   % (sl, m.sum(), em.mean(), ek.mean(), em.mean() / ek.mean(), d.mean(), ct(d, mk[m])))
    print(" | ".join(out))
