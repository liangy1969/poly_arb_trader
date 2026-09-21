"""Directional evaluation of the exceedance classifiers (train_exceed.py --save-pred): the up/down AUCs are inflated by
volatility (both heads rise together in volatile states), so the tradable quantity is the DIRECTION score
s = P(up >= X) - P(down >= X). For each (X, h) and tag: (1) AUC of s for the sign of the move among rows with |move| >= X
(pure direction skill), (2) in the top/bottom 5% of s: realized rate of up/down exceedance, mean signed move (cents),
and net per trade after a taker cost (fee 0.07 p(1-p) at the row's mid + half a 1c spread), (3) the same at the 1%
tails of s. TEST rows only (VAL is the selection slice). Usage: exceed_eval.py TAG [TAG2 ...]"""
import glob
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
FUTI = {"mid1": 0, "mid5": 1, "mid25": 2, "mid150": 3, "mid15": 4}
HNAME = {"mid5": "1s", "mid15": "3s", "mid25": "5s", "mid150": "30s"}
days = sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz")))
te = [p for p in days if os.path.basename(p)[:10] >= "2026-09-01"]
MID, FUT = [], []
for p in te:
    z = np.load(p); tte = z["ctx"][:, 1]; sel = np.where((tte >= 60) & (tte <= 300))[0]
    fut4 = z["fut"][sel].reshape(len(sel), 4, 3)[:, :, 0]
    m3 = np.load(os.path.join(SP, "midmove_v10_fut", os.path.basename(p).replace(".npz", ".fut.npz")))["mid3s"][sel].astype(np.float32)
    MID.append(z["ctx"][sel, 0]); FUT.append(np.column_stack([fut4, m3]))
mid = np.concatenate(MID); fut = np.concatenate(FUT)
cost = 100 * (0.07 * mid * (1 - mid)) + 0.5                     # taker fee at the mid + half a 1c spread, cents
for tag in sys.argv[1:]:
    print("\n=== %s === (TEST 09-01..04; s = P(up) - P(down); cost = fee at mid + 0.5c)" % tag)
    print("%-5s %-4s | %-14s | %-38s | %-38s | %s" % ("X", "h", "dir AUC |mv|>=X", "top 5% of s: up-rate%, move, net/trade", "bottom 5%: down-rate%, move, net", "top/bottom 1%: net/trade"))
    for f in sorted(glob.glob("data/nowcast/exceed/%s_*_s0.npz" % tag), key=lambda f: (list(HNAME).index(os.path.basename(f).split("_")[1]), float(os.path.basename(f).split("_x")[1].split("_")[0]))):
        b = os.path.basename(f); h = b.split("_")[1]; x = float(b.split("_x")[1].split("_")[0])
        z = np.load(f); p = z["p_te"]; ok = z["okte"]
        mv = 100 * (fut[:, FUTI[h]] - mid); m = ok & np.isfinite(mv)
        s = p[:, 0] - p[:, 1]; s, mvk, ck = s[m], mv[m], cost[m]
        big = np.abs(mvk) >= x
        dauc = roc_auc_score((mvk[big] > 0).astype(int), s[big]) if big.sum() > 100 and 0 < (mvk[big] > 0).mean() < 1 else float("nan")
        hi, lo = np.quantile(s, 0.95), np.quantile(s, 0.05); h1, l1 = np.quantile(s, 0.99), np.quantile(s, 0.01)
        top, bot = s >= hi, s <= lo; t1, b1 = s >= h1, s <= l1
        print("%-5g %-4s | %.3f          | %5.1f%% %+5.2fc  net %+5.2fc (n=%6d)     | %5.1f%% %+5.2fc  net %+5.2fc (n=%6d)     | %+5.2fc / %+5.2fc" % (
            x, HNAME[h], dauc, 100 * (mvk[top] >= x).mean(), mvk[top].mean(), (mvk[top] - ck[top]).mean(), top.sum(),
            100 * (mvk[bot] <= -x).mean(), mvk[bot].mean(), (-mvk[bot] - ck[bot]).mean(), bot.sum(), (mvk[t1] - ck[t1]).mean(), (-mvk[b1] - ck[b1]).mean()))
