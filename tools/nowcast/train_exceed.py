"""Exceedance classifier: P(mid moves UP by >= X cents at horizon h) and P(DOWN by >= X) as two sigmoid heads on the
standard 43-input feature set (same MLP as the nowcast: 2x256 ReLU). Trained on 07-25..08-26, VAL 08-27..31 (selection),
TEST 09-01..04 (report). Label = mid at the horizon (fut column / +3 s side-file) minus mid now, in cents, at the
horizon (NOT the path maximum). For each (X, h): AUC of each head on VAL/TEST, base rate, and the realized
exceedance rate + mean signed move in the top-decile score bucket; a context-only model gives the AUC floor.
Usage: train_exceed.py [--x 1,1.5,2,3,5] [--h mid5,mid15,mid25,mid150] [--seed 0] [--epochs 8] [--ctx-only] [--tag T] [--save-pred DIR]"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--x", default="1,1.5,2,3,5"); ap.add_argument("--h", default="mid5,mid15,mid25,mid150")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--ctx-only", action="store_true"); ap.add_argument("--tag", default="ex"); ap.add_argument("--save-pred", default="")
ap.add_argument("--tte-min", type=float, default=60); ap.add_argument("--tte-max", type=float, default=300)
a = ap.parse_args()
torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed)
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
EPS = 1e-4
FEAT = "hist,kbs,kas,kmi,pbs,lf_flow_tfi_1s,lf_flow_tfi_5s,lf_flow_tfi_30s,lf_flow_vol_1s,lf_flow_n_1s,lf_flow_n_5s,lf_depth_band5,lf_depth_band10,lf_depth_band25,lf_depth_depth10,lf_depth_spread_bps,lf_depth_dimb5_1s,lf_depth_dimb5_5s,lf_depth_dband10_1s"
LAGS = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30])
FUTI = {"mid1": 0, "mid5": 1, "mid25": 2, "mid150": 3, "mid15": 4}
HNAME = {"mid1": "0.2s", "mid5": "1s", "mid15": "3s", "mid25": "5s", "mid150": "30s"}
XS = [float(x) for x in a.x.split(",")]; HS = a.h.split(",")


def lg(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def load(paths):
    X, MID, FUT, MK = [], [], [], []
    groups = FEAT.split(",")
    for di, p in enumerate(paths):
        z = np.load(p); tte = z["ctx"][:, 1]
        sel = np.where((tte >= a.tte_min) & (tte <= a.tte_max))[0]
        if len(sel) == 0:
            continue
        mid0, sig, spr = z["ctx"][sel, 0], z["ctx"][sel, 2], z["ctx"][sel, 3]; l0 = lg(mid0); F = []
        if not a.ctx_only:
            mh = z["mh"][sel][:, LAGS - 1].astype(np.float32)
            F.append(lg(mh) - l0[:, None]); F.append(z["ph"][sel][:, LAGS - 1].astype(np.float32) / sig[:, None])
            mx = z["mx"][sel]
            for g, col, scale in (("kbs", 0, 1), ("kas", 1, 1), ("kim", 2, 1), ("kmi", 3, 100), ("pbs", 4, 1), ("pas", 5, 1)):
                if g in groups:
                    F.append(scale * mx[:, col:col + 1])
            zl = np.load(os.path.join(SP, "midmove_v10_lake", os.path.basename(p).replace(".npz", ".lake.npz")))
            ln = [str(x) for x in zl["names"]]; want = [i for i, nm in enumerate(ln) if ("lf_" + nm) in groups]
            F.append(zl["lk"][sel][:, want].astype(np.float32))
        F.append(np.column_stack([np.log(np.maximum(tte[sel], 1)), sig, 100 * spr, l0, mid0 * (1 - mid0)]))
        fut4 = z["fut"][sel].reshape(len(sel), 4, 3)[:, :, 0]
        fp = os.path.join(SP, "midmove_v10_fut", os.path.basename(p).replace(".npz", ".fut.npz"))
        m3 = np.load(fp)["mid3s"][sel].astype(np.float32)
        X.append(np.concatenate(F, 1).astype(np.float32)); MID.append(mid0); FUT.append(np.column_stack([fut4, m3])); MK.append(di * 10000 + z["tk"][sel])
    return np.concatenate(X), np.concatenate(MID), np.concatenate(FUT), np.concatenate(MK)


days = sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz")))
split = {"tr": [], "va": [], "te": []}
for p in days:
    d = os.path.basename(p)[:10]
    split["tr" if d <= "2026-08-26" else "va" if d <= "2026-08-31" else "te"].append(p)
D = {k: load(v) for k, v in split.items()}
mu, sd = D["tr"][0].mean(0), D["tr"][0].std(0) + 1e-6
for k in D:
    D[k][0][:] = (D[k][0] - mu) / sd
NF = D["tr"][0].shape[1]
print("features %d (%s) | rows tr %d va %d te %d" % (NF, "ctx-only" if a.ctx_only else "standard 43", len(D["tr"][0]), len(D["va"][0]), len(D["te"][0])), flush=True)


def labels(k, h, x):
    X_, mid, fut, mk = D[k]
    mv = 100 * (fut[:, FUTI[h]] - mid)
    ok = np.isfinite(mv)
    return ok, mv, np.column_stack([(mv >= x), (mv <= -x)]).astype(np.float32)


def make_net():
    return nn.Sequential(nn.Linear(NF, a.hidden), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 2))


def predict(net, k):
    net.eval(); out = []
    with torch.no_grad():
        Xk = torch.from_numpy(D[k][0])
        for i in range(0, len(Xk), 65536):
            out.append(torch.sigmoid(net(Xk[i:i + 65536])).numpy())
    return np.concatenate(out)


def top_bucket(p, y, mv, q=0.9):
    thr = np.quantile(p, q); m = p >= thr
    return 100 * y[m].mean(), mv[m].mean(), m.sum()


print("%-5s %-4s | %-9s | %-31s | %-31s | %-22s | %s" % ("X", "h", "base% up/dn", "AUC up VAL/TEST", "AUC down VAL/TEST", "top-decile UP: rate%, move", "top-decile DOWN: rate%, move"))
for h in HS:
    for x in XS:
        oktr, _, ytr = labels("tr", h, x); okva, mvva, yva = labels("va", h, x); okte, mvte, yte = labels("te", h, x)
        Xt = torch.from_numpy(D["tr"][0][oktr]); Yt = torch.from_numpy(ytr[oktr])
        net = make_net(); opt = torch.optim.Adam(net.parameters(), lr=a.lr); lossf = nn.BCEWithLogitsLoss()
        best, bstate, bad = 1e9, None, 0
        for ep in range(a.epochs):
            net.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 8192):
                j = perm[i:i + 8192]; opt.zero_grad(); loss = lossf(net(Xt[j]), Yt[j]); loss.backward(); opt.step()
            pv = predict(net, "va")[okva]; v = float(lossf(torch.from_numpy(np.log(pv / (1 - pv) + 1e-12)), torch.from_numpy(yva[okva])))
            if v < best - 1e-6:
                best, bad, bstate = v, 0, {k_: t_.clone() for k_, t_ in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    break
        net.load_state_dict(bstate)
        pva, pte = predict(net, "va")[okva], predict(net, "te")[okte]
        yv, yt = yva[okva], yte[okte]; mv_v, mv_t = mvva[okva], mvte[okte]
        auc = lambda y, p: roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
        au_v, au_t = auc(yv[:, 0], pva[:, 0]), auc(yt[:, 0], pte[:, 0]); ad_v, ad_t = auc(yv[:, 1], pva[:, 1]), auc(yt[:, 1], pte[:, 1])
        ru, mu_, nu = top_bucket(pte[:, 0], yt[:, 0], mv_t); rd, md_, nd = top_bucket(pte[:, 1], yt[:, 1], -mv_t)
        print("%-5g %-4s | %4.1f / %4.1f | %.3f / %.3f (base %.3f)      | %.3f / %.3f                  | %5.1f%% (base %4.1f%%), %+5.2fc | %5.1f%% (base %4.1f%%), %+5.2fc" % (
            x, HNAME[h], 100 * yt[:, 0].mean(), 100 * yt[:, 1].mean(), au_v, au_t, 0.5, ad_v, ad_t, ru, 100 * yt[:, 0].mean(), mu_, rd, 100 * yt[:, 1].mean(), md_), flush=True)
        if a.save_pred:
            os.makedirs(a.save_pred, exist_ok=True)
            np.savez_compressed(os.path.join(a.save_pred, "%s_%s_x%g_s%d.npz" % (a.tag, h, x, a.seed)), p_va=predict(net, "va"), p_te=predict(net, "te"), okva=okva, okte=okte)
