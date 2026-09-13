"""Outcome study: can a learned fair BEAT the observed market on settlement BCE?

  --target   settle | mid1 | mid5 | mid25 | mid150   (mid = future mid at +0.2/1/5/30s)
  --market   none | feature | residual  (residual: logit p = logit(mid_t) + f(x))
  --feat     hist,book,bookl,kz,act      (ctx = log tte, sigma, spread always)
  --arch     mlp | had | film | gbm       (gbm: LightGBM binary, init_score = logit mid, settle only)
  --pretrain mid1 --pre-epochs N          (two-stage: mid target first, then settle fine-tune at --ft-lr)
Selection on VAL outcome BCE (08-27..31); TEST = Sept. Reports outcome BCE of
model vs market vs (d, market-clustered t) on ALL / TRIG, plus market-KL.
"""
import argparse
import gc
import glob
import os
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="settle"); ap.add_argument("--market", default="residual")
ap.add_argument("--feat", default="hist,book,bookl"); ap.add_argument("--arch", default="mlp")
ap.add_argument("--lookback", type=float, default=6); ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--patience", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--pretrain", default=""); ap.add_argument("--pre-epochs", type=int, default=6)
ap.add_argument("--ft-lr", type=float, default=2e-4)
ap.add_argument("--ft-head", action="store_true", help="fine-tune stage: train only the output layer")
ap.add_argument("--tag", default=""); ap.add_argument("--save-pred", default="")
ap.add_argument("--save-model", default="", help="torch checkpoint (weights + standardization + spec)")
ap.add_argument("--tte-min", type=float, default=60); ap.add_argument("--tte-max", type=float, default=300)
ap.add_argument("--lags", default="", help="comma list of lag ticks (200ms units) overriding --lookback")
ap.add_argument("--init", default="", help="checkpoint to initialise from (fine-tune on --target at --ft-lr; VAL-guarded: never worse than the init)")
ap.add_argument("--cpu", action="store_true", help="train on CPU (laptop GPU thermal limit)")
a = ap.parse_args()
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
DEV = "cpu" if a.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4); torch.manual_seed(a.seed)
EPS = 1e-4
LAGS = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300])
use = np.array([int(x) for x in a.lags.split(",")]) if a.lags else LAGS[LAGS <= a.lookback * 5 + 0.5]
GROUPS = set(a.feat.split(","))
FUTI = {"mid1": 0, "mid5": 1, "mid25": 2, "mid150": 3, "mid15": 4}   # mid15 = +3 s from midmove_v10_fut (build_fut.py)
# minimal print-only lake subsets (aggTrade-reproducible live): top permutation-importance features
SUB = {"tfi": ["flow_tfi_1s", "flow_tfi_5s", "flow_tfi_30s"], "vol": ["flow_vol_1s", "flow_vol_5s", "flow_vol_30s"],
       "n": ["flow_n_1s", "flow_n_5s"], "fother": ["flow_big_30s", "flow_burst", "flow_vwap1_bps"],
       "dimb": ["depth_imb1", "depth_imb5", "depth_imb20", "depth_micro_bps"],
       "dband": ["depth_band5", "depth_band10", "depth_band25", "depth_depth10", "depth_spread_bps"],
       "ddelta": ["depth_dimb5_1s", "depth_dimb5_5s", "depth_dband10_1s"], "liq": ["liq_buy_30s", "liq_sell_30s", "liq_net_300s"]}
LAKEG = lambda G: any(g in {"lake", "lflow", "ldepth", "lliq"} or g in LMIN or g.startswith("lake-") or g.startswith("lsub_") or g.startswith("lf_") for g in G)
LMIN = {"lmin2": ["flow_n_1s", "flow_tfi_1s"], "lmin5": ["flow_n_1s", "flow_tfi_1s", "flow_tfi_5s", "flow_vwap1_bps", "flow_burst"]}
days = sorted(glob.glob(os.path.join(SP, "midmove_v10", "*.npz")))
split = {"tr": [], "va": [], "te": []}
for p in days:
    d = os.path.basename(p)[:10]
    split["tr" if d <= "2026-08-26" else "va" if d <= "2026-08-31" else "te"].append(p)


def lg(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def load(paths):
    X, MID, SET, FUT, MK, TR_, M1 = [], [], [], [], [], [], []
    for di, p in enumerate(paths):
        z = np.load(p); tte = z["ctx"][:, 1]
        sel = np.where((tte >= a.tte_min) & (tte <= a.tte_max))[0]
        if len(sel) == 0:
            continue
        mid0, sig, spr = z["ctx"][sel, 0], z["ctx"][sel, 2], z["ctx"][sel, 3]; l0 = lg(mid0); F = []
        if "hist" in GROUPS:
            mh = z["mh"][sel][:, use - 1].astype(np.float32)
            F.append(lg(mh) - l0[:, None]); F.append(z["ph"][sel][:, use - 1].astype(np.float32) / sig[:, None])
        if "hmid" in GROUPS:             # mid-history lags only (pruning)
            mh = z["mh"][sel][:, use - 1].astype(np.float32); F.append(lg(mh) - l0[:, None])
        if "hperp" in GROUPS:            # perp-history lags only (pruning)
            F.append(z["ph"][sel][:, use - 1].astype(np.float32) / sig[:, None])
        if "histp" in GROUPS:            # LEVEL model: history vs the LAG-1 mid (no current-mid leak), perp vs now
            mh = z["mh"][sel][:, use - 1].astype(np.float32); l1_ = lg(mh[:, 0])
            F.append(lg(mh) - l1_[:, None]); F.append(z["ph"][sel][:, use - 1].astype(np.float32) / sig[:, None])
        if "histraw" in GROUPS:          # price-history-model representation: raw levels, zero-meaned cents
            mh = z["mh"][sel][:, use - 1].astype(np.float32)
            F.append(100 * (mh - 0.5)); F.append(z["ph"][sel][:, use - 1].astype(np.float32) / sig[:, None])
        mx = z["mx"][sel]
        if "book" in GROUPS:
            F.append(np.column_stack([mx[:, 0], mx[:, 1], mx[:, 2], 100 * mx[:, 3], mx[:, 4], mx[:, 5], mx[:, 6]]))
        # venue split of the book block: Kalshi (sizes, imbalance, microprice) vs perp (sizes, imbalance)
        if "kbook" in GROUPS:
            F.append(np.column_stack([mx[:, 0], mx[:, 1], mx[:, 2], 100 * mx[:, 3]]))
        if "pbook" in GROUPS:
            F.append(np.column_stack([mx[:, 4], mx[:, 5], mx[:, 6]]))
        if "ksz" in GROUPS:
            F.append(np.column_stack([mx[:, 0], mx[:, 1]]))
        if "kimb" in GROUPS:
            F.append(np.column_stack([mx[:, 2], 100 * mx[:, 3]]))
        if "psz" in GROUPS:
            F.append(np.column_stack([mx[:, 4], mx[:, 5]]))
        if "pimb" in GROUPS:
            F.append(mx[:, 6:7])
        # single book columns (pruning): kbs/kas = Kalshi bid/ask size, kim = Kalshi imbalance, kmi = Kalshi microprice, pbs/pas = perp sizes
        for g, col, scale in (("kbs", 0, 1), ("kas", 1, 1), ("kim", 2, 1), ("kmi", 3, 100), ("pbs", 4, 1), ("pas", 5, 1)):
            if g in GROUPS:
                F.append(scale * mx[:, col:col + 1])
        # lake side-file groups (build_lake.py): lake = all | lflow = flow_* | ldepth = depth_* | lliq = liq_*
        if LAKEG(GROUPS):
            zl = np.load(os.path.join(SP, "midmove_v10_lake", os.path.basename(p).replace(".npz", ".lake.npz"))); ln = [str(x) for x in zl["names"]]; LK = zl["lk"][sel]
            selg = set()
            for g in GROUPS:
                if g == "lake": selg |= set(ln)
                elif g == "lflow": selg |= {n for n in ln if n.startswith("flow_")}
                elif g == "ldepth": selg |= {n for n in ln if n.startswith("depth_")}
                elif g == "lliq": selg |= {n for n in ln if n.startswith("liq_")}
                elif g in LMIN: selg |= set(LMIN[g])
                elif g.startswith("lake-"): selg |= set(ln) - set(SUB[g[5:]])
                elif g.startswith("lsub_"): selg |= set(SUB[g[5:]])
                elif g.startswith("lf_"): selg.add(g[3:])          # one lake feature by name
            want = [i for i, nm in enumerate(ln) if nm in selg]
            F.append(LK[:, want].astype(np.float32))
        if "act" in GROUPS:
            F.append(np.column_stack([mx[:, 7] / 50.0, mx[:, 8], mx[:, 9] / 10.0, mx[:, 10]]))
        if "bookl" in GROUPS:
            bl = z["bl"][sel].copy(); bl[:, 7:14] *= 100; bl[:, 21:35] *= 100; F.append(bl)
        if "kz" in GROUPS:
            kz = z["kz"][sel].copy(); kz[:, 0:2] /= 100.0; kz[:, 4:6] *= 100.0; F.append(kz)
        mid1 = z["mh"][sel][:, 0].astype(np.float32)
        if a.market == "prev":           # context at lag 1 only: spread, logit and variance of the lag-1 mid
            l1_ = lg(mid1); cx = [np.log(np.maximum(tte[sel], 1)), sig, 100 * z["sh"][sel][:, 0].astype(np.float32), l1_, mid1 * (1 - mid1)]
        else:
            cx = [np.log(np.maximum(tte[sel], 1)), sig, 100 * spr]
            if a.market in ("feature", "residual"):
                cx += [l0, mid0 * (1 - mid0)]
        F.append(np.column_stack(cx))
        X.append(np.concatenate(F, 1).astype(np.float32)); MID.append(mid0)
        SET.append(z["settle"][sel])
        fut4 = z["fut"][sel].reshape(len(sel), 4, 3)[:, :, 0]
        fp = os.path.join(SP, "midmove_v10_fut", os.path.basename(p).replace(".npz", ".fut.npz"))
        m3 = np.load(fp)["mid3s"][sel].astype(np.float32) if os.path.exists(fp) else np.full(len(sel), np.nan, np.float32)
        FUT.append(np.column_stack([fut4, m3]))
        MK.append(di * 10000 + z["tk"][sel]); TR_.append(z["trig"][sel]); M1.append(mid1); del z
    c = np.concatenate
    return c(X), c(MID), c(SET), c(FUT), c(MK), c(TR_).astype(bool), c(M1)


D = {k: load(split[k]) for k in ("tr", "va", "te")}
gc.collect()
mu, sd = D["tr"][0].mean(0), D["tr"][0].std(0) + 1e-6
for k in D:
    D[k][0][:] = (D[k][0] - mu) / sd


def targets(k, tgt):
    X, mid, st, fut, mk, tr = D[k][:6]
    if tgt == "settle":
        ok = np.isfinite(st); return ok, st
    if tgt == "cur":                       # level target: the CURRENT mid
        return np.ones(len(mid), bool), mid
    y = fut[:, FUTI[tgt]]; ok = np.isfinite(y); return ok, y


NF = D["tr"][0].shape[1]
def basev(k):
    """market logit added to the net output: current mid (residual) | lag-1 mid (prev) | 0"""
    if a.market == "residual":
        return torch.from_numpy(lg(D[k][1]).astype(np.float32))
    if a.market == "prev":
        return torch.from_numpy(lg(D[k][6]).astype(np.float32))
    return torch.zeros(len(D[k][1]))
NCTX = 3 + (2 if a.market != "none" else 0)
SIDX = list(range(NF - NCTX, NF))


class HAD(nn.Module):
    def __init__(s, nin, h, drop):
        super().__init__(); s.a = nn.Linear(nin, h); s.b = nn.Linear(nin, h); s.c = nn.Linear(nin, h)
        s.drop = nn.Dropout(drop); s.h2 = nn.Linear(h, h); s.out = nn.Linear(h, 1)

    def forward(s, x):
        return s.out(torch.relu(s.h2(s.drop(torch.relu(s.c(x) + s.a(x) * s.b(x))))))


class FILM(nn.Module):
    def __init__(s, nin, sidx, h, drop):
        super().__init__(); s.register_buffer("sidx", torch.tensor(sidx, dtype=torch.long))
        s.g = nn.Linear(len(sidx), h); s.b = nn.Linear(len(sidx), h); s.f = nn.Linear(nin, h)
        s.drop = nn.Dropout(drop); s.h2 = nn.Linear(h, h); s.out = nn.Linear(h, 1)

    def forward(s, x):
        st = x[:, s.sidx]; z = torch.relu(s.f(x)) * (1 + torch.tanh(s.g(st))) + s.b(st)
        return s.out(torch.relu(s.h2(s.drop(z))))


def make_net():
    if a.arch == "had":
        return HAD(NF, a.hidden, a.dropout).to(DEV)
    if a.arch == "film":
        return FILM(NF, SIDX, a.hidden, a.dropout).to(DEV)
    return nn.Sequential(nn.Linear(NF, a.hidden), nn.ReLU(), nn.Dropout(a.dropout),
                         nn.Linear(a.hidden, a.hidden), nn.ReLU(), nn.Linear(a.hidden, 1)).to(DEV)


def bce_np(logit, y):
    p = np.clip(1 / (1 + np.exp(-logit)), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def predict_logit(net, k):
    X, mid = D[k][0], D[k][1]; Xt = torch.from_numpy(X); out = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(Xt), 65536):
            out.append(net(Xt[i:i + 65536].to(DEV)).squeeze(1).cpu())
    return (torch.cat(out) + basev(k)).numpy()


def loss_fn(net, X, b, y, tgt, mid):
    f = net(X).squeeze(1) + b
    if tgt == "settle":
        return nn.functional.binary_cross_entropy_with_logits(f, y)
    pc = 100 * (torch.sigmoid(f) - mid); return nn.functional.mse_loss(pc, 100 * (y - mid))


def train(net, tgt, epochs, lr, patience, head_only=False, guard=False):
    ok, y = targets("tr", tgt); X = torch.from_numpy(D["tr"][0][ok]); yt = torch.from_numpy(y[ok].astype(np.float32))
    b = basev("tr")[ok]; mid = torch.from_numpy(D["tr"][1][ok].astype(np.float32))
    okv, yv = targets("va", tgt)
    params = [q for n_, q in net.named_parameters() if (not head_only) or n_.startswith("5.") or n_.startswith("out.")]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=a.wd)
    best, bstate, bad, ep_used = 1e18, None, 0, 0
    if guard:                                    # fine-tuning: the initial model is the incumbent
        lv = predict_logit(net, "va")[okv]
        best = float(bce_np(lv, yv[okv]).mean()) if tgt == "settle" else float(np.mean((100 * (1 / (1 + np.exp(-lv)) - D["va"][1][okv]) - 100 * (yv[okv] - D["va"][1][okv])) ** 2))
        bstate = {k_: x.detach().clone() for k_, x in net.state_dict().items()}
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(yt))
        for i in range(0, len(yt), 8192):
            j = perm[i:i + 8192]; opt.zero_grad()
            loss = loss_fn(net, X[j].to(DEV), b[j].to(DEV), yt[j].to(DEV), tgt, mid[j].to(DEV))
            loss.backward(); opt.step()
        lv = predict_logit(net, "va")[okv]
        v = float(bce_np(lv, yv[okv]).mean()) if tgt == "settle" else \
            float(np.mean((100 * (1 / (1 + np.exp(-lv)) - D["va"][1][okv]) - 100 * (yv[okv] - D["va"][1][okv])) ** 2))
        ep_used = ep + 1
        if v < best - 1e-7:
            best, bad = v, 0; bstate = {k_: x.detach().clone() for k_, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(bstate); return ep_used


if a.arch == "gbm":
    import lightgbm as lgb
    ok, y = targets("tr", "settle"); okv, yv = targets("va", "settle")
    gbm = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=500,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                             n_jobs=4, verbose=-1, random_state=a.seed)
    gbm.fit(D["tr"][0][ok], y[ok].astype(int), init_score=lg(D["tr"][1][ok]) if a.market == "residual" else None,
            eval_set=[(D["va"][0][okv], yv[okv].astype(int))],
            eval_init_score=[lg(D["va"][1][okv]) if a.market == "residual" else None],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    ep_used = int(gbm.best_iteration_ or 0)
    logits = {k: gbm.predict(D[k][0], raw_score=True) + (lg(D[k][1]) if a.market == "residual" else 0) for k in ("va", "te")}
else:
    net = make_net()
    if a.init:
        ck0 = torch.load(a.init, map_location="cpu"); assert int(ck0["nf"]) == NF and ck0["feat"] == a.feat, "init checkpoint spec mismatch"
        net.load_state_dict(ck0["state"]); net.to(DEV)
        ep_used = train(net, a.target, a.epochs, a.ft_lr, a.patience, head_only=a.ft_head, guard=True)
    elif a.pretrain:
        train(net, a.pretrain, a.pre_epochs, a.lr, 99)
        ep_used = train(net, a.target, a.epochs, a.ft_lr, a.patience, head_only=a.ft_head)
    else:
        ep_used = train(net, a.target, a.epochs, a.lr, a.patience)
    logits = {k: predict_logit(net, k) for k in ("va", "te")}
    if a.save_model:
        torch.save({"state": {k_: v.cpu() for k_, v in net.state_dict().items()}, "mu": mu, "sd": sd,
                    "arch": a.arch, "hidden": a.hidden, "dropout": a.dropout, "feat": a.feat,
                    "lookback": a.lookback, "market": a.market, "target": a.target, "sidx": SIDX,
                    "nf": NF, "use": use.tolist()}, a.save_model)

row = ["%-9s %-7s mkt=%-8s %-5s %-24s s%d ep%-2d nf%-3d" % (a.tag, a.target, a.market, a.arch, a.feat, a.seed, ep_used, NF)]
for k, lab in (("va", "VAL"), ("te", "TEST")):
    X, mid, st, fut, mk, tr = D[k][:6]; okk = np.isfinite(st); L = logits[k]
    bm, bk = bce_np(L[okk], st[okk]), bce_np(lg(mid[okk]), st[okk]); dd = bm - bk
    per = {}
    for kk, v in zip(mk[okk], dd):
        per.setdefault(kk, []).append(v)
    e = np.array([np.mean(v) for v in per.values()]); t_ = e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))
    Hm = bce_np(lg(mid), mid); klm = 1e4 * (bce_np(L, mid) - Hm).mean()
    row.append("%s BCE model %.5f mkt %.5f d %+.5f t %+5.2f | mktKL %5.2f" % (lab, bm.mean(), bk.mean(), dd.mean(), t_, klm))
    if k == "te":
        tt = tr[okk]; dt = dd[tt]; per = {}
        for kk, v in zip(mk[okk][tt], dt):
            per.setdefault(kk, []).append(v)
        e = np.array([np.mean(v) for v in per.values()])
        row.append("TRIG d %+.5f t %+5.2f" % (dt.mean(), e.mean() / (e.std(ddof=1) / np.sqrt(len(e)))))
if a.save_pred:
    np.savez_compressed(a.save_pred, logit_te=logits["te"], logit_va=logits["va"])
print(" | ".join(row), flush=True)
