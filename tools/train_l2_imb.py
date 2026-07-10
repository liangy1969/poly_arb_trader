#!/usr/bin/env python3
"""Fair-surface ablation: binance perp MID (from local L2 lake) + order-book
imbalance features.

Same protocol as train_venues.py (arch, seed, chronological 90/10 event split,
causal (db,dr) fit on tte>300s -> predict tte<=300s, KL(mkt||fair) core window
300-60s, paired outcome-BCE). Extra book features are z-scored on the train
set and fed to the MLP alongside [z', log tau]; per-event params stay on the
price channel only.

Ablations (feature sets added to [z', log tau]):
    base    []
    imb1    top-of-book qty imbalance
    imbK    imb1 + depth-{5,20,100} level-sum imbalances
    band    qty imbalance within +/-{5,10,25} bps of mid
    moff    microprice offset (bps)
    sprd    spread (bps) - non-directional control
    all     imbK + band + moff

Usage: python train_l2_imb.py <data_dir_with_rows> <l2feat.csv.gz>
"""
import csv
import gzip
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HID = 32
EPOCHS = 25
BATCH = 16384
LR = 2e-3
P_CLIP = 0.01
SEED = 7
VAL_FRAC = 0.10
FIT_TTE_S = 300
FIT_STEPS = 150
FIT_LR = 0.05
FFILL_S = 5
CLIP = 0.005
# Asset-scale priors: BTC defaults; e.g. ETH (~$1.7k): RHO0~4, B_SCALE~1.5
RHO0 = float(os.environ.get("RHO0", "150"))
SKIP_LAST_S = float(os.environ.get("SKIP_LAST_S", "0"))
B_SCALE = float(os.environ.get("B_SCALE", "50"))
# TARGET=outcome trains the surface on settlement labels instead of the
# market prob (the per-event eval fit ALWAYS uses market mid - that is the
# only online-fittable quantity).
TARGET = os.environ.get("TARGET", "market")

FEATS = ["imb1", "imb5", "imb20", "imb100", "band5", "band10", "band25", "moff_bps", "spread_bps", "mom15", "mom60", "mom180"]
ABLATIONS = [
    ("base", []),
    ("imb1", ["imb1"]),
    ("imbK", ["imb1", "imb5", "imb20", "imb100"]),
    ("band", ["band5", "band10", "band25"]),
    ("moff", ["moff_bps"]),
    ("sprd", ["spread_bps"]),
    ("all", ["imb1", "imb5", "imb20", "imb100", "band5", "band10", "band25", "moff_bps"]),
    ("mom", ["mom60"]),
    ("momK", ["mom15", "mom60", "mom180"]),
]
# ABL=base,imb1 : run only the named ablations (default: all of them)
_abl = [a for a in os.environ.get("ABL", "").split(",") if a]
if _abl:
    ABLATIONS = [(n, e) for n, e in ABLATIONS if n in _abl]


def load_feats(path):
    cols = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                cols[k].append(float(v))
    T = np.array(cols["t"], dtype=np.int64)
    o = np.argsort(T)
    T = T[o]
    mid = np.array(cols["mid"])[o]
    base_feats = [k for k in FEATS if not k.startswith("mom")]
    X = np.stack([np.array(cols[k])[o] for k in base_feats], axis=1)
    # momentum features: k-second underlying return, from the same series
    lut = dict(zip(T.tolist(), mid.tolist()))
    moms = []
    for k in (15, 60, 180):
        prev = np.array([lut.get(int(t) - k, np.nan) for t in T])
        moms.append(mid - prev)
    X = np.concatenate([X, np.stack(moms, axis=1)], axis=1)
    ok = ~np.isnan(X).any(axis=1)
    return T[ok], mid[ok], X[ok]


def load_kalshi(data_dir, t_lo, t_hi):
    ev = {}
    with open(os.path.join(data_dir, "events_meta.csv")) as f:
        meta = {r["ticker"]: r for r in csv.DictReader(f) if t_lo <= int(r["open_ts"]) <= t_hi}
    cols = defaultdict(lambda: {"t": [], "tte": [], "prob": []})
    with gzip.open(os.path.join(data_dir, "rows.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            if r["ticker"] not in meta:
                continue
            c = cols[r["ticker"]]
            c["t"].append(int(r["t"]))
            c["tte"].append(float(r["tte_s"]))
            c["prob"].append(float(r["prob"]))
    for t, c in cols.items():
        ev[t] = {
            "t": np.array(c["t"]),
            "tte": np.array(c["tte"]),
            "prob": np.array(c["prob"]),
            "strike": float(meta[t]["strike"]),
            "outcome": 1 if meta[t]["result"] == "yes" else 0,
            "open_ts": int(meta[t]["open_ts"]),
        }
    return ev


def attach(ev, T, mid, X):
    for d in ev.values():
        idx = np.searchsorted(T, d["t"], side="right") - 1
        ok = idx >= 0
        ok[ok] &= (d["t"][ok] - T[idx[ok]]) <= FFILL_S
        d["px"] = np.where(ok, mid[np.maximum(idx, 0)], np.nan)
        xf = np.full((len(d["t"]), X.shape[1]), np.nan)
        xf[ok] = X[idx[ok]]
        d["X"] = xf


class Surface(nn.Module):
    def __init__(self, n_extra):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + n_extra, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1)
        )

    def forward(self, zp, log_tau, extra):
        x = torch.cat([zp.unsqueeze(-1), log_tau.unsqueeze(-1), extra], dim=-1)
        return self.net(x).squeeze(-1)


def logit_of(net, px, tte, extra, b, s):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((px - b) / s) / tau.sqrt()
    return net(zp, tau.log(), extra)


def run(extras, ev, order, fcols, train_all=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fidx = [fcols.index(c) for c in extras]
    ids = order
    n_val = max(1, int(len(ids) * VAL_FRAC))
    tr_ids, val_ids = (ids, []) if train_all else (ids[:-n_val], ids[-n_val:])

    eidx, tte, px, prob, xf = [], [], [], [], []
    for i, t in enumerate(tr_ids):
        d = ev[t]
        ok = ~np.isnan(d["px"])
        if SKIP_LAST_S > 0:
            ok &= d["tte"] > SKIP_LAST_S
        eidx.append(np.full(ok.sum(), i))
        tte.append(d["tte"][ok])
        px.append(d["px"][ok])
        if TARGET == "outcome":
            prob.append(np.full(ok.sum(), float(d["outcome"])))
        else:
            prob.append(d["prob"][ok])
        xf.append(d["X"][ok][:, fidx] if fidx else np.zeros((ok.sum(), 0)))
    eidx = np.concatenate(eidx).astype(np.int64)
    tte_a = np.concatenate(tte)
    px_a = np.concatenate(px)
    prob_a = np.clip(np.concatenate(prob), P_CLIP, 1 - P_CLIP)
    xf_a = np.concatenate(xf)
    mu = xf_a.mean(0) if fidx else np.zeros(0)
    sd = xf_a.std(0) + 1e-9 if fidx else np.ones(0)
    xf_a = np.clip((xf_a - mu) / sd, -5, 5)
    strikes = torch.tensor([ev[t]["strike"] for t in tr_ids], dtype=torch.float32)

    net = Surface(len(fidx))
    d_b = nn.Parameter(torch.zeros(len(tr_ids)))
    d_r = nn.Parameter(torch.zeros(len(tr_ids)))
    rho = nn.Parameter(torch.tensor(math.log(RHO0)))
    opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [d_b, d_r, rho]}], lr=LR)
    E = torch.tensor(eidx)
    TT = torch.tensor(tte_a, dtype=torch.float32)
    SP = torch.tensor(px_a, dtype=torch.float32)
    XF = torch.tensor(xf_a, dtype=torch.float32)
    Y = torch.tensor(prob_a, dtype=torch.float32)
    cnt = np.bincount(eidx, minlength=len(tr_ids)).astype(np.float64)
    W = torch.tensor(1.0 / np.maximum(cnt[eidx], 1), dtype=torch.float32)
    n = len(eidx)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            ix = perm[i : i + BATCH]
            e = E[ix]
            lo = logit_of(net, SP[ix], TT[ix], XF[ix], strikes[e] + B_SCALE * d_b[e], torch.exp(rho + d_r[e]))
            bce = nn.functional.binary_cross_entropy_with_logits(lo, Y[ix], weight=W[ix], reduction="sum") / W[ix].sum()
            opt.zero_grad()
            bce.backward()
            opt.step()
    for p in net.parameters():
        p.requires_grad_(False)
    rho_f = rho.detach()

    # RESID_MOM=1: logistic residual outcome ~ logit_fair + beta*mom_norm,
    # beta fit on the train rows (using their trained per-event params).
    beta = 0.0
    mom_col = fcols.index("mom60")
    if os.environ.get("RESID_MOM", "") == "1":
        with torch.no_grad():
            lo_tr = logit_of(net, SP, TT, XF, strikes[E] + B_SCALE * d_b[E].detach(), torch.exp(rho_f + d_r[E].detach()))
        mom_raw = []
        for i, t in enumerate(tr_ids):
            d = ev[t]
            ok = ~np.isnan(d["px"])
            if SKIP_LAST_S > 0:
                ok &= d["tte"] > SKIP_LAST_S
            mom_raw.append(d["X"][ok][:, mom_col])
        mom_a = np.concatenate(mom_raw)
        m_mu, m_sd = mom_a.mean(), mom_a.std() + 1e-9
        mz = torch.tensor(np.clip((mom_a - m_mu) / m_sd, -5, 5), dtype=torch.float32)
        yo = torch.tensor(np.concatenate([np.full(int((~np.isnan(ev[t]["px"]) & (ev[t]["tte"] > SKIP_LAST_S if SKIP_LAST_S > 0 else ~np.isnan(ev[t]["px"]))).sum()) if False else int(len(x)), float(ev[t]["outcome"])) for t, x in zip(tr_ids, mom_raw)]), dtype=torch.float32)
        b_p = torch.zeros(1, requires_grad=True)
        a_p = torch.zeros(1, requires_grad=True)
        ob = torch.optim.Adam([b_p, a_p], lr=0.05)
        Wt = W
        for _ in range(200):
            ob.zero_grad()
            lo2 = lo_tr + a_p + b_p * mz
            loss = nn.functional.binary_cross_entropy_with_logits(lo2, yo, weight=Wt, reduction="sum") / Wt.sum()
            loss.backward()
            ob.step()
        beta = float(b_p)
        print(f"  RESID_MOM: beta={beta:+.4f} alpha={float(a_p):+.4f} (mom60 z-scored, train outcome fit)")
        globals()["_RESID"] = (beta, float(a_p), m_mu, m_sd, mom_col)

    per_ev = {}  # ticker -> dict of per-event metrics (for pairing)
    for t in val_ids:
        d = ev[t]
        ok = ~np.isnan(d["px"])
        tte_v, px_v, prob_v = d["tte"][ok], d["px"][ok], d["prob"][ok]
        xf_v = np.clip((d["X"][ok][:, fidx] - mu) / sd, -5, 5) if fidx else np.zeros((ok.sum(), 0))
        fit = tte_v > FIT_TTE_S
        pred = tte_v <= FIT_TTE_S
        if fit.sum() < 60 or pred.sum() < 60:
            continue
        db = torch.zeros(1, requires_grad=True)
        dr = torch.zeros(1, requires_grad=True)
        o2 = torch.optim.Adam([db, dr], lr=FIT_LR)
        ts = torch.tensor(px_v[fit], dtype=torch.float32)
        tt = torch.tensor(tte_v[fit], dtype=torch.float32)
        xx = torch.tensor(xf_v[fit], dtype=torch.float32)
        y = torch.tensor(np.clip(prob_v[fit], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
        for _ in range(FIT_STEPS):
            o2.zero_grad()
            lo = logit_of(net, ts, tt, xx, d["strike"] + B_SCALE * db, torch.exp(rho_f + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            o2.step()
        with torch.no_grad():
            lo = logit_of(
                net,
                torch.tensor(px_v[pred], dtype=torch.float32),
                torch.tensor(tte_v[pred], dtype=torch.float32),
                torch.tensor(xf_v[pred], dtype=torch.float32),
                d["strike"] + B_SCALE * db,
                torch.exp(rho_f + dr),
            )
            fair = torch.sigmoid(lo).numpy()
        mid_p = np.clip(prob_v[pred], CLIP, 1 - CLIP)
        f = np.clip(fair, CLIP, 1 - CLIP)
        if "_RESID" in globals():
            _b, _a, _mu, _sd, _mc = globals()["_RESID"]
            mzv = np.clip((d["X"][ok][:, _mc][pred] - _mu) / _sd, -5, 5)
            lo_adj = np.log(f / (1 - f)) + _a + _b * mzv
            f_adj = np.clip(1 / (1 + np.exp(-lo_adj)), CLIP, 1 - CLIP)
        else:
            f_adj = None
        bce = -(mid_p * np.log(f) + (1 - mid_p) * np.log(1 - f))
        H = -(mid_p * np.log(mid_p) + (1 - mid_p) * np.log(1 - mid_p))
        kl_row = bce - H
        core = tte_v[pred] > 60
        o = d["outcome"]
        bo_f = -(o * np.log(f) + (1 - o) * np.log(1 - f))
        bo_m = -(o * np.log(mid_p) + (1 - o) * np.log(1 - mid_p))
        if f_adj is not None:
            bo_adj = -(o * np.log(f_adj) + (1 - o) * np.log(1 - f_adj))
            if core.sum() >= 30:
                m_extra_adj = bo_adj[core].mean()
        else:
            m_extra_adj = None
        m = {"kl": kl_row.mean()}
        if core.sum() >= 30:
            m["kl_core"] = kl_row[core].mean()
            m["out_model_core"] = bo_f[core].mean()
            m["out_market_core"] = bo_m[core].mean()
            if f_adj is not None:
                m["out_adj_core"] = bo_adj[core].mean()
        if (~core).sum() >= 30:
            m["kl_last"] = kl_row[~core].mean()
        per_ev[t] = m
    return {"rho": float(rho_f), "per_ev": per_ev, "n_train": len(tr_ids),
            "net": net, "mu": mu.tolist(), "sd": sd.tolist()}


def agg(per_ev, key):
    v = [m[key] for m in per_ev.values() if key in m]
    return float(np.mean(v)) if v else float("nan")


def main():
    data_dir, feat_path = sys.argv[1], sys.argv[2]
    T, mid, X = load_feats(feat_path)
    print(f"features: {len(T):,} secs  {T.min()} -> {T.max()}")
    cap = int(os.environ.get("CAP_TS", "0"))  # exclude events opening at/after this
    ev = load_kalshi(data_dir, int(T.min()), min(int(T.max()) - 900, cap - 1) if cap else int(T.max()) - 900)
    attach(ev, T, mid, X)
    # keep events with decent coverage
    order = [t for t in sorted(ev, key=lambda t: ev[t]["open_ts"])
             if (~np.isnan(ev[t]["px"])).sum() >= 600]
    print(f"events with kalshi+L2 coverage: {len(order)}")

    # Export mode: EXPORT=<path> EXTRAS=<comma-list or empty> — train on ALL
    # events (train_all) and dump a sim_50ms-compatible JSON with extras/mu/sd.
    export = os.environ.get("EXPORT", "")
    if export:
        extras = [c for c in os.environ.get("EXTRAS", "").split(",") if c]
        r = run(extras, ev, order, FEATS, train_all=True)
        net = r["net"]
        layers = [{"w": m.weight.tolist(), "b": m.bias.tolist()} for m in net.net if isinstance(m, nn.Linear)]
        out = {
            "arch": {"hidden": HID, "clamp": 2.0, "act": "tanh", "logit": "raw", "mode": "direct"},
            "features": "zp=(px-b)/s/sqrt(tte/900); u_in=[zp, log(tte/900)] + extras",
            "rho_bar": r["rho"],
            "b_prior": "strike",
            "b_scale": B_SCALE,
            "extras": extras,
            "mu": r["mu"],
            "sd": r["sd"],
            "layers": layers,
            "train": {"venue": "lake_perp_mid", "events": len(order), "cap_ts": cap, "target": "market_prob"},
        }
        os.makedirs(os.path.dirname(export), exist_ok=True)
        json.dump(out, open(export, "w"))
        print(f"exported extras={extras} ({len(order)} events, s~${math.exp(r['rho']):.0f}) -> {export}")
        return

    results = {}
    base_pe = None
    print(f"\nEVAL core window tte 300-60s; paired deltas vs base (negative = better than base)")
    print(f"{'ablation':>8} {'val':>4} {'s~$':>5} {'KL_core':>9} {'dKL(t)':>16} {'outBCE mdl':>11} {'outBCE mkt':>11} {'dOut(t)':>16}  (all / last60)")
    for name, extras in ABLATIONS:
        r = run(extras, ev, order, FEATS)
        pe = r["per_ev"]
        results[name] = r
        if name == "base":
            base_pe = pe
        # paired deltas vs base on common events
        dkl, dout = [], []
        for t, m in pe.items():
            if base_pe and t in base_pe and "kl_core" in m and "kl_core" in base_pe[t]:
                dkl.append(m["kl_core"] - base_pe[t]["kl_core"])
                dout.append(m["out_model_core"] - base_pe[t]["out_model_core"])
        def tstat(x):
            x = np.array(x)
            return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan")
        dkl_s = f"{np.mean(dkl):+.5f}({tstat(dkl):+.1f})" if dkl and name != "base" else "-"
        dout_s = f"{np.mean(dout):+.5f}({tstat(dout):+.1f})" if dout and name != "base" else "-"
        adj = agg(pe, 'out_adj_core')
        adj_s = f" adjOut {adj:.4f}" if adj == adj else ""
        print(
            f"{name:>8} {len(pe):>4} {math.exp(r['rho']):>5.0f} {agg(pe,'kl_core'):>9.5f} {dkl_s:>16} "
            f"{agg(pe,'out_model_core'):>11.4f} {agg(pe,'out_market_core'):>11.4f} {dout_s:>16}  "
            f"({agg(pe,'kl'):.4f} / {agg(pe,'kl_last'):.4f}){adj_s}",
            flush=True,
        )


if __name__ == "__main__":
    main()
