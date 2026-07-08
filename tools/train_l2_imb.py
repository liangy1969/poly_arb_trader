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

FEATS = ["imb1", "imb5", "imb20", "imb100", "band5", "band10", "band25", "moff_bps", "spread_bps"]
ABLATIONS = [
    ("base", []),
    ("imb1", ["imb1"]),
    ("imbK", ["imb1", "imb5", "imb20", "imb100"]),
    ("band", ["band5", "band10", "band25"]),
    ("moff", ["moff_bps"]),
    ("sprd", ["spread_bps"]),
    ("all", ["imb1", "imb5", "imb20", "imb100", "band5", "band10", "band25", "moff_bps"]),
]


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
    X = np.stack([np.array(cols[k])[o] for k in FEATS], axis=1)
    return T, mid, X


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


def run(extras, ev, order, fcols):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fidx = [fcols.index(c) for c in extras]
    ids = order
    n_val = max(1, int(len(ids) * VAL_FRAC))
    tr_ids, val_ids = ids[:-n_val], ids[-n_val:]

    eidx, tte, px, prob, xf = [], [], [], [], []
    for i, t in enumerate(tr_ids):
        d = ev[t]
        ok = ~np.isnan(d["px"])
        eidx.append(np.full(ok.sum(), i))
        tte.append(d["tte"][ok])
        px.append(d["px"][ok])
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
    rho = nn.Parameter(torch.tensor(math.log(150.0)))
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
            lo = logit_of(net, SP[ix], TT[ix], XF[ix], strikes[e] + 50.0 * d_b[e], torch.exp(rho + d_r[e]))
            bce = nn.functional.binary_cross_entropy_with_logits(lo, Y[ix], weight=W[ix], reduction="sum") / W[ix].sum()
            opt.zero_grad()
            bce.backward()
            opt.step()
    for p in net.parameters():
        p.requires_grad_(False)
    rho_f = rho.detach()

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
            lo = logit_of(net, ts, tt, xx, d["strike"] + 50.0 * db, torch.exp(rho_f + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            o2.step()
        with torch.no_grad():
            lo = logit_of(
                net,
                torch.tensor(px_v[pred], dtype=torch.float32),
                torch.tensor(tte_v[pred], dtype=torch.float32),
                torch.tensor(xf_v[pred], dtype=torch.float32),
                d["strike"] + 50.0 * db,
                torch.exp(rho_f + dr),
            )
            fair = torch.sigmoid(lo).numpy()
        mid_p = np.clip(prob_v[pred], CLIP, 1 - CLIP)
        f = np.clip(fair, CLIP, 1 - CLIP)
        bce = -(mid_p * np.log(f) + (1 - mid_p) * np.log(1 - f))
        H = -(mid_p * np.log(mid_p) + (1 - mid_p) * np.log(1 - mid_p))
        kl_row = bce - H
        core = tte_v[pred] > 60
        o = d["outcome"]
        bo_f = -(o * np.log(f) + (1 - o) * np.log(1 - f))
        bo_m = -(o * np.log(mid_p) + (1 - o) * np.log(1 - mid_p))
        m = {"kl": kl_row.mean()}
        if core.sum() >= 30:
            m["kl_core"] = kl_row[core].mean()
            m["out_model_core"] = bo_f[core].mean()
            m["out_market_core"] = bo_m[core].mean()
        if (~core).sum() >= 30:
            m["kl_last"] = kl_row[~core].mean()
        per_ev[t] = m
    return {"rho": float(rho_f), "per_ev": per_ev, "n_train": len(tr_ids)}


def agg(per_ev, key):
    v = [m[key] for m in per_ev.values() if key in m]
    return float(np.mean(v)) if v else float("nan")


def main():
    data_dir, feat_path = sys.argv[1], sys.argv[2]
    T, mid, X = load_feats(feat_path)
    print(f"features: {len(T):,} secs  {T.min()} -> {T.max()}")
    ev = load_kalshi(data_dir, int(T.min()), int(T.max()) - 900)
    attach(ev, T, mid, X)
    # keep events with decent coverage
    order = [t for t in sorted(ev, key=lambda t: ev[t]["open_ts"])
             if (~np.isnan(ev[t]["px"])).sum() >= 600]
    print(f"events with kalshi+L2 coverage: {len(order)}")

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
        print(
            f"{name:>8} {len(pe):>4} {math.exp(r['rho']):>5.0f} {agg(pe,'kl_core'):>9.5f} {dkl_s:>16} "
            f"{agg(pe,'out_model_core'):>11.4f} {agg(pe,'out_market_core'):>11.4f} {dout_s:>16}  "
            f"({agg(pe,'kl'):.4f} / {agg(pe,'kl_last'):.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
