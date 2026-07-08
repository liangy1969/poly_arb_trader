#!/usr/bin/env python3
"""Per-venue fair-surface training + KL(market||fair) comparison.

Trains one direct-mode surface per price venue on the SAME events (identical
architecture, hyperparameters, seed, chronological 90/10 event split) and
compares, on held-out events under the standard online protocol (causal
(Δb,Δρ) fit on tte>300s at 1s grid → predict tte<=300s):

    KL(market ‖ fair) = BCE(fair vs mid) − H(mid)     (lower = tracks better)
    outcome-BCE (model vs market)                     (secondary diagnostic)

Venues: binance_s (spot column already in rows.csv.gz), plus every venue in
venue_prices.csv.gz (kraken / coinbase / binance_p), forward-filled ≤30s onto
the same per-second grid.

Usage: python train_venues.py <data_dir_with_rows> <venue_prices.csv.gz>
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
FFILL_S = 30
CLIP = 0.005
RHO0 = float(os.environ.get("RHO0", "150"))
SKIP_LAST_S = float(os.environ.get("SKIP_LAST_S", "0"))
B_SCALE = float(os.environ.get("B_SCALE", "50"))


def load_kalshi(data_dir, tickers):
    ev = {}
    with open(os.path.join(data_dir, "events_meta.csv")) as f:
        meta = {r["ticker"]: r for r in csv.DictReader(f) if r["ticker"] in tickers}
    cols = defaultdict(lambda: {"t": [], "tte": [], "prob": [], "spot": []})
    with gzip.open(os.path.join(data_dir, "rows.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            if r["ticker"] not in meta:
                continue
            c = cols[r["ticker"]]
            c["t"].append(int(r["t"]))
            c["tte"].append(float(r["tte_s"]))
            c["prob"].append(float(r["prob"]))
            c["spot"].append(float(r["spot"]))
    for t, c in cols.items():
        ev[t] = {
            "t": np.array(c["t"]),
            "tte": np.array(c["tte"]),
            "prob": np.array(c["prob"]),
            "binance_s": np.array(c["spot"]),
            "strike": float(meta[t]["strike"]),
            "outcome": 1 if meta[t]["result"] == "yes" else 0,
            "open_ts": int(meta[t]["open_ts"]),
        }
    return ev


def load_venues(path):
    per = defaultdict(lambda: defaultdict(dict))  # venue -> ticker -> {sec: px}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            per[r["venue"]][r["ticker"]][int(r["t"])] = float(r["px"])
    return per


def align(series, grid):
    """ffill venue {sec: px} onto grid secs (<=FFILL_S stale); NaN where absent."""
    out = np.full(len(grid), np.nan)
    if not series:
        return out
    ks = np.array(sorted(series))
    vs = np.array([series[k] for k in ks])
    idx = np.searchsorted(ks, grid, side="right") - 1
    ok = idx >= 0
    ok[ok] &= (grid[ok] - ks[idx[ok]]) <= FFILL_S
    out[ok] = vs[idx[ok]]
    return out


class Surface(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1)
        )

    def forward(self, zp, log_tau):
        return self.net(torch.stack([zp, log_tau], dim=-1)).squeeze(-1)


def logit_of(net, spot, tte, b, s):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((spot - b) / s) / tau.sqrt()
    return net(zp, tau.log())


def run_venue(name, ev, order, train_all=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ids = [t for t in order if not np.all(np.isnan(ev[t][name]))]
    n_ev = len(ids)
    n_val = max(1, int(n_ev * VAL_FRAC))
    tr_ids, val_ids = (ids, []) if train_all else (ids[:-n_val], ids[-n_val:])

    eidx, tte, px, prob = [], [], [], []
    for i, t in enumerate(tr_ids):
        d = ev[t]
        ok = ~np.isnan(d[name])
        if SKIP_LAST_S > 0:
            ok &= d["tte"] > SKIP_LAST_S
        eidx.append(np.full(ok.sum(), i))
        tte.append(d["tte"][ok])
        px.append(d[name][ok])
        prob.append(d["prob"][ok])
    eidx = np.concatenate(eidx).astype(np.int64)
    tte_a = np.concatenate(tte)
    px_a = np.concatenate(px)
    prob_a = np.clip(np.concatenate(prob), P_CLIP, 1 - P_CLIP)
    strikes = torch.tensor([ev[t]["strike"] for t in tr_ids], dtype=torch.float32)

    net = Surface()
    d_b = nn.Parameter(torch.zeros(len(tr_ids)))
    d_r = nn.Parameter(torch.zeros(len(tr_ids)))
    rho = nn.Parameter(torch.tensor(math.log(RHO0)))
    opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [d_b, d_r, rho]}], lr=LR)
    E = torch.tensor(eidx)
    TT = torch.tensor(tte_a, dtype=torch.float32)
    SP = torch.tensor(px_a, dtype=torch.float32)
    Y = torch.tensor(prob_a, dtype=torch.float32)
    cnt = np.bincount(eidx, minlength=len(tr_ids)).astype(np.float64)
    W = torch.tensor(1.0 / np.maximum(cnt[eidx], 1), dtype=torch.float32)
    n = len(eidx)
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            ix = perm[i : i + BATCH]
            e = E[ix]
            lo = logit_of(net, SP[ix], TT[ix], strikes[e] + B_SCALE * d_b[e], torch.exp(rho + d_r[e]))
            bce = nn.functional.binary_cross_entropy_with_logits(lo, Y[ix], weight=W[ix], reduction="sum") / W[ix].sum()
            opt.zero_grad()
            bce.backward()
            opt.step()
    for p in net.parameters():
        p.requires_grad_(False)
    rho_f = rho.detach()

    # ── held-out eval: online protocol ──
    kls, kls_core, kls_last, out_m, out_k = [], [], [], [], []
    out_m_core, out_k_core = [], []
    for t in val_ids:
        d = ev[t]
        ok = ~np.isnan(d[name])
        tte_v, px_v, prob_v = d["tte"][ok], d[name][ok], d["prob"][ok]
        fit = tte_v > FIT_TTE_S
        pred = tte_v <= FIT_TTE_S
        if fit.sum() < 60 or pred.sum() < 60:
            continue
        db = torch.zeros(1, requires_grad=True)
        dr = torch.zeros(1, requires_grad=True)
        o2 = torch.optim.Adam([db, dr], lr=FIT_LR)
        ts = torch.tensor(px_v[fit], dtype=torch.float32)
        tt = torch.tensor(tte_v[fit], dtype=torch.float32)
        y = torch.tensor(np.clip(prob_v[fit], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
        for _ in range(FIT_STEPS):
            o2.zero_grad()
            lo = logit_of(net, ts, tt, d["strike"] + B_SCALE * db, torch.exp(rho_f + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            o2.step()
        with torch.no_grad():
            lo = logit_of(
                net,
                torch.tensor(px_v[pred], dtype=torch.float32),
                torch.tensor(tte_v[pred], dtype=torch.float32),
                d["strike"] + B_SCALE * db,
                torch.exp(rho_f + dr),
            )
            fair = torch.sigmoid(lo).numpy()
        mid = np.clip(prob_v[pred], CLIP, 1 - CLIP)
        f = np.clip(fair, CLIP, 1 - CLIP)
        bce = -(mid * np.log(f) + (1 - mid) * np.log(1 - f))
        H = -(mid * np.log(mid) + (1 - mid) * np.log(1 - mid))
        kl_row = bce - H
        core = tte_v[pred] > 60  # (60, 300]: settlement-endgame excluded
        kls.append(kl_row.mean())
        if core.sum() >= 30:
            kls_core.append(kl_row[core].mean())
        if (~core).sum() >= 30:
            kls_last.append(kl_row[~core].mean())
        o = d["outcome"]
        bo_f = -(o * np.log(f) + (1 - o) * np.log(1 - f))
        bo_m = -(o * np.log(mid) + (1 - o) * np.log(1 - mid))
        out_m.append(bo_f.mean())
        out_k.append(bo_m.mean())
        if core.sum() >= 30:
            out_m_core.append(bo_f[core].mean())
            out_k_core.append(bo_m[core].mean())
    return {
        "n_events": n_ev,
        "n_val": len(kls),
        "rho": float(rho_f),
        "kl": float(np.mean(kls)) if kls else float("nan"),
        "kl_core": float(np.mean(kls_core)) if kls_core else float("nan"),
        "kl_last": float(np.mean(kls_last)) if kls_last else float("nan"),
        "out_model": float(np.mean(out_m)) if out_m else float("nan"),
        "out_market": float(np.mean(out_k)) if out_k else float("nan"),
        "out_model_core": float(np.mean(out_m_core)) if out_m_core else float("nan"),
        "out_market_core": float(np.mean(out_k_core)) if out_k_core else float("nan"),
        "net": net,
    }


def main():
    data_dir = sys.argv[1]
    venues_path = sys.argv[2]
    per_venue = load_venues(venues_path)
    tickers = set().union(*[set(v.keys()) for v in per_venue.values()])
    ev = load_kalshi(data_dir, tickers)
    order = sorted(ev, key=lambda t: ev[t]["open_ts"])
    print(f"events with kalshi+venue data: {len(order)}")
    # attach venue columns on the kalshi grid
    for vname, per_tick in per_venue.items():
        for t in order:
            ev[t][vname] = align(per_tick.get(t, {}), ev[t]["t"])

    # Export mode: ONLY=<venue> TRAIN_ALL=1 EXPORT=<path> — train one venue on
    # ALL its events and dump sim_50ms-compatible weights JSON.
    only = os.environ.get("ONLY", "")
    export = os.environ.get("EXPORT", "")
    if only and export:
        r = run_venue(only, ev, order, train_all=os.environ.get("TRAIN_ALL", "") == "1")
        net = r["net"]
        layers = [
            {"w": m.weight.tolist(), "b": m.bias.tolist()}
            for m in net.net
            if isinstance(m, nn.Linear)
        ]
        out = {
            "arch": {"hidden": HID, "clamp": 2.0, "act": "tanh", "logit": "raw", "mode": "direct"},
            "features": "zp=(px-b)/s/sqrt(tte/900); u_in=[zp, log(tte/900)]",
            "rho_bar": r["rho"],
            "b_prior": "strike",
            "b_scale": B_SCALE,
            "layers": layers,
            "train": {"venue": only, "events": r["n_events"], "target": "market_prob"},
        }
        os.makedirs(os.path.dirname(export), exist_ok=True)
        json.dump(out, open(export, "w"))
        print(f"exported {only} surface ({r['n_events']} events, s~${math.exp(r['rho']):.0f}) -> {export}")
        return

    names = ["binance_s"] + sorted(per_venue.keys())
    print(f"\nEVAL WINDOW tte 300-60s ONLY (final minute excluded from fit AND test):")
    print(f"{'venue':>10} {'n_ev':>5} {'val':>4} {'s~$':>5} {'KL':>8} {'outBCE model':>13} {'outBCE mkt':>11}   (all-window: KL / last60s KL)")
    for name in names:
        r = run_venue(name, ev, order)
        print(
            f"{name:>10} {r['n_events']:>5} {r['n_val']:>4} {math.exp(r['rho']):>5.0f} "
            f"{r['kl_core']:>8.4f} {r['out_model_core']:>13.4f} {r['out_market_core']:>11.4f}   "
            f"({r['kl']:.4f} / {r['kl_last']:.4f})"
        )


if __name__ == "__main__":
    main()
