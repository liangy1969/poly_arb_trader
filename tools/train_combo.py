#!/usr/bin/env python3
"""3-venue combo surface: fair = MLP(z'_spot, z'_perp, z'_coinbase, log tau).

Per-event, per-venue bias b_{e,v} = strike + 50*db_{e,v} (3/event) + one shared
scale s_e = exp(rho_bar + dr_e) -> 4 online-fittable params. The MLP sees the
three normalized prices jointly, so cross-venue structure (perp-spot basis
state, settlement-chain vs discovery divergence) becomes a feature.

Same protocol as train_venues.py: chronological 90/10 event split, causal
per-event fit on tte>300s, eval on tte 300-60s (KL vs market mid + outcome-BCE
model vs market). Compare against the single-venue table.

Usage: python train_combo.py <data_dir_with_rows> <venue_prices.csv.gz>
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_venues as TV  # loaders + constants

VENUES = ["binance_s", "binance_p", "coinbase"]
HID = 32


class Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1)
        )

    def forward(self, zps, log_tau):  # zps: (..., 3)
        x = torch.cat([zps, log_tau.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def logit_of(net, px3, tte, b3, s3):
    """px3/b3/s3: (...,3) — per-venue bias AND scale; tte: (...)"""
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((px3 - b3) / s3) / tau.sqrt().unsqueeze(-1)
    return net(zp, tau.log())


def main():
    data_dir = sys.argv[1]
    venues_path = sys.argv[2]
    per_venue = TV.load_venues(venues_path)
    tickers = set(per_venue["coinbase"].keys()) & set(per_venue["binance_p"].keys())
    ev = TV.load_kalshi(data_dir, tickers)
    order = sorted(ev, key=lambda t: ev[t]["open_ts"])
    for vname in ["binance_p", "coinbase"]:
        for t in order:
            ev[t][vname] = TV.align(per_venue[vname].get(t, {}), ev[t]["t"])

    # events need decent coverage of ALL venues
    usable = []
    for t in order:
        d = ev[t]
        ok = ~(np.isnan(d["binance_p"]) | np.isnan(d["coinbase"]))
        if ok.sum() >= 500:
            usable.append(t)
    n_ev = len(usable)
    n_val = max(1, int(n_ev * TV.VAL_FRAC))
    tr_ids, val_ids = usable[:-n_val], usable[-n_val:]
    print(f"combo events: {n_ev} (train {len(tr_ids)}, val {len(val_ids)})")

    torch.manual_seed(TV.SEED)
    np.random.seed(TV.SEED)

    eidx, tte, prob, px = [], [], [], []
    for i, t in enumerate(tr_ids):
        d = ev[t]
        ok = ~(np.isnan(d["binance_p"]) | np.isnan(d["coinbase"]))
        eidx.append(np.full(ok.sum(), i))
        tte.append(d["tte"][ok])
        prob.append(d["prob"][ok])
        px.append(np.stack([d[v][ok] for v in VENUES], axis=1))
    eidx = np.concatenate(eidx).astype(np.int64)
    tte_a = np.concatenate(tte)
    prob_a = np.clip(np.concatenate(prob), TV.P_CLIP, 1 - TV.P_CLIP)
    px_a = np.concatenate(px)
    strikes = torch.tensor([ev[t]["strike"] for t in tr_ids], dtype=torch.float32)

    net = Combo()
    d_b = nn.Parameter(torch.zeros(len(tr_ids), 3))
    d_r = nn.Parameter(torch.zeros(len(tr_ids), 3))  # per-venue scale
    rho = nn.Parameter(torch.full((3,), math.log(150.0)))  # per-venue population scale
    opt = torch.optim.Adam([{"params": net.parameters()}, {"params": [d_b, d_r, rho]}], lr=TV.LR)
    E = torch.tensor(eidx)
    TT = torch.tensor(tte_a, dtype=torch.float32)
    PX = torch.tensor(px_a, dtype=torch.float32)
    Y = torch.tensor(prob_a, dtype=torch.float32)
    cnt = np.bincount(eidx, minlength=len(tr_ids)).astype(np.float64)
    W = torch.tensor(1.0 / np.maximum(cnt[eidx], 1), dtype=torch.float32)
    n = len(eidx)
    for ep in range(TV.EPOCHS):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, TV.BATCH):
            ix = perm[i : i + TV.BATCH]
            e = E[ix]
            b3 = strikes[e].unsqueeze(-1) + 50.0 * d_b[e]
            s3 = torch.exp(rho + d_r[e])  # (batch, 3)
            lo = logit_of(net, PX[ix], TT[ix], b3, s3)
            bce = nn.functional.binary_cross_entropy_with_logits(lo, Y[ix], weight=W[ix], reduction="sum") / W[ix].sum()
            opt.zero_grad()
            bce.backward()
            opt.step()
            tot += float(bce.detach()) * len(ix)
        if (ep + 1) % 5 == 0:
            sv = [f"{math.exp(float(r)):.0f}" for r in rho]
            print(f"epoch {ep + 1}/{TV.EPOCHS} bce={tot / n:.4f} s~$[{','.join(sv)}]", flush=True)
    for p in net.parameters():
        p.requires_grad_(False)
    rho_f = rho.detach()

    # ── held-out eval (core window 300-60s) ──
    kls_core, out_m_core, out_k_core = [], [], []
    for t in val_ids:
        d = ev[t]
        ok = ~(np.isnan(d["binance_p"]) | np.isnan(d["coinbase"]))
        tte_v = d["tte"][ok]
        prob_v = d["prob"][ok]
        px_v = np.stack([d[v][ok] for v in VENUES], axis=1)
        fit = tte_v > TV.FIT_TTE_S
        core = (tte_v <= TV.FIT_TTE_S) & (tte_v > 60)
        if fit.sum() < 60 or core.sum() < 60:
            continue
        db = torch.zeros(3, requires_grad=True)
        dr = torch.zeros(3, requires_grad=True)
        o2 = torch.optim.Adam([db, dr], lr=TV.FIT_LR)
        pxf = torch.tensor(px_v[fit], dtype=torch.float32)
        ttf = torch.tensor(tte_v[fit], dtype=torch.float32)
        y = torch.tensor(np.clip(prob_v[fit], TV.P_CLIP, 1 - TV.P_CLIP), dtype=torch.float32)
        for _ in range(TV.FIT_STEPS):
            o2.zero_grad()
            lo = logit_of(net, pxf, ttf, d["strike"] + 50.0 * db, torch.exp(rho_f + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            o2.step()
        with torch.no_grad():
            lo = logit_of(
                net,
                torch.tensor(px_v[core], dtype=torch.float32),
                torch.tensor(tte_v[core], dtype=torch.float32),
                d["strike"] + 50.0 * db,
                torch.exp(rho_f + dr),
            )
            fair = torch.sigmoid(lo).numpy()
        mid = np.clip(prob_v[core], TV.CLIP, 1 - TV.CLIP)
        f = np.clip(fair, TV.CLIP, 1 - TV.CLIP)
        bce = -(mid * np.log(f) + (1 - mid) * np.log(1 - f))
        H = -(mid * np.log(mid) + (1 - mid) * np.log(1 - mid))
        kls_core.append((bce - H).mean())
        o = d["outcome"]
        out_m_core.append((-(o * np.log(f) + (1 - o) * np.log(1 - f))).mean())
        out_k_core.append((-(o * np.log(mid) + (1 - o) * np.log(1 - mid))).mean())

    sv = [f"{math.exp(float(r)):.0f}" for r in rho_f]
    print(f"\nCOMBO (3-venue, per-venue scale) — eval tte 300-60s, val={len(kls_core)} events, s~$[{','.join(sv)}]")
    print(f"  KL(mkt||fair) = {np.mean(kls_core):.4f}")
    print(f"  outBCE model  = {np.mean(out_m_core):.4f}")
    print(f"  outBCE market = {np.mean(out_k_core):.4f}")
    print(f"  model - market = {np.mean(out_m_core) - np.mean(out_k_core):+.4f}  (<0 = model ahead)")


if __name__ == "__main__":
    main()
