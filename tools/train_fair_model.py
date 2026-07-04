#!/usr/bin/env python3
"""Offline trainer for the KXBTC15M fair-probability surface (phase 2).

Model (user design, 2026-07-04):
    z    = (spot - b_e) / s_e          per-event affine normalization
    z'   = z / sqrt(tau')              tau' = tte_s/900 — standardized distance
    fair = sigmoid( z' * exp(u(z', log tau')) )
where u is a small MLP learned OFFLINE and FROZEN online, while (b_e, s_e)
are per-event: trainable here (amortized over all events), fitted online
from the live event's own history.

Structural anchors (identifiability):
  - fair(z=0) = 0.5 exactly (logit ∝ z'), odd-symmetric-ish surface
  - b_e = strike_e + Δb_e with a prior pulling Δb→0 (strike is known!)
  - s_e = exp(ρ̄ + Δρ_e), Δρ prior → population vol level ρ̄ is learned
  - u tanh-clamped to ±CLAMP so the surface stays sane in the tails

Loss: BCE against the observed MARKET probability (user-selected target),
rows weighted so each event contributes equally. Chronological event split.
Eval extras: outcome-BCE diagnostic + the ONLINE REHEARSAL — freeze u, fit
(Δb, Δρ) per val-event on the first `fit_min` minutes only, predict the rest;
report gap stats and whether market moves TOWARD fair afterward (the edge).

Usage:  py tools/train_fair_model.py data/model out/fair_model
Needs:  numpy, torch (CPU fine; model is tiny).
"""
import csv
import gzip
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

CLAMP = 2.0
HID = 32
EPOCHS = 25
BATCH = 16384
LR = 2e-3
# MODE=structured (default): logit = z'*exp(u(z',logtau)), BCE + priors + wd.
# MODE=direct (ablation):    logit = MLP(z',logtau) raw, BCE ONLY (no priors/wd).
MODE = os.environ.get("MODE", "structured")
LAMBDA_B = 0.0 if MODE == "direct" else 1.0e-4  # prior on (Δb/50)^2
LAMBDA_R = 0.0 if MODE == "direct" else 1.0e-3  # prior on Δρ^2
WD = 0.0 if MODE == "direct" else 1e-5
P_CLIP = 0.01
SEED = 7
VAL_FRAC = 0.10
REHEARSAL_FIT_S = 600  # online rehearsal: fit (b,s) while tte > this
GAP_HORIZON_S = 30  # does the market move toward fair over this horizon?


def load(data_dir):
    events = {}  # ticker -> dict
    with open(os.path.join(data_dir, "events_meta.csv")) as f:
        for r in csv.DictReader(f):
            events[r["ticker"]] = {
                "strike": float(r["strike"]),
                "open_ts": int(r["open_ts"]),
                "idx": None,
            }
    # chronological order for the split
    order = sorted(events, key=lambda k: events[k]["open_ts"])
    for i, t in enumerate(order):
        events[t]["idx"] = i

    cols = {"eidx": [], "tte": [], "spot": [], "prob": [], "outcome": [], "t": []}
    with gzip.open(os.path.join(data_dir, "rows.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            e = events.get(r["ticker"])
            if e is None:
                continue
            cols["eidx"].append(e["idx"])
            cols["tte"].append(float(r["tte_s"]))
            cols["spot"].append(float(r["spot"]))
            cols["prob"].append(float(r["prob"]))
            cols["outcome"].append(int(r["outcome"]))
            cols["t"].append(int(r["t"]))
    arr = {k: np.asarray(v) for k, v in cols.items()}
    strikes = np.zeros(len(order))
    for t in order:
        strikes[events[t]["idx"]] = events[t]["strike"]
    return arr, strikes, order


class Surface(nn.Module):
    """structured: logit = z' * exp(clamp(u(z', log tau'))) — anchored surface.
    direct (ablation): logit = MLP(z', log tau') raw — no anchors."""

    def __init__(self, mode=MODE):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(2, HID), nn.Tanh(), nn.Linear(HID, HID), nn.Tanh(), nn.Linear(HID, 1)
        )

    def forward(self, zp, log_tau):
        x = torch.stack([zp, log_tau], dim=-1)
        raw = self.net(x).squeeze(-1)
        if self.mode == "direct":
            return raw
        u = CLAMP * torch.tanh(raw / CLAMP)
        return zp * torch.exp(u)


def logit_of(surface, spot, tte, b, s):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    z = (spot - b) / s
    zp = z / tau.sqrt()
    return surface(zp, tau.log())


def bce_on_probs(logit, target):
    return nn.functional.binary_cross_entropy_with_logits(logit, target)


def online_rehearsal(surface, arr, strikes, ev_ids, rho_bar):
    """Freeze u; per event fit (Δb, Δρ) on tte>REHEARSAL_FIT_S rows; predict the
    rest. Returns per-row (gap, future market move) for the edge diagnostic."""
    gaps, moves, briers, m_briers = [], [], [], []
    for e in ev_ids:
        m = arr["eidx"] == e
        tte, spot, prob, t = (arr[k][m] for k in ("tte", "spot", "prob", "t"))
        outc = arr["outcome"][m][0] if m.any() else 0
        fit = tte > REHEARSAL_FIT_S
        if fit.sum() < 60 or (~fit).sum() < 60:
            continue
        db = torch.zeros(1, requires_grad=True)
        dr = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([db, dr], lr=0.05)
        ts_f = torch.tensor(spot[fit], dtype=torch.float32)
        tt_f = torch.tensor(tte[fit], dtype=torch.float32)
        y_f = torch.tensor(np.clip(prob[fit], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
        strike = float(strikes[e])
        for _ in range(150):
            opt.zero_grad()
            lo = logit_of(surface, ts_f, tt_f, strike + 50 * db, torch.exp(rho_bar + dr))
            loss = bce_on_probs(lo, y_f) + LAMBDA_B * db.pow(2).sum() + LAMBDA_R * dr.pow(2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = tte <= REHEARSAL_FIT_S
            lo = logit_of(
                surface,
                torch.tensor(spot[pred], dtype=torch.float32),
                torch.tensor(tte[pred], dtype=torch.float32),
                strike + 50 * db,
                torch.exp(rho_bar + dr),
            )
            fair = torch.sigmoid(lo).numpy()
        mkt = prob[pred]
        tp = t[pred]
        briers.append(np.mean((fair - outc) ** 2))
        m_briers.append(np.mean((mkt - outc) ** 2))
        # gap -> subsequent market move (does market close the gap?)
        idx = np.argsort(tp)
        tp_s, mkt_s, fair_s = tp[idx], mkt[idx], fair[idx]
        j = np.searchsorted(tp_s, tp_s + GAP_HORIZON_S)
        ok = j < len(tp_s)
        gaps.append((fair_s - mkt_s)[ok])
        moves.append((mkt_s[j[ok]] - mkt_s[ok]))
    gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    moves = np.concatenate(moves) if moves else np.zeros(0)
    return gaps, moves, float(np.mean(briers)) if briers else float("nan"), float(np.mean(m_briers)) if m_briers else float("nan")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/model"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "out/fair_model"
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    arr, strikes, order = load(data_dir)
    n_ev = len(order)
    n_val = max(1, int(n_ev * VAL_FRAC))
    val_ids = set(range(n_ev - n_val, n_ev))  # most recent events
    print(f"events={n_ev} rows={len(arr['tte'])} val_events={n_val}")

    is_val = np.isin(arr["eidx"], list(val_ids))
    tr = ~is_val
    # per-event equal weights
    cnt = np.bincount(arr["eidx"], minlength=n_ev).astype(np.float64)
    w = 1.0 / cnt[arr["eidx"]]

    dev = torch.device("cpu")
    surface = Surface().to(dev)
    d_b = nn.Parameter(torch.zeros(n_ev))  # Δb in units of $50
    d_r = nn.Parameter(torch.zeros(n_ev))
    rho_bar = nn.Parameter(torch.tensor(math.log(150.0)))
    opt = torch.optim.Adam(
        [
            {"params": surface.parameters(), "weight_decay": WD},
            {"params": [d_b, d_r, rho_bar]},
        ],
        lr=LR,
    )

    T = {
        k: torch.tensor(arr[k][tr], dtype=torch.float32)
        for k in ("tte", "spot", "prob")
    }
    T["eidx"] = torch.tensor(arr["eidx"][tr], dtype=torch.long)
    T["w"] = torch.tensor(w[tr], dtype=torch.float32)
    T["prob"] = T["prob"].clamp(P_CLIP, 1 - P_CLIP)
    strikes_t = torch.tensor(strikes, dtype=torch.float32)
    n_tr = len(T["tte"])

    for ep in range(EPOCHS):
        perm = torch.randperm(n_tr)
        tot = 0.0
        for i in range(0, n_tr, BATCH):
            ix = perm[i : i + BATCH]
            e = T["eidx"][ix]
            b = strikes_t[e] + 50.0 * d_b[e]
            s = torch.exp(rho_bar + d_r[e])
            lo = logit_of(surface, T["spot"][ix], T["tte"][ix], b, s)
            bce = nn.functional.binary_cross_entropy_with_logits(
                lo, T["prob"][ix], weight=T["w"][ix], reduction="sum"
            ) / T["w"][ix].sum()
            loss = bce + LAMBDA_B * d_b[e].pow(2).mean() + LAMBDA_R * d_r[e].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(bce.detach()) * len(ix)
        print(f"epoch {ep + 1}/{EPOCHS} train_bce={tot / n_tr:.4f} rho_bar={float(rho_bar):.3f} (s~${math.exp(float(rho_bar)):.0f})", flush=True)

    # ── eval: online rehearsal on validation events ──
    surface.eval()
    for p in surface.parameters():
        p.requires_grad_(False)
    gaps, moves, brier_f, brier_m = online_rehearsal(
        surface, arr, strikes, sorted(val_ids), rho_bar.detach()
    )
    print("\n=== ONLINE REHEARSAL (val events, fit on tte>600s, predict tte<=600s) ===")
    print(f"fair-vs-market gap: mean={gaps.mean():+.4f} p50(|gap|)={np.median(np.abs(gaps)):.4f} p90={np.quantile(np.abs(gaps), 0.9):.4f}")
    if len(gaps) > 100:
        slope = float(np.polyfit(gaps, moves, 1)[0])
        big = np.abs(gaps) > 0.05
        print(f"gap->market-move({GAP_HORIZON_S}s) slope={slope:+.3f}  (>0 = market moves toward fair = edge)")
        if big.any():
            print(f"|gap|>5c rows: {big.sum()}  mean signed capture={float(np.mean(np.sign(gaps[big]) * moves[big])):+.4f}")
    print(f"Brier vs OUTCOME (diagnostic): model={brier_f:.4f}  market={brier_m:.4f}")

    # ── export ──
    layers = []
    for mod in surface.net:
        if isinstance(mod, nn.Linear):
            layers.append({"w": mod.weight.tolist(), "b": mod.bias.tolist()})
    export = {
        "arch": {"hidden": HID, "clamp": CLAMP, "act": "tanh", "logit": "raw" if MODE == "direct" else "zp*exp(u)", "mode": MODE},
        "features": "zp=(spot-b)/s/sqrt(tte/900); u_in=[zp, log(tte/900)]",
        "rho_bar": float(rho_bar),
        "b_prior": "strike",
        "b_scale": 50.0,
        "layers": layers,
        "train": {"events": n_ev, "rows": int(len(arr["tte"])), "target": "market_prob"},
    }
    with open(os.path.join(out_dir, "fair_model.json"), "w") as f:
        json.dump(export, f)
    print(f"\nexported -> {out_dir}/fair_model.json")


if __name__ == "__main__":
    main()
