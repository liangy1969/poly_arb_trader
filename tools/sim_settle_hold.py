#!/usr/bin/env python3
"""Hold-to-settlement strategy simulation on the fair-probability model.

Strategy (user spec, 2026-07-04): when |fair − market| ≥ δ, buy YES if
fair > market else NO, ONE contract, FIRST trigger per event only (matches
the live 1-net-per-market cap), hold to settlement. Sweep δ.

Faithful to the live procedure: uses the EXPORTED fair_model.json (same
artifact the Rust processor will read), per-event (Δb, Δρ) fitted ONLY on
the event's first 5 minutes (tte > 600s), trading only on tte ≤ 600s rows.
Validation events only (chronological last 10% — never seen in training).

Costs: entry pays last-trade prob + SPREAD_C half-spread + the Kalshi taker
fee 0.07·p·(1−p); settlement pays $1/0 fee-free. Gross column = at prob,
no fee (upper bound).

Usage: python sim_settle_hold.py <data_dir> <fair_model.json>
"""
import csv
import gzip
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

P_CLIP = 0.01
FIT_TTE_S = int(os.environ.get("FIT_TTE_S", "600"))  # fit on tte > this
TRADE_TTE_S = int(os.environ.get("TRADE_TTE_S", str(FIT_TTE_S)))  # trade on tte <= this
FIT_STEPS = 150
FIT_LR = 0.05
VAL_FRAC = 0.10
SPREAD_C = 0.01
FEE_RATE = 0.07
DELTAS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
# FIT_SWEEP mode: comma-sep fit boundaries; trade window FIXED at TRADE_TTE_S
# for every variant so the comparison is apples-to-apples.
FIT_SWEEP = [int(x) for x in os.environ.get("FIT_SWEEP", "").split(",") if x]


def load(data_dir):
    events = {}
    with open(os.path.join(data_dir, "events_meta.csv")) as f:
        for r in csv.DictReader(f):
            events[r["ticker"]] = {"strike": float(r["strike"]), "open_ts": int(r["open_ts"]), "idx": None}
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


def build_surface(js):
    hid = js["arch"]["hidden"]
    net = nn.Sequential(nn.Linear(2, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for m, lw in zip(lin, js["layers"]):
        m.weight.data = torch.tensor(lw["w"], dtype=torch.float32)
        m.bias.data = torch.tensor(lw["b"], dtype=torch.float32)
    for p in net.parameters():
        p.requires_grad_(False)
    mode = js["arch"].get("mode", "structured")
    clamp = js["arch"].get("clamp", 2.0)

    def forward(zp, log_tau):
        x = torch.stack([zp, log_tau], dim=-1)
        raw = net(x).squeeze(-1)
        if mode == "direct":
            return raw
        u = clamp * torch.tanh(raw / clamp)
        return zp * torch.exp(u)

    return forward


def logit_of(fwd, spot, tte, b, s):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((spot - b) / s) / tau.sqrt()
    return fwd(zp, tau.log())


def fee(p):
    return FEE_RATE * p * (1.0 - p)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/model"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "out/fair_model/fair_model.json"
    js = json.load(open(model_path))
    fwd = build_surface(js)
    rho_bar = torch.tensor(float(js["rho_bar"]))
    b_scale = float(js.get("b_scale", 50.0))

    arr, strikes, order = load(data_dir)
    n_ev = len(order)
    val_ids = sorted(range(n_ev - max(1, int(n_ev * VAL_FRAC)), n_ev))
    print(f"events={n_ev} val_events={len(val_ids)} model_mode={js['arch'].get('mode')}")

    def simulate(fit_tte, trade_tte):
        """Fit (Δb,Δρ) on tte>fit_tte; trade on tte<=trade_tte. Returns
        (trades[delta], gap_abs_all) — gap stats over the trade window."""
        trades = {d: [] for d in DELTAS}
        gap_abs = []
        for e in val_ids:
            m = arr["eidx"] == e
            tte, spot, prob, t = (arr[k][m] for k in ("tte", "spot", "prob", "t"))
            outc = int(arr["outcome"][m][0]) if m.any() else 0
            fit = tte > fit_tte
            pred = tte <= trade_tte
            if fit.sum() < 45 or pred.sum() < 45:
                continue
            db = torch.zeros(1, requires_grad=True)
            dr = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([db, dr], lr=FIT_LR)
            ts_f = torch.tensor(spot[fit], dtype=torch.float32)
            tt_f = torch.tensor(tte[fit], dtype=torch.float32)
            y_f = torch.tensor(np.clip(prob[fit], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
            strike = float(strikes[e])
            for _ in range(FIT_STEPS):
                opt.zero_grad()
                lo = logit_of(fwd, ts_f, tt_f, strike + b_scale * db, torch.exp(rho_bar + dr))
                nn.functional.binary_cross_entropy_with_logits(lo, y_f).backward()
                opt.step()
            with torch.no_grad():
                lo = logit_of(
                    fwd,
                    torch.tensor(spot[pred], dtype=torch.float32),
                    torch.tensor(tte[pred], dtype=torch.float32),
                    strike + b_scale * db,
                    torch.exp(rho_bar + dr),
                )
                fair = torch.sigmoid(lo).numpy()
            mkt = prob[pred]
            tp = t[pred]
            tt = tte[pred]
            idx = np.argsort(tp)
            fair, mkt, tt = fair[idx], mkt[idx], tt[idx]
            gap = fair - mkt
            gap_abs.append(np.abs(gap))
            for d in DELTAS:
                hit = np.nonzero(np.abs(gap) >= d)[0]
                if len(hit) == 0:
                    continue
                i = hit[0]
                side_yes = gap[i] > 0
                p_entry = mkt[i] + SPREAD_C if side_yes else (1.0 - mkt[i]) + SPREAD_C
                p_entry = float(np.clip(p_entry, 0.01, 0.99))
                p_gross = float(np.clip(mkt[i] if side_yes else 1.0 - mkt[i], 0.01, 0.99))
                won = outc == 1 if side_yes else outc == 0
                trades[d].append((won, p_entry + fee(p_entry), p_gross, abs(gap[i]), tt[i], side_yes))
        return trades, (np.concatenate(gap_abs) if gap_abs else np.zeros(0))

    def report(trades, gap_abs, label):
        print(f"\n--- {label} ---  gap p50={np.median(gap_abs):.3f} p90={np.quantile(gap_abs, 0.9):.3f}")
        print(f"{'delta':>6} {'n':>4} {'win%':>6} {'avg|gap|':>8} {'avg_tte':>8} {'gross/tr':>9} {'net/tr':>8} {'total_net':>10} {'yes%':>5}")
        for d in DELTAS:
            T = trades[d]
            if not T:
                print(f"{d:>6} {0:>4}")
                continue
            won = np.array([x[0] for x in T])
            cost = np.array([x[1] for x in T])
            gcost = np.array([x[2] for x in T])
            gaps = np.array([x[3] for x in T])
            ttes = np.array([x[4] for x in T])
            yes = np.array([x[5] for x in T])
            net = won.astype(float) - cost
            gross = won.astype(float) - gcost
            print(
                f"{d:>6} {len(T):>4} {100 * won.mean():>5.1f}% {gaps.mean():>8.3f} {ttes.mean():>7.0f}s "
                f"{gross.mean():>+9.4f} {net.mean():>+8.4f} {net.sum():>+10.2f} {100 * yes.mean():>4.0f}%"
            )

    if FIT_SWEEP:
        print(f"FIT-WINDOW SWEEP: trade window FIXED at tte<={TRADE_TTE_S}s")
        for fit_tte in FIT_SWEEP:
            tr, ga = simulate(fit_tte, TRADE_TTE_S)
            report(tr, ga, f"fit on first {(900 - fit_tte) / 60:.1f}min (tte>{fit_tte}s)")
    else:
        tr, ga = simulate(FIT_TTE_S, TRADE_TTE_S)
        report(tr, ga, f"fit tte>{FIT_TTE_S}s, trade tte<={TRADE_TTE_S}s")
    print("\n(1 contract/event, first trigger only, hold to settlement)")
    print(f"(net = entry at prob+{SPREAD_C:.2f} half-spread + taker fee {FEE_RATE}*p*(1-p); gross = at prob, no fee)")


if __name__ == "__main__":
    main()
