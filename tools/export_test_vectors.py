#!/usr/bin/env python3
"""Export PyTorch ground-truth vectors for the Rust FairSurface/fit parity
tests (DESIGN_FAIR_RIDE §7 gates 1-2).

Outputs (into crates/processor/testdata/):
  surface_vectors.json  500 random (px, tte_s, b, s) -> logit (torch f32)
  fit_case.json         one real event's 1s rows + the torch fit results:
                        stage1 = cold fit on tte>300 (150 steps),
                        stage2 = warm refit on tte>240 (60 steps),
                        each with (db, dr) and the fair curve on 300>=tte>60.

Usage: python export_test_vectors.py <model.json> <sampler.csv.gz> <out_dir>
The event is auto-picked: first event in the file with >=120 usable fit rows
and full trade-window coverage.
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

P_CLIP = 0.01
MAX_SPREAD = 0.15


def build(js):
    hid = js["arch"]["hidden"]
    net = nn.Sequential(nn.Linear(2, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for m, lw in zip(lin, js["layers"]):
        m.weight.data = torch.tensor(lw["w"], dtype=torch.float32)
        m.bias.data = torch.tensor(lw["b"], dtype=torch.float32)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def logit_of(net, px, tte, b, s):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((px - b) / s) / tau.sqrt()
    return net(torch.stack([zp, tau.log()], dim=-1)).squeeze(-1)


def main():
    model_path, csv_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    js = json.load(open(model_path))
    net = build(js)
    rho_bar, b_scale = float(js["rho_bar"]), float(js["b_scale"])
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. surface vectors ──
    rng = np.random.default_rng(7)
    n = 500
    b0 = 62000.0
    tte = rng.uniform(1.0, 900.0, n)
    s = np.exp(rho_bar + rng.uniform(-1.0, 1.0, n))
    b = b0 + rng.uniform(-200.0, 200.0, n)
    px = b + rng.uniform(-4.0, 4.0, n) * s * np.sqrt(np.clip(tte / 900.0, 1e-4, 1.0))
    # f64 reference: the parity gate verifies the Rust math implements the
    # same FUNCTION; f32 price-scale cancellation (px−b at ~$62k) would add
    # ~1e-3 logit noise that has nothing to do with implementation parity.
    net64 = build(js).double()
    with torch.no_grad():
        lo = logit_of(
            net64,
            torch.tensor(px, dtype=torch.float64),
            torch.tensor(tte, dtype=torch.float64),
            torch.tensor(b, dtype=torch.float64),
            torch.tensor(s, dtype=torch.float64),
        ).numpy()
    json.dump(
        {"rho_bar": rho_bar, "b_scale": b_scale,
         "rows": [dict(px=float(a), tte_s=float(t), b=float(bb), s=float(ss), logit=float(l))
                  for a, t, bb, ss, l in zip(px, tte, b, s, lo)]},
        open(os.path.join(out_dir, "surface_vectors.json"), "w"),
    )
    print(f"surface_vectors: {n} rows")

    # ── 2. fit case: pick a well-covered event from the sampler csv ──
    ev = {}
    with gzip.open(csv_path, "rt") as f:
        for r in csv.DictReader(f):
            try:
                cb_b, cb_a = float(r["cb_bid"]), float(r["cb_ask"])
                ybid, yask = float(r["ybid"]), float(r["yask"])
                age = float(r["cb_age_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            if cb_b <= 0 or cb_a <= 0 or not (0 <= age <= 5000):
                continue
            if ybid <= 0 or yask <= 0 or yask <= ybid or yask - ybid > MAX_SPREAD:
                continue
            tte = int(r["tte_ms"]) / 1000.0
            if tte <= 0 or tte > 900:
                continue
            sec = int(r["ts_ms"]) // 1000
            ev.setdefault(r["ticker"], {})[sec] = (tte, (cb_b + cb_a) / 2.0, (ybid + yask) / 2.0)

    pick = None
    for tick, rows in ev.items():
        a = sorted(rows.values(), key=lambda x: -x[0])  # tte desc = chronological
        fit_rows = [x for x in a if x[0] > 300.0]
        core = [x for x in a if 60.0 < x[0] <= 300.0]
        if len(fit_rows) >= 120 and len(core) >= 200:
            pick = (tick, a)
            break
    assert pick, "no suitable event found"
    tick, rows = pick
    print(f"fit case event: {tick}  rows={len(rows)}")

    # strike from the meta cache next to the csv if present, else infer via
    # ticker suffix — for the test we just need a REALISTIC anchor; use the
    # median cb price rounded to strike grid (the fit absorbs the rest).
    meta_path = os.path.join(os.path.dirname(csv_path), "meta_cache.json")
    strike = None
    if os.path.exists(meta_path):
        strike = (json.load(open(meta_path)).get(tick) or {}).get("strike")
    if strike is None:
        strike = round(float(np.median([p for _, p, _ in rows])) / 250.0) * 250.0
    strike = float(strike)

    def torch_fit(sub, init, steps):
        db = torch.tensor([init[0]], requires_grad=True)
        dr = torch.tensor([init[1]], requires_grad=True)
        opt = torch.optim.Adam([db, dr], lr=0.05)
        ts = torch.tensor([p for _, p, _ in sub], dtype=torch.float32)
        tt = torch.tensor([t for t, _, _ in sub], dtype=torch.float32)
        y = torch.tensor(np.clip([m for _, _, m in sub], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
        for _ in range(steps):
            opt.zero_grad()
            lo = logit_of(net, ts, tt, strike + b_scale * db, torch.exp(torch.tensor(rho_bar) + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            opt.step()
        return float(db), float(dr)

    def fair_curve(sub, db, dr):
        with torch.no_grad():
            lo = logit_of(
                net,
                torch.tensor([p for _, p, _ in sub], dtype=torch.float32),
                torch.tensor([t for t, _, _ in sub], dtype=torch.float32),
                torch.tensor(strike + b_scale * db, dtype=torch.float32),
                torch.tensor(math.exp(rho_bar + dr), dtype=torch.float32),
            )
            return torch.sigmoid(lo).numpy().tolist()

    core = [x for x in rows if 60.0 < x[0] <= 300.0]
    fit1 = [x for x in rows if x[0] > 300.0]
    db1, dr1 = torch_fit(fit1, (0.0, 0.0), 150)
    fit2 = [x for x in rows if x[0] > 240.0]
    db2, dr2 = torch_fit(fit2, (db1, dr1), 60)
    json.dump(
        {
            "ticker": tick, "strike": strike, "rho_bar": rho_bar, "b_scale": b_scale,
            "rows": [dict(tte_s=t, px=p, mid=m) for t, p, m in rows],
            "stage1": {"fit_tte_gt": 300.0, "steps": 150, "init": [0.0, 0.0],
                       "db": db1, "dr": dr1, "fair_core": fair_curve(core, db1, dr1)},
            "stage2": {"fit_tte_gt": 240.0, "steps": 60, "init": [db1, dr1],
                       "db": db2, "dr": dr2, "fair_core": fair_curve(core, db2, dr2)},
            "core_rows": [dict(tte_s=t, px=p, mid=m) for t, p, m in core],
        },
        open(os.path.join(out_dir, "fit_case.json"), "w"),
    )
    print(f"fit_case: fit1 n={len(fit1)} (db={db1:+.4f} dr={dr1:+.4f})  "
          f"fit2 n={len(fit2)} (db={db2:+.4f} dr={dr2:+.4f})  core n={len(core)}")


if __name__ == "__main__":
    main()
