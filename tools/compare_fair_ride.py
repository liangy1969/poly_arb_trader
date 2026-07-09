#!/usr/bin/env python3
"""Equivalence check for the Rust FairRide pipeline (DESIGN_FAIR_RIDE §7
gate 3): re-implement the exact pipeline in Python with TORCH fits
(autograd Adam = the authoritative reference) on the same sampler CSV, then
compare against the replay binary's outputs:

  1. fair series (1s grid): max/RMS |Δfair|
  2. signals: matched within ±tol_ms on (target, direction), plus counts

Usage: python compare_fair_ride.py <samples.csv> <meta_cache.json>
       <model.json> <replay_prefix>
"""
import csv
import json
import math
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

P_CLIP = 0.01
# rule constants (must mirror FairRideCfg defaults)
DELTA, SHARE_MIN, OPEN_MIN, REARM = 0.05, 0.75, 0.005, 0.02
TTE_MIN, TTE_MAX, MAX_ENTRIES = 60.0, 300.0, 255
LOOK_MIN_MS, LOOK_MAX_MS = 1000, 10000
STALE_MS, MAX_SPREAD, REF_AGE_MS = 1500, 0.15, 5000
# calibrator constants (CalibCfg defaults)
FIRST_TTE, REFIT_EVERY, LAST_TTE = 300.0, 60.0, 60.0
STEPS_FIRST, STEPS_REFIT, LR, MIN_ROWS, SAMPLE_MS = 150, 60, 0.05, 60, 1000


def build(js):
    hid = js["arch"]["hidden"]
    net = nn.Sequential(nn.Linear(2, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
    for m, lw in zip([m for m in net if isinstance(m, nn.Linear)], js["layers"]):
        m.weight.data = torch.tensor(lw["w"], dtype=torch.float32)
        m.bias.data = torch.tensor(lw["b"], dtype=torch.float32)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def main():
    csv_path, meta_path, model_path, prefix = sys.argv[1:5]
    js = json.load(open(model_path))
    net = build(js)
    rho_bar, b_scale = float(js["rho_bar"]), float(js["b_scale"])
    meta = json.load(open(meta_path))

    def logit_of(px, tte, b, s):
        px = torch.as_tensor(px, dtype=torch.float32)
        tte = torch.as_tensor(tte, dtype=torch.float32)
        tau = (tte / 900.0).clamp(1e-4, 1.0)
        zp = ((px - b) / s) / tau.sqrt()
        return net(torch.stack([zp, tau.log()], dim=-1)).squeeze(-1)

    def fair_of(px, tte, strike, db, dr):
        with torch.no_grad():
            lo = logit_of([px], [tte], strike + b_scale * db, math.exp(rho_bar + dr))
            return float(torch.sigmoid(lo)[0])

    def torch_fit(rows, strike, init, steps):
        db = torch.tensor([init[0]], requires_grad=True)
        dr = torch.tensor([init[1]], requires_grad=True)
        opt = torch.optim.Adam([db, dr], lr=LR)
        ts = torch.tensor([p for _, p, _ in rows], dtype=torch.float32)
        tt = torch.tensor([t for t, _, _ in rows], dtype=torch.float32)
        y = torch.tensor(np.clip([m for _, _, m in rows], P_CLIP, 1 - P_CLIP), dtype=torch.float32)
        for _ in range(steps):
            opt.zero_grad()
            lo = logit_of(ts, tt, strike + b_scale * db, torch.exp(torch.tensor(rho_bar) + dr))
            nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
            opt.step()
        return float(db), float(dr)

    class Ev:
        def __init__(self, strike, expiry_ns):
            self.strike = strike
            self.expiry = expiry_ns
            self.rows = []       # calib samples (tte_s, px, mid)
            self.last_sample = 0
            self.ybid = self.yask = 0.0
            self.y_ns = 0
            self.db = self.dr = 0.0
            self.fitted = False
            self.boundary = FIRST_TTE
            self.ring = []       # (ts_ns, px, mid)
            self.entries = 0
            self.armed = True

    evs = {}
    cb_px, cb_ns = float("nan"), 0
    last_cb = (0.0, 0.0)
    signals = []   # (ts_ms, ticker, direction)
    fair_rows = [] # (ts_ms, ticker, fair)
    last_fair_ms = defaultdict(int)

    def calib_tick(now):
        for tick, e in evs.items():
            tte = (e.expiry - now) / 1e9
            if tte <= 0:
                continue
            if now - e.last_sample >= SAMPLE_MS * 1_000_000:
                ref_fresh = cb_ns > 0 and now - cb_ns <= STALE_MS * 1_000_000
                y_fresh = e.y_ns > 0 and now - e.y_ns <= STALE_MS * 1_000_000
                two = e.ybid > 0 and e.yask > e.ybid
                if ref_fresh and y_fresh and two and e.yask - e.ybid <= MAX_SPREAD:
                    e.rows.append((tte, cb_px, 0.5 * (e.ybid + e.yask)))
                    e.last_sample = now
            if tte <= e.boundary and e.boundary >= LAST_TTE and len(e.rows) >= MIN_ROWS:
                init = (e.db, e.dr) if e.fitted else (0.0, 0.0)
                steps = STEPS_REFIT if e.fitted else STEPS_FIRST
                e.db, e.dr = torch_fit(e.rows, e.strike, init, steps)
                e.fitted = True
                e.boundary -= REFIT_EVERY

    def evaluate(tick, e, now):
        tte = (e.expiry - now) / 1e9
        if tte <= 0:
            return
        if not (cb_px > 0) or now - cb_ns > REF_AGE_MS * 1_000_000:
            return
        if not (e.ybid > 0 and e.yask > e.ybid and e.yask - e.ybid <= MAX_SPREAD) or now - e.y_ns > STALE_MS * 1_000_000:
            return
        mid = 0.5 * (e.ybid + e.yask)
        # ring upkeep
        while e.ring and now - e.ring[0][0] > LOOK_MAX_MS * 1_200_000:
            e.ring.pop(0)
        push = (now, cb_px, mid)
        if not e.fitted or not (TTE_MIN < tte <= TTE_MAX):
            e.ring.append(push)
            return
        fair = fair_of(cb_px, tte, e.strike, e.db, e.dr)
        gap = fair - mid
        if not e.armed:
            if abs(gap) <= REARM:
                e.armed = True
            e.ring.append(push)
            return
        if e.entries >= MAX_ENTRIES or abs(gap) < DELTA:
            e.ring.append(push)
            return
        lo = now - LOOK_MAX_MS * 1_000_000
        hi = now - LOOK_MIN_MS * 1_000_000
        then = next((x for x in reversed(e.ring) if lo <= x[0] <= hi), None)
        if then is None:
            e.ring.append(push)
            return
        ts_then, px_then, mid_then = then
        tte_then = (e.expiry - ts_then) / 1e9
        fair_then = fair_of(px_then, tte_then, e.strike, e.db, e.dr)
        side = 1.0 if gap > 0 else -1.0
        mp = side * (fair - fair_then)
        xp = -side * (mid - mid_then)
        tot = mp + xp
        if tot > OPEN_MIN and mp / tot > SHARE_MIN:
            e.entries += 1
            e.armed = False
            signals.append((now // 1_000_000, tick, 1 if gap > 0 else -1))
        e.ring.append(push)

    with open(csv_path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            ts_ms = int(r["ts_ms"])
            now = ts_ms * 1_000_000
            tick = r["ticker"]
            tte_ms = int(r["tte_ms"])
            if tte_ms <= 0:
                continue
            if tick not in evs:
                strike = (meta.get(tick) or {}).get("strike")
                if strike is None:
                    continue
                evs[tick] = Ev(float(strike), now + tte_ms * 1_000_000)
                calib_tick(now)
            try:
                cb_b, cb_a = float(r["cb_bid"]), float(r["cb_ask"])
                age = float(r["cb_age_ms"])
            except (ValueError, TypeError):
                cb_b = cb_a = 0.0
                age = -1
            # cb change event (before this row's kalshi book update)
            if cb_b > 0 and cb_a > 0 and age >= 0 and (cb_b, cb_a) != last_cb:
                last_cb = (cb_b, cb_a)
                cb_px, cb_ns = 0.5 * (cb_b + cb_a), now
                calib_tick(now)
                for tk, e in evs.items():
                    evaluate(tk, e, now)
            # kalshi book event
            ybid, yask = float(r["ybid"]), float(r["yask"])
            if ybid > 0 and yask > 0:
                e = evs[tick]
                e.ybid, e.yask, e.y_ns = ybid, yask, now
                calib_tick(now)
                evaluate(tick, e, now)
                if e.fitted and ts_ms - last_fair_ms[tick] >= 1000 and cb_px > 0:
                    last_fair_ms[tick] = ts_ms
                    tte = (e.expiry - now) / 1e9
                    if tte > 0:
                        fair_rows.append((ts_ms, tick, fair_of(cb_px, tte, e.strike, e.db, e.dr)))

    # ── compare vs replay outputs ──
    ref_fair = {}
    with open(prefix + "_fair.csv") as f:
        for r in csv.DictReader(f):
            ref_fair[(int(r["ts_ms"]), r["ticker"])] = float(r["fair"])
    diffs = []
    for ts, tick, fv in fair_rows:
        rv = ref_fair.get((ts, tick))
        if rv is not None:
            diffs.append(abs(fv - rv))
    d = np.array(diffs)
    print(f"fair series: python n={len(fair_rows)} rust n={len(ref_fair)} joined={len(d)}")
    if len(d):
        print(f"  |dfair|: max={d.max():.5f}  rms={np.sqrt((d**2).mean()):.5f}  p99={np.percentile(d,99):.5f}")

    ref_sigs = []
    with open(prefix + "_signals.csv") as f:
        for r in csv.DictReader(f):
            tick = r["target"].replace("kalshi.", "").replace(".YES", "")
            ref_sigs.append((int(r["ts_ms"]), tick, int(r["direction"])))
    tol = 500  # ms
    used = set()
    matched = 0
    for ts, tick, d_ in signals:
        for j, (rts, rtick, rd_) in enumerate(ref_sigs):
            if j in used or rtick != tick or rd_ != d_ or abs(rts - ts) > tol:
                continue
            used.add(j)
            matched += 1
            break
    print(f"signals: python={len(signals)} rust={len(ref_sigs)} matched(±{tol}ms)={matched}")
    only_py = [s for s in signals if s not in [(r, t, dd) for (r, t, dd) in signals[:0]]]
    if matched < max(len(signals), len(ref_sigs)):
        pys = set((t, d_) for _, t, d_ in signals)
        rss = [(ts, t, d_) for j, (ts, t, d_) in enumerate(ref_sigs) if j not in used]
        print("  unmatched rust:", rss[:8])
        mts = {j for j in range(len(ref_sigs)) if j in used}
        # list python signals that found no rust partner
        used2 = set()
        un_py = []
        for ts, tick, d_ in signals:
            hit = False
            for j, (rts, rtick, rd_) in enumerate(ref_sigs):
                if j in used2 or rtick != tick or rd_ != d_ or abs(rts - ts) > tol:
                    continue
                used2.add(j)
                hit = True
                break
            if not hit:
                un_py.append((ts, tick, d_))
        print("  unmatched python:", un_py[:8])


if __name__ == "__main__":
    main()
