#!/usr/bin/env python3
"""Fine-tune + hold-to-settle simulation on the 50ms live-sampler data.

Protocol (user spec, 2026-07-04):
  1. events from data/samples (one clock, real books) split 50/50 chronologically
  2. FINE-TUNE the offline MLP (init = fair_model.json) on the first half
     (target: book MID, per-event (Δb,Δρ) as in offline training, low LR)
  3. TEST the strategy on the second half with REAL ORDER-BOOK prices:
     buy YES at the ask (yask), buy NO at the NO-ask (= 1 − ybid), + taker fee;
     hold to settlement. First trigger per event, gap = fair − mid.
  4. per-event (Δb,Δρ) on the test half: fit on tte>300s, trade on tte<=300s
     (the winning window from the fit-window sweep).

Meta (strike + result) is fetched from the public Kalshi API per ticker and
cached next to the data.

Usage: python sim_50ms.py <samples.csv.gz> <fair_model.json> [out_model.json]
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn

P_CLIP = 0.01
FIT_TTE_S = int(os.environ.get("FIT_TTE_S", "300"))  # fit (db,dr) on tte>this, trade on tte<=this
FIT_STEPS = 150
FIT_LR = 0.05
TUNE_EPOCHS = 8
TUNE_LR = 5e-4
FEE_RATE = 0.07
DELTAS = [float(x) for x in os.environ.get("DELTAS", "0.03,0.05,0.075,0.10,0.15,0.20").split(",")]
TUNE_RESAMPLE = 20  # 50ms -> 1s rows for fine-tuning
SIM_RESAMPLE = int(os.environ.get("SIM_RESAMPLE", "5"))  # 1 = full 50ms trigger scan
MAX_SPREAD = 0.15  # ignore rows with a wider YES book (thin/next-window noise)
# Diagnostics: PERSIST_ROWS = consecutive scan rows |gap|>=delta must HOLD
# before entering (debounce book flickers; 8 rows @250ms = 2s). NO_TUNE=1
# skips fine-tuning (tests the original weights as a control).
PERSIST_ROWS = int(os.environ.get("PERSIST_ROWS", "1"))
NO_TUNE = os.environ.get("NO_TUNE", "") == "1"
# ALL_TEST=1: no split — every usable event is a test event (base weights only).
ALL_TEST = os.environ.get("ALL_TEST", "") == "1"
# EXIT_MODE: "settle" (hold to settlement) | "revert" (exit when the gap closes
# to <= EXIT_EPS, capped at EXIT_HORIZON_S seconds; settle only as fallback).
EXIT_MODE = os.environ.get("EXIT_MODE", "settle")
# FIT_MODE: "static" (one (db,dr) fit on tte>FIT_TTE_S) | "expand" (refit at
# every minute boundary on ALL history so far, warm-started — each episode's
# gap uses the freshest causal calibration, like the live rolling refit).
FIT_MODE = os.environ.get("FIT_MODE", "static")
# ENTRY_MODE (settle mode): "taker" (lift the ask + fee) | "maker" (rest a bid
# at our side's best bid; FILLS only if the opposing ask later crosses down
# through it before settlement; fee-free). Measures maker adverse selection.
ENTRY_MODE = os.environ.get("ENTRY_MODE", "taker")
# No NEW entries at/below this tte (resting maker orders may still fill later).
ENTRY_MIN_TTE_S = float(os.environ.get("ENTRY_MIN_TTE_S", "0"))
# Maker fill window: bid rests this many seconds then cancels (0 = to settle).
MAKER_FILL_S = float(os.environ.get("MAKER_FILL_S", "0"))
# Only events whose data starts at/after this unix-seconds time (0 = all).
MIN_OPEN_TS = float(os.environ.get("MIN_OPEN_TS", "0"))
EXIT_EPS = float(os.environ.get("EXIT_EPS", "0.02"))
EXIT_HORIZON_S = float(os.environ.get("EXIT_HORIZON_S", "120"))
# FEAT=path: per-timestamp extra-feature CSV (t_ms,<name>,...). When set, rows
# without a feature within FEAT_STALE_MS are DROPPED (for every model, so row
# sets stay identical across model variants). The model JSON's "extras" list
# selects which columns feed the net (z-scored with the JSON's mu/sd).
FEAT = os.environ.get("FEAT", "")
FEAT_STALE_MS = float(os.environ.get("FEAT_STALE_MS", "1000"))
# FEAT_OFFSET_MS: added to feature timestamps before the as-of join. Use with
# server-time-keyed features to model feed latency: offset=100 means a book
# state stamped T becomes visible to the strategy at local time T+100ms.
FEAT_OFFSET_MS = float(os.environ.get("FEAT_OFFSET_MS", "0"))
# FEAT_NATIVE=1: compute imb1 from the sampler's own perp_bid_sz/perp_ask_sz
# columns (post-3759b5c rows) — causally exact, no join. Rows without sizes
# are dropped (for every model, keeping row sets identical).
FEAT_NATIVE = os.environ.get("FEAT_NATIVE", "") == "1"
# PX_FROM_FEAT=1: use the FEAT file's `mid` column (as-of joined, offset
# applied) as the MODEL PRICE INPUT instead of the sampler's perp column —
# lets other-venue surfaces be simulated with their own venue's price.
# Execution (kalshi book) stays from the sampler.
PX_FROM_FEAT = os.environ.get("PX_FROM_FEAT", "") == "1"
# PX_NATIVE=cb: model price = the sampler's own coinbase quote mid (cb_bid/
# cb_ask, age<=5s) — the online-collected settlement-chain price. Rows
# without a usable cb quote are dropped.
PX_NATIVE = os.environ.get("PX_NATIVE", "")
# MAX_ENTRIES_PER_EVENT: cap stacked episodes per event per delta (0 = off).
# Motivation (2026-07-08 event-level study): losses concentrate in trending
# events where the gap keeps re-opening — 25-trade one-sided pileups; you
# stack the most exactly when you're most wrong.
MAX_EV = int(os.environ.get("MAX_ENTRIES_PER_EVENT", "0"))
# TRADES_OUT=path: dump per-trade rows (settle mode) for clustered stats.
TRADES_OUT = os.environ.get("TRADES_OUT", "")

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_meta(tickers, cache_path):
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    for t in tickers:
        if t in cache:
            continue
        try:
            with urllib.request.urlopen(f"{KALSHI}/markets/{t}", timeout=15) as r:
                m = json.load(r)["market"]
            cache[t] = {
                "strike": m.get("floor_strike"),
                "result": m.get("result", ""),
                "close_ts": m.get("close_time", ""),
            }
        except Exception as e:  # noqa: BLE001
            print(f"  meta fetch {t}: {e}")
            cache[t] = {"strike": None, "result": "", "close_ts": ""}
        time.sleep(0.1)
    json.dump(cache, open(cache_path, "w"))
    return cache


def load_samples(path):
    """-> per-ticker dict of numpy arrays (ts_ms, tte_s, perp_mid, ybid, yask)."""
    ev = {}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for r in csv.DictReader(f):
            try:
                ybid, yask = float(r["ybid"]), float(r["yask"])
                if ybid <= 0.0 or yask <= 0.0 or yask - ybid > MAX_SPREAD or yask <= ybid:
                    continue
                tte = int(r["tte_ms"])
                if tte <= 0 or tte > 900_000:
                    continue
                pbs, pas = r.get("perp_bid_sz"), r.get("perp_ask_sz")
                imb1 = float("nan")
                if pbs is not None and pas is not None:
                    b_sz, a_sz = float(pbs), float(pas)
                    if b_sz + a_sz > 0:
                        imb1 = (b_sz - a_sz) / (b_sz + a_sz)
                cbmid = float("nan")
                cb_b, cb_a, cb_age = r.get("cb_bid"), r.get("cb_ask"), r.get("cb_age_ms")
                if cb_b is not None and cb_a is not None and cb_age is not None:
                    b_px, a_px, age = float(cb_b), float(cb_a), float(cb_age)
                    if b_px > 0 and a_px > 0 and 0 <= age <= 5000:
                        cbmid = (b_px + a_px) / 2.0
                e = ev.setdefault(r["ticker"], [])
                e.append(
                    (
                        int(r["ts_ms"]),
                        tte / 1000.0,
                        (float(r["perp_bid"]) + float(r["perp_ask"])) / 2.0,
                        ybid,
                        yask,
                        imb1,
                        cbmid,
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
    out = {}
    for t, rows in ev.items():
        rows.sort()
        a = np.array(rows, dtype=np.float64)
        out[t] = {"ts": a[:, 0], "tte": a[:, 1], "spot": a[:, 2], "ybid": a[:, 3], "yask": a[:, 4], "imb1n": a[:, 5], "cbmid": a[:, 6]}
    return out


def build_surface(js):
    hid = js["arch"]["hidden"]
    n_extra = len(js.get("extras", []))
    net = nn.Sequential(nn.Linear(2 + n_extra, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for m, lw in zip(lin, js["layers"]):
        m.weight.data = torch.tensor(lw["w"], dtype=torch.float32)
        m.bias.data = torch.tensor(lw["b"], dtype=torch.float32)
    return net, js["arch"].get("mode", "direct"), js["arch"].get("clamp", 2.0)


def make_fwd(net, mode, clamp):
    def fwd(zp, log_tau, extra=None):
        cols = [zp.unsqueeze(-1), log_tau.unsqueeze(-1)]
        if extra is not None and extra.shape[-1] > 0:
            cols.append(extra)
        x = torch.cat(cols, dim=-1)
        raw = net(x).squeeze(-1)
        if mode == "direct":
            return raw
        u = clamp * torch.tanh(raw / clamp)
        return zp * torch.exp(u)

    return fwd


def logit_of(fwd, spot, tte, b, s, extra=None):
    tau = (tte / 900.0).clamp(1e-4, 1.0)
    zp = ((spot - b) / s) / tau.sqrt()
    return fwd(zp, tau.log(), extra)


def load_feat_file(path):
    """FEAT csv -> (t_ms sorted int64 array, {col: value array}).

    Accepts either a `t_ms` column (ms) or a `t` column (unix seconds, the
    1s-grid extractor format)."""
    op = gzip.open if path.endswith(".gz") else open
    cols = None
    rows = []
    with op(path, "rt") as f:
        rd = csv.reader(f)
        cols = next(rd)
        for r in rd:
            rows.append(tuple(float(x) for x in r))
    a = np.array(rows, dtype=np.float64)
    tcol = "t_ms" if "t_ms" in cols else "t"
    ti = cols.index(tcol)
    t = a[:, ti] * (1 if tcol == "t_ms" else 1000)
    o = np.argsort(t)
    a = a[o]
    return t[o].astype(np.int64), {c: a[:, i] for i, c in enumerate(cols) if c not in ("t_ms", "t")}


def fee(p):
    return FEE_RATE * p * (1.0 - p)


def fit_event(fwd, spot, tte, target, strike, rho_bar, b_scale, steps=FIT_STEPS, init=None, extra=None):
    db = torch.tensor([init[0]] if init else [0.0], requires_grad=True)
    dr = torch.tensor([init[1]] if init else [0.0], requires_grad=True)
    opt = torch.optim.Adam([db, dr], lr=FIT_LR)
    ts = torch.tensor(spot, dtype=torch.float32)
    tt = torch.tensor(tte, dtype=torch.float32)
    xx = torch.tensor(extra, dtype=torch.float32) if extra is not None else None
    y = torch.tensor(np.clip(target, P_CLIP, 1 - P_CLIP), dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad()
        lo = logit_of(fwd, ts, tt, strike + b_scale * db, torch.exp(rho_bar + dr), xx)
        nn.functional.binary_cross_entropy_with_logits(lo, y).backward()
        opt.step()
    return db.detach(), dr.detach()


def main():
    samples_paths = sys.argv[1].split(",")  # comma-separated CSVs (multi-day)
    model_path = sys.argv[2]
    out_model = sys.argv[3] if len(sys.argv) > 3 else ""
    js = json.load(open(model_path))
    net, mode, clamp = build_surface(js)
    fwd = make_fwd(net, mode, clamp)
    rho_bar = torch.tensor(float(js["rho_bar"]), requires_grad=True)
    b_scale = float(js.get("b_scale", 50.0))

    ev = {}
    for p in samples_paths:
        for t, d in load_samples(p).items():
            if t in ev:  # same ticker spanning a file boundary: concatenate
                ev[t] = {k: np.concatenate([ev[t][k], d[k]]) for k in d}
            else:
                ev[t] = d

    # ── extra features: align onto each event's rows; DROP rows w/o fresh
    # feature (identical row sets whether or not the model consumes them) ──
    extras_names = js.get("extras", [])
    if PX_NATIVE == "cb":
        kept = dropped = 0
        for t, d in ev.items():
            ok = ~np.isnan(d["cbmid"])
            d["spot"] = d["cbmid"]
            for k in list(d):
                d[k] = d[k][ok]
            dropped += int((~ok).sum())
            kept += int(ok.sum())
        print(f"native cb price: kept {kept:,} rows, dropped {dropped:,} (no fresh cb quote)")
    if FEAT_NATIVE:
        if [c for c in extras_names if c != "imb1"]:
            sys.exit(f"FEAT_NATIVE only provides imb1; model wants {extras_names}")
        mu = np.array(js.get("mu", [0.0] * len(extras_names)))
        sd = np.array(js.get("sd", [1.0] * len(extras_names)))
        dropped = kept = 0
        for t, d in ev.items():
            ok = ~np.isnan(d["imb1n"])
            if extras_names:
                d["X"] = np.clip((d["imb1n"][:, None] - mu) / sd, -5, 5)
            else:
                d["X"] = np.zeros((len(d["ts"]), 0))
            for k in list(d):
                if k != "X":
                    d[k] = d[k][ok]
            d["X"] = d["X"][ok]
            dropped += int((~ok).sum())
            kept += int(ok.sum())
        print(f"native imb1: kept {kept:,} rows, dropped {dropped:,} (no sizes)")
    elif FEAT:
        ft, fcols = load_feat_file(FEAT)
        ft = ft + int(FEAT_OFFSET_MS)
        mu = np.array(js.get("mu", [0.0] * len(extras_names)))
        sd = np.array(js.get("sd", [1.0] * len(extras_names)))
        dropped = kept = 0
        for t, d in ev.items():
            idx = np.searchsorted(ft, d["ts"].astype(np.int64), side="right") - 1
            ok = idx >= 0
            ok[ok] &= (d["ts"][ok] - ft[idx[ok]]) <= FEAT_STALE_MS
            if PX_FROM_FEAT:
                d["spot"] = np.where(ok, fcols["mid"][np.maximum(idx, 0)], np.nan)
            if extras_names:
                raw = np.stack([fcols[c][np.maximum(idx, 0)] for c in extras_names], axis=1)
                d["X"] = np.clip((raw - mu) / sd, -5, 5)
            else:
                d["X"] = np.zeros((len(d["ts"]), 0))
            for k in list(d):
                if k != "X":
                    d[k] = d[k][ok]
            d["X"] = d["X"][ok]
            dropped += int((~ok).sum())
            kept += int(ok.sum())
        print(f"feature align: kept {kept:,} rows, dropped {dropped:,} (stale>{FEAT_STALE_MS:.0f}ms)")
    else:
        if extras_names:
            sys.exit(f"model needs extras {extras_names} but FEAT not set")
        for d in ev.values():
            d["X"] = np.zeros((len(d["ts"]), 0))
    meta = fetch_meta(sorted(ev), os.path.join(os.path.dirname(samples_paths[0]), "meta_cache.json"))
    # usable: settled, strike known, coverage in both fit and trade regions
    usable = []
    for t, d in ev.items():
        m = meta.get(t, {})
        if m.get("strike") is None or m.get("result") not in ("yes", "no"):
            continue
        need_trade = min(2000, int(FIT_TTE_S * 20 * 0.5))  # >=50% coverage of the trade window
        if (d["tte"] > FIT_TTE_S).sum() < 2000 or (d["tte"] <= FIT_TTE_S).sum() < need_trade:
            continue  # need ~100s+ of fit rows and decent trade-window coverage
        usable.append(t)
    if MIN_OPEN_TS > 0:
        usable = [t for t in usable if ev[t]["ts"][0] >= MIN_OPEN_TS * 1000.0]
    usable.sort(key=lambda t: ev[t]["ts"][0])
    n = len(usable)
    if ALL_TEST:
        tune_ev, test_ev = [], usable
    else:
        tune_ev, test_ev = usable[: n // 2], usable[n // 2 :]
    print(f"events: total={len(ev)} usable={n} tune={len(tune_ev)} test={len(test_ev)}")

    # ── fine-tune on the first half (target = book mid, 1s resample) ──
    rows = []
    for i, t in enumerate(tune_ev):
        d = ev[t]
        sl = slice(None, None, TUNE_RESAMPLE)
        mid = (d["ybid"][sl] + d["yask"][sl]) / 2.0
        rows.append(
            (
                np.full(len(mid), i),
                d["tte"][sl],
                d["spot"][sl],
                mid,
                d["X"][sl],
            )
        )
    if rows:
        eidx = np.concatenate([r[0] for r in rows]).astype(np.int64)
        tte_a = np.concatenate([r[1] for r in rows])
        spot_a = np.concatenate([r[2] for r in rows])
        mid_a = np.concatenate([r[3] for r in rows])
        x_a = np.concatenate([r[4] for r in rows])
    else:  # ALL_TEST: nothing to tune on
        eidx = np.zeros(0, dtype=np.int64)
        tte_a = spot_a = mid_a = np.zeros(0)
        x_a = np.zeros((0, len(extras_names)))
    strikes = torch.tensor([meta[t]["strike"] for t in tune_ev], dtype=torch.float32)
    print(f"fine-tune rows: {len(eidx)} (1s grid)")

    d_b = nn.Parameter(torch.zeros(len(tune_ev)))
    d_r = nn.Parameter(torch.zeros(len(tune_ev)))
    for p in net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(
        [{"params": net.parameters(), "lr": TUNE_LR}, {"params": [d_b, d_r, rho_bar], "lr": 2e-3}]
    )
    E = torch.tensor(eidx)
    TT = torch.tensor(tte_a, dtype=torch.float32)
    SP = torch.tensor(spot_a, dtype=torch.float32)
    XA = torch.tensor(x_a, dtype=torch.float32)
    Y = torch.tensor(np.clip(mid_a, P_CLIP, 1 - P_CLIP), dtype=torch.float32)
    cnt = np.bincount(eidx, minlength=len(tune_ev)).astype(np.float64)
    W = torch.tensor(1.0 / cnt[eidx], dtype=torch.float32)
    n_rows = len(eidx)
    for ep in range(0 if (NO_TUNE or not rows) else TUNE_EPOCHS):
        perm = torch.randperm(n_rows)
        tot = 0.0
        for i in range(0, n_rows, 8192):
            ix = perm[i : i + 8192]
            e = E[ix]
            lo = logit_of(fwd, SP[ix], TT[ix], strikes[e] + b_scale * d_b[e], torch.exp(rho_bar + d_r[e]), XA[ix])
            bce = nn.functional.binary_cross_entropy_with_logits(lo, Y[ix], weight=W[ix], reduction="sum") / W[ix].sum()
            opt.zero_grad()
            bce.backward()
            opt.step()
            tot += float(bce.detach()) * len(ix)
        print(f"tune epoch {ep + 1}/{TUNE_EPOCHS} bce={tot / n_rows:.4f}")
    for p in net.parameters():
        p.requires_grad_(False)
    rho_fixed = rho_bar.detach()

    if out_model:
        layers = [
            {"w": m.weight.tolist(), "b": m.bias.tolist()} for m in net if isinstance(m, nn.Linear)
        ]
        js2 = dict(js)
        js2["layers"] = layers
        js2["rho_bar"] = float(rho_fixed)
        js2["train"] = {**js.get("train", {}), "finetuned_on": f"{len(tune_ev)} 50ms events"}
        json.dump(js2, open(out_model, "w"))
        print(f"fine-tuned model -> {out_model}")

    # ── simulate on the second half with REAL book prices ──
    trades = {d: [] for d in DELTAS}
    unfilled = {d: 0 for d in DELTAS}
    tdump = []
    for t in test_ev:
        d = ev[t]
        outc = 1 if meta[t]["result"] == "yes" else 0
        strike = float(meta[t]["strike"])
        sl = slice(None, None, TUNE_RESAMPLE)  # 1s grid for the fits
        pred = d["tte"] <= FIT_TTE_S
        ss = slice(None, None, SIM_RESAMPLE)  # trigger-scan grid
        spot_p = d["spot"][pred][ss]
        tte_p = d["tte"][pred][ss]
        ybid_p = d["ybid"][pred][ss]
        yask_p = d["yask"][pred][ss]
        x_p = d["X"][pred][ss]
        if FIT_MODE == "expand":
            # refit (db,dr) at each minute boundary on ALL history so far
            # (warm-started); rows in (B-60, B] use the fit from tte>B.
            fair = np.full(len(tte_p), np.nan)
            init = None
            for B in range(FIT_TTE_S, 0, -60):
                fitm = d["tte"] > B
                if fitm[sl].sum() < 60:
                    continue
                mid_f = ((d["ybid"] + d["yask"]) / 2.0)[fitm][sl]
                db, dr = fit_event(
                    fwd, d["spot"][fitm][sl], d["tte"][fitm][sl], mid_f, strike,
                    rho_fixed, b_scale,
                    steps=FIT_STEPS if init is None else 60, init=init,
                    extra=d["X"][fitm][sl],
                )
                init = (float(db), float(dr))
                seg = (tte_p <= B) & (tte_p > B - 60)
                if seg.any():
                    with torch.no_grad():
                        lo = logit_of(
                            fwd,
                            torch.tensor(spot_p[seg], dtype=torch.float32),
                            torch.tensor(tte_p[seg], dtype=torch.float32),
                            strike + b_scale * db,
                            torch.exp(rho_fixed + dr),
                            torch.tensor(x_p[seg], dtype=torch.float32),
                        )
                        fair[seg] = torch.sigmoid(lo).numpy()
            ok = ~np.isnan(fair)
            spot_p, tte_p, ybid_p, yask_p, fair = (
                spot_p[ok], tte_p[ok], ybid_p[ok], yask_p[ok], fair[ok]
            )
        else:
            fit = d["tte"] > FIT_TTE_S
            mid_fit = (d["ybid"][fit] + d["yask"][fit]) / 2.0
            db, dr = fit_event(
                fwd, d["spot"][fit][sl], d["tte"][fit][sl], mid_fit[sl], strike, rho_fixed, b_scale,
                extra=d["X"][fit][sl],
            )
            with torch.no_grad():
                lo = logit_of(
                    fwd,
                    torch.tensor(spot_p, dtype=torch.float32),
                    torch.tensor(tte_p, dtype=torch.float32),
                    strike + b_scale * db,
                    torch.exp(rho_fixed + dr),
                    torch.tensor(x_p, dtype=torch.float32),
                )
                fair = torch.sigmoid(lo).numpy()
        mid_p = (ybid_p + yask_p) / 2.0
        gap = fair - mid_p
        for dl in DELTAS:
            # EPISODE semantics (2026-07-06): trade EVERY excursion of the gap
            # past the threshold, not just the first. Armed -> enter when
            # |gap|>=dl holds PERSIST_ROWS rows; disarm on entry; re-arm once
            # |gap|<=EXIT_EPS (hysteresis: one trade per dislocation episode).
            # revert: sequential (position closes, then re-arm). settle: each
            # episode stacks one independently-held contract.
            armed = True
            run = 0
            k = 0
            n_entered = 0
            N = len(gap)
            while k < N:
                ag = abs(gap[k])
                if not armed:
                    if ag <= EXIT_EPS:
                        armed = True
                        run = 0
                    k += 1
                    continue
                run = run + 1 if ag >= dl else 0
                if run < PERSIST_ROWS:
                    k += 1
                    continue
                # ── ENTER at k ──
                if ENTRY_MIN_TTE_S > 0 and tte_p[k] <= ENTRY_MIN_TTE_S:
                    break  # entry window closed for this event
                if MAX_EV and n_entered >= MAX_EV:
                    break  # per-event exposure cap reached
                n_entered += 1
                run = 0
                armed = False
                side_yes = gap[k] > 0
                if EXIT_MODE == "settle" and ENTRY_MODE == "maker":
                    # Rest a bid at our side's current best bid; it FILLS only if
                    # the opposing ask later crosses down through it (adverse
                    # selection: fills concentrate where price moved against us).
                    inwin = (
                        (tte_p[k] - tte_p[k + 1 :]) <= MAKER_FILL_S
                        if MAKER_FILL_S > 0
                        else np.ones(len(tte_p) - k - 1, dtype=bool)
                    )
                    if side_yes:
                        P = ybid_p[k]
                        hit = np.nonzero((yask_p[k + 1 :] <= P + 1e-9) & inwin)[0] if P > 0.005 else []
                    else:
                        P = 1.0 - yask_p[k]  # NO best bid
                        hit = np.nonzero((ybid_p[k + 1 :] >= (1.0 - P) - 1e-9) & inwin)[0] if P > 0.005 else []
                    if len(hit) == 0:
                        unfilled[dl] += 1
                        k += 1
                        continue
                    won = outc == 1 if side_yes else outc == 0
                    trades[dl].append((won, float(P), abs(gap[k]), tte_p[k], side_yes))  # fee-free
                    k += 1
                    continue
                p_entry = yask_p[k] if side_yes else 1.0 - ybid_p[k]
                if p_entry >= 0.99:
                    k += 1
                    continue
                won = outc == 1 if side_yes else outc == 0
                cost = p_entry + fee(p_entry)
                if EXIT_MODE == "settle" and TRADES_OUT:
                    # 1s-lookback trigger decomposition: how much of the gap at
                    # entry was opened by the MODEL moving (fair) vs the MARKET
                    # moving (mid) over the last second.
                    j = k - 1
                    while j > 0 and (tte_p[j] - tte_p[k]) < 1.0 and (k - j) < 40:
                        j -= 1
                    if j >= 0 and (tte_p[j] - tte_p[k]) >= 1.0 and (tte_p[j] - tte_p[k]) <= 10.0:
                        dfair = fair[k] - fair[j]
                        dmid = mid_p[k] - mid_p[j]
                    else:
                        dfair = dmid = float("nan")
                    # chase: executable entry price (our side's ask) at +100/300/500ms
                    # after the trigger row — quantifies fill slippage vs order latency.
                    def ask_at(kk):
                        return yask_p[kk] if side_yes else 1.0 - ybid_p[kk]
                    fut = []
                    for h in (0.1, 0.3, 0.5):
                        jj = k
                        while jj + 1 < len(tte_p) and (tte_p[k] - tte_p[jj]) < h:
                            jj += 1
                        fut.append(ask_at(jj) if (tte_p[k] - tte_p[jj]) >= h and (tte_p[k] - tte_p[jj]) <= h + 1.0 else float("nan"))
                    tdump.append((dl, t, won, cost, abs(gap[k]), tte_p[k], side_yes, dfair, dmid,
                                  ask_at(k), fut[0], fut[1], fut[2]))
                if EXIT_MODE == "revert":
                    elapsed = tte_p[k] - tte_p[k + 1 :]
                    closed = np.abs(gap[k + 1 :]) <= EXIT_EPS
                    cand = np.nonzero(closed | (elapsed >= EXIT_HORIZON_S))[0]
                    if len(cand):
                        j = k + 1 + cand[0]
                        how = "revert" if closed[cand[0]] else "horizon"
                        if side_yes:
                            ret_taker = (ybid_p[j] - fee(ybid_p[j])) - cost
                            ret_maker = yask_p[j] - cost
                        else:
                            no_bid = 1.0 - yask_p[j]
                            no_ask = 1.0 - ybid_p[j]
                            ret_taker = (no_bid - fee(no_bid)) - cost
                            ret_maker = no_ask - cost
                        hold_s = float(elapsed[cand[0]])
                        trades[dl].append((won, cost, abs(gap[k]), tte_p[k], side_yes, ret_taker, ret_maker, how, hold_s))
                        armed = how == "revert"  # gap<=eps at a revert exit
                        k = j + 1
                        continue
                    ret = float(won) - cost
                    trades[dl].append((won, cost, abs(gap[k]), tte_p[k], side_yes, ret, ret, "settle", float(tte_p[k])))
                    break  # rode to settlement; event over
                else:
                    trades[dl].append((won, cost, abs(gap[k]), tte_p[k], side_yes))
                k += 1

    if TRADES_OUT:
        with open(TRADES_OUT, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["delta", "ticker", "won", "cost", "gap", "tte", "side_yes", "dfair1s", "dmid1s",
                        "ask0", "ask100", "ask300", "ask500"])
            for r in tdump:
                w.writerow([r[0], r[1], int(r[2]), f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[5]:.1f}", int(r[6]),
                            f"{r[7]:.4f}", f"{r[8]:.4f}", f"{r[9]:.3f}", f"{r[10]:.3f}", f"{r[11]:.3f}", f"{r[12]:.3f}"])
        print(f"trades -> {TRADES_OUT} ({len(tdump)} rows)")

    print(f"\n=== TEST ({len(test_ev)} events): fit tte>{FIT_TTE_S}s, trade tte<={FIT_TTE_S}s, REAL bid/ask entries, exit={EXIT_MODE} ===")
    if EXIT_MODE == "revert":
        print(f"{'delta':>6} {'n':>4} {'revert%':>8} {'hzn%':>5} {'stl%':>5} {'hold_s':>7} {'taker/tr':>9} {'maker/tr':>9} {'tot_taker':>9} {'tot_maker':>9}")
        for dl in DELTAS:
            T = trades[dl]
            if not T:
                print(f"{dl:>6} {0:>4}")
                continue
            rt = np.array([x[5] for x in T])
            rm = np.array([x[6] for x in T])
            how = np.array([x[7] for x in T])
            hold = np.array([x[8] for x in T])
            print(
                f"{dl:>6} {len(T):>4} {100 * (how == 'revert').mean():>7.0f}% {100 * (how == 'horizon').mean():>4.0f}% "
                f"{100 * (how == 'settle').mean():>4.0f}% {hold.mean():>6.0f}s {rt.mean():>+9.4f} {rm.mean():>+9.4f} "
                f"{rt.sum():>+9.2f} {rm.sum():>+9.2f}"
            )
        # ── return by entry-TTE bucket (maker-exit column) ──
        buckets = [(300 - 60 * k, 300 - 60 * (k + 1), f"{300 - 60 * k}-{300 - 60 * (k + 1)}s") for k in range(5)]
        print(f"\nmaker-exit net/tr by ENTRY TTE: {'delta':>6} " + "".join(f"{name:>20}" for _, _, name in buckets))
        for dl in DELTAS:
            T = trades[dl]
            if not T:
                continue
            rm = np.array([x[6] for x in T])
            rt = np.array([x[5] for x in T])
            ttes = np.array([x[3] for x in T])
            cells = []
            for hi, lo, _ in buckets:
                m = (ttes <= hi) & (ttes > lo)
                if m.sum() == 0:
                    cells.append(f"{'-':>20}")
                else:
                    cells.append(f"{rm[m].mean():>+8.3f}/{rt[m].mean():>+7.3f} n={m.sum():<3}")
            print(f"{'':>31}{dl:>6} " + "".join(cells))
        print("   (cells: maker/taker net per trade)")
        print(f"\n(exit: gap<= {EXIT_EPS} or {EXIT_HORIZON_S:.0f}s horizon; taker = sell at bid + fee, maker = rest at ask fee-free [optimistic fills])")
    else:
        fill_col = ENTRY_MODE == "maker"
        print(f"{'delta':>6} {'n':>4} " + ("{:>6} ".format("fill%") if fill_col else "") + f"{'win%':>6} {'avg|gap|':>8} {'avg_tte':>8} {'avg_cost':>8} {'net/tr':>8} {'total':>8} {'yes%':>5}")
        for dl in DELTAS:
            T = trades[dl]
            if not T:
                print(f"{dl:>6} {0:>4}")
                continue
            won = np.array([x[0] for x in T])
            cost = np.array([x[1] for x in T])
            gaps = np.array([x[2] for x in T])
            ttes = np.array([x[3] for x in T])
            yes = np.array([x[4] for x in T])
            net = won.astype(float) - cost
            fr = f"{100 * len(T) / max(1, len(T) + unfilled[dl]):>5.0f}% " if fill_col else ""
            print(
                f"{dl:>6} {len(T):>4} " + fr + f"{100 * won.mean():>5.1f}% {gaps.mean():>8.3f} {ttes.mean():>7.0f}s "
                f"{cost.mean():>8.3f} {net.mean():>+8.4f} {net.sum():>+8.2f} {100 * yes.mean():>4.0f}%"
            )
        # ── P&L by ENTRY PRICE bucket ──
        pbuckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        print(f"\nP&L by ENTRY PRICE: {'delta':>6} " + "".join(f"{f'{lo:.1f}-{hi:.1f}':>22}" for lo, hi in pbuckets))
        for dl in DELTAS:
            T = trades[dl]
            if not T:
                continue
            won = np.array([x[0] for x in T])
            cost = np.array([x[1] for x in T])
            net = won.astype(float) - cost
            cells = []
            for lo, hi in pbuckets:
                m = (cost >= lo) & (cost < hi)
                if m.sum() == 0:
                    cells.append(f"{'-':>22}")
                else:
                    cells.append(f"{net[m].mean():>+8.3f} n={m.sum():<3} w={100 * won[m].mean():>3.0f}%")
            print(f"{'':>19}{dl:>6} " + "".join(cells))

        # ── P&L by entry-TTE bucket ──
        buckets = [(300 - 60 * k, 300 - 60 * (k + 1), f"{300 - 60 * k}-{300 - 60 * (k + 1)}s") for k in range(5)]
        print(f"\nP&L by ENTRY TTE:  {'delta':>6} " + "".join(f"{name:>22}" for _, _, name in buckets))
        for dl in DELTAS:
            T = trades[dl]
            if not T:
                continue
            won = np.array([x[0] for x in T])
            cost = np.array([x[1] for x in T])
            ttes = np.array([x[3] for x in T])
            net = won.astype(float) - cost
            cells = []
            for hi, lo, _ in buckets:
                m = (ttes <= hi) & (ttes > lo)
                if m.sum() == 0:
                    cells.append(f"{'-':>22}")
                else:
                    cells.append(f"{net[m].mean():>+8.3f} n={m.sum():<3} w={100 * won[m].mean():>3.0f}%")
            print(f"{'':>18}{dl:>6} " + "".join(cells))
        print("\n(1 contract/event, first trigger, hold to settle; entry = real ask + taker fee)")


if __name__ == "__main__":
    main()
