#!/usr/bin/env python3
"""Standing analysis harness for the online 50ms sampler data.

Replaces the scratchpad one-offs. One run scores N models x M thresholds and
emits the five standard views:

  1. STRATEGY   net/trade, totals, event-clustered t per (model, delta)
  2. BY DATE    the same, per UTC date  -- decay/regime is visible here first
  3. BY TTE     the same, per entry-tte bucket
  4. BCE        fair vs MARKET Brier/log-loss against SETTLEMENT, per tte
                bucket, paired per-event (does the model know more than the
                quote?)
  5. CLOSURE    per trade: how long until |fair-mid| < eps, and when it closes,
                what fraction of the closure the MARKET traveled vs the FAIR
                retreating (market_share + fair_share = 1)

Every mean is reported with an EVENT-CLUSTERED t (trades inside one event are
not independent -- the gap re-opens and you stack). Green cells at n<100 have
repeatedly failed to reproduce; treat any cell without a clustered |t|>2 AND a
fresh-data replication as noise.

FRESH SPLIT: --fresh-from YYYY-MM-DD reports every table twice, in-sample vs
fresh. Use it. The frozen 2026-07-10 candidate looked significant (t=+2.8) on
the days it was chosen on and printed -15.3c/trade (t=-2.6) on the first
genuinely unseen slice; models are selected on the left panel and only the
right panel is evidence.

Usage:
  python tools/analyze_online.py \
      --samples "data/s10.csv.gz,data/s11.csv.gz" \
      --model cb=models/fair-cb-x60/fair_model.json \
      --model l2base=models/fair-l2base-x60/fair_model.json \
      --deltas 0.03,0.05 --tte 300:120 --fit-window 120 \
      --fresh-from 2026-07-11 --out-dir out/analysis

Model spec:  label=path[:px=cb|perp]
  price input defaults from the surface's train.venue (coinbase -> cb quote,
  else binance perp mid); features are reconstructed natively from the
  sampler's own columns (imb1 from perp sizes; basis/dbasis/mom from the price
  series) so the join is causally exact -- no cross-collector recv-time join.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_50ms as sim  # noqa: E402  (fit/forward math: one source of truth)

BUCKETS = [(300, 240), (240, 180), (180, 120), (120, 60), (60, 0)]
# extended pre-window buckets, prepended in main() when --tte MAX > 300
BUCKETS_EARLY = [(900, 600), (600, 450), (450, 300)]
SCAN_STRIDE = int(os.environ.get("SCAN_STRIDE", "1"))   # 50ms trigger grid (default since
# 2026-07-26: closest to the live tick-driven rule — live/sim count ratio ~1.2x vs 1.6x at
# the old 250ms default of 5; sub-250ms whipsaw crossings are where live diverged)
FIT_STRIDE = 20   # 1s fit grid


# ── stats ────────────────────────────────────────────────────────────────────
def clustered_t(tickers, values):
    """t-stat of the mean, clustering by event (sum within event, then t)."""
    if len(values) < 2:
        return float("nan")
    per = defaultdict(float)
    for tk, v in zip(tickers, values):
        per[tk] += v
    a = np.array(list(per.values()))
    if len(a) < 3 or a.std() == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a))))


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    if len(d) < 3 or d.std() == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))


def cell(rows):
    """rows: list of (ticker, net). -> compact 'net/tr n t' string."""
    if not rows:
        return f"{'--':>21}"
    net = np.array([r[1] for r in rows])
    t = clustered_t([r[0] for r in rows], net)
    return f"{100 * net.mean():>+6.1f}c n={len(net):<4d} t={t:>+5.2f}"


# ── model / feature prep ─────────────────────────────────────────────────────
def parse_model(spec):
    if "=" not in spec:
        sys.exit(f"--model needs label=path (got {spec!r})")
    label, rest = spec.split("=", 1)
    parts = rest.split(":px=")
    path = parts[0]
    js = json.load(open(path))
    px = parts[1] if len(parts) > 1 else (
        "cb" if "coinbase" in str(js.get("train", {}).get("venue", "")) else "perp"
    )
    # two-price surface (px2cb_mom): coinbase mid is the SECOND PRICE CHANNEL,
    # scaled by exp(rho_cb + dr) = s * cb_mult with the shared per-event (b, dr).
    rho_cb = js.get("rho_cb")
    cb_mult = math.exp(rho_cb - js["rho_bar"]) if rho_cb is not None else None
    # three-price surface: kraken BBO mid as the THIRD channel
    rho_kr = js.get("rho_kr")
    kr_mult = math.exp(rho_kr - js["rho_bar"]) if rho_kr is not None else None
    return {"label": label, "path": path, "js": js, "px": px,
            "extras": js.get("extras", []), "cb_mult": cb_mult, "kr_mult": kr_mult}


def prepare(ev, m):
    """Apply the model's price input + native feature reconstruction, in place
    on a COPY of the event dict. Rows the model cannot see (no fresh cb quote,
    no lookback for momentum) are dropped."""
    js, extras, px = m["js"], m["extras"], m["px"]
    mu = np.array(js.get("mu", [0.0] * len(extras)))
    sd = np.array(js.get("sd", [1.0] * len(extras)))
    out = {}
    kept = dropped = 0
    for t, d0 in ev.items():
        d = {k: v.copy() for k, v in d0.items()}
        if px == "cb":
            d["spot"] = d["cbmid"]
        elif px == "kr":
            # kraken BBO mid (online venue columns) as the price input
            d["spot"] = d["krmid"]
        ts = d["ts"]

        def lag(series, k_s):
            idx = np.searchsorted(ts, ts - k_s * 1000.0, side="right") - 1
            ok = idx >= 0
            ok[ok] &= (ts[ok] - ts[idx[ok]]) <= (k_s * 1000.0 + 2000.0)
            col = np.full(len(ts), np.nan)
            col[ok] = series[idx[ok]]
            return col

        # adj overlay: reconstruct the adj head's features and precompute the
        # per-row logit shift d["ADJ"] = w·z + b. NaN rows (probe warmup /
        # missing imb columns) stay NaN — the '+adj' twin simply cannot trade
        # there; the BASE model's row mask is NOT affected.
        ajs = m.get("adj_js")
        if ajs is not None:
            an = ajs["extras"]
            amu = np.array(ajs["mu"]); asd = np.array(ajs["sd"])
            Xa = np.full((len(ts), len(an)), np.nan)
            for ci, name in enumerate(an):
                if name.startswith("pmom"):
                    lg = lag(d["spot"], int(name[4:]))
                    Xa[:, ci] = (d["spot"] - lg) / lg
                elif name.startswith("dmid"):
                    q = np.clip((d["ybid"] + d["yask"]) / 2.0, 0.01, 0.99)
                    lmid = np.log(q / (1.0 - q))
                    Xa[:, ci] = lmid - lag(lmid, int(name[4:]))
                elif name == "imb1":
                    Xa[:, ci] = d["imb1n"]
                elif name in d:
                    Xa[:, ci] = d[name]
                else:
                    sys.exit(f"adj overlay: cannot reconstruct feature {name!r}")
            Za = np.clip((Xa - amu) / asd, -5, 5)
            d["ADJ"] = Za @ np.array(ajs["w"]) + float(ajs.get("b", 0.0))

        if extras:
            needs_cb = any(c.startswith(("basis", "dbasis")) for c in extras)
            bas = d["cbmid"] - d["spot"] if needs_cb else None
            X = np.full((len(ts), len(extras)), np.nan)
            for ci, name in enumerate(extras):
                if name.startswith("pcbmom"):
                    # PERCENT momentum of the CB mid (1s cb-tick impulse etc.);
                    # same lag tolerance as offline emom (k+2s)
                    lg = lag(d["cbmid"], int(name[6:]))
                    X[:, ci] = (d["cbmid"] - lg) / lg
                elif name.startswith("pkrmom"):
                    # PERCENT momentum of the kraken BBO mid
                    lg = lag(d["krmid"], int(name[6:]))
                    X[:, ci] = (d["krmid"] - lg) / lg
                elif name.startswith("pmom"):
                    # PERCENT momentum (the 2p model's %mom)
                    lg = lag(d["spot"], int(name[4:]))
                    X[:, ci] = (d["spot"] - lg) / lg
                elif name.startswith("mom"):
                    X[:, ci] = d["spot"] - lag(d["spot"], int(name[3:]))
                elif name == "basis":
                    X[:, ci] = bas
                elif name.startswith("dbasis"):
                    X[:, ci] = bas - lag(bas, int(name[6:]))
                elif name == "imb1":
                    X[:, ci] = d["imb1n"]
                elif name.startswith("dmid"):
                    # Kalshi-mid logit change over k seconds (adj-model input);
                    # clip matches the offline P_CLIP=0.01
                    q = np.clip((d["ybid"] + d["yask"]) / 2.0, 0.01, 0.99)
                    lmid = np.log(q / (1.0 - q))
                    X[:, ci] = lmid - lag(lmid, int(name[4:]))
                elif name in d:
                    # sampler-native column (band5, vsurge120_1200, ...): the
                    # ONLINE-computed feature value, fed by name.
                    X[:, ci] = d[name]
                else:
                    sys.exit(f"{m['label']}: cannot reconstruct feature {name!r} "
                             f"from sampler columns")
            d["X"] = np.clip((X - mu) / sd, -5, 5)
        else:
            d["X"] = np.zeros((len(ts), 0))

        ok = ~np.isnan(d["spot"])
        if m["cb_mult"] is not None:
            # two-price surface: a fresh cb quote is a PRICE input on every row
            ok &= ~np.isnan(d["cbmid"])
        if m.get("kr_mult") is not None:
            ok &= ~np.isnan(d["krmid"])
        if extras:
            ok &= ~np.isnan(d["X"]).any(axis=1)
        for k in list(d):
            d[k] = d[k][ok]
        dropped += int((~ok).sum())
        kept += int(ok.sum())
        if ok.sum() > 0:            # skip events with 0 usable rows — old-schema
            out[t] = d              # days (pre-Jul-8) lack cb/sizes -> features NaN
    print(f"  [{m['label']}] px={m['px']} extras={extras or '-'}: "
          f"kept {kept:,} rows, dropped {dropped:,}")
    return out


def load_live_calib(path):
    """Parse the live calibrator's `fit:` records (trader-events.log extract)
    -> {ticker: sorted [(tte_boundary_s, db, dr)]}. Used by --calib-from-live
    to run the sim on the LIVE calibration instead of refitting offline —
    for consistency evaluation, this removes fit-protocol divergence (the
    fairlog replay shows fair agreement ~4e-5 with logged params vs ~1e-3
    with offline refits)."""
    import re as _re
    pat = _re.compile(r"fit: calib kalshi\.(KXBTC15M-\S+?)\.YES tte=(\d+)s rows=\d+ "
                      r"db=([+-][\d.]+) dr=([+-][\d.]+)")
    out = {}
    for ln in open(path, errors="ignore"):
        mm = pat.search(ln)
        if mm:
            out.setdefault(mm.group(1), []).append(
                (float(mm.group(2)), float(mm.group(3)), float(mm.group(4))))
    for k in out:
        out[k].sort(key=lambda x: -x[0])
    return out


def fair_series(d, m, strike, rho, b_scale, fwd, tte_max, fit_window, refit_step=60, demean_window=0.0, latency_ms=0.0, live_cal=None, fit_stride=None):
    """Windowed refit at each `refit_step`-second boundary; returns the causal
    fair series on the scan grid (the live Calibrator's semantics). If
    `demean_window>0`, also returns `gbar` = the trailing mean of (fair-mid) over
    that past window, recomputed with the CURRENT params at 1s resolution."""
    fs = fit_stride if fit_stride is not None else FIT_STRIDE
    fit_sl = slice(None, None, fs)
    scan = d["tte"] <= tte_max
    ss = slice(None, None, SCAN_STRIDE)
    spot_p, tte_p = d["spot"][scan][ss], d["tte"][scan][ss]
    ybid_p, yask_p = d["ybid"][scan][ss], d["yask"][scan][ss]
    ts_p, x_p = d["ts"][scan][ss], d["X"][scan][ss]
    # two-price surface: cb mid rides along everywhere spot does
    cbm = m["cb_mult"]
    cb_p = d["cbmid"][scan][ss] if cbm is not None else None
    krm = m.get("kr_mult")
    kr_p = d["krmid"][scan][ss] if krm is not None else None
    age_p = d["cb_age"][scan][ss] if "cb_age" in d else None
    fair = np.full(len(tte_p), np.nan)
    gbar = np.full(len(tte_p), np.nan)
    # per-row calibration in effect (db, dr) — lets the closure analysis
    # recompute a FROZEN-params fair and split the fair move into
    # input-driven vs refit-absorbed (the refit is fit TO the mid, so each
    # boundary mechanically pulls fair toward the mid; counting that as
    # "capitulation" overstates the model's retreat).
    dbs = np.full(len(tte_p), np.nan)
    drs = np.full(len(tte_p), np.nan)
    init = None
    for B in range(int(tte_max), 0, -int(refit_step)):
        if live_cal is not None:
            # consistency mode: take the LIVE calibrator's logged fit at this
            # boundary (nearest within half a step); no fit here -> live had
            # none either (warmup/grace) -> carry the previous params, or skip
            # the segment entirely if live never calibrated yet.
            cand = [c for c in live_cal if abs(c[0] - B) <= refit_step / 2.0]
            if cand:
                nearest = min(cand, key=lambda c: abs(c[0] - B))
                init = (nearest[1], nearest[2])
            if init is None:
                continue
            db, dr = init
        else:
            fitm = d["tte"] > B
            if fit_window > 0:
                fitm &= d["tte"] <= B + fit_window
            if fitm[fit_sl].sum() < (30 if fit_window > 0 else 60):
                continue
            mid_f = ((d["ybid"] + d["yask"]) / 2.0)[fitm][fit_sl]
            db, dr = sim.fit_event(
                fwd, d["spot"][fitm][fit_sl], d["tte"][fitm][fit_sl], mid_f, strike,
                rho, b_scale, steps=sim.FIT_STEPS if init is None else 60, init=init,
                extra=d["X"][fitm][fit_sl],
                cb=d["cbmid"][fitm][fit_sl] if cbm is not None else None, cb_mult=cbm,
                kr=d["krmid"][fitm][fit_sl] if krm is not None else None, kr_mult=krm,
            )
            init = (float(db), float(dr))
        seg = (tte_p <= B) & (tte_p > B - int(refit_step))
        if seg.any():
            with torch.no_grad():
                lo = sim.logit_of(
                    fwd,
                    torch.tensor(spot_p[seg], dtype=torch.float32),
                    torch.tensor(tte_p[seg], dtype=torch.float32),
                    strike + b_scale * db, torch.exp(rho + dr),
                    torch.tensor(x_p[seg], dtype=torch.float32),
                    torch.tensor(cb_p[seg], dtype=torch.float32) if cbm is not None else None,
                    cbm,
                    torch.tensor(kr_p[seg], dtype=torch.float32) if krm is not None else None,
                    krm,
                )
                fair[seg] = torch.sigmoid(lo).numpy()
                dbs[seg] = float(db)
                drs[seg] = float(dr)
            if demean_window > 0:
                # baseline gap: recompute (fair-mid) on a low-freq (1s) grid over
                # the past-window + segment tte range using the CURRENT (db,dr),
                # then a trailing mean per scan point = "the mean gap the model
                # currently sees". Entry later fires on gap - gbar.
                gm = (d["tte"] > B - int(refit_step)) & (d["tte"] <= B + demean_window)
                gi = np.where(gm)[0][::fs]
                if len(gi) >= 5:
                    g_tte = d["tte"][gi]
                    with torch.no_grad():
                        glo = sim.logit_of(
                            fwd, torch.tensor(d["spot"][gi], dtype=torch.float32),
                            torch.tensor(g_tte, dtype=torch.float32),
                            strike + b_scale * db, torch.exp(rho + dr),
                            torch.tensor(d["X"][gi], dtype=torch.float32),
                            torch.tensor(d["cbmid"][gi], dtype=torch.float32) if cbm is not None else None,
                            cbm,
                            torch.tensor(d["krmid"][gi], dtype=torch.float32) if krm is not None else None,
                            krm)
                        g_fair = torch.sigmoid(glo).numpy()
                    g_gap = g_fair - (d["ybid"][gi] + d["yask"][gi]) / 2.0
                    order = np.argsort(g_tte)  # ascending tte = back in time
                    st = g_tte[order]
                    csum = np.concatenate([[0.0], np.cumsum(g_gap[order])])
                    for kk in np.where(seg)[0]:
                        a0 = np.searchsorted(st, tte_p[kk], side="left")
                        a1 = np.searchsorted(st, tte_p[kk] + demean_window, side="right")
                        if a1 - a0 >= 5:
                            gbar[kk] = (csum[a1] - csum[a0]) / (a1 - a0)
    # post-latency executable book: the traded-side ask `latency_ms` AFTER each
    # scan row. The signal is decided at the row; the fill only lands after the
    # order round-trip, by which time the book has drifted (usually toward fair,
    # eroding the edge — and running away entirely on the fastest signals).
    if latency_ms > 0:
        ts_full = d["ts"][scan]           # full-res ts within the window (monotone)
        yb_full, ya_full = d["ybid"][scan], d["yask"][scan]
        idx = np.clip(np.searchsorted(ts_full, ts_p + latency_ms, side="left"),
                      0, len(ts_full) - 1)
        fill_ybid, fill_yask = yb_full[idx].copy(), ya_full[idx].copy()
        elapsed = ts_full[idx] - ts_p
        bad = (elapsed < latency_ms * 0.5) | (elapsed > latency_ms + 250.0)
        fill_ybid[bad], fill_yask[bad] = ybid_p[bad], yask_p[bad]  # tail/gap: no drift
    else:
        fill_ybid, fill_yask = ybid_p, yask_p
    ok = ~np.isnan(fair)
    return (ts_p[ok], tte_p[ok], ybid_p[ok], yask_p[ok], fair[ok], gbar[ok],
            fill_ybid[ok], fill_yask[ok], dbs[ok], drs[ok],
            spot_p[ok], x_p[ok], cb_p[ok] if cbm is not None else None,
            age_p[ok] if age_p is not None else None,
            kr_p[ok] if krm is not None else None)


# ── trade generation ─────────────────────────────────────────────────────────
# Gap-distribution buckets by moneyness (Kalshi YES mid). The live `gapstats`
# log found a persistent ~5c |fair-mid| AT the money; this checks whether the
# offline sim reproduces it (i.e. it's model misspecification, not a live-only
# artifact). tte floor matches the live entry window (entry_min_tte_s).
GAP_MID_BUCKETS = [
    ("0.05-0.20", 0.05, 0.20), ("0.20-0.35", 0.20, 0.35), ("0.35-0.45", 0.35, 0.45),
    ("0.45-0.55 ATM", 0.45, 0.55), ("0.55-0.65", 0.55, 0.65),
    ("0.65-0.80", 0.65, 0.80), ("0.80-0.95", 0.80, 0.95),
]
GAP_TTE_MIN = 60.0


def gap_report(a, models):
    """Print the |fair-mid| gap distribution bucketed by moneyness — the offline
    analogue of the live `gapstats` log, to test whether the persistent ATM gap
    is expected model behaviour."""
    print()
    print(f"=== GAP (fair - mid) distribution by MONEYNESS, tte {GAP_TTE_MIN:.0f}-{a.tte_max:.0f}s "
          "(offline; cf. live gapstats) ===")
    print("   per-event gaps are ~one-signed but the sign varies by event, so the aggregate")
    print("   'signed' mean cancels; compare 'mean|gap|' with the live ~0.05 at the money.")
    for m in models:
        gd = a.gap_dist.get(m["label"], {})
        if not any(gd.values()):
            continue
        over_hdr = " ".join(f">={d:.2f}" for d in a.deltas)
        print(f"\n[{m['label']}]")
        print(f"{'mid bucket':>16} {'n':>9} {'mean|gap|':>9} {'signed':>8} "
              f"{'p50':>7} {'p90':>7} {'p99':>7} {'max':>7}  {over_hdr}")
        for name, _, _ in GAP_MID_BUCKETS:
            g = np.array(gd.get(name, []))
            if len(g) == 0:
                continue
            ag = np.abs(g)
            over = " ".join(f"{100 * (ag >= d).mean():>5.0f}%" for d in a.deltas)
            print(f"{name:>16} {len(g):>9} {ag.mean():>9.4f} {g.mean():>+8.4f} "
                  f"{np.percentile(ag, 50):>7.4f} {np.percentile(ag, 90):>7.4f} "
                  f"{np.percentile(ag, 99):>7.4f} {ag.max():>7.4f}  {over}")


# Entry-price (YES mid) buckets — the moneyness dimension, added alongside the
# tte dimension in every section. The live executor's ATM band is [0.30,0.70].
PRICE_BUCKETS = [(0.02, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 0.98)]


def price_bucket_of(mid):
    for lo, hi in PRICE_BUCKETS:
        if lo <= mid < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return "other"


def _atm_mark(pname):
    """'*' if a price-bucket label is entirely inside the live ATM band [0.30,0.70]."""
    try:
        lo, hi = (float(x) for x in pname.split("-"))
        return "*" if lo >= 0.30 and hi <= 0.70 else " "
    except ValueError:
        return " "


def event_calib_report(a, models):
    """Event-level calibration error: per event, (mean fair - mean mid) over the
    trading span; |.| averaged over events (isolates persistent bias from noise)."""
    print()
    print(f"=== EVENT-LEVEL calibration: |mean_fair - mean_mkt| over the span "
          f"(tte {GAP_TTE_MIN:.0f}-{a.tte_max:.0f}s), averaged over events ===")
    for m in models:
        rows = a.event_calib.get(m["label"], [])
        if not rows:
            continue
        bias = np.array([r[0] for r in rows]); mmid = np.array([r[1] for r in rows])
        print(f"\n[{m['label']}]  n_events={len(rows)}")
        print(f"   mean |mean_fair - mean_mkt| = {np.abs(bias).mean() * 100:6.2f}c"
              f"   (signed {bias.mean() * 100:+.2f}c, median|.| {np.median(np.abs(bias)) * 100:.2f}c)")
        ws_raw = a.win_signed.get(m["label"], [])
        if ws_raw:
            sg = np.array([r[0] for r in ws_raw]); ab = np.array([r[1] for r in ws_raw]); asg = np.abs(sg)
            flip = 100.0 * (np.sign(sg[1:]) != np.sign(sg[:-1])).mean() if len(sg) > 1 else float("nan")
            ratio = (asg / np.maximum(ab, 1e-9)).mean()
            print(f"   60s-window signed mean (live gapstats cadence, {len(sg)} windows):")
            print(f"      |signed|: mean {asg.mean() * 100:.2f}c  p90 {np.percentile(asg, 90) * 100:.2f}c  "
                  f"p99 {np.percentile(asg, 99) * 100:.2f}c  max {asg.max() * 100:.2f}c")
            print(f"      |signed|/|gap| ratio = {ratio:.2f}  (near 1 => the gap is ONE-SIGNED within the 60s window, as live shows)")
            print(f"      sign flips across adjacent windows {flip:.0f}% of the time  => it REVERSES over the event")
            print(f"   -> 60s |signed| ({asg.mean() * 100:.2f}c, tail to {asg.max() * 100:.1f}c) >> event-span |bias| "
                  f"({np.abs(bias).mean() * 100:.2f}c): the live large signed mean is a real one-signed gap in that window")
        print(f"   {'span-mid bucket':>16} {'n_ev':>5} {'mean|bias|':>10} {'signed':>8}")
        for name, lo, hi in GAP_MID_BUCKETS:
            sel = (mmid >= lo) & (mmid < hi)
            if sel.sum() >= 1:
                print(f"   {name:>16} {int(sel.sum()):>5} {np.abs(bias[sel]).mean() * 100:>9.2f}c "
                      f"{bias[sel].mean() * 100:>+7.2f}c")


def generate_trades(sig, m_label, gap, fair, mid_p, tte_p, ybid_p, yask_p, ts_p, outc, t, a,
                    fill_ybid, fill_yask, fx_fair=None, cb_age_p=None):
    """Arm/gate/entry episode loop for ONE entry signal on ONE event -> trade
    dicts labeled `m_label`. `sig` (raw gap or demeaned gap) drives the threshold,
    direction and re-arm; the ride gate and closure use the RAW fair/mid/gap.

    --trigger move: a DIFFERENT trigger — fire when the 1s FAIR move
    |dfair_1s| >= delta AND |dfair_1s| >= move_k * |dmid_1s| (the fair moved,
    the market hasn't followed), direction = sign(dfair_1s). No level-gap
    requirement, no ride gate (the dominance test IS the gate). Episode:
    disarm on fire, re-arm when |dfair_1s| <= delta * move_rearm."""
    out = []
    dfair_arr = dmid_arr = None
    if a.trigger == "move":
        # vectorized 1s-lookback moves: youngest j with tte_j >= tte_k + 1s
        # (<= +10s). NOTE: uses the stored causal fair at both endpoints — a
        # refit inside the lookback (~7% of rows at 15s cadence) leaks a
        # param jump into dfair, same caveat as the pre-43bea62 ride gate.
        idx = np.searchsorted(-tte_p, -(tte_p + 1.0), side="right") - 1
        okl = (idx >= 0) & ((tte_p[np.clip(idx, 0, None)] - tte_p) <= 10.0)
        dfair_arr = np.full(len(tte_p), np.nan)
        dmid_arr = np.full(len(tte_p), np.nan)
        ii = np.clip(idx, 0, None)
        dfair_arr[okl] = (fair - fair[ii])[okl]
        dmid_arr[okl] = (mid_p - mid_p[ii])[okl]
    for dl in a.deltas:
        armed, run, k, n_ent = True, 0, 0, 0
        dis_dir = 0.0   # sign of the disarming crossing (directional re-arm)
        last_cross_ms = -1e18   # ts of the previous delta-crossing (any outcome)
        while k < len(sig):
            if a.trigger == "move":
                mv, dm = dfair_arr[k], dmid_arr[k]
                ag = 0.0 if math.isnan(mv) else abs(mv)
                if not armed:
                    if ag <= dl * a.move_rearm:
                        armed, run = True, 0
                    k += 1
                    continue
                if not (ag >= dl and not math.isnan(dm) and ag >= a.move_k * abs(dm)):
                    k += 1
                    continue
                if tte_p[k] <= a.tte_min:
                    break
                if a.cap and n_ent >= a.cap:
                    break
                n_ent += 1
                armed = False
                side_yes = mv > 0
                s = 1.0 if side_yes else -1.0
                dfair1, dmid1 = mv, dm
                push, pull = s * dfair1, -s * dmid1
                tot = push + pull
                share = push / tot if tot and not math.isnan(tot) else float("nan")
                is_ride = True
            else:
                ag = abs(sig[k])
                if not armed:
                    # DIRECTIONAL re-arm (live semantics): same-direction re-fires
                    # revert to within rearm_eps; a sign flip re-arms immediately.
                    if sig[k] * dis_dir <= a.rearm_eps:
                        armed, run = True, 0
                    k += 1
                    continue
                run = run + 1 if ag >= dl else 0
                if run < 1:
                    k += 1
                    continue
                if tte_p[k] <= a.tte_min:
                    break            # entry window closed
                if a.cap and n_ent >= a.cap:
                    break            # per-event exposure cap
                n_ent += 1
                run, armed = 0, False
                side_yes = sig[k] > 0
                dis_dir = 1.0 if side_yes else -1.0
                s = 1.0 if side_yes else -1.0
                prev_cross_ms = last_cross_ms
                last_cross_ms = ts_p[k]   # EVERY crossing disarms + stamps, even
                                          # gate-rejected/vetoed ones (live parity)

                # 1s-lookback RIDE gate — LIVE SEMANTICS (rule.rs): the lookback
                # reference is the youngest sample >= lookback_min (1s) old, up
                # to lookback_max (10s); fair_then is RECOMPUTED from the
                # lookback row's raw inputs under the CURRENT calibration
                # (refit-jump immune), not read from the stored causal series.
                j = k - 1
                while j > 0 and (tte_p[j] - tte_p[k]) < a.lookback_s and (k - j) < 400:
                    j -= 1
                if j >= 0 and a.lookback_s <= (tte_p[j] - tte_p[k]) <= a.lookback_s + 9.0:
                    fair_then = fx_fair(j, k) if fx_fair is not None else float("nan")
                    if math.isnan(fair_then):
                        fair_then = fair[j]
                    dfair1, dmid1 = fair[k] - fair_then, mid_p[k] - mid_p[j]
                else:
                    dfair1 = dmid1 = float("nan")
                push, pull = s * dfair1, -s * dmid1
                tot = push + pull
                share = push / tot if tot and not math.isnan(tot) else float("nan")
                # open_min CONVENTION (user, 2026-08-03): defaults to the gap
                # threshold (delta) unless explicitly overridden
                open_eff = a.ride_open if a.ride_open is not None else dl
                is_ride = bool(tot > open_eff and share > a.ride_share)
                if a.gate == "ride" and not is_ride:
                    k += 1
                    continue
                if a.gate == "fade" and is_ride:
                    k += 1
                    continue
                if a.gate == "overreact":
                    gap_then = gap[k] - dfair1 + dmid1  # = fair[j] - mid[j]
                    ok = (not math.isnan(dfair1)
                          and dfair1 * dmid1 > 0
                          and abs(dmid1) >= a.overreact_k * abs(dfair1)
                          and s * gap_then <= -a.overreact_flip)
                    if not ok:
                        k += 1
                        continue

                # trade-only vetoes: the crossing above already consumed the
                # armed state (and stamped last_cross_ms), so unlike row-level
                # exclusion these CANNOT unlock/relocate later episodes.
                # (a) stale-signal veto: the fair at this row consumed a cb
                #     quote older than the threshold -> disarm but don't trade.
                if a.stale_veto_ms > 0 and cb_age_p is not None and \
                        not math.isnan(cb_age_p[k]) and cb_age_p[k] > a.stale_veto_ms:
                    a.vetoes.append({"model": m_label, "delta": dl, "why": "stale",
                                     "date": utc_date(ts_p[k]), "ticker": t})
                    k += 1
                    continue
                # (b) disarm cooldown: no trade within N s of the PREVIOUS
                #     delta-crossing (any direction, traded or not) -> whipsaw
                #     bursts push the stamp forward and stay untraded.
                if a.disarm_cd_s > 0 and (ts_p[k] - prev_cross_ms) < a.disarm_cd_s * 1000.0:
                    a.vetoes.append({"model": m_label, "delta": dl, "why": "cooldown",
                                     "date": utc_date(ts_p[k]), "ticker": t})
                    k += 1
                    continue

            # fill price: signal-instant ask, or (with latency) the post-latency
            # ask — but MISS if it ran more than the chase budget above the
            # signal-instant ask (the order can't catch a fast market).
            sig_ask = yask_p[k] if side_yes else 1.0 - ybid_p[k]
            if a.latency_ms > 0:
                post_ask = fill_yask[k] if side_yes else 1.0 - fill_ybid[k]
                if post_ask > sig_ask + a.chase_c + 1e-9:
                    a.misses.append({
                        "model": m_label, "delta": dl,
                        "pbucket": price_bucket_of(float(mid_p[k])),
                        "date": utc_date(ts_p[k]), "ticker": t,
                        "tte": float(tte_p[k]), "bucket": bucket_of(tte_p[k]),
                    })
                    k += 1
                    continue
                p_entry = post_ask
            else:
                p_entry = sig_ask
            if p_entry >= 0.99:
                k += 1
                continue
            cost = p_entry + sim.fee(p_entry)
            won = int(outc == 1 if side_yes else outc == 0)

            # closure: first raw |fair-mid| < eps after entry, and who traveled.
            # close_df_fx = the fair move with the ENTRY calibration frozen
            # (inputs-only); close_df - close_df_fx = the refit-absorbed part
            # (each refit is fit to the mid, mechanically closing the gap from
            # the fair side — that is calibration absorption, not the model
            # changing its mind).
            close_s = close_df = close_dm = mkt_sh = fair_sh = float("nan")
            close_df_fx = float("nan")
            jj_close = None
            for jj in range(k + 1, len(tte_p)):
                if abs(gap[jj]) < a.close_eps:
                    jj_close = jj
                    close_s = float(tte_p[k] - tte_p[jj])
                    close_df = float(fair[jj] - fair[k])
                    close_dm = float(mid_p[jj] - mid_p[k])
                    if fx_fair is not None:
                        f_fx = fx_fair(jj, k)
                        if not math.isnan(f_fx):
                            close_df_fx = float(f_fx - fair[k])
                    g0 = gap[k]
                    if abs(g0) > 1e-9:
                        mkt_sh = close_dm / g0
                        fair_sh = -close_df / g0
                    break
            # capped variant: measure at min(closure, close_cap) — a FIXED horizon,
            # so the measurement isn't conditioned on the endogenous closure time,
            # and never-closed trades are included (measured at the cap).
            cap_s = cap_df = cap_dm = cap_df_fx = float("nan")
            if a.close_cap > 0:
                jc = None
                for jj in range(k + 1, len(tte_p)):
                    if tte_p[k] - tte_p[jj] > a.close_cap:
                        break
                    jc = jj
                    if jj_close is not None and jj >= jj_close:
                        break
                if jc is not None:
                    cap_s = float(tte_p[k] - tte_p[jc])
                    cap_df = float(fair[jc] - fair[k])
                    cap_dm = float(mid_p[jc] - mid_p[k])
                    if fx_fair is not None:
                        f_fx = fx_fair(jc, k)
                        if not math.isnan(f_fx):
                            cap_df_fx = float(f_fx - fair[k])

            out.append({
                "model": m_label, "delta": dl, "ticker": t,
                "date": utc_date(ts_p[k]), "ts_ms": int(ts_p[k]),
                "tte": float(tte_p[k]), "bucket": bucket_of(tte_p[k]),
                "pbucket": price_bucket_of(float(mid_p[k])),
                "side_yes": int(side_yes), "gap": float(abs(sig[k])),
                "fair0": float(fair[k]), "mid0": float(mid_p[k]),
                "slip": float(p_entry - sig_ask),
                "cost": float(cost), "won": won, "net": float(won - cost), "outc": int(outc),
                "dfair1s": dfair1, "dmid1s": dmid1, "ride_share": share,
                "is_ride": int(is_ride), "close_s": close_s,
                "close_dfair": close_df, "close_dmid": close_dm,
                "close_dfair_fx": close_df_fx,
                "cap_s": cap_s, "cap_dfair": cap_df, "cap_dmid": cap_dm,
                "cap_dfair_fx": cap_df_fx,
                "mkt_share": mkt_sh, "fair_share": fair_sh,
                # cb quote age at entry: lets the stale-veto threshold curve be
                # computed post-hoc (veto = strict subset, no state coupling)
                "cb_age": float(cb_age_p[k]) if cb_age_p is not None else float("nan"),
            })
            k += 1
    return out


def simulate_adj(m, ev, meta, a):
    """adj-model path: fair = sigmoid(logit(mid) + w·z + b) — a LINEAR head on
    market-observable features, NO surface, NO per-event (db, dr) calibration.
    gap = fair − mid ≈ adj·p(1−p) = the predicted forward mid move, so the
    standard |gap| > delta entry ladder IS "enter when adj > threshold" in
    price units. Everything downstream (latency fills, fees, ride/none gate,
    BCE, closure, price buckets) is the shared machinery."""
    js = m["js"]
    w = np.array(js["w"], dtype=np.float64)
    b0 = float(js.get("b", 0.0))
    pc = float(js.get("p_clip", 0.01))
    trades, bce_rows = [], []
    for t in sorted(ev, key=lambda x: ev[x]["ts"][0]):
        d = ev[t]
        mm = meta.get(t, {})
        if mm.get("strike") is None or mm.get("result") not in ("yes", "no"):
            continue
        if (d["tte"] <= a.tte_max).sum() < 1000:
            continue  # need trade-window coverage (no calibration history needed)
        outc = 1 if mm["result"] == "yes" else 0
        scan = d["tte"] <= a.tte_max
        ss = slice(None, None, SCAN_STRIDE)
        tte_p = d["tte"][scan][ss]
        ybid_p, yask_p = d["ybid"][scan][ss], d["yask"][scan][ss]
        ts_p, x_p = d["ts"][scan][ss], d["X"][scan][ss]
        if len(tte_p) < 50:
            continue
        mid_p = (ybid_p + yask_p) / 2.0
        q = np.clip(mid_p, pc, 1.0 - pc)
        fair = 1.0 / (1.0 + np.exp(-(np.log(q / (1.0 - q)) + x_p @ w + b0)))
        gap = fair - mid_p
        # post-latency executable book (same semantics as fair_series)
        if a.latency_ms > 0:
            ts_full = d["ts"][scan]
            yb_full, ya_full = d["ybid"][scan], d["yask"][scan]
            idx = np.clip(np.searchsorted(ts_full, ts_p + a.latency_ms, side="left"),
                          0, len(ts_full) - 1)
            fill_ybid, fill_yask = yb_full[idx].copy(), ya_full[idx].copy()
            elapsed = ts_full[idx] - ts_p
            bad = (elapsed < a.latency_ms * 0.5) | (elapsed > a.latency_ms + 250.0)
            fill_ybid[bad], fill_yask[bad] = ybid_p[bad], yask_p[bad]
        else:
            fill_ybid, fill_yask = ybid_p, yask_p

        gd = a.gap_dist.setdefault(m["label"], {name: [] for name, _, _ in GAP_MID_BUCKETS})
        win = (tte_p > GAP_TTE_MIN) & (tte_p <= a.tte_max)
        for name, lomid, himid in GAP_MID_BUCKETS:
            sel = win & (mid_p >= lomid) & (mid_p < himid)
            if sel.any():
                gd[name].extend(gap[sel].tolist())
        span = (tte_p > GAP_TTE_MIN) & (tte_p <= a.tte_max)
        if span.sum() >= 20:
            a.event_calib.setdefault(m["label"], []).append(
                (float(fair[span].mean() - mid_p[span].mean()), float(mid_p[span].mean())))
        for hi, lo in BUCKETS:
            bsel = (tte_p <= hi) & (tte_p > lo)
            if bsel.sum() < 20:
                continue
            fc = np.clip(fair[bsel], 0.005, 0.995)
            mc = np.clip(mid_p[bsel], 0.005, 0.995)
            bf = float(-(outc * np.log(fc) + (1 - outc) * np.log(1 - fc)).mean())
            bm = float(-(outc * np.log(mc) + (1 - outc) * np.log(1 - mc)).mean())
            bce_rows.append({
                "ticker": t, "date": utc_date(ts_p[bsel][0]), "bucket": f"{lo:.2f}-{hi:.2f}",
                "bce_fair": bf, "bce_mkt": bm, "d": bf - bm,
                "brier_fair": float(((fc - outc) ** 2).mean()),
                "brier_mkt": float(((mc - outc) ** 2).mean()),
                "mae": float(np.abs(fair[bsel] - mid_p[bsel]).mean()),
            })
        # raw-gap entries only (no demeaned variant: the adj head has no
        # per-event calibration whose bias dm30 would strip)
        trades += generate_trades(gap, m["label"], gap, fair, mid_p,
                                  tte_p, ybid_p, yask_p, ts_p, outc, t, a,
                                  fill_ybid, fill_yask, fx_fair=None, cb_age_p=None)
    return trades, bce_rows


def simulate(m, ev, meta, a):
    """-> (trades, bce_rows). One pass per event; all deltas scored together."""
    if m["js"].get("kind") == "adj":
        return simulate_adj(m, ev, meta, a)
    net_s, mode, clamp = sim.build_surface(m["js"])
    fwd = sim.make_fwd(net_s, mode, clamp)
    rho = torch.tensor(float(m["js"]["rho_bar"]))
    b_scale = float(m["js"].get("b_scale", 50.0))
    trades, bce_rows = [], []

    for t in sorted(ev, key=lambda x: ev[x]["ts"][0]):
        d = ev[t]
        mm = meta.get(t, {})
        if mm.get("strike") is None or mm.get("result") not in ("yes", "no"):
            continue
        if (d["tte"] > a.tte_max).sum() < 2000 or (d["tte"] <= a.tte_max).sum() < 1000:
            continue  # need calibration history + trade-window coverage
        outc = 1 if mm["result"] == "yes" else 0
        strike = float(mm["strike"])
        lc = a.live_calib.get(t) if getattr(a, "live_calib", None) else None
        if getattr(a, "live_calib", None) and lc is None:
            continue  # consistency mode: live never calibrated this event
        (ts_p, tte_p, ybid_p, yask_p, fair, gbar, fill_ybid, fill_yask,
         dbs_p, drs_p, spot_p, x_p, cb_p, cb_age_p, kr_p) = fair_series(
            d, m, strike, rho, b_scale, fwd, a.tte_max, a.fit_window, a.refit_step,
            a.demean_window, a.latency_ms, live_cal=lc, fit_stride=a.fit_stride)
        if len(tte_p) < 50:
            continue
        mid_p = (ybid_p + yask_p) / 2.0
        gap = fair - mid_p
        cbm = m["cb_mult"]
        krm = m.get("kr_mult")
        rho_f = float(rho)

        def fx_fair(jj, kk):
            """fair at row jj recomputed under row kk's (db, dr) — the
            frozen-calibration fair used to strip refit jumps from closure."""
            if math.isnan(dbs_p[kk]):
                return float("nan")
            with torch.no_grad():
                lo = sim.logit_of(
                    fwd,
                    torch.tensor(spot_p[jj:jj + 1], dtype=torch.float32),
                    torch.tensor(tte_p[jj:jj + 1], dtype=torch.float32),
                    strike + b_scale * float(dbs_p[kk]),
                    math.exp(rho_f + float(drs_p[kk])),
                    torch.tensor(x_p[jj:jj + 1], dtype=torch.float32),
                    torch.tensor(cb_p[jj:jj + 1], dtype=torch.float32) if cb_p is not None else None,
                    cbm,
                    torch.tensor(kr_p[jj:jj + 1], dtype=torch.float32) if kr_p is not None else None,
                    krm,
                )
                return float(torch.sigmoid(lo)[0])

        # gap distribution by moneyness (offline analogue of the live `gapstats`
        # log): every fair-model row in the entry window, bucketed by mid level.
        gd = a.gap_dist.setdefault(m["label"], {name: [] for name, _, _ in GAP_MID_BUCKETS})
        win = (tte_p > GAP_TTE_MIN) & (tte_p <= a.tte_max)
        for name, lomid, himid in GAP_MID_BUCKETS:
            sel = win & (mid_p >= lomid) & (mid_p < himid)
            if sel.any():
                gd[name].extend(gap[sel].tolist())

        # event-level calibration: mean(fair) - mean(mid) over the trading span
        # (the persistent per-event bias; intra-event oscillations average out).
        # |.| is aggregated over events in the report. Stored with the span-mean
        # mid so we can bucket the bias by moneyness.
        span = (tte_p > GAP_TTE_MIN) & (tte_p <= a.tte_max)
        if span.sum() >= 20:
            a.event_calib.setdefault(m["label"], []).append(
                (float(fair[span].mean() - mid_p[span].mean()), float(mid_p[span].mean())))
        # per-60s-window signed gap mean — REPRODUCES the live gapstats cadence.
        # If |signed| here is large (like live) while the event-span bias above is
        # small, the "large signed mean" is the 60s window catching a one-signed
        # gap that REVERSES across windows (not a live bug).
        for hi, lo in BUCKETS:
            wm = (tte_p <= hi) & (tte_p > lo) & (tte_p > GAP_TTE_MIN)
            if wm.sum() >= 20:
                a.win_signed.setdefault(m["label"], []).append(
                    (float(gap[wm].mean()), float(np.abs(gap[wm]).mean())))

        # optional: dump the causal fair series for replay-parity checks
        if a.dump_fair:
            for k in range(len(tte_p)):
                a.fair_rows.append((t, int(ts_p[k]), float(tte_p[k]), float(fair[k]), float(mid_p[k])))

        # ── (4) settlement BCE, fair vs market, per tte bucket, per event ──
        for hi, lo in BUCKETS:
            b = (tte_p <= hi) & (tte_p > lo)
            if b.sum() < 20:
                continue
            fc = np.clip(fair[b], 0.005, 0.995)
            mc = np.clip(mid_p[b], 0.005, 0.995)
            bf = float(-(outc * np.log(fc) + (1 - outc) * np.log(1 - fc)).mean())
            bm = float(-(outc * np.log(mc) + (1 - outc) * np.log(1 - mc)).mean())
            bce_rows.append({
                "ticker": t, "date": utc_date(ts_p[b][0]), "bucket": f"{hi}-{lo}s",
                "bce_fair": bf, "bce_mkt": bm, "d": bf - bm,
                "brier_fair": float(((fc - outc) ** 2).mean()),
                "brier_mkt": float(((mc - outc) ** 2).mean()),
                "mae": float(np.abs(fair[b] - mid_p[b]).mean()),
            })

        # ── (4b) settlement BCE by PRICE bucket (all rows in the entry window,
        # unconditional on any trade -> delta-independent, unlike a trade-
        # conditional BCE). Shares bce_rows; the label distinguishes it. ──
        for lo, hi in PRICE_BUCKETS:
            b = (mid_p >= lo) & (mid_p < hi) & (tte_p <= a.tte_max) & (tte_p > a.tte_min)
            if b.sum() < 20:
                continue
            fc = np.clip(fair[b], 0.005, 0.995)
            mc = np.clip(mid_p[b], 0.005, 0.995)
            bf = float(-(outc * np.log(fc) + (1 - outc) * np.log(1 - fc)).mean())
            bm = float(-(outc * np.log(mc) + (1 - outc) * np.log(1 - mc)).mean())
            bce_rows.append({
                "ticker": t, "date": utc_date(ts_p[b][0]), "bucket": f"{lo:.2f}-{hi:.2f}",
                "bce_fair": bf, "bce_mkt": bm, "d": bf - bm,
                "brier_fair": float(((fc - outc) ** 2).mean()),
                "brier_mkt": float(((mc - outc) ** 2).mean()),
                "mae": float(np.abs(fair[b] - mid_p[b]).mean()),
            })

        # entry STRATEGIES: raw gap, plus the DEMEANED gap (gap - trailing mean,
        # recomputed with the current params) when --demean-window > 0. Demeaned
        # trades are labeled `<model>.dm<W>` so ONE run reports raw vs demeaned
        # side by side in every section.
        strats = [("", gap, gap, fair, fx_fair)]
        if a.demean_window > 0:
            strats.append((f".dm{int(a.demean_window)}", gap - gbar, gap, fair, fx_fair))
        # '+adj' twin: fair shifted by the adj head's predicted forward mid
        # move (logit space). Same rows, same fills, same gate machinery — the
        # ride gate's push now includes the fast forecast. dm variant reuses
        # the BASE gbar (adj mean-reverts in ~10s; its 30s trailing mean is
        # ~0, so gap2 − gbar ≈ demeaned base gap + adj innovation).
        if "ADJ" in d:
            adjv = d["ADJ"][np.searchsorted(d["ts"], ts_p)]
            fc = np.clip(fair, 0.005, 0.995)
            fair2 = 1.0 / (1.0 + np.exp(-(np.log(fc / (1.0 - fc)) + adjv)))
            gap2 = fair2 - mid_p
            strats.append(("+adj", gap2, gap2, fair2, None))
            if a.demean_window > 0:
                strats.append((f"+adj.dm{int(a.demean_window)}", gap2 - gbar, gap2, fair2, None))
        for suffix, sig, g_, f_, fx_ in strats:
            trades += generate_trades(sig, m["label"] + suffix, g_, f_, mid_p,
                                      tte_p, ybid_p, yask_p, ts_p, outc, t, a,
                                      fill_ybid, fill_yask, fx_fair=fx_,
                                      # live parity: the stale veto only guards
                                      # 2-price surfaces (rule.rs two_price())
                                      cb_age_p=cb_age_p if cbm is not None else None)
    return trades, bce_rows


def utc_date(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc).strftime("%Y-%m-%d")


def bucket_of(tte):
    for hi, lo in BUCKETS:
        if lo < tte <= hi:
            return f"{hi}-{lo}s"
    return "other"


# ── reporting ────────────────────────────────────────────────────────────────
def panels(trades, fresh_from):
    if not fresh_from:
        return [("ALL", trades)]
    ins = [t for t in trades if t["date"] < fresh_from]
    fresh = [t for t in trades if t["date"] >= fresh_from]
    return [(f"IN-SAMPLE (< {fresh_from})", ins), (f"FRESH (>= {fresh_from})", fresh)]


def report(trades, bce_rows, a, models):
    # labels = base models, each followed by its demean variants ("<model>.dm<W>")
    # so raw vs demeaned sit adjacent in the return/closure sections.
    seen = {t["model"] for t in trades}
    labels = []
    for m in models:
        labels.append(m["label"])
        labels += sorted(l for l in seen if l.startswith(m["label"] + ".dm"))
        labels += sorted(l for l in seen if l.startswith(m["label"] + "+adj"))

    lat = a.latency_ms > 0
    for tag, T in panels(trades, a.fresh_from):
        # misses (chase-cap blocked signals) under the SAME date split as T
        if not a.fresh_from:
            M = a.misses
        elif tag.startswith("IN-SAMPLE"):
            M = [x for x in a.misses if x["date"] < a.fresh_from]
        else:
            M = [x for x in a.misses if x["date"] >= a.fresh_from]
        print(f"\n{'=' * 100}\n### {tag}   [{len(T)} trades]\n{'=' * 100}")
        if not T:
            continue

        # ── 1. strategy summary ──
        fillhdr = f", {a.latency_ms:.0f}ms latency + {a.chase_c*100:.0f}c chase" if lat else ""
        print(f"\n--- 1. STRATEGY (gate={a.gate}, entry {a.tte_max}-{a.tte_min}s, "
              f"exit=settle, taker{fillhdr}) ---")
        extra = f"{'fill%':>6s} {'slip':>6s} " if lat else ""
        print(f"{'model':12s} {'delta':>6s} {'n':>5s} {'ev':>4s} {extra}{'win%':>6s} "
              f"{'cost':>6s} {'net/tr':>8s} {'total':>8s} {'t':>6s}")
        for lb in labels:
            for dl in a.deltas:
                R = [t for t in T if t["model"] == lb and t["delta"] == dl]
                if not R:
                    continue
                net = np.array([r["net"] for r in R])
                t = clustered_t([r["ticker"] for r in R], net)
                if lat:
                    nm = sum(1 for x in M if x["model"] == lb and x["delta"] == dl)
                    fr = 100 * len(R) / (len(R) + nm)
                    slip = 100 * np.mean([r["slip"] for r in R])
                    extra = f"{fr:>5.0f}% {slip:>+5.1f} "
                else:
                    extra = ""
                print(f"{lb:12s} {dl:>6.3f} {len(R):>5d} "
                      f"{len({r['ticker'] for r in R}):>4d} {extra}"
                      f"{100 * np.mean([r['won'] for r in R]):>5.1f}% "
                      f"{100 * np.mean([r['cost'] for r in R]):>5.1f} "
                      f"{100 * net.mean():>+7.1f}c {net.sum():>+8.2f} {t:>+6.2f}")

        # ── 2. by date ──
        print(f"\n--- 2. BY DATE (net/trade, n) ---")
        dates = sorted({t["date"] for t in T})
        for dl in a.deltas:
            print(f"  delta={dl}")
            for lb in labels:
                cells = []
                for dt in dates:
                    R = [t for t in T if t["model"] == lb and t["delta"] == dl
                         and t["date"] == dt]
                    cells.append(f"{100 * np.mean([r['net'] for r in R]):>+6.1f}"
                                 f"({len(R):>3d})" if R else f"{'--':>11}")
                print(f"    {lb:10s} " + " ".join(
                    f"{dt[5:]}:{c}" for dt, c in zip(dates, cells)))

        # ── 3. by tte bucket ──
        print(f"\n--- 3. BY ENTRY TTE BUCKET ---")
        bnames = [f"{hi}-{lo}s" for hi, lo in BUCKETS if lo >= a.tte_min - 1e-9]
        print(f"{'model':12s} {'delta':>6s} | " + " | ".join(f"{b:^21}" for b in bnames))
        for lb in labels:
            for dl in a.deltas:
                cells = []
                for b in bnames:
                    R = [(t["ticker"], t["net"]) for t in T
                         if t["model"] == lb and t["delta"] == dl and t["bucket"] == b]
                    cells.append(cell(R))
                print(f"{lb:12s} {dl:>6.3f} | " + " | ".join(cells))

        # ── 3b. by entry PRICE bucket (moneyness; * = live ATM band) ──
        print(f"\n--- 3b. BY ENTRY PRICE BUCKET (YES mid at entry; * = live ATM [0.30,0.70]) ---")
        pnames = [f"{lo:.2f}-{hi:.2f}" for lo, hi in PRICE_BUCKETS]
        print(f"{'model':12s} {'delta':>6s} | " + " | ".join(f"{_atm_mark(p) + p:^21}" for p in pnames))
        for lb in labels:
            for dl in a.deltas:
                cells = []
                for p in pnames:
                    R = [(t["ticker"], t["net"]) for t in T
                         if t["model"] == lb and t["delta"] == dl and t["pbucket"] == p]
                    cells.append(cell(R))
                print(f"{lb:12s} {dl:>6.3f} | " + " | ".join(cells))

        # ── 3c. FILL RATE by price bucket (chase cap starves the fastest
        # signals — where the edge is; fill% = filled / (filled+missed)) ──
        if lat:
            print(f"\n--- 3c. FILL RATE by entry PRICE bucket "
                  f"(filled/signals; {a.chase_c*100:.0f}c chase, {a.latency_ms:.0f}ms) ---")
            print(f"{'model':12s} {'delta':>6s} | " + " | ".join(f"{_atm_mark(p) + p:^15}" for p in pnames))
            for lb in labels:
                for dl in a.deltas:
                    cells = []
                    for p in pnames:
                        nf = sum(1 for t in T if t["model"] == lb and t["delta"] == dl and t["pbucket"] == p)
                        nm = sum(1 for x in M if x["model"] == lb and x["delta"] == dl and x["pbucket"] == p)
                        tot = nf + nm
                        cells.append(f"{100*nf/tot:>4.0f}% ({tot:>4d})" if tot else f"{'--':>11}")
                    print(f"{lb:12s} {dl:>6.3f} | " + " | ".join(cells))

    # ── 4. settlement BCE (all rows; not trade-conditional) ──
    print(f"\n{'=' * 100}\n### 4. SETTLEMENT BCE: fair vs MARKET, per tte bucket "
          f"(per-event paired)\n{'=' * 100}")
    print("negative d = model beats the market's own quote at predicting settlement")
    for tag, keep in ([("ALL", None)] if not a.fresh_from else
                      [(f"IN-SAMPLE (< {a.fresh_from})", lambda r: r["date"] < a.fresh_from),
                       (f"FRESH (>= {a.fresh_from})", lambda r: r["date"] >= a.fresh_from)]):
        print(f"\n{tag}")
        print(f"{'model':12s} {'bucket':>10s} {'n_ev':>5s} {'bceF':>7s} {'bceM':>7s} "
              f"{'d':>8s} {'t':>6s} {'brierF':>7s} {'brierM':>7s} {'|f-m| MAE':>9s}")
        for m in models:
            rows_m = [r for r in bce_rows if r["model"] == m["label"]
                      and (keep is None or keep(r))]
            for hi, lo in BUCKETS:
                R = [r for r in rows_m if r["bucket"] == f"{hi}-{lo}s"]
                if len(R) < 3:
                    continue
                d = np.array([r["d"] for r in R])
                print(f"{m['label']:12s} {f'{hi}-{lo}s':>10s} {len(R):>5d} "
                      f"{np.mean([r['bce_fair'] for r in R]):>7.4f} "
                      f"{np.mean([r['bce_mkt'] for r in R]):>7.4f} "
                      f"{d.mean():>+8.4f} {paired_t(d):>+6.2f} "
                      f"{np.mean([r['brier_fair'] for r in R]):>7.4f} "
                      f"{np.mean([r['brier_mkt'] for r in R]):>7.4f} "
                      f"{np.mean([r['mae'] for r in R]):>9.4f}")

    # ── 4b. settlement BCE by PRICE bucket (all rows, per-event paired;
    # delta-independent — measures calibration by moneyness, not the trade set) ──
    print(f"\n{'=' * 100}\n### 4b. SETTLEMENT BCE by ENTRY-PRICE bucket "
          f"(YES mid; * = live ATM [0.30,0.70])\n{'=' * 100}")
    print("negative d = model beats the market's own quote at predicting settlement")
    for tag, keep in ([("ALL", None)] if not a.fresh_from else
                      [(f"IN-SAMPLE (< {a.fresh_from})", lambda r: r["date"] < a.fresh_from),
                       (f"FRESH (>= {a.fresh_from})", lambda r: r["date"] >= a.fresh_from)]):
        print(f"\n{tag}")
        print(f"{'model':12s} {'price bkt':>12s} {'n_ev':>5s} {'bceF':>7s} {'bceM':>7s} "
              f"{'d':>8s} {'t':>6s} {'|f-m| MAE':>9s}")
        for m in models:
            rows_m = [r for r in bce_rows if r["model"] == m["label"]
                      and (keep is None or keep(r))]
            for lo, hi in PRICE_BUCKETS:
                lab = f"{lo:.2f}-{hi:.2f}"
                R = [r for r in rows_m if r["bucket"] == lab]
                if len(R) < 3:
                    continue
                d = np.array([r["d"] for r in R])
                print(f"{m['label']:12s} {_atm_mark(lab) + lab:>12s} {len(R):>5d} "
                      f"{np.mean([r['bce_fair'] for r in R]):>7.4f} "
                      f"{np.mean([r['bce_mkt'] for r in R]):>7.4f} "
                      f"{d.mean():>+8.4f} {paired_t(d):>+6.2f} "
                      f"{np.mean([r['mae'] for r in R]):>9.4f}")

    # ── 5. closure: WHO MOVED. Headline first (see memory: closure-analysis-
    # format) — the two signed mean moves, then the date x model breakdown of
    # the same pair. The normalized shares are bimodal (a third of trades
    # overshoot, a third go negative), so their medians mislead; the raw signed
    # moves obey  mkt_move - fair_move = gap_at_entry,  which IS gap closure.
    C_all = [t for t in trades if not math.isnan(t["close_s"])]

    def moves(R):
        """-> (mean market move, mean fair move, mean entry gap) in CENTS,
        signed by trade direction (+ = moved the way the trade bets)."""
        if not R:
            return float("nan"), float("nan"), float("nan")
        s = np.array([1.0 if r["side_yes"] else -1.0 for r in R])
        mk = 100 * s * np.array([r["close_dmid"] for r in R])
        fr = 100 * s * np.array([r["close_dfair"] for r in R])
        g0 = 100 * np.array([r["gap"] for r in R])
        return mk.mean(), fr.mean(), g0.mean()

    def fair_split(R):
        """Split the mean fair move into (inputs-only, refit-absorbed) cents,
        over the subset with a frozen-params recompute available."""
        Rf = [r for r in R if not math.isnan(r.get("close_dfair_fx", float("nan")))]
        if not Rf:
            return float("nan"), float("nan"), 0
        s = np.array([1.0 if r["side_yes"] else -1.0 for r in Rf])
        fx = 100 * s * np.array([r["close_dfair_fx"] for r in Rf])
        rf = 100 * s * np.array([r["close_dfair"] - r["close_dfair_fx"] for r in Rf])
        return fx.mean(), rf.mean(), len(Rf)

    print(f"\n{'=' * 100}\n### 5. CLOSURE: who moved, entry -> first |fair-mid| < "
          f"{100 * a.close_eps:.0f}c\n{'=' * 100}")
    mk, fr, g0 = moves(C_all)
    print(f"  n = {len(C_all)} closed trades of {len(trades)} "
          f"({100 * len(C_all) / max(1, len(trades)):.0f}% close before settle)")
    print(f"  mean gap at entry : {g0:+6.2f}c")
    print()
    print(f"  mean MARKET move  : {mk:+6.2f}c   (toward the trade)")
    print(f"  mean FAIR   move  : {fr:+6.2f}c   (away from the trade)")
    fx_mv, rf_mv, n_fx = fair_split(C_all)
    if n_fx:
        print(f"      of which  inputs-only (frozen calib): {fx_mv:+6.2f}c"
              f"   refit-absorbed: {rf_mv:+6.2f}c   (n={n_fx})")
    print()
    print(f"  -> the market did {100 * mk / (abs(mk) + abs(fr) + 1e-9):.0f}% of the closing, "
          f"the model capitulated the rest.  (identity: mkt - fair = gap)")
    if n_fx:
        print(f"  -> refit-excluded: market {mk:+.2f}c vs model-opinion {fx_mv:+.2f}c "
              f"(the refit-absorbed {rf_mv:+.2f}c is the calibrator being pulled to the "
              f"mid at each refit, not the model changing its mind)")

    if a.close_cap > 0:
        Rc = [t for t in trades if not math.isnan(t.get("cap_s", float("nan")))]
        if Rc:
            s = np.array([1.0 if r["side_yes"] else -1.0 for r in Rc])
            mkc = (100 * s * np.array([r["cap_dmid"] for r in Rc])).mean()
            frc = (100 * s * np.array([r["cap_dfair"] for r in Rc])).mean()
            Rf = [r for r in Rc if not math.isnan(r["cap_dfair_fx"])]
            sf = np.array([1.0 if r["side_yes"] else -1.0 for r in Rf])
            fxc = (100 * sf * np.array([r["cap_dfair_fx"] for r in Rf])).mean() if Rf else float("nan")
            rfc = (100 * sf * np.array([r["cap_dfair"] - r["cap_dfair_fx"] for r in Rf])).mean() if Rf else float("nan")
            g0c = (100 * np.array([r["gap"] for r in Rc])).mean()
            resid = g0c - (mkc - frc)
            print(f"\n--- 5c. CAPPED @ {a.close_cap:.0f}s: moves at min(closure, cap); "
                  f"includes never-closed (n={len(Rc)} of {len(trades)}) ---")
            print(f"  gap0={g0c:+.2f}c   MARKET {mkc:+.2f}c   FAIR {frc:+.2f}c "
                  f"(inputs-only {fxc:+.2f}c, refit {rfc:+.2f}c)   residual gap at horizon {resid:+.2f}c")

    print(f"\n--- 5b. same two moves, BY ENTRY PRICE BUCKET  (* = live ATM [0.30,0.70]) ---")
    for lo, hi in PRICE_BUCKETS:
        lab = f"{lo:.2f}-{hi:.2f}"
        Rp = [t for t in C_all if t.get("pbucket") == lab]
        if len(Rp) < 5:
            continue
        mkp, frp, g0p = moves(Rp)
        print(f"  {_atm_mark(lab)}{lab}  n={len(Rp):>4d}  gap0={g0p:+5.1f}c  "
              f"mkt={mkp:+5.2f}c  fair={frp:+5.2f}c  mkt_share={100 * mkp / (abs(mkp) + abs(frp) + 1e-9):>3.0f}%")

    print(f"\n--- 5a. same two numbers, BY DATE x MODEL  (m = market, f = fair, cents) ---")
    dates = sorted({t["date"] for t in trades})
    for m in models:
        lb = m["label"]
        cells = []
        for dt in dates:
            R = [t for t in C_all if t["model"] == lb and t["date"] == dt]
            if not R:
                cells.append(f"{'--':>18}")
                continue
            mk, fr, _ = moves(R)
            cells.append(f"m{mk:>+5.1f} f{fr:>+5.1f} n{len(R):>3d}")
        print(f"  {lb:10s} " + "  ".join(f"{dt[5:]} {c}" for dt, c in zip(dates, cells)))

    print(f"\n--- 5b. per model x delta (+ closure timing) ---")
    for tag, T in panels(trades, a.fresh_from):
        print(f"\n{tag}")
        print(f"{'model':12s} {'delta':>6s} {'n':>5s} {'closed%':>8s} {'med_s':>6s} "
              f"{'gap0':>6s} {'mkt_mv':>7s} {'fair_mv':>8s} {'fx_mv':>7s} {'refit':>7s} "
              f"| {'net|WON':>9s} {'net|LOST':>9s}")
        for lb in labels:
            for dl in a.deltas:
                R = [t for t in T if t["model"] == lb and t["delta"] == dl]
                C = [r for r in R if not math.isnan(r["close_s"])]
                if not R or not C:
                    continue
                mk, fr, g0 = moves(C)
                fx_mv, rf_mv, _ = fair_split(C)
                w = [r["net"] for r in C if r["won"]]
                l = [r["net"] for r in C if not r["won"]]
                print(f"{lb:12s} {dl:>6.3f} {len(R):>5d} "
                      f"{100 * len(C) / len(R):>7.0f}% "
                      f"{np.median([r['close_s'] for r in C]):>6.1f} {g0:>6.2f} "
                      f"{mk:>+7.2f} {fr:>+8.2f} {fx_mv:>+7.2f} {rf_mv:>+7.2f} | "
                      f"{100 * np.mean(w) if w else float('nan'):>+8.1f}c "
                      f"{100 * np.mean(l) if l else float('nan'):>+8.1f}c")
    print("\n(a real convergence edge would show market >> |fair|: the market comes to a")
    print(" fair that stays put. An even split means the gap closes half because we were")
    print(" wrong -- the gap is a trigger, not a mispricing.)")


# ── main ─────────────────────────────────────────────────────────────────────
def sharpe_report(trades, a):
    """Daily-return Sharpe per model x delta (daily return = sum(net)/sum(cost)
    that day = return on capital deployed; annualized x sqrt(365), 24/7 market).
    Splits in-sample/fresh when --fresh-from is set."""
    print(f"\n{'=' * 100}\n### SHARPE (daily return = day sum(net)/sum(cost); annualized x sqrt(365))\n{'=' * 100}")
    for tag, T in panels(trades, a.fresh_from):
        if not T:
            continue
        print(f"\n{tag}  [{len(T)} trades]")
        print(f"{'model':14s} {'delta':>5s} {'n':>5s} {'days':>4s} {'mean/day':>9s} "
              f"{'std/day':>8s} {'Sharpe/d':>9s} {'Sharpe/yr':>9s} {'tot_ret':>8s}")
        for lb in sorted({t['model'] for t in T}):
            for dl in a.deltas:
                R = [t for t in T if t['model'] == lb and t['delta'] == dl]
                if not R:
                    continue
                byday = defaultdict(lambda: [0.0, 0.0])  # [net, cost]
                for t in R:
                    byday[t['date']][0] += t['net']; byday[t['date']][1] += t['cost']
                dret = np.array([n / c for n, c in byday.values() if c > 0])
                if len(dret) < 2:
                    continue
                sd = dret.std(ddof=1)
                shd = dret.mean() / sd if sd > 0 else float('nan')
                tot = sum(n for n, _ in byday.values()) / sum(c for _, c in byday.values())
                print(f"{lb:14s} {dl:>5.2f} {len(R):>5d} {len(dret):>4d} {100 * dret.mean():>+8.2f}% "
                      f"{100 * sd:>7.2f}% {shd:>+9.2f} {shd * math.sqrt(365):>+9.1f} {100 * tot:>+7.1f}%")


def main():
    p = argparse.ArgumentParser(
        description="Standing analysis harness for online 50ms sampler data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--samples", required=True,
                   help="comma-separated sampler CSV(.gz) paths")
    p.add_argument("--model", action="append", required=True, metavar="label=path[:px=cb|perp]",
                   help="repeatable; price input defaults from surface train.venue")
    p.add_argument("--deltas", default="0.03,0.05", help="entry |gap| thresholds")
    p.add_argument("--tte", default="300:60", metavar="MAX:MIN",
                   help="entry window in seconds to expiry")
    p.add_argument("--fit-window", type=float, default=120.0,
                   help="rolling (db,dr) fit window in s (0 = expanding)")
    p.add_argument("--fit-sample-ms", type=float, default=1000.0,
                   help="calibration fit sample resolution (ms); live=1000. "
                        "converted to a scan-grid stride internally (50ms grid)")
    p.add_argument("--refit-step", type=float, default=15.0,
                   help="refit frequency in s (default 15 = the LIVE calibrator's "
                        "cadence, 2026-08-10; was 60 — 4x fewer fits, faster runs, "
                        "but a different (db,dr) path that misses live's marginal "
                        "crossings; see the Aug-8/9 loss debug)")
    p.add_argument("--demean-window", type=float, default=0.0,
                   help="if >0, entry fires on gap MINUS its trailing mean over "
                        "this window (s), the mean recomputed with the current "
                        "params at 1s resolution (fire on deviations from the "
                        "standing bias, not the bias itself; 0 = off/raw gap)")
    p.add_argument("--gate", choices=("ride", "fade", "none", "overreact"), default="ride",
                   help="ride = only gaps opened by the model (share>ride-share); "
                        "overreact = fade a market overshoot (gap flipped past fair, "
                        "mid moved >> fair, same direction)")
    p.add_argument("--ride-share", type=float, default=0.75)
    p.add_argument("--lookback-s", type=float, default=1.0,
                   help="ride-gate lookback (s): reference = youngest sample at "
                        "least this old (live lookback_min_ms/1000; default 1.0)")
    p.add_argument("--ride-open", type=float, default=None,
                   help="ride-gate open_min; DEFAULT = the gap threshold (delta), "
                        "per the 2026-08-03 convention (live open_min: 0.02 = delta)")
    p.add_argument("--overreact-k", type=float, default=2.0,
                   help="overreact gate: mid must have moved >= this x |fair move|")
    p.add_argument("--overreact-flip", type=float, default=0.01,
                   help="overreact gate: the gap must have been >= this on the OTHER side 1s ago")
    p.add_argument("--rearm-eps", type=float, default=0.02,
                   help="gap must fall below this to re-arm (one trade/episode)")
    p.add_argument("--trigger", choices=("gap", "move"), default="gap",
                   help="gap = level trigger |fair-mid| >= delta (default). "
                        "move = 1s-DYNAMICS trigger: fire when |dfair_1s| >= delta "
                        "AND |dfair_1s| >= move-k * |dmid_1s| (fair moved, market "
                        "hasn't followed), direction = sign(dfair_1s); no level-gap "
                        "requirement, --gate ignored")
    p.add_argument("--move-k", type=float, default=2.0,
                   help="move trigger: fair-vs-mid 1s move dominance ratio")
    p.add_argument("--move-rearm", type=float, default=0.5,
                   help="move trigger: re-arm when |dfair_1s| <= delta * this")
    p.add_argument("--calib-from-live", default="", metavar="PATH",
                   help="consistency mode: read the LIVE calibrator's fit: records "
                        "(trader-events.log extract) and use its (db,dr) per boundary "
                        "instead of refitting offline; events live never calibrated "
                        "are skipped")
    p.add_argument("--close-cap", type=float, default=0.0, metavar="SECONDS",
                   help="also measure fair/mid moves at min(closure, cap) — a fixed "
                        "horizon uncorrelated with the endogenous closure time; "
                        "includes never-closed trades. 0 = off")
    p.add_argument("--cb-stale-ms", type=float, default=5000.0,
                   help="max age of the cb quote a fair may consume; staler rows "
                        "are excluded for 2-price models (live CB_STALE_NS parity = 5000)")
    p.add_argument("--stale-veto-ms", type=float, default=100.0,
                   help="trade-only veto: suppress the ENTRY (crossing still "
                        "disarms) when the fair's cb quote is older than this; "
                        "0=off. Default 100 = LIVE parity (stale_veto_ms: 100, "
                        "2026-08-03); applies to 2-price models only")
    p.add_argument("--disarm-cd-s", type=float, default=0.0,
                   help="trade-only veto: no entry within N s of the previous "
                        "delta-crossing (any direction, traded or not); 0=off")
    p.add_argument("--close-eps", type=float, default=0.01,
                   help="closure threshold for analysis 5")
    p.add_argument("--latency-ms", type=float, default=0.0,
                   help="order round-trip latency: the fill executes against the "
                        "book this many ms AFTER the signal row (market drifts "
                        "meanwhile). 0 = instant fill at the signal instant "
                        "(legacy). Live executor is ~100.")
    p.add_argument("--chase-c", type=float, default=0.01,
                   help="max chase (dollars): with --latency-ms>0, MISS the fill "
                        "if the post-latency traded-side ask has run more than "
                        "this above the signal-instant ask (order can't catch a "
                        "fast market). Default 0.01 = the live 1c chase.")
    p.add_argument("--cap", type=int, default=0,
                   help="max entries per event per delta (0 = uncapped, the "
                        "default; pass e.g. 3 to bound the per-event tail, "
                        "which is ~6x the median event). NOTE: uncapped inflates "
                        "raw P&L/Sharpe via stacked non-independent bets in "
                        "trending events -- read the EVENT-CLUSTERED t, which "
                        "discounts the stacking, as the honest metric.")
    p.add_argument("--fresh-from", default="", metavar="YYYY-MM-DD",
                   help="split every table in-sample vs fresh at this UTC date")
    p.add_argument("--meta-cache", default="",
                   help="Kalshi meta cache json (default: beside the samples)")
    p.add_argument("--adj-overlay", default="", metavar="PATH",
                   help="adj-head JSON (kind=adj): every surface model also "
                        "runs a '+adj' twin with fair2 = sigmoid(logit(fair) + "
                        "w.z + b) — the fast mid-forecast stacked on the "
                        "calibrated fair; same rows, fills and gate machinery")
    p.add_argument("--out-dir", default="", help="write trades/bce/summary CSVs here")
    p.add_argument("--dump-fair", default="", metavar="PATH",
                   help="also write the causal fair series (ticker,ts_ms,tte_s,"
                        "fair,mid) here — for replay-parity checks")
    a = p.parse_args()

    # fit sample resolution -> scan-grid stride (grid = 50ms * SCAN_STRIDE)
    a.fit_stride = max(1, round(a.fit_sample_ms / (50.0 * SCAN_STRIDE)))

    a.live_calib = None
    if a.calib_from_live:
        a.live_calib = load_live_calib(a.calib_from_live)
        print(f"live-calib mode: {len(a.live_calib)} events with logged fits "
              f"from {a.calib_from_live}")

    a.fair_rows = []
    a.gap_dist = {}
    a.event_calib = {}
    a.win_signed = {}
    a.misses = []   # signals blocked by the chase cap (post-latency ask ran away)
    a.vetoes = []   # trade-only vetoes (stale-signal / disarm-cooldown)
    a.deltas = [float(x) for x in a.deltas.split(",")]
    a.tte_max, a.tte_min = (float(x) for x in a.tte.split(":"))
    if a.tte_max > 300:
        BUCKETS[:0] = [b for b in BUCKETS_EARLY if b[1] < a.tte_max]
    paths = a.samples.split(",")
    models = [parse_model(s) for s in a.model]
    if a.adj_overlay:
        adj_js = json.load(open(a.adj_overlay))
        assert adj_js.get("kind") == "adj", "--adj-overlay expects an adj-head JSON"
        for m in models:
            if m["js"].get("kind") != "adj":       # surfaces only
                m["adj_js"] = adj_js
        print(f"adj overlay: {a.adj_overlay} (h={adj_js.get('h')}) -> "
              f"'+adj' twin for every surface model")

    print(f"loading {len(paths)} sample file(s)...")
    ev = {}
    for pth in paths:
        for t, d in sim.load_samples(pth, cb_stale_ms=a.cb_stale_ms).items():
            if t in ev:  # same event spanning a file boundary: concatenate,
                ev[t] = {k: np.concatenate([ev[t][k], d[k]]) for k in d}  # then re-sort
            else:
                ev[t] = d
    for t, d in ev.items():
        o = np.argsort(d["ts"])
        for k in d:
            d[k] = d[k][o]
    cache = a.meta_cache or os.path.join(os.path.dirname(paths[0]), "meta_cache.json")
    meta = sim.fetch_meta(sorted(ev), cache)
    print(f"events: {len(ev)}  (meta cache: {cache})")

    trades, bce_rows = [], []
    for m in models:
        evm = prepare(ev, m)
        tr, bc = simulate(m, evm, meta, a)
        for r in bc:
            r["model"] = m["label"]
        trades += tr
        bce_rows += bc
        print(f"  [{m['label']}] {len(tr)} trades, {len(bc)} event-bucket BCE rows")

    if a.vetoes:
        cnt = Counter((v["model"], v["delta"], v["why"]) for v in a.vetoes)
        print("  trade-only vetoes (crossing still disarmed):")
        for (ml, dl, why), n in sorted(cnt.items()):
            print(f"    {ml:14s} d={dl:g} {why:9s} {n}")

    report(trades, bce_rows, a, models)
    sharpe_report(trades, a)
    gap_report(a, models)
    event_calib_report(a, models)

    if a.dump_fair and a.fair_rows:
        with open(a.dump_fair, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ticker", "ts_ms", "tte_s", "fair", "mid"])
            for r in a.fair_rows:
                w.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.6f}", f"{r[4]:.6f}"])
        print(f"\n-> {a.dump_fair} ({len(a.fair_rows)} fair rows)")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for name, rows in (("trades", trades), ("bce", bce_rows)):
            if not rows:
                continue
            fp = os.path.join(a.out_dir, f"{name}.csv")
            with open(fp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\n-> {fp} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
