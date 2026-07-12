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
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_50ms as sim  # noqa: E402  (fit/forward math: one source of truth)

BUCKETS = [(300, 240), (240, 180), (180, 120), (120, 60), (60, 0)]
SCAN_STRIDE = 5   # 250ms trigger grid
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
    return {"label": label, "path": path, "js": js, "px": px,
            "extras": js.get("extras", [])}


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
        ts = d["ts"]

        def lag(series, k_s):
            idx = np.searchsorted(ts, ts - k_s * 1000.0, side="right") - 1
            ok = idx >= 0
            ok[ok] &= (ts[ok] - ts[idx[ok]]) <= (k_s * 1000.0 + 2000.0)
            col = np.full(len(ts), np.nan)
            col[ok] = series[idx[ok]]
            return col

        if extras:
            needs_cb = any(c.startswith(("basis", "dbasis")) for c in extras)
            bas = d["cbmid"] - d["spot"] if needs_cb else None
            X = np.full((len(ts), len(extras)), np.nan)
            for ci, name in enumerate(extras):
                if name.startswith("mom"):
                    X[:, ci] = d["spot"] - lag(d["spot"], int(name[3:]))
                elif name == "basis":
                    X[:, ci] = bas
                elif name.startswith("dbasis"):
                    X[:, ci] = bas - lag(bas, int(name[6:]))
                elif name == "imb1":
                    X[:, ci] = d["imb1n"]
                else:
                    sys.exit(f"{m['label']}: cannot reconstruct feature {name!r} "
                             f"from sampler columns")
            d["X"] = np.clip((X - mu) / sd, -5, 5)
        else:
            d["X"] = np.zeros((len(ts), 0))

        ok = ~np.isnan(d["spot"])
        if extras:
            ok &= ~np.isnan(d["X"]).any(axis=1)
        for k in list(d):
            d[k] = d[k][ok]
        dropped += int((~ok).sum())
        kept += int(ok.sum())
        out[t] = d
    print(f"  [{m['label']}] px={m['px']} extras={extras or '-'}: "
          f"kept {kept:,} rows, dropped {dropped:,}")
    return out


def fair_series(d, m, strike, rho, b_scale, fwd, tte_max, fit_window):
    """Windowed expanding refit at each 60s boundary; returns the causal fair
    series on the scan grid (the live Calibrator's semantics)."""
    fit_sl = slice(None, None, FIT_STRIDE)
    scan = d["tte"] <= tte_max
    ss = slice(None, None, SCAN_STRIDE)
    spot_p, tte_p = d["spot"][scan][ss], d["tte"][scan][ss]
    ybid_p, yask_p = d["ybid"][scan][ss], d["yask"][scan][ss]
    ts_p, x_p = d["ts"][scan][ss], d["X"][scan][ss]
    fair = np.full(len(tte_p), np.nan)
    init = None
    for B in range(int(tte_max), 0, -60):
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
        )
        init = (float(db), float(dr))
        seg = (tte_p <= B) & (tte_p > B - 60)
        if seg.any():
            with torch.no_grad():
                lo = sim.logit_of(
                    fwd,
                    torch.tensor(spot_p[seg], dtype=torch.float32),
                    torch.tensor(tte_p[seg], dtype=torch.float32),
                    strike + b_scale * db, torch.exp(rho + dr),
                    torch.tensor(x_p[seg], dtype=torch.float32),
                )
                fair[seg] = torch.sigmoid(lo).numpy()
    ok = ~np.isnan(fair)
    return (ts_p[ok], tte_p[ok], ybid_p[ok], yask_p[ok], fair[ok])


# ── trade generation ─────────────────────────────────────────────────────────
def simulate(m, ev, meta, a):
    """-> (trades, bce_rows). One pass per event; all deltas scored together."""
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
        ts_p, tte_p, ybid_p, yask_p, fair = fair_series(
            d, m, strike, rho, b_scale, fwd, a.tte_max, a.fit_window)
        if len(tte_p) < 50:
            continue
        mid_p = (ybid_p + yask_p) / 2.0
        gap = fair - mid_p

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

        for dl in a.deltas:
            armed, run, k, n_ent = True, 0, 0, 0
            while k < len(gap):
                ag = abs(gap[k])
                if not armed:
                    if ag <= a.rearm_eps:
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
                side_yes = gap[k] > 0
                s = 1.0 if side_yes else -1.0

                # 1s-lookback trigger attribution (the RIDE gate): did the gap
                # open because the MODEL moved (ride a real reprice) or because
                # the MARKET moved away (fade informed flow -> adverse select)?
                j = k - 1
                while j > 0 and (tte_p[j] - tte_p[k]) < 1.0 and (k - j) < 40:
                    j -= 1
                if j >= 0 and 1.0 <= (tte_p[j] - tte_p[k]) <= 10.0:
                    dfair1, dmid1 = fair[k] - fair[j], mid_p[k] - mid_p[j]
                else:
                    dfair1 = dmid1 = float("nan")
                push, pull = s * dfair1, -s * dmid1
                tot = push + pull
                share = push / tot if tot and not math.isnan(tot) else float("nan")
                is_ride = bool(tot > a.ride_open and share > a.ride_share)
                if a.gate == "ride" and not is_ride:
                    k += 1
                    continue
                if a.gate == "fade" and is_ride:
                    k += 1
                    continue

                p_entry = yask_p[k] if side_yes else 1.0 - ybid_p[k]
                if p_entry >= 0.99:
                    k += 1
                    continue
                cost = p_entry + sim.fee(p_entry)
                won = int(outc == 1 if side_yes else outc == 0)

                # ── (5) closure: first |fair-mid| < eps after entry, and who
                # traveled. gap0 = fair0-mid0; closing requires dmid-dfair=gap0,
                # so market_share = dmid/gap0 and fair_share = -dfair/gap0 sum
                # to 1. market_share ~1 => the market came to us (the thesis);
                # ~0 => our fair retreated to the market (the model was wrong).
                close_s = close_df = close_dm = mkt_sh = fair_sh = float("nan")
                for jj in range(k + 1, len(tte_p)):
                    if abs(gap[jj]) < a.close_eps:
                        close_s = float(tte_p[k] - tte_p[jj])
                        close_df = float(fair[jj] - fair[k])
                        close_dm = float(mid_p[jj] - mid_p[k])
                        g0 = gap[k]
                        if abs(g0) > 1e-9:
                            mkt_sh = close_dm / g0
                            fair_sh = -close_df / g0
                        break

                trades.append({
                    "model": m["label"], "delta": dl, "ticker": t,
                    "date": utc_date(ts_p[k]), "ts_ms": int(ts_p[k]),
                    "tte": float(tte_p[k]), "bucket": bucket_of(tte_p[k]),
                    "side_yes": int(side_yes), "gap": float(abs(gap[k])),
                    "fair0": float(fair[k]), "mid0": float(mid_p[k]),
                    "cost": float(cost), "won": won, "net": float(won - cost),
                    "dfair1s": dfair1, "dmid1s": dmid1, "ride_share": share,
                    "is_ride": int(is_ride), "close_s": close_s,
                    "close_dfair": close_df, "close_dmid": close_dm,
                    "mkt_share": mkt_sh, "fair_share": fair_sh,
                })
                k += 1
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
    labels = [m["label"] for m in models]

    for tag, T in panels(trades, a.fresh_from):
        print(f"\n{'=' * 100}\n### {tag}   [{len(T)} trades]\n{'=' * 100}")
        if not T:
            continue

        # ── 1. strategy summary ──
        print(f"\n--- 1. STRATEGY (gate={a.gate}, entry {a.tte_max}-{a.tte_min}s, "
              f"exit=settle, taker) ---")
        print(f"{'model':12s} {'delta':>6s} {'n':>5s} {'ev':>4s} {'win%':>6s} "
              f"{'cost':>6s} {'net/tr':>8s} {'total':>8s} {'t':>6s}")
        for lb in labels:
            for dl in a.deltas:
                R = [t for t in T if t["model"] == lb and t["delta"] == dl]
                if not R:
                    continue
                net = np.array([r["net"] for r in R])
                t = clustered_t([r["ticker"] for r in R], net)
                print(f"{lb:12s} {dl:>6.3f} {len(R):>5d} "
                      f"{len({r['ticker'] for r in R}):>4d} "
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

    print(f"\n{'=' * 100}\n### 5. CLOSURE: who moved, entry -> first |fair-mid| < "
          f"{100 * a.close_eps:.0f}c\n{'=' * 100}")
    mk, fr, g0 = moves(C_all)
    print(f"  n = {len(C_all)} closed trades of {len(trades)} "
          f"({100 * len(C_all) / max(1, len(trades)):.0f}% close before settle)")
    print(f"  mean gap at entry : {g0:+6.2f}c")
    print()
    print(f"  mean MARKET move  : {mk:+6.2f}c   (toward the trade)")
    print(f"  mean FAIR   move  : {fr:+6.2f}c   (away from the trade)")
    print()
    print(f"  -> the market did {100 * mk / (abs(mk) + abs(fr) + 1e-9):.0f}% of the closing, "
          f"the model capitulated the rest.  (identity: mkt - fair = gap)")

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
              f"{'gap0':>6s} {'mkt_mv':>7s} {'fair_mv':>8s} | {'net|WON':>9s} {'net|LOST':>9s}")
        for lb in labels:
            for dl in a.deltas:
                R = [t for t in T if t["model"] == lb and t["delta"] == dl]
                C = [r for r in R if not math.isnan(r["close_s"])]
                if not R or not C:
                    continue
                mk, fr, g0 = moves(C)
                w = [r["net"] for r in C if r["won"]]
                l = [r["net"] for r in C if not r["won"]]
                print(f"{lb:12s} {dl:>6.3f} {len(R):>5d} "
                      f"{100 * len(C) / len(R):>7.0f}% "
                      f"{np.median([r['close_s'] for r in C]):>6.1f} {g0:>6.2f} "
                      f"{mk:>+7.2f} {fr:>+8.2f} | "
                      f"{100 * np.mean(w) if w else float('nan'):>+8.1f}c "
                      f"{100 * np.mean(l) if l else float('nan'):>+8.1f}c")
    print("\n(a real convergence edge would show market >> |fair|: the market comes to a")
    print(" fair that stays put. An even split means the gap closes half because we were")
    print(" wrong -- the gap is a trigger, not a mispricing.)")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Standing analysis harness for online 50ms sampler data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--samples", required=True,
                   help="comma-separated sampler CSV(.gz) paths")
    p.add_argument("--model", action="append", required=True, metavar="label=path[:px=cb|perp]",
                   help="repeatable; price input defaults from surface train.venue")
    p.add_argument("--deltas", default="0.03,0.05", help="entry |gap| thresholds")
    p.add_argument("--tte", default="300:120", metavar="MAX:MIN",
                   help="entry window in seconds to expiry")
    p.add_argument("--fit-window", type=float, default=120.0,
                   help="rolling (db,dr) fit window in s (0 = expanding)")
    p.add_argument("--gate", choices=("ride", "fade", "none"), default="ride",
                   help="ride = only gaps opened by the model (share>ride-share)")
    p.add_argument("--ride-share", type=float, default=0.75)
    p.add_argument("--ride-open", type=float, default=0.005)
    p.add_argument("--rearm-eps", type=float, default=0.02,
                   help="gap must fall below this to re-arm (one trade/episode)")
    p.add_argument("--close-eps", type=float, default=0.01,
                   help="closure threshold for analysis 5")
    p.add_argument("--cap", type=int, default=3,
                   help="max entries per event per delta (0 = uncapped; the "
                        "uncapped tail is ~6x the median event)")
    p.add_argument("--fresh-from", default="", metavar="YYYY-MM-DD",
                   help="split every table in-sample vs fresh at this UTC date")
    p.add_argument("--meta-cache", default="",
                   help="Kalshi meta cache json (default: beside the samples)")
    p.add_argument("--out-dir", default="", help="write trades/bce/summary CSVs here")
    a = p.parse_args()

    a.deltas = [float(x) for x in a.deltas.split(",")]
    a.tte_max, a.tte_min = (float(x) for x in a.tte.split(":"))
    paths = a.samples.split(",")
    models = [parse_model(s) for s in a.model]

    print(f"loading {len(paths)} sample file(s)...")
    ev = {}
    for pth in paths:
        for t, d in sim.load_samples(pth).items():
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

    report(trades, bce_rows, a, models)

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
