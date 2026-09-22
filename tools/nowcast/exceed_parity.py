"""Live-vs-offline parity for the exceed rule (DESIGN_LIVE_SIM_CONSISTENCY level 2 for kind=exceed).
Joins the trader's `exceedfeat` rows (43-vector every 10 s per market) and `exceed` state rows (score every 1 s) to
the harness dump (analyze_online.py --dump-exceed, every valid 50 ms scan row) by ticker + nearest ts (<= tol ms),
and compares: per-feature |diff| (mean/p95/max, and the share within tolerance), the score s and p_up/p_dn, and the
fired signals (live `signal` json vs harness trades.csv on the same period). Feature names follow the trainer's order.
Usage: exceed_parity.py LIVE_LOG DUMP.npz [trades.csv] [--tol-ms 60] [--from HH:MM] [--to HH:MM]"""
import argparse
import json
import re

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("live"); ap.add_argument("dump"); ap.add_argument("trades", nargs="?", default="")
ap.add_argument("--tol-ms", type=int, default=60); ap.add_argument("--from", dest="t0", default=""); ap.add_argument("--to", dest="t1", default="")
a = ap.parse_args()
NAMES = (["mid_lag%d" % j for j in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)] + ["perp_lag%d" % j for j in (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)]
         + ["kbs", "kas", "kmi", "pbs"]
         + ["flow_tfi_1s", "flow_vol_1s", "flow_tfi_5s", "flow_tfi_30s", "flow_n_1s", "flow_n_5s", "depth_spread_bps", "depth_depth10",
            "depth_band5", "depth_band10", "depth_band25", "depth_dimb5_1s", "depth_dimb5_5s", "depth_dband10_1s"]
         + ["log_tte", "sigma", "spread_c", "l0", "mid_var"])

# ── live rows ──
feat, state, sigs = [], [], []
rx_feat = re.compile(r"exceedfeat: (\S+) ts=(\d+) tte=([\d.]+) x=(\S+)")
rx_state = re.compile(r"INFO exceed: (kalshi\S+) ts=(\d+) tte=([\d.]+) mid=([\d.]+) s=([+-][\d.]+) pup=([\d.]+) pdn=([\d.]+) armed=(\d)")
for line in open(a.live, encoding="utf-8", errors="replace"):
    m = rx_feat.search(line)
    if m:
        feat.append((m.group(1), int(m.group(2)), float(m.group(3)), np.array([float(v) for v in m.group(4).split(",")]))); continue
    m = rx_state.search(line)
    if m:
        state.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)), float(m.group(7)), int(m.group(8)))); continue
    if '"strategy":"exceed"' in line:
        js = json.loads(line[line.index("{"):])
        r = dict(re.findall(r"(\w+)=([+-]?[\d.]+)", js["reason"]))
        sigs.append((js["target"], js["ts_ns"] // 1_000_000, float(r["s"]), float(r["mid"]), float(r["tte"]), js["direction"]))
print("live: %d feature rows, %d state rows, %d signals" % (len(feat), len(state), len(sigs)))

# ── harness dump ──
z = np.load(a.dump, allow_pickle=True)
H = pd.DataFrame({"ticker": z["ticker"].astype(str), "ts": z["ts"], "tte": z["tte"], "mid": z["mid"], "p_up": z["p_up"], "p_dn": z["p_dn"]})
HX = z["x"]; H["s"] = H.p_up - H.p_dn
H["ticker"] = H.ticker.str.replace("^kalshi\\.", "", regex=True).str.replace("\\.YES$", "", regex=True)
print("harness: %d rows, %d markets, ts %s .. %s" % (len(H), H.ticker.nunique(), pd.to_datetime(H.ts.min(), unit="ms"), pd.to_datetime(H.ts.max(), unit="ms")))


def norm(t):
    return re.sub(r"^kalshi\.", "", re.sub(r"\.YES$", "", t))


def window(ts):
    d = pd.to_datetime(ts, unit="ms")
    ok = np.ones(len(d), bool)
    if a.t0:
        ok &= d >= pd.Timestamp(d.min().strftime("%Y-%m-%d ") + a.t0)
    if a.t1:
        ok &= d <= pd.Timestamp(d.min().strftime("%Y-%m-%d ") + a.t1)
    return ok


def join(rows, key_ts=1, key_tk=0):
    """nearest harness row per live row (same ticker, |dt| <= tol)"""
    out = []
    by = {t: g.sort_values("ts") for t, g in H.groupby("ticker")}
    for r in rows:
        g = by.get(norm(r[key_tk]))
        if g is None:
            out.append(None); continue
        ts = g.ts.to_numpy(); i = np.searchsorted(ts, r[key_ts])
        cand = [k for k in (i - 1, i) if 0 <= k < len(ts)]
        k = min(cand, key=lambda k: abs(int(ts[k]) - r[key_ts])) if cand else None
        out.append(g.index[k] if k is not None and abs(int(ts[k]) - r[key_ts]) <= a.tol_ms else None)
    return out

# ── features ──
if feat:
    ok = window(np.array([r[1] for r in feat]))
    feat = [r for r, o in zip(feat, ok) if o]
    idx = join(feat)
    pairs = [(r, i) for r, i in zip(feat, idx) if i is not None]
    print("\n=== FEATURES: %d live rows, %d matched to a harness row within %d ms" % (len(feat), len(pairs), a.tol_ms))
    if pairs:
        L = np.array([r[3] for r, _ in pairs]); R = HX[[i for _, i in pairs]]
        D = L - R
        print("%-18s | %9s %9s %9s | %8s %8s | %s" % ("feature", "mean|d|", "p95|d|", "max|d|", "live sd", "corr", "note"))
        for j, n in enumerate(NAMES[:L.shape[1]]):
            d = np.abs(D[:, j]); sd = L[:, j].std()
            c = np.corrcoef(L[:, j], R[:, j])[0, 1] if sd > 1e-9 and R[:, j].std() > 1e-9 else float("nan")
            note = "" if d.mean() <= 0.05 * max(sd, 1e-9) or d.max() < 1e-3 else "<-- differs"
            print("%-18s | %9.4f %9.4f %9.4f | %8.4f %8.3f | %s" % (n, d.mean(), np.percentile(d, 95), d.max(), sd, c, note))
        dt = np.array([abs(int(H.ts[i]) - r[1]) for r, i in pairs])
        print("join |dt|: median %d ms, max %d ms" % (np.median(dt), dt.max()))

# ── score ──
if state:
    ok = window(np.array([r[1] for r in state]))
    state = [r for r, o in zip(state, ok) if o]
    idx = join(state)
    pairs = [(r, i) for r, i in zip(state, idx) if i is not None]
    print("\n=== SCORE: %d live state rows, %d matched" % (len(state), len(pairs)))
    if pairs:
        ls = np.array([r[4] for r, _ in pairs]); hs = H.s.to_numpy()[[i for _, i in pairs]]
        lp = np.array([r[5] for r, _ in pairs]); hp = H.p_up.to_numpy()[[i for _, i in pairs]]
        lm = np.array([r[3] for r, _ in pairs]); hm = H.mid.to_numpy()[[i for _, i in pairs]]
        d = np.abs(ls - hs)
        print("s: mean|d| %.4f p95 %.4f max %.4f | corr %.4f | live sd %.4f | mid mean|d| %.5f" % (d.mean(), np.percentile(d, 95), d.max(), np.corrcoef(ls, hs)[0, 1], ls.std(), np.abs(lm - hm).mean()))
        print("p_up: mean|d| %.4f max %.4f" % (np.abs(lp - hp).mean(), np.abs(lp - hp).max()))
        for thr in (0.26, 0.34, 0.42, 0.48):
            both = ((np.abs(ls) >= thr) & (np.abs(hs) >= thr)).sum(); lo = (np.abs(ls) >= thr).sum(); ho = (np.abs(hs) >= thr).sum()
            print("  |s|>=%.2f: live %d rows, harness %d rows at the same instants, both %d" % (thr, lo, ho, both))
        big = np.argsort(-d)[:8]
        print("  largest score disagreements (ts, ticker, live s, harness s, live mid, harness mid):")
        for k in big:
            r, i = pairs[k]
            print("    %s %s live %+.3f harness %+.3f mid %.4f/%.4f" % (pd.to_datetime(r[1], unit="ms").strftime("%H:%M:%S.%f")[:-3], norm(r[0])[-12:], ls[k], hs[k], lm[k], hm[k]))

# ── signals ──
if a.trades:
    T = pd.read_csv(a.trades, usecols=["ticker", "ts_ms", "side_yes", "gap", "mid0", "tte", "model", "delta"])
    T = T[np.isclose(T.delta, T.delta.max())]
    ok = window(T.ts_ms.to_numpy()); T = T[ok]
    print("\n=== SIGNALS in the window: live %d, harness (latency 0 = every crossing) %d" % (len([s for s in sigs if window(np.array([s[1]]))[0]]), len(T)))
    print("  live:")
    for t, ts, s, mid, tte, d in sigs:
        if not window(np.array([ts]))[0]:
            continue
        near = T[(T.ticker.str.replace("^kalshi\\.", "", regex=True).str.replace("\\.YES$", "", regex=True) == norm(t)) & ((T.ts_ms - ts).abs() <= 2000)]
        print("    %s %s dir %+d s %+.3f mid %.4f tte %.0f | harness fire within 2 s: %s" % (
            pd.to_datetime(ts, unit="ms").strftime("%H:%M:%S.%f")[:-3], norm(t)[-12:], d, s, mid, tte,
            ", ".join("%+.0f ms s=%+.3f" % (r.ts_ms - ts, r.gap * (1 if r.side_yes else -1)) for r in near.itertuples()) or "NONE"))
    print("  harness fires with no live signal within 2 s:")
    for r in T.itertuples():
        tk = norm(r.ticker)
        if not any(norm(t) == tk and abs(ts - r.ts_ms) <= 2000 for t, ts, *_ in sigs):
            print("    %s %s dir %+d |s| %.3f mid %.4f tte %.0f" % (pd.to_datetime(r.ts_ms, unit="ms").strftime("%H:%M:%S.%f")[:-3], tk[-12:], 1 if r.side_yes else -1, r.gap, r.mid0, r.tte))
