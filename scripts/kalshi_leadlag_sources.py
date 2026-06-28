#!/usr/bin/env python3
"""Comparative lead-lag: for each BTC price source, how does it lead/lag Kalshi P(up)?
The source whose moves most PREDATE Kalshi's P(up) changes is the best predictor.

Per-Kalshi-window cross-correlation of each source's log-returns vs ΔP(up), pooled
across windows, scanned over lags. Peak lag < 0 ⇒ the source LEADS Kalshi.

  py scripts/kalshi_leadlag_sources.py --dir data/leadlag-collect --grid 20ms
"""
import argparse, glob
import numpy as np
import pandas as pd
from kalshi_arb_study import load_books, grid

SOURCES = [
    ("binance(tunnel)", "binance",   lambda s: s.startswith("binance.usdt_perp")),
    ("binanceus(spot)", "binanceus", lambda s: s == "binanceus.BTC"),
    ("coinbase(spot)",  "coinbase",  lambda s: s == "coinbase.BTC"),
    ("databento(MBT)",  "databento", lambda s: s.startswith("databento.")),
]


def load_src(dir, venue, filt):
    return load_books(glob.glob(f"{dir}/stream=book/venue={venue}/**/events.jsonl", recursive=True), filt)


SRC_BY_NAME = {n: (v, f) for n, v, f in SOURCES}


def event_times(dir, ref_name, grid_str, step_ms, win_ms, bps, cooldown_ms):
    """Rising-edge times where |ref move| over `win_ms` ≥ `bps`, with a refractory."""
    venue, filt = SRC_BY_NAME[ref_name]
    g = grid(load_src(dir, venue, filt), grid_str)
    if g.empty:
        return pd.DatetimeIndex([])
    win = max(1, round(win_ms / step_ms))
    move = (np.log(g) - np.log(g).shift(win)).abs() * 1e4
    trig = (move >= bps).to_numpy()
    idx = g.index
    cd = cooldown_ms * 1e6
    out, last, prev = [], None, False
    ii = idx.view("int64")
    for k in range(len(trig)):
        on = bool(trig[k])
        if on and not prev and (last is None or ii[k] - last >= cd):
            out.append(idx[k]); last = ii[k]
        prev = on
    return pd.DatetimeIndex(out)


def mask_near(idx, ev, pad_ms):
    if len(ev) == 0:
        return np.zeros(len(idx), bool)
    e = ev.view("int64"); ii = idx.view("int64"); pad = pad_ms * 1e6
    pos = np.searchsorted(e, ii)
    keep = np.zeros(len(ii), bool)
    for k in (pos - 1, pos):
        kk = np.clip(k, 0, len(e) - 1)
        keep |= np.abs(ii - e[kk]) <= pad
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/leadlag-collect")
    ap.add_argument("--grid", default="20ms")
    ap.add_argument("--span-ms", type=int, default=300)
    ap.add_argument("--min-window-pts", type=int, default=300)
    ap.add_argument("--event-bps", type=float, default=0.0, help=">0 conditions on |ref move|≥bps windows")
    ap.add_argument("--event-ref", default="binance(tunnel)")
    ap.add_argument("--event-win-ms", type=int, default=200)
    ap.add_argument("--event-pad-ms", type=int, default=600)
    ap.add_argument("--event-cooldown-ms", type=int, default=1500)
    ap.add_argument("--mode", default="btc", help="btc | sp")
    ap.add_argument("--kalshi-substr", default="", help="restrict kalshi instruments containing this")
    a = ap.parse_args()
    global SRC_BY_NAME
    if a.mode == "sp":
        sources = [("ES(CME)", "databento", lambda s: s == "databento.ES"),
                   ("NQ(CME)", "databento", lambda s: s == "databento.NQ")]
        if a.event_ref == "binance(tunnel)":
            a.event_ref = "ES(CME)"
    else:
        sources = SOURCES
    SRC_BY_NAME = {n: (v, f) for n, v, f in sources}
    step_ms = pd.Timedelta(a.grid).total_seconds() * 1000
    lags = list(range(-round(a.span_ms / step_ms), round(a.span_ms / step_ms) + 1))

    kal = load_src(a.dir, "kalshi", lambda s: s.endswith(".YES") and a.kalshi_substr in s)
    if kal.empty:
        print("no kalshi YES yet"); return
    kwins = [(inst, grid(seg, a.grid)) for inst, seg in kal.groupby("inst")]
    kwins = [(i, p) for i, p in kwins if len(p) >= a.min_window_pts]
    span_h = (kal["ts_ns"].max() - kal["ts_ns"].min()) / 3.6e12

    ev = pd.DatetimeIndex([])
    if a.event_bps > 0:
        ev = event_times(a.dir, a.event_ref, a.grid, step_ms, a.event_win_ms, a.event_bps, a.event_cooldown_ms)
        print(f"CONDITIONED on |{a.event_ref} move|>={a.event_bps}bps/{a.event_win_ms}ms  "
              f"-> {len(ev)} events, +/-{a.event_pad_ms}ms windows")
    print(f"kalshi YES ticks={len(kal):,}  windows={kal['inst'].nunique()} (usable {len(kwins)})  span={span_h:.2f}h  grid={a.grid}\n")

    print(f"  {'source':>16} {'ticks':>9} {'peak_corr':>9} {'lead_ms':>8}  {'curve (ms:corr near 0)':>20}")
    results = []
    for name, venue, filt in sources:
        btc = load_src(a.dir, venue, filt)
        if btc.empty:
            print(f"  {name:>16} {'0':>9}  (no data)"); continue
        bg = grid(btc, a.grid)
        # pool the cross-products across windows
        num = {L: 0.0 for L in lags}
        den_b = {L: 0.0 for L in lags}
        den_p = {L: 0.0 for L in lags}
        cnt = {L: 0 for L in lags}
        for inst, pg in kwins:
            j = pd.concat({"mid": bg.reindex(pg.index), "pup": pg}, axis=1, sort=True).ffill().dropna()
            if len(j) < 50:
                continue
            br = np.log(j["mid"]).diff().to_numpy()
            dp = j["pup"].diff().to_numpy()
            keep = mask_near(j.index, ev, a.event_pad_ms) if a.event_bps > 0 else np.ones(len(j), bool)
            for L in lags:
                x = br
                y = np.roll(dp, -L)
                k = keep
                # valid overlap (drop wrapped ends)
                if L >= 0:
                    x, y, k = x[1:len(x) - L], (y[1:len(y) - L] if L > 0 else y[1:]), k[1:len(k) - L] if L > 0 else k[1:]
                else:
                    x, y, k = x[1 - L:], y[1 - L:], k[1 - L:]
                m = k & ~(np.isnan(x) | np.isnan(y))
                x, y = x[m], y[m]
                if len(x) < 20:
                    continue
                num[L] += float((x * y).sum())
                den_b[L] += float((x * x).sum())
                den_p[L] += float((y * y).sum())
                cnt[L] += len(x)
        corr = {L: (num[L] / np.sqrt(den_b[L] * den_p[L]) if den_b[L] > 0 and den_p[L] > 0 else np.nan) for L in lags}
        corr = {L: c for L, c in corr.items() if not np.isnan(c)}
        if not corr:
            print(f"  {name:>16} {len(btc):>9}  (insufficient overlap)"); continue
        pk = max(corr, key=corr.get)
        near = "  ".join(f"{int(L*step_ms):+d}:{corr[L]:+.3f}" for L in sorted(corr) if abs(L) <= 2)
        print(f"  {name:>16} {len(btc):>9} {corr[pk]:>+9.3f} {pk*step_ms:>+7.0f}  {near}")
        results.append((name, corr[pk], pk * step_ms))

    if results:
        print("\nranked by LEAD over Kalshi (most-negative lead_ms = earliest predictor):")
        for name, c, lead in sorted(results, key=lambda r: r[2]):
            print(f"  {name:>16}  lead={lead:+.0f}ms  peak_corr={c:+.3f}")


if __name__ == "__main__":
    main()
