#!/usr/bin/env python3
"""Rust-collector analog of crypto_collector/arb_study/kalshi_event_study.py.

Reads the Rust recorder JSONL (data/kalshi-arb-collect/stream=book/venue={binance,kalshi})
and runs the same lead-lag + event study:
  - cross-correlation lead-lag of perp returns vs ΔP(up)
  - perp-move trigger (|ret over trig_s| >= bps, rising edge, refractory) → signed forward
    Kalshi response by P(up) bucket; the decisive LAG (Kalshi flat in trigger) vs MOVED split.

The point: the Rust feeds are fresher than the Python 50ms-snapshot / 136ms-jitter perp, so
this is a cleaner measurement (the perp SOCKS tunnel ~100ms confound still present).

  py scripts/kalshi_arb_study.py --dir data/kalshi-arb-collect --grid 50ms --bps 3 --trig-s 0.2
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd


def load_books(paths, keep):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                i = line.find('"Book"')
                if i < 0:
                    continue
                try:
                    b = json.loads(line)["payload"]["Book"]
                except Exception:
                    continue
                inst = b["instrument"]
                if not keep(inst):
                    continue
                bid = b.get("bids") or []
                ask = b.get("asks") or []
                if not bid or not ask:
                    continue
                rows.append((b["recv_ts_ns"], (bid[0][0] + ask[0][0]) / 2.0, inst))
    return pd.DataFrame(rows, columns=["ts_ns", "mid", "inst"])


def grid(df, freq):
    s = df.set_index(pd.to_datetime(df["ts_ns"], unit="ns"))["mid"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.resample(freq).last().ffill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/kalshi-arb-collect")
    ap.add_argument("--grid", default="50ms")
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--trig-s", type=float, default=0.2)
    ap.add_argument("--cooldown-s", type=float, default=5.0)
    ap.add_argument("--min-window-pts", type=int, default=200)
    a = ap.parse_args()

    bpat = glob.glob(f"{a.dir}/stream=book/venue=binance/**/events.jsonl", recursive=True)
    kpat = glob.glob(f"{a.dir}/stream=book/venue=kalshi/**/events.jsonl", recursive=True)
    btc = load_books(bpat, lambda s: s.startswith("binance."))
    kal = load_books(kpat, lambda s: s.endswith(".YES"))
    if btc.empty or kal.empty:
        print(f"no data yet (btc={len(btc)} kalshi_yes={len(kal)})"); return
    span_h = (btc["ts_ns"].max() - btc["ts_ns"].min()) / 3.6e12
    print(f"perp ticks={len(btc):,} | kalshi YES ticks={len(kal):,} | windows(tickers)={kal['inst'].nunique()} | span={span_h:.1f}h")

    bg_full = grid(btc, a.grid)
    step = pd.Timedelta(a.grid).total_seconds()
    kt = max(1, round(a.trig_s / step))
    HZ = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
    SHOW_H = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]   # horizons shown in the matrices (sub-1s + early)
    PRE_BINS = [("pre<=0", -1e9, 0), ("0-1", 0, 1), ("1-3", 1, 3), ("3-10", 3, 10), (">10", 10, 1e9)]
    hz_k = [max(1, round(h / step)) for h in HZ]
    cd = max(1, round(a.cooldown_s / step))
    BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

    # ---- cross-correlation lead-lag (pooled, on the perp grid) ----
    pup_all = grid(kal, a.grid)
    j = pd.concat({"mid": bg_full, "pup": pup_all}, axis=1).ffill().dropna()
    br = np.log(j["mid"]).diff()
    dp = j["pup"].diff()
    lags = range(-10, 11)  # * grid step
    cc = [(L, br.corr(dp.shift(-L))) for L in lags]
    pk = max(cc, key=lambda t: (t[1] if not np.isnan(t[1]) else -9))
    print(f"lead-lag peak corr={pk[1]:+.3f} at {pk[0]*step*1000:+.0f}ms "
          f"({'P(up) LEADS perp' if pk[0] < 0 else 'P(up) LAGS perp' if pk[0] > 0 else 'synchronous'})")

    # ---- event study, per window ----
    rows = []
    n_edges = 0
    for inst, seg in kal.groupby("inst"):
        pg = grid(seg, a.grid)
        if len(pg) < a.min_window_pts:
            continue
        al = pd.concat({"mid": bg_full.reindex(pg.index), "pup": pg}, axis=1).ffill().dropna()
        if len(al) < kt + max(hz_k) + 5:
            continue
        mid = al["mid"].to_numpy(); pup = al["pup"].to_numpy(); n = len(al)
        bps = np.full(n, np.nan)
        bps[kt:] = (np.log(mid[kt:]) - np.log(mid[:-kt])) * 1e4
        edges = [i for i in range(kt + 1, n - max(hz_k))
                 if abs(bps[i]) >= a.bps and abs(bps[i - 1]) < a.bps]
        n_edges += len(edges)
        last = -10**9
        for i in edges:
            if i - last < cd:
                continue
            last = i
            d = 1.0 if bps[i] > 0 else -1.0
            rec = {"p0": pup[i] * 100, "dir": d, "pre": d * (pup[i] - pup[i - kt]) * 100}
            for h, hk in zip(HZ, hz_k):
                rec[f"r{h}"] = d * (pup[i + hk] - pup[i]) * 100
            rows.append(rec)
    if not rows:
        print(f"{n_edges} raw edges but 0 independent events (need more data)"); return
    df = pd.DataFrame(rows)
    nev = len(df)
    try:  # cache the events table so display tweaks don't re-parse the raw JSONL
        import os
        os.makedirs("data/event_cache", exist_ok=True)
        df.to_parquet(f"data/event_cache/ev_b{a.bps:g}_t{a.trig_s:g}_{a.grid}.parquet")
    except Exception:
        pass
    print(f"\n[trigger {a.bps:.0f}bps / {a.trig_s}s, {a.cooldown_s:.0f}s refractory] "
          f"{n_edges} edges -> {nev} events")

    # LAG vs MOVED (absolute, by how much the token moved IN the trigger window).
    # 'lag' = token didn't move with the perp during the window (pre<=0) -> the
    # genuinely front-runnable case the arb thesis needs.
    print(f"\n[lag vs moved] token in-window move (pre), {nev} events:")
    band = [("pre<=0c  (LAG: flat/against perp)", df["pre"] <= 0),
            ("0<pre<=1c (barely moved)",          (df["pre"] > 0) & (df["pre"] <= 1)),
            ("1<pre<=3c",                         (df["pre"] > 1) & (df["pre"] <= 3)),
            ("3<pre<=10c",                        (df["pre"] > 3) & (df["pre"] <= 10)),
            ("pre>10c  (MOVED a lot)",            df["pre"] > 10)]
    for lab, m in band:
        print(f"    {lab:>34}: {int(m.sum()):>3} ({100*m.mean():>2.0f}%)")
    nlag = int((df["pre"] <= 0).sum())
    print(f"    => genuinely LAGGING (pre<=0): {nlag}/{nev} = {100*nlag/nev:.0f}%   "
          f"| already MOVED (pre>0): {nev-nlag}/{nev} = {100*(nev-nlag)/nev:.0f}%")
    # do the lagging events PAY (catch up)? this is the whole arb thesis.
    for lab, sub in [("LAG  (pre<=0)", df[df["pre"] <= 0]), ("MOVED(pre>0)", df[df["pre"] > 0])]:
        if not len(sub):
            continue
        def dh(c):
            nz = sub[c][sub[c] != 0]
            return f"{(nz>0).mean()*100:>3.0f}%(n={len(nz)})" if len(nz) else "--"
        print(f"    {lab}: +1s={sub['r1.0'].mean():+.2f} +2s={sub['r2.0'].mean():+.2f} "
              f"+5s={sub['r5.0'].mean():+.2f} +10s={sub['r10.0'].mean():+.2f} +30s={sub['r30.0'].mean():+.2f}c  dh@1s={dh('r1.0')}")

    print(f"\n[global] signed P(up) move in perp dir (pre={df['pre'].mean():+.2f}c):")
    print(f"    {'horizon':>8} {'mean_c':>7} {'dir_hit':>8} {'flat%':>6} {'n_move':>7}")
    for h in HZ:
        c = df[f"r{h}"]; nz = c[c != 0]
        dh = (nz > 0).mean() * 100 if len(nz) else float("nan")
        print(f"    {'+'+format(h,'g')+'s':>8} {c.mean():>+7.2f} {dh:>7.0f}% {(c==0).mean()*100:>5.0f}% {len(nz):>7}")
    print("    (dir_hit = of events where P(up) moved, % in perp direction)")

    # ---- forward ΔP(up) cents at horizons OUT TO 30s, as matrices ----
    hdr = " ".join(f"{'+'+format(h,'g')+'s':>7}" for h in SHOW_H)
    def matrix(title, groups):
        print(f"\n{title}")
        print(f"    {'group':>8} {'n':>4} {'pre_c':>6}  {hdr}")
        for lab, mask in groups:
            b = df[mask]
            if not len(b):
                print(f"    {lab:>8} {0:>4}"); continue
            vals = " ".join(f"{b[f'r{h}'].mean():>+7.2f}" for h in SHOW_H)
            sm = " *sm" if len(b) < 10 else ""
            print(f"    {lab:>8} {len(b):>4} {b['pre'].mean():>+6.2f}  {vals}{sm}")

    # ---- full per-price-bucket stats (reactivity, forward, fee, net) ----
    print(f"\n[per P(up) price bucket — full stats]  fee_rt = round-trip taker = 14·p·(1−p) c/share")
    print(f"    {'bucket':>8} {'n':>4} {'up/dn':>7} {'lag%':>5} {'pre_c':>6} "
          f"{'+1s':>6} {'+3s':>6} {'+5s':>6} {'dh@1s':>6} {'fee_rt':>6} {'net+3s':>7}")
    for lo, hi in BUCKETS:
        b = df[(df["p0"] >= lo) & (df["p0"] < hi)]
        if not len(b):
            print(f"    {f'{lo}-{hi}':>8} {0:>4}"); continue
        nup = int((b["dir"] > 0).sum()); ndn = len(b) - nup
        lagp = 100 * (b["pre"] <= 0).mean()
        nz = b["r1.0"][b["r1.0"] != 0]
        dh = f"{(nz>0).mean()*100:>3.0f}%" if len(nz) else "--"
        pmid = (lo + hi) / 200.0
        fee = 14.0 * pmid * (1 - pmid)            # round-trip taker, cents/share
        net = b["r3.0"].mean() - fee
        sm = " *" if len(b) < 10 else ""
        print(f"    {f'{lo}-{hi}':>8} {len(b):>4} {f'{nup}/{ndn}':>7} {lagp:>4.0f}% {b['pre'].mean():>+6.2f} "
              f"{b['r1.0'].mean():>+6.2f} {b['r3.0'].mean():>+6.2f} {b['r5.0'].mean():>+6.2f} {dh:>6} "
              f"{fee:>6.2f} {net:>+7.2f}{sm}")
    matrix("[by pre bucket]  in-window move (pre) bucketed → forward ΔP(up) cents:",
           [(lab, (df["pre"] > lo) & (df["pre"] <= hi)) for lab, lo, hi in PRE_BINS])

    tot = df["pre"].mean() + df["r1.0"].mean()
    if abs(tot) > 1e-9:
        print(f"\n[catch-up] {df['r1.0'].mean()/tot*100:.0f}% of the (pre+1s) move lands AFTER the event "
              f"(predate); {df['pre'].mean()/tot*100:.0f}% already in by event time (synchronous).")


if __name__ == "__main__":
    main()
