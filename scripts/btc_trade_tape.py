#!/usr/bin/env python3
"""Around each Binance-perp synced move (>=bps over win), characterise the Kalshi
BTC-15m trade tape: how many trades, what volume, sweep pattern (one taker crossing
multiple price levels), and buy/sell direction vs the perp move.

  py scripts/btc_trade_tape.py --bps 3 --win-ms 200
"""
import argparse, glob, json
import numpy as np
import pandas as pd

DIR = "data/leadlag-collect"


def load_perp():
    ts, mid = [], []
    for f in glob.glob(f"{DIR}/stream=book/venue=binance/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                e = json.loads(ln); b = e["payload"]["Book"]
            except Exception:
                continue
            if not b.get("instrument", "").startswith("binance.usdt_perp") or not b["bids"] or not b["asks"]:
                continue
            ts.append(e["ts_ns"]); mid.append((b["bids"][0][0] + b["asks"][0][0]) / 2)
    o = np.argsort(ts)
    s = pd.Series(np.array(mid)[o], index=pd.to_datetime(np.array(ts)[o], utc=True))
    return s[~s.index.duplicated(keep="last")]


def load_trades():
    rows = []
    for f in glob.glob(f"{DIR}/stream=trade/venue=kalshi/**/events.jsonl", recursive=True):
        for ln in open(f, encoding="utf-8"):
            try:
                t = json.loads(ln)["payload"]["Trade"]
            except Exception:
                continue
            if "KXBTC15M" not in t.get("instrument", ""):
                continue
            rows.append((json.loads(ln)["ts_ns"], t["price"], t["qty"], t["side"], t.get("exch_ts_ns", 0)))
    rows.sort()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), \
        np.array([r[2] for r in rows]), np.array([r[3] for r in rows], dtype=object), \
        np.array([r[4] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=float, default=3.0)
    ap.add_argument("--win-ms", type=int, default=200)
    ap.add_argument("--pre-ms", type=int, default=200)
    ap.add_argument("--post-ms", type=int, default=400)
    ap.add_argument("--refractory-ms", type=int, default=1000)
    a = ap.parse_args()
    perp = load_perp().resample("50ms").last().ffill().dropna()
    tts, tpx, tqty, tside, texch = load_trades()
    hrs = (perp.index[-1] - perp.index[0]).total_seconds() / 3600
    print(f"perp ticks(50ms)={len(perp):,}  kalshi BTC trades={len(tts):,}  span={hrs:.2f}h")

    win = max(1, a.win_ms // 50)
    mv = (np.log(perp) - np.log(perp).shift(win))
    mag = (mv.abs() * 1e4).to_numpy(); sgn = np.sign(mv).to_numpy()
    ii = np.asarray(perp.index.view("int64"))
    events = []; last = None; prev = False
    for k in range(len(mag)):
        on = mag[k] >= a.bps
        if on and not prev and (last is None or ii[k] - last >= a.refractory_ms * 1e6):
            events.append((ii[k], sgn[k])); last = ii[k]
        prev = on
    print(f">={a.bps}bps/{a.win_ms}ms perp moves: {len(events)} ({len(events)/hrs:.0f}/h)\n")

    # baseline: random 600ms windows -> trades per window
    base_rate = len(tts) / (hrs * 3600) * ((a.pre_ms + a.post_ms) / 1000)

    n_tr, vol, n_lvl, aligned_q, total_q, swept, with_trade = [], [], [], [], [], 0, 0
    for t0, d in events:
        lo, hi = t0 - a.pre_ms * 1e6, t0 + a.post_ms * 1e6
        m = (tts >= lo) & (tts < hi)
        if m.sum() == 0:
            n_tr.append(0); continue
        with_trade += 1
        px, q, sd, ex = tpx[m], tqty[m], tside[m], texch[m]
        n_tr.append(m.sum()); vol.append(q.sum())
        n_lvl.append(len(set(np.round(px, 4))))
        # aligned: perp up -> taker BUYS yes ; perp down -> taker SELLS yes
        want = "Buy" if d > 0 else "Sell"
        aq = q[sd == want].sum(); aligned_q.append(aq); total_q.append(q.sum())
        # sweep: >=2 fills sharing an exch_ts but different prices
        for et in set(ex):
            sub = ex == et
            if sub.sum() >= 2 and len(set(np.round(px[sub], 4))) >= 2:
                swept += 1; break

    nt = np.array(n_tr)
    print(f"  windows with >=1 trade: {with_trade}/{len(events)} ({100*with_trade/len(events):.0f}%)   baseline ~{base_rate:.1f} trades/equal-window")
    print(f"  trades / move-window : mean={nt.mean():.1f}  median={np.median(nt):.0f}  p90={np.percentile(nt,90):.0f}  max={nt.max()}")
    if vol:
        v = np.array(vol)
        print(f"  volume / move-window : mean={v.mean():.0f}  median={np.median(v):.0f}  p90={np.percentile(v,90):.0f} contracts (~${np.median(v)*0.5:.0f} at 50c)")
        print(f"  distinct price levels: mean={np.mean(n_lvl):.1f}  max={max(n_lvl)}   sweeps(>=2 lvl same ts): {swept}/{with_trade} ({100*swept/max(with_trade,1):.0f}%)")
        af = np.sum(aligned_q) / max(np.sum(total_q), 1e-9)
        print(f"  direction: {100*af:.0f}% of volume ALIGNED with perp move (buy-yes on up / sell-yes on down)")


if __name__ == "__main__":
    main()
