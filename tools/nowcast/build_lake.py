"""Lake features for the compact nowcast rows: per-day 100ms bars from the E:\\crypto lake
(binance USDT_PERP BTCUSDT: trade prints w/ aggressor side, l2_snapshot 7.8Hz x 500 levels,
liquidations), then sampled at the midmove_v10 rows (bar = floor((ts_ms - LAG_MS)/100), features as
of that bar; the box's perp view is aligned with exch_ts to +-25ms, LAG_MS=50 keeps it causal).
Writes <SP>/midmove_v10_lake/<day>.lake.npz : lk (n_rows x k float32, NaN -> 0 after 2s ffill), names.
Groups (names prefixed): flow_* (trades), depth_* (book), liq_* (liquidations).
Usage: build_lake.py DAY [DAY ...]"""
import glob
import os
import sys

import numpy as np
import pandas as pd

try:
    import orjson
    loads = orjson.loads
except ImportError:  # pragma: no cover
    import json
    loads = json.loads

BASE = r"E:\crypto\data\parquet"
SP = r"C:/Users/fatli/AppData/Local/Temp/claude/e--poly-crypto-trader/0ed64f57-c300-45f3-b675-113fb239783c/scratchpad"
V10 = os.path.join(SP, os.environ.get("V10_OUT", "midmove_v10"))
BAR_MS, LAG_MS, FF_BARS = 100, 50, 20
BARS = os.environ.get("LAKE_BARS", "") == "1"          # write full-day bar matrices for the harness join instead
BARS_DIR = r"E:/poly/crypto_trader/data/nowcast/lake_bars"
BIG = 1.0            # BTC
BANDS = (5.0, 10.0, 25.0)   # bps


def files(stream, day):
    return sorted(glob.glob(os.path.join(BASE, f"stream={stream}", "venue=binance", "market=USDT_PERP", "symbol=" + os.environ.get("LAKE_SYMBOL", "BTCUSDT"), f"date={day}", "*.parquet")))


def read(stream, day, cols):
    fs = files(stream, day)
    parts = [pd.read_parquet(f, columns=cols) for f in fs]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def roll(x, w):
    c = np.cumsum(np.concatenate([[0.0], x]))
    return c[w:] - c[:-w] if w < len(c) else np.full(len(x), np.nan)


def rsum(s, w):
    return pd.Series(s).rolling(w, min_periods=1).sum().to_numpy()


def build_day(day):
    v10p = os.path.join(V10, day + ".npz")
    if os.path.exists(v10p):
        ts = np.load(v10p)["ts"].astype(np.int64)
    else:                      # bars mode on a day outside the training set: nothing to sample
        assert BARS, "no v10 rows for %s" % day
        ts = np.zeros(0, np.int64)
    day0 = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    nb = 24 * 3600 * 1000 // BAR_MS + 10
    # ── trades → bars
    tr = read("trade", day, ["exch_ts_ns", "price", "qty", "aggressor_side"])
    tr["bar"] = ((tr.exch_ts_ns // 1_000_000 - day0) // BAR_MS).astype(np.int64)
    tr = tr[(tr.bar >= 0) & (tr.bar < nb)]
    buy = tr.aggressor_side.eq("BUY").to_numpy()
    qb = np.bincount(tr.bar, weights=np.where(buy, tr.qty, 0.0), minlength=nb)
    qs = np.bincount(tr.bar, weights=np.where(buy, 0.0, tr.qty), minlength=nb)
    n = np.bincount(tr.bar, minlength=nb).astype(float)
    big = np.bincount(tr.bar, weights=(tr.qty >= BIG).astype(float), minlength=nb)
    notional = np.bincount(tr.bar, weights=(tr.price * tr.qty).to_numpy(), minlength=nb)
    vol = qb + qs
    F = {}
    for w, lab in ((10, "1s"), (50, "5s"), (300, "30s")):
        b, s = rsum(qb, w), rsum(qs, w)
        F["flow_tfi_" + lab] = (b - s) / np.maximum(b + s, 1e-6)
        F["flow_vol_" + lab] = np.log1p(b + s)
    F["flow_n_1s"] = rsum(n, 10); F["flow_n_5s"] = rsum(n, 50)
    F["flow_big_30s"] = rsum(big, 300)
    v60 = rsum(vol, 600); F["flow_burst"] = np.log1p(rsum(vol, 10)) - np.log1p(v60 / 60.0)      # 1s volume vs 60s per-second average
    vw1 = rsum(notional, 10) / np.maximum(rsum(vol, 10), 1e-9)                                  # 1s VWAP (NaN-ish where no trades)
    # ── l2 snapshots → last per bar, top-20 levels + bps bands
    sn = read("l2_snapshot", day, ["exch_ts_ns", "bids_json", "asks_json"])
    sn["bar"] = ((sn.exch_ts_ns // 1_000_000 - day0) // BAR_MS).astype(np.int64)
    sn = sn[(sn.bar >= 0) & (sn.bar < nb)].sort_values("exch_ts_ns").drop_duplicates("bar", keep="last")
    D = {k: np.full(nb, np.nan) for k in ("mid", "spread_bps", "imb1", "imb5", "imb20", "micro_bps", "depth10")}
    for w in BANDS:
        D["band%d" % int(w)] = np.full(nb, np.nan)
    bars = sn.bar.to_numpy()
    for i, (bj, aj) in enumerate(zip(sn.bids_json.to_numpy(), sn.asks_json.to_numpy())):
        b = loads(bj); a = loads(aj)
        if not b or not a:
            continue
        bp = np.array([x[0] for x in b[:60]]); bq = np.array([x[1] for x in b[:60]])
        ap = np.array([x[0] for x in a[:60]]); aq = np.array([x[1] for x in a[:60]])
        k = bars[i]; mid = 0.5 * (bp[0] + ap[0])
        D["mid"][k] = mid; D["spread_bps"][k] = 1e4 * (ap[0] - bp[0]) / mid
        D["imb1"][k] = (bq[0] - aq[0]) / max(bq[0] + aq[0], 1e-9)
        b5, a5 = bq[:5].sum(), aq[:5].sum(); D["imb5"][k] = (b5 - a5) / max(b5 + a5, 1e-9)
        b20, a20 = bq[:20].sum(), aq[:20].sum(); D["imb20"][k] = (b20 - a20) / max(b20 + a20, 1e-9)
        D["micro_bps"][k] = 1e4 * ((bp[0] * aq[0] + ap[0] * bq[0]) / max(bq[0] + aq[0], 1e-9) - mid) / mid
        for w in BANDS:
            mb = bp >= mid * (1 - w * 1e-4); ma = ap <= mid * (1 + w * 1e-4)
            sb, sa = bq[mb].sum(), aq[ma].sum()
            D["band%d" % int(w)][k] = (sb - sa) / max(sb + sa, 1e-9)
            if w == 10.0:
                D["depth10"][k] = np.log1p(sb + sa)
    # ffill (<= 2s) the bar-level book state
    dfD = pd.DataFrame(D).ffill(limit=FF_BARS)
    for k in ("spread_bps", "imb1", "imb5", "imb20", "micro_bps", "depth10") + tuple("band%d" % int(w) for w in BANDS):
        F["depth_" + k] = dfD[k].to_numpy()
    imb5 = dfD["imb5"].to_numpy(); mid = dfD["mid"].to_numpy()
    F["depth_dimb5_1s"] = imb5 - np.roll(imb5, 10); F["depth_dimb5_5s"] = imb5 - np.roll(imb5, 50)
    F["depth_dband10_1s"] = dfD["band10"].to_numpy() - np.roll(dfD["band10"].to_numpy(), 10)
    F["flow_vwap1_bps"] = np.where(np.isfinite(vw1) & (rsum(vol, 10) > 0), 1e4 * (vw1 - mid) / mid, 0.0)
    # ── liquidations
    lq = read("liquidation", day, ["exch_ts_ns", "price", "qty", "side"])
    if len(lq):
        lq["bar"] = ((lq.exch_ts_ns // 1_000_000 - day0) // BAR_MS).astype(np.int64); lq = lq[(lq.bar >= 0) & (lq.bar < nb)]
        lb = np.bincount(lq.bar, weights=np.where(lq.side.eq("BUY"), lq.price * lq.qty, 0.0), minlength=nb)
        ls = np.bincount(lq.bar, weights=np.where(lq.side.eq("BUY"), 0.0, lq.price * lq.qty), minlength=nb)
    else:
        lb = ls = np.zeros(nb)
    F["liq_buy_30s"] = np.log1p(rsum(lb, 300) / 1e3); F["liq_sell_30s"] = np.log1p(rsum(ls, 300) / 1e3)
    F["liq_net_300s"] = np.log1p(rsum(lb, 3000) / 1e3) - np.log1p(rsum(ls, 3000) / 1e3)
    names = list(F.keys())
    M = np.column_stack([F[k] for k in names]).astype(np.float32)
    # ── sample at the v10 rows (causal: bar of ts - LAG_MS)
    rb = np.clip((ts - day0 - LAG_MS) // BAR_MS, 0, nb - 1)
    X = M[rb]
    ok = np.isfinite(X).all(1) if len(ts) else np.zeros(0, bool)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    if BARS:   # full-day 100ms bar matrix (post-ffill) for tools/analyze_online.py lake groups
        os.makedirs(BARS_DIR, exist_ok=True)
        np.savez_compressed(os.path.join(BARS_DIR, day + ".bars.npz"), M=M, day0=np.int64(day0), names=np.array(names))
    else:
        os.makedirs(V10 + "_lake", exist_ok=True)
        np.savez_compressed(os.path.join(V10 + "_lake", day + ".lake.npz"), lk=X, names=np.array(names), ok=ok)
    print("%s: rows %d | features %d | complete rows %.1f%% | snapshots %d trades %d liq %d" % (day, len(ts), len(names), 100 * ok.mean() if len(ok) else float("nan"), len(sn), len(tr), len(lq)), flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        try:
            build_day(d)
        except Exception as e:  # noqa: BLE001
            print("%s: FAILED %s" % (d, e), flush=True)
