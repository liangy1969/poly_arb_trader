# Extract per-second Binance USDT-perp book features from the e:\crypto L2 lake.
#
# For each UTC second: take the LAST l2_snapshot in that second (matches the
# ffill semantics of the Kalshi trade-print grid) and compute mid, spread and
# a family of order-book imbalance signals:
#   imb1            top-of-book qty imbalance (bq-aq)/(bq+aq)
#   imb5/20/100     depth-K level-sum imbalance
#   band5/10/25     qty imbalance within +/-{5,10,25} bps of mid
#   moff_bps        microprice offset in bps: ((a1*bq1+b1*aq1)/(bq1+aq1)-mid)/mid*1e4
#
# Usage: python extract_l2_features.py <start-date> <end-date> <out.csv.gz>
import sys, os, glob, gzip, json
from datetime import date, timedelta
import numpy as np
import pandas as pd

try:
    import orjson
    loads = orjson.loads
except ImportError:
    loads = json.loads

LAKE = r"E:\crypto\data\parquet\stream=l2_snapshot\venue=binance\market=USDT_PERP\symbol=BTCUSDT"
BANDS_BPS = (5.0, 10.0, 25.0)
KS = (1, 5, 20, 100)

def features(bids, asks):
    b1, bq1 = bids[0]; a1, aq1 = asks[0]
    mid = 0.5 * (b1 + a1)
    out = {"mid": mid, "spread_bps": (a1 - b1) / mid * 1e4}
    for k in KS:
        bs = sum(q for _, q in bids[:k]); asum = sum(q for _, q in asks[:k])
        out[f"imb{k}"] = (bs - asum) / (bs + asum) if bs + asum > 0 else 0.0
    for w in BANDS_BPS:
        lo, hi = mid * (1 - w / 1e4), mid * (1 + w / 1e4)
        bs = sum(q for p, q in bids if p >= lo)
        asum = sum(q for p, q in asks if p <= hi)
        out[f"band{int(w)}"] = (bs - asum) / (bs + asum) if bs + asum > 0 else 0.0
    micro = (a1 * bq1 + b1 * aq1) / (bq1 + aq1)
    out["moff_bps"] = (micro - mid) / mid * 1e4
    return out

def main():
    d0, d1, outp = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
    days = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    all_rows = []
    for day in days:
        fs = sorted(glob.glob(os.path.join(LAKE, f"date={day}", "*.parquet")))
        if not fs:
            print(f"{day}: NO FILES", flush=True); continue
        picks = {}  # sec -> (recv_ns, bids_json, asks_json); keep max recv per sec
        for f in fs:
            df = pd.read_parquet(f, columns=["recv_ts_ns", "bids_json", "asks_json"])
            sec = (df.recv_ts_ns // 1_000_000_000).values
            # last row per second within file (rows are ~time-ordered; use argmax of recv per sec)
            idx = pd.Series(np.arange(len(df))).groupby(sec).last().values
            for i in idx:
                s = int(sec[i])
                r = df.recv_ts_ns.iat[i]
                if s not in picks or r > picks[s][0]:
                    picks[s] = (r, df.bids_json.iat[i], df.asks_json.iat[i])
        rows = []
        for s in sorted(picks):
            _, bj, aj = picks[s]
            try:
                out = features(loads(bj), loads(aj))
            except Exception:
                continue
            out["t"] = s
            rows.append(out)
        all_rows.extend(rows)
        print(f"{day}: files={len(fs)} secs={len(rows)}", flush=True)
    df = pd.DataFrame(all_rows)
    cols = ["t", "mid", "spread_bps"] + [f"imb{k}" for k in KS] + [f"band{int(w)}" for w in BANDS_BPS] + ["moff_bps"]
    df[cols].to_csv(outp, index=False, compression="gzip", float_format="%.6g")
    print(f"wrote {outp}: {len(df):,} rows, {df.t.min()} -> {df.t.max()}")

if __name__ == "__main__":
    main()
