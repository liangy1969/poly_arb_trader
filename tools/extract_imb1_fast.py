# Full-cadence (~105ms) top-of-book imbalance from the e:\crypto perp L2 lake.
# Only level 1 is parsed (string prefix, no full JSON) so every snapshot is kept.
# Output: t_ms, imb1, mid   -- for aligning onto the 50ms sampler grid.
#
# Usage: python extract_imb1_fast.py <start-date> <end-date> <out.csv.gz> [ts_col]
#   ts_col: recv_ts_ns (default) | exch_ts_ns (Binance server stamp)
import sys, os, glob, gzip
from datetime import date, timedelta
import pandas as pd

LAKE = r"E:\crypto\data\parquet\stream=l2_snapshot\venue=binance\market=USDT_PERP\symbol=BTCUSDT"

def top(s):
    # "[[58457.3, 9.997], ..." -> (58457.3, 9.997)
    p, q = s[2 : s.index("]")].split(",")
    return float(p), float(q)

def main():
    d0, d1, outp = date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]), sys.argv[3]
    ts_col = sys.argv[4] if len(sys.argv) > 4 else "recv_ts_ns"
    days = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    with gzip.open(outp, "wt") as out:
        out.write("t_ms,imb1,mid\n")
        for day in days:
            fs = sorted(glob.glob(os.path.join(LAKE, f"date={day}", "*.parquet")))
            n = 0
            for f in fs:
                df = pd.read_parquet(f, columns=[ts_col, "bids_json", "asks_json"])
                for r, bj, aj in zip(df[ts_col].values, df.bids_json.values, df.asks_json.values):
                    try:
                        b1, bq = top(bj)
                        a1, aq = top(aj)
                    except Exception:
                        continue
                    if bq + aq <= 0:
                        continue
                    out.write(f"{r // 1_000_000},{(bq - aq) / (bq + aq):.5f},{(b1 + a1) / 2:.2f}\n")
                    n += 1
            print(f"{day}: files={len(fs)} rows={n}", flush=True)
    print(f"wrote {outp}")

if __name__ == "__main__":
    main()
