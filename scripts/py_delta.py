import sys, glob, pandas as pd, numpy as np

# find a book_ticker USDT_PERP BTCUSDT parquet
pats = glob.glob(r"E:\crypto\collector\data\parquet\stream=book_ticker\venue=binance\market=USDT_PERP\symbol=BTCUSDT\**\*.parquet", recursive=True)
if not pats:
    pats = glob.glob(r"E:\crypto\collector\data\parquet\stream=book_ticker\venue=binance\**\*.parquet", recursive=True)
print("found", len(pats), "book_ticker files; using last")
f = sorted(pats)[-1]
print("file:", f)
df = pd.read_parquet(f)
print("columns:", list(df.columns))
print("rows:", len(df))
print("sample:", df.head(2).to_dict("records"))

cands = {c.lower(): c for c in df.columns}
def pick(*names):
    for n in names:
        for lc, orig in cands.items():
            if n in lc:
                return orig
    return None
recv = pick("recv_ts", "recv_ns", "recv", "client_ts", "local_ts", "ingest_ts", "ts_recv")
exch = pick("transact_ts", "exch_ts", "event_ts", "server_ts", "ts_exch", "_t_ms", "ts_event", "exch")
print("recv col:", recv, "| exch col:", exch)
if recv and exch:
    r = pd.to_numeric(df[recv], errors="coerce").astype("float64")
    e = pd.to_numeric(df[exch], errors="coerce").astype("float64")
    def to_ms(x):
        m = x.dropna().median()
        if m > 1e17: return x / 1e6   # ns
        if m > 1e14: return x / 1e3   # us? unlikely
        return x                      # ms
    d = (to_ms(r) - to_ms(e)).dropna()
    print(f"delta (recv-exch) ms: n={len(d)} min={d.min():.1f} p50={d.median():.1f} "
          f"p90={d.quantile(.9):.1f} p99={d.quantile(.99):.1f} max={d.max():.1f}")
    print(f"jitter above min: p50-min={d.median()-d.min():.1f}ms  p99-min={d.quantile(.99)-d.min():.1f}ms")
