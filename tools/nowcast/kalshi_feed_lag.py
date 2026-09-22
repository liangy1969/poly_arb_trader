"""Kalshi order-book FEED LAG vs the matching engine, from the venue-timestamped trade tape (tools/kalshi_trades_poller.py,
`created_time` = Kalshi's clock) and the trader's 50 ms sampler book (local clock, same WS feed the executor uses).
For every tape trade that CONSUMES a touch level (taker_side "no" hits the YES bid at p: the local best bid must fall
below p; taker "yes" lifts the YES ask at p: the local best ask must rise above p), the lag is the time from the venue's
trade creation to the first local sample whose touch has moved past p, given the local touch still sat at p (or better)
at the trade time. Reports the lag distribution overall, by hour, and for "burst" trades (>= 3 touch trades within 1 s).
Usage: kalshi_feed_lag.py SAMPLER_BOOK.csv[.gz] TAPE.csv[.gz] [--max-lag-ms 3000]"""
import argparse

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("book"); ap.add_argument("tape"); ap.add_argument("--max-lag-ms", type=int, default=3000)
a = ap.parse_args()
B = pd.read_csv(a.book, usecols=["ts_ms", "ticker", "ybid", "yask"], dtype={"ts_ms": np.int64, "ticker": str, "ybid": float, "yask": float})
B = B[(B.ybid > 0) & (B.yask > B.ybid)]
T = pd.read_csv(a.tape, usecols=["created_time", "ticker", "yes_price", "count", "taker_side"])
T["t_ms"] = (pd.to_datetime(T.created_time, utc=True, format="ISO8601").astype("int64") // 1_000_000).astype(np.int64)
T = T[T.t_ms.between(B.ts_ms.min(), B.ts_ms.max())].sort_values("t_ms")
print("book rows %d (%d markets) | tape trades in range %d" % (len(B), B.ticker.nunique(), len(T)))
rows = []
for tk, g in T.groupby("ticker"):
    b = B[B.ticker == tk].sort_values("ts_ms")
    if len(b) < 100:
        continue
    ts, bid, ask = b.ts_ms.to_numpy(), b.ybid.to_numpy(), b.yask.to_numpy()
    # dedupe: one observation per (t_ms, price, side) — a sweep prints many fills at one level
    g = g.drop_duplicates(["t_ms", "yes_price", "taker_side"])
    tt = g.t_ms.to_numpy(); n_near = np.array([((tt >= t - 1000) & (tt <= t)).sum() for t in tt])
    for (t, p, side), nn in zip(g[["t_ms", "yes_price", "taker_side"]].itertuples(index=False), n_near):
        i = np.searchsorted(ts, t, side="right") - 1          # local state at the venue trade time
        if i < 0:
            continue
        # a venue print at p on the bid side means the venue's best bid is AT OR BELOW p right after t; the lag is the
        # time until the LOCAL best bid first shows <= p, given the local bid still sat ABOVE p at t (the local feed had
        # not yet seen the move). Same for lifted asks with >= p. (A level is hit many times before it empties, so
        # "until the level disappears" would measure exhaustion, not lag.)
        if side == "no":
            if bid[i] <= p + 1e-9:
                rows.append((tk, t, p, side, nn, 0, "local_already_past")); continue
            j = i + np.argmax(bid[i:] <= p + 1e-9) if (bid[i:] <= p + 1e-9).any() else -1
        else:
            if ask[i] >= p - 1e-9:
                rows.append((tk, t, p, side, nn, 0, "local_already_past")); continue
            j = i + np.argmax(ask[i:] >= p - 1e-9) if (ask[i:] >= p - 1e-9).any() else -1
        if j < 0 or ts[j] - t > a.max_lag_ms:
            rows.append((tk, t, p, side, nn, np.nan, "never/too_long")); continue
        rows.append((tk, t, p, side, nn, ts[j] - t, "ok"))
R = pd.DataFrame(rows, columns=["ticker", "t_ms", "p", "side", "n_near", "lag_ms", "kind"])
print("classified: %s" % R.kind.value_counts().to_dict())
ok = R[R.kind == "ok"].copy()
print("\nlocal-book lag after a venue touch trade (ms): n=%d  median %.0f  mean %.0f  p25 %.0f  p75 %.0f  p90 %.0f  p99 %.0f" % (
    len(ok), ok.lag_ms.median(), ok.lag_ms.mean(), *np.percentile(ok.lag_ms, [25, 75, 90, 99])))
print("(50 ms sampler: a lag below ~50 ms is not resolvable; 'local_already_past' = the local book had already moved when the venue printed, i.e. lag <= 0 or the level was pulled earlier)")
ok["hour"] = pd.to_datetime(ok.t_ms, unit="ms").dt.strftime("%H")
print("\nby hour (UTC): n, median, p90")
print(ok.groupby("hour").lag_ms.agg(["size", "median", lambda v: np.percentile(v, 90)]).rename(columns={"<lambda_0>": "p90"}).round(0).to_string())
print("\nby burst size (touch trades in the preceding 1 s): n, median, p90")
ok["burst"] = pd.cut(ok.n_near, [0, 1, 2, 4, 8, 1000], labels=["1", "2", "3-4", "5-8", "9+"])
print(ok.groupby("burst", observed=True).lag_ms.agg(["size", "median", lambda v: np.percentile(v, 90)]).rename(columns={"<lambda_0>": "p90"}).round(0).to_string())
print("\nby side: n, median")
print(ok.groupby("side").lag_ms.agg(["size", "median"]).round(0).to_string())
