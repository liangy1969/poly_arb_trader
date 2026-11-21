"""Would a passive quote inside the spread of slow, BTC-driven event markets ever fill — and would
those fills make money? A measurement tool; it places no orders.

collect   For each Kalshi market in the crypto driver group (from corr_theses.json, plus --extra
          tickers), page the public trade tape (/markets/trades, authenticated host) from the last
          stored trade forward (first run: from --since, default the start of our snapshots) and
          append to data/xmarket/flow/kalshi_<ticker>.csv (trade_id, ts, taker_side, yes_price, count).
          Polymarket no longer serves public trades (all data-api trade endpoints return 410 Gone),
          so Polymarket members are skipped.

analyze   For each market, join every trade with the last book snapshot before it, and simulate a
          maker who quotes ONE TICK INSIDE the spread on both sides whenever the spread is >= 3c
          (bid+1c / ask-1c), so any taker would have hit that maker first:
            taker buys YES  -> maker sold YES at ask-1c ; markout = fill - mid(t+h)
            taker sells YES -> maker bought YES at bid+1c; markout = mid(t+h) - fill
          Kalshi makers pay no fee. Reported per market: trades and contracts per day, fills the
          maker would have taken, mean markout per contract at 30 min and 2 h, and the same split
          by whether BTC moved >= 0.3% in the 15 minutes before the trade (fills the maker would
          have AVOIDED by repricing with BTC are the lag-driven, adverse ones).
          Caveat: the book is from the last 15-minute snapshot, so "inside the spread" is judged
          against a possibly stale quote.

Usage: maker_flow_probe.py collect [--since 2026-11-12] [--extra T1,T2]
       maker_flow_probe.py analyze
"""
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = "data/xmarket"
FLOW = os.path.join(OUT, "flow")


def signed_get(path):
    import base64
    import http.client
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = serialization.load_pem_private_key(open(r"C:/Users/fatli/.poly/kalshi_private.txt", "rb").read(), password=None)
    for att in range(5):
        try:
            conn = http.client.HTTPSConnection("external-api.kalshi.com", timeout=30)
            ts = str(int(time.time() * 1000))
            sig = key.sign((ts + "GET" + path.split("?")[0]).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
            conn.request("GET", path, headers={"KALSHI-ACCESS-KEY": "e547e0c2-c92f-4d1a-aaef-58d6d589032d",
                                               "KALSHI-ACCESS-TIMESTAMP": ts, "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
                                               "User-Agent": "Mozilla/5.0"})
            return json.loads(conn.getresponse().read().decode())
        except Exception:
            time.sleep(3 * (att + 1))
    raise RuntimeError("kalshi request failed: " + path)


def members(extra):
    th = json.load(open(os.path.join(OUT, "corr_theses.json")))
    grp = next(t for t in th if "Crypto price level" in t["driver"])
    ks = [(m["id"], 1 if m.get("sign", "+") == "+" else -1) for m in grp["members"] if m["venue"] == "kalshi"]
    for t in extra:
        if t and t not in [k for k, _ in ks]:
            ks.append((t, 1))
    return ks


def parse_ts(s):
    return pd.Timestamp(s).timestamp()          # tolerant of Kalshi's variable-length fractional seconds


def cmd_collect(since, extra):
    os.makedirs(FLOW, exist_ok=True)
    since_ts = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())
    for tk, _ in members(extra):
        path = os.path.join(FLOW, "kalshi_%s.csv" % tk)
        seen, last = set(), since_ts
        if os.path.exists(path):
            old = pd.read_csv(path)
            seen = set(old.trade_id)
            if len(old):
                last = int(old.ts.max()) - 60
        rows, cursor = [], ""
        while True:
            q = "/trade-api/v2/markets/trades?ticker=%s&limit=1000&min_ts=%d" % (tk, last)
            js = signed_get(q + ("&cursor=" + cursor if cursor else ""))
            for t in js.get("trades", []):
                if t["trade_id"] in seen:
                    continue
                seen.add(t["trade_id"])
                rows.append((t["trade_id"], parse_ts(t["created_time"]), t.get("taker_side"),
                             float(t.get("yes_price_dollars") or 0), float(t.get("count_fp") or t.get("count") or 0)))
            cursor = js.get("cursor") or ""
            if not cursor:
                break
            time.sleep(0.3)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["trade_id", "ts", "taker_side", "yes_price", "count"])
            w.writerows(rows)
        print("%-34s +%d trades" % (tk, len(rows)), flush=True)
        time.sleep(0.5)


def btc_moves(t0, t1):
    days = pd.date_range(pd.to_datetime(t0, unit="s").normalize(), pd.to_datetime(t1, unit="s").normalize()).strftime("%Y-%m-%d")
    fs = []
    for d in days:
        fs += glob.glob(r"E:/crypto/data/parquet/stream=trade/venue=binance/market=USDT_PERP/symbol=BTCUSDT/date=%s/*.parquet" % d)
    if not fs:
        return pd.Series(dtype=float)
    P = pd.concat([pd.read_parquet(f, columns=["exch_ts_ns", "price"]) for f in fs])
    P["m"] = (P.exch_ts_ns // 1_000_000_000 // 60) * 60          # 1-minute bars
    close = P.groupby("m").price.last().sort_index()
    return close


def cmd_analyze():
    S = pd.read_csv(os.path.join(OUT, "snapshots.csv"))
    S = S[(S.venue == "kalshi") & (S.yes_bid > 0) & (S.yes_ask > 0) & (S.yes_ask < 1)].copy()
    S["mid"] = 0.5 * (S.yes_bid + S.yes_ask)
    files = sorted(glob.glob(os.path.join(FLOW, "kalshi_*.csv")))
    if not files:
        print("no trade tapes yet — run collect first")
        return
    tmin = min(pd.read_csv(f).ts.min() for f in files if os.path.getsize(f) > 40)
    btc = btc_moves(tmin, time.time())
    bt, bp = btc.index.to_numpy(), btc.to_numpy()

    def btc_move_before(t):
        """BTC return over the 15 minutes ending at t (from 1-minute closes)."""
        if len(bt) == 0:
            return np.nan
        i1 = np.searchsorted(bt, t, side="right") - 1
        i0 = np.searchsorted(bt, t - 900, side="right") - 1
        if i0 < 0 or i1 < 0:
            return np.nan
        return bp[i1] / bp[i0] - 1

    tot = []
    print("%-28s %6s %8s %7s %7s | %9s %9s | %9s %9s" % ("market", "days", "trades/d", "ctr/d", "fills", "mk30m_c", "mk2h_c", "quiet2h_c", "lag2h_c"))
    for f in files:
        tk = os.path.basename(f)[7:-4]
        T = pd.read_csv(f)
        s = S[S.id == tk].sort_values("ts")
        if len(T) == 0 or len(s) < 20:
            continue
        sts, sb, sa, sm = s.ts.to_numpy(), s.yes_bid.to_numpy(), s.yes_ask.to_numpy(), s.mid.to_numpy()
        T = T[(T.ts >= sts[0]) & (T.ts <= sts[-1])]
        days = (sts[-1] - sts[0]) / 86400
        fills = []
        for r in T.itertuples():
            i = np.searchsorted(sts, r.ts, side="right") - 1
            if i < 0:
                continue
            bid, ask = sb[i], sa[i]
            if ask - bid < 0.03 - 1e-9:          # no room to quote inside
                continue
            out = {}
            for h, lab in ((1800, "m30"), (7200, "m2h")):
                j = np.searchsorted(sts, r.ts + h, side="right") - 1
                out[lab] = sm[j] if j > i else np.nan
            if r.taker_side == "yes":           # taker bought YES -> maker sold at ask-1c
                px = ask - 0.01
                mk30, mk2h = px - out["m30"], px - out["m2h"]
            else:                               # taker sold YES -> maker bought at bid+1c
                px = bid + 0.01
                mk30, mk2h = out["m30"] - px, out["m2h"] - px
            fills.append((r.count, mk30, mk2h, abs(btc_move_before(r.ts)) >= 0.003))
        if not fills:
            print("%-28s %6.1f %8.1f %7.0f %7d | (spread always < 3c)" % (tk[:28], days, len(T) / days, T["count"].sum() / days, 0))
            continue
        F = pd.DataFrame(fills, columns=["n", "mk30", "mk2h", "lag"])
        w = lambda d, c: (d[c] * d.n).sum() / d.n[d[c].notna()].sum() if d[c].notna().any() else np.nan
        q, l = F[~F.lag], F[F.lag]
        print("%-28s %6.1f %8.1f %7.0f %7d | %+9.2f %+9.2f | %+9.2f %+9.2f" % (
            tk[:28], days, len(T) / days, T["count"].sum() / days, len(F),
            100 * w(F, "mk30"), 100 * w(F, "mk2h"), 100 * w(q, "mk2h"), 100 * w(l, "mk2h")))
        F["tk"] = tk
        tot.append(F)
    if tot:
        A = pd.concat(tot)
        w2 = (A.mk2h * A.n).sum() / A.n[A.mk2h.notna()].sum()
        q = A[~A.lag]
        wq = (q.mk2h * q.n).sum() / q.n[q.mk2h.notna()].sum()
        print("\nALL: %d maker fills, %.0f contracts | markout at 2h %+.2fc per contract | quiet-BTC fills only %+.2fc (%d fills)" % (
            len(A), A.n.sum(), 100 * w2, 100 * wq, len(q)))
        print("(markout = what the maker's fill is worth 2h later vs the mid; positive = the spread capture survived)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    extra = sys.argv[sys.argv.index("--extra") + 1].split(",") if "--extra" in sys.argv else []
    if cmd == "collect":
        since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else "2026-11-12"
        cmd_collect(since, extra)
    else:
        cmd_analyze()
