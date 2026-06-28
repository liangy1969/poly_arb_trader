#!/usr/bin/env python3
"""Collect Hyperliquid (US-direct perp) + Coinbase + Binance.US (US-direct spot) on
ONE local recv clock, then lead-lag. Question: does HL lead the US spot venues
(=> a US-accessible leading source, no tunnel) or lag them (=> useless)?
"""
import websocket, json, threading, time, sys
import numpy as np, pandas as pd

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 230
data = {"hl": [], "cb": [], "bus": [], "bp": []}
running = True


def bp():  # Binance.com PERP via SOCKS tunnel (the price leader)
    ws = websocket.create_connection(
        "wss://fstream.binance.com/ws/btcusdt@bookTicker", timeout=14,
        http_proxy_host="127.0.0.1", http_proxy_port=1080, proxy_type="socks5h")
    ws.settimeout(10)
    while running:
        try:
            m = ws.recv(); r = time.time_ns()
        except Exception:
            break
        try:
            d = json.loads(m)
        except Exception:
            continue
        if "b" in d and "a" in d:
            data["bp"].append((r, (float(d["b"]) + float(d["a"])) / 2))


def hl():
    ws = websocket.create_connection("wss://api.hyperliquid.xyz/ws", timeout=12)
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "bbo", "coin": "BTC"}}))
    ws.settimeout(8)
    while running:
        try:
            m = ws.recv(); r = time.time_ns()
        except Exception:
            break
        try:
            d = json.loads(m)
        except Exception:
            continue
        if d.get("channel") == "bbo":
            b = d.get("data", {}).get("bbo")
            if b and b[0] and b[1]:
                data["hl"].append((r, (float(b[0]["px"]) + float(b[1]["px"])) / 2))


def cb():
    ws = websocket.create_connection("wss://ws-feed.exchange.coinbase.com", timeout=12)
    ws.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}))
    ws.settimeout(8)
    while running:
        try:
            m = ws.recv(); r = time.time_ns()
        except Exception:
            break
        try:
            d = json.loads(m)
        except Exception:
            continue
        if d.get("type") == "ticker" and d.get("best_bid") and d.get("best_ask"):
            data["cb"].append((r, (float(d["best_bid"]) + float(d["best_ask"])) / 2))


def bus():
    ws = websocket.create_connection("wss://stream.binance.us:9443/ws/btcusd@bookTicker", timeout=12)
    ws.settimeout(8)
    while running:
        try:
            m = ws.recv(); r = time.time_ns()
        except Exception:
            break
        try:
            d = json.loads(m)
        except Exception:
            continue
        if "b" in d and "a" in d:
            data["bus"].append((r, (float(d["b"]) + float(d["a"])) / 2))


for f in (hl, cb, bus, bp):
    threading.Thread(target=f, daemon=True).start()
print(f"collecting {SECS}s ...")
time.sleep(SECS)
running = False
time.sleep(1)


def ser(rows):
    s = pd.Series([m for _, m in rows], index=pd.to_datetime([t for t, _ in rows]))
    return s[~s.index.duplicated(keep="last")].sort_index().resample("50ms").last().ffill().dropna()


S = {k: ser(v) for k, v in data.items() if v}
for k in data:
    print(f"  {k}: {len(data[k])} updates ({len(data[k])/SECS:.1f}/s)")


def leadlag(a, b, span=600, step=50):
    j = pd.concat({"a": S[a], "b": S[b]}, axis=1).dropna()
    ra = np.log(j["a"]).diff().to_numpy(); rb = np.log(j["b"]).diff().to_numpy()
    res = {}
    for L in range(-span // step, span // step + 1):
        y = np.roll(rb, -L); x = ra
        m = ~(np.isnan(x) | np.isnan(y))
        if L > 0:
            x2, y2 = x[1:-L][m[1:-L]], y[1:-L][m[1:-L]]
        elif L < 0:
            x2, y2 = x[1 - L:][m[1 - L:]], y[1 - L:][m[1 - L:]]
        else:
            x2, y2 = x[1:][m[1:]], y[1:][m[1:]]
        if len(x2) > 30 and x2.std() > 0 and y2.std() > 0:
            res[L * step] = float(np.corrcoef(x2, y2)[0, 1])
    if res:
        pk = max(res, key=res.get)
        print(f"\n{a} vs {b}: peak corr {res[pk]:+.3f} at lag {pk:+d}ms  (lag<0 => {a} LEADS {b})")
        near = "  ".join(f"{l:+d}:{res[l]:+.2f}" for l in sorted(res) if abs(l) <= 200)
        print(f"   {near}")


if "hl" in S and "bp" in S:
    leadlag("hl", "bp")     # <<< the key one: HL perp (US-direct) vs Binance perp (tunneled)
if "cb" in S and "bp" in S:
    leadlag("cb", "bp")
if "hl" in S and "cb" in S:
    leadlag("hl", "cb")
