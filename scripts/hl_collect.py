#!/usr/bin/env python3
"""Long-running collector: Hyperliquid perp + Binance perp (tunnel) + Coinbase +
Binance.US, all on one local recv clock, written to CSV with per-thread reconnect.
  py scripts/hl_collect.py [seconds]
"""
import websocket, json, threading, time, sys, os

SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 14400
DIR = "data/hl_compare"
os.makedirs(DIR, exist_ok=True)
running = True
files = {k: open(f"{DIR}/{k}.csv", "w", buffering=1 << 16) for k in ("hl", "bp", "cb", "bus")}


def loop(name, conn, parse):
    while running:
        try:
            ws = conn(); ws.settimeout(20)
            while running:
                m = ws.recv(); r = time.time_ns()
                v = parse(m)
                if v is not None:
                    files[name].write(f"{r},{v}\n")
        except Exception:
            time.sleep(2)


def c_hl():
    ws = websocket.create_connection("wss://api.hyperliquid.xyz/ws", timeout=12)
    ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "bbo", "coin": "BTC"}}))
    return ws


def p_hl(m):
    d = json.loads(m)
    if d.get("channel") == "bbo":
        b = d.get("data", {}).get("bbo")
        if b and b[0] and b[1]:
            return (float(b[0]["px"]) + float(b[1]["px"])) / 2
    return None


def c_bp():
    return websocket.create_connection(
        "wss://fstream.binance.com/ws/btcusdt@bookTicker", timeout=14,
        http_proxy_host="127.0.0.1", http_proxy_port=1080, proxy_type="socks5h")


def p_bp(m):
    d = json.loads(m)
    if "b" in d and "a" in d:
        return (float(d["b"]) + float(d["a"])) / 2
    return None


def c_cb():
    ws = websocket.create_connection("wss://ws-feed.exchange.coinbase.com", timeout=12)
    ws.send(json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker"]}))
    return ws


def p_cb(m):
    d = json.loads(m)
    if d.get("type") == "ticker" and d.get("best_bid") and d.get("best_ask"):
        return (float(d["best_bid"]) + float(d["best_ask"])) / 2
    return None


def c_bus():
    return websocket.create_connection("wss://stream.binance.us:9443/ws/btcusd@bookTicker", timeout=12)


def p_bus(m):
    d = json.loads(m)
    if "b" in d and "a" in d:
        return (float(d["b"]) + float(d["a"])) / 2
    return None


for nm, c, p in [("hl", c_hl, p_hl), ("bp", c_bp, p_bp), ("cb", c_cb, p_cb), ("bus", c_bus, p_bus)]:
    threading.Thread(target=loop, args=(nm, c, p), daemon=True).start()
print(f"collecting {SECS}s -> {DIR}/  (hl, bp, cb, bus)")
t_end = time.time() + SECS
while time.time() < t_end:
    time.sleep(10)
    for f in files.values():
        f.flush()
running = False
time.sleep(1)
for f in files.values():
    f.flush()
