"""Record the full order book and trade stream of selected Kalshi markets, event by event.

Feeds the maker shadow study: replaying these files with the trade tape lets us simulate a quote
waiting its turn in line (Kalshi fills the best price first, then the earliest order).

Connects to Kalshi's market-data websocket (same host, path and RSA-PSS signing as the live
trader's collector), subscribes to orderbook_delta (snapshot + every change) and trade for the
configured tickers, and appends every message verbatim, with a local receive timestamp, to
data/xmarket/books/YYYY-MM-DD.jsonl. Raw messages are stored unparsed on purpose, so a field
rename on Kalshi's side cannot silently corrupt the record. Reconnects with backoff; a fresh
snapshot arrives after every reconnect.

A pidfile makes the launcher idempotent: run it every few minutes from a scheduled task and only
one copy stays alive.
Usage: book_recorder.py [--tickers KXBTCVSGOLD-26,...]
"""
import asyncio
import base64
import json
import os
import sys
import time

import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

ROOT = r"E:\poly\crypto_trader"
OUT = os.path.join(ROOT, "data", "xmarket", "books")
PIDFILE = os.path.join(ROOT, "data", "xmarket", "book_recorder.pid")
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"
KEY_ID = "e547e0c2-c92f-4d1a-aaef-58d6d589032d"
PEM = r"C:/Users/fatli/.poly/kalshi_private.txt"
DEFAULT_TICKERS = ["KXBTCVSGOLD-26"]


def already_running():
    if not os.path.exists(PIDFILE):
        return False
    try:
        pid = int(open(PIDFILE).read().strip())
    except Exception:
        return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259                                       # STILL_ACTIVE


def auth_headers():
    key = serialization.load_pem_private_key(open(PEM, "rb").read(), password=None)
    ts = str(int(time.time() * 1000))
    sig = key.sign((ts + "GET" + WS_PATH).encode(), padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256())
    return {"KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


class DailyLog:
    def __init__(self):
        os.makedirs(OUT, exist_ok=True)
        self.day, self.f, self.n = None, None, 0

    def write(self, raw):
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if day != self.day:
            if self.f:
                self.f.close()
            self.day = day
            self.f = open(os.path.join(OUT, day + ".jsonl"), "a", encoding="utf-8")
        self.f.write('{"rt":%d,"m":%s}\n' % (int(time.time() * 1000), raw))
        self.n += 1
        self.f.flush()          # low-volume market: flush every message so a kill never loses data


async def run(tickers):
    log = DailyLog()
    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL, additional_headers=auth_headers(), ping_interval=20, max_size=2 ** 24) as ws:
                await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                          "params": {"channels": ["orderbook_delta", "trade"], "market_tickers": tickers}}))
                backoff = 1
                print("connected; subscribed %s" % tickers, flush=True)
                async for raw in ws:
                    log.write(raw if isinstance(raw, str) else raw.decode())
        except Exception as e:
            print("ws error: %s; reconnect in %ds" % (e, backoff), flush=True)
            if log.f:
                log.f.flush()
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


def main():
    if sys.stdout is None or "pythonw" in sys.executable.lower():   # windowless launch: log to a file
        f = open(os.path.join(ROOT, "data", "xmarket", "book_recorder.log"), "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = f
    tickers = DEFAULT_TICKERS
    if "--tickers" in sys.argv:
        tickers = sys.argv[sys.argv.index("--tickers") + 1].split(",")
    if already_running():
        print("already running")
        return
    open(PIDFILE, "w").write(str(os.getpid()))
    asyncio.run(run(tickers))


if __name__ == "__main__":
    main()
