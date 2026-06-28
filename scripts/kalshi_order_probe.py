#!/usr/bin/env python3
"""Read-only validation of the Kalshi order integration: RSA-PSS REST auth, order
RTT, and the real order/fill response shapes (to confirm the field names the Rust
KalshiVenue adapter parses). Places NO order. Also prints the exact IOC order body
the adapter WOULD send (dry run), without sending it.
"""
import base64, http.client, json, os, sys, time

# Env-configurable so the same probe targets prod (read-only) or demo (paper trade).
#   demo: KALSHI_HOST=external-api.demo.kalshi.co KALSHI_KEY_ID=<demo> KALSHI_PEM=<demo.pem>
KEY_ID = os.environ.get("KALSHI_KEY_ID", "")    # set via env — never hardcode
PEM = os.environ.get("KALSHI_PEM", "")          # path to your Kalshi RSA private key PEM
HOST = os.environ.get("KALSHI_HOST", "external-api.kalshi.com")
IS_DEMO = "demo" in HOST

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

key = serialization.load_pem_private_key(open(PEM, "rb").read(), password=None)
conn = http.client.HTTPSConnection(HOST, timeout=10)  # reused -> warm RTT


def sign(method, path):
    ts = str(int(time.time() * 1000))
    sig = key.sign((ts + method + path).encode(),
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
                   hashes.SHA256())
    return ts, base64.b64encode(sig).decode()


def get(path):
    sign_path = path.split("?")[0]  # Kalshi signs the path WITHOUT the query string
    ts, sig = sign("GET", sign_path)
    t0 = time.time()
    conn.request("GET", path, headers={
        "KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig, "User-Agent": "Mozilla/5.0"})
    r = conn.getresponse(); body = r.read().decode()
    return r.status, (time.time() - t0) * 1000, body


def keys_of(body, path):
    try:
        d = json.loads(body)
        arr = d.get(path, [])
        return list(arr[0].keys()) if arr else f"(empty {path})"
    except Exception as e:
        return f"(parse err {e})"


print("=== 1) AUTH + RTT: GET /portfolio/balance (read-only, warm keep-alive) ===")
rtts = []
for i in range(8):
    code, rtt, body = get("/trade-api/v2/portfolio/balance")
    rtts.append(rtt)
print(f"  status={code}  cold(1st)={rtts[0]:.0f}ms")
warm = sorted(rtts[1:])
print(f"  WARM rtt: min={warm[0]:.0f} median={warm[len(warm)//2]:.0f} max={warm[-1]:.0f} ms  (the real per-order latency)")
print(f"  body: {body[:160]}")

print("\n=== 2) ORDER object shape: GET /portfolio/orders?limit=1 ===")
code, rtt, body = get("/trade-api/v2/portfolio/orders?limit=1")
print(f"  status={code} rtt={rtt:.0f}ms")
print(f"  order fields: {keys_of(body, 'orders')}")

print("\n=== 3) FILL object shape: GET /portfolio/fills?limit=1 ===")
code, rtt, body = get("/trade-api/v2/portfolio/fills?limit=1")
print(f"  status={code} rtt={rtt:.0f}ms")
print(f"  fill fields: {keys_of(body, 'fills')}")

print(f"\n=== 4) ORDER PLACEMENT (host={HOST}, demo={IS_DEMO}) ===")
# mirrors venue_kalshi.rs map_order: buy YES, 1 contract, IOC limit at <cents>.
ph = "--place-order" in sys.argv
ticker = sys.argv[sys.argv.index(ph_flag := "--place-order") + 1] if ph else "KXBTC15M-EXAMPLE"
cents = int(sys.argv[sys.argv.index(ph_flag) + 2]) if ph else 55
body = {"ticker": ticker, "client_order_id": "probe-" + str(int(time.time())),
        "action": "buy", "side": "yes", "count": 1, "type": "limit",
        "yes_price": cents, "time_in_force": "immediate_or_cancel"}
print(f"  POST https://{HOST}/trade-api/v2/portfolio/orders")
print(f"  body: {json.dumps(body)}")
if not ph:
    print("  (DRY RUN — pass '--place-order <TICKER> <CENTS>' to send; DEMO host only)")
elif not IS_DEMO:
    print("  REFUSED: --place-order is allowed against a DEMO host only (real-money safety).")
else:
    sp = "/trade-api/v2/portfolio/orders"
    ts, sig = sign("POST", sp)
    t0 = time.time()
    conn.request("POST", sp, body=json.dumps(body), headers={
        "KALSHI-ACCESS-KEY": KEY_ID, "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": sig, "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/json"})
    r = conn.getresponse(); resp = r.read().decode(); rtt = (time.time() - t0) * 1000
    print(f"  DEMO order: status={r.status}  rtt={rtt:.0f}ms")
    print(f"  RESPONSE (real shape -> confirms adapter parsing): {resp}")
