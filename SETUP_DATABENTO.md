# Setup — Databento account for the BTC reference feed

Goal: get a Databento **API key** + **live access to a BTC dataset**, verify the
symbol/schema stream data, then hand the key to the collector via
`DATABENTO_API_KEY`. The BTC feed is signal-only (never traded), so this is read-only
market data — no trading credentials, no funds.

> Cost note: Databento live data is **paid** (platform fee + per-dataset live fees; CME
> data also carries exchange fees). Confirm the price for your chosen dataset in the portal
> before subscribing — I can't quote current pricing. Start with **one symbol + `mbp-1`** to
> keep it minimal.

---

## 1. Create the account
1. Go to **https://databento.com** → **Sign up**. Use a real email (US data licensing
   asks for org / professional-vs-non-professional status).
2. Complete the profile. For personal research, "non-professional" is usually cheaper —
   answer truthfully; it affects exchange fees.

## 2. Get your API key
1. In the portal: **Settings → API keys** (or **Keys**).
2. Create a key. It looks like **`db-xxxxxxxxxxxxxxxxxxxxxxxxxxxx`**. Treat it as a secret.
3. You'll put it in an env var for the collector (step 6) — **do not commit it to the repo**
   (same handling as the Kalshi key).

## 3. Pick the dataset + symbol (the key decision)
Open **Datasets** in the portal and check what your account is licensed for. Two paths:

- **Preferred — 24/7 crypto spot** (matches Kalshi's hours + settlement): look for a
  **crypto spot** dataset (e.g. a Coinbase BTC-USD feed). If present, note its **dataset
  code** and the **BTC-USD symbol**. This avoids the weekend gap.
- **Fallback — CME BTC futures**: dataset **`GLBX.MDP3`** (CME Globex). Symbols:
  - **`BTC.c.0`** — continuous front-month full BTC future (5 BTC), `stype_in = continuous`
  - **`MBT.c.0`** — micro BTC future (0.1 BTC), cheaper to license, same price signal
  - ⚠️ CME is **closed weekends** (Fri 16:00 → Sun 17:00 CT) + a daily 16:00–17:00 CT break,
    so you'd have **no BTC signal then**. Fine for weekday testing; a real 24/7 strategy
    wants the spot option above.

Schema for either: **`mbp-1`** (top-of-book best bid/ask — the analog of Binance bookTicker).

## 4. Enable Live access
1. Databento separates **Historical** (batch/REST) and **Live** (streaming). The collector
   needs **Live**. In the portal, enable/subscribe **Live** for your chosen dataset.
2. Review the cost shown for **live** on that dataset (platform + data + any CME exchange
   fee). Subscribe to the minimum that covers your symbol + `mbp-1`.

## 5. Verify it works BEFORE building the collector
Test the exact dataset/symbol/schema with the Python SDK (no Rust needed yet):
```bash
pip install databento
```
```python
import databento as db
client = db.Live(key="db-...")              # your key
client.subscribe(
    dataset="GLBX.MDP3",                     # or your crypto-spot dataset code
    schema="mbp-1",
    stype_in="continuous",                   # "raw_symbol" for a spot/explicit symbol
    symbols=["BTC.c.0"],                     # or the BTC-USD spot symbol
)
for record in client:                        # prints live BBO updates
    print(record)                            # look for bid_px / ask_px / ts_event
    break                                    # (remove to keep streaming)
```
If you see live records with `bid_px` / `ask_px` / `ts_event`, the dataset+symbol+schema
are correct and licensed. Note the exact **dataset code**, **symbol**, and **stype_in** that
worked — those go straight into `databento:` config.

(If live errors with an entitlement message, the dataset isn't licensed for live on your
plan — fix in step 4. You can also sanity-check the symbol via a tiny **Historical** query
first, which is cheaper.)

## 6. Hand the key to the collector
Set the key as an environment variable (the collector reads `DATABENTO_API_KEY`, falling
back to the `databento.api_key` config field):
```powershell
# PowerShell (current session)
$env:DATABENTO_API_KEY = "db-..."
# or persist for your user:
setx DATABENTO_API_KEY "db-..."
```
Then in the run config:
```yaml
reference_source: databento
databento:
  instrument: "databento.glbx.BTC"   # canonical id the system uses as processor.reference
  dataset: "GLBX.MDP3"               # or your crypto-spot dataset code
  schema: "mbp-1"
  stype_in: "continuous"             # or "raw_symbol"
  symbols: ["BTC.c.0"]               # or the spot symbol
  # api_key omitted -> read from $DATABENTO_API_KEY
processor:
  reference: "databento.glbx.BTC"    # point the trigger at the new feed
```

---

## What to send me once set up
After step 5 succeeds, tell me:
1. the **dataset code**, **symbol**, and **stype_in** that streamed (e.g. `GLBX.MDP3` /
   `BTC.c.0` / `continuous`), and
2. whether you went **spot (24/7)** or **CME futures (weekday)**.

I'll then implement `crates/collector-databento`, wire the `reference_source` selector,
do a smoke run (verifying the recv−exch latency is now single-digit-to-tens of ms, not the
~133 ms tunnel), and re-run the lead-lag study to confirm the residual lead collapses.
