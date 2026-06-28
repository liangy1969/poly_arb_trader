# Polymarket US setup (the CFTC-regulated, US-legal venue)

How to set up **Polymarket US** (QCEX LLC — a CFTC-registered Designated Contract
Market) for both manual and **programmatic** trading. This is the **US-legal**
path; it is a *different platform* from the offshore crypto venue in
[SETUP_POLYMARKET.md](SETUP_POLYMARKET.md), which is **non-US only**.

> What's different vs offshore: **no wallet / private key / gas / pUSD / on-chain
> approvals.** Auth is an **Ed25519 API key**, funding is **USD via an FCM/broker**,
> settlement is **USD**, and markets are addressed by **slug** (e.g. `btc-100k`),
> not Polygon token ids. See [POLYMARKET_VENUE_FINDINGS.md](POLYMARKET_VENUE_FINDINGS.md).

> Verify everything against the official docs at <https://docs.polymarket.us> at
> setup time — the US API launched 2026-02-16 and is still young.

---

## 0. The credential model (what you actually guard)

| layer | what it is | secret |
|---|---|---|
| **Account** | KYC'd identity tied to your SSN | n/a (no key) |
| **Funds** | USD held by an approved **FCM** (futures commission merchant) | n/a (bank-rails) |
| **API key** | an **Ed25519** key pair issued in the developer portal | `POLYMARKET_KEY_ID` + `POLYMARKET_SECRET_KEY` |

The API secret is **less catastrophic than a wallet private key** (funds sit with
the FCM; withdrawals go to your linked bank, not anywhere an order can send them),
but it can still place/cancel orders — guard it, and it's **revocable** in the
portal (unlike a wallet key).

---

## 1. Eligibility (check first)

- **US person** with a **Social Security Number** (required for the US API).
- **Eligible state.** Federally legal as a CFTC DCM, but some states (e.g. TN, NV)
  restrict event contracts as gambling — confirm yours is supported.
- An **iOS device** for KYC (the verification flow runs through the iOS app).

---

## 2. Create the account + complete KYC

1. Download the **Polymarket US iOS app** and create an account with **Apple,
   Google, or email**. **Remember which method** — switching sign-in methods later
   can break your API-key access.
2. Complete **KYC/AML**:
   - **SSN**,
   - **Government photo ID** (driver's license, passport, or state ID),
   - **Proof of address** dated within ~90 days (utility bill, bank statement, or
     government letter) matching your details,
   - **Live selfie** if requested.
3. **Timeline:** often instant (automated); manual review 3–5 business days
   (commonly 24–48 h). Support / invite codes: `support@polymarket.us`.

---

## 3. Fund the account (USD via the FCM)

After KYC you're routed to an approved **FCM** that holds your **segregated USD**
and processes deposits/withdrawals through normal banking rails (this is why it's
USD, not crypto).

| Method | Speed / notes |
|---|---|
| **ACH** | free, 1–3 business-day settlement |
| **Debit card** | instant, capped ~$1,000/day |
| **Bank wire** | for larger accounts (≈$25k+) |
| **Apple Pay** | instant |

No Polymarket-charged deposit fees. Deposits show as **instant buying power**, but
**withdrawals require the deposit to fully clear** (~3–4 business days); ACH
withdrawals settle in ~18–48 h.

---

## 4. Get API access (for the bot)

1. Go to the **developer portal: <https://polymarket.us/developer>** and **sign in
   with the same method** you used in the app.
2. **Create a new key** → you receive a **Key ID** and a **Secret Key**.
3. **The secret is shown only once — copy it immediately** into your secrets store
   (§6). If lost or exposed, **revoke** it in the portal and create a new one.

---

## 5. Connect to the API

**Base URLs**
- REST: `https://api.polymarket.us/v1/`
- WebSocket: per docs (public market channel + private user channel); confirm the
  exact URL in <https://docs.polymarket.us>.
- Surface: ~23 REST endpoints + 2 WS endpoints.

**Auth — Ed25519 request signing.** Per endpoint call:

```
message   = "{timestamp}{METHOD}{path}"          # ms epoch + HTTP verb + path
signature = base64( Ed25519_sign(secret_key, message) )
```

Headers:

| Header | Value |
|---|---|
| `X-PM-Access-Key` | your Key ID |
| `X-PM-Timestamp` | current ms since epoch (must be within **30 s** of server time) |
| `X-PM-Signature` | base64 Ed25519 signature of the message |
| `Content-Type` | `application/json` |

> ⚠️ The documented signed message is `timestamp+method+path` and appears **not to
> include the request body**. Before trusting that for `POST` (order) calls,
> confirm against the live docs whether the body must be signed — and prefer the
> official SDK, which handles signing correctly.

**Official SDKs (recommended — don't hand-roll the signing):**

- **Python:** `pip install polymarket-us`
  ```python
  import os
  from polymarket_us import PolymarketUS

  client = PolymarketUS(
      key_id=os.environ["POLYMARKET_KEY_ID"],
      secret_key=os.environ["POLYMARKET_SECRET_KEY"],   # base64 Ed25519 private key
  )

  book = client.markets.book("btc-100k")                 # order book by slug

  order = client.orders.create({
      "marketSlug": "btc-100k-2025",
      "intent": "ORDER_INTENT_BUY_LONG",
      "type":   "ORDER_TYPE_LIMIT",
      "price":  {"value": "0.55", "currency": "USD"},
      "quantity": 100,
      "tif": "TIME_IN_FORCE_GOOD_TILL_CANCEL",
  })
  ```
  Async WebSocket (private user channel):
  ```python
  ws = client.ws.private()
  ws.on("order_update", lambda d: print(d))
  await ws.connect()
  await ws.subscribe("order-sub-1", "SUBSCRIPTION_TYPE_ORDER")
  ```
- **TypeScript:** official `polymarket-us-typescript` SDK.

---

## 6. Store the API secret (reuse the out-of-tree pattern)

Same discipline as the wallet key, applied to `POLYMARKET_SECRET_KEY`:

- Put it in the **out-of-tree secrets file** (`C:\Users\fatli\.poly\secrets.env`,
  ACL-locked) or the **OS keychain** — never in a file inside the repo. The
  `.claude/settings.json` deny rule already blocks tools from reading `secrets*`.
- It's **shown once**; if it leaks, **revoke + reissue** in the portal (a real
  advantage over the wallet key — instant rotation, no funds to drain on-chain).

```bash
# C:\Users\fatli\.poly\secrets.env  (outside the repo)
POLYMARKET_KEY_ID=...                  # UUID
POLYMARKET_SECRET_KEY=...              # base64 Ed25519 private key — guard it
```

---

## 7. Preflight (read-only) before trading

1. **Auth works** — an authenticated read (account balance, or a market book)
   returns 200 (401 = signing/clock issue; check the 30 s timestamp window).
2. **Funds present** — account balance shows your cleared USD buying power.
3. **WS** — connect the private user channel; a clean subscribe means order/fill
   events will stream.
4. Then place a **small limit order**, confirm it appears, and **cancel** it.

---

## 8. What this means for this repo (Rust)

- The offshore [collector](crates/collector-polymarket/src/collector.rs) (public
  `book` WS on `ws-subscriptions-clob.polymarket.com`) and the planned EOA executor
  **do not apply** to Polymarket US — different host, different auth, different
  instruments.
- **There is no official Rust SDK** (Python + TypeScript only). For the Rust
  workspace you either:
  - implement **Ed25519 signing in Rust** (`ed25519-dalek`) against
    `api.polymarket.us/v1/` + the WS channels, or
  - run a **Python `polymarket-us` sidecar** and bridge it to the bus.
- Market data + order semantics (CLOB, WS) carry over conceptually; only the
  **onboarding, funding, auth, and identifiers** change.

---

## 9. Checklist

- [ ] US person, eligible state, SSN ready
- [ ] iOS app account created (note the sign-in method)
- [ ] KYC/AML approved
- [ ] USD funded via FCM (ACH/debit/wire/Apple Pay); buying power visible
- [ ] API key created at polymarket.us/developer; **secret copied once**
- [ ] `POLYMARKET_KEY_ID` / `POLYMARKET_SECRET_KEY` in out-of-tree secrets / keychain
- [ ] Auth preflight 200 (timestamp within 30 s); balance + WS verified
- [ ] Small test order placed + cancelled
- [ ] Decided Rust path: native Ed25519 client vs Python sidecar

> Status: the US API launched 2026-02-16 and is young — confirm base URLs, the
> exact signing scheme (esp. whether POST bodies are signed), endpoint list, and
> rate limits against <https://docs.polymarket.us> at implementation time.

Sources: <https://docs.polymarket.us/api-reference/authentication> ·
<https://github.com/Polymarket/polymarket-us-python> ·
<https://www.quantvps.com/blog/polymarket-us-api-available> ·
<https://agentbets.ai/guides/polymarket-us-api-guide/> ·
<https://www.tradetheoutcome.com/how-the-kyc-process-works-on-polymarket-us-and-required-documents/>
