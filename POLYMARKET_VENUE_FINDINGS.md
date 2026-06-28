# Polymarket venue findings: offshore vs Polymarket US

_Investigation date: 2026-06-20. Not legal/tax advice._

## TL;DR

- The **current implementation targets the offshore (international) Polymarket**
  crypto venue — **not Polymarket US**.
- **Polymarket US (QCEX)** is a **separate CFTC-regulated Designated Contract
  Market** with different hosts, **Ed25519 application-gated auth**, **USD funding
  via an FCM/broker**, mandatory **KYC**, and **its own order books**.
- For a **US person**, the offshore EOA/crypto path is legally restricted; the
  compliant path is **Polymarket US**.
- `SETUP_POLYMARKET.md` (EOA, pUSD, on-chain approvals) documents the **offshore**
  model only.

## 1. What the current code uses (offshore)

From [collector.rs:25-26](crates/collector-polymarket/src/collector.rs#L25-L26):

| Purpose | Endpoint |
|---|---|
| Market metadata | `https://gamma-api.polymarket.com` |
| Order book (WS) | `wss://ws-subscriptions-clob.polymarket.com/ws/market` |

- Subscribes to the **public `book` channel** ([collector.rs:435-439](crates/collector-polymarket/src/collector.rs#L435-L439)) — **unauthenticated, read-only** (no API key, no signing).
- Endpoints are **hardcoded constants** — no config override.
- The (planned, P2) executor/CLOB adapter is also offshore: EOA EIP-712 order
  signing + pUSD collateral.

## 2. Polymarket US (QCEX) — the regulated US venue

- **Entity:** QCEX LLC, CFTC-registered **Designated Contract Market** (launched
  2025-12-03; from Polymarket's ~$112M QCX acquisition).
- **Onboarding:** full **KYC/AML** (government ID, SSN, proof of residency, selfie).
- **Funding:** **USD** via bank/ACH to an associated **FCM/broker**; **USD-settled**.
  No crypto, no EOA, no pUSD.
- **API:** CFTC-regulated **CLOB**; ~23 REST + 2 WS endpoints; **FIX**;
  **Ed25519 auth**; access requires an **application + integration testing**.
  Contacts: `onboarding@polymarket.us`, `support@polymarket.us`, `fix@polymarket.us`.
- **State eligibility:** federally legal as a DCM but restricted in some states
  (e.g., TN, NV treat event contracts as gambling).

## 3. Side-by-side

| | Current code (offshore `.com`) | Polymarket US (QCEX) |
|---|---|---|
| Order-book host | `ws-subscriptions-clob.polymarket.com` | different host (polymarket.us / QCEX) |
| Metadata host | `gamma-api.polymarket.com` | different |
| Auth | none (public `book` channel) | **Ed25519, application-gated** |
| Funding | crypto / EOA / pUSD | **USD via FCM/broker** |
| Settlement | crypto (pUSD) | **USD** |
| KYC | none | **full KYC/AML** |
| Liquidity / prices | offshore venue | **separate DCM venue** |
| US persons | restricted | **the legal path** |

## 4. Implications for this project

- The collector pulls the **offshore** order book — the **wrong venue** if the
  goal is US trading (separate liquidity and prices, not just a different URL).
- To target Polymarket US: repoint hosts, switch to **Ed25519** auth, replace the
  EOA/pUSD funding model with **USD via broker/FCM**, and **apply for API access**.
- The credential to protect changes from a **wallet private key** → an
  **Ed25519 API key** (no on-chain "money key", no gas, no approvals).
- **Geoblocking:** offshore *trading* is blocked for US persons; the offshore
  *data* WS may or may not be reachable from the US.

## 5. US tax (brief)

- Moving your own crypto **Coinbase ↔ your EOA is not taxable**, but it **breaks
  Coinbase's basis chain** (transferred-in = "non-covered" on Form 1099-DA) — you
  track basis yourself.
- **Taxable:** crypto-to-crypto swaps, gas spent in POL, and **Polymarket P&L**.
- **Character is unsettled:** capital gains (property) vs **§1256** (regulated DCM
  contracts) vs gambling income — consult a crypto-savvy CPA.

## 6. Order-book depth comparison

**Result: a like-for-like comparison is _not possible_.** The offshore book is
public and measurable; the Polymarket US book is **auth-gated (HTTP 401)** and
uses **different USD-settled instruments**. Live probe on 2026-06-20:

### Method

- **Offshore:** `GET https://clob.polymarket.com/book?token_id=<id>` (token ids
  from Gamma); depth = Σ(price × size) per side. Point-in-time snapshot.
- **US:** probed candidate hosts for a public book endpoint.

### Offshore (reachable from this US machine, no auth)

Representative two-sided markets (snapshot):

| Market | best bid | best ask | spread | levels (bid/ask) | bid $depth | ask $depth |
|---|---|---|---|---|---|---|
| Will Japan win on 2026-06-21? | 0.69 | 0.70 | 0.01 | 60 / 30 | $1.02M | $2.29M |
| France win 2026 WC | 0.197 | 0.198 | 0.001 | 137 / 198 | $5.69M | $25.5M |
| Netherlands win 2026 WC | 0.058 | 0.059 | 0.001 | 56 / 172 | $0.04M | $26.0M |
| Germany win 2026 WC | 0.053 | 0.054 | 0.001 | 50 / 196 | $0.12M | $25.1M |
| Strait of Hormuz traffic normal | 0.07 | 0.08 | 0.01 | 7 / 83 | $0.05M | $0.41M |

Offshore books are **deep** (≈$1M–$25M+ per side on liquid markets) with **tight
spreads** (0.001–0.01). Longshot/outcome markets are one-sided (huge ask, ~no bid).

### Polymarket US (gated)

| Host probed | Result |
|---|---|
| `api.polymarket.us` | **HTTP 401** — exists, requires auth |
| `clob.polymarket.us` | DNS does not resolve |
| `gamma-api.polymarket.us` | DNS does not resolve |
| `polymarket.us` | HTTP 200 (site) |

The US API exists but **every endpoint is Ed25519 / approved-application gated**
(401 without credentials), and US instruments are **USD-settled contracts with
different identifiers** (not Polygon CTF `token_id`s). So the US book **cannot be
fetched without approved API access**, and an offshore `token_id` is not valid
there.

### Takeaways

- **Offshore market _data_ is not geoblocked** from this US machine (the data
  endpoints responded) — but *trading* offshore is still restricted for US persons.
- A real depth comparison requires **approved Polymarket US API credentials**;
  until then only the offshore venue is measurable.
- The venues are **separate order books**, so offshore depth is **not a proxy** for
  US depth — the regulated US DCM is newer/intermediated and its liquidity must be
  verified directly with US API access.

## Sources

- Polymarket US / CFTC DCM: <https://www.prnewswire.com/news-releases/polymarket-receives-cftc-approval-of-amended-order-of-designation-enabling-intermediated-us-market-access-302625833.html>
- Polymarket US API: <https://www.quantvps.com/blog/polymarket-us-api-available>, <https://www.polymarketexchange.com/developers.html>
- US access / KYC: <https://polymarket.review/us-access.html>
- Offshore CLOB docs: <https://docs.polymarket.com/>
