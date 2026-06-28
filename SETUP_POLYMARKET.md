# Polymarket account & secrets setup

How to provision the account, wallet, and API credentials the **real**
`polymarket_clob` venue adapter (DESIGN_EXECUTION §4/§7/§10, P2) needs. None of
this is required for the **sim** venue (`executor.venue.adapter: sim`), which
places no real orders — provision this only when moving to testnet/mainnet.

> Custody warning: the private key below **is** the money. Anyone with it can
> drain the wallet. It is never committed, never logged, never sent on the bus.
> Keep it in an env var / OS keychain / secrets manager, not in any YAML file.

> **V2 update (since 28 Apr 2026):** Polymarket migrated to **CTF Exchange V2**
> with a new collateral token **pUSD** (an ERC-20 backed 1:1 by USDC) and new
> exchange addresses. The old `py-clob-client` is archived (read-only since
> 11 May 2026) — use **`py-clob-client-v2`**. Old V1 approvals no longer apply.
> The contract addresses, the pUSD wrap step (§2c), and the API-creds method
> name (§3) below all reflect V2. These gate **real funds** — re-verify every
> address against <https://docs.polymarket.com/resources/contracts> and
> PolygonScan at setup time before approving or sending anything.

---

## 0. The two-layer auth model (why there are several secrets)

Polymarket's CLOB uses **two** credential layers — you need both:

| layer | what it is | used for | secret |
|---|---|---|---|
| **L1 — wallet** | an Ethereum (Polygon) private key | EIP-712 **signing** of every order; on-chain pUSD + token custody | `POLY_PRIVATE_KEY` |
| **L2 — API key** | a CLOB API key/secret/passphrase derived *from* the L1 key | authenticating REST/WS calls (place/cancel, `GET /data/trades`, user-channel) | `POLY_API_KEY` / `POLY_API_SECRET` / `POLY_API_PASSPHRASE` |

L1 proves ownership of funds and signs orders; L2 is a lighter, revocable
credential the client uses for ordinary API traffic so the raw key isn't sent on
every request. The L2 triplet is **derived deterministically** from the L1 key
(see §3) — you don't pick it.

One more wrinkle — **the funder (proxy) address**. Polymarket UI accounts trade
through a **proxy wallet** (a smart-contract wallet the UI deploys), not the EOA
of your private key directly. Orders are signed by the EOA but funded from the
proxy. The adapter must know which signature type / funder address to use:

| account origin | `signature_type` | funder address |
|---|---|---|
| **MetaMask / EOA key** you control directly, funded directly | `0` (EOA) | the EOA address (= address of `POLY_PRIVATE_KEY`) |
| **Polymarket UI ("Email/Magic")** proxy wallet | `1` (POLY_PROXY) | the proxy address shown in the UI |
| **Polymarket UI (browser wallet)** proxy | `2` (POLY_GNOSIS_SAFE) | the proxy/safe address |
| **Deposit wallet** (V2 ERC-1271 flow for new API users) | `3` | the deposit-wallet address |

For a clean programmatic setup, prefer a **dedicated EOA** (`signature_type 0`)
you fund directly — it avoids the proxy indirection entirely. Use the proxy
types only if you're trading an existing UI account's balance.

### Why a dedicated EOA — the mechanism (gas, proxies, latency)

**Gas, in one line.** Every state-changing on-chain action on Polygon costs
*gas*, paid in the native token **POL**, by whoever *originates* the transaction.
EVM has two account types: an **EOA** (a keypair — the only thing that can *start*
a transaction, and it pays its own gas) and a **contract account** (code, no key,
can't self-originate). That asymmetry is the whole EOA-vs-proxy story.

- **EOA path (`sig_type 0`).** Your key holds the funds and signs everything. For
  on-chain actions (wrap, the six approvals, redeeming winnings) *you* originate
  the tx and pay POL. You depend on no one — submit straight to any Polygon RPC.
- **Proxy path (`sig_type 1/2/3`).** Funds live in a smart-contract wallet your
  key controls indirectly. You sign an off-chain *message*; Polymarket's
  **Relayer** wraps it into a transaction and **pays the gas** (a Gas-Station-
  Network / meta-tx pattern). Gasless — but adds indirection (funder ≠ signer),
  an extra contract + Relayer dependency, and more failure modes.

**What actually costs an EOA gas — and what doesn't.** The CLOB is
hybrid-decentralised: **placing/cancelling orders is a free off-chain signed
message to the API** (no chain, no gas, no Relayer — true for *every* account
type). Matching is off-chain. Only **settlement** hits the chain, and the
**operator pays that gas** for everyone. So an EOA pays POL only for the one-time
**wrap + approvals** and the occasional **redeem** — never per order.

**Latency.** Because order submission is off-chain for both EOA and proxy, the two
are **latency-equivalent for trading** — with one trap: the **Magic/email proxy**
(`sig_type 1`, and the V2 Magic wallet on `sig_type 3`) custodies your key as an
MPC/TSS *share*, so **every order signature needs a round-trip to Magic/Privy's
signing server** (added latency + known programmatic-signing breakage). A
*self-controlled* key (an EOA, or a Safe/deposit wallet whose key you hold) signs
locally — instant. A protocol-level **~250 ms taker delay** on marketable orders
(§1) floors taker latency for everyone, so this is **not** an HFT venue.

**Decision for this bot:** the EOA is both the **latency-clean** *and* the
**simplest-to-automate** choice (local signing, standard ECDSA, best client
support), at the cost of keeping a little POL (§2b). Choose a proxy only if gasless
is a hard requirement *and* you run a **self-controlled** Safe/deposit wallet —
never the email/Magic account, which is the trap for programmatic trading.

---

## 1. Pick the network first

| | testnet | mainnet |
|---|---|---|
| chain | Polygon **Amoy** | Polygon **mainnet** |
| `chainId` | `80002` | `137` |
| `executor.venue.network` | `testnet` | `mainnet` |
| funds | test USDC from a faucet | **real USDC** |
| liquidity | thin/none | real |
| what it proves | adapter plumbing (signing, L2 auth, FAK/GTD, cancel, reconcile) | real fills, fees, the 250 ms taker delay, slippage |

Do testnet first (DESIGN_EXECUTION §10). It validates the entire adapter with
**zero financial risk**, just no realistic fills.

> Reality check: `py-clob-client-v2` accepts `chain_id=80002` (Amoy), but
> Polymarket does **not** run a public sandbox/CLOB or liquidity on Amoy — there
> is no paper-trading mode. Testnet therefore exercises only *client plumbing*
> (key gen, signing, L2 auth), not real order matching. The practical validation
> path is **mainnet at a `$1` cap** (`max_order_usdc: 1.0`), per §10. Treat
> testnet as a smoke test, not proof of fills.

---

## 2. Create & fund the wallet

**2a. Generate a dedicated EOA** — do not reuse a personal wallet. A fresh key
isolates trading funds and limits the blast radius if it ever leaks. Generate it
**offline** if you can, and never paste the key into shell history, a chat, or any
networked tool.

*Option A — Python (no extra tooling):*

```python
# pip install eth-account
from eth_account import Account
acct = Account.create()
print("address:", acct.address)        # public → POLY_FUNDER_ADDRESS
print("private:", acct.key.hex())      # secret → POLY_PRIVATE_KEY
```

*Option B — `cast` (Foundry).* Install on Ubuntu with
`curl -L https://foundry.paradigm.xyz | bash && foundryup`; on Windows run it
under WSL or Git Bash. Then:

```bash
cast wallet new
# → Address: 0xABC...     Private key: 0x123...
```

Then wire it up:
- Record the **address** (public — this is `POLY_FUNDER_ADDRESS` for
  `signature_type 0`) and the **private key** (secret → `POLY_PRIVATE_KEY`).
- For an EOA the **signer *is* the funder**, so set `POLY_SIGNATURE_TYPE=0` and
  `POLY_FUNDER_ADDRESS` = the address of the key.
- Sanity-check the key derives that address:
  `cast wallet address --private-key 0x...` must print the same address.
- Store both in your secrets manager / `.env` now (see §4) — they're needed from
  §2c onward.

**2b. Get gas (POL).** Polygon's native token POL pays for the *on-chain* steps
only — the wrap (§2c), the six approvals (§2d), and later the occasional **redeem**
of resolved positions. Order placement/cancellation is off-chain and costs **no**
gas (§0), so this is essentially a one-time top-up: each tx is sub-cent, and
**~$1–2 of POL lasts effectively forever**. (A self-controlled proxy via the
Relayer skips this entirely — §0.)

Get POL **over the Polygon network** (not Ethereum) into your EOA — fastest first:

1. **CEX buy + withdraw** (most reliable) — buy POL on Coinbase / Kraken (US) →
   withdraw to your EOA on **Polygon**. Mind the withdrawal minimum (often more
   than the ~$1 you strictly need; a few dollars is fine). Coinbase waives *USDC*
   Polygon withdrawal fees, but a *POL* withdrawal carries a small network fee —
   still pennies.
2. **In-wallet card on-ramp** (quickest) — MetaMask / wallet "Buy" with card or
   Apple Pay drops POL in minutes.
3. **Faucet** (free) — a small live-mainnet drip; see the detailed steps below.
4. **DEX swap** — only once you already hold a little POL (a swap itself needs
   gas), or via a gasless-swap aggregator.

**Detailed: getting free POL (Stakely faucet).** As of 2026, **Stakely is the
only reliable free *mainnet* POL faucet** — the official Polygon faucet is
deprecated, and QuickNode/Alchemy hand out only Amoy *testnet* POL. Steps:

1. Open <https://stakely.io/faucet/polygon-pol> (no sign-up).
2. Paste your EOA address (`POLY_FUNDER_ADDRESS`) and **double-check it** — drips
   go to whatever you enter, with no undo.
3. Solve the captcha and confirm you need gas for real on-chain use.
4. Post the public **X (Twitter)** message it generates, including your request ID.
5. Submit — POL lands at your address shortly after.

> Reality check: the drip is **small — typically ~1–2 transactions' worth**, while
> the full EOA setup is **~8 txs** (1 wrap-approve + 1 wrap + 6 allowance txs).
> Faucets are rate-limited (≈ one drip per period per address/IP), so a single
> request may not cover everything. Either drip again after the cooldown, or just
> buy ~$1–2 of POL (option 1) — the dollar cost is pennies and it's instant. Free
> POL is genuinely free but slow; bought POL is trivially cheap and immediate.

Testnet (Amoy) free POL — for the §1 smoke test only: the Alchemy Amoy faucet
(<https://www.alchemy.com/faucets/polygon-amoy>, requires ~0.001 ETH on L1 to
qualify) or the QuickNode faucet (<https://faucet.quicknode.com/polygon>, requires
an X post, one drip per 12 h).

Confirm it landed before §2c/§2d:
`cast balance <address> --rpc-url https://polygon-rpc.com` (or PolygonScan) should
show a non-zero POL balance.

**2c. Get USDC, then wrap to pUSD.** V2 settles in **pUSD**, not raw USDC, so
this is two parts.

First, get USDC on Polygon into your EOA:
- mainnet: buy USDC on a CEX → withdraw to your EOA **over the Polygon network**
  (not Ethereum), or bridge from another chain, or swap POL→USDC on a Polygon
  DEX. Native USDC (Circle) is `0x3c499c542cEF5E3811e1192ce70d8cc03d5c3359`;
  bridged USDC.e is `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`. The Onramp
  docs describe wrapping **USDC.e** — confirm which token the current
  `CollateralOnramp` accepts before moving size, and test with a few dollars.
- testnet: Amoy test-USDC faucet / bridge (subject to the §1 reality check).

Then wrap USDC → pUSD via the `CollateralOnramp`
(`0x93070a847efEf7F70739046A929D47a521F5B8ee`):
1. `USDC.approve(onramp, amount)`
2. `onramp.wrap(usdcAddress, yourAddress, amount)` — `amount` in 6 decimals.

> Shortcut: depositing through Polymarket's official **Bridge auto-wraps to
> pUSD**, skipping the manual `wrap()` call. The two steps above are the explicit
> path for a pure programmatic EOA.

**2d. Approve allowances (V2)** — the CLOB exchange contracts must be allowed to
move your pUSD and outcome (CTF) tokens. Old V1 approvals **do not carry over**.
Approve **three spenders** for **two token standards** — six one-time on-chain
txs per wallet. `py-clob-client-v2` has **no built-in allowance helper**; do this
with web3.py. Without allowances every order **rejects** (and §9's reject-counter
trips the kill switch — the intended signal that setup is incomplete).

| contract | address | approve via |
|---|---|---|
| CTF Exchange V2 | `0xE111180000d2663C0091e4f400237545B87B996B` | spender |
| Neg Risk CTF Exchange V2 | `0xe2222d279d744050d28e00520010520000310F59` | spender |
| Neg Risk Adapter | `0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296` | spender |
| pUSD collateral (ERC-20) | `0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB` | `approve(spender, MAX)` ×3 |
| Conditional Tokens / CTF (ERC-1155) | `0x4D97DCd97eC945f40cF65F87097ACe5EA0476045` | `setApprovalForAll(spender, true)` ×3 |

```python
import os
from dotenv import load_dotenv                      # pip install python-dotenv
from web3 import Web3
from web3.constants import MAX_INT
from web3.middleware import ExtraDataToPOAMiddleware

load_dotenv()                                        # in-repo .env → POLY_ENV_FILE
load_dotenv(os.environ["POLY_ENV_FILE"])             # out-of-tree secrets (§4)

w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

PRIV = os.environ["POLY_PRIVATE_KEY"]
PUB  = w3.to_checksum_address(os.environ["POLY_FUNDER_ADDRESS"])  # your EOA
CHAIN_ID = 137

PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"   # pUSD collateral (ERC20)
CTF  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"   # Conditional Tokens (ERC1155)
SPENDERS = [
    "0xE111180000d2663C0091e4f400237545B87B996B",     # CTF Exchange V2
    "0xe2222d279d744050d28e00520010520000310F59",     # Neg Risk CTF Exchange V2
    "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",     # Neg Risk Adapter
]

ERC20   = '[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"}]'
ERC1155 = '[{"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"type":"function"}]'

pusd = w3.eth.contract(address=w3.to_checksum_address(PUSD), abi=ERC20)
ctf  = w3.eth.contract(address=w3.to_checksum_address(CTF),  abi=ERC1155)
nonce = w3.eth.get_transaction_count(PUB)

def send(fn):
    global nonce
    tx = fn.build_transaction({"chainId": CHAIN_ID, "from": PUB, "nonce": nonce})
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIV)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    w3.eth.wait_for_transaction_receipt(h, timeout=600)
    nonce += 1
    print("ok", h.hex())

for s in SPENDERS:
    s = w3.to_checksum_address(s)
    send(pusd.functions.approve(s, int(MAX_INT, 0)))   # ERC20 collateral
    send(ctf.functions.setApprovalForAll(s, True))      # ERC1155 positions
```

Requires `web3>=7`. **Verify the addresses against the official contracts page
and PolygonScan, and run a small test trade first** — these approvals authorise
movement of real funds.

> Naked short is mechanically impossible (DESIGN_EXECUTION §15.3): no margin,
> balance+allowance are checked, and the swap reverts if you don't hold the
> tokens. The allowance step is what makes legitimate BUY/SELL go through.

---

## 3. Derive the L2 API credentials

With the L1 key funded, derive (or create) the CLOB API key. Use
**`py-clob-client-v2`** (`pip install py-clob-client-v2`) — the original
`py-clob-client` is archived and won't work against V2 contracts:

```python
import os
from dotenv import load_dotenv                  # pip install python-dotenv
from py_clob_client_v2 import ClobClient

load_dotenv(); load_dotenv(os.environ["POLY_ENV_FILE"])   # in-repo .env → out-of-tree secrets

HOST = "https://clob.polymarket.com"           # confirm V2 host per docs
KEY  = os.environ["POLY_PRIVATE_KEY"]           # L1, from secrets.env
CHAIN_ID = 137                                  # 80002 for Amoy

client = ClobClient(host=HOST, key=KEY, chain_id=CHAIN_ID)
creds = client.create_or_derive_api_key()       # deterministic from the L1 key
print(creds.api_key, creds.api_secret, creds.api_passphrase)
```

> Note the renamed method: V2 uses `create_or_derive_api_key()` (V1 was
> `create_or_derive_api_creds()`).

It is idempotent — same L1 key → same triplet, so you can re-derive instead of
storing if you prefer. These three become `POLY_API_KEY`, `POLY_API_SECRET`,
`POLY_API_PASSPHRASE`.

---

## 4. Where the secrets live (and don't)

The adapter reads secrets from the **environment**, never from `config/*.yaml`
(YAML is for non-secret tuning and is safe to commit). The YAML selects the
*network and funder type*; the env supplies the *secret material*.

```bash
# .env (gitignored) or your secrets manager — never committed
export POLY_PRIVATE_KEY=0x...          # L1 — the money. guard it.
export POLY_API_KEY=...                # L2
export POLY_API_SECRET=...
export POLY_API_PASSPHRASE=...
export POLY_FUNDER_ADDRESS=0x...       # EOA address (sig_type 0) or proxy (1/2/3)
export POLY_SIGNATURE_TYPE=0           # 0 EOA | 1 POLY_PROXY | 2 GNOSIS_SAFE | 3 DEPOSIT_WALLET
```

```yaml
# config/*.yaml — non-secret; selects network + funder mode only
executor:
  venue:
    adapter: polymarket_clob
    network: testnet            # or mainnet
    max_order_usdc: 1.0         # hard cap while validating on mainnet
```

Add `.env`, `*.key`, and any `secrets*` to `.gitignore`. The adapter must
**fail closed** at startup if any required env var is missing — never fall back
to a default key. Recommended guard rails:

- log only the **address** (public) and the **last 4 chars** of the API key,
  never the private key or secret;
- on mainnet, refuse to start if `max_order_usdc` is unset or above your
  validating cap;
- keep a separate key per environment (testnet key ≠ mainnet key).

### This repo's secret layout (kept out of the watched tree)

To keep the private key out of any file an assistant/editor/tool can read, the
real secrets live **outside the repo**; the in-repo `.env` only points at them:

- **`C:\Users\fatli\.poly\secrets.env`** — the real values (private key, funder,
  API creds). ACL-locked to your user, and outside the repo so it is never
  surfaced by the workspace file-watcher.
- **in-repo `.env`** — non-secret pointer only:
  `POLY_ENV_FILE=C:\Users\fatli\.poly\secrets.env`.
- **`.claude/settings.json`** — a `permissions.deny` rule blocks assistant tools
  from reading `.env` / `secrets*` / `*.key` (defense-in-depth).

Helper scripts load both, secrets file last:

```python
import os
from dotenv import load_dotenv     # pip install python-dotenv
load_dotenv()                       # in-repo .env → POLY_ENV_FILE
load_dotenv(os.environ["POLY_ENV_FILE"])   # out-of-tree secrets (real values)
```

---

## 5. Verify before trading

A read-only preflight (no orders) confirms the credentials end-to-end. Confirm
the exact V2 endpoint paths against the current CLOB docs (they changed in the
V2 migration):

1. **L2 auth works** — the open-orders endpoint returns 200 with an empty list
   (it is per-account; an auth failure is 401).
2. **Balances/allowances (check both layers — they answer different questions):**
   - **On-chain = source of truth** (read via a Polygon RPC, *not* Polymarket):
     `balanceOf` on pUSD, `allowance(owner, spender)` on pUSD for the three V2
     spenders, and CTF `isApprovedForAll(owner, operator)` for each — all must be
     non-zero/true.
   - **CLOB cache = what actually gates orders:** `GET /balance-allowance`
     (`get_balance_allowance`) returns the operator's *cached* view. It can desync
     from chain — if on-chain is correct but this shows `0`, call
     `update_balance_allowance` to force a re-sync, then re-check. Orders reject on
     the CLOB's view, not the chain's, so **both must agree** before trading.
3. **Funder matches** — the address the CLOB attributes your orders to equals
   `POLY_FUNDER_ADDRESS`.
4. **WS user channel** — connect the authenticated user channel; a successful
   subscribe (no auth error) means fills will stream.

Only after all four pass should `executor.enabled: true` with
`adapter: polymarket_clob` be set — and even then start on **testnet**, then
mainnet at `$1` sizes (DESIGN_EXECUTION §10).

---

## 6. Checklist

- [ ] `py-clob-client-v2` installed (not the archived `py-clob-client`)
- [ ] Dedicated EOA generated; address + private key recorded securely
- [ ] Gas (POL) funded on the target chain
- [ ] USDC funded (correct token/network)
- [ ] USDC wrapped to **pUSD** (via `CollateralOnramp` or the Bridge auto-wrap)
- [ ] Allowances approved for **V2** (pUSD ERC-20 + CTF ERC-1155 × 3 spenders)
- [ ] L2 API creds derived (`create_or_derive_api_key`)
- [ ] Secrets in env / secrets manager (never in YAML/git)
- [ ] `signature_type` + funder address set to match the account origin
- [ ] Read-only preflight (§5) all green
- [ ] mainnet smoke at `$1` cap (testnet has no real CLOB — see §1)

> Status: this document is the prerequisite for the P2 `polymarket_clob`
> adapter, updated for **CTF Exchange V2** (28 Apr 2026). The exact V2 CLOB host,
> the pUSD/Onramp and exchange addresses, and the per-market `neg_risk` / fee
> flags are confirmed against live docs at implementation time
> (<https://docs.polymarket.com/resources/contracts>, DESIGN_EXECUTION
> §15.2/§15.7). Contract addresses here gate real funds — re-verify on PolygonScan
> before approving.
