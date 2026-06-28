# DESIGN — Multi-Venue Support (Polymarket + Kalshi)

Status: design approved (2026-06-21). Scope: make the trading system venue-generic so
the active prediction market is **chosen by config** — Polymarket *or* Kalshi, one at a
time. Cross-venue/simultaneous trading is explicitly out of scope (see §7).

This complements `DESIGN_EXECUTION.md` (the executor) — read that first for the
SignalIntake → RiskGate → TradeEngine → OrderManager → PositionManager pipeline.

---

## 1. Goal & non-goals

- **Goal.** Run the existing perp-signal → prediction-market strategy against **Kalshi**
  BTC up/down markets instead of Polymarket, selecting the venue via config, with both
  adapters retained (switchable). The reference signal (Binance BTC perp bookTicker) is
  unchanged and venue-independent.
- **Non-goal (this phase).** Trading both venues at once, or cross-venue arbitrage. The
  signal is a single perp move mapped to a single prediction market; there is no
  cross-venue leg. One active prediction venue at a time.

---

## 2. What is already venue-generic (verified)

The collector/bus layer was already designed venue-agnostic, and the **Kalshi collector
already conforms** — it emits the same bus contract as Polymarket. Nothing below changes.

| Layer | Why it is already generic |
|---|---|
| **Bus contract** | Both collectors emit identical `Payload::{Book, Trade, Meta}`. Prices are canonicalized to **[0,1]** (Kalshi parses on-wire cents ÷100). `BookUpdate`/`TradeTick`/`TradeSignal` carry no venue fields. |
| **Instrument string** | Uniform convention `<venue>.<market_id>.<outcome>` — `polymarket.0x<cid>.UP\|DOWN`, `kalshi.<ticker>.YES\|NO`. |
| **Market metadata** | `MarketMeta` already carries per-market `min_order_size` / `tick_size` / `fee_rate` (Polymarket from Gamma; Kalshi hardcoded/`None` → config fallback). |
| **Processor link** | The `InstrumentLinker` matches catalog events on `kind` (`processor.link_kind`), not venue (`crates/processor/src/state.rs:158`). Already works for Kalshi's `"15m_updown"`. |
| **Price gate** | `yes_bucket` operates in [0,1] — valid for both venues. |
| **Venue trait** | `TradingVenue` (submit + async fill stream) is the single market boundary; `SimVenue` is venue-neutral except fee + taker delay (both already config-shaped). |

**Consequence:** this work touches **only the executor** (four spots) plus config/app wiring.

---

## 3. Remaining Polymarket coupling (the work)

| # | Coupling | Location | Fix |
|---|---|---|---|
| 1 | Topic subscription hardcoded `market.polymarket.#` / `.catalog` | `executor/src/module.rs:70,81` | Subscribe to `market.{venue}.#` from config |
| 2 | `cid_of` strips literal `"polymarket."` | `executor/src/types.rs:25` | Generic `market_id_of` (split on first/last `.`) |
| 3 | `traded_instrument` derives complement via `.UP`→`.DOWN` | `executor/src/types.rs:12` | `complement()` via `VenueSpec` outcome pair |
| 4 | Fee `rate·p·(1−p)·qty` + `taker_delay_ms: 250` assume Polymarket | `executor/src/venue.rs:45`, `SimCfg` | Take delay + fee model from `VenueSpec` |

The mirror (`executor/src/mirror.rs:49,87`) keys meta by `cid_of`, so it rides on #2.

---

## 4. Design — the `VenueSpec` descriptor

All per-venue facts live in **one** descriptor, inferred from the instrument prefix, so no
venue string literal appears in the trade path.

```rust
struct VenueSpec {
    prefix: &'static str,   // "polymarket" | "kalshi"  (== bus topic segment)
    yes_label: &'static str,// "UP"         | "YES"
    no_label: &'static str, // "DOWN"       | "NO"
    taker_delay_ms: u64,    // 250          | 0          (Kalshi: no marketable-order delay)
    fee: FeeModel,          // { rate, rounding }  (PM continuous; Kalshi ceil-to-cent)
    maker_rebate: f64,      // PM ~0.2 of pool | Kalshi 0.0
}
```

Generic instrument helpers (replace `cid_of`/`traded_instrument`):

```rust
fn venue_of(inst: &str)     -> &'static VenueSpec   // by first '.' segment
fn market_id_of(inst: &str) -> Option<&str>         // middle segment
fn complement(inst: &str)   -> String               // swap yes/no label via the spec
```

`<venue>.<market_id>.<outcome>` parses by splitting on the **first and last** `.`. Safe for
both venues: Polymarket cids (hex) and Kalshi tickers (`KXBTC15M-240621-14C`, dashes not
dots) contain no internal `.`.

**Location:** a new `executor/src/venue_spec.rs`. (It could later move to `core` if the
processor needs it, but the processor is already kind-generic, so the executor is the only
consumer in this phase.) The registry is a small `const` table keyed by prefix.

---

## 5. Per-component changes (Phase 1 — sim, zero financial risk)

1. **`executor/src/module.rs`** — subscribe to `market.{venue}.#` and
   `market.{venue}.catalog`, where `{venue}` = `executor.venue.market` (the active
   prediction venue). The `signal.#` subscription is already venue-agnostic.
2. **`executor/src/types.rs`** — `cid_of` → `market_id_of` (generic split);
   `traded_instrument` → `complement` via the `VenueSpec` outcome pair. Unit tests cover
   both `polymarket.0xabc.UP↔DOWN` and `kalshi.KX….YES↔NO`.
3. **`executor/src/mirror.rs`** — key `meta`/`winners` by `market_id_of` (one-line swap;
   logic identical). Doc comment generalized from `market.polymarket.#`.
4. **`executor/src/venue.rs` + `SimCfg`** — `taker_delay_ms` and the fee model resolved
   from the `VenueSpec` (Kalshi: delay 0, fee ceil-to-cent, **no maker rebate**). The
   `p·(1−p)` shape is shared, so this is a parameterization, not a rewrite.
5. **Config + app wiring**
   - per-collector `enabled` flag → start only the active venue's collector;
   - `processor.link_kind` set to the active venue's kind (`5m_updown` | `15m_updown`);
   - `executor.venue.market: "kalshi" | "polymarket"` → drives the subscription + spec;
   - `executor.venue.adapter` stays the *real-adapter* selector (`sim` now;
     `kalshi_exchange` / `polymarket_clob` are P2).

**Untouched:** processor rule engine, gate/risk, PositionManager, the near-expiry
liquidator (+ its double-sell race fix), and the hold/chase/maker study probes — all
venue-neutral once #1–#4 land.

---

## 6. Kalshi-specific differences that affect tuning (not architecture)

| Dimension | Polymarket | Kalshi | Impact |
|---|---|---|---|
| Marketable-order delay | 250 ms | **none (0 ms)** — *verified*: no Kalshi speed bump; an order-execution delay was *proposed* at the CFTC (Dec 2025) but **not implemented**. | The *chase* dynamic (250 ms drift) largely vanishes; re-validate `chase_c` (likely ~0). Monitor the proposed delay. |
| Window length | 5 min | **15 min** | Hold/threshold optima are window-dependent; re-run the hold-period probe. Re-tune `min_tte`, `hold_ms`, `force_window`. |
| Fee | `0.07·p·(1−p)`, continuous, taker-only, ~20% maker rebate | **verified** `KXBTC15M`: `fee_type=quadratic`, `multiplier=1` → `round_up(0.07·C·P·(1−P))`, **taker-only, makers FREE, NO rebate**. Round-up is per-order to a centicent with a 1¢-overpayment rebate accumulator → ≈ continuous in effect. | Sim's continuous `0.07·p·(1−p)·qty` ≈ the effective Kalshi taker fee (within ~1¢/order). No maker rebate kills the maker-exit *rebate* case (maker exit could still save the taker fee, but it was already a wash on PM *with* the rebate). |
| Min order size | 5 (Gamma) | 1 (whole contracts) | Already per-market via `MarketMeta`. |
| Settlement | USDC, on-chain | USD, REST | Sim unaffected; only the real adapter (P2) differs. |
| Outcome labels | UP / DOWN | YES / NO | `VenueSpec` outcome pair. |

---

## 7. Phasing

- **Phase 1 (this change).** The §4–§5 generic plumbing. Validate by running the **sim
  executor against the live Kalshi book** (same zero-risk setup used for Polymarket), then
  re-tune the 15-minute params with the existing study probes.
- **Phase 2 (P2, future).** A `KalshiExchange` `TradingVenue` adapter — RSA-signed REST
  order placement + the user fill stream — parallel to the planned `PolymarketClob`,
  selected by `executor.venue.adapter`. The trait already supports it; no further generic
  work needed.

---

## 8. Risks / open items

- **Fee fidelity — RESOLVED.** Verified against the Kalshi API (`/series/KXBTC15M`):
  `fee_type=quadratic`, `fee_multiplier=1` → taker fee `round_up(0.07·C·P·(1−P))`,
  **taker-only**, makers free, **no rebate**. The sim's continuous `0.07·p·(1−p)·qty`
  matches the *effective* fee within ~1¢/order (Kalshi's per-order centicent round-up has
  a 1¢-overpayment rebate accumulator, so it averages back to ≈ continuous). `fee_rate
  0.07` in the Kalshi configs is correct. To stay robust to per-series changes, the Kalshi
  collector could later read `fee_type`/`fee_multiplier` from `/series` into `MarketMeta`.
- **Proposed order-execution delay (monitor).** Kalshi filed a *proposed* order-execution
  delay with the CFTC (~Dec 2025), under review, **not implemented** and with no published
  duration or BTC carve-out. If it ships, set `KALSHI.taker_delay_ms` accordingly — the
  `VenueSpec` already isolates that one number.
- **No taker delay** removes a Polymarket quirk the strategy was tuned around — the entry
  fill is immediate, so entry slippage / chase behavior must be re-measured, not assumed.
- **15-minute windows** change every study optimum (hold, threshold, force-liquidity
  window). Treat all tuned constants as Polymarket-specific until re-run on Kalshi.
- **Single-venue invariant.** The one-active-trade slot and single `VenueSpec` assume one
  venue is live. Running both collectors with a single `link_kind` is safe (only the
  matching kind links), but the inactive collector wastes resources — gate it off.
