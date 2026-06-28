# DESIGN — Trade Execution & Position Management

Status: draft for review. Detail design of **§7 Trade Executor** of
`DESIGN_TRADING_SYSTEM.md` — covers trade construction, order types, entry
**unfill processing**, **exit execution**, the order/trade state machines, and
the **Position Manager**. Everything here implements the semantics of the
validated backtest (`e:\poly\crypto_collector\arb_study\backtest.py`); the
mapping is made explicit in §1.

---

## 0. Scope & vocabulary

- **Trade** — one round trip: entry leg → hold → exit leg (+ settlement
  fallback). The unit the backtest counts. At most **one active Trade** at a
  time (v1, matching the backtest's `last_exit` gate).
- **Order** — one venue order attempt. A Trade owns 1..N orders (entry
  attempts + exit attempts).
- **Traded token** — the executor never sells short. Signal direction maps to
  a token to **buy**:

  | signal `direction` | traded instrument | entry side | exit side |
  |---|---|---|---|
  | `+1` (perp up-tick) | `polymarket.<cid>.UP` | BUY | SELL |
  | `-1` (perp down-tick) | `polymarket.<cid>.DOWN` | BUY | SELL |

  Buying DOWN ≡ selling UP (complementary books; the CLOB matching engine
  crosses complementary orders). This sidesteps "can't sell what you don't
  hold" and keeps both legs plain BUY-then-SELL. CTF split/merge (on-chain) is
  **not** used — too slow for a 5-minute market.

All prices are USDC per share in `[0,1]` (f64) in core; cents only in
analytics output. The venue adapter converts to exact decimal strings at the
boundary (§4.3).

**Units convention.** Anything suffixed `_ns` (and the symbol `now`) is an
**absolute nanosecond** wall-clock stamp (`i64`, the bus `ts_ns` clock).
Anything suffixed `_ms` (and all config knobs) is a **millisecond duration**.
The two never mix implicitly: whenever a timestamp delta is compared against a
duration, the duration is promoted to ns via `MS = 1_000_000` (`ns/ms`). The
expressions below are written explicitly (`stale_ms·MS`) so the conversion is
never lost in implementation. `tte` is the lone duration the linker exposes
directly in ms (`MarketState::tte_ms`), so tte-vs-`_ms` comparisons stay in ms.

---

## 1. Backtest ↔ executor parameter mapping

The executor is the live re-implementation of `backtest()`. Every gate there
has exactly one counterpart here:

| backtest (`backtest.py`) | executor |
|---|---|
| `\|prev_bps\| ≥ θ` signal | `PerpMoveRule.threshold_bps` (processor, upstream) |
| entry = first YES tick after signal+latency | entry attempt against live book, immediately on signal |
| `MAX_ENTRY_GAP_S = 1.0` (skip if no tick ≤ 1 s) | `entry.ttl_ms = 1000` — abandon entry, never chase past it |
| `5 ≤ ymid ≤ 95` | `yes_bucket` re-checked **at execution time** on the traded token |
| entry price = `e_ask` (cross, top of book) | entry = marketable limit **FAK** at `best_ask` (§5) |
| hold `H` from entry tick | `hold_ms` timer from **first entry fill** ts (§6.1) |
| exit price = `x_bid` (cross) | exit = marketable limit **FAK** at `best_bid`, escalation ladder on misses (§6) |
| forced exit at market block end | exit deadline = `expiry_ts_ns − deadline_buffer_ms·MS`; then hold-to-resolution (§6.4) |
| one position at a time (`st < last_exit` skip) | PositionManager one-trade gate + `cooldown_ms` (§8.4) |
| top-of-book size assumed available | size capped at `depth_frac × top size` (§3.2) |
| cents/contract accounting | `TradeRecord` P&L fields (§11) |

> **Economics gate (important).** The backtest's `cross` P&L is net of the YES
> spread but **not** net of Polymarket taker fees. The short-term crypto
> up/down markets are exactly the segment where Polymarket applies/changes
> taker fees; `fee_rate_bps` must be read **from market metadata at discovery**
> (catalog event, §11), never assumed zero. **On a BUY the taker fee is taken
> in shares** → received `qty < ordered size` (§3.2, §8.2). The Python backtest
> must net this out (§10, §15 P0). If `expected_edge_c − fee_c(p) <
> min_net_edge_c`, the risk gate rejects the signal (§9) — this can disable
> whole price buckets.

> **Latency floor (important — verified live, Jun 2026).** Polymarket applies a
> **250 ms taker delay** to marketable orders on crypto/finance up/down markets
> — *our* exact market type. Every FAK (entry **and** each exit rung) is held
> ~250 ms before validation re-runs and it matches/rests, and **it cannot be
> cancelled inside that window**. This is a *mandatory* floor on top of
> network+compute latency, so realistic signal→match is ~300–400 ms, not the
> optimistic backtest cells. The backtest's latency-decay (Part A) shows the
> edge shrinks with latency, so this is potentially **edge-determining**:
> **re-run the economics with latency ≥ 250 ms before any live switch.** Detect
> per market via `GET /clob-markets/{cid}` (taker-delay flag, §15); fold it into
> the backtest re-run and the latency budget (§10, main doc §10).

---

## 2. Executor internal architecture

One executor module (tokio task group). Components are single-owner; they
exchange plain function calls / mpsc inside the module, and only bus events
outside it (share-nothing across modules is preserved).

```
        signal.<strategy>            market.polymarket.#        market.polymarket.catalog
              │ sub (Block)                │ sub (Conflate{instrument})      │ sub (Block)
              ▼                            ▼                                 ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ EXECUTOR                                                                               │
│                                                                                        │
│  SignalIntake ──▶ RiskGate ──▶ TradeEngine  (≤ 1 active Trade; state machine §2.1)     │
│                      ▲              │   ▲                                              │
│                      │ reads        │   │ exit-due / deadline timers                   │
│                PositionManager ◀────┤   └── ExitScheduler (sleep_until)                │
│                  (§8)  ▲            │ order intents                                    │
│                        │ fills      ▼                                                  │
│                        └──── OrderManager (§7) ──▶ TradingVenue: PolymarketClob       │
│                                     ▲                       │   (testnet | mainnet)    │
│  ExecBookMirror (§2.2) ─────────────┴── prices live orders ─┘                          │
└──────────────│─────────────────────────────────────────────────────────────────────────┘
               ▼ publish
   exec.report   exec.fill   exec.position   exec.trade        (port exec.ctl: kill/flatten)
```

- **SignalIntake** — drains `signal.*`; drops signals older than
  `signal.ttl_ms` on arrival (stale-by-queueing protection; the sub policy is
  `Block`, so this is the only place a signal can die silently — it is still
  counted + logged).
- **RiskGate** — pure pre-trade checks (§9). No I/O.
- **TradeEngine** — owns the active `Trade` state machine (§2.1); the only
  component that creates order intents.
- **OrderManager** — per-order lifecycle, idempotent ids, ack timeouts,
  reconciliation (§7).
- **ExitScheduler** — arms `sleep_until(first_entry_fill_ts_ns + hold_ms·MS)`
  and `sleep_until(exit_deadline_ns)`; fires events back into TradeEngine.
- **PositionManager** — positions, cash, P&L, settlement, the one-trade gate
  (§8). Consumes fills from OrderManager, resolution from catalog events.
- **ExecBookMirror** — executor-local top-N book per *currently relevant*
  token (the live market's UP/DOWN), built from the same `market.polymarket.#`
  events the processor consumes. Live orders must be priced off a current book
  (entry cap = `best_ask`, exit floor = `best_bid`), and sizing reads its depth
  (§3.2). This is a derived projection like the processor's `MarketState` —
  **not** shared
  memory; the main doc's "executor subscribes `signal.*`" is extended to
  "+ `market.polymarket.#` (conflated) + catalog" (Δ to §2/§9 of the main doc,
  listed in §12 here).

### 2.1 Trade state machine

```
          ┌──────────────────────────── cooldown_ms ────────────────────────────┐
          ▼                                                                     │
   ┌──────┐  signal   ┌──────────┐  any fill   ┌─────────┐  exit due   ┌─────────┐  flat   ┌────────┐
   │ Idle │ ────────▶ │ Entering │ ──────────▶ │ Holding │ ──────────▶ │ Exiting │ ──────▶ │ Closed │
   └──────┘  (gates   └──────────┘             └─────────┘  (timer or  └─────────┘         └────────┘
       ▲      pass)        │                                 forced §6.3)   │
       │                   │ no fill within entry.ttl_ms                    │ deadline hit, qty > 0
       │                   ▼                                                ▼
       │              ┌───────────┐                              ┌───────────────────┐ resolved ┌─────────┐
       └──────────────│ Abandoned │                              │ PendingResolution │ ───────▶ │ Settled │
        (cooldown)    └───────────┘                              └─────(§6.4, §8.3)──┘          └─────────┘
```

| state | meaning | exits |
|---|---|---|
| `Idle` | no trade; signals evaluated | gates pass → `Entering` |
| `Entering` | entry FAK attempt loop running (§5) | first fill → `Holding`; ttl/abandon → `Abandoned` |
| `Holding` | `filled_qty > 0`; exit timer armed | timer or forced trigger → `Exiting` |
| `Exiting` | exit attempt ladder running (§6) | flat → `Closed`; deadline & `qty > 0` → `PendingResolution` |
| `Closed` | flat; `TradeRecord` emitted; cooldown starts | → `Idle` |
| `Abandoned` | entry never filled; no position ever existed | → `Idle` (cooldown still applies — the signal was consumed) |
| `PendingResolution` | still long at exit deadline; await market resolution | catalog `resolved` → `Settled` → `Idle`. **Alert-level event**, not a normal path. |

Every transition publishes an `exec.report` (and `exec.trade` on terminal
states) and is journaled (§8.5) **before** the next order intent is issued —
crash between transitions recovers deterministically.

---

## 3. From `TradeSignal` to `ExecutionPlan`

### 3.1 Plan construction (TradeEngine, on gated signal)

```rust
pub struct TradePlan {
    pub trade_id: String,           // "{strategy}-{signal_ts_ns}" — deterministic
    pub instrument: String,         // UP or DOWN per direction (§0 table)
    pub token_id: String,           // CLOB asset id (from catalog meta)
    pub size_shares: f64,           // §3.2
    pub entry: EntryPolicy,         // FAK ladder params (§5)
    pub hold_ms: u64,               // from signal.plan
    pub exit: ExitPolicy,           // Cross{ladder} | Passive{wait→ladder} (§6)
    pub exit_deadline_ns: i64,      // expiry_ts_ns − deadline_buffer_ms·MS (§6.4)
    pub signal_snapshot: SignalCtx, // signal ts, trigger, book at decision time (analytics)
}
```

Execution-time re-validation (the book has moved since the rule fired —
microseconds in-proc, but re-check anyway; identical checks protect replay):

1. target token book fresh: `now − mirror.recv_ts_ns ≤ stale_ms·MS`;
2. traded-token `best_ask ∈ yes_bucket` (bucket is symmetric so it applies to
   either token's own price);
3. `tte_ms ≥ min_tte_ms` where `min_tte_ms ≥ hold_ms + exit_window_ms +
   deadline_buffer_ms` (a position must never be *designed* to straddle
   resolution; `tte_ms = (target.expiry_ts_ns − now) / MS`);
4. `now − signal.ts_ns ≤ signal.ttl_ms·MS`;
5. edge-consumption: `best_ask − signal.trigger.yes_price ≤ max_adverse_c`
   (price already ran → the move we trade is spent → abandon).

Any failure → reject with reason, `exec.report{Rejected}`, no state change.

### 3.2 Sizing

v1 = fixed target size, capped by displayed liquidity (the backtest only ever
assumed top-of-book):

```
size = min(cfg.size_shares, depth_frac × top_ask_size(traded token))
if size < venue.min_order_size → abandon (no trade)        // min size from market meta
```

No averaging-in, no pyramiding. Notional cap is enforced by the risk gate
(§9), not by sizing.

> **Fee-in-shares (BUY).** Polymarket charges the taker fee *in shares* on a
> buy, so a 20-share order credits **< 20** shares (verified on the live
> 15-minute crypto markets — the root of py-clob-client #245's "can't sell the
> full size"). Two consequences: (a) the position is whatever **actually
> filled**, so the exit sizes from the held `qty` — never the ordered size
> (§6, §8.2); (b) `size_shares` is a *gross* target — net exposure is slightly
> lower. The held `qty` is read from the fill events / a balance query, never
> assumed equal to the request.

---

## 4. Order types & venue mapping

### 4.1 Logical order intents → Polymarket CLOB

The CLOB is limit-only; "market" orders are marketable limits with an
immediate time-in-force. We use exactly three intent types:

| intent | use | CLOB mapping | rationale |
|---|---|---|---|
| **TakeNow** (marketable limit, IOC) | entry; exit ladder steps | limit order, `FAK` (fill-and-kill) | immediate, accepts partials, leaves **nothing resting**; price field = worst acceptable price (cap/floor), bounding slippage. Never naked-market: the book can move between decision and arrival. |
| **RestUntil** (passive, expiring) | passive exit mode only (§6.2) | limit order, `GTD` (expiry = escalation time + venue clock skew margin) | self-cancelling — if the executor dies, the order dies with it. Plain `GTC` is forbidden in v1 (an orphan GTC on a 5-minute binary market is an unbounded risk). |
| **Cancel** | passive→cross escalation; flatten | cancel by order id | — |

`FOK` is **not** used: all-or-nothing forfeits the partial fills the strategy
happily takes (backtest is per-share P&L; a partial is just a smaller trade).

### 4.2 Order identity & idempotency

```
client_id = "{trade_id}:{leg}:{attempt}"        // OUR handle; leg ∈ {E, X}, attempt = 0,1,2…
```

`client_id` is our application id, unique per attempt. The **`venue_order_id`
is taken from the POST `/order` response** — *not* computed locally. On the ack
the OrderManager records the `client_id ↔ venue_order_id` pair (journaled), and
that recorded pair is the authoritative link for the rest of the order's life.
We deliberately **do not** depend on reproducing the venue's EIP-712 order hash
(that chain — orderID = the hash, byte-exact field reproduction, neg-risk
domain, amount-rounding parity — is unverified and brittle); the signing salt
may be the client lib's default (random).

*Idempotency is structural, not id-based.* **One order in flight per leg** +
**reconcile-before-resubmit** (§7) means we never blindly re-POST, so there is
no duplicate-order risk an id scheme would need to dedupe. The only case with no
returned id is a **lost ack** (POST sent, response dropped). Reconciliation then
matches the single in-flight order against the per-user open-orders /
recent-trades feed on `(token, side, price, size)` within the attempt window and
**adopts the venue's id from the match** (§7, §8.6). Under the one-in-flight
invariant that match is unambiguous — a future multi-order-in-flight mode (P3)
is the only thing that would force revisiting this.

### 4.3 Numeric boundary rules

Market metadata supplies `tick_size` (0.01, or 0.001 in the price extremes)
and `min_order_size`. Conversion f64 → exact decimal at the adapter, rounding
**conservatively**: BUY cap rounds *down* to tick, SELL floor rounds *up* —
never round into a worse price than the policy allows. Sizes round down to
the size step. A rounded-to-zero order → abandon attempt.

---

## 5. Entry execution & unfill processing

Entry is a bounded **attempt loop** of TakeNow orders. Design stance:
**abandon, don't chase** — the edge is a ~1–2 s lag; paying up erodes exactly
the cents being captured.

```
on Entering:
  attempt = 0
  loop:
    book = mirror(traded_token);  re-run gates §3.1 (2)–(5)        // any fail → Abandon
    cap  = book.best_ask + entry.chase_c                            // default chase_c = 0
    if cap − signal_ask > entry.max_chase_total_c → Abandon
    submit TakeNow{BUY, price=cap, size=remaining}                  // FAK
    ── FAK resolves immediately (full / partial / none) ──
    if filled_any → record fills → HOLDING (clock = first fill ts)  // partial: see below
    attempt += 1
    if attempt ≥ entry.max_attempts            → Abandon
    if now − signal.ts_ns ≥ entry.ttl_ms·MS    → Abandon            // = MAX_ENTRY_GAP analog
    sleep entry.retry_delay_ms                                      // let the book refresh
```

**Unfill outcomes:**

| outcome | handling |
|---|---|
| **No fill** (book moved above cap before arrival) | retry per loop above; bounded by `max_attempts` (default 3) ∧ `ttl_ms` (default 1000) ∧ `max_chase_total_c`. Then `Abandoned` — emit `exec.report{Expired}`, `exec.trade{abandoned}`, cooldown. A missed entry is a non-event economically; it must never become a worse entry. |
| **Partial fill** | **accept and proceed** — transition to `Holding` with `filled_qty`; do *not* top up (a second buy at a later ts has a different, worse expected edge and smears the entry timestamp the hold timer is anchored to). FAK guarantees the remainder is dead at the venue — nothing to cancel. |
| **Reject** (balance/allowance/min-size/rate-limit) | `Abandoned` + reject-counter; `N` consecutive rejects (default 3) trips the kill switch (§9) — rejects mean our model of the venue/account is wrong. |
| **Ack timeout / unknown** | freeze the loop; OrderManager reconciles (§7.2). If the order turns out filled → `Holding` (hold clock = venue fill ts); if dead → resume loop if ttl allows. Never submit attempt *k+1* while attempt *k* is unknown — that is how double positions happen. |

---

## 6. Exit execution

The exit leg is the risk-critical leg: entry can be abandoned, **exit
cannot** — once long, the only outcomes are sold or resolved. Three layers:
the scheduled exit, the escalation ladder, and the resolution fallback.

### 6.1 Scheduling

`exit_due_ns = first_entry_fill_ts_ns + hold_ms·MS` (anchored at the *first* fill —
mirrors the backtest's entry-tick anchor; partials don't extend the hold).
ExitScheduler also always arms the independent **deadline timer**
`exit_deadline_ns = expiry_ts_ns − deadline_buffer_ms·MS` at Holding-entry; whichever
fires first moves the trade to `Exiting`. (`min_tte` gating §3.1 normally
makes the hold timer fire first by construction.)

### 6.2 Modes

**`cross` (default — what the backtest validated):**

```
on Exiting:                       // remaining = current position qty
  k = 0
  loop every exit.retry_interval_ms (first iteration immediately):
    book  = mirror(traded_token)               // stale book: see §6.3(c)
    floor = max(book.best_bid − k·exit.step_c,  book.best_bid − exit.max_slip_c,  tick)
    submit TakeNow{SELL, price=floor, size=remaining}     // FAK; deeper floor sweeps levels
    remaining -= filled;  if remaining == 0 → Closed
    k += 1                                     // ladder: bid, bid−1c, bid−2c, … capped
    if now ≥ exit_deadline_ns → PendingResolution (§6.4)
```

Attempt 0 is exactly the backtest's `x_bid` exit. The ladder only deepens on
*misses* (thin/moving book) — in the common case the first FAK clears at the
bid and the ladder never engages.

> **Taker-delay constraint (§1 latency floor).** On these markets each FAK is
> held ~250 ms before it resolves, and **cannot be cancelled in that window**.
> So `retry_interval_ms` must be **≥ the taker delay** (a rung can't resolve
> faster than ~250 ms anyway — `250` is a floor, not a free parameter), and the
> per-rung result isn't known for ~250 ms. Budget the exit window accordingly:
> with ~2–3 rungs each ~250 ms+, `cross_window_ms` should allow ~1 s, well
> inside `deadline − exit_due`. For **passive**, the no-cancel-in-window rule
> means the GTD cancel-confirm before falling through to cross can itself lag —
> size `passive_wait_ms` so the cancel resolves before the cross ladder needs
> to start.

**`passive` (config option; P3 to enable by default — unvalidated by the
backtest):** at exit-due, place `RestUntil{SELL, price = best_ask − tick}`
(improve the offer, first in queue) with GTD expiry = `now +
passive_wait_ms·MS`. On fill → Closed. On expiry/timeout → cancel-confirm →
fall through to the **cross ladder** above. Passive wins ~the spread when it
works; the ladder bounds the cost when it doesn't. `passive_wait_ms` must
satisfy `(passive_wait_ms + cross_window_ms)·MS ≤ exit_deadline_ns − exit_due_ns`.

### 6.3 Forced-exit triggers (any of these moves Holding → Exiting immediately)

a. **deadline proximity** — deadline timer fired before hold timer (clock
   slippage, late fill);
b. **kill switch / `exec.ctl flatten`** — operational flatten;
c. **stale target book while holding** (`now − mirror.recv_ts_ns > stale_ms·MS`) —
   exit *anyway*: quote from the last known book with the floor widened by
   `stale_extra_slip_c`, and alert. Holding a 5-minute binary blind is worse
   than a bad exit print;
d. **catalog `expired` arrives early** while still Holding — skip straight to
   the ladder with whatever book remains (books typically get pulled around
   expiry; this is the last-resort window).

No take-profit / stop-loss in v1 — the backtest edge is a pure time exit;
adding path-dependent exits changes the measured distribution (P3 research).

### 6.4 Terminal fallback: hold to resolution

If the deadline passes with `remaining > 0` (book emptied / venue down):

- stop quoting (post-deadline books on these markets are noise);
- trade → `PendingResolution`; position → `PendingResolution` in the PM;
- **alert** — this is an incident-grade event: a mid-bucket entry (~50c) held
  to resolution is a ±50c coin flip against a +1–2c designed edge. The
  `min_tte` + deadline-buffer parameters exist to make this rare; its
  frequency is a tracked health metric (> X/day → halt and re-examine);
- settlement: on catalog `resolved {winner}` the PM books `qty × (1 if
  instrument == winner else 0)` (§8.3). Live: redemption is a separate
  on-chain `redeemPositions` step (P2 — auto-redeem task; cash reconciles
  after the tx).

---

## 7. Order state machine (OrderManager)

```
 Created ──submit──▶ Submitted ──venue ack──▶ Open ──▶ PartFilled ──▶ Filled
     │                   │                     │            │
     │                   │ ack timeout         │ cancel     │ FAK auto-kill remainder
     │                   ▼                     ▼            ▼
     └─ pre-send fail   Unknown ──reconcile──▶ {Filled | Dead}      Dead = Cancelled|Rejected|Expired
        → Dead
```

- **Live fills (per-user, authenticated):** all fill tracking uses the
  **account-scoped** channels. Primary = CLOB **user-channel** WS (order +
  trade events, real-time). Pull fallback/backfill = CLOB
  **`GET /data/trades`** (L2-auth; returns *only this account's* trades, the
  address implicit in the credentials). Neither is the **public** trade tape
  (`data-api.polymarket.com/trades`, across-users) — that carries no
  fees/cost-basis and is *not* used for fills; the public market channel feeds
  only the collector's book (main doc §5). Fills dedupe by
  `(order_id, venue_trade_id)` — WS+pull overlap must not double-apply (PM
  applies idempotently, §8.2). Cadence, id mapping, and edge cases: **§8.6**.
- **Ack timeout (`place` no response in `ack_timeout_ms`, default 2000):**
  state → `Unknown`. Reconcile by **per-user query** — `GET /data/orders`
  (open) + `GET /data/trades` (fills), filtered to this token and matched on
  `(side, size, price)` within the attempt window (unambiguous — only one order
  was in flight). Found-open → **adopt its `venue_order_id`**, treat as Open
  (then cancel if the intent was FAK-immediate); found-filled → adopt id +
  apply fills; not-found → Dead. **No new order on this leg until Unknown
  resolves** (§5) — this invariant, not an id scheme, is what prevents a double
  order.
- **Cancel is not done until confirmed:** cancel-requested orders stay
  non-terminal until the venue confirms (or reconcile proves them dead);
  passive→cross escalation waits for that confirmation before crossing
  (otherwise a late passive fill + a cross fill = double sell → short).

---

## 8. Position Manager

Single authority on **what we hold and what it's worth**. Lives inside the
executor crate but is bus-shaped: inputs are fills + catalog events + marks,
outputs are `exec.position` — it can be lifted out to its own module later
without code surgery.

### 8.1 Data model

```rust
pub struct PositionManager {
    cash_usdc: f64,                        // USDC balance, reconciled from the venue (§8.5)
    positions: HashMap<String, Position>,  // instrument → position (≤ 1 nonzero in v1)
    active_trade: Option<TradeId>,         // the one-trade gate
    cooldown_until_ns: i64,
    daily: DailyStats,                     // realized, fees, n_trades, max_drawdown; UTC reset
    last_trade_cursor: TradeCursor,        // GET /data/trades high-water mark (§8.6); journaled
    seen_fills: HashMap<VenueTradeId, FillStatus>, // dedup + provisional/confirmed tracking (§8.2)
}

pub struct Position {
    pub instrument: String,           // "polymarket.<cid>.UP" | ".DOWN"
    pub token_id: String,
    pub qty: f64,                     // shares, ≥ 0 — LONG-ONLY invariant (§8.2)
    pub avg_cost: f64,                // USDC/share, fee-inclusive
    pub realized_pnl: f64,            // net of fees
    pub fees_paid: f64,
    pub mark_px: f64,                 // conservative: exit-side best bid (config: bid|mid)
    pub upnl: f64,                    // qty × (mark_px − avg_cost)
    pub status: PosStatus,            // Open | PendingResolution | Settled
    pub expiry_ts_ns: i64,
}
```

### 8.2 Fill application (idempotent, status-aware, long-only)

Live trades settle **on-chain**, so a fill is *provisional* until final. Every
fill carries `status ∈ {matched, confirmed, failed}` (§8.6) and a
`venue_trade_id` (the venue's fill-record id — **not** our round-trip
`trade_id`; see the id ladder in §8.6). `apply_fill` keys on `venue_trade_id`
and is both **idempotent** (same id+status replays to a no-op) and
**reversible** (a later `failed` backs out an earlier `matched`). It applies the
position math **exactly once** per fill, on whichever of `matched`/`confirmed`
arrives first:

For a BUY, `f.qty` is the **shares actually received** — i.e. fee-reduced, since
the taker fee is charged in shares (§3.2) — *not* the ordered size. So `qty`
always reflects the real holding, and the exit sizes from it.

```
apply_fill(f):                  // f: venue_trade_id, order_id, status, side, qty(received), px, fee
  prev = seen_fills.get(f.venue_trade_id)      // None | matched | confirmed | failed
  if prev == f.status: return                  // exact replay → no-op
  applied = prev ∈ {matched, confirmed}
  match (f.status, applied):
    (matched | confirmed, false):              // first application, via either status
      BUY :  avg_cost = (qty·avg_cost + f.qty·f.px + f.fee)/(qty+f.qty);  qty += f.qty
      SELL:  assert f.qty ≤ qty + ε;   realized += f.qty·(f.px − avg_cost) − f.fee;  qty −= f.qty
      cash:  BUY −(f.qty·f.px + f.fee);  SELL +(f.qty·f.px − f.fee)
    (confirmed, true):  pass                   // already applied at `matched` → just finalize status
    (failed,    true):  undo the BUY/SELL math above for f.qty    // reverse the provisional fill
    (failed,    false): pass                    // never applied → nothing to reverse
  seen_fills[f.venue_trade_id] = f.status;  emit exec.position
```

- **matched** → apply optimistically (for risk/position you *are* holding it),
  flagged pending-confirm;
- **confirmed** → finalize (realized P&L is now settled, not provisional) —
  no re-application of qty/cash;
- **failed** → reverse the provisional application (rare: stale maker / reverted
  tx; a FAK against a live book almost always confirms).

The long-only `assert` is a **hard invariant**: a violation means a
double-applied or phantom fill → kill switch + reconcile, never "go negative".

### 8.3 Resolution settlement

On catalog `resolved {cid, winner}` with a nonzero position in that market:
`payout_px = 1.0` if `instrument`'s outcome == winner else `0.0`;
`realized += qty × (payout_px − avg_cost)`; `cash += qty × payout_px`;
`qty = 0`; status → `Settled`; emit `exec.position` + close the
PendingResolution trade with a `TradeRecord{settled}`.

**Required collector extension (Δ to main doc §5):** the lifecycle scheduler
currently drops a market ~10 s after expiry. Add a post-`Expired` step: poll
Gamma for resolution (`outcomePrices` / UMA status) every `resolve_poll_s`
(2 s) up to `resolve_timeout_s` (600 s), publish
`market.polymarket.catalog {status: resolved, winner: UP|DOWN}`, then drop.
Cheap (one market at a time, REST poll, off hot path) and it is the *only*
input the PM needs for settlement. Poll may be skipped when the executor
reports no position — but v1 keeps it unconditional (it also timestamps
resolution latency for research). Timeout without resolution → alert, manual
handling (live funds are safe — redemption is available whenever UMA
finalizes).

### 8.4 The one-trade gate & cooldown

`try_begin_trade(trade_id)` atomically checks
`active_trade == None ∧ all qty == 0 ∧ now ≥ cooldown_until_ns` and claims the
slot — called by the RiskGate, this *is* the structural one-position-at-a-time
guarantee (the rule-level `cooldown_ms` upstream is advisory; this one is
load-bearing). `end_trade()` (Closed/Abandoned/Settled) sets
`cooldown_until_ns = now + cooldown_ms·MS` and frees the slot.

### 8.5 Persistence, restart & reconciliation

- **Journal:** every transition/fill is appended (sqlite via the storage
  crate, off-path bounded channel) *in addition to* the bus Recorder. The
  journal is the PM's recovery source (incl. `last_trade_cursor`, §8.6); the
  Recorder stream is for research/replay.
- **Restart (executor running — testnet or mainnet):** order of operations —
  1. **cancel-all** open orders for our account on the CLOB (cancel-on-start
     policy: unknown resting orders are risk, not state to adopt);
  2. fetch venue truth: token balances (data-API positions / on-chain
     `balanceOf`) + recent trades;
  3. diff vs journal. **Venue is truth; the journal explains it.** Mismatch →
     adopt venue numbers, alert, and (config `halt_on_drift`, default true)
     hold the kill switch until acknowledged;
  4. any open position found → per `flatten_on_restart` (default **true**):
     synthesize a trade in `Exiting` and run the §6.2 ladder immediately;
     else adopt as Holding with exit due now.
- **Steady-state reconcile (between restarts):** the startup diff is *also* run
  on the §8.6 triggers — after every `end_trade()` (expect flat) and on
  user-channel WS reconnect. `GET /data/trades` since `last_trade_cursor`
  rebuilds the fill history; `balanceOf`/data-API `/positions` give net qty.
  Drift beyond ε → same policy as restart: adopt venue, alert, `halt_on_drift`.
- **Marking:** on each mirror book update of a held instrument, recompute
  `upnl`; publish `exec.position` throttled (on Δ > ε or 500 ms heartbeat).
  Mark = **best bid** of the held token (what flattening would plausibly
  realize), not mid — `daily_loss_limit` (§9) keys off this conservative
  number.

### 8.6 Pull-based fill ingestion (`GET /data/trades`)

The user-channel WS is the real-time fill source; **`GET /data/trades`** is the
**per-user, authenticated** pull that backstops it (gaps, restarts,
ack-timeouts) and supplies the authoritative settlement status + realized fee.
It returns **only this account's** trades (L2-auth, address implicit) — never
the public across-users tape (§7).

**Triggered, not constantly polled** (rate-limit-friendly — no pull when idle):

| trigger | scope of pull | purpose |
|---|---|---|
| an order is non-terminal | `?market=<cid>` since cursor, every `trade_poll_ms` | confirm a FAK fill the WS hasn't delivered yet; bounded to the order's brief active window |
| `place` ack-timeout (§7) | `?market=<cid>`, attribute-match the in-flight order (§4.2) | resolve `Unknown` — did the lost-ack order trade? |
| user-channel WS (re)connect | since `last_trade_cursor` → head | backfill every fill missed during the gap *before* trusting the stream again |
| after `end_trade()` (expect flat) | `?market=<cid>` since cursor | flat-check — trades must net to a zero balance, else drift (§8.5) |
| startup reconcile (§8.5) | all touched markets since cursor | rebuild **cost basis + realized P&L** (balances give qty only) |

**The four identifiers.** Two are ours, two are the venue's; they nest:

| id | domain | shape | one per | assigned by |
|---|---|---|---|---|
| `trade_id` | ours | `{strategy}-{signal_ts_ns}` | round-trip **Trade** (entry→exit) | us (§3.1) |
| `client_id` | ours | `{trade_id}:{leg}:{attempt}` | order **attempt** | us (§4.2) |
| `order_id` (= `taker_order_id`) | venue | EIP-712 order hash | submitted **order** | venue — returned on the POST ack (recovered via reconcile on a lost ack) |
| `venue_trade_id` | venue | opaque `id` on a `/data/trades` row / WS trade | **match/fill event** | the venue |

**Resolution** (raw fill row → our signal):
`row.id` = `venue_trade_id` → `row.taker_order_id` = our `order_id`
→ *(OrderManager map `venue_order_id → client_id`, recorded from the POST ack)* `client_id`
→ *(parse prefix `client_id.split(':')[0]`)* `trade_id` → the Trade + its signal.

The load-bearing properties:

- **`order_id` comes from the venue, not from us** — the POST ack returns it and
  the OrderManager records `venue_order_id ↔ client_id` (§4.2); a fill row's
  `taker_order_id` is looked up in that map to recover `client_id`. The one case
  the map can miss — a fill for an order whose ack was lost — *is* the §7
  reconcile trigger (attribute-match the in-flight order, adopt the id). We do
  not pre-compute the id.
- **`venue_trade_id` is not derivable** — only the venue assigns it; we learn it
  from the WS/pull feed. It is **globally unique per fill**, so it is the dedupe
  key: `apply_fill` is idempotent on it (§8.2).
- **one `order_id` → ≥1 `venue_trade_id`** (a marketable order sweeping several
  levels, or partials seen across polls) — each summed by `apply_fill`.
- **one `trade_id` → many `client_id`** (entry retries `E:0,E:1…` + exit ladder
  `X:0,X:1…`), each → its own `order_id` → its own `venue_trade_id`(s).
- `taker_order_id` is simply `order_id` *as seen from the trade row* when we are
  the taker — always, for FAK. Maker side (P3 passive): our `order_id` appears in
  the row's `maker_orders[]` instead, and the fill is attributed from there.

**Cursor / high-water mark.** The PM persists `last_trade_cursor` (last applied
`id` / `match_time`). Backfill pages newest-first until it reaches the cursor,
applies the new rows **oldest-first** (so `avg_cost` builds in trade order),
then advances + journals the cursor → a restart resumes from exactly where it
stopped.

**Edge cases:**

| case | handling |
|---|---|
| same fill on WS + pull | dedupe by `venue_trade_id` → second is a no-op (§8.2) |
| `matched` seen, `confirmed` never arrives | held pending-confirm; later pull resolves it; unresolved past `trade_confirm_timeout_s` → alert |
| `matched` → `failed` | reverse the provisional fill (§8.2); if it moved us off-flat, re-run the affected gate |
| one order, many trades (partials) | each is a distinct `venue_trade_id`, summed by `apply_fill`; exit `remaining` reconciles to the sum |
| pull returns a trade for an order we marked `Dead` | adopt it — venue is truth: apply the fill + alert (the lost-ack-actually-filled case) |
| WS down past the poll window | the `trade_poll_ms` pull keeps fills current; **both** WS and REST down → stale-venue kill switch (§9), exits still forced (§6.3) |
| backfill exceeds `max_backfill_pages` | stop, alert, `halt_on_drift` — a gap that large means something is wrong |
| duplicate `venue_trade_id`, different amounts | corruption → kill switch + reconcile; never silently overwrite |

*P2 verification:* exact field names (`id`, `taker_order_id`, `maker_orders`,
`status`, `match_time`), the `status` enum values, and pagination
(`next_cursor` vs `before`/`after`) are confirmed against the live CLOB /
`py-clob-client` during the live phase; the *scoping* (per-account) is not in
question.

---

## 9. Risk gate (pre-trade checks, in order)

| # | check | on breach |
|---|---|---|
| 1 | kill switch off (manual via `exec.ctl`, or auto-tripped) | reject |
| 2 | venue adapter healthy; mirror fresh (`now − recv_ts_ns ≤ stale_ms·MS`) | reject |
| 3 | `try_begin_trade` — no active trade, no position, cooldown elapsed | reject |
| 4 | re-validation set §3.1 (bucket, tte, ttl, edge-consumption) | reject |
| 5 | net-edge: expected edge − fee(`fee_rate_bps`, price) ≥ `min_net_edge_c` | reject |
| 6 | size ≥ venue min; order notional ≤ `max_order_notional` | reject |
| 7 | `cash ≥ size·price·(1 + fee_buffer)` (live: USDC balance + allowance) | reject |
| 8 | rate: trades this minute < `max_trades_per_min` | reject |
| 9 | `daily.realized + Σupnl > −daily_loss_limit` | **trip kill switch** |

Auto-trips of the kill switch: rule 9; N consecutive venue rejects (§5);
long-only invariant violation (§8.2); reconcile drift (§8.5); persistent
stale data while Idle. Tripping never blocks **exits** — the kill switch
gates *entries*; an active trade always runs its exit ladder (and `flatten`
forces one).

---

## 10. Validation & fill modeling (no paper venue)

There is **no local PaperVenue** — Polymarket offers no official paper mode, and
rather than maintain a second fill simulator we split validation into the two
things that actually need proving:

**Strategy / fill economics → the existing Python backtest.** `arb_study/
backtest.py` is the fill-model oracle: it already models entry-at-ask /
exit-at-bid, the YES spread, hold, and one-position-at-a-time. The frictions
this design surfaced are folded **into that backtest** before any live switch
(§15 P0): the **250 ms taker delay** (re-run at latency ≥ 250 ms) and the
**taker fee** — Polymarket's current per-category formula is
`fee = feeRate × p × (1−p) × qty` (taker-only, symmetric, peaks at p=0.5;
*not* the older `min(p,1−p)` form), and **crypto markets carry the highest rate,
`feeRate ≈ 0.07` → peak ≈1.75% at 50¢** (verified docs.polymarket.com/trading/
fees, Jun 2026). Deducted in shares on BUY. The backtest — not a live-code sim —
is what says the edge survives.

**Execution pipeline → replay + testnet + small mainnet.**

| stage | what it proves | how |
|---|---|---|
| **replay** (P1) | collectors→processor→rules produce the **same signals** the backtest does | feed recorded streams through the real processor + `PerpMoveRule`; diff signal ts / direction / target vs the backtest (*signal parity*) |
| **Amoy testnet** (P2) | the `PolymarketClob` adapter plumbing — EIP-712 signing, L2 auth, order format, FAK/GTD, cancel, `venue_order_id`-from-ack, reconcile | run the executor against the testnet CLOB (chainId 80002); no financial risk, but no realistic liquidity |
| **small mainnet** (P2) | real fills, real fees, real 250 ms delay, real slippage | `$1` sizes, FAK only; the `TradeRecord` stream (§11) measured live is compared against the backtest's per-trade expectations |

The executor always places **real** orders (testnet or mainnet) — the `venue`
config selects the *network*, not paper-vs-live. The `TradingVenue` trait is
kept for one impl (`PolymarketClob`) plus test mocks.

---

## 11. Events, topics & records (Δ + additions)

| name | mode | new? | content |
|---|---|---|---|
| `exec.trade` | pub/sub | **new** | terminal `TradeRecord` per trade (below) |
| `exec.ctl` | claimed port (executor) | **new** | `Kill`, `Resume`, `Flatten` ops commands |
| `market.polymarket.catalog` | pub/sub | extended | adds `status: resolved, winner` (§8.3) |
| `exec.report` / `exec.fill` / `exec.position` | pub/sub | as main doc | per-order reports / fills / PM snapshots |

```rust
pub struct TradeRecord {            // the backtest row, measured live — column-compatible
    pub trade_id: String, pub outcome: TradeOutcome, // Closed|Abandoned|Settled
    pub direction: i8, pub instrument: String,
    pub signal_ts_ns: i64, pub trigger: Trigger,      // move_bps, window_ms, yes_price
    pub entry: LegSummary, pub exit: LegSummary,      // qty, vwap, fees, n_fills, first/last ts
    pub hold_actual_ms: u64,
    pub pnl_gross: f64, pub pnl_net: f64,             // cross-equivalent & net-of-fees, USDC
    pub slippage_entry_c: f64, pub slippage_exit_c: f64, // vs signal-time ask / exit-due bid
    pub lat_signal_to_submit_ms: f64, pub lat_submit_to_ack_ms: f64, pub lat_ack_to_fill_ms: f64,
}
```

`TradeRecord` is the parity instrument, used two ways (§10): **(a) signal
parity** — replaying a recorded stream through processor + rules must reproduce
the backtest's entry/exit *signals* (ts, direction, target); **(b) fill parity**
— the small-mainnet `TradeRecord` stream (entry ask, exit bid, fees, P&L) is
compared against the backtest's per-trade expectations. Together they are the
acceptance test for this design.

---

## 12. Changes required to DESIGN_TRADING_SYSTEM.md

1. **§5 collector:** add post-`Expired` resolution poll + `resolved` catalog
   status (§8.3 here).
2. **§2/§9:** executor additionally subscribes `market.polymarket.#`
   (Conflate) + `market.polymarket.catalog` for the ExecBookMirror; add
   topics `exec.trade`, port `exec.ctl`.
3. **§8 data model:** `OrderType` enum → the three intents of §4.1
   (marketable-limit FAK / GTD passive / cancel); add `TradeRecord`.

---

## 13. Config (executor section, YAML)

```yaml
executor:
  venue:
    adapter: polymarket_clob    # only adapter; no paper venue
    network: testnet            # testnet (Amoy, chainId 80002) | mainnet (137)
    max_order_usdc: 1.0         # hard cap while validating on mainnet
  sizing:
    size_shares: 20             # fixed target size, v1
    depth_frac: 0.5             # cap vs displayed top-of-book size
  entry:
    chase_c: 0.0                # cap = ask + chase_c per attempt (0 = top-of-book only)
    max_chase_total_c: 0.01     # abandon if ask ran > 1c past signal-time ask
    max_attempts: 3
    retry_delay_ms: 100
    ttl_ms: 1000                # = backtest MAX_ENTRY_GAP_S
  exit:
    mode: cross                 # cross | passive
    retry_interval_ms: 250
    step_c: 0.01                # ladder deepening per miss
    max_slip_c: 0.05            # floor vs bid at exit-due
    stale_extra_slip_c: 0.03    # §6.3(c)
    deadline_buffer_ms: 5000    # be flat ≥ 5 s before expiry
    passive_wait_ms: 1500       # passive mode only
  hold_ms: 1000                 # H (signal.plan may override per strategy)
  risk:
    yes_bucket: [0.05, 0.95]
    min_tte_ms: 15000
    min_net_edge_c: 0.3
    max_order_notional: 25.0
    max_trades_per_min: 6
    daily_loss_limit: 50.0
    cooldown_ms: 2000
    stale_ms: 1500
    max_consecutive_rejects: 3
    fee_buffer: 0.02
  position:
    mark: bid                   # bid | mid
    flatten_on_restart: true
    halt_on_drift: true
    reconcile_after_trade: true # §8.6 flat-check after each end_trade()
    reconcile_on_reconnect: true# §8.6 backfill on user-channel WS reconnect
    journal_path: data/exec_journal.sqlite
  order:
    ack_timeout_ms: 2000
    poll_ms: 500                # live REST order-status poll while orders open
    trade_poll_ms: 500          # GET /data/trades pull while an order is non-terminal (§8.6)
    trade_confirm_timeout_s: 120 # alert if a matched fill never reaches confirmed (§8.6)
    max_backfill_pages: 50      # cursor-backfill page cap before halt_on_drift (§8.6)
```

(The 250 ms taker delay and `fee_rate_bps` are venue facts read from market
metadata per market (§15), not config knobs — there is no local fill sim to
parameterize.)

Defaults mirror the backtest's best-validated cell (θ per rule config,
hold = 1 s, latency budget ≤ 200 ms) — change via config, never code.

---

## 14. Failure matrix

| failure | during | behavior |
|---|---|---|
| no entry fill / book ran away | Entering | bounded retries → Abandoned (never chase) §5 |
| venue reject ×N | Entering | Abandoned + kill switch after `max_consecutive_rejects` |
| place ack timeout | any | Unknown → reconcile before any further order on that leg §7 |
| user-channel WS drop | live, any | REST poll fallback continues fill tracking §7 |
| target book stale | Holding | forced exit, widened floor, alert §6.3(c) |
| book empty / venue down at exit | Exiting | ladder exhausts → PendingResolution + alert §6.4 |
| catalog `expired` early | Holding | immediate ladder on remaining book §6.3(d) |
| executor crash | mid-trade | journal replay; live: cancel-all → reconcile → flatten §8.5 |
| reconcile drift | startup | adopt venue, alert, hold kill switch (`halt_on_drift`) §8.5 |
| resolution never arrives | PendingResolution | poll timeout → alert, manual; funds redeemable later §8.3 |
| daily loss breach | any | kill switch — entries blocked, exits always allowed §9 |

---

## 15. Open questions (tracked, not blocking P1)

1. **P0 — re-run economics with the live frictions (§1 gates).** Two
   *verified* facts must enter the backtest before any live switch, because
   each can flip a few-cent edge negative: (a) the **250 ms taker delay** on
   these markets → re-run with latency ≥ 250 ms (the backtest's latency-decay
   shows the edge shrinks with latency); (b) the **taker fee** — pull
   `fee_rate_bps` from live 5-minute markets and net it out (charged in shares
   on BUY, USDC on SELL). Until this passes, live stays off.
2. **Taker-delay + fee detection (live wiring)** — confirm the per-market
   taker-delay flag via `GET /clob-markets/{cid}` and the fee rate via the CLOB
   `GET fee rate` endpoint (per-token bps). Formula is
   `fee = feeRate × p × (1−p) × qty`; crypto `feeRate ≈ 0.07` (resolved Jun 2026,
   docs.polymarket.com/trading/fees) — still read per-market rather than hardcode
   (fees change and are category-specific). Both flow into market meta (the risk
   gate + the backtest re-run read them).
3. **Settlement vs. hold (resolved, Jun 2026)** — a SELL is **not** gated on the
   BUY settling on-chain: live 15-min crypto traders round-trip buy→sell, the
   only blocker being fee-reduced size (py-clob-client #245), not a settlement
   lockout. So a ~1 s hold is executable and **`matched` is the correct fill
   indicator** (§8.2). On-chain settlement is ~2 s (Bor); **naked short is
   mechanically impossible** (no margin; balance+allowance checked; atomic
   swap that reverts if tokens aren't held).
4. **Order-id source (decided)** — `venue_order_id` is taken from the POST ack,
   *not* computed from a deterministic hash; a lost ack reconciles by per-user
   attribute matching (§4.2, §7). Revisit only if a multi-order-in-flight mode
   (P3) makes the attribute match ambiguous.
5. **Partial-fill top-up** — v1 never tops up (§5); revisit if live partials
   are frequent and the residual-size edge measurably persists.
6. **Passive exit** — enable after a small-mainnet A/B (`exit.mode`) shows
   spread capture beats miss cost (no paper venue to A/B it offline; the
   backtest can bound it but can't model queue position).
7. **negRisk flag** — these markets' signing path (exchange vs negRisk
   adapter contract) is market metadata; the live adapter must read it, not
   assume (P2, signing layer).
