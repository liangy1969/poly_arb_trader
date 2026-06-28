# DESIGN — Arbitrage Trading System (Rust, event-pipeline architecture)

Status: draft for review. Implements the "Polymarket **Up** token catches up
after a BTC perp tick" strategy validated in `e:\poly\crypto_collector\arb_study\`
(see §0 for all reference paths) — **trade the prediction
market only** (the BTC perp is the *signal*, not a traded leg).

**Decisions locked in:** Rust / async (tokio) · in-process bus behind a swappable
trait · **no paper venue** — strategy validated by the Python backtest + replay,
execution on **Amoy testnet → small mainnet** · **execution = Polymarket
only** (no perp hedge for now).

> Note (scope of the chosen strategy): trading the Up token *without* the perp
> hedge is the **naked directional** variant — +EV in the backtest but it carries
> full BTC beta (a momentum+lag bet) and is exposed to regime/drift risk. The
> architecture keeps a hedge leg trivially addable later (just a second leg +
> a Binance trading adapter); for now the executor is single-venue.

---

## 0. Reference code (absolute paths — this doc moves to the new project)

| What | Path |
|---|---|
| **Crypto collector (Python reference for `collector-binance`)** | `E:\crypto\collector\` |
| · stateful L2 book + buffered-snapshot/diff resync | `E:\crypto\collector\collector\adapters\l2_book.py` |
| · Binance adapter, WS client, frame parsers (`U/u/pu`, exch/recv ts) | `E:\crypto\collector\collector\adapters\binance\` |
| · working config incl. SOCKS proxy, 100 ms cadence | `E:\crypto\collector\config\local.yaml` |
| **Polymarket collector (Python reference for `collector-polymarket`)** | `e:\poly\crypto_collector\collector\polymarket\` |
| · slug prediction, window-START convention, `endDate` expiry | `e:\poly\crypto_collector\collector\polymarket\discovery.py` |
| · stateful Up/Down book (snapshot + `price_change` deltas) | `e:\poly\crypto_collector\collector\polymarket\book.py` |
| · Gamma/CLOB/WS endpoints & framing | `e:\poly\crypto_collector\collector\polymarket\client.py` |
| · frame normalization (incl. the fixed `_pick_end_raw` expiry order) | `e:\poly\crypto_collector\collector\polymarket\normalizer.py` |
| · orchestration (per-market tasks, WS resubscribe) | `e:\poly\crypto_collector\collector\polymarket\collector.py` |
| · implementation reference doc (§16) | `e:\poly\crypto_collector\DESIGN_PREDICTION_MARKETS.md` |
| **Strategy research & backtests (parameters, expected P&L)** | `e:\poly\crypto_collector\arb_study\` |
| · the backtest this system implements (θ, hold, latency, buckets) | `e:\poly\crypto_collector\arb_study\backtest.py` |
| · tick-level predictiveness (slope vs horizon/bucket) | `e:\poly\crypto_collector\arb_study\tick_predict2.py` |
| · loaders / market-id conventions / run notes | `e:\poly\crypto_collector\arb_study\analyze_lag.py`, `README.md` |
| **Collected data (for replay/parity testing)** | `E:\crypto\collector\data\parquet\`, `e:\poly\crypto_collector\data\parquet\` |

---

## 1. Goals & principles

- **Virtual distributed / share-nothing.** Modules communicate *only* through an
  event pipeline; no shared memory. A module can later move to its own process or
  host by swapping the bus transport — **no module code changes**.
- **Modular & extensible.** New venue = new collector. New strategy = new rule.
  New execution venue = new trading adapter.
- **One code path for backtest and live.** Processor + rules run identically on
  a recorded or live event stream → research and production share logic (the
  stream is recordable and replayable). Strategy P&L / fill modeling lives in the
  Python backtest; the executor places real orders (testnet/mainnet) and is not
  in the replay loop.
- **Hot path minimal, non-blocking, low-latency** (the reason for Rust): the
  backtest showed reaction ≤100–200 ms is decisive. Persistence/analytics run off
  the critical path. Single clock (`ts_ns`) on every event for ordering + latency.

---

## 2. System architecture

```
┌──────────────────────────────  EVENT BUS (in-proc, trait Bus)  ──────────────────────────────┐
│  pub/sub topics: market.*  signal.*  exec.*           |           claimed port: exec.intent  │
└─────────┬───────────────────┬─────────────────────┬───────────────────────┬──────────────────┘
          │ publish           │ publish             │ subscribe market.*    │ subscribe signal.*
          │ market.binance.*  │ market.polymkt.*    │ publish  signal.*     │ publish  exec.*
          │                   │                     │ (+ exec.* feedback)   │ claim  exec.intent
┌─────────┴─────────┐ ┌───────┴────────────┐ ┌──────┴─────────────────┐ ┌───┴──────────────┐
│ Collector: binance│ │ Collector: polymkt │ │ Market Event Processor │ │  Trade Executor  │
│ perp WS (SIGNAL   │ │ CLOB/WS (book +    │ │ • merge → global state │ │ Polymarket only  │
│ only, not traded) │ │ catalog / expiry)  │ │ • persist (off-path)   │ │ • risk/limits    │
│ → market.binance.*│ │ → market.polymkt.* │ │ • rule engine          │ │ • testnet|mainnet│
└───────────────────┘ └────────────────────┘ │ → signal.*             │ │ → exec.*         │
                                             └────────────────────────┘ └──────────────────┘

flow:  collectors ──market.*──▶ processor ──signal.*──▶ executor ──orders──▶ Polymarket (testnet|mainnet)
                                    ▲                          │
                                    └─── exec.* (fills) ───────┘
```

A **Supervisor** task starts/stops modules, watches health, restarts on failure.

---

## 3. The event pipeline (`Bus`) — the central piece

One bus, two delivery modes:

- **Pub/Sub (topics)** — many-to-many fan-out. Hierarchical dotted topics with
  wildcards (`market.binance.#`, `market.*.*.BTCUSDT.book`). For market data + signals.
- **Ports (claimed)** — exactly-one-owner point-to-point. `claim_port` errors if
  already owned. For order intents → the single executor.

A central **router task** owns the subscription table; `publish` hands an
`Arc<Event>` to the router, which pattern-matches and forwards to each
subscriber's channel (no payload cloning on fan-out). Per-subscriber **bounded**
channels with a backpressure **policy**: market data = `Conflate{key}` (keep only
latest per instrument) or `DropOldest`; signals/exec = `Block` (never silently
drop an order). A **Recorder** subscriber can persist the whole stream for replay.

```rust
#[derive(Clone)]
pub struct Event {
    pub topic: Topic,          // interned dotted key
    pub source: &'static str,  // emitting module
    pub ts_ns: i64,            // creation wall-clock (ns)
    pub seq: u64,              // per-source monotonic
    pub payload: Payload,
}

pub enum Payload {            // one enum = no dynamic dispatch on the hot path
    Book(BookUpdate), Trade(TradeTick), Liq(Liquidation), Meta(MarketMeta),
    Signal(TradeSignal), Order(OrderRequest), Report(ExecutionReport), Position(PositionUpdate),
}

pub enum Policy { Conflate(fn(&Event)->u64), DropOldest, Block }

pub trait Bus: Send + Sync {
    fn publish(&self, ev: Event);
    fn subscribe(&self, pattern: &str, maxq: usize, policy: Policy) -> Subscription; // Receiver<Arc<Event>>
    fn claim_port(&self, port: &str, maxq: usize) -> PortInbox;                      // exclusive
    fn send(&self, port: &str, ev: Event);
}
// InProcBus (tokio mpsc/router) now; BrokerBus (NATS/redis/zmq) later — same trait.
```

---

## 4. Module framework

```rust
#[async_trait::async_trait]
pub trait Module: Send {
    fn name(&self) -> &'static str;
    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()>; // claim/subscribe, spawn tasks
    async fn stop(&mut self) -> anyhow::Result<()>;
    fn health(&self) -> Health;                                          // Ok | Degraded | Down
}
```

---

## 5. Module 1 — Data Collectors (full Rust, one per venue)

Both collectors are **reimplemented in Rust** (tokio + `tokio-tungstenite` WS,
`reqwest` REST), porting the logic from the existing Python collectors as the
reference spec. Each WS frame normalizes to a canonical payload and publishes on
`market.<venue>.<market>.<symbol>.<stream>` (carrying `exch_ts_ns` + `recv_ts_ns`).

**`collector-binance` — perp signal feed** (port from `E:\crypto\collector`):
- `wss …/stream` depth@100 ms diff + aggTrade + forceOrder(liquidation) subs;
- stateful **L2 book** with buffered-snapshot + diff resync (the `U/u/pu`
  update-id continuity, crossed-book → resync). Only top-of-book is needed for the
  rule, but maintain the book for correctness. **This book is collector-local**
  (feed plumbing — a diff stream is uninterpretable without it); it is *not*
  shared state. Only normalized `BookUpdate` events (top-of-book / top-N) cross
  the bus; the processor's `MarketState` is a derived projection of those events,
  never a replica of the venue book;
- optional SOCKS5 proxy (Binance Global), mirroring the Python `socks_proxy`;
- publishes `market.binance.usdt_perp.BTCUSDT.{book,trade,liquidation}` — **signal only, not traded.**

**`collector-polymarket` — tradeable book + catalog** (port from
`e:\poly\crypto_collector\collector\polymarket\`, see §0):

*Scope:* only the configured short-term series (`btc-updown-5m`) is tracked — the
series is deterministic via slug prediction, so the data collector's tag-discovery
and event-scan loops are **not ported**. The collector actively manages the active
market set with a small registry and three cooperating tasks:

```
ActiveMarketSet:  cid → { slug, up_token, down_token, start_ns, end_ns, phase }
phase:            Pending ─▶ Upcoming ─▶ Live ─▶ Expired (─▶ dropped after grace)
                  (tokens     (listed,    (start ≤ now    (now ≥ end_ns;
                   not yet     book pre-    < end_ns;      unsubscribe after
                   parseable)  subscribed)  the TARGET)    unsub_grace_s ≈ 10 s)
```

1. **Discovery loop** (poll Gamma every `discovery_interval_s`, default 5 s):
   - predict slugs: `start = ⌊now / window⌋ · window`; slugs `{prefix}-{start + k·window}`
     for k = 0..n_upcoming (slug timestamp = window **START** — fixed convention);
   - for each slug not in the registry: `GET /markets?slug=` (browser User-Agent —
     the default UA gets 403). If listed → parse `conditionId`, `clobTokenIds`
     (Up/Down), and full-timestamp **`endDate`** as the authoritative expiry
     (never the date-only `endDateIso` — the midnight bug). Insert as Upcoming.
   - not yet listed (markets appear only near their window; outside listing hours,
     never) → retry next tick, not an error. Tokens missing → stay Pending, retry.

2. **Lifecycle scheduler** (event-driven — `sleep_until` the next `start_ns`/`end_ns`
   in the registry; no polling):
   - at `start_ns`: phase → Live; publish `market.polymarket.catalog`
     `{instrument, kind, start, expiry, status: live}` — the processor's
     InstrumentLinker repoints the BTC-perp link to this market's UP token at the
     boundary;
   - at `end_ns`: phase → Expired; publish `status: expired` (linker drops the
     target; rules idle until the next Live). After `unsub_grace_s` drop +
     unsubscribe. No long post-expiry grace — that was for trade-tape capture,
     not trading;
   - also publish `status: upcoming` on first discovery so downstream can pre-warm.

3. **WS task** (one CLOB `market`-channel connection):
   - desired asset set = tokens of all Upcoming + Live markets. Subscribing
     Upcoming **before** the boundary means the next window's book is already
     streaming at rollover — no data gap when the linker switches;
   - the CLOB WS fixes the asset list at subscribe time → on any set change,
     reconnect/resubscribe (the Python reference behavior);
   - per-token stateful book (`book` snapshot + `price_change` deltas,
     hash-checked) → `BookUpdate` (top-N) on `market.polymarket.<cid>.book`;
     `last_trade_price` → `.trade`.

*Failure modes:* Gamma unreachable → keep serving the current Live market until
its expiry, keep retrying discovery (alert if no Upcoming exists within one window
of rollover). WS drop → reconnect, fresh `book` snapshot rebuilds state (brief
gap). Rollover with nothing listed (outside listing hours) → no target, rules
idle — correct behavior, not an error.

**Parity gate.** Before a Rust collector is trusted, run it side-by-side with the
Python one on the same live feed and diff the normalized events (mids, update-ids,
expiries). The strategy depends on correct data, so each collector ships only once
it matches.

---

## 6. Module 2 — Market Event Processor

Subscribes `market.*`. Three jobs:

**(a) Global market state with merging** — one `InstrumentState` per canonical
instrument (full field spec in §8: top-of-book + sizes, top-N depth for the
target, last trade, ts/seq/staleness, lifecycle, mid-history ring), all venues
in one store (ids are venue-qualified — no cross-venue book averaging);
**rolling windows** the rules need (perp mid over the last X ms); an **InstrumentLinker**
that resolves the strategy *target* (current live Polymarket Up token) from the
*reference* (BTC perp). The linker is driven purely by `market.polymarket.catalog`
lifecycle events (§5): it repoints on `live`, clears on `expired` — it never
infers liveness from clocks or prices itself. `MarketState` is a
**derived projection of `BookUpdate` events** — top-of-book + windows, venue-
agnostic. The authoritative venue L2 books (update-ids, resync) live inside the
collectors (§5) and never cross the bus.

**(b) Persistence** to parquet/sqlite, **off the hot path** (bounded channel → writer task).

**(c) Rule engine** — runs registered rules per relevant event; emits
`TradeSignal` on `signal.<strategy>`.

```rust
pub trait Rule: Send {
    fn id(&self) -> &str;
    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal>;
}

/// Fire when |perp mid % change over past window_ms| >= threshold_bps.
/// direction = sign(move); target = current live Polymarket Up (via linker).
/// Gates: yes_price in [lo,hi] bucket, cooldown_ms, max signals/min, and
/// min_tte_ms — no signal when target expiry - now < hold_ms + buffer, so a
/// position never straddles the window's resolution.
pub struct PerpMoveRule {
    pub reference: String, pub window_ms: u64, pub threshold_bps: f64,
    pub yes_bucket: (f64, f64), pub cooldown_ms: u64, pub min_tte_ms: u64,
}
```

Components: `MarketStateStore`, `RollingWindow` (ring buffer), `InstrumentLinker`,
`RuleEngine`, `Persistor`.

---

## 7. Module 3 — Trade Executor (Polymarket only)

> Detail design: `DESIGN_EXECUTION.md` — order types, entry unfill processing,
> exit execution/escalation, order & trade state machines, Position Manager,
> risk gate, validation approach. That doc also lists three small deltas to this
> one (catalog `resolved` status, executor market-data subscription, new
> `exec.trade` topic / `exec.ctl` port).

Subscribes `signal.*` (or claims port `exec.intent`). Per signal:

1. **ExecutionPlan** — a single leg (for now): a Polymarket **Up** order in the
   signal direction (`+1` buy / `-1` sell-via-No), with `hold_ms` then exit
   (`cross` the spread or `passive`). One-position-at-a-time / cooldown matches
   the backtest.
2. **Risk gate** (pre-trade): max position/notional, max open orders, price
   sanity, global **kill-switch**, throttle. Breach → reject.
3. **Submit** via a `TradingVenue` adapter; manage lifecycle (ack, fill, timeout,
   cancel/replace, exit at `hold_ms`).
4. Emit `ExecutionReport` / `Fill` / `PositionUpdate` on `exec.*`.

```rust
#[async_trait::async_trait]
pub trait TradingVenue: Send + Sync {
    async fn place(&self, o: OrderRequest) -> anyhow::Result<ExecutionReport>;
    async fn cancel(&self, client_id: &str) -> anyhow::Result<ExecutionReport>;
}
```

Adapter (one, no paper venue):
- **`PolymarketClob`** — places signed CLOB orders. Selectable network:
  **Amoy testnet** (chainId 80002, no financial risk — validates signing/auth/
  order format) or **mainnet** (137). Polymarket CLOB orders are **EIP-712
  signed** → needs `ethers-rs` signing + key management; that's the main
  execution complexity. There is no `PaperVenue` — Polymarket has no official
  paper mode, so rather than maintain a second fill simulator, the **strategy is
  validated by the Python backtest + replay** and **execution on testnet → small
  mainnet** (`DESIGN_EXECUTION.md` §10).

`venue.network: testnet | mainnet` selects the network; everything upstream is
identical. The executor always places **real** orders.

---

## 8. Data model

```rust
// ---- market payloads ----
pub struct BookUpdate { pub instrument: String, pub bids: Vec<(f64,f64)>, pub asks: Vec<(f64,f64)>,
                        pub update_id: Option<u64>, pub exch_ts_ns: i64, pub recv_ts_ns: i64 }
pub struct TradeTick  { pub instrument: String, pub price: f64, pub qty: f64, pub side: Side,
                        pub exch_ts_ns: i64, pub recv_ts_ns: i64 }
pub struct Liquidation{ pub instrument: String, pub price: f64, pub qty: f64, pub side: Side, pub recv_ts_ns: i64 }
pub enum MarketStatus { Upcoming, Live, Expired }
pub struct MarketMeta { pub instrument: String,            // "polymarket.<cid>.UP"
                        pub kind: String,                  // "5m_updown"
                        pub status: MarketStatus,          // lifecycle phase (catalog events)
                        pub start_ts_ns: Option<i64>, pub expiry_ts_ns: Option<i64> }

// ---- signal ----
pub struct TradeSignal {
    pub strategy: String, pub ts_ns: i64, pub reason: String,
    pub reference: String,        // "binance.usdt_perp.BTCUSDT"
    pub target: String,           // "polymarket.<cid>.UP"
    pub direction: i8,            // +1 buy / -1 sell
    pub trigger: Trigger,         // { move_bps, window_ms, yes_price }
    pub plan: ExecutionPlan,
    pub ttl_ms: u64,
}

// ---- execution ----
pub enum Side { Buy, Sell }
pub enum OrderType { Limit, Market }
pub struct OrderRequest { pub instrument: String, pub side: Side, pub otype: OrderType,
                          pub price: Option<f64>, pub size: f64, pub tif: Tif, pub client_id: String }
pub struct ExecutionPlan{ pub legs: Vec<OrderRequest>, pub hold_ms: u64, pub exit: ExitKind }
pub struct ExecutionReport { pub client_id: String, pub venue_order_id: String, pub status: OrderStatus,
                             pub filled: f64, pub avg_price: f64, pub fee: f64, pub ts_ns: i64 }
pub struct PositionUpdate { pub instrument: String, pub net: f64, pub avg_price: f64, pub upnl: f64, pub ts_ns: i64 }

// ---- processor-internal state ----

pub enum InstrumentKind { CryptoPerp, CryptoSpot, PredictionOutcome }

/// One per canonical instrument, updated in-place per market.* event.
pub struct InstrumentState {
    pub instrument: String,            // canonical id
    pub kind: InstrumentKind,

    // top of book — what the rules read
    pub best_bid: f64, pub bid_sz: f64,
    pub best_ask: f64, pub ask_sz: f64,
    pub mid: f64,                      // (bid+ask)/2; NaN until both sides seen
    pub spread: f64,

    // shallow depth — PredictionOutcome only; for sizing + exit-fill estimation
    pub depth: Option<Depth>,          // top-N levels, as published by the collector

    pub last_trade: Option<LastTrade>, // px, qty, side, ts (from .trade events)

    // timestamps & quality — staleness gate + latency accounting
    pub exch_ts_ns: i64,               // venue stamp of the update that set this state
    pub recv_ts_ns: i64,               // collector receive stamp
    pub apply_ts_ns: i64,              // processor apply stamp (apply-recv = bus latency)
    pub seq: u64,                      // per-source seq of last applied event
    pub stale: bool,                   // watchdog sets if now - recv_ts_ns > stale_ms

    // lifecycle — PredictionOutcome only; written by catalog events
    pub status: Option<MarketStatus>,  // Upcoming | Live | Expired
    pub start_ts_ns: Option<i64>,
    pub expiry_ts_ns: Option<i64>,     // drives the min_tte gate

    // history — what move_bps() reads
    pub ring: RollingWindow,           // (ts_ns, mid) samples
}

pub struct Depth     { pub bids: Vec<(f64, f64)>, pub asks: Vec<(f64, f64)> } // best-first
pub struct LastTrade { pub price: f64, pub qty: f64, pub side: Side, pub ts_ns: i64 }

/// Fixed-capacity ring of (ts_ns, mid); push on mid change; no alloc on hot path.
/// Capacity sized from max rule window × max tick rate (e.g. 1 s × ~20/s → 256 slots).
pub struct RollingWindow { /* buf: Box<[(i64,f64)]>, head, len, horizon_ms */ }
impl RollingWindow {
    pub fn push(&mut self, ts_ns: i64, mid: f64);
    pub fn asof(&self, ts_ns: i64) -> Option<f64>;     // latest sample <= ts (binary search)
    pub fn move_bps(&self, now_ns: i64, window_ms: u64) -> Option<f64>;
    // (mid_now / mid_asof(now - window) - 1) * 1e4; None if no sample spans the window
}

pub struct MarketState {
    pub instruments: HashMap<String, InstrumentState>,
    pub links: HashMap<String, String>,                // reference -> current Live target
}
impl MarketState {
    pub fn on_event(&mut self, ev: &Event);            // merge (semantics below)
    pub fn get(&self, inst: &str) -> Option<&InstrumentState>;
    pub fn target_of(&self, reference: &str) -> Option<&InstrumentState>;
    pub fn move_bps(&self, inst: &str, window_ms: u64) -> Option<f64>;
    pub fn tte_ms(&self, inst: &str, now_ns: i64) -> Option<i64>;
}
```

**Merge semantics** (`on_event`): `Book` → replace top-of-book/depth, update
`exch/recv/apply_ts`, `seq`, clear `stale`, push `(recv_ts_ns, mid)` to `ring`
iff mid changed. `Trade` → set `last_trade` only. `Meta` (catalog) → set
`status/start/expiry`, creating the entry if the book hasn't arrived yet.
Guard: drop events older than the applied `recv_ts_ns` (defensive; per-source
order is already preserved by the bus). Note "merging" here means **one global
store across venues** — instrument ids are venue-qualified, so books are never
averaged or merged *between* venues; cross-venue composites would be new derived
instruments, out of scope for v1.
```

**Canonical instrument id:** `"<venue>.<market>.<symbol>"` (`binance.usdt_perp.BTCUSDT`)
and `"polymarket.<condition_id>.<outcome>"` (`polymarket.0xabc….UP`).

---

## 9. Topic / port conventions

| name | mode | producer → consumer |
|---|---|---|
| `market.<venue>.<market>.<symbol>.<stream>` | pub/sub | collectors → processor |
| `market.polymarket.catalog` | pub/sub | poly collector → processor (expiries) |
| `signal.<strategy>` | pub/sub | processor → executor |
| `exec.report` / `exec.fill` / `exec.position` | pub/sub | executor → processor/risk/recorder |
| `exec.intent` | claimed port | (optional) → executor (point-to-point) |

---

## 10. Concurrency & latency (tokio)

- Multi-threaded tokio runtime, single process. Bus router = one task; each
  module = one or more tasks. Swap `InProcBus`→`BrokerBus` to distribute later.
- Hot path (collector recv → publish → rule → signal → executor → place) is
  alloc-light (`Arc<Event>`, interned topics, enum payloads — no dyn on the path);
  persistence/analytics offloaded to writer tasks.
- Instrument **end-to-end latency** `signal_ts → order_ack_ts`; in-proc logic is
  µs-scale, so the *controllable* budget is dominated by WS-in + order-out network.
  Keep our portion < ~100 ms.
- **Venue taker delay (floor we don't control):** Polymarket holds marketable
  orders on crypto up/down markets ~**250 ms** before matching (verified Jun
  2026; see `DESIGN_EXECUTION.md` §1). This is *added* to our latency, so
  realistic signal→match is ~300–400 ms regardless of how fast we are — and the
  backtest's latency-decay makes it potentially edge-determining. The economics
  must be re-validated at latency ≥ 250 ms before live (exec §15 P0).
- **Replay parity:** record raw events; replay through the same processor + rules
  → live and research code paths are identical (signal parity vs the Python
  backtest). The executor places real orders and is not in the replay loop.

---

## 11. Cargo workspace layout

```
trading/                      (cargo workspace)
├── crates/
│   ├── core/         # Event, Payload, data model, Bus/Module/Rule/TradingVenue traits
│   ├── bus/          # InProcBus (router, channels, policies); later BrokerBus
│   ├── collector-binance/    # perp signal feed (reimpl of E:\crypto\collector, §0)
│   ├── collector-polymarket/ # book + catalog (reimpl of e:\poly\crypto_collector\collector\polymarket, §0)
│   ├── processor/    # MarketState, RollingWindow, InstrumentLinker, RuleEngine, Persistor
│   ├── executor/     # risk, order mgmt, TradingVenue: PolymarketClob (testnet|mainnet)
│   └── storage/      # parquet/sqlite writer (arrow2/parquet)
└── app/              # binary: load config, build bus, wire modules, Supervisor, run
```
Key deps: `tokio`, `tokio-tungstenite`, `reqwest`, `serde`/`serde_json`,
`anyhow`, `tracing`, `arrow`/`parquet`; live-only: `ethers` (CLOB EIP-712 signing).

---

## 12. Phasing, failure, config

- **P1 (signals, no executor):** `core` + `bus` + `Module`/Supervisor; both Rust
  collectors publishing (each parity-tested vs its Python reference, §5); processor
  (merge+persist); `PerpMoveRule` → signals; Recorder. Validation = replay recorded
  streams through processor + rules and diff signals vs the **Python backtest**
  (signal parity). **No executor, no orders.**
- **P2 (execution: testnet → small mainnet):** executor (risk, order mgmt,
  OrderManager, PositionManager) + `PolymarketClob` adapter (EIP-712 signing, key
  mgmt) on **Amoy testnet** to prove the plumbing, then **$1 mainnet** for real
  fills; position reconciliation, kill-switch. The fee/latency economics must
  pass the backtest re-run first (`DESIGN_EXECUTION.md` §15 P0).
- **P3:** more rules/strategies; (optional) Binance trading adapter to re-enable
  the perp hedge; `BrokerBus` for true multi-process; metrics/dashboards.
- **Failure:** supervisor restarts modules; bus backpressure policies; idempotent
  `client_id`; reconcile positions on (re)start; kill-switch on stale data / breach.
- **Config (YAML):** bus (queue sizes, policies); collectors (reuse existing
  venue/symbol config); rules (`threshold_bps`, `window_ms`, `yes_bucket`,
  `cooldown_ms`); executor (`hold_ms`, exit, sizes, risk limits,
  `venue.network: testnet|mainnet`); storage paths.

---

## 13. Decisions

- **Runtime:** Rust / async (tokio), single process.
- **Bus:** in-process (`InProcBus`) behind the `Bus` trait; `BrokerBus` swappable later.
- **Execution:** **no paper venue** — single `PolymarketClob` adapter, network
  switch **testnet → mainnet**. Strategy validated by the Python backtest +
  replay (signal parity); execution validated on Amoy testnet then $1 mainnet
  (`DESIGN_EXECUTION.md` §10). Polymarket has no official paper mode, so a local
  fill sim would be a second oracle to maintain — the backtest already is one.
- **Strategy legs:** **Polymarket Up token only** (naked directional); perp is the
  *signal*, not traded. Hedge leg is a later add (P3) — a second leg + Binance adapter.
- **Collectors:** **full Rust reimplementation** of both (chosen) — largest P1
  chunk, gated by the §5 parity test. Hybrid / Python-bridge options were rejected
  in favour of a clean, truly in-process, low-latency system.
```
