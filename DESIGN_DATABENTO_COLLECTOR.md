# DESIGN — Databento BTC Reference Collector

Status: design (2026-06-24). Goal: replace the SG-tunneled Binance perp BTC stream
with a **US-direct Databento feed** to remove the ~100 ms tunnel latency — the #1
confound in the lead-lag study (`KALSHI_PERP_ARB_REPORT.md` §6, and the live finding
that the perp `recv_ts` lands ~100 ms after the direct Kalshi feed).

The BTC feed is **signal-only** (the perp move triggers the signal; it is never
traded), so this is a pure data-source swap behind the existing `BookUpdate` contract.

---

## 1. Why Databento

- **US-direct, low latency.** The home machine is ~65 ms from AWS us-east (Ohio), where
  Kalshi is. A Databento live gateway in the US puts the BTC feed on the *same* latency
  footing as Kalshi-direct, eliminating the ~100 ms Binance→Singapore→home asymmetry that
  manufactured the "P(up) leads perp" artifact.
- **Nanosecond exchange timestamps.** Databento DBN records carry `ts_event` (exchange
  matching-engine time, ns) and `ts_recv` (gateway capture, ns) — far better than Binance
  bookTicker's millisecond `E`. Enables a clean lead-lag join.
- **Normalized + reliable.** One schema/format across venues, with reconnection/gap
  handling on Databento's side, vs hand-rolling a raw exchange WS.

Cost: Databento live data is **paid** (subscription + per-dataset fees). See §8.

---

## 2. Source selection (the key decision)

The reference must (a) be a liquid BTC price, (b) be US-direct/low-latency, and ideally
(c) cover Kalshi's **24/7** trading hours. Two options:

| option | dataset / symbol | hours | notes |
|---|---|---|---|
| **A. Crypto spot (preferred IF available)** | a 24/7 crypto venue Databento carries (e.g. Coinbase BTC-USD), schema `mbp-1` | **24/7** | matches Kalshi's 24/7 + spot settlement reference; **verify it's in your Databento catalog** |
| **B. CME BTC futures (sure fallback)** | `GLBX.MDP3`, symbol `BTC.c.0` (continuous front month) or `MBT.c.0` (micro), schema `mbp-1` | **weekday only** (Sun 17:00 → Fri 16:00 CT, daily 16:00–17:00 CT break; **closed weekends**) | colocated CME, sub-ms; basis vs spot is stable over a 0.2 s window so the *move* signal is unaffected |

**Recommendation:** check the Databento portal for a **24/7 crypto spot** dataset first
(option A) — it matches Kalshi's hours and settlement. If none is licensed for your
account, **CME futures (option B)** work for the move signal but **lose weekend coverage**
(no signal Fri 16:00 CT → Sun 17:00 CT). For a 24/7 strategy that gap matters; flag it.

(Aside: a free 24/7 alternative is a *direct* Coinbase WS from the US machine — no tunnel,
no Databento cost — but you'd hand-roll the feed and timestamps. Databento buys you
normalization + ns exchange times. Your call on the tradeoff.)

Schema: **`mbp-1`** (market-by-price, top level) is the direct analog of Binance
bookTicker — one record per BBO change with best bid/ask px+sz. (`trades` is available too
if we later want last-trade.)

---

## 3. Collector architecture

New crate **`crates/collector-databento`**, mirroring `collector-binance`, using the
**official `databento` Rust crate** (crates.io) for the Live API.

```
collector-databento/
  src/
    lib.rs        # exports DatabentoCollector, DatabentoCfg
    collector.rs  # LiveClient connect/auth/subscribe + stream loop + reconnect
    book.rs       # Mbp1Msg -> Ticker mapping (price scaling, UNDEF handling)
  Cargo.toml      # deps: databento, tokio, arb-core, arb-bus, serde
```

### Config (`DatabentoCfg`)
```rust
pub struct DatabentoCfg {
    pub instrument: String,   // canonical id we publish, e.g. "databento.glbx.BTC"
    pub dataset: String,      // "GLBX.MDP3" (CME) | the crypto-spot dataset code
    pub schema: String,       // "mbp-1"
    pub stype_in: String,     // "continuous" (BTC.c.0) | "raw_symbol"
    pub symbols: Vec<String>, // ["BTC.c.0"]  (or the spot symbol)
    pub api_key: String,      // or env DATABENTO_API_KEY (key is secret; keep out of repo)
    pub top_n: usize,         // 1 for mbp-1
}
```

### Stream loop (illustrative — verify against the current `databento` crate version)
```rust
let mut client = LiveClient::builder()
    .key(api_key)?            // "db-..." from env DATABENTO_API_KEY
    .dataset(&cfg.dataset)
    .build().await?;

client.subscribe(
    Subscription::builder()
        .symbols(cfg.symbols.clone())
        .schema(Schema::Mbp1)
        .stype_in(SType::Continuous)   // or RawSymbol
        .build()
).await?;
client.start().await?;

let topic = format!("market.databento.{}.book", symbol);   // == reference instrument
while let Some(rec) = client.next_record().await? {
    if let Some(m) = rec.get::<Mbp1Msg>() {
        let l = &m.levels[0];
        if l.bid_px == UNDEF_PRICE || l.ask_px == UNDEF_PRICE { continue; }
        let bid = l.bid_px as f64 / FIXED_PRICE_SCALE;  // 1e9 nano-dollars -> $
        let ask = l.ask_px as f64 / FIXED_PRICE_SCALE;
        publish_book(&bus, &topic, &cfg.instrument, &Ticker {
            bid, ask, bid_sz: l.bid_sz as f64, ask_sz: l.ask_sz as f64,
            exch_ns: m.hd.ts_event as i64,   // ns since epoch, exchange time (already ns!)
            recv_ns: now_ns(),               // LOCAL clock — same basis as Kalshi, for the join
        });
    }
}
```
The `publish_book` and `BookUpdate { instrument, bids:[(bid,sz)], asks:[(ask,sz)],
exch_ts_ns, recv_ts_ns }` are exactly today's — **no downstream change**. Reconnect with
backoff on `next_record` error, same as the Binance session loop.

Notes:
- **`recv_ts_ns = now_ns()` (local)** — keep the same clock as the Kalshi collector so the
  cross-stream lead-lag join is offset-cancelling (as the report relies on). `exch_ts_ns =
  ts_event` is carried for reference/QA.
- **Price scale:** DBN prices are i64 × 1e-9; divide by `FIXED_PRICE_SCALE` (1e9). Skip
  `UNDEF_PRICE` (i64::MAX) levels.

---

## 4. Integration (reference-source selector)

Make the reference collector config-selectable, parallel to the venue selector:

1. **`AppCfg`**: add `pub databento: DatabentoCfg`, and a top-level `reference_source:
   "binance" | "databento"` (default "binance").
2. **`app/src/bin/live.rs`**: start the matching reference collector only —
   ```rust
   match cfg.reference_source.as_str() {
       "databento" => databento.start(bus.clone()).await?,
       _           => binance.start(bus.clone()).await?,
   }
   ```
3. **`processor.reference`**: point at the databento instrument (e.g.
   `databento.glbx.BTC`). The processor's `move_bps` / trigger logic is unchanged — it
   keys on whatever `reference` names.
4. **Recorder / analysis**: unchanged. The recorder captures `market.databento.#` like any
   book stream; the study scripts already glob `stream=book/venue=binance` — add a
   `venue=databento` glob (one-line change) or symlink the loader.

Everything else (mirror, rule, hold/chase logic, the Kalshi side) is untouched.

---

## 5. Validation

Re-run `scripts/kalshi_arb_study.py` / `leadlag_resolution.py` with the databento
reference. Expected: the residual **~10–20 ms "P(up) leads perp"** lead **collapses
toward 0** (no tunnel), confirming the prior lead was network skew, not a real Kalshi
lead. The event-study conclusions (synchronous, momentum-only, fee-bound) should be
unchanged in substance but cleaner. If a *real* lead appears once the tunnel is gone, that
would be the first genuinely tradeable signal — the whole reason to run this test.

---

## 6. Risks / caveats

- **Hours (option B):** CME futures are closed weekends + a daily hour — no signal then.
  Confirm whether a 24/7 spot dataset is available before committing to futures.
- **Cost:** live data is paid; size the subscription to one symbol + `mbp-1`.
- **Symbology:** continuous `BTC.c.0` requires Databento's continuous resolution on the
  live feed; if unavailable, subscribe to the explicit front-month contract and roll it.
- **Crate API drift:** the `databento` Rust crate evolves — verify `LiveClient` /
  `Subscription` / `Schema` names against the installed version before coding.
- **Basis (futures):** futures ≠ spot in *level*, but the *0.2 s move* is ~identical, so
  the trigger is unaffected; only matters if we ever use the level directly.

---

## 7. Phasing

- **P1:** the collector crate + config + app wiring + a smoke run logging
  `market.databento.BTC.book` (recv−exch delta should be single-digit-to-tens of ms, US).
- **P2:** swap `reference_source: databento`, run the collector alongside Kalshi, collect,
  re-run the lead-lag/event study to confirm the residual lead is gone.

---

## 8. Account setup — see the accompanying instructions (next message / SETUP_DATABENTO.md)
```
