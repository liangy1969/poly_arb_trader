# DESIGN: FairRide — the attribution-gated coinbase ride strategy in the live trader

Status: PROPOSED 2026-07-09. Scope: **signal generation only** (BTC,
coinbase-input surface). Executor wiring deferred. Implements the spec frozen
in DESIGN_FAIR_MODEL.md §10.5, restricted to its strongest cell:

> BTC · coinbase-price surface (x60) · enter only when the MODEL side opened
> the gap (1s model-share > 0.75) · |gap| ≥ 0.05 · 60s < tte ≤ 300s ·
> ≤3 entries/event · re-arm at |gap| ≤ 0.02.

Evidence base: ride class +3.0–9.4¢/trade across every BTC cell (both models,
both deltas), cold post-freeze slice +8.2–18.4¢/trade, 100ms chase cost
~1.2–2.2¢ (median 0). Ungated trading is a proven null — the gate IS the
strategy.

## 1. Dataflow (one new feed, one new rule, one new fitter)

```
 coinbase WS ──► CoinbaseCollector ──► market.coinbase.spot.BTC-USD.book ─┬────────────┐
 kalshi WS ────► KalshiCollector ────► market.kalshi.<mkt>.YES.book ──────┤            │
                 (meta + STRIKE) ────► market.kalshi.catalog (Meta) ──────┤            │
                                                                          ▼            ▼
                                                             Processor(MarketState)  Calibrator (own task,
                                                                          │           fits on blocking thread)
                                                            FairRideRule (per event)    │
                                                              │  ├─ FairSurface (MLP)   │ calib.<market>
                                                              │  ├─ latest Calib ◄──────┘ {Δb,Δρ,seq,rows,bce}
                                                              │  └─ ride gate + caps
                                                              ▼
                                                      signal.fair_ride ──► (executor later)
                                                              │
                                                     pxprobe / trader-events.log (shadow)
```

The binance perp feed keeps running (sampler, heartbeat) but plays no role in
this rule. The 50ms sampler is untouched. The **Calibrator is a separate bus
module on its own task**: the processor's hot path never runs a fit.

## 2. Component A — CoinbaseCollector (new crate `collector-coinbase`)

Today the coinbase quote lives in a sampler-private task behind a mutex; the
processor can't see it. Promote it to a first-class collector:

- Same tungstenite `ticker`-channel code as live.rs (lift & move), direct US
  connection, reconnect w/ backoff.
- Publishes `Payload::Book` (single-level, like the perp bookTicker path):
  instrument `coinbase.spot.BTC-USD`, `bids=[(best_bid, best_bid_size)]`,
  `asks=[...]`, `exch_ts_ns` = the RFC3339 `time` field (server stamp — we
  already parse it), `recv_ts_ns` local.
- Config section `coinbase: { enabled, product, ws_url }`.
- live.rs sampler MIGRATES to consuming this instrument off the bus (deletes
  the private task/mutex; one WS connection instead of two).

Trigger cadence note: the ticker channel prints on trades. That is exactly
what the ride gate wants (a real settlement-chain print), and its measured
staleness (median gap ~0.2–1s, age column) is already reflected in every sim
number we trust.

## 3. Component B — strike on MarketMeta (small core + kalshi change)

`MarketMeta` gains `strike: Option<f64>`. The kalshi discovery already GETs
the market objects; parse `floor_strike` and forward it. The rule refuses to
arm an event whose meta lacks a strike (b prior is strike-anchored).

## 4. Component C — FairSurface (pure-Rust model runtime, `processor/src/fair.rs`)

- Loads `fair_model.json` (the exported x60 coinbase surface) at startup:
  arch check (`mode=direct`, hidden=32), layers, `rho_bar`, `b_scale`.
- Forward: `τ = clamp(tte/900, 1e-4, 1)`, `z′ = ((px − b)/s)/√τ`,
  `logit = MLP([z′, ln τ])`, `fair = σ(logit)` — ~40 lines, f32, no deps.
- Ships as a repo artifact `models/fair-cb-x60.json` (30KB; reproducible from
  tools/train_venues.py with the recorded env). Config points at the path.
- Parity gate: `tools/export_test_vectors.py` dumps (px,tte,b,s,logit)
  tuples from torch; a Rust unit test asserts |Δlogit| < 1e-4 on all of them.
  **No live signal until this test exists and passes.**

## 5. Component D — Calibrator (separate module, own task: `crates/processor/src/calib.rs` or crate `calibrator`)

A first-class bus `Module` (like Recorder/Processor), NOT rule-internal —
the fit runs in parallel and the signal path never blocks on it.

- **Own subscription**: `market.#` conflated; maintains its own lightweight
  per-event view (strike from Meta, cb px, kalshi YES mid) — share-nothing
  with the processor, consistent with the rest of the system.
- **Sampling**: per active event, append `(tte_s, cb_px, kalshi_mid)` at 1s
  cadence, skipping rows where either side is stale (>1.5s), the YES spread
  > 0.15, or the book is one-sided — the sim's row filters.
- **Schedule**: first fit when tte first ≤ 300s (requires ≥60 rows; else
  keep waiting — the rule stays disarmed); warm-started refits when tte
  first ≤ 240/180/120/60, each on ALL rows collected so far. Exactly the
  sim's `FIT_MODE=expand`.
- **Objective**: BCE(σ(logit(px_i; b, s)), clip(mid_i, .01, .99)),
  `b = strike + b_scale·Δb`, `s = exp(rho_bar + Δρ)`.
- **Optimizer**: Adam (β=0.9/0.999, eps 1e-8), lr 0.05; 150 steps first fit,
  60 per refit — same as sim. Gradients via central finite differences on
  the 2 params (4 batch-forwards/step; ≤300 rows × 150 steps ≈ sub-10ms).
- **Threading**: the module's tokio task detects boundaries and runs each
  fit via `spawn_blocking` (dedicated OS thread pool) so the 2-core box's
  async workers — collectors, processor, sampler — are never starved even
  during the 150-step first fit. Results return to the module task, which
  publishes.
- **Output**: new payload variant `Payload::Calib(CalibUpdate)` on topic
  `calib.<market_id>`: `{instrument, seq, fitted_at_tte_s, rows, d_b, d_rho,
  bce, ts_ns}`. Recorder/events-log capture it for free (audit trail); a
  `fit`-target log line mirrors it into trader-events.log.
- **Failure containment**: calibrator death only stops NEW signals (rule
  guard below), never corrupts them; it restarts stateless (re-accumulates
  the current event or waits for the next one).

## 6. Component E — FairRideRule (`processor/src/rule.rs`)

Per-event state: `latest_calib: Option<CalibUpdate>`, ring buffer of raw
`(ts_ns, cb_px, kalshi_mid)` (≥1.2s deep, event-driven), `entries: u8`,
`armed: bool`. The processor's subscription widens to `market.#` + `calib.#`
(or the calibrator publishes under `market.calib.*` to reuse the pattern);
`Payload::Calib` events just update `latest_calib` — no computation.

On every book event for the coinbase reference or a live kalshi target:

1. **Eligibility**: meta has strike; `latest_calib` present AND not stale
   (fitted_at within the current schedule window +30s grace — a dead
   calibrator disarms the rule rather than trading on old params);
   `60s < tte ≤ 300s`; kalshi fresh (≤1.5s) and spread ≤ 0.15; cb quote age
   ≤ 5s.
2. `fair = surface(cb_px; b, s)`, `gap = fair − kalshi_mid`, `side = sign`.
3. **Hysteresis/cap**: if `!armed`: re-arm when `|gap| ≤ 0.02`, else return.
   If `entries ≥ 3`: return (event done).
4. **Threshold**: `|gap| ≥ 0.05`.
5. **Ride gate**: take the youngest ring sample ≥1s old (reject if none in
   (1s, 10s]). Compute Δfair = fair(px_now) − fair(px_then) and
   Δmid = mid_now − mid_then, **both ends evaluated under the CURRENT
   (Δb,Δρ)** — the ring stores raw px/mid, not cached fairs, so refit jumps
   can never masquerade as model pushes (a deliberate improvement over the
   sim's known contamination; expect slightly FEWER signals than sim).
   `mp = side·Δfair`, `xp = −side·Δmid`; require `mp + xp > 0.005` and
   `mp/(mp+xp) > 0.75`.
6. **Emit** `TradeSignal{ strategy: "fair_ride", direction: side, reference:
   coinbase instrument, target: kalshi market, reason: "gap=… share=…
   fair=… mid=… tte=…", ttl_ms, hold_ms }`; `entries += 1`, `armed = false`.

Config (new nested section, flat in ProcCfg or `processor.fair_ride`):

```yaml
processor:
  strategy: "fair_ride"
  reference: "coinbase.spot.BTC-USD"
  fair_ride:
    model_path: "models/fair-cb-x60.json"
    delta: 0.05          # |gap| threshold      (spec)
    share_min: 0.75      # ride gate            (spec)
    open_min: 0.005      # 1s gap-opening floor (spec)
    rearm_eps: 0.02      # hysteresis           (spec/sim)
    entry_min_tte_s: 60  # x60 surfaces extrapolate below this (spec)
    entry_max_tte_s: 300
    max_entries_per_event: 3   # tail cap        (spec)
    lookback_ms: [1000, 10000] # ride window bounds
    calib_grace_s: 30          # disarm if calibrator falls behind schedule

calibrator:                    # the separate fitting module
  enabled: true
  reference: "coinbase.spot.BTC-USD"
  model_path: "models/fair-cb-x60.json"   # same surface as the rule
  first_tte_s: 300
  refit_every_s: 60
  steps_first: 150
  steps_refit: 60
  lr: 0.05
  min_rows: 60
  sample_ms: 1000
  max_spread: 0.15
  stale_ms: 1500
```

Every number above is the frozen spec / sim protocol — nothing new is tuned.
`FairSurface` is shared: both the calibrator (for fitting) and the rule (for
fair evaluation) load the same JSON; the Calib event carries a model-file
hash so a mismatched pair refuses to arm.

## 7. Validation gates (in order; each gates the next)

1. **Surface parity** (unit test): Rust forward == torch on exported vectors.
2. **Fit parity** (unit test): Rust EventFit on a canned event's rows lands
   within tolerance of the Python fit's (Δb,Δρ) and per-row fair (<0.005
   prob everywhere in the trade window). Adam+finite-diff vs torch autograd
   won't bit-match; the tolerance is on OUTPUT fair, which is what trades.
3. **Replay parity** (offline binary `replay_fair_ride`): stream a day's
   sampler CSV through MarketState+rule; diff emitted signals against
   `sim_50ms.py` ride-gated trades for the same day/config. Expected
   agreement: same events, entry times within ±250ms (sim scans a grid; the
   rule is event-driven), a few sim-only signals near refit boundaries (the
   contamination we removed). Disagreements beyond that are bugs.
4. **Shadow live**: `executor.enabled: false`; signals + the existing
   pxprobe (50ms price path after each signal) land in trader-events.log.
   Run ≥3 days; score signal rate, share distribution, and pxprobe-implied
   P&L vs the sim on the same days' sampler data.
5. **Executor wiring** — deferred by design. (When it comes: IOC entry at
   ask+chase already exists; the reconciler exit is validated; hold =
   to-settlement is NEW for the executor — needs a "no exit, settle" mode
   and per-event exposure accounting aligned with max_entries_per_event.)

## 8. Explicitly out of scope / deferred decisions

- Executor changes (above). ETH (cold ride cells all red — wait for the
  week's data). imb1 features (needs perp sizes in the rule; additive later).
  Position sizing beyond 1 contract/signal. FIX order path (halves chase
  cost; orthogonal). Coinbase level2 book feed (only if ticker-print cadence
  proves limiting in shadow).

## 9. Honest risks

- **The edge itself is n≈150 trades / 2 days.** Shadow mode exists to grow n
  cold before any capital. The gate's cold record is BTC-only.
- Ticker-channel staleness means some "rides" trigger on prints already
  seconds old; the sim shares this property (numbers already include it).
- Surface is frozen (trained ≤Jul 4); calibration drift over weeks is
  absorbed by (Δb,Δρ) only in level/scale, not shape. Plan: weekly retrain
  with the recorded recipe, new file, replay-parity re-run.
- Kalshi mid as fit target inherits MM quirks at wide spreads; the 0.15
  spread filter is the only guard (same as offline).

## 10. Implementation notes — surface & optimizer

**Surface**: 3 dense layers from JSON (2→32→32→1, tanh), explicit-loop
matvec, f64 throughout (serde-native, deterministic; f32↔f64 delta ~1e-6
≪ the 1e-4 parity tolerance). ~1.2k MACs/forward. BCE in the stable
with-logits form.

**Fit**: full-batch, deterministic. Gradients by CENTRAL FINITE DIFFERENCE
on the 2 params (ε=1e-4, 4 batch-evals/step) — chosen over analytic backprop
because 2 params × few hundred rows makes compute irrelevant (≈270M MACs
worst case ≈ ms on a blocking thread) and a hand-derived chain rule's bugs
would masquerade as optimizer noise. Analytic gradient (dL/dlogit = σ−y
chained through tanh layers; dzp/dΔb = −b_scale/(s√τ), dzp/dΔρ = −zp) is the
recorded optimization if ever needed.

**Adam**: lr .05, β .9/.999, ε 1e-8, bias correction, 150/60 fixed steps,
and — matching the sim's fit_event exactly — FRESH moment state each
(re)fit, params warm-started. No convergence test: determinism + sim parity
over cleverness.

**Parity tolerance is output-space**: |fair_rust − fair_python| < 0.005
across the trade window on a canned real event (the (Δb,Δρ) surface has a
soft ridge; parameter-space comparison would be brittle while fair is what
trades). Surface-only parity: |Δlogit| < 1e-4 on exported random vectors.
