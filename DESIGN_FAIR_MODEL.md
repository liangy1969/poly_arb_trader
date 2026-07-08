# DESIGN — Fair-Probability Model (KXBTC15M)

Status: v1 adopted (`direct` mode, 2026-07-04). Replaces the threshold-triage
strategy (`PerpMoveRule`: ≥3bps/200ms perp move → chase the Kalshi reprice)
with **probability modeling**: predict the market's fair probability from the
perp price and time-to-expiry, and trade the gap between fair and the current
market price. The execution layer of `DESIGN_EXECUTION.md` (IOC entry, exit
reconciler, caps, real-balance kill switch) is unchanged and consumed as-is.

---

## 0. Why the pivot

Two months of triage-era evidence (live + shadow studies) converged on one
identity: with ~107ms REST order flight, a threshold-triggered **taker chase
pays the move it predicts** — `fill ≈ signal price + reprice`, leaving
`edge ≈ −(spread + fee)` before retrace risk. Latency (FIX), exits (maker
reconciler), and hold length were each fixed in turn; the sign never flipped
(final record: 8 fills, −$0.26). The bottleneck was never execution — it was
that a binary trigger knows nothing about *value*. A fair-value model prices
every regime explicitly (including the near-settle gamma whipsaws that a
trigger can only chase) and turns the strategy from "react to prints" into
"quote a mispricing".

## 1. Model

For an event (one 15-minute up/down market) with strike `K` and remaining
time `tte`:

```
τ  = tte / 900                      (fraction of window remaining, (0,1])
z  = (spot − b_e) / s_e             per-event affine normalization
z′ = z / √τ                         standardized distance (Brownian scaling)
fair = σ( MLP(z′, log τ) )          direct-logit MLP  ("direct" mode, v1)
```

- **`b_e` (bias)** — the dollar level where fair ≈ 0.5: strike + CF-vs-spot
  basis. Per-event.
- **`s_e` (scale)** — the event's vol yardstick (dollars per unit
  uncertainty over the full window). Per-event.
- **`√τ` in the input** — variance of the remaining move grows ∝ τ, so σ of
  the remaining move ∝ √τ: the same $ distance is worth more certainty as
  settle approaches (this is the settle-gamma effect). Encoded in the input
  so the MLP learns *deviations* from diffusion, not the singularity itself.
- **MLP** — 2→32→32→1, tanh; ~1.2k weights. Learned OFFLINE from thousands
  of historical events and **frozen online**.
- **Split of labor (the core of the user's design):** the *shape* of the
  probability surface is universal → offline; the *event-specifics* (strike
  basis, vol regime) are exactly two numbers → fitted online from the live
  event's own history.

### 1.1 Structured vs direct (ablation, 2026-07-04)

Two variants were trained on identical data/seed:

| | structured | **direct (adopted)** |
|---|---|---|
| logit | `z′·exp(u(z′,logτ))`, u clamped ±2 | raw `MLP(z′, logτ)` |
| loss | BCE + priors (Δb², Δρ²) + wd | **BCE only** |
| anchors | fair(K)=0.5 exact, quasi-monotone | none |
| median \|gap\| / p90 | 1.9¢ / 11.8¢ | **1.4¢ / 6.2¢** |
| gap→move(30s) slope | +0.028 | **+0.088** |
| capture @ \|gap\|>5¢ | +0.48¢/30s | **+0.89¢/30s** |
| Brier vs outcome | 0.1139 | **0.1098** (market: 0.1076) |

The anchored form was too restrictive (forces odd symmetry through 0.5;
couples the tails); its larger "gaps" were substantially model error. The
identifiability concern about prior-free (b,s) did **not** materialize in
rehearsal. Residual risk of `direct`: unanchored extrapolation in thin
corners (extreme z′, tiny τ) — sanity-check the surface before each model
ships (§5). Both modes remain switchable in the trainer (`MODE=`).

## 2. Calibration target

**Market probability** (chosen 2026-07-04 over settle outcomes): BCE against
the observed Kalshi price. `fair` = "what the market itself would price at
this (z′, τ)"; a gap is a deviation from the market's own average map, and
the trade is a bet on reversion to that map (consistent with the observed
MM-reprice behavior). Brier-vs-outcome is tracked as a diagnostic only; the
market remains slightly better outcome-calibrated than the model (0.1076 vs
0.1098), as expected for this target.

## 3. Data

Two complementary pipelines:

**3.1 REST backfill (breadth)** — `tools/backfill_kxbtc.py`, runs on the
Ohio box (needs the Tokyo tunnel for Binance): for every settled KXBTC15M
market (~96/day; 5,649 events / 5.07M rows over 60 days), joins per second:
Binance **spot** 1s klines (spot ≈ CF settle source; perp/spot basis is
absorbed by `b_e`), Kalshi tick trades (last-trade YES price, ffill ≤30s),
and market meta — `floor_strike` (exact strike) and `result` /
`expiration_value` (exact labels). Limits: 1s bars, trade prints only (no
book), unknown cross-venue clock skew (±1–2s during fast moves ⇒ phantom
gaps). Good for surface *shape*; poor for microstructure truth.

**3.2 Live 50ms sampler (depth)** — `run.sample_dir` in the `live` binary
(commit 494cf33): every 50ms tick, one row per active Kalshi market with the
CURRENT perp top + YES book top/sizes + tte — **one clock, both sources,
real book**. Covers through settle (`kalshi.min_tte_s: 0`) so the gamma
regime is captured at full resolution. Daily CSVs in `data/samples/`
(~150–200MB/day; cron gzips old days). This is the retraining + validation
substrate as it accumulates, and exactly the data shape the online processor
consumes.

## 4. Training

`tools/train_fair_model.py`, conda env `pytorch2`
(`D:\ProgramData\Anaconda3\envs\pytorch2\python.exe`).

- Per-event params: `b_e = strike + 50·Δb_e`, `s_e = exp(ρ̄ + Δρ_e)` (ρ̄ a
  learned population level; log-param keeps s > 0; $50 scaling equalizes
  gradient magnitudes).
- Loss (`direct`): BCE against market prob clipped to [0.01, 0.99]; rows
  weighted `1/n_rows(event)` so every event counts equally.
- Adam lr 2e-3, batch 16,384, 25 epochs (plateaus ~20), seed 7.
- Split: **chronological by event**, last 10% = validation (rows within an
  event are autocorrelated — random row splits would leak).
- Export: `fair_model.json` (weights + mode + ρ̄) for the Rust processor.

## 5. Validation gates (in order; each gates the next)

1. **Online rehearsal** (in-trainer): freeze the MLP; per held-out event fit
   only `(Δb, Δρ)` on rows with tte > 600s (Adam, 150 steps — the offline
   stand-in for the live rolling refit); predict tte ≤ 600s. Metrics: gap
   distribution, **gap→subsequent-market-move slope** (>0 = market closes
   gaps toward fair = tradeable reversion), capture @ thresholds, Brier.
   v1 (`direct`): slope +0.088, capture +0.89¢/30s @ >5¢ — real but below
   the ~2–4¢ round-trip cost at that cut. Pending: capture broken down by
   gap-size × horizon × tte-bucket to locate the tradeable region, plus
   surface sanity plots (monotonicity/tails, the `direct`-mode risk).
2. **Shadow mode** (phase 4a): the Rust processor runs live with `dry_run`,
   logging (fair, market, gap, would-trade) against real books — measures
   capture at live prices with zero risk.
3. **Live** (phase 4b): existing execution layer; 1-contract sizing, caps,
   $-floor kill switch as validated in the triage era.

## 6. Online architecture (phase 3, to build)

New processor rule `FairGapRule` (replaces `PerpMoveRule` in the signal
path); executor untouched:

- Maintain per-event history of (perp top, YES book, tte) — the sampler's
  in-memory twin.
- Every ~1s: refit `(b_e, s_e)` against the event's history-so-far with the
  frozen MLP (2 params; central finite differences = 5 forward passes per
  step, microseconds; warm-started from the previous fit; recency weighting
  TBD from rehearsal tuning).
- Cold start: first ~2 min of an event trade nothing; `b ← strike` prior
  (from catalog meta) + `s ← ρ̄` until the fit stabilizes.
- Per tick: `fair = σ(MLP(z′, logτ))`; emit a signal when
  `|fair − market| > threshold(τ) + fees + spread buffer`, side = the cheap
  one, with the gap size attached for sizing/priority.
- MLP forward in Rust is a hand-rolled ~30-line function reading
  `fair_model.json`; bit-compat verified against a Python golden vector.

## 7. Risks / open items

- **Capture < costs at the naive cut** — the whole go/no-go rests on the
  gap-size × horizon breakdown and the 50ms-data retrain (clock-skew phantom
  gaps in the backfill dilute measured capture both ways).
- **`direct` extrapolation** in thin corners — gate each export on surface
  sanity plots; consider re-adding a *weak* prior only if plots misbehave.
- **Regime drift** — the surface is a snapshot of market behavior; retrain
  cadence (weekly?) once the 50ms pipeline accumulates.
- **Market-prob target circularity** — the model cannot flag where the
  market is *systematically* wrong, only where it deviates from its own map;
  if reversion edge stays sub-cost, revisit the outcome-calibrated target
  (one env var in the trainer).
- Kalshi taker fee `0.07·p·(1−p)` and spread must be inside the threshold;
  maker entries (resting at fair-adjusted prices) are the natural next step
  since the reconciler infrastructure already supports resting orders.

## 8. Status (2026-07-04)

| piece | state |
|---|---|
| Backfill (60d, 5,649 events) | DONE (`data/model/` on box) |
| 50ms sampler | RUNNING (`data/samples/`, executor disabled) |
| Trainer + ablation | DONE — `direct` adopted; artifacts in scratchpad, weights export working |
| Eval deep-dive (gap×horizon×tte) | NEXT |
| Rust `FairGapRule` | to build (phase 3) |
| Shadow → live | gated on the above |

## 9. VERDICT (2026-07-06, on 239 events of 50ms one-clock data)

Every test below on real bid/ask books, causal per-event `(b,s)` fits,
strictly out-of-sample surface (trained on the May–Jun backfill). Tools:
`tools/sim_50ms.py`, `tools/train_lead.py`.

**The model is an excellent coincident tracker and nothing more.**

1. **Tracking:** held-out KL vs the market mid ≈ 0.008–0.013 nats — the
   2-input surface + per-event affine calibration nearly replicates the
   market's map. The architecture works exactly as designed.
2. **No lead:** predicting `mid(t+n)`, the optimal blend weight on `fair`
   is **α\* = 0.00 for every n ∈ [1s, 60s]**; a residual-on-martingale MLP
   trained directly on future mids learns `g → 0` (and hurts OOS at 30s).
3. **No outcome edge at any tte:** market beats model on outcome-BCE in
   every minute bucket (paired t < 2 everywhere the model looked ahead early
   at n=109 — regressed to null at n=197); the market's advantage in the
   FINAL minute is the only significant cell (+0.073, t = 2.7): the CF
   settlement index converges and the market sees it, the perp doesn't.
4. **Strategies (episode triggers, expanding refits, real costs):** every
   cell of {taker, maker-∞, maker-10s} × {δ = 1–20¢} × {5 entry minutes} ×
   {settle, revert} lies within ~1σ of zero. Maker entries are DOMINATED by
   taker at all δ: the fill itself is adverse information (win% −5 to −9
   pts, immediate — no fill-window escapes it). Reversion gaps close fast
   (93–98% in 10–30s) but mostly from the fair side (perp noise), costing
   the round trip.
5. **Incidental positives that decayed with data or methodology fixes:**
   late-window settle pocket (+19¢@n=7 → +9.7¢@n=12 → negative under
   expanding fits); early-window BCE advantage (died at n=197); last-minute
   reversion column (static-fit artifact).

**Pre-registered for ONE-SHOT confirmation on ~Jul 13 fresh data (no
further slicing of the current dataset):**
- (a) settle-hold, expanding fit, taker, |gap| ≥ 15¢, tte ∈ (60s, 300s]
  — observed +6.6¢/trade, n=72, t ≈ 1.2;
- (b) the minute-4 (tte 240–180s) column — positive in ~every table
  (+1.6…+5.1¢), never individually significant.

**Forward forks:** new information inputs (order flow, book imbalance, the
CF index itself) — the only route left to outcome edge; or the market-making
reframe (two-sided quoting anchored on the model's calibration — its actual
demonstrated strength); or park with the collector accumulating.

## 10. THE MICROSTRUCTURE ARC (2026-07-07/08) — two assets, three venues, and where the null actually lives

One research day, following §9's "new information inputs" fork. Every offline
claim below used the standard protocol (chronological split, causal (Δb,Δρ)
fit tte>300s, core window 300–60s); every strategy number is settle-mode,
expanding fits, taker entries, real books.

### 10.1 Data infrastructure built / discovered
- **`e:\crypto` lake** (discovered): 200GB multi-venue microstructure archive
  — binance/bybit/okx × SPOT+PERP × BTC/ETH, full-depth L2 @~105ms
  (500 lvls), trades w/ aggressor, funding/OI/liquidations, mark+index,
  2026-05-24→ongoing, zero day gaps, ns timestamps (exch/recv/ingest).
  Also `e:\poly\crypto_collector`: full-depth **Kalshi** books Jun 21–24
  (308 events, ~117ms) — Kalshi-side depth still untested.
- **Sampler upgrades** (the 50ms collector, now 23 columns): perp + coinbase
  top-of-book **sizes**; **server-time ages** (`ts_ms − age` = exchange
  stamp; perp feed measured ~128ms behind Binance's clock, coinbase ~13ms);
  **binance SPOT** feed (second collector via tunnel; spot bookTicker has no
  exchange ts — spot_age is local-only); **second app instance for
  KXETH15M** (per-asset config: `run.perp_instrument`, `run.cb_product`,
  `TRADER_EVENTS_LOG`; kill patterns: `pkill -f "kalshi-{trade,eth}\.yaml$"`,
  never combined with relaunch in one ssh command).
- **Directive**: simulations use ONLY online-collected sampler data. Lake
  joins remain legitimate for offline training/calibration studies.

### 10.2 Methodology lessons (each cost us a wrong number first)
1. **Recv-time cross-collector joins flatter fast features.** The perp-imb1
   strategy "edge" (+0.106/ev, t=+1.44) collapsed to +0.029/ev (t=+0.36)
   when the book was re-joined on Binance server time + 100ms feed latency.
   ~Half the apparent gain was sub-100ms freshness production cannot have —
   even though the two clocks agree to <50ms (price-diff xcorr). Native
   sampler columns are causally exact by construction; prefer them always.
2. **Event-cluster everything.** +$16.6/day headlines deflate to t≈0.7; the
   P&L distribution is fat-tailed in BOTH directions at event level.
3. **Episode stacking is auto-adverse.** Median event ≈ 0; the tails are
   25-trade one-sided pileups — the gap re-opens exactly when you're most
   wrong, so uncapped size concentrates in the worst events (worst −$8.9 vs
   best +$2.9). Caps (1–3/event) truncate both tails: variance control, not
   edge (the Jul4–7 uncapped imb1 "+$16.6" was pileups that happened to win
   — flipped to −$8.8 under cap=3).
4. **Final-minute rows must be excluded from surface TRAINING** (was only
   excluded from eval; user-caught). Retraining all six surfaces with
   SKIP_LAST_S=60 flipped same-day BTC base from −$21.6 to +$3.2 (win% 39.5
   →51.8): much of what looked like adverse selection was settlement-regime
   contamination leaking through the smooth log-τ axis. x60 surfaces are the
   standard; they extrapolate below 60s → pair with ENTRY_MIN_TTE_S=60.

### 10.3 Features and assets (offline, standard protocol)
- **Perp book imbalance (BTC)**: tiny, internally consistent KL gain (~2%,
  |t|≤2.0), spread control clean — real information, economically nil.
- **ETH transfers wholesale**: same architecture, priors rescaled (s≈$4,
  b_scale $1.5). Perp 0.00518 < spot 0.00584 KL (venue ordering replicates);
  model ≥ market on outcome-BCE in core window on both inputs. Imbalance
  features STRONGER on ETH (six |t|≥2.5 cells; thin market → book carries
  more unpriced info) — still ~0.0004 nats, not tradable alone.
- **Other venues**: bybit tracks worse than binance on both symbols (0.00679
  BTC / 0.00600 ETH); coinbase-ETH ties spot in core but is WORSE in the
  final minute (sparse prints lag the endgame — opposite of BTC, where
  coinbase owns it).

### 10.4 The central discovery: trigger attribution (ride vs fade)
Decompose each entry's gap-opening over the prior 1s: model_push = s·Δfair
vs market_pull = −s·Δmid (they sum exactly to the signed Δgap).
- **Fading the Kalshi move loses everywhere** it's informed (BTC: −1.7 to
  −9.6¢/trade; the 0.6–0.8 entry-price bucket is its worst face — buying
  favorites at 70¢ that win 56–64%).
- **Riding your input venue's own move is the only persistent green**: with
  clean (x60) surfaces at δ=0.05, the ride class is positive in **6/6**
  model×asset cells (+0.3¢ to +15.9¢/trade), fade negative in 5/6.
- Ride quality orders by settlement relevance: perp < perp+imb1 < coinbase
  (settlement-chain prints; positive in 4/4 coinbase cells even pre-x60).
- Features mostly SHIFT trades between classes (imb1 converts fades into
  rides) rather than improving either class.
- ETH at δ=0.03 dissolves into noise (model moves ≈ quote flutter on the
  thin book) — the gate needs the trigger large vs the venue's noise floor.

### 10.5 The pre-registered spec (frozen 2026-07-08, score cold on fresh days)
**Enter only when the MODEL side opened the gap (1s model-share > 0.75),
δ ≥ 0.05, ENTRY_MIN_TTE_S=60, per-event cap ≤3, x60 surfaces, both assets;
coinbase-input surfaces are the primary candidates.** Every input is
computable live from collected columns. Expectations stated in advance: the
ride/fade split survives; absolute profitability after costs remains the
open question (attribution t's are 1.5–1.9 unclustered — suggestive only).

### 10.6 Open
- Coinbase-native imbalance surface (needs ~2 weeks of sampler cb sizes).
- Kalshi-side depth features (crypto_collector books, Jun 21–24).
- BRTI-replica collector remains the terminal info play.
- okx offline venue cell (extraction done, table not run) — low priority.
