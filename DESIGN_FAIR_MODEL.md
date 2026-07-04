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
