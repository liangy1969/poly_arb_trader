# DESIGN — ETH Fair-Probability Model (KXETH15M)

Status: draft (2026-07-22). Applies the full BTC program (`DESIGN_FAIR_MODEL.md`,
`DESIGN_FAIR_RIDE.md`) to KXETH15M, encoding every methodological learning from
the BTC iteration so ETH starts where BTC finished — the two-price surface,
the live-fit evaluation protocol, the settled feature verdicts, and the honest
statistical bars. Everything here reuses the existing tools; no new pipeline.

---

## 0. What transfers from BTC (the learnings ledger)

Adopt as-is (each was established on BTC at full n under the live-fit protocol):

1. **Two-price parameterization beats feature-decomposition.** Feed BOTH venue
   prices as normalized channels — `u = [z′_perp, z′_cb, ln τ] + extras`,
   `z′_v = ((px_v − b_e)/s_v)/√τ` with **per-venue global scales** `s_v =
   exp(ρ_v + Δρ_e)` and **shared per-event** `(b_e, Δρ_e)`. On BTC this beat
   the basis/dbasis/imb1 (px2imb-style) decomposition with fewer inputs
   (KL 0.00382 vs 0.00408), and the venue scales genuinely differ
   (s_perp ≈ $151 vs s_cb ≈ $136 — the leader runs wider than the settlement
   source). The shared (b, Δρ) means the existing FD-Adam calibrator fits both
   channels with no new parameters (Rust `px2cb_mom`, commit 5074a9d).
2. **%mom momentum is the one durable extra.** `pmom{k} = (p(t)−p(t−k))/p(t−k)`
   on the PERP, k ∈ {60, 120}, z-scored. Settled sub-questions: % vs raw-$ is a
   wash (both z-scored); cb-side momentum is redundant (perp leads).
3. **band5 is the best orderbook feature** — qty imbalance within ±5bps of the
   deep book's own mid. Only feature family with |t|>2 on outcome (dOut t −2.5)
   in the BTC screen; band width barely matters (5/10/25 identical); groups
   overfit (imbK actively hurts; bandK adds nothing over band5 alone).
   CAVEAT carried forward: the spread control also moved (t −1.5), so the
   band-specific margin is the −2.5-vs−1.5 gap — screen ETH with the same
   control and demand the same margin.
4. **Light outcome blend, W_OUT ≈ 0.1.** `L = 0.9·BCE(fair, mid) +
   0.1·BCE(fair, outcome)`, per-event weighted. The optimum sat at ~0.1 on
   three BTC model families in a row (broad minimum, overfits by 0.15); gains
   stack ~additively with band5 on the outcome axis (overlap on the KL axis).
5. **Train on the FULL window (60, 900], never a tte sub-band.** SKIP_LAST_S=60
   (drop the degenerate endgame minute). Restricting training to 60–420
   degraded everything: the 420–900 region anchors the surface shape and the
   feature normalization even though inference only queries tte ≤ 420.
6. **Distill to the mid, not the outcome.** The Kalshi mid is the dense,
   well-calibrated teacher; the settlement is one noisy bit/event. Outcome
   enters ONLY through the light blend (see 4).

Known nulls — do NOT re-spend on these without new cause: volume surge
(vsurge/vol60), order-flow imbalance, liquidations, ΔOI, funding, premium,
basis-level as a feature (non-stationary → rails the clamp), depth-sum
imbalances beyond imb1, cb-side momentum, W_OUT ≥ 0.25.

---

## 1. Training data

Four sources, mirroring BTC. Target range: as much as the lake covers
(BTC used 2026-05-24 → 07-04, 2,981 events after coverage filters; expect a
similar count for ETH).

| # | data | source | tool | notes |
|---|------|--------|------|-------|
| 1 | KXETH15M events (strike, settle, per-second Kalshi mid) | Kalshi REST | `SERIES=KXETH15M python tools/backfill_kxbtc.py …` | writes `events_meta.csv` + `rows.csv.gz` → keep in a NEW dir `data/model-eth/` (do not mix with BTC's `data/model/`) |
| 2 | perp L2 features (mid, spread, imb*, band*, moff) | `E:\crypto` lake | `python tools/extract_l2_features.py <d0> <d1> out.csv.gz USDT_PERP ETHUSDT` | lake has ETHUSDT l2_snapshot from 2026-05-24 ✓ |
| 3 | coinbase ETH-USD per-second prices | coinbase public REST (direct, no tunnel) | the `backfill_venues.fetch_coinbase` runner (see `scratchpad/backfill_cb_full.py` pattern), `CB_PRODUCT=ETH-USD` | **tz lesson (2026-07-21): `calendar.timegm` fix is already in `tools/backfill_venues.py`; any cb data fetched BEFORE that fix on a local (non-UTC) box carries a +25200s key shift** |
| 4 | fresh online slice for OOS | box `data/samples-eth/` (running since 2026-07-15) | scp / analyze_online | confirm the rich column format (cb_bid/ask + perp sizes) before relying on it |

Build the **matched feature file** exactly as BTC (`scratchpad/build_cb_full.py`
pattern): perp mid/L2 from the lake extraction + `basis = cb − perp` column
from source 3, inner-joined on the second. `cb = px + basis` reconstructs the
cb channel in the trainer.

Coverage gates (same as BTC): event needs ≥600 non-NaN seconds; report the
event count and date range with every result.

---

## 2. Model & features

The **candidate set** (train all four, in this order — each was a real layer
on BTC):

| model | extras | purpose |
|---|---|---|
| base | [] | price-only floor |
| 2p+%mom | pmom60, pmom120 | the BTC deployed config |
| 2p+%mom+band5 | + band5 | the BTC best cell |
| control | + spread_bps instead of band5 | the null control — MUST come out ≈ base |

All with: mid target + blend W_OUT=0.1; arch 2→(3+k)→32→32→1 tanh direct;
seed 7; 25 epochs; per-event inverse-count weights; z-score extras on train
rows only, clip ±5.

**ETH scale priors** (the only asset-specific knobs): `RHO0` and `B_SCALE`
must be rescaled to ETH's dollar vol. The trainer's standing note suggests
RHO0 ≈ 4, B_SCALE ≈ 1.5 for ETH (~$1.7k spot; BTC uses 150/50 at ~$65k spot).
**Recalibrate at build time**: set RHO0 to the empirical median 15-min |move|
scale so `exp(rho)` lands near the fitted s (BTC sanity: fitted s ≈ $120–150
vs RHO0 150), and B_SCALE so `b = strike + B_SCALE·Δb` spans a few ticks of
strike ladder with |Δb| ≲ 1. Wrong priors show up as fitted |Δρ| ≫ 1 or the
fit walking b across strikes — check the first ablation run's per-event params.

---

## 3. Evaluation protocol (the part BTC got wrong first — do not shortcut)

1. **Chronological 90/10 event split.** Train on the first 90%, evaluate on
   the last 10% (most recent). No shuffling.
2. **Per-event calibration under the LIVE regime: rolling 120s fit, refit
   every 60s** — to predict tte ∈ (B−60, B], fit (Δb, Δρ) on tte ∈ (B, B+120],
   warm-started, B stepping 300→60. This is `FIT_WINDOW_S=120` /
   `REFIT_STEP_S=60` in `tools/train_l2_imb.py` (now the default).
   **The BTC lesson:** the old static fit (tte>300, all 600s) over-pins b and
   made every feature look useless; under the live fit the features tripled in
   value and became significant. A result that only exists under one fit
   regime is an artifact — report the live-fit number.
3. **Metrics on the tradeable core, tte (60, 300]:**
   - `KL_core` = KL(market mid ‖ fair) — calibration to the mid.
   - `outBCE` = fair's log-loss vs settlement; **`outBCE_mkt`** = the MID's
     log-loss — the bar. Report both; "beats the market" means
     outBCE < outBCE_mkt with the paired t.
   - Paired per-event deltas vs `base` (and vs the incremental predecessor),
     **event-clustered t** (event = one observation).
4. **Fresh-slice forward test** on the box ETH sampler (source 4) via the
   `analyze-online` skill — §4 BCE + §1 strategy sim with `--latency-ms 100
   --chase-c 0.01`. In-regime val is selection; the fresh slice is evidence.
5. **Return sims: three BTC traps, all mandatory checks:**
   - **YES/NO side split** — a one-sided P&L is drift/beta, not alpha (two
     confirmed false positives on BTC: +6.7c and +6.9c, both YES-only).
   - **Entry-grid effect** — a 1s-sampled entry grid acts as a persistence
     filter worth ~+4c/trade vs the live continuous scan on the SAME fair;
     only the harness's scan grid predicts live behavior.
   - **Vetted harness only** (`tools/analyze_online.py` / the skill) — bespoke
     entry loops reproduce the engine only when carefully validated; use the
     standing one.

## 4. Honest bars (unchanged from BTC)

- |event-clustered t| > 2 **and** a fresh-slice replication before believing
  any cell; sub-significant improvements are reported as directional only.
- Every screen includes the **spread control**; a feature's claim is its
  margin over the control, not over base.
- Never promote the currently-best cell (winner's curse: 4+ burned models).
- Expect the end state: on BTC the best model (KL 0.00309 / outBCE 0.2507)
  still LOSES to the market's own quote (0.2498) at settlement. The honest
  goal for ETH is the same frontier: best-calibrated fair for gap/execution
  work — not settlement alpha, unless ETH's (thinner) Kalshi book proves
  otherwise. If ETH's market BCE margin is wider than BTC's, that is the
  interesting result — quantify it.

---

## 5. Deliverables & gates

1. `data/model-eth/` + matched feature file + a data-quality note (events,
   range, cb basis mu/sd — check the basis level is sane, cf BTC's tz bug).
2. Ablation table (base / 2p+%mom / +band5 / control) at full n, live fit,
   with paired t's — the ETH analogue of the BTC table.
3. Blend sweep W ∈ {0, 0.05, 0.1, 0.15} on the winner (expect ~0.1 optimum;
   a different optimum is itself a finding).
4. Fresh-slice §4 BCE + return sim (side-split) via the skill.
5. If (and only if) the ETH model clears the BTC frontier: export
   `models/fair-2pmom-cbbasis-eth.json` (`px2cb_mom` — the Rust surface is
   symbol-agnostic; golden vectors + parity test mandatory, same as
   `surface_parity_2p_vs_torch`), and shadow-probe it on the box ETH process
   (`config/kalshi-eth.yaml`) before any trading-path swap.

Operational notes for the online leg (from the BTC deploy): the calibrator +
rule cross-check the model hash — swap both `model_path` lines; verify with a
`fairlog` replay (online ≡ offline to ~4e-5 median on BTC); band5 online needs
the `binance_depth` collector (`ethusdt@depth@100ms`, top-500 — coverage gate
matters MORE for ETH: ±5bps ≈ $0.85, check the published depth reaches it);
YAML edits on the box config go through `python3 -c "yaml.safe_load(...)"`
before restart.
