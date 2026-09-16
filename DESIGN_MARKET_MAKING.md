# DESIGN: Market-making strategy on KXBTC15M with the nowcast as quote center

Status: PLAN (2026-09-09). Nothing in this document is deployed. Each phase ends in a gate whose
bar is written here before the data for it is examined. The program's history (four losing deploys
below the bar, all at t ~ 1-2 on the discovery slice) is the reason for the gates.

## 0. Objective and the decision this plan produces

Build a two-sided resting-quote strategy on the Kalshi KXBTC15M contracts in which the compact
200 ms nowcast (`models/resid-compact-btc.json`; Rust port on branch `resid-fair-port`) sets the
quote center and the pull / quiet policy, and decide with pre-registered bars whether to run it
with real size. Makers on Kalshi event contracts pay no fee (verified 2026-09-09: Kalshi
"Makers and Takers"; series object `fee_type: quadratic, fee_multiplier: 1`, no maker-fee field);
the July-2026 maker-fee tiers apply to the perps. Ticks: 1c in 10c-90c, 0.1c in the tails; minimum
unit one hundredth of a contract (Rule 13.1(a), effective 2026-01-26).

## 1. What is already measured (13 sampler days, 2026-08-27..09-08, tte 60-300 s)

The nowcast as a signal for a TAKER or a passive ENTRY is closed:

| use | result | file |
|---|---|---|
| taker, hold to settle | -1.1..-3.4c/trade at every delta; fee 1.3-1.5c + half-spread 0.5-1.5c consume the +2.9c gross at delta .02 | `tools/nowcast/fill_ladder.py` |
| scalp to reconvergence | market closes only 0.71c of the gap; deterministic loss | `closure_cells2.py` |
| passive entry at the touch | adversely selected: filled signals -3..-5c, unfilled +1..+14c | `maker_feasibility.py` |
| settlement probability | parity with the market in every prediction bucket (5.7M rows) | `bce_buckets.py` |

The nowcast as a MAKER input is open. Under the pessimistic price-through fill proxy
(`mm_apply.py`, 281k decisions, 1231 events): naive touch quoting -0.456c per decision; pull the
flagged side at 0.25c -19%; one-tick skew on the flagged side -21%; quote only when the model
agrees with the market (|fair-mid| < 0.25c, 89% of the time) -28%. All three slices agree.
The proxy cannot see uninformed fills at the touch, which is the maker's income; that is the
Phase 0 measurement.

## 2. Phases and gates

### Phase 0 - measure the maker economics (1 day of wiring, 1-2 weeks of tape)

Wiring: persist Kalshi trade prints with the taker side. The collector already parses
`taker_outcome_side` into `TradeTick.side` on the bus but nothing stores it. First implementation:
`tools/kalshi_trades_poller.py`, a standalone poller on the box reading the public trades endpoint
for the open KXBTC15M markets every 2 s and appending to `data/samples/trades-YYYY-MM-DD.csv`
(no trader restart, no config change). Later: the same stream from the bus into the sampler.

Analysis (join prints to the 50 ms book and to the nowcast dump `--dump-fair`):
- taker arrival rate at the touch by side, price region, tte;
- share of prints followed by a mid move against the resting side within 1 s and 6 s, unconditional
  and conditional on the nowcast flag (informed vs uninformed split);
- realized spread per fill; queue-adjusted fill probability for an order posted at the touch
  (queue = displayed size ahead at posting);
- maker P&L decomposition per fill: spread capture - adverse selection, at zero fee.

GATE 0: expected P&L per fill > 0 under the nowcast pull-or-quiet policy at t > 2 over >= 1000
events, replicated on a second, later week. Fail = stop; the uninformed flow is too thin.

### Phase 1 - offline maker simulator with print-based fills (3-4 days)

- Fills from prints: a resting order fills when taker volume at its level exceeds the displayed
  queue ahead of it; conservative and optimistic queue assumptions bracket the truth.
- Policy grid, CAPPED, selected on the validation slice only: quote center (mid vs nowcast fair),
  pull threshold, quiet threshold, one-tick skew, Avellaneda-Stoikov inventory skew with
  binary-settlement risk, size, tte window with a hard flatten before expiry.
- Scaffolding: `tools/nowcast/mm_defense.py`, `mm_apply.py` (+ inventory, settlement, print fills).

GATE 1: the selected policy positive on the untouched test slice at t > 2, same sign every day,
not carried by a handful of events.

### Phase 2 - Rust build (1-2 weeks)

- `MakerRule` beside `FairRideRule` consuming the resid fair (`kind: "resid"`, branch
  `resid-fair-port`), producing target quotes per market from the Phase 1 policy.
- Resting-order manager (new): post-only GTC, cancel/replace throttled inside Kalshi rate limits,
  per-market state machine, position truth from authoritative polls via the existing reconciler
  pattern (the old maker exit was removed after an oversell; this is non-negotiable), exposure caps
  per market and total, daily loss limit, balance kill switch, near-expiry flatten.
- Level-2 parity: replay sampler + prints through the Rust quoter; quotes and fills must match the
  simulator (DESIGN_LIVE_SIM_CONSISTENCY.md ladder).

### Phase 3 - micro-live (1-2 weeks)

Quote 0.01 contracts (half a cent of notional at 50c) to obtain real queue positions and real fill
rates for our own orders. GATE 3: realized fill rate and P&L per fill inside the simulator's
bracket over >= 500 events.

### Phase 4 - scale with caps

Size ladder 0.01 -> 1 -> N, each step held for a pre-registered event count with the forward bar
t > 2 on unseen events; enrol in a Liquidity Incentive Program if one appears for KXBTC15M
(none listed on 2026-09-09; the crypto-leader, metals and energy 15-minute series have them:
target size 1000 contracts, period reward, `GET /incentive_programs`).

## 2b. Phase 0 measurements (running log; one day is NOT the gate)

### Day 1: 2026-09-10 (tape 06:13-23:59 UTC, 64 markets, 1.57M prints; `tools/nowcast/maker_tape.py`)

Tape shape: 24.6 prints/s (fills, bursty: median inter-arrival 0 s), median print 9.9 contracts (mean 73, p90 161,
69% fractional), 51% taker-YES. Touch prints inside the 60-300 s window: ~1,150 per market-minute, 55-110 contracts
each, 63k-113k contracts per market-minute at the touch; displayed queue at the touch median ~1,400 contracts.
Spread = one tick everywhere (1c in 10-90c, 0.1c in the tails); 0.3% of sampler book rows crossed (dropped).

Maker at the touch, mark-to-mid, zero fee, every touch print treated as a 1-contract fill (flow-sampling proxy,
no queue model, no inventory), market-clustered t over 64 markets:

| policy | P&L/fill @6 s | @30 s | @60 s | beta-neutral @6 s |
|---|---|---|---|---|
| naive touch quoting | +0.06c (t 1.8) | +0.15c (t 1.5) | +0.10c (t 0.7) | +0.05c (t 1.3) |
| quiet-only (quote when \|fair-mid\| <= 0.25c, 83% of fills) | +0.10c (t 3.4) | +0.13c (t 1.3) | +0.09c (t 0.6) | +0.09c (t 3.2) |
| pull policy (drop the flagged side, 11% of fills) | +0.17c (t 5.1) | +0.19c (t 1.6) | +0.15c (t 1.0) | +0.16c (t 4.9) |
| 1c-tick region (0.10-0.90) only | +0.09c (t 1.8) | +0.22c (t 1.7) | +0.11c (t 0.6) | +0.13c (t 2.8) |
| 1c-tick region + pull policy | +0.32c (t 5.7) | +0.43c (t 2.8) | +0.31c (t 1.6) | +0.33c (t 7.4) |

Decomposition (all touch fills): capture +0.30c, adverse move at 6 s -0.23c. By nowcast flag (compact 0.2 s
model): flagged-against fills -0.87c @6 s (10.8%), quiet +0.11c (82.8%), flagged-for +1.18c (5.9%). Tails (0.1c
tick): capture 0.05c, adverse ~0, P&L ~0. Print size: <1 contract -0.21c @6 s (the only losing size class),
1-10 +0.18c, 10-100 +0.16c, >=100 +0.22c. Directional split is large this day (long-YES fills -6c @60 s,
short-YES +6c: the day's drift, cancels in the beta-neutral column).

Same day, same fills, with the STANDARD 43-input 1 s nowcast as the flag (`models/resid-prune5-btc.json`; it sits
~0.35c from the mid on average vs 0.13c for the compact model, so it flags more fills: 22% against, 15% for):

| policy (1 s fair) | P&L/fill @6 s | @30 s | @60 s | beta-neutral @6 s |
|---|---|---|---|---|
| quiet-only (63% of fills) | +0.10c (t 2.6) | +0.10c (t 1.0) | +0.13c (t 0.9) | +0.09c (t 2.5) |
| pull policy (drop the 22% flagged against) | +0.25c (t 6.0) | +0.24c (t 1.6) | +0.29c (t 1.5) | +0.24c (t 6.1) |
| 1c-tick region + pull | +0.52c (t 6.3) | +0.65c (t 3.0) | +0.72c (t 2.5) | +0.48c (t 9.1) |
| 1c-tick region + quiet-only | +0.28c (t 4.4) | +0.44c (t 3.3) | +0.48c (t 2.5) | +0.27c (t 4.6) |

By flag: against -0.58c @6 s (22.2%), quiet +0.12c (62.6%), for +0.82c @6 s / +1.04c @30 s (14.8%). The pull rule
beats quiet-only because the favourable 15% is the maker's best income; the 1 s fair roughly doubles the compact
model's per-fill P&L under the same rule (+0.25c vs +0.17c; +0.52c vs +0.32c in the 1c region) and keeps
significance out to 30-60 s in the 1c region.

Reading: uninformed flow exists and is abundant; the naive maker is roughly break-even at the touch; the nowcast
pull removes the informed 11% of fills and lifts P&L/fill to +0.17c (t 5) in the proxy. Caveats that keep this
below GATE 0: one day; the proxy fills on every print (a real 1-contract order behind a 1,400 queue fills on a
subset, and adverse selection concentrates on the fills that do happen); no inventory or expiry risk; the 30 s
and 60 s marks lose significance as directional variance enters. Next: replicate daily (poller running), then
Phase 1 print-based queue fills.

### Pull rule vs favourable-only (day 1, 1 s fair, 64 markets, touch fills 60-300 s)

| policy (fills taken) | share of touch fills | P&L/fill @6 s | @30 s | @60 s | bid in the book (1c region) |
|---|---|---|---|---|---|
| pull rule, quote unless flagged against (g >= -0.25c) | 77.7% | +0.25c (t 6.0) | +0.24c | +0.29c | 73.6% of the time |
| favourable-only, quote only when flagged for (g > +0.25c) | 14.8% | +0.72c (t 7.6) | +0.81c | +0.91c | 30.8% |
| pull rule, 1c region | 35.1% | +0.52c (t 6.3) | +0.65c | +0.72c | |
| favourable-only, 1c region | 13.4% | +0.94c (t 6.2) | +0.99c | +1.22c | |

Favourable fills stay good as g grows (g 1-2c: +1.02c @6 s, +2.22c @30 s), unlike the passive-ENTRY study on big
gaps: a resting order that is hit while the model says the move is in its favour is the late/uninformed flow.
Hit rate by state (fills share / time share): quiet 1.46x, favourable 0.48x, against 0.84x - the touch is hit
most in the quiet state, least when the flow is already going our way. Income per unit time in the 1c region:
pull rule ~0.31c per unit vs favourable-only ~0.14c (2.3x), because the pull rule is in the book 2.4x longer and
the quiet fills earn +0.28c; favourable-only earns ~2x per fill and per contract of inventory. Pull rule is the
default when fills are presence-limited (tiny size vs abundant flow); favourable-only when inventory-limited
(small Q). Both are Phase 1 grid rows; the difference is only whether the quiet zone is quoted.

## 2c. Quoting rule v0 (what Phase 1 simulates and Phase 2 builds)

Standard signal (user decision 2026-09-12): the 43-input pruned 1 s nowcast `models/resid-prune5-btc.json`
(fair f, gap g = f - mid in cents). Live twin needs the depth-band features from a deeper perp book, or the
print-only subset; the compact 0.2 s model is the fallback live signal.

Inputs on every evaluation tick (200 ms grid, and immediately on a book change): best bid b / ask a and displayed
sizes, our resting orders, f and g, inventory q (YES-equivalent contracts, signed), time to expiry tau, tick size
(1c in 10-90c, 0.1c in the tails).

R0 Eligibility. Quote only while 0.10 <= mid <= 0.90 (the 0.1c-tick tails capture 0.05c and earn nothing) and
   tau in [90 s, 300 s]. For tau < 90 s quote only the side that reduces |q|; at tau < 45 s cancel everything.
   Residual inventory: micro-live holds to settlement (a taker flatten costs ~1.5c at the money, more than the
   position's expected value); at scale, flatten with a taker order at tau = 45 s.
R1 Price = the touch, always. Bid at b, ask at a. The spread is one tick, so improving would cross, and a quote
   one tick behind the touch fills only when the level is swept, which is adverse by construction. The nowcast
   decides WHETHER a side is quoted, never the level.
R2 Side selection = the pull rule (measured day 1: fills the model flags against -0.87c at 6 s, 11% of fills).
   Bid quoted iff g >= -theta; ask quoted iff g <= +theta; theta = 0.25c. Both sides are on ~83% of the time.
R3 Inventory. Per-market cap Q. q >= +Q: bid off; q <= -Q: ask off. Between, the accumulating side uses a
   tighter threshold theta * (1 - |q|/Q) (a reservation-price shift of f by lambda * q, expressed as a threshold),
   the reducing side keeps theta. This is Avellaneda-Stoikov skew in a one-tick book: skew changes which side is
   quoted, not where.
R4 Update loop (reconcile targets to resting orders; never modify in place, a replace loses priority anyway):
   - for each side compute target = touch price or None (R0-R3);
   - resting order present and (target None or price != resting price) -> cancel;
   - no resting order and target present -> place post-only GTC at target, size S;
   - asymmetric hysteresis: pull immediately when the flag fires (the taker arrives within ~1 s; REST cancel
     ~100 ms is the latency budget); re-join a side pulled by the flag only after its own condition has cleared with
     margin for 1 s: bid when g >= -theta/2 held 1 s, ask when g <= +theta/2 held 1 s (the favourable direction of g is
     never a reason to stay off), so a marginal flag does not churn the queue position (every pull sends us to the
     back of a ~1,400-contract queue); cancels for other reasons (touch move, inventory, tau) re-join without the hold;
   - action budget <= 2 per second per side (verify the Kalshi tier limits before Phase 3);
   - position truth from authoritative polls via the reconciler pattern; local fill accounting is advisory only.
R5 Touch moves. If the touch moves away from our resting order (we are now behind), cancel and re-join at the new
   touch unless R2 says that side is off. If others cancel and our order is alone at the touch with displayed size
   below a minimum (Phase 1 knob, start 100 contracts), pull: a lone quote is the one being picked off.
R6 Size. S = 0.01 contract in micro-live (Phase 3), then 1, then N with the forward bar at each step. At N expect
   partial fills behind the queue.
R7 Expected economics from the day-1 proxy: +0.17c per fill with the pull rule, +0.32c inside the 1c-tick region,
   at 6 s mark-to-mid, zero fee; ~1,150 touch prints per market-minute so fill count is not the constraint. The
   proxy fills on every print; the real fill subset, the re-join cost and inventory variance are what Phase 1
   measures. Knobs for the Phase 1 grid: theta, the re-join rule, Q and lambda, the lone-quote minimum, the tau
   window, S.

## 2d. Phase 1 result, day 1: QUEUE-based fills reverse the proxy (`tools/nowcast/maker_queue_sim.py`)

Model: one contract per side at the touch, joining BEHIND the displayed size; the queue ahead drains with prints
at our price on our side and is bounded by the displayed size (cancels ahead); a fill when prints exhaust the
queue (partial allowed) or when a sweep prints beyond our level; decisions on the 50 ms rows with 300 ms cancel /
150 ms post latency; R0-R5 as in §2c (1c region, tau 90-300 s, Q = 5, lone-quote < 100, re-join hold 1 s).
2026-09-10, 64 markets, standard 1 s fair as the flag, market-clustered t.

| policy | fills / market | P&L per fill @6 s | @30 s | settle per contract | per-market P&L to settlement |
|---|---|---|---|---|---|
| naive | 120 | -1.05c (t -9.3) | -1.17c (t -6.5) | -1.65c (t -3.9) | -85c (t -4.1) |
| pull (theta 0.25c) | 32 | -0.90c (t -4.0) | -0.21c (t -0.5) | -0.71c (t -0.8) | -3c (t -0.2) |
| favourable-only | 16 | -0.89c (t -2.8) | -0.57c (t -1.0) | -2.18c (t -1.2) | -26c (t -1.5) |
| pull, queue ahead halved | 38 | -0.58c (t -3.1) | -0.20c | -0.83c | -14c (t -0.8) |

Where the fills come from (pull): 44% queue-exhaustion fills at +0.04c @6 s / +1.53c @30 s, 56% SWEEP fills
(the level eaten through) at -1.49c / -0.64c; 82-90% of fills arrive within 1 s of posting (a 1,400 queue lasts
~2 s at 500-900 contracts/s per side). Time priority is not the constraint; the direction of the consumption is.
Front-of-queue variants (post only when the displayed size <= 200 / 500): pull -1.39c / -1.49c @6 s, naive -1.08c,
favourable-only 1 fill per market at -1.82c -> being first in line = being picked off (the lone-quote rule).

Sensitivity (pull): cancel latency 300 ms -0.90c @6 s -> 50 ms -0.45c (t -1.9) -> 0 ms -0.06c (t -0.3, per-market
+13c t 0.8); threshold 0.10c at 300 ms -0.65c, at 50 ms -0.80c (more churn). Even the zero-latency bound is
break-even: sweep fills remain 39% at -0.78c.

Measured latency and print-driven reaction (2026-09-12): the live REST order path on the box (us-east-2, warm
connection) is median 10 ms, p75 14 ms, p90 71 ms, p99 153 ms signal-to-fill over 2,699 IOC fills; perp ticks reach
the box within ~50 ms; the book WS is in-region. Re-run with latency SAMPLED from that distribution (lognormal,
median 10 ms, p90 ~46 ms) and, as the higher-frequency variant, a sweep guard that reacts to every print at our
level at tape speed (cancel when prints at our price in the last 100 ms consumed >= f of the queue ahead):

| pull rule, measured latency | fills / market | P&L per fill @6 s | sweep share | per-market to settlement |
|---|---|---|---|---|
| no guard | 16 | -0.26c (t -1.1) | 44% at -1.27c | +7.6c (t +0.5) |
| sweep guard f = 0.5 | 13 | -0.24c (t -1.0) | 55% | -4.7c (t -0.3) |
| sweep guard f = 0.25 | 12 | -0.32c (t -1.2) | 60% | -13.6c (t -0.9) |
| sweep guard f = 0.1 | 12 | -0.28c (t -1.0) | 62% | -20.5c (t -1.7) |
| naive + guard 0.5 | 111 | -0.87c (t -6.5) | 48% | -56c (t -2.9) |

Reacting faster to the tape does not remove the sweep fills: a level is eaten within one print burst, so by the
time the first print at our level is visible the rest of the sweep is already matching; the guard only adds
cancel/re-post churn (actions per market 186 -> 232-298). At measured latency the pull maker sits at -0.26c per
fill, statistically zero on one day but negative in point estimate, with the same 44% sweep share.

Placement-timing tactics from the large-tick literature (Gould-Bonart queue imbalance as the one-tick-ahead
predictor; Lehalle-Mounjid cancel-on-imbalance eroded by latency; Moallemi-Yuan queue-position value; Huang-Lehalle-
Rosenbaum queue-reactive intensities), tested 2026-09-13 on the day-1 tape, pull rule, measured latency:

| tactic | fills / market | P&L per fill @6 s | sweep share | per-market to settlement |
|---|---|---|---|---|
| imbalance pull: cancel when own-side share of the touch sizes < 0.30 | 10 | +0.09c (t +0.35) | 48% | +1.2c (t +0.1) |
| imbalance pull < 0.40 | 7.5 | -0.32c (t -0.9) | 52% | -10.6c |
| imbalance post gate: post only when own share >= 0.50 | 15 | -0.51c (t -2.0) | 44% | -11.4c |
| post gate 0.60 + pull 0.35 | 8 | +0.05c (t +0.15) | 52% | -14.5c |
| naive + imbalance pull 0.35 | 24 | -0.64c (t -2.4) | 54% | -46c (t -2.3) |
| freshness: post only within 500 ms of level formation | 13 | -0.54c (t -2.3) | 44% | -8.0c |
| freshness 500 ms + imbalance pull 0.35 | 7 | -0.34c (t -1.0) | 49% | +0.7c |

The queue-imbalance pull at 0.30 is the first configuration with a non-negative point estimate under queue fills
(+0.09c per fill, +1.2c per market) but it is statistically zero, and 0.40 over-pulls. The sweep share never
drops below ~44%: the sweeps that hurt arrive without a visible imbalance warning at 50 ms resolution. The
imbalance alone does not rescue naive quoting. Freshness (front of the queue by timing) is worse, consistent with
the front-of-queue result above.

Anticipating the level ("ahead of the current bid", 2026-09-13): when g toward a side exceeds a jump threshold and the
OPPOSITE touch level has just cleared (ask moved up for a bid; bid moved down for an ask), post at the price that just
cleared instead of the current touch: it rests as the new best quote, first in a queue nobody displays yet, one tick
ahead of the crowd. Valid for `jump-hold` 500 ms while it does not cross. Pull rule, measured latency, lone rule off:

| variant | fills / market | P&L per fill @6 s | jump fills (share, @6 s) | per-market to settlement |
|---|---|---|---|---|
| reference (pull, lone off) | 17 | -0.28c (t -1.3) | - | +5.8c (t +0.4) |
| jump at g > 0.25c | 18 | -0.14c (t -0.7) | 54%: queue-type +0.18c, sweep-type -0.66c | +9.9c (t +0.7) |
| jump at g > 0.50c | 17 | -0.07c (t -0.3) | 52%: queue-type -0.09c, sweep-type -0.20c | +9.7c (t +0.6) |
| jump at g > 1.00c | 16 | -0.43c (t -1.7) | 52% | +8.3c |
| jump 0.50 + imbalance pull 0.30 | 10 | +0.07c (t +0.2) | 53%: queue-type +0.74c, sweep-type -0.31c | -3.9c |
| favourable-only + jump 0.50 | 5 | -0.34c (t -0.7) | 48%: queue-type +2.51c | -33c (t -2.0) |

Mechanism: a jumped order that gets swept loses -0.2..-0.7c instead of the -1.4..-1.6c of a sweep at the old level,
because it sits at the front of a thin fresh level and the adverse move is at most the tick it just gained; the
queue-type fills at the new level are ~0 to +0.7c. It halves the per-fill loss of the pull rule on day 1 (-0.28c ->
-0.07c) and is the most promising placement rule tested; still statistically zero on one day (t -0.3, per-market
+9.7c t 0.6). Replication on further tape days is the next step before any build.

Day 2 replication (2026-09-12 tape, a quiet Saturday: 96 markets, 296k prints, sweeps 28% of fills vs 44% on 09-10):
naive -0.38c/fill @6 s (t -4.5, -26c/market); pull at the touch +0.11c (t +0.8, -11c/market); pull + jump 0.50 +0.04c
(t +0.2, -21c/market; jump-queue fills +0.61c, jump-sweep -0.68c); pull + jump 0.50 + imbalance pull 0.30 +0.06c
(t +0.4, -3.7c/market). Two-day read: the pull rule at the touch is -0.28c / +0.11c per fill, the jump -0.07c / +0.04c:
both average to ~0; the jump narrows the day-to-day spread (it helps on the busy day where sweeps dominate, adds nothing
on the quiet day). Per-market settlement totals are negative on both days for every variant except day-1 jump.

Perp-tick pull (2026-09-15, "quote faster by leveraging the model's leading input"): cancel a side when the binance
perp moved >= x bps against it within 300 ms, evaluated at perp-tick cadence (lake prints, +40 ms feed delay, measured
cancel latency), 1.5 s cool-down. Pull rule, both tape days:

| perp threshold | 09-10 P&L/fill @6 s (sweep share) | 09-12 P&L/fill @6 s (sweep share) |
|---|---|---|
| none (reference) | -0.28c (44%) | +0.11c (28%) |
| 2 bps (91k triggers/day) | -0.37c (44%) | +0.15c (29%) |
| 3 bps (47k) | -0.28c (43%) | +0.08c (28%) |
| 5 bps (18k) | -0.50c (44%) | +0.18c (28%) |
| 3 bps + jump 0.5 | -0.11c (41%) | +0.01c (27%) |

The sweep share does not move at any threshold: the Kalshi sweeps that hit a resting order are not preceded by a
2-5 bps perp move in the prior 300 ms, so the perp tick is not an earlier warning at this resolution. Faster
reaction (tape-speed sweep guard, perp-tick pull, 0 ms latency bound) never removes the sweep fills; only the
nowcast flag (pull) and the jump change the per-fill number.

Joining the queue early on the prediction (2026-09-16). Three legal forms in a one-tick book: (1) rest before the
flow (priority accrues only while resting; the lever is the re-join hold after a flag clears); (2) STEP BACK instead
of pulling: when a side is flagged against, rest it one tick behind the touch so it is already queued at the level
the predicted move goes to (level-2 depth proxied by the touch size, conservative); (3) the jump (post at the level
that just cleared, first in a fresh queue). Pull rule, measured latency, both days (reference pull: -0.28c / +0.11c):

| variant | 09-10 P&L/fill @6 s (fills/mkt, actions/mkt) | 09-12 P&L/fill @6 s (fills/mkt) |
|---|---|---|
| re-join hold 0 ms | -0.46c t -2.6 (38, 504) | +0.07c (38) |
| re-join hold 250 ms | -0.41c t -1.9 (27, 329) | +0.08c (30) |
| step back at 0.25c | -0.55c t -3.2 (49, 557) | -0.20c t -1.4 (52) |
| step back at 0.50c | -0.54c t -1.9 (19, 178) | -0.21c t -1.1 (24) |
| step back 0.25c + jump 0.5 | -0.53c t -2.9 (51) | -0.24c t -2.0 (52) |

Faster re-joins add fills and churn and lose more on the busy day (the 1 s hold stands). Stepping back is worse than
pulling on BOTH days: the flagged sweeps travel more than one tick often enough that the order waiting at the next
level is the next victim (sweep-type fills -1.3..-1.7c, 3x the fill count of the pull), and the queue-type fills at
the new level (+0.3..+0.4c) do not cover it. The only early-queue form that pays is the jump, because it waits for
the level to actually clear and lands at the front of a fresh queue rather than behind an existing level-2 queue.

SIX-DAY RESULT (2026-09-16; tape days 09-10..15, 545 markets, `tools/nowcast/run_days.sh` + `maker_days.py`; queue
fills, measured latency, lone rule off; market-clustered t over all day x market cells; per-market settlement P&L
counts zero-fill markets as 0):

| variant | P&L/fill @6 s | @30 s | settle per contract | per-MARKET settlement P&L | fills/mkt | sweep share |
|---|---|---|---|---|---|---|
| naive | -0.57c (t -14.1) | -0.74c | -1.31c (t -8.9) | -55.7c (t -8.4) | 157 | 35% |
| pull (theta 0.25c) | -0.13c (t -1.6) | -0.04c | -0.80c (t -1.6) | -8.6c (t -1.6) | 16 | 37% |
| pull + jump 0.5 | -0.06c (t -0.7) | 0.00c | -0.74c (t -1.5) | -11.0c (t -1.9) | 16 | 37% |
| pull + imbalance pull 0.30 | -0.13c (t -1.3) | -0.17c | -1.59c (t -2.5) | -11.5c (t -2.3) | 10 | 42% |
| pull + jump + imbalance | -0.11c (t -1.2) | -0.28c | -1.88c (t -2.8) | -9.8c (t -1.8) | 10 | 41% |
| favourable-only | -0.32c (t -1.7) | -0.43c | -1.30c | +0.9c (t +0.2) | 4 | 52% |

Per day, pull P&L/fill @6 s: 09-10 -0.31c, 09-11 -0.05c, 09-12 +0.12c, 09-13 +0.16c, 09-14 -0.26c, 09-15 -0.62c;
pull + jump: -0.09 / +0.06 / +0.03 / +0.11 / -0.15 / -0.39c. Excluding the 09-11 lake-gap day changes nothing
(pull -0.15c / -12.2c per market, t -2.0). Reading: with six days the pull family is a small, consistent loss per
fill (2 of 6 days positive) and a consistent ~10c loss per market to settlement (t -1.6..-2.3), i.e. the inventory
carried to expiry costs more than the marks suggest; the jump halves the per-fill loss but not the per-market one;
the imbalance pull lowers fills and raises the sweep share; favourable-only trades too little to matter.

VERDICT (day 1, confirmed on six days above): GATE 1 not met. Under realistic fills the touch maker on KXBTC15M loses ~1c per fill at 6 s in
every configuration; the nowcast pull only cuts the fill count and brings the per-market total to ~0 (-3c, t -0.2).
The +0.25c/fill of §2b was the flow-sampling proxy counting fills the queue never delivers. What would have to be
true for Phase 2 to proceed: a fill model calibrated on REAL micro-live fills that shows sweep fills below ~40% of
fills at <= 300 ms, or a sub-100 ms cancel path (FIX) plus a day-replicated positive per-market total. Until then
the maker build stops at Phase 1; the trade tape keeps accumulating for replication.

## 3. Risks

- Uninformed flow may be too thin: this market's takers are fast perp-followers (the informed flow
  a maker pays). Phase 0 answers this.
- Inventory, not the fair, is the strategy's risk: binary settlement, 15-minute expiry, 5-10c sweeps
  in 100 ms. A one-contract inventory at settlement is a +-50c coin flip.
- New order-management code on a real-money path with an oversell in its history.
- Fee rules are moving (perps maker fees; a maker/taker fee swap for orders resting < 5 s on combos).
  Re-verify before Phase 3.
- Institutional makers on Kalshi; the edge is a 20-28% cut in adverse selection, not a moat.
- Live evaluation cadence: the Rust rule evaluates on perp ticks too (2.1x the simulated signal
  count); `eval_on_ref: false` reproduces the simulated semantics.

## 5. Strategy landscape (2026-09-16): what has been tested, what the literature offers, what is left

Literature scanned this round: Feil and Nendel, "Optimal Market Making in Prediction Markets" (arXiv 2607.17991, July
2026): binary-settlement HJB with a terminal penalty gamma*q^2*p(1-p); optimal quotes skew against inventory, spreads
narrow into resolution except near p = 0.5 where settlement risk peaks; the numerical gain is almost entirely risk
reduction (terminal |q| 49 -> 15, VaR -87%) at unchanged expected P&L (12.39 vs 12.47) - i.e. active unwinding buys
safety, not profit; no adverse-selection term in the model. Turbine, "5,000 strategy backtest on KXBTC15M" (30 days
to 2026-04-29): 102 of 4,904 strategies profitable, all one family (fade a hard intra-window move, size 100), median
strategy ROI -14.5%. "Market Informedness and Market-Maker Profitability" (arXiv 2606.05882): makers suffer most when
the informed segment is small but reversal-prone. OddsShopper on the last minute: settlement = 60-sample CF index
average, 45 samples locked before the final 15 s, index-reading programs dominate the final seconds. Plus the earlier
set: Avellaneda-Stoikov, Glosten-Milgrom, Gould-Bonart, Lehalle-Mounjid, Moallemi-Yuan, queue-reactive.

Families tested on our data (harness / queue simulator), all at real fees and measured latency:

| family | best cell found | verdict |
|---|---|---|
| taker on the nowcast gap (0.2 s / 1 s / 3 s / 5 s targets; delta ladder; stability and open gates; tails; bands) | w5 r0.5 delta 0.5c +0.46c (t 0.4) on 09-05..12 | dead: market completes the tick within 1 s, fee+spread > move; every grid cell fails its fresh slice |
| fade the nowcast at large gaps (reverse trades, both legs costed) | OOS bands -0.2..-4c | dead: both directions lose the round trip |
| panic fade (model-free, the Turbine archetype), 79 days | every cell -2..-4c (t -3.6..-7.1) | dead: an April regime |
| settlement fine-tuning of the nowcast | harmful OOS (+0.002 BCE, t 2.4) | closed |
| maker at the touch: naive | -0.57c/fill, -56c/market (t -8) | dead |
| maker: nowcast pull (theta 0.25c) | -0.13c/fill (t -1.6), -8.6c/market (t -1.6), 6 days | ~zero per fill, consistent loss to settlement |
| maker: favourable-only | -0.32c/fill, +0.9c/market on 4 fills/market | too thin to matter |
| maker: jump (post at the level that just cleared) | -0.06c/fill (t -0.7), -11c/market | best per fill, not per market |
| maker: queue-imbalance pull / post gates, level freshness, step-back, faster re-join | all <= pull | none rescue the sweep fills |
| maker: reaction speed (tape sweep guard, perp-tick pull, 0 ms bound) | sweep share unchanged | speed is not the lever |
| maker: inventory caps Q = 1 / 2 | -5.3c / -7.3c per market (t -3.6 / -2.9) | smaller loss, more certain loss |
| maker: taker flatten at tau < 45 s | -14.2c/market | worse than holding: the drift is done by then, the fee is extra |
| maker: price regions (ATM only, wings only) | -3.5c / -8.5c per market | no region is positive |
| maker: volatility regime gate (60 s mid range <= 4/6/10c) | 0.3-3.3 fills/market, per-market -0.3..-5.6c | trades away the volume, remainder not positive |

What the six-day maker numbers say structurally: per-fill marks at 6-30 s are ~0 for the pull family, the loss appears
by settlement (~-10c per market, t -1.6..-2.3) and is NOT recoverable by flattening at 45 s, so it is adverse
selection realised over minutes (the maker ends up holding the side the informed flow left it with), consistent with
the informedness paper. The Feil-Nendel prescription (unwind actively) reduces variance here, not the mean (Q = 1 is
the practical version: -5.3c/market with t -3.6).

Not testable with current data or infrastructure: (a) fills measured on a real micro-live order (the one unknown
that could move every maker number; needs the order manager enabled and 0.01-contract quotes); (b) level-2 Kalshi
depth (the recorder's raw book stream on the box, unexamined); (c) delta-hedging inventory on the perp (needs a
Binance futures account; hedge cost ~0.5-1.3c per contract-equivalent vs a 0.5c capture, so unlikely to pay);
(d) other series (KXETH15M, the 15-min leaders with Liquidity Incentive Programs) where a maker rebate would change
the arithmetic - the poller can be pointed at them; (e) the endgame index-average trade, already found dominated by
faster index readers in the earlier program.

## 4. References

- Avellaneda & Stoikov (2008) reservation price / inventory skew; Glosten & Milgrom (1985) adverse
  selection; Stoikov microprice; "Toward Black-Scholes for Prediction Markets: a market maker's
  handbook" (arXiv 2510.15205): spread ~ 1/sqrt(tau), shrinking inventory limits near expiry,
  short-horizon fair as the quote anchor.
- Buergi, Deng, Whelan (UCD, Jan 2026) "Makers and Takers: the economics of the Kalshi prediction
  market": favorite-longshot bias (<10c buyers lose > 60%), makers out-earn takers.
- Kalshi Rule 13.1 (CFTC filing 2026-01-12), fee schedule July 2026, API changelog, `GET
  /incentive_programs`.
- Memory notes: `analyze-online-harness.md` (all measurements above), `kalshi-fees-and-delay.md`.
