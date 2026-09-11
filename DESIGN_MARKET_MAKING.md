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

Reading: uninformed flow exists and is abundant; the naive maker is roughly break-even at the touch; the nowcast
pull removes the informed 11% of fills and lifts P&L/fill to +0.17c (t 5) in the proxy. Caveats that keep this
below GATE 0: one day; the proxy fills on every print (a real 1-contract order behind a 1,400 queue fills on a
subset, and adverse selection concentrates on the fills that do happen); no inventory or expiry risk; the 30 s
and 60 s marks lose significance as directional variance enters. Next: replicate daily (poller running), then
Phase 1 print-based queue fills.

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
