# DESIGN_LIVE_SIM_CONSISTENCY — validating live signals against the simulation

**Every trader change or model deploy must pass this ladder.** The offline
harness (`tools/analyze_online.py`) is the source of truth for strategy
evidence; the live trader is a different implementation (Rust, tick-driven,
executor-gated) of the same math. This doc is the standing procedure for
proving the two agree — and for locating the bug when they don't.

Why it exists — each of these shipped and was caught only by this process:

| Incident | Symptom | Root cause |
|---|---|---|
| Episode-logic bug (fixed 94eaf4b) | live fired 2-5× more than sim on the same fair series | rule disarmed only on an actual fire; sim disarms on EVERY δ-crossing before the gate |
| Replay feed-freshness bug (3656542) | offline replay produced ZERO signals on flat-market events the live shadow traded | replaying from 50ms snapshots on value-change starved the calibrator's ref-freshness gate |
| max_order_usdc silent rejection (removed 7075f3e) | live fill stream missing the whole [0.5,0.8] band | $1 client-side cap from the size-1 era rejected every order ≥$0.50 after size→2; 61 orders (~43% of band-passers) dropped with no funnel accounting |
| Signal-log rounding (0e9fdd0) | live fires looked ~0.5s misaligned vs reconstruction | `tte:.0f` in the signal reason; now `.1f` + 4dp gap/fair/mid |

---

## 0. Data sources

| Source | What | Notes |
|---|---|---|
| box `data/trader-events.log` | append-only tracing targets: `signal`, `chase`, `exit`, `executor`/`exec`, `hold`, `maker`, `pxprobe`, `featstats`, `fairlog`, `fit`, `gapstats` | survives restarts — the ONLY durable analysis stream ([live.rs](app/src/bin/live.rs) Targets filter) |
| box `/tmp/live-trade.log` | full stdout incl. startup lines | TRUNCATED by every `start_collectors.sh` — never rely on it for series |
| `data/samples/YYYY-MM-DD.csv(.gz)` | 50ms sampler rows (perp px/sizes, cb bid/ask, kalshi book, feature extras from 2026-07-26) | pull box→local `data/samples/`; rich format from 2026-07-12 |
| `data/samples/meta_cache.json` | per-event strike + settlement outcome | `fetch_meta` fills gaps (needs network) |
| `tools/analyze_online.py` | THE sim harness (see the `analyze-online` skill) | `SCAN_STRIDE=1` env → 50ms grid (default 5 = 250ms) |
| `tools/verify_online_fair.py` | Level-1/2 replay check (fairlog + calib vs offline) | promoted from the 2026-07-22 verification |
| `tools/sim_live_signals.py` | live-signal fill/drift analysis vs sampler | P0 = resting ask immediately BEFORE the signal |

Log line formats (grep-able):
- `fairlog: <inst> tte=<s.1f> px=<perp> cb=<cb> mid=<.4f> fair=<.4f> db=<+.5f> dr=<+.5f>` — every in-window event, 10s cadence ([rule.rs](crates/processor/src/rule.rs))
- `fit`/`calib … tte=<B>s rows=<n> db= dr=` — per fit boundary
- `signal: *** SIGNAL #n *** {json}` — reason carries `gap= share= fair= mid= tte=<.1f>s px= entry#`
- `chase: CHASE trade=<id> signal_ask= fill_ask= drift_c= ask_sz= filled=<bool>`
- `gapstats: delta_stats n= mean_abs= mean_signed= p50/p90/p99/max= ge0.03=` — full pre-gate gap distribution, 60s

---

## 1. The consistency ladder

Run the levels in order; each isolates one layer, so a failure localizes the
bug. Levels 1-2 are pre-deploy + first-hours; 3-4 need ~a day of live data.

### Level 1 — Surface parity (exact, pre-deploy)

The Rust `FairSurface` forward must equal the f64 torch forward.

- Golden test: `surface_parity_2p_vs_torch` (600 vectors via
  `tools/export_test_vectors.py`), bar **|Δlogit| < 1e-4**. Required whenever
  the surface arch, feature set, or export path changes.
- Hash gate: calibrator AND rule startup lines must log the **same model
  hash** (the rule refuses mismatched hashes). Check after every restart:
  `grep -E "calibrator up|rule up" /tmp/live-trade.log`.

### Level 2 — Fair-value + calibrator replay (online ≡ offline)

Does the live pipeline (feeds → features → rolling fit → fair) produce the
same number the offline sim would? Uses the `fairlog` stream vs the sampler.

```
# on box: grep -E "fairlog|fit" data/trader-events.log > /tmp/fairlog.txt ; pull it + the sampler day
<pytorch2> tools/verify_online_fair.py fairlog.txt data/samples/<day>.csv[.gz] [model.json]
```

Three checks, three bars (reference = the seqft3e3 full-day run, 2026-07-24,
n=1,712 aligned fairlog samples; judge on median/p90, NOT max — the max is a
fast-move order statistic that grows with n):

| Check | Isolates | Pass bar (healthy reference) |
|---|---|---|
| A: model(logged px/cb + sampler features, logged db/dr) vs logged fair | live feature reconstruction (%mom etc.) | median \|Δfair\| ≤ ~1e-4 (ref 4e-5), p90 ≤ ~1e-3 (ref 5e-4); tail to ~0.04 = features sampled within the ±1.5s alignment tolerance during a fast move |
| B: model(sampler px/cb/features) vs logged fair | full pipeline incl. feed timing | median ≤ ~1e-4 (ref 4e-5), p90 ≤ ~5e-3 (ref 3.1e-3); fat tail ONLY where dpx is large (live-vs-sampler price diffs of tens of $ on sub-100ms moves are genuine timing, not bugs) |
| C: logged (db,dr) vs offline rolling refit on sampler rows (same window/step) | calibrator | \|Δdb\| ≤ 0.010, \|Δdr\| ≤ 0.015 per boundary, BCE decreasing (needs `fit` lines in the input grep, not just fairlog) |

A failing A = feature reconstruction bug. A ok / B failing beyond the fast-move
tail = feed handling. C failing = fit protocol drift (window, step, grace, init).

### Level 3 — Signal reconciliation (counts, episodes, timelines)

Parse live `signal` records; run the harness on the SAME dates, model, δ,
gate, tte window, at live cadence (`--refit-step 15`), and compare.

**Counts will NOT match 1:1 — the residual ratio is structural.** Measured
(seqft3e3 era, δ0.03 ride): live/sim ≈ **1.6× at 250ms grid, ≈1.2× at 50ms
grid** (`SCAN_STRIDE=1`), excess concentrated in whipsaw events. The live
rule evaluates on every tick; crossings that live entirely between grid
samples are real but mostly economically unfillable. Ratios well beyond
~1.5× at 50ms (or sim > live) mean a semantics bug — see §2.

What MUST match:
- **Same events, same directions.** An event the sim trades that live never
  signals (or vice versa) needs an explanation (restart quirk, feed outage —
  see §2), not a shrug.
- **Fair agreement at mutually-sampled instants ≤ 0.5c.** For each live
  signal, compare its logged fair/mid against the sampler row immediately
  before it.

**Timeline case study (do at least one per deploy):** take a live signal with
no sim counterpart, align by exact `ts_ns`, and lay out the live fair/mid/px
against the bracketing 50ms rows. The benign signature: live fires mid-grid
during a sub-100ms perp move, the bracketing rows straddle the δ threshold
(e.g. −0.030 / +0.056 around a −0.119 live gap), and fair matches at every
shared instant. Anything else = investigate.
GOTCHA: always compare against the sample **immediately BEFORE** the signal,
never the nearest — the nearest lands post-MM-reprice on fast moves and
manufactures phantom disagreement.

### Level 4 — Execution funnel audit (signals → orders → fills)

The harness models **NO executor constraints** — it is a signal-level
evaluator (no price band, no size, no balance latch, no cooldown, no
client-side caps). So live trades ⊂ live signals, and the funnel must be
**fully accounted**:

```
signals (signal target)
  → risk-gate rejects, each named: yes_bucket band / tte / cooldown /
    trades-per-min / stale / balance-latch  (executor target)
  → chase attempts (chase target)
  → fills (filled=true) vs misses (adverse-jump declines)
```

**Every non-attempted signal must map to a named gate with a count.** An
unexplained gap IS the bug — max_order_usdc hid 61 rejects (~43% of
band-passers) exactly here. Healthy reference (seqft era, pre-cap-fix):
202 signals → 66 band → (61 cap, now removed) → 75 attempts → 60 fills (80%).

Fill realism: single-IOC +1c chase fills flat-or-favorable drift and misses
adverse jumps; sim `--latency-ms 100 --chase-c 0.01` approximates this
(validated ~56% naive → ~80% live because the chase self-selects). When
comparing P&L, either re-slice sim `trades.csv` to the live-executable set
(band, size) or report the excluded slices explicitly.

---

## 2. Expected structural divergences (do NOT chase to zero)

1. **Evaluation grain:** live = tick-driven; sim = 50/250ms grid. Bounds the
   count ratio (§L3). Irreducible without a tick-level sim.
2. **Episode semantics knobs:** both sides disarm on every δ-crossing before
   the gate (sim semantics, adopted in 94eaf4b) — but `rearm_eps` differs
   (live 0.01, sim 0.02): shallow 1-2c oscillation events re-arm slightly
   differently.
3. **Ride-gate lookback:** the live rule recomputes fair-then from its raw
   ring under CURRENT params (refit-jump immune); the sim uses stored
   fair[j]. Near-threshold share values can differ.
4. **Restart quirk:** a market already live at restart misses its calib fits
   and signals for that one event.
5. **Feed timing:** the sampler is a 50ms snapshot of a continuous stream;
   on sub-100ms moves the live px and the nearest sampler px genuinely
   differ (up to ~$7 observed). This is the fat tail in Level-2 check B.
6. **Executor constraints** exist only live (§L4).

## 3. Hard-won gotchas (each cost a debugging session)

- **Replay feeds need message-freshness, not value-change** — feeding a
  reconstructed stream only when the value changes stalls `ts−age` freshness
  gates on flat markets. Applies to EVERY continuously-published reference
  (perp AND coinbase). (3656542)
- **Compare to the pre-signal sample** (see §L3 gotcha).
- **Persistent series must go to `trader-events.log`** — `start_collectors.sh`
  truncates the stdout log; ~26h of probe series were destroyed across 3
  restarts before 8aab4fc routed analysis targets to the append-only file.
  Any NEW analysis series must be added to the Targets filter in live.rs.
- **Log precision must support alignment**: tte `.1f`, prices 4dp. `.0f` made
  live fires look ~0.5s misaligned.
- **Config edits on the box**: never naive string-splice (a `replace("kalshi:")`
  once landed inside a header comment). Always back up, then
  `python3 -c "import yaml; yaml.safe_load(open('config/kalshi-trade.yaml'))"`
  BEFORE restarting.
- **`pgrep -f` self-match**: a waiter loop grepping for the build it launched
  matches its own command line and spins forever.
- **Seed-noise floor** (offline evals): ±0.0004 KL / ±0.0005 outBCE between
  identical-config reruns — differences within that were never real.

## 4. Deploy checklist

**Before** (offline):
1. Offline eval at LIVE cadence (`--refit-step 15`, w120) with `--fresh-from`;
   headline the event-clustered t on the fresh panel only. Keep TRADE-GATE
   parity with the live rule: `--stale-veto-ms` (harness default 100 =
   live `stale_veto_ms: 100`, 2026-08-01, 2p models only; crossing still
   disarms), `--rearm-eps`, `--demean-window` must mirror the box config —
   any run at other values must say so next to its numbers.
2. Surface changed? → golden parity test + fresh export; note the model hash.
3. `cargo build --workspace` + `cargo test` locally.

**At deploy** (box):
4. Back up `config/kalshi-trade.yaml` (`/tmp/kalshi-trade.yaml.bak-<tag>`);
   edit; YAML-validate. Code via git push→pull (never scp tracked files).
5. Release build (`nice -19 cargo build --release --bin live -j1`), then
   `pkill -f "kalshi-trade\.yaml$"` + `scripts/start_collectors.sh`
   (separate ssh commands).
6. Verify startup: calibrator + rule log the SAME new hash, `executor up`
   line (size, venue), `kalshi venue` line, zero ERROR, heartbeats, sampler
   file growing, `df -h`.

**After** (first hours → days):
7. First events: Level-2 replay (`verify_online_fair.py`) on the fresh
   fairlog; `gapstats` sanity (mean_abs/p99/ge-δ rate in line with the sim's
   gap distribution).
8. ~1 day: Level-3 signal reconciliation + one timeline case study; Level-4
   funnel audit — every reject named, fill rate ~within the reference.
9. ~Days: live KL/outcome-BCE from fairlog vs the offline tables (10s
   in-window samples, model-era filtered); calibration curves once n≳500
   events. Only then compare live P&L to sim — and split YES/NO (one-sided
   P&L = drift/beta, not alpha).
