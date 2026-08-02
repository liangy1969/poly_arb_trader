//! Rule engine + `PerpMoveRule` (DESIGN §6). A rule is a pure function of an
//! event + the current `MarketState`, returning zero or more signals.

use arb_core::event::{Event, Payload};
use arb_core::model::{TradeSignal, Trigger};

use crate::state::MarketState;

pub trait Rule: Send {
    fn id(&self) -> &str;
    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal>;
}

pub struct RuleEngine {
    rules: Vec<Box<dyn Rule>>,
}

impl RuleEngine {
    pub fn new(rules: Vec<Box<dyn Rule>>) -> Self {
        RuleEngine { rules }
    }

    pub fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal> {
        let mut out = Vec::new();
        for r in self.rules.iter_mut() {
            out.extend(r.on_event(ev, state));
        }
        out
    }
}

/// Fire when `|perp mid % change over window_ms| >= threshold_bps`, gated by
/// the target's yes-price bucket, cooldown, and min time-to-expiry.
pub struct PerpMoveRule {
    pub strategy: String,
    pub reference: String,
    pub window_ms: u64,
    pub threshold_bps: f64,
    pub yes_bucket: (f64, f64),
    pub cooldown_ms: u64,
    pub min_tte_ms: i64,
    pub hold_ms: u64,
    pub ttl_ms: u64,
    last_signal_ns: i64,
}

impl PerpMoveRule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: String,
        reference: String,
        window_ms: u64,
        threshold_bps: f64,
        yes_bucket: (f64, f64),
        cooldown_ms: u64,
        min_tte_ms: i64,
        hold_ms: u64,
        ttl_ms: u64,
    ) -> Self {
        PerpMoveRule {
            strategy,
            reference,
            window_ms,
            threshold_bps,
            yes_bucket,
            cooldown_ms,
            min_tte_ms,
            hold_ms,
            ttl_ms,
            last_signal_ns: i64::MIN,
        }
    }
}

impl Rule for PerpMoveRule {
    fn id(&self) -> &str {
        &self.strategy
    }

    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal> {
        // Only react to reference-perp book updates.
        match &ev.payload {
            Payload::Book(b) if b.instrument == self.reference => {}
            _ => return Vec::new(),
        }
        let now = ev.ts_ns;

        if self.last_signal_ns != i64::MIN
            && now - self.last_signal_ns < (self.cooldown_ms as i64) * 1_000_000
        {
            return Vec::new();
        }

        let move_bps = match state.move_bps(&self.reference, now, self.window_ms) {
            Some(m) => m,
            None => return Vec::new(),
        };
        if move_bps.abs() < self.threshold_bps {
            return Vec::new();
        }

        // Resolve target via linker; copy out what we need before re-borrowing state.
        let (target_id, yes) = match state.target_of(&self.reference) {
            Some(t) => (t.instrument.clone(), t.mid),
            None => return Vec::new(),
        };
        if !(yes.is_finite() && yes >= self.yes_bucket.0 && yes <= self.yes_bucket.1) {
            return Vec::new();
        }
        match state.tte_ms(&target_id, now) {
            Some(tte) if tte >= self.min_tte_ms => {}
            _ => return Vec::new(),
        }

        // Did the prediction token already move over the SAME window? (cents)
        let target_move_c = state
            .move_abs(&target_id, now, self.window_ms)
            .map(|d| d * 100.0)
            .unwrap_or(f64::NAN);

        self.last_signal_ns = now;
        let direction = if move_bps > 0.0 { 1 } else { -1 };
        vec![TradeSignal {
            strategy: self.strategy.clone(),
            ts_ns: now,
            reason: format!(
                "perp move {move_bps:+.2}bps over {}ms, token {target_move_c:+.2}c",
                self.window_ms
            ),
            reference: self.reference.clone(),
            target: target_id,
            direction,
            trigger: Trigger { move_bps, window_ms: self.window_ms, yes_price: yes, target_move_c },
            hold_ms: self.hold_ms,
            ttl_ms: self.ttl_ms,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arb_core::event::{Event, Payload};
    use arb_core::model::*;

    const PERP: &str = "binance.usdt_perp.BTCUSDT";
    const UP: &str = "polymarket.0xabc.UP";
    const BASE: i64 = 1_700_000_000_000_000_000;

    fn perp_book(ts: i64, px: f64) -> Event {
        Event::new(
            "market.binance.usdt_perp.BTCUSDT.book",
            "t",
            ts,
            0,
            Payload::Book(BookUpdate {
                instrument: PERP.into(),
                bids: vec![(px - 1.0, 5.0)],
                asks: vec![(px + 1.0, 5.0)],
                update_id: None,
                exch_ts_ns: ts,
                recv_ts_ns: ts,
            }),
        )
    }

    fn live_market(st: &mut MarketState) {
        st.on_event(&Event::new(
            "market.polymarket.0xabc.catalog",
            "t",
            BASE,
            0,
            Payload::Meta(MarketMeta {
                instrument: UP.into(),
                kind: "5m_updown".into(),
                status: MarketStatus::Live,
                start_ts_ns: Some(BASE),
                expiry_ts_ns: Some(BASE + 300_000_000_000),
                winner: None,
                min_order_size: None,
                tick_size: None,
                fee_rate: None,
                strike: None,
            }),
        ));
        st.on_event(&Event::new(
            "market.polymarket.0xabc.book",
            "t",
            BASE,
            0,
            Payload::Book(BookUpdate {
                instrument: UP.into(),
                bids: vec![(0.49, 100.0)],
                asks: vec![(0.51, 100.0)],
                update_id: None,
                exch_ts_ns: BASE,
                recv_ts_ns: BASE,
            }),
        ));
    }

    fn run(st: &mut MarketState, rule: &mut PerpMoveRule) -> usize {
        let mut fired = 0;
        for k in 0..=12i64 {
            let ts = BASE + k * 100_000_000;
            let px = 50000.0 * (1.0 + (k as f64) * 0.5 / 10000.0); // +0.5bps/step
            let ev = perp_book(ts, px);
            st.on_event(&ev);
            fired += rule.on_event(&ev, st).len();
        }
        fired
    }

    fn rule() -> PerpMoveRule {
        PerpMoveRule::new("perp_move".into(), PERP.into(), 1000, 3.0, (0.05, 0.95), 2000, 15000, 1000, 1000)
    }

    #[test]
    fn fires_on_move_with_live_target() {
        let mut st = MarketState::new(PERP.into(), "5m_updown".into(), 512, 5000);
        live_market(&mut st);
        let mut r = rule();
        assert!(run(&mut st, &mut r) >= 1, "expected a signal once the 1s window shows >=3bps");
    }

    #[test]
    fn no_target_no_signal() {
        let mut st = MarketState::new(PERP.into(), "5m_updown".into(), 512, 5000);
        let mut r = rule();
        assert_eq!(run(&mut st, &mut r), 0, "no Live target -> no signal");
    }

    #[test]
    fn cooldown_limits_to_one() {
        let mut st = MarketState::new(PERP.into(), "5m_updown".into(), 512, 5000);
        live_market(&mut st);
        let mut r = rule();
        // 2000ms cooldown over a ~1200ms feed -> at most one signal.
        assert_eq!(run(&mut st, &mut r), 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FairRideRule — the attribution-gated coinbase ride strategy
// (DESIGN_FAIR_RIDE §6). Thin by design: the surface is frozen, the (Δb,Δρ)
// arrive as Payload::Calib from the Calibrator; this rule only evaluates
// gates. The ring buffer stores RAW (px, mid) so the 1s lookback re-evaluates
// BOTH ends under the current params — refit jumps can never register as
// model pushes (deliberate improvement over the sim's known contamination).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde::Deserialize;

use arb_core::model::CalibUpdate;

use crate::fair::{FairSurface, FeatureState, MAX_EXTRA};

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct FairRideCfg {
    pub model_path: String,
    /// Reference PRICE instrument (must match the calibrator's).
    pub reference: String,
    /// BASIS reference (coinbase mid) for px2imb `basis = cb − perp`; only used
    /// when the surface has extras. Must match the calibrator's.
    pub basis_reference: String,
    /// |fair − mid| entry threshold (the frozen spec: 0.05).
    pub delta: f64,
    /// Ride gate: model share of the 1s gap-opening (spec: 0.75).
    pub share_min: f64,
    /// Ride gate: minimum signed gap-opening over the lookback (spec: 0.005).
    pub open_min: f64,
    /// Re-arm hysteresis (spec: 0.02).
    pub rearm_eps: f64,
    pub entry_min_tte_s: f64,
    pub entry_max_tte_s: f64,
    pub max_entries_per_event: u8,
    pub lookback_min_ms: i64,
    pub lookback_max_ms: i64,
    /// Calib considered stale once tte is more than one refit period + grace
    /// below its boundary → disarm rather than trade old params.
    pub refit_every_s: f64,
    pub calib_grace_s: f64,
    pub stale_ms: i64,
    pub max_spread: f64,
    pub ref_max_age_ms: i64,
    pub hold_ms: u64,
    pub ttl_ms: u64,
    /// Demeaned trigger (0 = off = raw gap, unchanged behavior): entry fires on
    /// gap MINUS its trailing mean over this window (s) — deviations from the
    /// standing bias, not the bias itself (the sim's --demean-window). The
    /// baseline is a ~1s-grain (ts, gap) ring kept per event; it is PREFILLED
    /// (recomputed from the raw sample ring under the NEW params) on every
    /// calib update, so the per-tick cost is one append + O(1) running mean —
    /// no per-tick surface recomputes, and the baseline is calibration-clean
    /// (a refit shifts gap and baseline in lockstep, like the sim's gbar).
    pub demean_window_s: f64,
    /// Stale-signal veto (0 = off): suppress the ENTRY (the crossing still
    /// disarms — trade-only veto, sim's --stale-veto-ms) when the cb quote the
    /// current fair consumed is older than this many ms. A cb-CONFIRMATION
    /// gate: only trade a dislocation the settlement-anchor venue has itself
    /// just repriced (sweep 2026-08-01: interior optimum ~100ms; vetoed
    /// entries = perp-led, fade-prone).
    pub stale_veto_ms: f64,
}

impl Default for FairRideCfg {
    fn default() -> Self {
        FairRideCfg {
            model_path: "models/fair-cb-x60.json".into(),
            reference: "coinbase.BTC".into(),
            basis_reference: "coinbase.BTC".into(),
            delta: 0.05,
            share_min: 0.75,
            open_min: 0.005,
            rearm_eps: 0.02,
            entry_min_tte_s: 60.0,
            entry_max_tte_s: 300.0,
            max_entries_per_event: 3,
            lookback_min_ms: 1000,
            lookback_max_ms: 10000,
            refit_every_s: 60.0,
            calib_grace_s: 30.0,
            stale_ms: 1500,
            max_spread: 0.15,
            ref_max_age_ms: 5000,
            hold_ms: 0,
            ttl_ms: 500,
            demean_window_s: 0.0,
            stale_veto_ms: 0.0,
        }
    }
}

struct RideState {
    calib: Option<CalibUpdate>,
    /// (ts_ns, reference px mid, px2 (coinbase mid; NAN on one-price surfaces),
    /// kalshi YES mid, raw extra features) — raw, params-free, so the 1s
    /// lookback re-evaluates BOTH ends under the current (Δb,Δρ). Feats are the
    /// extras captured at that sample (all-zero for cb). `fair_then` uses these
    /// so refit jumps can't masquerade as pushes.
    ring: VecDeque<(i64, f64, f64, f64, [f64; MAX_EXTRA])>,
    entries: u8,
    armed: bool,
    /// last `fairlog` sample time for this event (online-vs-offline check).
    fairlog_ns: i64,
    /// demeaned-trigger baseline: (ts_ns, gap) at ~1s grain under the CURRENT
    /// params (prefilled from `ring` on every calib update); `base_sum` keeps
    /// the running sum so the mean is O(1) per eval.
    base_hist: VecDeque<(i64, f64)>,
    base_sum: f64,
    base_last_ns: i64,
    /// Sign of the trigger signal at the last disarm (+1/-1; 0 before any).
    /// Re-arm is DIRECTIONAL: `sig * disarm_dir <= rearm_eps` — same-direction
    /// re-fires must revert to within rearm_eps, but a SIGN FLIP re-arms
    /// immediately (an opposite-direction dislocation is a new episode; the
    /// old |sig| band skipped fast flips, e.g. +0.037 -> -0.05 in one tick
    /// stayed disarmed through the whole opposite move).
    disarm_dir: f64,
}

impl RideState {
    fn new() -> Self {
        RideState {
            calib: None,
            ring: VecDeque::new(),
            entries: 0,
            armed: true,
            fairlog_ns: 0,
            base_hist: VecDeque::new(),
            base_sum: 0.0,
            base_last_ns: 0,
            disarm_dir: 0.0,
        }
    }
}

/// Emit a `gapstats` distribution line at most this often (wall-clock, ns).
const GAP_FLUSH_NS: i64 = 60_000_000_000;
/// Safety cap so a runaway-active market flushes before the Vec grows unbounded.
const GAP_MAX_SAMPLES: usize = 50_000;

/// Summary of the `gap = fair − mid` samples over one flush window. Pure so it
/// can be unit-tested; percentiles are on |gap| (the magnitude compared to δ).
struct GapSummary {
    n: usize,
    mean: f64,      // signed mean (bias)
    abs_mean: f64,  // mean |gap| (typical magnitude)
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
    ge_delta: usize, // how many |gap| >= delta (would-be triggers)
}

fn gap_summary(samples: &[f64], delta: f64) -> Option<GapSummary> {
    let n = samples.len();
    if n == 0 {
        return None;
    }
    let mean = samples.iter().sum::<f64>() / n as f64;
    let ge_delta = samples.iter().filter(|g| g.abs() >= delta).count();
    let mut abs: Vec<f64> = samples.iter().map(|g| g.abs()).collect();
    abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let abs_mean = abs.iter().sum::<f64>() / n as f64;
    let pct = |p: f64| abs[(((p / 100.0) * (n as f64 - 1.0)).round() as usize).min(n - 1)];
    Some(GapSummary { n, mean, abs_mean, p50: pct(50.0), p90: pct(90.0), p99: pct(99.0), max: abs[n - 1], ge_delta })
}

/// Log the distribution of the accumulated gaps and clear the buffer.
fn flush_gap_stats(samples: &mut Vec<f64>, delta: f64) {
    if let Some(s) = gap_summary(samples, delta) {
        tracing::info!(
            target: "gapstats",
            "delta_stats n={} mean_abs={:.4} mean_signed={:+.4} p50={:.4} p90={:.4} p99={:.4} max={:.4} ge{:.2}={}",
            s.n, s.abs_mean, s.mean, s.p50, s.p90, s.p99, s.max, delta, s.ge_delta,
        );
    }
    samples.clear();
}

pub struct FairRideRule {
    pub cfg: FairRideCfg,
    surface: Arc<FairSurface>,
    model_hash: u64,
    /// px2imb feature reconstruction (basis/dbasis/imb1); inactive for cb.
    feats: FeatureState,
    evs: HashMap<String, RideState>,
    /// Rolling `gap = fair − mid` samples (one per fair-model invocation) for the
    /// periodic `gapstats` distribution log; `gap_flush_ns` is the last flush time.
    gap_samples: Vec<f64>,
    gap_flush_ns: i64,
    /// last `featstats` (live extra-feature values) flush time.
    feat_flush_ns: i64,
}

impl FairRideRule {
    pub fn new(cfg: FairRideCfg, surface: Arc<FairSurface>, model_hash: u64) -> Self {
        let feats = FeatureState::new(surface.extras.clone());
        FairRideRule {
            cfg,
            surface,
            model_hash,
            feats,
            evs: HashMap::new(),
            gap_samples: Vec::new(),
            gap_flush_ns: 0,
            feat_flush_ns: 0,
        }
    }

    fn eval(&mut self, inst: &str, state: &MarketState, now: i64) -> Option<TradeSignal> {
        let cfg = &self.cfg;
        let t = state.get(inst)?;
        let r = state.get(&cfg.reference)?;
        let expiry = t.expiry_ts_ns?;
        let strike = t.strike?;
        let tte_s = (expiry - now) as f64 / 1e9;
        if tte_s <= 0.0 {
            self.evs.remove(inst);
            return None;
        }
        // freshness / book-quality gates (mirror the sim's row filters)
        let ref_mid = r.mid;
        if !(ref_mid > 0.0) || now - r.recv_ts_ns > cfg.ref_max_age_ms * 1_000_000 {
            return None;
        }
        let (ybid, yask) = (t.best_bid, t.best_ask);
        if !(ybid > 0.0 && yask > ybid && yask - ybid <= cfg.max_spread)
            || now - t.recv_ts_ns > cfg.stale_ms * 1_000_000
        {
            return None;
        }
        let mid = 0.5 * (ybid + yask);

        // px2imb: reconstruct the extra features NOW. None (no dbasis lookback
        // or stale coinbase) ⇒ this row is not a valid scan point, so skip it
        // (no fair, no ring push) — exactly as the sim drops NaN-feature rows.
        // Empty (all-zero) for the cb surface.
        let feats_now = if self.feats.active() {
            match self.feats.feats(now) {
                Some(f) => f,
                None => return None,
            }
        } else {
            [0.0; MAX_EXTRA]
        };

        // periodic featstats: log the live extra-feature values so the online
        // reconstruction (basis/dbasis/imb1/mom/vsurge) can be validated vs offline.
        if self.feats.active() && now - self.feat_flush_ns >= 60_000_000_000 {
            let kv: Vec<String> = self
                .surface
                .extras
                .iter()
                .zip(feats_now.iter())
                .map(|(n, v)| format!("{}={:.5}", n, v))
                .collect();
            tracing::info!(target: "featstats", "feats {}", kv.join(" "));
            self.feat_flush_ns = now;
        }

        // two-price surface: the coinbase mid is a PRICE INPUT — stale cb (>5s)
        // ⇒ not a valid scan point (mirror the sim dropping NaN-cb rows).
        let px2_now = if self.surface.two_price() {
            match self.feats.cb_price(now) {
                Some(c) => c,
                None => return None,
            }
        } else {
            f64::NAN
        };

        let st = self.evs.entry(inst.to_string()).or_insert_with(RideState::new);
        // ring upkeep (always, so history exists before the first calib). With
        // the demeaned trigger the raw ring must also span the demean window
        // (the calib-update prefill recomputes the baseline from it).
        let keep_ms = cfg.lookback_max_ms.max((cfg.demean_window_s * 1000.0) as i64);
        while st.ring.front().map_or(false, |&(ts, _, _, _, _)| now - ts > keep_ms * 1_200_000) {
            st.ring.pop_front();
        }
        let push = (now, ref_mid, px2_now, mid, feats_now);

        // calib gates
        let c = match st.calib.as_ref() {
            Some(c) => c,
            None => {
                st.ring.push_back(push);
                return None;
            }
        };
        if c.model_hash != self.model_hash
            // stale calib → disarm (a dead calibrator stops NEW signals)
            || tte_s < c.fitted_at_tte_s - cfg.refit_every_s - cfg.calib_grace_s
            || !(tte_s > cfg.entry_min_tte_s && tte_s <= cfg.entry_max_tte_s)
        {
            st.ring.push_back(push);
            return None;
        }
        let (d_b, d_rho) = (c.d_b, c.d_rho);

        let fair = self.surface.fair(ref_mid, px2_now, tte_s, strike, d_b, d_rho, &feats_now);
        let gap = fair - mid;

        // periodic fairlog: one full input/output sample per event every 10s so
        // the ONLINE fair can be replayed offline (same px/cb/tte/db/dr) and
        // asserted equal — the online-vs-offline consistency check.
        if now - st.fairlog_ns >= 10_000_000_000 {
            st.fairlog_ns = now;
            tracing::info!(
                target: "fairlog",
                "{} tte={:.1} px={:.2} cb={:.2} mid={:.4} fair={:.4} db={:+.5} dr={:+.5}",
                inst, tte_s, ref_mid, px2_now, mid, fair, d_b, d_rho
            );
        }

        // record the gap for the periodic distribution log (every fair-model
        // invocation, before any gate — so it captures the FULL |fair−mid|
        // distribution, not just the |gap|>=delta crossings that fire).
        self.gap_samples.push(gap);
        if self.gap_flush_ns == 0 {
            self.gap_flush_ns = now; // first sample: set the baseline, don't flush
        } else if now - self.gap_flush_ns >= GAP_FLUSH_NS || self.gap_samples.len() >= GAP_MAX_SAMPLES {
            flush_gap_stats(&mut self.gap_samples, cfg.delta);
            self.gap_flush_ns = now;
        }

        // demeaned trigger: `sig` = gap − trailing mean (raw gap when off).
        // Baseline upkeep is O(1): evict beyond the window, append at ~1s
        // grain; the heavy recompute happens only at calib updates (prefill).
        let mut sig = gap;
        if cfg.demean_window_s > 0.0 {
            let win_ns = (cfg.demean_window_s * 1e9) as i64;
            while st.base_hist.front().map_or(false, |&(ts, _)| now - ts > win_ns) {
                if let Some((_, g0)) = st.base_hist.pop_front() {
                    st.base_sum -= g0;
                }
            }
            if now - st.base_last_ns >= 1_000_000_000 {
                st.base_hist.push_back((now, gap));
                st.base_sum += gap;
                st.base_last_ns = now;
            }
            if st.base_hist.len() < 5 {
                // baseline not established (event start / post-restart) —
                // mirror the sim's NaN semantics: no trigger activity at all.
                st.ring.push_back(push);
                return None;
            }
            sig = gap - st.base_sum / st.base_hist.len() as f64;
        }

        // hysteresis / cap (on the TRIGGER signal `sig`; raw gap still drives
        // gapstats/fairlog above and the ride gate below)
        if !st.armed {
            // DIRECTIONAL re-arm: same-direction re-fires must revert to within
            // rearm_eps of zero, but a sign flip vs the disarming crossing
            // re-arms immediately (sig * disarm_dir goes negative).
            if sig * st.disarm_dir <= cfg.rearm_eps {
                st.armed = true;
                // episode transition log (target "episode"): re-arm, with how much
                // slack there was to the threshold — a small margin means the arm
                // state is knife-edge and could diverge from the offline replay on
                // sub-cent fair differences.
                tracing::info!(
                    target: "episode",
                    "{} RE-ARM tte={:.1} fair={:.4} mid={:.4} gap={:+.4} margin={:+.4}",
                    inst, tte_s, fair, mid, sig, cfg.rearm_eps - sig * st.disarm_dir
                );
            }
            st.ring.push_back(push);
            return None;
        }
        if st.entries >= cfg.max_entries_per_event || sig.abs() < cfg.delta {
            st.ring.push_back(push);
            return None;
        }

        // SIM EPISODE SEMANTICS (chosen 2026-07-16 to match analyze_online):
        // a |gap|>=delta crossing CONSUMES the armed state regardless of the
        // ride-gate outcome ("one trade per dislocation"). Disarm + count HERE,
        // before the gate; a gate rejection (or missing lookback) then just
        // returns without re-arming, so the next fire must wait for the gap to
        // reset below rearm_eps. (Previously the rule disarmed only on an actual
        // fire, which re-fired far more often than the sim.)
        st.entries += 1;
        st.armed = false;
        st.disarm_dir = if sig > 0.0 { 1.0 } else { -1.0 };
        let entry_no = st.entries;

        // ride gate: youngest ring sample >= lookback_min old
        let lo = now - cfg.lookback_max_ms * 1_000_000;
        let hi = now - cfg.lookback_min_ms * 1_000_000;
        let then = st.ring.iter().rev().find(|&&(ts, _, _, _, _)| ts <= hi && ts >= lo).copied();
        let (ts_then, px_then, px2_then, mid_then, feats_then) = match then {
            Some(x) => x,
            None => {
                tracing::info!(
                    target: "episode",
                    "{} DISARM tte={:.1} fair={:.4} mid={:.4} gap={:+.4} entry#{} gate=NO-LOOKBACK",
                    inst, tte_s, fair, mid, sig, entry_no
                );
                st.ring.push_back(push);
                return None;
            }
        };
        let tte_then = (expiry - ts_then) as f64 / 1e9;
        let fair_then = self.surface.fair(px_then, px2_then, tte_then, strike, d_b, d_rho, &feats_then);
        let side = if sig > 0.0 { 1.0 } else { -1.0 };
        let mp = side * (fair - fair_then);
        let xp = -side * (mid - mid_then);
        let tot = mp + xp;
        let share = if tot != 0.0 { mp / tot } else { 0.0 };
        // episode transition log: a δ-crossing consumed the armed state (DISARM);
        // record the fair/mid, the 1s-lookback reference (fair_then/mid_then), the
        // gate decomposition (push = model's fair move toward the trade, pull =
        // market's mid retreat) and the outcome. A live gate REJECT fires no
        // signal but still disarms — now fully visible instead of inferred, and
        // reconcilable line-for-line with the offline replay's gate.
        let back_ms = (now - ts_then) as f64 / 1e6;
        if !(tot > cfg.open_min && share > cfg.share_min) {
            tracing::info!(
                target: "episode",
                "{} DISARM tte={:.1} fair={:.4} mid={:.4} gap={:+.4} entry#{} gate=REJECT fair_then={:.4} mid_then={:.4} back_ms={:.0} push={:+.4} pull={:+.4} tot={:+.4} share={:.2} (open_min={:.3} share_min={:.2})",
                inst, tte_s, fair, mid, sig, entry_no, fair_then, mid_then, back_ms, mp, xp, tot, share, cfg.open_min, cfg.share_min
            );
            st.ring.push_back(push);
            return None;
        }
        // stale-signal veto (trade-only, sim parity): the crossing above
        // already consumed the armed state; suppress only the ENTRY when the
        // cb quote the fair consumed is older than stale_veto_ms.
        if cfg.stale_veto_ms > 0.0 && self.surface.two_price() {
            let cb_age = self.feats.cb_age_ms(now);
            if cb_age as f64 > cfg.stale_veto_ms {
                tracing::info!(
                    target: "episode",
                    "{} DISARM tte={:.1} fair={:.4} mid={:.4} gap={:+.4} entry#{} gate=PASS veto=STALE cb_age_ms={}",
                    inst, tte_s, fair, mid, sig, entry_no, cb_age
                );
                st.ring.push_back(push);
                return None;
            }
        }
        tracing::info!(
            target: "episode",
            "{} DISARM tte={:.1} fair={:.4} mid={:.4} gap={:+.4} entry#{} gate=PASS fair_then={:.4} mid_then={:.4} back_ms={:.0} push={:+.4} pull={:+.4} tot={:+.4} share={:.2}",
            inst, tte_s, fair, mid, sig, entry_no, fair_then, mid_then, back_ms, mp, xp, tot, share
        );

        // ── SIGNAL ──
        st.ring.push_back(push);
        Some(TradeSignal {
            strategy: "fair_ride".into(),
            ts_ns: now,
            reason: format!(
                "gap={sig:+.4} share={share:.2} fair={fair:.4} mid={mid:.4} tte={tte_s:.1}s px={ref_mid:.2} entry#{entry_no}"
            ),
            reference: cfg.reference.clone(),
            target: inst.to_string(),
            direction: if sig > 0.0 { 1 } else { -1 },
            trigger: Trigger {
                move_bps: mp * 100.0,          // model push, in cents
                window_ms: cfg.lookback_min_ms as u64,
                yes_price: if sig > 0.0 { yask } else { 1.0 - ybid },
                target_move_c: (mid - mid_then) * 100.0,
            },
            hold_ms: cfg.hold_ms,
            ttl_ms: cfg.ttl_ms,
        })
    }
}

impl Rule for FairRideRule {
    fn id(&self) -> &str {
        "fair_ride"
    }

    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal> {
        match &ev.payload {
            Payload::Calib(c) if c.reference == self.cfg.reference => {
                let st = self.evs.entry(c.instrument.clone()).or_insert_with(RideState::new);
                st.calib = Some(c.clone());
                // demeaned trigger: PREFILL the baseline under the NEW params —
                // recompute (fair - mid) from the raw sample ring at ~1s grain
                // so the trailing mean is calibration-clean (the refit shifts
                // gap and baseline in lockstep; the sim's gbar semantics) and
                // the per-tick eval never recomputes history. ~window_s
                // surface forwards once per refit — microseconds.
                if self.cfg.demean_window_s > 0.0 {
                    st.base_hist.clear();
                    st.base_sum = 0.0;
                    st.base_last_ns = 0;
                    if let Some(t) = state.get(&c.instrument) {
                        if let (Some(strike), Some(expiry)) = (t.strike, t.expiry_ts_ns) {
                            for &(ts, px, px2, mid, feats) in st.ring.iter() {
                                if ts - st.base_last_ns < 1_000_000_000 {
                                    continue;
                                }
                                let tte = (expiry - ts) as f64 / 1e9;
                                if tte <= 0.0 {
                                    continue;
                                }
                                let f = self.surface.fair(px, px2, tte, strike, c.d_b, c.d_rho, &feats);
                                st.base_hist.push_back((ts, f - mid));
                                st.base_sum += f - mid;
                                st.base_last_ns = ts;
                            }
                        }
                    }
                }
                Vec::new()
            }
            Payload::Book(b) if b.instrument == self.cfg.reference => {
                // reference (price) moved: refresh features FIRST (perp mid +
                // imb1 sizes) so this tick's eval sees the current snapshot,
                // then evaluate every tracked event.
                if self.feats.active() || self.surface.two_price() {
                    if let (Some(&(bid, bsz)), Some(&(ask, asz))) = (b.bids.first(), b.asks.first()) {
                        if bid > 0.0 && ask > 0.0 {
                            self.feats.on_perp(b.recv_ts_ns, 0.5 * (bid + ask), bsz, asz);
                        }
                    }
                }
                let insts: Vec<String> = self.evs.keys().cloned().collect();
                let now = b.recv_ts_ns;
                insts.iter().filter_map(|i| self.eval(i, state, now)).collect()
            }
            Payload::Trade(t)
                if self.feats.active()
                    && t.instrument.strip_suffix(".vol") == Some(self.cfg.reference.as_str()) =>
            {
                // perp cumulative volume: feed vsurge; no eval (perp/YES ticks drive it)
                self.feats.on_perp_trade(t.recv_ts_ns, t.qty);
                Vec::new()
            }
            Payload::Book(b)
                if (self.feats.active() || self.surface.two_price())
                    && b.instrument == self.cfg.basis_reference =>
            {
                // coinbase moved: update basis/px2; no eval (perp/YES ticks drive it)
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.feats.on_cb(b.recv_ts_ns, 0.5 * (bid + ask));
                    }
                }
                Vec::new()
            }
            Payload::Book(b) if b.instrument.ends_with(".YES") => {
                let inst = b.instrument.clone();
                let now = b.recv_ts_ns;
                self.eval(&inst, state, now).into_iter().collect()
            }
            Payload::Meta(m) => {
                use arb_core::model::MarketStatus::*;
                if matches!(m.status, Expired | Resolved) {
                    self.evs.remove(&m.instrument);
                } else if m.strike.is_some() {
                    self.evs.entry(m.instrument.clone()).or_insert_with(RideState::new);
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod gap_tests {
    use super::gap_summary;

    #[test]
    fn gap_summary_stats() {
        // gaps 0.001, 0.002, …, 0.100
        let samples: Vec<f64> = (1..=100).map(|i| i as f64 * 0.001).collect();
        let s = gap_summary(&samples, 0.03).expect("non-empty");
        assert_eq!(s.n, 100);
        assert!((s.mean - 0.0505).abs() < 1e-9, "mean {}", s.mean);
        assert!((s.abs_mean - 0.0505).abs() < 1e-9);
        assert!((s.max - 0.100).abs() < 1e-9);
        // |gap| >= 0.03 → values 0.030..=0.100 = 71 samples
        assert_eq!(s.ge_delta, 71);
        assert!(s.p50 > 0.045 && s.p50 < 0.056, "p50 {}", s.p50);
        assert!(s.p90 > 0.085 && s.p90 < 0.095, "p90 {}", s.p90);
        assert!(s.p99 > 0.095, "p99 {}", s.p99);
    }

    #[test]
    fn gap_summary_empty() {
        assert!(gap_summary(&[], 0.03).is_none());
    }

    #[test]
    fn gap_summary_signed_vs_abs() {
        // symmetric signs → signed mean ~0, |gap| mean positive
        let s = gap_summary(&[-0.02, 0.02, -0.04, 0.04], 0.03).unwrap();
        assert!(s.mean.abs() < 1e-9, "signed mean {}", s.mean);
        assert!((s.abs_mean - 0.03).abs() < 1e-9);
        assert_eq!(s.ge_delta, 2); // the two |0.04|
    }
}
