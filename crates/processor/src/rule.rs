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

use crate::fair::FairSurface;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct FairRideCfg {
    pub model_path: String,
    /// Reference price instrument (must match the calibrator's).
    pub reference: String,
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
}

impl Default for FairRideCfg {
    fn default() -> Self {
        FairRideCfg {
            model_path: "models/fair-cb-x60.json".into(),
            reference: "coinbase.BTC".into(),
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
        }
    }
}

struct RideState {
    calib: Option<CalibUpdate>,
    /// (ts_ns, reference px mid, kalshi YES mid) — raw, params-free.
    ring: VecDeque<(i64, f64, f64)>,
    entries: u8,
    armed: bool,
}

impl RideState {
    fn new() -> Self {
        RideState { calib: None, ring: VecDeque::new(), entries: 0, armed: true }
    }
}

pub struct FairRideRule {
    pub cfg: FairRideCfg,
    surface: Arc<FairSurface>,
    model_hash: u64,
    evs: HashMap<String, RideState>,
}

impl FairRideRule {
    pub fn new(cfg: FairRideCfg, surface: Arc<FairSurface>, model_hash: u64) -> Self {
        FairRideRule { cfg, surface, model_hash, evs: HashMap::new() }
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

        let st = self.evs.entry(inst.to_string()).or_insert_with(RideState::new);
        // ring upkeep (always, so history exists before the first calib)
        while st.ring.front().map_or(false, |&(ts, _, _)| now - ts > cfg.lookback_max_ms * 1_200_000) {
            st.ring.pop_front();
        }
        let push = (now, ref_mid, mid);

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

        let fair = self.surface.fair(ref_mid, tte_s, strike, d_b, d_rho);
        let gap = fair - mid;

        // hysteresis / cap
        if !st.armed {
            if gap.abs() <= cfg.rearm_eps {
                st.armed = true;
            }
            st.ring.push_back(push);
            return None;
        }
        if st.entries >= cfg.max_entries_per_event || gap.abs() < cfg.delta {
            st.ring.push_back(push);
            return None;
        }

        // ride gate: youngest ring sample >= lookback_min old
        let lo = now - cfg.lookback_max_ms * 1_000_000;
        let hi = now - cfg.lookback_min_ms * 1_000_000;
        let then = st.ring.iter().rev().find(|&&(ts, _, _)| ts <= hi && ts >= lo).copied();
        let (ts_then, px_then, mid_then) = match then {
            Some(x) => x,
            None => {
                st.ring.push_back(push);
                return None;
            }
        };
        let tte_then = (expiry - ts_then) as f64 / 1e9;
        let fair_then = self.surface.fair(px_then, tte_then, strike, d_b, d_rho);
        let side = if gap > 0.0 { 1.0 } else { -1.0 };
        let mp = side * (fair - fair_then);
        let xp = -side * (mid - mid_then);
        let tot = mp + xp;
        if !(tot > cfg.open_min && mp / tot > cfg.share_min) {
            st.ring.push_back(push);
            return None;
        }

        // ── SIGNAL ──
        st.entries += 1;
        st.armed = false;
        st.ring.push_back(push);
        let entry_no = st.entries;
        let share = mp / tot;
        Some(TradeSignal {
            strategy: "fair_ride".into(),
            ts_ns: now,
            reason: format!(
                "gap={gap:+.3} share={share:.2} fair={fair:.3} mid={mid:.3} tte={tte_s:.0}s px={ref_mid:.2} entry#{entry_no}"
            ),
            reference: cfg.reference.clone(),
            target: inst.to_string(),
            direction: if gap > 0.0 { 1 } else { -1 },
            trigger: Trigger {
                move_bps: mp * 100.0,          // model push, in cents
                window_ms: cfg.lookback_min_ms as u64,
                yes_price: if gap > 0.0 { yask } else { 1.0 - ybid },
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
                self.evs.entry(c.instrument.clone()).or_insert_with(RideState::new).calib = Some(c.clone());
                Vec::new()
            }
            Payload::Book(b) if b.instrument == self.cfg.reference => {
                // reference moved: evaluate every tracked event
                let insts: Vec<String> = self.evs.keys().cloned().collect();
                let now = b.recv_ts_ns;
                insts.iter().filter_map(|i| self.eval(i, state, now)).collect()
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
