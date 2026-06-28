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
