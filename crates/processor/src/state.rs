//! `MarketState` — a derived projection of `market.*` events, plus the
//! InstrumentLinker that resolves the strategy target from the reference
//! (DESIGN §6/§8). One global store; ids are venue-qualified.

use std::collections::HashMap;

use arb_core::event::{Event, Payload};
use arb_core::model::*;

use crate::window::RollingWindow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstrumentKind {
    CryptoPerp,
    PredictionOutcome,
    Other,
}

impl InstrumentKind {
    pub fn infer(id: &str) -> Self {
        if id.starts_with("binance.") {
            InstrumentKind::CryptoPerp
        } else if id.starts_with("polymarket.") {
            InstrumentKind::PredictionOutcome
        } else {
            InstrumentKind::Other
        }
    }
}

/// One per canonical instrument, updated in place per market.* event.
#[derive(Clone, Debug)]
pub struct InstrumentState {
    pub instrument: String,
    pub kind: InstrumentKind,
    pub best_bid: f64,
    pub bid_sz: f64,
    pub best_ask: f64,
    pub ask_sz: f64,
    pub mid: f64,
    pub spread: f64,
    pub last_trade: Option<LastTrade>,
    pub exch_ts_ns: i64,
    pub recv_ts_ns: i64,
    pub apply_ts_ns: i64,
    pub seq: u64,
    pub stale: bool,
    pub status: Option<MarketStatus>,
    pub start_ts_ns: Option<i64>,
    pub expiry_ts_ns: Option<i64>,
    pub strike: Option<f64>,
    pub ring: RollingWindow,
}

impl InstrumentState {
    pub fn new(instrument: String, kind: InstrumentKind, ring_cap: usize, horizon_ms: u64) -> Self {
        InstrumentState {
            instrument,
            kind,
            best_bid: 0.0,
            bid_sz: 0.0,
            best_ask: 0.0,
            ask_sz: 0.0,
            mid: f64::NAN,
            spread: 0.0,
            last_trade: None,
            exch_ts_ns: 0,
            recv_ts_ns: 0,
            apply_ts_ns: 0,
            seq: 0,
            stale: false,
            status: None,
            strike: None,
            start_ts_ns: None,
            expiry_ts_ns: None,
            ring: RollingWindow::new(ring_cap, horizon_ms),
        }
    }

    pub fn apply_book(&mut self, b: &BookUpdate, apply_ts_ns: i64) {
        if let Some(&(p, s)) = b.bids.first() {
            self.best_bid = p;
            self.bid_sz = s;
        }
        if let Some(&(p, s)) = b.asks.first() {
            self.best_ask = p;
            self.ask_sz = s;
        }
        let new_mid = if self.best_bid > 0.0 && self.best_ask > 0.0 {
            (self.best_bid + self.best_ask) / 2.0
        } else {
            f64::NAN
        };
        let changed = new_mid.is_finite() && (self.mid.is_nan() || (new_mid - self.mid).abs() > 1e-9);
        self.spread = self.best_ask - self.best_bid;
        self.exch_ts_ns = b.exch_ts_ns;
        self.recv_ts_ns = b.recv_ts_ns;
        self.apply_ts_ns = apply_ts_ns;
        self.stale = false;
        if changed {
            self.ring.push(b.recv_ts_ns, new_mid);
        }
        self.mid = new_mid;
    }
}

#[derive(Clone, Debug)]
pub struct MarketState {
    pub instruments: HashMap<String, InstrumentState>,
    /// reference id -> current Live target id (InstrumentLinker output).
    pub links: HashMap<String, String>,
    link_ref: String,
    link_kind: String,
    ring_cap: usize,
    ring_horizon_ms: u64,
}

impl MarketState {
    pub fn new(link_ref: String, link_kind: String, ring_cap: usize, ring_horizon_ms: u64) -> Self {
        MarketState {
            instruments: HashMap::new(),
            links: HashMap::new(),
            link_ref,
            link_kind,
            ring_cap,
            ring_horizon_ms,
        }
    }

    pub fn on_event(&mut self, ev: &Event) {
        let (cap, hor) = (self.ring_cap, self.ring_horizon_ms);
        match &ev.payload {
            Payload::Book(b) => {
                let st = self.instruments.entry(b.instrument.clone()).or_insert_with(|| {
                    InstrumentState::new(b.instrument.clone(), InstrumentKind::infer(&b.instrument), cap, hor)
                });
                st.seq = ev.seq;
                st.apply_book(b, ev.ts_ns);
            }
            Payload::Trade(t) => {
                if let Some(st) = self.instruments.get_mut(&t.instrument) {
                    st.last_trade = Some(LastTrade { price: t.price, qty: t.qty, side: t.side, ts_ns: t.recv_ts_ns });
                }
            }
            Payload::Meta(m) => {
                let st = self.instruments.entry(m.instrument.clone()).or_insert_with(|| {
                    InstrumentState::new(m.instrument.clone(), InstrumentKind::infer(&m.instrument), cap, hor)
                });
                st.status = Some(m.status);
                st.start_ts_ns = m.start_ts_ns;
                st.expiry_ts_ns = m.expiry_ts_ns;
                if m.strike.is_some() {
                    st.strike = m.strike;
                }
                self.linker_on_meta(m);
            }
            _ => {}
        }
    }

    /// InstrumentLinker: repoint the reference->target link on Live, clear on
    /// Expired/Resolved. Driven purely by catalog lifecycle (DESIGN §6).
    fn linker_on_meta(&mut self, m: &MarketMeta) {
        if m.kind != self.link_kind {
            return;
        }
        match m.status {
            MarketStatus::Live => {
                self.links.insert(self.link_ref.clone(), m.instrument.clone());
            }
            MarketStatus::Expired | MarketStatus::Resolved => {
                if self.links.get(&self.link_ref).map(String::as_str) == Some(m.instrument.as_str()) {
                    self.links.remove(&self.link_ref);
                }
            }
            MarketStatus::Upcoming => {}
        }
    }

    pub fn get(&self, inst: &str) -> Option<&InstrumentState> {
        self.instruments.get(inst)
    }

    pub fn target_of(&self, reference: &str) -> Option<&InstrumentState> {
        let t = self.links.get(reference)?;
        self.instruments.get(t)
    }

    pub fn move_bps(&self, inst: &str, now_ns: i64, window_ms: u64) -> Option<f64> {
        self.instruments.get(inst)?.ring.move_bps(now_ns, window_ms)
    }

    /// Raw mid change over the window (price units). For a [0,1] token, ×100 = cents.
    pub fn move_abs(&self, inst: &str, now_ns: i64, window_ms: u64) -> Option<f64> {
        self.instruments.get(inst)?.ring.move_abs(now_ns, window_ms)
    }

    pub fn tte_ms(&self, inst: &str, now_ns: i64) -> Option<i64> {
        self.instruments.get(inst)?.expiry_ts_ns.map(|e| (e - now_ns) / 1_000_000)
    }
}
