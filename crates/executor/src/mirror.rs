//! `ExecBookMirror` (DESIGN_EXECUTION §2.2) — an executor-local projection of the
//! active venue's `market.<venue>.#`: top-of-book per outcome (entry cap / exit
//! floor / sizing) and lifecycle meta per market (expiry/status/winner for tte +
//! settlement). Meta is keyed by `market_id_of` so a traded outcome (UP/DOWN or
//! YES/NO) joins back to the market the catalog published on the YES outcome.
//!
//! Like the processor's `MarketState`, this is a derived view, **not** shared
//! memory across modules. Within the executor it lives behind a mutex shared by
//! the mirror task (writer) and the trade task (reader).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use arb_core::event::{Event, Payload};
use arb_core::model::MarketStatus;

use crate::types::{BookDepth, BookSource, BookTop};
use crate::venue_spec::market_id_of;

#[derive(Clone, Copy, Debug)]
pub struct MarketMetaLite {
    pub status: MarketStatus,
    pub start_ns: i64,
    pub expiry_ns: i64,
    pub min_order_size: Option<f64>,
    pub tick_size: Option<f64>,
    pub fee_rate: Option<f64>,
}

#[derive(Default)]
pub struct ExecBookMirror {
    books: HashMap<String, BookDepth>,       // instrument -> full top-N ladder
    meta: HashMap<String, MarketMetaLite>,   // cid -> lifecycle meta
    winners: HashMap<String, String>,        // cid -> winning instrument (resolved)
    hist: HashMap<String, VecDeque<(i64, f64)>>, // instrument -> recent (recv_ts, best_ask), ~2s
    trades: HashMap<String, VecDeque<(i64, f64)>>, // instrument -> recent (recv_ts, qty), ~2s
}

impl ExecBookMirror {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one `market.polymarket.*` event into the projection.
    pub fn on_event(&mut self, ev: &Event) {
        match &ev.payload {
            Payload::Book(b) => {
                self.books.insert(
                    b.instrument.clone(),
                    BookDepth { bids: b.bids.clone(), asks: b.asks.clone(), recv_ts_ns: b.recv_ts_ns },
                );
                // Rolling top-of-book ask history (~2s) for the pre-signal move.
                let ask = b.asks.first().map(|&(p, _)| p).unwrap_or(0.0);
                let h = self.hist.entry(b.instrument.clone()).or_default();
                h.push_back((b.recv_ts_ns, ask));
                let cutoff = b.recv_ts_ns - 2_000 * 1_000_000;
                while h.front().is_some_and(|&(t, _)| t < cutoff) {
                    h.pop_front();
                }
            }
            Payload::Meta(m) => {
                if let Some(mid) = market_id_of(&m.instrument) {
                    self.meta.insert(
                        mid.to_string(),
                        MarketMetaLite {
                            status: m.status,
                            start_ns: m.start_ts_ns.unwrap_or(0),
                            expiry_ns: m.expiry_ts_ns.unwrap_or(0),
                            min_order_size: m.min_order_size,
                            tick_size: m.tick_size,
                            fee_rate: m.fee_rate,
                        },
                    );
                    if m.status == MarketStatus::Resolved {
                        if let Some(w) = &m.winner {
                            self.winners.insert(mid.to_string(), w.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn top(&self, instrument: &str) -> Option<BookTop> {
        self.books.get(instrument).map(|d| BookTop {
            best_bid: d.bids.first().map(|&(p, _)| p).unwrap_or(0.0),
            bid_sz: d.bids.first().map(|&(_, s)| s).unwrap_or(0.0),
            best_ask: d.asks.first().map(|&(p, _)| p).unwrap_or(0.0),
            ask_sz: d.asks.first().map(|&(_, s)| s).unwrap_or(0.0),
            recv_ts_ns: d.recv_ts_ns,
        })
    }

    pub fn depth(&self, instrument: &str) -> Option<BookDepth> {
        self.books.get(instrument).cloned()
    }

    pub fn meta_of(&self, instrument: &str) -> Option<MarketMetaLite> {
        market_id_of(instrument).and_then(|mid| self.meta.get(mid)).copied()
    }

    /// ms to expiry for the market the instrument belongs to (None if unknown).
    pub fn tte_ms(&self, instrument: &str, now_ns: i64) -> Option<i64> {
        self.meta_of(instrument).map(|m| (m.expiry_ns - now_ns) / 1_000_000)
    }

    /// Best-ask at-or-before `ts_ns` from the rolling history (None if no entry).
    fn ask_at(&self, instrument: &str, ts_ns: i64) -> Option<f64> {
        let h = self.hist.get(instrument)?;
        h.iter().rev().find(|&&(t, _)| t <= ts_ns).map(|&(_, a)| a)
    }

    /// Ask move (¢) on `instrument` from `from_ns` to `to_ns`, from the rolling
    /// history. None if either side lacks history (e.g. just started). Used for
    /// the pre-signal Kalshi move.
    pub fn ask_move_c(&self, instrument: &str, from_ns: i64, to_ns: i64) -> Option<f64> {
        Some((self.ask_at(instrument, to_ns)? - self.ask_at(instrument, from_ns)?) * 100.0)
    }

    /// Record a trade in the rolling ~15s tape, keyed by MARKET (not the traded
    /// outcome) so a `.NO` probe matches the `.YES`-keyed trade tape, and stamped
    /// by EXCHANGE time (`exch_ts`) — Kalshi trades are REST-polled ~1-3s late, so
    /// recv-time would put them outside the probe window. Fed by a non-conflated
    /// Block sub so sweep bursts aren't collapsed.
    pub fn on_trade(&mut self, instrument: &str, exch_ts_ns: i64, qty: f64) {
        let key = market_id_of(instrument).map(str::to_string).unwrap_or_else(|| instrument.to_string());
        let h = self.trades.entry(key).or_default();
        h.push_back((exch_ts_ns, qty));
        let cutoff = exch_ts_ns - 15_000 * 1_000_000;
        while h.front().is_some_and(|&(t, _)| t < cutoff) {
            h.pop_front();
        }
    }

    /// Contracts traded on `instrument`'s MARKET in (from_ns, to_ns] (exchange
    /// time). Query after the poll has caught up (delayed volume pass).
    pub fn traded_between(&self, instrument: &str, from_ns: i64, to_ns: i64) -> f64 {
        let key = market_id_of(instrument).map(str::to_string).unwrap_or_else(|| instrument.to_string());
        self.trades
            .get(&key)
            .map(|h| {
                h.iter().filter(|&&(t, _)| t > from_ns && t <= to_ns).map(|&(_, q)| q).sum()
            })
            .unwrap_or(0.0)
    }

    pub fn winner_of(&self, instrument: &str) -> Option<String> {
        market_id_of(instrument).and_then(|mid| self.winners.get(mid)).cloned()
    }
}

/// `BookSource` over the shared mirror — lets the venue re-read the latest book
/// after its taker delay (DESIGN_EXECUTION §5/§6: the order matches against the
/// then-current book, not the submit-time snapshot).
#[derive(Clone)]
pub struct MirrorSource(pub Arc<Mutex<ExecBookMirror>>);

impl BookSource for MirrorSource {
    fn top(&self, instrument: &str) -> Option<BookTop> {
        self.0.lock().unwrap().top(instrument)
    }
    fn depth(&self, instrument: &str) -> Option<BookDepth> {
        self.0.lock().unwrap().depth(instrument)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arb_core::event::Event;
    use arb_core::model::{BookUpdate, MarketMeta};

    fn book_ev(inst: &str, bid: f64, ask: f64, recv: i64) -> Event {
        Event::new(
            "market.polymarket.0xabc.book",
            "poly",
            recv,
            0,
            Payload::Book(BookUpdate {
                instrument: inst.into(),
                bids: vec![(bid, 100.0)],
                asks: vec![(ask, 80.0)],
                update_id: None,
                exch_ts_ns: recv,
                recv_ts_ns: recv,
            }),
        )
    }

    #[test]
    fn tracks_book_and_meta_across_tokens() {
        let mut m = ExecBookMirror::new();
        m.on_event(&book_ev("polymarket.0xabc.UP", 0.40, 0.42, 1000));
        m.on_event(&book_ev("polymarket.0xabc.DOWN", 0.58, 0.60, 1000));
        m.on_event(&Event::new(
            "market.polymarket.catalog",
            "poly",
            900,
            0,
            Payload::Meta(MarketMeta {
                instrument: "polymarket.0xabc.UP".into(),
                kind: "5m_updown".into(),
                status: MarketStatus::Live,
                start_ts_ns: Some(0),
                expiry_ts_ns: Some(300_000_000_000),
                winner: None,
                min_order_size: Some(5.0),
                tick_size: Some(0.01),
                fee_rate: Some(0.07),
            }),
        ));

        let up = m.top("polymarket.0xabc.UP").unwrap();
        assert!((up.best_ask - 0.42).abs() < 1e-9);
        // meta keyed on UP is reachable from the DOWN token (same cid).
        assert_eq!(m.tte_ms("polymarket.0xabc.DOWN", 0), Some(300_000));
        assert!(m.top("polymarket.0xabc.DOWN").is_some());
        // economics keyed on UP are reachable from the DOWN token too.
        assert_eq!(m.meta_of("polymarket.0xabc.DOWN").unwrap().min_order_size, Some(5.0));
        assert_eq!(m.meta_of("polymarket.0xabc.DOWN").unwrap().fee_rate, Some(0.07));
    }
}
