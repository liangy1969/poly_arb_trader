//! L2 order book + Binance futures depth parsing + the sequenced resync state
//! machine (ports `l2_book.py` seq_validation mode + `parsers.py`).
//!
//! Futures continuity: each depth event carries `U` (first id), `u` (last id),
//! `pu` (previous event's last id). After a REST snapshot `lastUpdateId = S`,
//! drop buffered events with `u <= S`, require the first kept event's `pu == S`,
//! then live continuity requires `pu == last_u`; a mismatch or crossed book
//! triggers resync.

use std::collections::BTreeMap;

use serde_json::Value;

fn key(p: f64) -> i64 {
    (p * 1_000_000.0).round() as i64
}

#[derive(Default, Clone, Debug)]
pub struct L2Book {
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
}

impl L2Book {
    pub fn rebuild(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        self.bids.clear();
        self.asks.clear();
        for &(p, q) in bids {
            if q > 0.0 {
                self.bids.insert(key(p), q);
            }
        }
        for &(p, q) in asks {
            if q > 0.0 {
                self.asks.insert(key(p), q);
            }
        }
    }

    pub fn update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        for &(p, q) in bids {
            let k = key(p);
            if q == 0.0 {
                self.bids.remove(&k);
            } else {
                self.bids.insert(k, q);
            }
        }
        for &(p, q) in asks {
            let k = key(p);
            if q == 0.0 {
                self.asks.remove(&k);
            } else {
                self.asks.insert(k, q);
            }
        }
    }

    pub fn crossed(&self) -> bool {
        match (self.bids.keys().next_back(), self.asks.keys().next()) {
            (Some(&b), Some(&a)) => b >= a,
            _ => false,
        }
    }

    pub fn top_n(&self, n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = self.bids.iter().rev().take(n).map(|(k, v)| (*k as f64 / 1_000_000.0, *v)).collect();
        let asks = self.asks.iter().take(n).map(|(k, v)| (*k as f64 / 1_000_000.0, *v)).collect();
        (bids, asks)
    }
}

#[derive(Clone, Debug)]
pub struct Delta {
    pub big_u: i64, // U — first update id in event
    pub u: i64,     // u — last update id in event
    pub pu: i64,    // pu — previous event's last update id (futures)
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub exch_ns: i64,
    pub recv_ns: i64,
}

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub last_update_id: i64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

fn num(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

fn levels(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|l| {
                    let arr = l.as_array()?;
                    Some((num(arr.get(0))?, num(arr.get(1))?))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Parse a combined-stream `depthUpdate` frame (`{stream, data:{...}}` or bare).
pub fn parse_delta(text: &str, recv_ns: i64) -> Option<Delta> {
    let v: Value = serde_json::from_str(text).ok()?;
    let data = v.get("data").unwrap_or(&v);
    if data.get("e").and_then(Value::as_str) != Some("depthUpdate") {
        return None;
    }
    let big_u = data.get("U")?.as_i64()?;
    let u = data.get("u")?.as_i64()?;
    let pu = data.get("pu").and_then(Value::as_i64).unwrap_or(-1);
    let exch_ms = data
        .get("T")
        .and_then(Value::as_i64)
        .or_else(|| data.get("E").and_then(Value::as_i64))
        .unwrap_or(recv_ns / 1_000_000);
    Some(Delta {
        big_u,
        u,
        pu,
        bids: levels(&data["b"]),
        asks: levels(&data["a"]),
        exch_ns: exch_ms * 1_000_000,
        recv_ns,
    })
}

pub fn parse_snapshot(v: &Value) -> Option<Snapshot> {
    Some(Snapshot {
        last_update_id: v.get("lastUpdateId")?.as_i64()?,
        bids: levels(&v["bids"]),
        asks: levels(&v["asks"]),
    })
}

/// Real-time top-of-book from a `bookTicker` frame.
#[derive(Clone, Debug)]
pub struct Ticker {
    pub bid: f64,
    pub bid_sz: f64,
    pub ask: f64,
    pub ask_sz: f64,
    pub u: i64,
    pub exch_ns: i64,
    pub recv_ns: i64,
}

/// Parse a combined-stream `bookTicker` frame: `{e,u,s,b,B,a,A,T,E}` (futures
/// includes `e:"bookTicker"`; spot omits it). No sequencing — each frame is the
/// complete top-of-book.
pub fn parse_book_ticker(text: &str, recv_ns: i64) -> Option<Ticker> {
    let v: Value = serde_json::from_str(text).ok()?;
    let data = v.get("data").unwrap_or(&v);
    if let Some(e) = data.get("e").and_then(Value::as_str) {
        if e != "bookTicker" {
            return None;
        }
    }
    let bid = num(data.get("b"))?;
    let ask = num(data.get("a"))?;
    let bid_sz = num(data.get("B")).unwrap_or(0.0);
    let ask_sz = num(data.get("A")).unwrap_or(0.0);
    let u = data.get("u").and_then(Value::as_i64).unwrap_or(0);
    let exch_ms = data
        .get("T")
        .and_then(Value::as_i64)
        .or_else(|| data.get("E").and_then(Value::as_i64))
        .unwrap_or(recv_ns / 1_000_000);
    Some(Ticker {
        bid,
        bid_sz,
        ask,
        ask_sz,
        u,
        exch_ns: exch_ms * 1_000_000,
        recv_ns,
    })
}

/// Sequenced book with buffered-snapshot replay + gap detection.
#[derive(Default)]
pub struct SeqBook {
    pub book: L2Book,
    pub last_u: i64,
    pub initialized: bool,
    first_applied: bool,
    buffer: Vec<Delta>,
}

impl SeqBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset_for_resync(&mut self) {
        self.initialized = false;
        self.first_applied = false;
        self.buffer.clear();
    }

    /// Apply a REST snapshot + replay buffered deltas. `true` = book ready
    /// (publish); `false` = first-delta/crossed mismatch → caller resyncs.
    pub fn apply_snapshot(&mut self, snap: &Snapshot, is_futures: bool) -> bool {
        self.book.rebuild(&snap.bids, &snap.asks);
        let s = snap.last_update_id;
        self.last_u = s;
        self.initialized = false;
        self.first_applied = false;

        // Keep events not fully covered by the snapshot (u >= S); the first
        // kept event must *straddle* S (`U <= S <= u`) — the official Binance
        // managed-book rule. (`pu == S` is too strict: S usually lands inside
        // an event's range, not on a boundary.)
        let _ = is_futures;
        let kept: Vec<Delta> = self.buffer.drain(..).filter(|d| d.u >= s).collect();
        if kept.is_empty() {
            self.initialized = true;
            return true;
        }
        let f = &kept[0];
        if !(f.big_u <= s && s <= f.u) {
            return false;
        }
        // The straddle (U <= S <= u) is required because the futures snapshot's
        // S lands *inside* an event's range, not on a boundary (verified live,
        // Jun 2026: 3/3 mid-event) — so `pu == S` (exact-boundary match) fails.
        // The straddle is a superset: it also accepts the boundary case. This
        // log lets you re-verify the regime against a different endpoint.
        tracing::debug!(
            "snapshot-sync: S={} first_kept[U={} u={} pu={}] -> {} (pu==S? {})",
            s, f.big_u, f.u, f.pu,
            if s == f.u { "boundary(S==u)" } else { "mid-event(U<=S<u)" },
            f.pu == s,
        );
        for d in &kept {
            self.book.update(&d.bids, &d.asks);
            self.last_u = d.u;
            if self.book.crossed() {
                return false;
            }
        }
        self.initialized = true;
        self.first_applied = true;
        true
    }

    /// Apply a live delta. `Some(true)` = applied (publish), `Some(false)` =
    /// buffered (pre-snapshot), `None` = gap/crossed → caller resyncs.
    pub fn on_delta(&mut self, d: Delta) -> Option<bool> {
        if !self.initialized {
            self.buffer.push(d);
            return Some(false);
        }
        if self.first_applied && d.pu != self.last_u {
            return None; // sequence gap
        }
        self.book.update(&d.bids, &d.asks);
        self.last_u = d.u;
        if self.book.crossed() {
            return None;
        }
        self.first_applied = true;
        Some(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(big_u: i64, u: i64, pu: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Delta {
        Delta { big_u, u, pu, bids, asks, exch_ns: 0, recv_ns: 0 }
    }

    #[test]
    fn parse_book_ticker_frame() {
        let text = r#"{"stream":"btcusdt@bookTicker","data":{"e":"bookTicker","u":400,"s":"BTCUSDT","b":"64000.0","B":"1.5","a":"64000.1","A":"2.0","T":2,"E":2}}"#;
        let t = parse_book_ticker(text, 0).unwrap();
        assert_eq!(t.bid, 64000.0);
        assert_eq!(t.ask, 64000.1);
        assert_eq!(t.bid_sz, 1.5);
        assert_eq!(t.ask_sz, 2.0);
        assert_eq!(t.u, 400);
        assert_eq!(t.exch_ns, 2_000_000);
    }

    #[test]
    fn snapshot_no_buffer_ready() {
        let mut sb = SeqBook::new();
        let snap = Snapshot {
            last_update_id: 100,
            bids: vec![(50000.0, 1.0)],
            asks: vec![(50001.0, 2.0)],
        };
        assert!(sb.apply_snapshot(&snap, true));
        assert!(sb.initialized);
        let (b, a) = sb.book.top_n(5);
        assert_eq!(b[0], (50000.0, 1.0));
        assert_eq!(a[0], (50001.0, 2.0));
    }

    #[test]
    fn buffered_deltas_replayed_with_straddle() {
        let mut sb = SeqBook::new();
        // deltas arrive before the snapshot (S=100)
        assert_eq!(sb.on_delta(d(97, 99, 96, vec![], vec![])), Some(false)); // u=99 < S, dropped
        // straddling event: U=100 <= S=100 <= u=102
        assert_eq!(sb.on_delta(d(100, 102, 99, vec![(50000.5, 3.0)], vec![])), Some(false));
        assert_eq!(sb.on_delta(d(103, 104, 102, vec![], vec![(50002.0, 1.0)])), Some(false));
        let snap = Snapshot { last_update_id: 100, bids: vec![(50000.0, 1.0)], asks: vec![(50001.0, 2.0)] };
        assert!(sb.apply_snapshot(&snap, true)); // first kept straddles S
        assert_eq!(sb.last_u, 104);
        let (b, _) = sb.book.top_n(5);
        assert_eq!(b[0], (50000.5, 3.0)); // delta raised best bid
    }

    #[test]
    fn first_delta_gap_resyncs() {
        let mut sb = SeqBook::new();
        // first kept event starts after S (U=103 > S=100) -> missed 101..102 -> gap
        sb.on_delta(d(103, 105, 102, vec![], vec![]));
        let snap = Snapshot { last_update_id: 100, bids: vec![(50000.0, 1.0)], asks: vec![(50001.0, 2.0)] };
        assert!(!sb.apply_snapshot(&snap, true));
    }

    #[test]
    fn live_gap_detected() {
        let mut sb = SeqBook::new();
        let snap = Snapshot { last_update_id: 100, bids: vec![(50000.0, 1.0)], asks: vec![(50001.0, 2.0)] };
        sb.apply_snapshot(&snap, true);
        assert_eq!(sb.on_delta(d(101, 101, 100, vec![], vec![])), Some(true)); // continuous
        assert_eq!(sb.on_delta(d(102, 103, 102, vec![], vec![])), None); // pu=102 != last_u=101 -> gap
    }

    #[test]
    fn crossed_book_resyncs() {
        let mut sb = SeqBook::new();
        let snap = Snapshot { last_update_id: 100, bids: vec![(50000.0, 1.0)], asks: vec![(50001.0, 2.0)] };
        sb.apply_snapshot(&snap, true);
        // bid jumps above ask -> crossed
        assert_eq!(sb.on_delta(d(101, 101, 100, vec![(50002.0, 1.0)], vec![])), None);
    }

    #[test]
    fn parse_delta_combined_frame() {
        let text = r#"{"stream":"btcusdt@depth@100ms","data":{"e":"depthUpdate","E":1,"T":2,"s":"BTCUSDT","U":10,"u":12,"pu":9,"b":[["50000.0","1.5"]],"a":[["50001.0","0"]]}}"#;
        let delta = parse_delta(text, 0).unwrap();
        assert_eq!(delta.big_u, 10);
        assert_eq!(delta.u, 12);
        assert_eq!(delta.pu, 9);
        assert_eq!(delta.bids[0], (50000.0, 1.5));
        assert_eq!(delta.asks[0], (50001.0, 0.0)); // 0 size = removal
    }
}
