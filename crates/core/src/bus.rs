//! Bus abstraction (DESIGN §3). `InProcBus` lives in the `arb-bus` crate.
//!
//! Two delivery shapes behind one `Subscription::recv()`:
//! - **Stream** (Block) — an unbounded mpsc; never drops. For signals/exec.
//! - **Conflate** — keeps only the *latest* event per key (instrument). A flood
//!   on one instrument can't back up the consumer: pending is bounded to the
//!   number of distinct keys, and each key always holds its newest event. For
//!   market data.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::Notify;

use crate::event::{Event, Payload};

pub enum Policy {
    /// Keep only the latest event per key (the fn maps an event to a key).
    Conflate(fn(&Event) -> u64),
    DropOldest,
    Block,
}

/// Conflating channel: latest-per-key, drained in first-seen-key order.
/// `map` + `order` live under a **single** mutex (a split would invite an ABBA
/// deadlock between push and pop).
#[derive(Default)]
struct ConflateInner {
    map: HashMap<u64, Arc<Event>>,
    order: VecDeque<u64>,
}

#[derive(Clone)]
pub struct ConflateChan {
    inner: Arc<Mutex<ConflateInner>>,
    notify: Arc<Notify>,
}

impl Default for ConflateChan {
    fn default() -> Self {
        Self::new()
    }
}

impl ConflateChan {
    pub fn new() -> Self {
        ConflateChan {
            inner: Arc::new(Mutex::new(ConflateInner::default())),
            notify: Arc::new(Notify::new()),
        }
    }

    pub fn push(&self, key: u64, ev: Arc<Event>) {
        {
            let mut g = self.inner.lock().unwrap();
            if g.map.insert(key, ev).is_none() {
                g.order.push_back(key);
            }
        }
        self.notify.notify_one();
    }

    fn try_pop(&self) -> Option<Arc<Event>> {
        let mut g = self.inner.lock().unwrap();
        while let Some(k) = g.order.pop_front() {
            if let Some(ev) = g.map.remove(&k) {
                return Some(ev);
            }
        }
        None
    }

    async fn recv(&self) -> Arc<Event> {
        loop {
            if let Some(ev) = self.try_pop() {
                return ev;
            }
            let notified = self.notify.notified();
            if let Some(ev) = self.try_pop() {
                return ev;
            }
            notified.await;
        }
    }

    /// True once only the bus retains a clone — the subscriber was dropped.
    pub fn subscriber_gone(&self) -> bool {
        Arc::strong_count(&self.notify) <= 1
    }
}

enum SubRx {
    Stream(UnboundedReceiver<Arc<Event>>),
    Conflate(ConflateChan),
}

pub struct Subscription {
    pub pattern: String,
    rx: SubRx,
}

impl Subscription {
    pub fn stream(pattern: String, rx: UnboundedReceiver<Arc<Event>>) -> Self {
        Subscription { pattern, rx: SubRx::Stream(rx) }
    }
    pub fn conflate(pattern: String, chan: ConflateChan) -> Self {
        Subscription { pattern, rx: SubRx::Conflate(chan) }
    }
    /// Next event. `None` only when a Stream's sender side is gone (Conflate
    /// recv runs until the task is dropped).
    pub async fn recv(&mut self) -> Option<Arc<Event>> {
        match &mut self.rx {
            SubRx::Stream(rx) => rx.recv().await,
            SubRx::Conflate(c) => Some(c.recv().await),
        }
    }
}

/// Exactly-one-owner point-to-point inbox (for `exec.intent` etc.).
pub struct PortInbox {
    pub port: String,
    pub rx: UnboundedReceiver<Event>,
}

pub trait Bus: Send + Sync {
    fn publish(&self, ev: Event);
    fn subscribe(&self, pattern: &str, maxq: usize, policy: Policy) -> Subscription;
    fn claim_port(&self, port: &str, maxq: usize) -> anyhow::Result<PortInbox>;
    fn send(&self, port: &str, ev: Event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use crate::model::BookUpdate;

    fn ev(inst: &str) -> Arc<Event> {
        Arc::new(Event::new(
            "t",
            "s",
            0,
            0,
            Payload::Book(BookUpdate {
                instrument: inst.into(),
                bids: vec![],
                asks: vec![],
                update_id: None,
                exch_ts_ns: 0,
                recv_ts_ns: 0,
            }),
        ))
    }

    fn inst_of(e: &Event) -> &str {
        match &e.payload {
            Payload::Book(b) => &b.instrument,
            _ => "",
        }
    }

    #[tokio::test]
    async fn conflate_keeps_latest_per_key_in_order() {
        let c = ConflateChan::new();
        c.push(1, ev("a")); // key 1
        c.push(2, ev("b")); // key 2
        c.push(1, ev("a2")); // key 1 updated -> conflated to latest
        // key 1 pops first (first-seen order), carrying the LATEST value
        let e1 = c.recv().await;
        let e2 = c.recv().await;
        assert_eq!(inst_of(&e1), "a2");
        assert_eq!(inst_of(&e2), "b");
        assert!(c.try_pop().is_none());
    }
}

/// Conflate key: `(payload-kind, instrument)` so each instrument keeps its
/// latest of each event kind (a Book and a catalog Meta for the same
/// instrument don't clobber each other).
pub fn key_by_instrument(ev: &Event) -> u64 {
    use std::hash::{Hash, Hasher};
    let (tag, inst): (u64, &str) = match &ev.payload {
        Payload::Book(b) => (1, &b.instrument),
        Payload::Trade(t) => (2, &t.instrument),
        Payload::Liq(l) => (3, &l.instrument),
        Payload::Meta(m) => (4, &m.instrument),
        Payload::Signal(s) => (5, &s.target),
        Payload::Latency(l) => (6, &l.name),
        Payload::ExecReport(r) => (7, &r.instrument),
        Payload::TradeRecord(t) => (8, &t.instrument),
        Payload::Position(p) => (9, &p.instrument),
        Payload::Calib(c) => (10, &c.instrument),
    };
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut h);
    inst.hash(&mut h);
    h.finish()
}
