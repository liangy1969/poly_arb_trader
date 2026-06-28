//! `InProcBus` — in-process pub/sub + claimed ports behind the `Bus` trait.
//!
//! Dispatch is synchronous on `publish`: it wraps the event in an `Arc`, walks
//! the subscription table, and forwards to each matching subscriber's channel
//! (no payload clone on fan-out — only the `Arc` is cloned). A dedicated router
//! task (DESIGN §3) is a later refactor; direct dispatch is simpler to reason
//! about for the bring-up.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use arb_core::bus::{Bus, ConflateChan, Policy, PortInbox, Subscription};
use arb_core::event::Event;

enum Sub {
    Stream { pat: Vec<String>, tx: UnboundedSender<Arc<Event>> },
    Conflate { pat: Vec<String>, key_fn: fn(&Event) -> u64, chan: ConflateChan },
}

impl Sub {
    fn pat(&self) -> &[String] {
        match self {
            Sub::Stream { pat, .. } | Sub::Conflate { pat, .. } => pat,
        }
    }
    fn alive(&self) -> bool {
        match self {
            Sub::Stream { tx, .. } => !tx.is_closed(),
            Sub::Conflate { chan, .. } => !chan.subscriber_gone(),
        }
    }
}

#[derive(Default)]
pub struct InProcBus {
    subs: Mutex<Vec<Sub>>,
    ports: Mutex<HashMap<String, UnboundedSender<Event>>>,
}

impl InProcBus {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Match a parsed dotted pattern against topic segments.
/// `*` matches exactly one segment; `#` matches zero or more (greedy via recursion).
fn seg_match(pat: &[String], top: &[&str]) -> bool {
    match pat.first() {
        None => top.is_empty(),
        Some(p) if p == "#" => (0..=top.len()).any(|i| seg_match(&pat[1..], &top[i..])),
        Some(p) => {
            !top.is_empty() && (p == "*" || p.as_str() == top[0]) && seg_match(&pat[1..], &top[1..])
        }
    }
}

impl Bus for InProcBus {
    fn publish(&self, ev: Event) {
        let arc = Arc::new(ev);
        let segs: Vec<&str> = arc.topic.split('.').collect();
        let mut subs = self.subs.lock().unwrap();
        subs.retain(|s| s.alive());
        for s in subs.iter() {
            if !seg_match(s.pat(), &segs) {
                continue;
            }
            match s {
                Sub::Stream { tx, .. } => {
                    let _ = tx.send(arc.clone());
                }
                Sub::Conflate { key_fn, chan, .. } => {
                    chan.push(key_fn(&arc), arc.clone());
                }
            }
        }
    }

    fn subscribe(&self, pattern: &str, _maxq: usize, policy: Policy) -> Subscription {
        let pat: Vec<String> = pattern.split('.').map(String::from).collect();
        match policy {
            Policy::Conflate(key_fn) => {
                let chan = ConflateChan::new();
                self.subs.lock().unwrap().push(Sub::Conflate { pat, key_fn, chan: chan.clone() });
                Subscription::conflate(pattern.to_string(), chan)
            }
            Policy::Block | Policy::DropOldest => {
                let (tx, rx) = unbounded_channel();
                self.subs.lock().unwrap().push(Sub::Stream { pat, tx });
                Subscription::stream(pattern.to_string(), rx)
            }
        }
    }

    fn claim_port(&self, port: &str, _maxq: usize) -> anyhow::Result<PortInbox> {
        let mut ports = self.ports.lock().unwrap();
        if ports.contains_key(port) {
            anyhow::bail!("port {port} already claimed");
        }
        let (tx, rx) = unbounded_channel();
        ports.insert(port.to_string(), tx);
        Ok(PortInbox { port: port.to_string(), rx })
    }

    fn send(&self, port: &str, ev: Event) {
        if let Some(tx) = self.ports.lock().unwrap().get(port) {
            let _ = tx.send(ev);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::seg_match;

    fn p(s: &str) -> Vec<String> {
        s.split('.').map(String::from).collect()
    }
    fn t(s: &str) -> Vec<&str> {
        s.split('.').collect()
    }

    #[test]
    fn exact_and_wildcards() {
        assert!(seg_match(&p("market.binance.book"), &t("market.binance.book")));
        assert!(seg_match(&p("market.*.book"), &t("market.binance.book")));
        assert!(!seg_match(&p("market.*.book"), &t("market.binance.trade")));
        assert!(seg_match(&p("market.#"), &t("market.binance.usdt_perp.BTCUSDT.book")));
        assert!(seg_match(&p("signal.#"), &t("signal.perp_move")));
        assert!(!seg_match(&p("market.#"), &t("signal.perp_move")));
        // '#' matches zero trailing segments
        assert!(seg_match(&p("market.#"), &t("market")));
    }
}
