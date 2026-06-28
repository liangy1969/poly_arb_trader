//! The Processor `Module`: subscribe `market.#`, merge into `MarketState`, run
//! the rule engine, publish `signal.<strategy>`.

use std::sync::Arc;

use async_trait::async_trait;

use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::model::LatencySample;
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::rule::{PerpMoveRule, RuleEngine};
use crate::state::MarketState;

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct ProcCfg {
    pub reference: String,
    pub strategy: String,
    pub link_kind: String,
    pub window_ms: u64,
    pub threshold_bps: f64,
    pub yes_bucket: (f64, f64),
    pub cooldown_ms: u64,
    pub min_tte_ms: i64,
    pub hold_ms: u64,
    pub ttl_ms: u64,
    pub ring_cap: usize,
    pub ring_horizon_ms: u64,
}

impl Default for ProcCfg {
    fn default() -> Self {
        // The backtest's validated cell (DESIGN_EXECUTION §13).
        ProcCfg {
            reference: "binance.usdt_perp.BTCUSDT".into(),
            strategy: "perp_move".into(),
            link_kind: "5m_updown".into(),
            window_ms: 1000,
            threshold_bps: 3.0,
            yes_bucket: (0.05, 0.95),
            cooldown_ms: 2000,
            min_tte_ms: 15000,
            hold_ms: 1000,
            ttl_ms: 1000,
            ring_cap: 512,
            ring_horizon_ms: 5000,
        }
    }
}

pub struct Processor {
    cfg: ProcCfg,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl Processor {
    pub fn new(cfg: ProcCfg) -> Self {
        Processor { cfg, handle: None }
    }
}

#[async_trait]
impl Module for Processor {
    fn name(&self) -> &'static str {
        "processor"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        // Conflate market data by instrument: a flood on one instrument can't
        // back up the processor; each instrument keeps only its latest book.
        let sub = bus.subscribe("market.#", 1024, Policy::Conflate(key_by_instrument));
        let mut state = MarketState::new(
            self.cfg.reference.clone(),
            self.cfg.link_kind.clone(),
            self.cfg.ring_cap,
            self.cfg.ring_horizon_ms,
        );
        let rule = PerpMoveRule::new(
            self.cfg.strategy.clone(),
            self.cfg.reference.clone(),
            self.cfg.window_ms,
            self.cfg.threshold_bps,
            self.cfg.yes_bucket,
            self.cfg.cooldown_ms,
            self.cfg.min_tte_ms,
            self.cfg.hold_ms,
            self.cfg.ttl_ms,
        );
        let mut engine = RuleEngine::new(vec![Box::new(rule)]);
        let mut seq = 0u64;

        let handle = tokio::spawn(async move {
            let mut sub = sub;
            while let Some(ev) = sub.recv().await {
                state.on_event(&ev);
                for s in engine.on_event(&ev, &state) {
                    // s.ts_ns == the triggering book tick's collector recv_ts_ns,
                    // so this is the full book-tick -> trade-event latency.
                    let ts = s.ts_ns;
                    let lat_us = (now_ns() - ts) as f64 / 1000.0;
                    tracing::info!(
                        target: "latency",
                        "book_tick -> signal: {:.0} µs | {}",
                        lat_us,
                        s.reason,
                    );
                    // Raw per-signal latency sample, recorded under
                    // stream=latency/venue=processor; joins back to the signal
                    // via origin_ts_ns (== signal ts_ns) + target.
                    bus.publish(Event::new(
                        "latency.processor.book_to_signal",
                        "processor",
                        now_ns(),
                        seq,
                        Payload::Latency(LatencySample {
                            name: "book_to_signal".into(),
                            latency_us: lat_us,
                            origin_ts_ns: ts,
                            strategy: s.strategy.clone(),
                            target: s.target.clone(),
                        }),
                    ));
                    seq += 1;

                    let topic = format!("signal.{}", s.strategy);
                    bus.publish(Event::new(topic, "processor", ts, seq, Payload::Signal(s)));
                    seq += 1;
                }
            }
        });
        self.handle = Some(handle);
        Ok(())
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
        Ok(())
    }

    fn health(&self) -> Health {
        Health::Ok
    }
}
