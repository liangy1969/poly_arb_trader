//! Feature probe — a **log-only** shadow. It reconstructs a model's extra
//! features (basis/dbasis/imb1/mom/vsurge) from the live feeds and logs them
//! every `flush_s` to the `featstats` target, for validating the online feature
//! extraction against the offline lake. It NEVER calibrates, emits signals, or
//! touches the executor — so it carries zero trading risk.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::task::JoinHandle;

use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::Payload;
use arb_core::module::{Health, Module};

use crate::fair::{FairSurface, FeatureState};

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct FeatureProbeCfg {
    pub enabled: bool,
    /// Model JSON whose `extras` list defines which features to reconstruct+log.
    /// Ignored (may be empty) when `extras` is set explicitly below.
    pub model_path: String,
    /// Explicit feature list to reconstruct+log; overrides the model's extras.
    /// Use this to probe only features NOT already logged by the trading path
    /// (the rule's own `featstats feats …` line covers the trading model's).
    pub extras: Vec<String>,
    /// Price reference instrument (perp) — carries mom/imb1/vol.
    pub reference: String,
    /// Basis reference instrument (coinbase) — carries basis/dbasis.
    pub basis_reference: String,
    /// DEEP-book instrument (the @depth collector) — carries band{k}. Empty =
    /// no depth feed (band features will never be ready).
    pub depth_reference: String,
    /// featstats log cadence (seconds).
    pub flush_s: u64,
}

impl Default for FeatureProbeCfg {
    fn default() -> Self {
        FeatureProbeCfg {
            enabled: false,
            model_path: String::new(),
            extras: Vec::new(),
            reference: "binance.usdt_perp.BTCUSDT".into(),
            basis_reference: "coinbase.BTC".into(),
            depth_reference: String::new(),
            flush_s: 60,
        }
    }
}

pub struct FeatureProbe {
    cfg: FeatureProbeCfg,
    handle: Option<JoinHandle<()>>,
}

impl FeatureProbe {
    pub fn new(cfg: FeatureProbeCfg) -> Self {
        FeatureProbe { cfg, handle: None }
    }
}

#[async_trait]
impl Module for FeatureProbe {
    fn name(&self) -> &'static str {
        "feature-probe"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        if !self.cfg.enabled {
            return Ok(());
        }
        // explicit `extras` wins; else load the surface only to read its list
        // (no fair/calib here either way).
        let extras = if !self.cfg.extras.is_empty() {
            self.cfg.extras.clone()
        } else {
            let bytes = std::fs::read(&self.cfg.model_path)?;
            FairSurface::from_json(std::str::from_utf8(&bytes)?)?.extras.clone()
        };
        let mut feats = FeatureState::new(extras.clone());
        let reference = self.cfg.reference.clone();
        let basis_reference = self.cfg.basis_reference.clone();
        let depth_reference = self.cfg.depth_reference.clone();
        let vol_inst = format!("{}.vol", reference);
        let flush_ns = (self.cfg.flush_s.max(1) as i64) * 1_000_000_000;
        let sub = bus.subscribe("market.#", 1024, Policy::Conflate(key_by_instrument));
        tracing::info!(
            "feature-probe up: extras={:?} ({})",
            extras,
            if self.cfg.extras.is_empty() { &self.cfg.model_path } else { "explicit cfg" }
        );
        self.handle = Some(tokio::spawn(async move {
            let mut sub = sub;
            let mut last_flush = 0i64;
            while let Some(ev) = sub.recv().await {
                match &ev.payload {
                    Payload::Book(b)
                        if !depth_reference.is_empty() && b.instrument == depth_reference =>
                    {
                        // deep perp book (@depth top-N) → band{k}
                        feats.on_depth(b.recv_ts_ns, &b.bids, &b.asks);
                    }
                    Payload::Book(b) if b.instrument == reference => {
                        if let (Some(&(bid, bsz)), Some(&(ask, asz))) =
                            (b.bids.first(), b.asks.first())
                        {
                            if bid > 0.0 && ask > 0.0 {
                                feats.on_perp(b.recv_ts_ns, 0.5 * (bid + ask), bsz, asz);
                            }
                        }
                    }
                    Payload::Book(b) if b.instrument == basis_reference => {
                        if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first())
                        {
                            if bid > 0.0 && ask > 0.0 {
                                feats.on_cb(b.recv_ts_ns, 0.5 * (bid + ask));
                            }
                        }
                    }
                    Payload::Trade(t) if t.instrument == vol_inst => {
                        feats.on_perp_trade(t.recv_ts_ns, t.qty); // cumulative volume
                    }
                    _ => {}
                }
                let now = ev.ts_ns;
                if now - last_flush >= flush_ns {
                    last_flush = now;
                    match feats.feats(now) {
                        Some(f) => {
                            let kv: Vec<String> = extras
                                .iter()
                                .zip(f.iter())
                                .map(|(n, v)| format!("{}={:.5}", n, v))
                                .collect();
                            tracing::info!(target: "featstats", "probe {}", kv.join(" "));
                        }
                        None => tracing::info!(
                            target: "featstats",
                            "probe (features not ready — warming up)"
                        ),
                    }
                }
            }
        }));
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
