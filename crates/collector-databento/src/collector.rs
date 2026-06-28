//! `DatabentoCollector` — CME BTC futures top-of-book (mbp-1 BBO) via the Databento
//! Live API. Signal-only reference feed: maps each `Mbp1Msg` to the standard
//! `BookUpdate` and publishes `market.databento.<root>.book` (e.g. `databento.MBT`).
//! US-direct, sub-ms exchange→gateway latency, nanosecond `ts_event`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::task::JoinHandle;

use arb_core::bus::Bus;
use arb_core::event::{Event, Payload};
use arb_core::model::BookUpdate;
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use databento::dbn::{Dataset, Mbp1Msg, SType, Schema, SymbolMappingMsg, FIXED_PRICE_SCALE, UNDEF_PRICE};
use databento::live::Subscription;
use databento::LiveClient;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct DatabentoCfg {
    /// Off by default — needs DATABENTO_API_KEY + a GLBX.MDP3 live entitlement.
    pub enabled: bool,
    /// Continuous symbols to stream, e.g. ["MBT.v.0", "BTC.v.0"]. Single-symbol is
    /// labelled directly; multi-symbol resolves labels via SymbolMappingMsg.
    pub symbols: Vec<String>,
    pub reconnect_base_ms: u64,
    pub reconnect_max_ms: u64,
}

impl Default for DatabentoCfg {
    fn default() -> Self {
        DatabentoCfg {
            enabled: false,
            symbols: vec!["MBT.v.0".into()],
            reconnect_base_ms: 1000,
            reconnect_max_ms: 30000,
        }
    }
}

pub struct DatabentoCollector {
    cfg: DatabentoCfg,
    handle: Option<JoinHandle<()>>,
}

impl DatabentoCollector {
    pub fn new(cfg: DatabentoCfg) -> Self {
        DatabentoCollector { cfg, handle: None }
    }
}

#[async_trait]
impl Module for DatabentoCollector {
    fn name(&self) -> &'static str {
        "collector-databento"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        self.handle = Some(tokio::spawn(run_loop(self.cfg.clone(), bus)));
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

/// "MBT.v.0" -> "MBT".
fn root_of(sym: &str) -> String {
    sym.split('.').next().unwrap_or(sym).to_string()
}

async fn run_loop(cfg: DatabentoCfg, bus: Arc<dyn Bus>) {
    let mut backoff = cfg.reconnect_base_ms;
    loop {
        match session(&cfg, &bus).await {
            Ok(()) => backoff = cfg.reconnect_base_ms,
            Err(e) => {
                tracing::warn!("databento session ended ({e}) -> reconnect in {backoff}ms");
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.reconnect_max_ms);
            }
        }
    }
}

async fn session(cfg: &DatabentoCfg, bus: &Arc<dyn Bus>) -> anyhow::Result<()> {
    let mut client = LiveClient::builder()
        .key_from_env()? // DATABENTO_API_KEY
        .dataset(Dataset::GlbxMdp3)
        .build()
        .await?;
    client
        .subscribe(
            Subscription::builder()
                .symbols(cfg.symbols.clone())
                .schema(Schema::Mbp1)
                .stype_in(SType::Continuous)
                .build(),
        )
        .await?;
    client.start().await?;
    tracing::info!("databento live up: {:?} mbp-1 (GLBX.MDP3)", cfg.symbols);

    // single-symbol: label directly; multi: resolve via SymbolMappingMsg.
    let single = (cfg.symbols.len() == 1).then(|| root_of(&cfg.symbols[0]));
    let mut id2root: HashMap<u32, String> = HashMap::new();
    let mut seq: u64 = 0;
    let scale = FIXED_PRICE_SCALE as f64;

    while let Some(rec) = client.next_record().await? {
        if let Some(sm) = rec.get::<SymbolMappingMsg>() {
            if let Ok(insym) = sm.stype_in_symbol() {
                id2root.insert(sm.hd.instrument_id, root_of(insym));
            }
            continue;
        }
        let Some(m) = rec.get::<Mbp1Msg>() else { continue };
        let l = &m.levels[0];
        if l.bid_px == UNDEF_PRICE || l.ask_px == UNDEF_PRICE {
            continue;
        }
        let root = single
            .clone()
            .or_else(|| id2root.get(&m.hd.instrument_id).cloned())
            .unwrap_or_else(|| format!("id{}", m.hd.instrument_id));
        let recv = now_ns();
        seq += 1;
        bus.publish(Event::new(
            format!("market.databento.{root}.book"),
            "databento",
            recv,
            seq,
            Payload::Book(BookUpdate {
                instrument: format!("databento.{root}"),
                bids: vec![(l.bid_px as f64 / scale, l.bid_sz as f64)],
                asks: vec![(l.ask_px as f64 / scale, l.ask_sz as f64)],
                update_id: Some(m.sequence as u64),
                exch_ts_ns: m.hd.ts_event as i64,
                recv_ts_ns: recv,
            }),
        ));
    }
    anyhow::bail!("databento stream ended")
}
