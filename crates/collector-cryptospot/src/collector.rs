//! Coinbase + Binance.US spot BTC top-of-book collectors (public WS, no auth).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::Value;
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use arb_core::bus::Bus;
use arb_core::event::{Event, Payload};
use arb_core::model::BookUpdate;
use arb_core::module::{Health, Module};
use arb_core::now_ns;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct CryptoSpotCfg {
    /// Off by default; set true to run the spot WS collectors.
    pub enabled: bool,
    pub coinbase: bool,
    pub binanceus: bool,
    pub coinbase_product: String, // "BTC-USD"
    pub binanceus_symbol: String, // "btcusd"
    pub reconnect_base_ms: u64,
    pub reconnect_max_ms: u64,
    pub stale_timeout_s: u64,
}

impl Default for CryptoSpotCfg {
    fn default() -> Self {
        CryptoSpotCfg {
            enabled: false,
            coinbase: true,
            binanceus: true,
            coinbase_product: "BTC-USD".into(),
            binanceus_symbol: "btcusd".into(),
            reconnect_base_ms: 1000,
            reconnect_max_ms: 30000,
            stale_timeout_s: 30,
        }
    }
}

pub struct CryptoSpotCollector {
    cfg: CryptoSpotCfg,
    handles: Vec<JoinHandle<()>>,
}

impl CryptoSpotCollector {
    pub fn new(cfg: CryptoSpotCfg) -> Self {
        CryptoSpotCollector { cfg, handles: Vec::new() }
    }
}

#[async_trait]
impl Module for CryptoSpotCollector {
    fn name(&self) -> &'static str {
        "collector-cryptospot"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        if self.cfg.coinbase {
            self.handles.push(tokio::spawn(coinbase_loop(self.cfg.clone(), bus.clone())));
        }
        if self.cfg.binanceus {
            self.handles.push(tokio::spawn(binanceus_loop(self.cfg.clone(), bus.clone())));
        }
        Ok(())
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        for h in self.handles.drain(..) {
            h.abort();
        }
        Ok(())
    }

    fn health(&self) -> Health {
        Health::Ok
    }
}

fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(|x| x.as_str().and_then(|s| s.parse().ok()).or_else(|| x.as_f64()))
}

fn publish(bus: &dyn Bus, venue: &'static str, bid: f64, bid_sz: f64, ask: f64, ask_sz: f64, recv: i64, seq: &mut u64) {
    *seq += 1;
    bus.publish(Event::new(
        format!("market.{venue}.BTC.book"),
        venue,
        recv,
        *seq,
        Payload::Book(BookUpdate {
            instrument: format!("{venue}.BTC"),
            bids: vec![(bid, bid_sz)],
            asks: vec![(ask, ask_sz)],
            update_id: None,
            exch_ts_ns: recv,
            recv_ts_ns: recv,
        }),
    ));
}

async fn coinbase_loop(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) {
    let mut backoff = cfg.reconnect_base_ms;
    loop {
        match coinbase_session(&cfg, &bus).await {
            Ok(()) => backoff = cfg.reconnect_base_ms,
            Err(e) => {
                tracing::warn!("coinbase ended ({e}) -> reconnect {backoff}ms");
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.reconnect_max_ms);
            }
        }
    }
}

async fn coinbase_session(cfg: &CryptoSpotCfg, bus: &Arc<dyn Bus>) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async("wss://ws-feed.exchange.coinbase.com").await?;
    let sub = serde_json::json!({
        "type": "subscribe",
        "product_ids": [cfg.coinbase_product],
        "channels": ["ticker"],
    });
    ws.send(Message::Text(sub.to_string().into())).await?;
    tracing::info!("coinbase ws connected: {} ticker", cfg.coinbase_product);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                if v.get("type").and_then(Value::as_str) != Some("ticker") {
                    continue;
                }
                let (Some(bid), Some(ask)) = (num(v.get("best_bid")), num(v.get("best_ask"))) else { continue };
                let bsz = num(v.get("best_bid_size")).unwrap_or(0.0);
                let asz = num(v.get("best_ask_size")).unwrap_or(0.0);
                publish(&**bus, "coinbase", bid, bsz, ask, asz, now_ns(), &mut seq);
            }
            Some(Ok(Message::Ping(p))) => {
                ws.send(Message::Pong(p)).await?;
            }
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}

async fn binanceus_loop(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) {
    let mut backoff = cfg.reconnect_base_ms;
    loop {
        match binanceus_session(&cfg, &bus).await {
            Ok(()) => backoff = cfg.reconnect_base_ms,
            Err(e) => {
                tracing::warn!("binance.us ended ({e}) -> reconnect {backoff}ms");
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.reconnect_max_ms);
            }
        }
    }
}

async fn binanceus_session(cfg: &CryptoSpotCfg, bus: &Arc<dyn Bus>) -> anyhow::Result<()> {
    let url = format!("wss://stream.binance.us:9443/ws/{}@bookTicker", cfg.binanceus_symbol);
    let (mut ws, _) = connect_async(&url).await?;
    tracing::info!("binance.us ws connected: {}@bookTicker", cfg.binanceus_symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                let (Some(bid), Some(ask)) = (num(v.get("b")), num(v.get("a"))) else { continue };
                let bsz = num(v.get("B")).unwrap_or(0.0);
                let asz = num(v.get("A")).unwrap_or(0.0);
                publish(&**bus, "binanceus", bid, bsz, ask, asz, now_ns(), &mut seq);
            }
            Some(Ok(Message::Ping(p))) => {
                ws.send(Message::Pong(p)).await?;
            }
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}
