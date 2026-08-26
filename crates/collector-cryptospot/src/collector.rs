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
    // BRTI constituents beyond coinbase (settlement-index siblings; the
    // KXBTC15M settlement BRTI is a depth-weighted consolidated mid of
    // coinbase+kraken+bitstamp+gemini+itbit+lmax+bullish+crypto.com books).
    pub kraken: bool,
    pub bitstamp: bool,
    pub gemini: bool,
    /// OKX spot. NOT a BRTI constituent — added 2026-08-25 as a fair-model
    /// feature after the venue-latency study found okx spot among the fastest
    /// feeds to this box (mean rank 2.14, tied with binance perp).
    /// ⚠️ BTC-USDT, so it carries a USDT basis that BTC-USD venues do not.
    pub okx: bool,
    pub coinbase_product: String, // "BTC-USD"
    pub binanceus_symbol: String, // "btcusd"
    pub kraken_symbol: String,    // "BTC/USD"
    pub bitstamp_symbol: String,  // "btcusd" (channel suffix)
    pub gemini_symbol: String,    // "btcusd"
    pub okx_symbol: String,       // "BTC-USDT"
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
            kraken: true,
            bitstamp: true,
            gemini: true,
            okx: true,
            coinbase_product: "BTC-USD".into(),
            binanceus_symbol: "btcusd".into(),
            kraken_symbol: "BTC/USD".into(),
            bitstamp_symbol: "btcusd".into(),
            gemini_symbol: "btcusd".into(),
            okx_symbol: "BTC-USDT".into(),
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
        if self.cfg.kraken {
            self.handles.push(tokio::spawn(venue_loop("kraken", kraken_session, self.cfg.clone(), bus.clone())));
        }
        if self.cfg.bitstamp {
            self.handles.push(tokio::spawn(venue_loop("bitstamp", bitstamp_session, self.cfg.clone(), bus.clone())));
        }
        if self.cfg.okx {
            self.handles.push(tokio::spawn(venue_loop("okx", okx_session, self.cfg.clone(), bus.clone())));
        }
        if self.cfg.gemini {
            self.handles.push(tokio::spawn(venue_loop("gemini", gemini_session, self.cfg.clone(), bus.clone())));
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

fn publish(bus: &dyn Bus, venue: &'static str, symbol: &str, bid: f64, bid_sz: f64, ask: f64, ask_sz: f64, recv: i64, seq: &mut u64) {
    publish_ts(bus, venue, symbol, bid, bid_sz, ask, ask_sz, recv, recv, seq);
}

fn publish_ts(bus: &dyn Bus, venue: &'static str, symbol: &str, bid: f64, bid_sz: f64, ask: f64, ask_sz: f64, exch: i64, recv: i64, seq: &mut u64) {
    *seq += 1;
    bus.publish(Event::new(
        format!("market.{venue}.{symbol}.book"),
        venue,
        recv,
        *seq,
        Payload::Book(BookUpdate {
            instrument: format!("{venue}.{symbol}"),
            bids: vec![(bid, bid_sz)],
            asks: vec![(ask, ask_sz)],
            update_id: None,
            exch_ts_ns: exch,
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
                let sym = cfg.coinbase_product.split('-').next().unwrap_or("BTC");
                publish(&**bus, "coinbase", sym, bid, bsz, ask, asz, now_ns(), &mut seq);
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
                publish(&**bus, "binanceus", "BTC", bid, bsz, ask, asz, now_ns(), &mut seq);
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

// ── BRTI-constituent venues (kraken / bitstamp / gemini) ─────────────────────
// One generic reconnect loop; each venue is a session fn that runs until error.
async fn venue_loop<F, Fut>(name: &'static str, session: F, cfg: CryptoSpotCfg, bus: Arc<dyn Bus>)
where
    F: Fn(CryptoSpotCfg, Arc<dyn Bus>) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let mut backoff = cfg.reconnect_base_ms;
    loop {
        match session(cfg.clone(), bus.clone()).await {
            Ok(()) => backoff = cfg.reconnect_base_ms,
            Err(e) => {
                tracing::warn!("{name} ended ({e}) -> reconnect {backoff}ms");
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.reconnect_max_ms);
            }
        }
    }
}

/// Kraken WS v2 `ticker`: BBO push on every best-quote change (JSON numbers).
async fn kraken_session(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async("wss://ws.kraken.com/v2").await?;
    let sub = serde_json::json!({
        "method": "subscribe",
        "params": {"channel": "ticker", "symbol": [cfg.kraken_symbol]},
    });
    ws.send(Message::Text(sub.to_string().into())).await?;
    tracing::info!("kraken ws connected: {} ticker", cfg.kraken_symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                if v.get("channel").and_then(Value::as_str) != Some("ticker") {
                    continue; // heartbeat / method acks
                }
                let Some(d) = v.get("data").and_then(|d| d.get(0)) else { continue };
                let (Some(bid), Some(ask)) = (num(d.get("bid")), num(d.get("ask"))) else { continue };
                let bsz = num(d.get("bid_qty")).unwrap_or(0.0);
                let asz = num(d.get("ask_qty")).unwrap_or(0.0);
                publish(&*bus, "kraken", "BTC", bid, bsz, ask, asz, now_ns(), &mut seq);
            }
            Some(Ok(Message::Ping(p))) => ws.send(Message::Pong(p)).await?,
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}

/// Bitstamp `order_book_<sym>`: ~100ms top-100 snapshots; we take level 0.
/// Carries a server `microtimestamp` -> real exch_ts (srv-age measurable).
async fn bitstamp_session(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async("wss://ws.bitstamp.net").await?;
    let chan = format!("order_book_{}", cfg.bitstamp_symbol);
    let sub = serde_json::json!({"event": "bts:subscribe", "data": {"channel": chan}});
    ws.send(Message::Text(sub.to_string().into())).await?;
    tracing::info!("bitstamp ws connected: {chan}");
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                match v.get("event").and_then(Value::as_str) {
                    Some("data") => {}
                    Some("bts:request_reconnect") => anyhow::bail!("server requested reconnect"),
                    _ => continue, // subscription ack etc.
                }
                let Some(d) = v.get("data") else { continue };
                let top = |side: &str| -> Option<(f64, f64)> {
                    let l = d.get(side)?.get(0)?;
                    Some((num(l.get(0))?, num(l.get(1)).unwrap_or(0.0)))
                };
                let (Some((bid, bsz)), Some((ask, asz))) = (top("bids"), top("asks")) else { continue };
                let exch = d
                    .get("microtimestamp")
                    .and_then(|x| x.as_str().and_then(|s| s.parse::<i64>().ok()))
                    .map(|us| us * 1_000)
                    .unwrap_or_else(now_ns);
                publish_ts(&*bus, "bitstamp", "BTC", bid, bsz, ask, asz, exch, now_ns(), &mut seq);
            }
            Some(Ok(Message::Ping(p))) => ws.send(Message::Pong(p)).await?,
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}

/// Gemini v1 marketdata with `top_of_book=true`: change events for the BBO
/// only; we hold the working BBO and publish on every change.
async fn gemini_session(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
    let url = format!(
        "wss://api.gemini.com/v1/marketdata/{}?top_of_book=true",
        cfg.gemini_symbol
    );
    let (mut ws, _) = connect_async(&url).await?;
    tracing::info!("gemini ws connected: {} top_of_book", cfg.gemini_symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    let (mut bid, mut bsz, mut ask, mut asz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                if v.get("type").and_then(Value::as_str) != Some("update") {
                    continue;
                }
                let Some(events) = v.get("events").and_then(Value::as_array) else { continue };
                let mut changed = false;
                for e in events {
                    if e.get("type").and_then(Value::as_str) != Some("change") {
                        continue;
                    }
                    let (Some(px), Some(rem)) = (num(e.get("price")), num(e.get("remaining"))) else { continue };
                    match e.get("side").and_then(Value::as_str) {
                        Some("bid") => { bid = px; bsz = rem; changed = true; }
                        Some("ask") => { ask = px; asz = rem; changed = true; }
                        _ => {}
                    }
                }
                if changed && bid > 0.0 && ask > 0.0 {
                    publish(&*bus, "gemini", "BTC", bid, bsz, ask, asz, now_ns(), &mut seq);
                }
            }
            Some(Ok(Message::Ping(p))) => ws.send(Message::Pong(p)).await?,
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}

/// OKX spot top-of-book via `bbo-tbt` (tick-by-tick BBO — the lowest-latency
/// public channel). Direct, NO proxy: OKX answers the Ohio box in ~200ms vs
/// ~450ms through the Tokyo relay, so routing it through the proxy would only
/// add latency to the feed we added for its speed.
async fn okx_session(cfg: CryptoSpotCfg, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
    let (mut ws, _) = connect_async("wss://ws.okx.com:8443/ws/v5/public").await?;
    let sub = serde_json::json!({
        "op": "subscribe",
        "args": [{"channel": "bbo-tbt", "instId": cfg.okx_symbol}]
    });
    ws.send(Message::Text(sub.to_string().into())).await?;
    tracing::info!("okx ws connected: bbo-tbt {}", cfg.okx_symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    let mut seq = 0u64;
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                let Ok(v) = serde_json::from_str::<Value>(t.as_str()) else { continue };
                if let Some(ev) = v.get("event").and_then(Value::as_str) {
                    if ev == "error" {
                        anyhow::bail!("okx subscribe error: {t}");
                    }
                    continue; // subscribe ack
                }
                let Some(d) = v.get("data").and_then(Value::as_array).and_then(|a| a.first())
                else { continue };
                // levels are ["px","sz","liqOrders","orders"]
                let top = |side: &str| -> Option<(f64, f64)> {
                    let l = d.get(side)?.as_array()?.first()?;
                    Some((num(l.get(0))?, num(l.get(1)).unwrap_or(0.0)))
                };
                let (Some((bid, bsz)), Some((ask, asz))) = (top("bids"), top("asks")) else { continue };
                let exch = d
                    .get("ts")
                    .and_then(|x| x.as_str().and_then(|s| s.parse::<i64>().ok()))
                    .map(|ms| ms * 1_000_000)
                    .unwrap_or_else(now_ns);
                publish_ts(&*bus, "okx", "BTC", bid, bsz, ask, asz, exch, now_ns(), &mut seq);
            }
            Some(Ok(Message::Ping(p))) => ws.send(Message::Pong(p)).await?,
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}
