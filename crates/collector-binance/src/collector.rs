//! `BinanceCollector` — perp depth (order book only) via the SOCKS5 proxy.
//! Connects the combined depth stream, fetches the REST snapshot through the
//! proxy, runs the sequenced resync, and publishes
//! `market.binance.usdt_perp.<symbol>.book` (DESIGN §5).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_socks::tcp::Socks5Stream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use arb_core::bus::Bus;
use arb_core::event::{Event, Payload};
use arb_core::model::BookUpdate;
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::book::{parse_book_ticker, parse_delta, parse_snapshot, L2Book, SeqBook, Snapshot, Ticker};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct BinanceCfg {
    pub symbol: String,            // "BTCUSDT"
    pub instrument: String,        // "binance.usdt_perp.BTCUSDT"
    pub ws_base: String,           // "wss://fstream.binance.com/public"
    pub rest_base: String,         // "https://fapi.binance.com"
    pub stream: String,            // "btcusdt@bookTicker" (top-of-book) | "btcusdt@depth@100ms" (L2 diff)
    pub snapshot_limit: u32,       // 1000 (depth mode only)
    pub socks_proxy: Option<String>, // "127.0.0.1:1080" (None = direct)
    pub top_n: usize,
    pub stale_timeout_s: u64,
    pub reconnect_base_ms: u64,
    pub reconnect_max_ms: u64,
}

impl Default for BinanceCfg {
    fn default() -> Self {
        BinanceCfg {
            symbol: "BTCUSDT".into(),
            instrument: "binance.usdt_perp.BTCUSDT".into(),
            ws_base: "wss://fstream.binance.com/public".into(),
            rest_base: "https://fapi.binance.com".into(),
            stream: "btcusdt@bookTicker".into(), // default: real-time top-of-book
            snapshot_limit: 1000,
            socks_proxy: Some("127.0.0.1:1080".into()),
            top_n: 10,
            stale_timeout_s: 5,
            reconnect_base_ms: 500,
            reconnect_max_ms: 30_000,
        }
    }
}

fn nxt(seq: &mut u64) -> u64 {
    let v = *seq;
    *seq += 1;
    v
}

fn host_port(ws_base: &str) -> (String, u16) {
    let s = ws_base
        .strip_prefix("wss://")
        .or_else(|| ws_base.strip_prefix("ws://"))
        .unwrap_or(ws_base);
    let hostpart = s.split('/').next().unwrap_or(s);
    let mut it = hostpart.splitn(2, ':');
    let host = it.next().unwrap_or(hostpart).to_string();
    let port = it.next().and_then(|p| p.parse().ok()).unwrap_or(443);
    (host, port)
}

async fn connect_ws(cfg: &BinanceCfg) -> anyhow::Result<Ws> {
    let url = format!("{}/stream?streams={}", cfg.ws_base, cfg.stream);
    let req = url.as_str().into_client_request()?;
    match &cfg.socks_proxy {
        Some(proxy) => {
            let (host, port) = host_port(&cfg.ws_base);
            let socks = Socks5Stream::connect(proxy.as_str(), (host.as_str(), port)).await?;
            let (ws, _) = tokio_tungstenite::client_async_tls(req, socks.into_inner()).await?;
            Ok(ws)
        }
        None => {
            let (ws, _) = tokio_tungstenite::connect_async(req).await?;
            Ok(ws)
        }
    }
}

fn snap_future(
    http: reqwest::Client,
    url: String,
    delay_ms: u64,
) -> Pin<Box<dyn Future<Output = anyhow::Result<Snapshot>> + Send>> {
    Box::pin(async move {
        if delay_ms > 0 {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        }
        let resp = http.get(&url).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        let snippet: String = text.chars().take(160).collect();
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| anyhow::anyhow!("snapshot not JSON (HTTP {status}): {e}: {snippet}"))?;
        // A geo-restricted REST endpoint returns a JSON error here, not a book.
        parse_snapshot(&v)
            .ok_or_else(|| anyhow::anyhow!("snapshot missing lastUpdateId (HTTP {status}): {snippet}"))
    })
}

fn publish_book(
    bus: &dyn Bus,
    cfg: &BinanceCfg,
    book: &L2Book,
    exch_ns: i64,
    recv_ns: i64,
    u: i64,
    seq: &mut u64,
) {
    let (bids, asks) = book.top_n(cfg.top_n);
    bus.publish(Event::new(
        format!("market.binance.usdt_perp.{}.book", cfg.symbol),
        "binance",
        recv_ns,
        nxt(seq),
        Payload::Book(BookUpdate {
            instrument: cfg.instrument.clone(),
            bids,
            asks,
            update_id: Some(u as u64),
            exch_ts_ns: exch_ns,
            recv_ts_ns: recv_ns,
        }),
    ));
}

/// One WS session: connect, snapshot+resync, stream until disconnect/gap.
/// Returns `Err` to trigger a reconnect.
async fn session(
    cfg: &BinanceCfg,
    http: &reqwest::Client,
    bus: &Arc<dyn Bus>,
    seq: &mut u64,
) -> anyhow::Result<()> {
    let mut ws = connect_ws(cfg).await?;
    tracing::info!("ws connected (depth): {}", cfg.stream);

    let snap_url = format!(
        "{}/fapi/v1/depth?symbol={}&limit={}",
        cfg.rest_base, cfg.symbol, cfg.snapshot_limit
    );
    let mut sb = SeqBook::new();
    let mut snap_fut = snap_future(http.clone(), snap_url.clone(), 0);
    let mut awaiting = true;
    let stale = Duration::from_secs(cfg.stale_timeout_s);

    loop {
        tokio::select! {
            res = &mut snap_fut, if awaiting => {
                awaiting = false;
                match res {
                    Ok(snap) => {
                        if sb.apply_snapshot(&snap, true) {
                            tracing::info!("book ready last_u={}", sb.last_u);
                            publish_book(&**bus, cfg, &sb.book, now_ns(), now_ns(), sb.last_u, seq);
                        } else {
                            tracing::warn!("snapshot/first-delta mismatch -> refetch");
                            sb.reset_for_resync();
                            snap_fut = snap_future(http.clone(), snap_url.clone(), 200);
                            awaiting = true;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("snapshot fetch failed: {e} -> retry");
                        snap_fut = snap_future(http.clone(), snap_url.clone(), 500);
                        awaiting = true;
                    }
                }
            }
            msg = tokio::time::timeout(stale, ws.next()) => {
                let msg = msg.map_err(|_| anyhow::anyhow!("stale stream"))?;
                match msg {
                    Some(Ok(Message::Text(t))) => {
                        if let Some(delta) = parse_delta(t.as_str(), now_ns()) {
                            let (exch, recv) = (delta.exch_ns, delta.recv_ns);
                            match sb.on_delta(delta) {
                                Some(true) => publish_book(&**bus, cfg, &sb.book, exch, recv, sb.last_u, seq),
                                Some(false) => {} // buffered pre-snapshot
                                None => {
                                    tracing::warn!("sequence gap / crossed book -> resync");
                                    sb.reset_for_resync();
                                    snap_fut = snap_future(http.clone(), snap_url.clone(), 0);
                                    awaiting = true;
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Ping(p))) => { let _ = ws.send(Message::Pong(p)).await; }
                    Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
                    Some(Ok(_)) => {}
                    Some(Err(e)) => anyhow::bail!("ws error: {e}"),
                }
            }
        }
    }
}

fn publish_ticker(bus: &dyn Bus, topic: &str, instrument: &str, tk: &Ticker, seq: &mut u64) {
    bus.publish(Event::new(
        topic.to_string(),
        "binance",
        tk.recv_ns,
        nxt(seq),
        Payload::Book(BookUpdate {
            instrument: instrument.to_string(),
            bids: vec![(tk.bid, tk.bid_sz)],
            asks: vec![(tk.ask, tk.ask_sz)],
            update_id: Some(tk.u as u64),
            exch_ts_ns: tk.exch_ns,
            recv_ts_ns: tk.recv_ns,
        }),
    ));
}

/// bookTicker session: real-time best bid/ask, no snapshot/resync. Each frame is
/// a self-contained top-of-book, published as a single-level `BookUpdate` on the
/// same `.book` topic — so the processor consumes it identically. Coalescing is
/// handled by the bus `Conflate` policy at the subscriber, not here.
async fn session_bookticker(
    cfg: &BinanceCfg,
    bus: &Arc<dyn Bus>,
    seq: &mut u64,
) -> anyhow::Result<()> {
    let mut ws = connect_ws(cfg).await?;
    tracing::info!("ws connected (bookTicker): {}", cfg.stream);
    let topic = format!("market.binance.usdt_perp.{}.book", cfg.symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale stream"))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                if let Some(tk) = parse_book_ticker(t.as_str(), now_ns()) {
                    publish_ticker(&**bus, &topic, &cfg.instrument, &tk, seq);
                }
            }
            Some(Ok(Message::Ping(p))) => {
                let _ = ws.send(Message::Pong(p)).await;
            }
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("ws error: {e}"),
        }
    }
}

async fn run_loop(cfg: BinanceCfg, http: reqwest::Client, bus: Arc<dyn Bus>) {
    let book_ticker = cfg.stream.contains("bookTicker");
    let mut backoff = cfg.reconnect_base_ms;
    let mut seq = 0u64;
    loop {
        let res = if book_ticker {
            session_bookticker(&cfg, &bus, &mut seq).await
        } else {
            session(&cfg, &http, &bus, &mut seq).await
        };
        match res {
            Ok(()) => backoff = cfg.reconnect_base_ms,
            Err(e) => {
                tracing::warn!("session ended ({e}) -> reconnect in {backoff}ms");
                tokio::time::sleep(Duration::from_millis(backoff)).await;
                backoff = (backoff * 2).min(cfg.reconnect_max_ms);
            }
        }
    }
}

pub struct BinanceCollector {
    cfg: BinanceCfg,
    handle: Option<JoinHandle<()>>,
}

impl BinanceCollector {
    pub fn new(cfg: BinanceCfg) -> Self {
        BinanceCollector { cfg, handle: None }
    }
}

#[async_trait]
impl Module for BinanceCollector {
    fn name(&self) -> &'static str {
        "collector-binance"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(10));
        if let Some(p) = &self.cfg.socks_proxy {
            builder = builder.proxy(reqwest::Proxy::all(format!("socks5://{p}"))?);
        }
        let http = builder.build()?;
        self.handle = Some(tokio::spawn(run_loop(self.cfg.clone(), http, bus)));
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

#[cfg(test)]
mod tests {
    use super::host_port;

    #[test]
    fn host_port_parsing() {
        assert_eq!(host_port("wss://fstream.binance.com/public"), ("fstream.binance.com".into(), 443));
        assert_eq!(host_port("wss://example.com:8443/x"), ("example.com".into(), 8443));
    }
}
