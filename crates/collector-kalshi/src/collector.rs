//! `KalshiCollector` — discovers the configured crypto series' open windows,
//! tracks the active market set + lifecycle, streams the `orderbook_delta` WS
//! channel (RSA-PSS signed), and publishes `market.kalshi.catalog` + `.book` /
//! `.trade`. Mirrors `PolyCollector`; ports the Python reference's Kalshi
//! collector (discovery + reflected book + REST fallback + trade tape).
//!
//! WS is the canonical book source; REST `GET /orderbook` is a connect-failure
//! fallback (also the sole source when no API key is configured).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::http::Request;
use tokio_tungstenite::tungstenite::Message;

use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::model::{BookUpdate, MarketMeta, MarketStatus, Side, TradeTick};
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::auth::Signer;
use crate::book::{num, parse_rest_orderbook, parse_ws_delta, parse_ws_snapshot, KalshiBook};

// external-api.* resolves directly to the us-east-2 ELB (Kalshi's engine region),
// no CDN hop — matches the WS host and is lowest-latency from a us-east-2 box.
const REST_BASE: &str = "https://external-api.kalshi.com/trade-api/v2";
const WS_URL: &str = "wss://external-api-ws.kalshi.com/trade-api/ws/v2";
const WS_PATH: &str = "/trade-api/ws/v2";

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct KalshiCfg {
    /// Crypto series to track, e.g. `["KXBTC15M", "KXETH15M"]`.
    pub series_tickers: Vec<String>,
    pub kind: String,
    pub discovery_interval_s: u64,
    pub n_windows: usize,
    pub min_tte_s: i64,
    pub max_tte_s: i64,
    pub unsub_grace_s: i64,
    pub top_n: usize,
    pub rest_base: String,
    pub ws_url: String,
    /// API key id (or env `KALSHI_KEY_ID`); empty → REST-only (no WS book).
    pub key_id: String,
    /// Path to the RSA private-key PEM (or env `KALSHI_PRIVATE_KEY_PATH`).
    pub private_key_path: String,
    pub user_agent: String,
    pub trades_enabled: bool,
    pub ob_poll_s: u64,
    pub trades_poll_s: u64,
    /// Per-series ATM reference: maps a strike-laddered series to the bus instrument
    /// whose mid marks "the money", so discovery tracks the at-the-money strikes.
    /// e.g. {KXINXU: databento.ES, KXWTI: databento.CL}. Series absent here use
    /// nearest-close ordering (fine for up/down series like KXBTC15M).
    pub atm_references: HashMap<String, String>,
}

impl Default for KalshiCfg {
    fn default() -> Self {
        KalshiCfg {
            series_tickers: vec!["KXBTC15M".into()],
            kind: "15m_updown".into(),
            discovery_interval_s: 30,
            n_windows: 3,
            min_tte_s: 10,
            max_tte_s: 3600,
            unsub_grace_s: 60,
            top_n: 10,
            rest_base: REST_BASE.into(),
            ws_url: WS_URL.into(),
            key_id: String::new(),
            private_key_path: String::new(),
            user_agent: "Mozilla/5.0".into(),
            trades_enabled: true,
            ob_poll_s: 2,
            trades_poll_s: 3,
            atm_references: HashMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Upcoming,
    Live,
    Expired,
}

#[derive(Clone, Debug)]
struct Market {
    ticker: String,
    title: String,
    open_ns: i64,
    close_ns: i64,
    phase: Phase,
    instrument: String,
}

/// Desired active set, pushed to the WS / poll / trade tasks on change.
#[derive(Clone, Default, PartialEq)]
struct Desired {
    tickers: Vec<String>,
    /// `(ticker, instrument)`, sorted for stable equality.
    ticker_inst: Vec<(String, String)>,
}

fn nxt(seq: &mut u64) -> u64 {
    let s = *seq;
    *seq += 1;
    s
}

fn phase_of(now: i64, open_ns: i64, close_ns: i64) -> Phase {
    if now < open_ns {
        Phase::Upcoming
    } else if now < close_ns {
        Phase::Live
    } else {
        Phase::Expired
    }
}

fn status_of(phase: Phase) -> MarketStatus {
    match phase {
        Phase::Upcoming => MarketStatus::Upcoming,
        Phase::Live => MarketStatus::Live,
        Phase::Expired => MarketStatus::Expired,
    }
}

/// RFC3339 (`2026-06-21T14:00:00Z`) → unix nanoseconds.
fn iso_ns(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .and_then(|dt| dt.timestamp_nanos_opt())
}

fn build_desired(markets: &HashMap<String, Market>) -> Desired {
    let mut tickers = Vec::new();
    let mut ticker_inst = Vec::new();
    for m in markets.values() {
        if m.phase == Phase::Expired {
            continue;
        }
        tickers.push(m.ticker.clone());
        ticker_inst.push((m.ticker.clone(), m.instrument.clone()));
    }
    tickers.sort();
    ticker_inst.sort();
    Desired { tickers, ticker_inst }
}

fn publish_catalog(bus: &dyn Bus, m: &Market, kind: &str, seq: &mut u64) {
    let meta = MarketMeta {
        instrument: m.instrument.clone(),
        kind: kind.to_string(),
        status: status_of(m.phase),
        start_ts_ns: Some(m.open_ns),
        expiry_ts_ns: Some(m.close_ns),
        winner: None,
        min_order_size: Some(1.0), // Kalshi trades whole contracts
        tick_size: Some(0.01),     // 1¢ tick
        fee_rate: None,            // consumers fall back to config
    };
    bus.publish(Event::new(
        "market.kalshi.catalog",
        "kalshi",
        now_ns(),
        nxt(seq),
        Payload::Meta(meta),
    ));
}

/// Publish BOTH outcome instruments for one market from the single book:
/// `kalshi.{ticker}.YES` (Up) and `kalshi.{ticker}.NO` (Down). They share the
/// `market.kalshi.{ticker}.book` topic and are distinguished by `instrument`
/// (the bus conflates per-instrument). The NO ladder is the mirror of YES —
/// publishing both gives the long-only executor a buyable instrument for each
/// direction (buy YES to bet Up, buy NO to bet Down).
fn publish_books(
    bus: &dyn Bus,
    ticker: &str,
    book: &KalshiBook,
    recv: i64,
    top_n: usize,
    seq: &mut u64,
) {
    let topic = format!("market.kalshi.{ticker}.book");
    for (suffix, (bids, asks)) in [
        ("YES", book.top_n(top_n)),
        ("NO", book.top_n_no(top_n)),
    ] {
        bus.publish(Event::new(
            topic.clone(),
            "kalshi",
            recv,
            nxt(seq),
            Payload::Book(BookUpdate {
                instrument: format!("kalshi.{ticker}.{suffix}"),
                bids,
                asks,
                update_id: None,
                exch_ts_ns: recv,
                recv_ts_ns: recv,
            }),
        ));
    }
}

// ───────────────────────────── REST helpers ─────────────────────────────

/// Strike from a laddered market ticker: "...-T7434.9999" -> 7434.9999.
fn strike_of(ticker: &str) -> Option<f64> {
    ticker.rsplit_once("-T").and_then(|(_, s)| s.parse::<f64>().ok())
}

async fn list_open_markets(
    http: &reqwest::Client,
    rest_base: &str,
    series: &str,
) -> anyhow::Result<Vec<Value>> {
    // Paginate: laddered series (e.g. KXINXU) have hundreds of strikes per event
    // across several events, easily exceeding one page — the at-the-money band can
    // sit past the first 1000, so a single page silently drops it.
    let mut out: Vec<Value> = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..6 {
        let mut url = format!("{rest_base}/markets?series_ticker={series}&status=open&limit=1000");
        if let Some(c) = &cursor {
            url.push_str("&cursor=");
            url.push_str(c);
        }
        let resp = http.get(&url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("HTTP {}", resp.status());
        }
        let data: Value = resp.json().await?;
        if let Some(arr) = data.get("markets").and_then(Value::as_array) {
            out.extend(arr.iter().cloned());
        }
        cursor = data
            .get("cursor")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from);
        if cursor.is_none() {
            break;
        }
    }
    Ok(out)
}

async fn get_orderbook(
    http: &reqwest::Client,
    rest_base: &str,
    ticker: &str,
) -> anyhow::Result<Value> {
    let url = format!("{rest_base}/markets/{ticker}/orderbook");
    let resp = http.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    Ok(resp.json().await?)
}

async fn get_trades(
    http: &reqwest::Client,
    rest_base: &str,
    ticker: &str,
    min_ts: Option<i64>,
) -> anyhow::Result<Vec<Value>> {
    let mut url = format!("{rest_base}/markets/trades?ticker={ticker}&limit=100");
    if let Some(t) = min_ts {
        url.push_str(&format!("&min_ts={t}"));
    }
    let resp = http.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}", resp.status());
    }
    let data: Value = resp.json().await?;
    Ok(data.get("trades").and_then(Value::as_array).cloned().unwrap_or_default())
}

// ───────────────────────────── Loop 1: lifecycle ────────────────────────

async fn lifecycle_task(
    cfg: KalshiCfg,
    http: reqwest::Client,
    bus: Arc<dyn Bus>,
    desired_tx: watch::Sender<Desired>,
    atm: HashMap<String, Arc<AtomicU64>>,
) {
    let mut markets: HashMap<String, Market> = HashMap::new();
    let mut last_desired = Desired::default();
    let mut seq = 0u64;
    let grace_ns = cfg.unsub_grace_s * 1_000_000_000;

    loop {
        let now = now_ns();

        // Discovery: list open markets per series, pick the nearest n_windows.
        for series in &cfg.series_tickers {
            let list = match list_open_markets(&http, &cfg.rest_base, series).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("kalshi discovery {series}: {e}");
                    continue;
                }
            };
            // (close, atm_dist, ticker, title, open). For strike-laddered series
            // (e.g. KXINXU) many markets share a close_time; if an ATM reference mid
            // is available we rank within a settlement by distance from it, so we
            // track the at-the-money strikes regardless of where the index sits.
            let atm_mid = atm
                .get(series)
                .map(|a| f64::from_bits(a.load(Ordering::Relaxed)))
                .unwrap_or(0.0);
            tracing::info!("kalshi discovery {series}: atm_mid={atm_mid:.2}");
            let mut cands: Vec<(i64, f64, String, String, i64)> = Vec::new();
            for m in &list {
                let ticker = match m.get("ticker").and_then(Value::as_str) {
                    Some(t) => t.to_string(),
                    None => continue,
                };
                let close = match m.get("close_time").and_then(Value::as_str).and_then(iso_ns) {
                    Some(c) => c,
                    None => continue,
                };
                let tte = close - now;
                if tte < cfg.min_tte_s * 1_000_000_000 || tte > cfg.max_tte_s * 1_000_000_000 {
                    continue;
                }
                let open = m
                    .get("open_time")
                    .and_then(Value::as_str)
                    .and_then(iso_ns)
                    .unwrap_or(now);
                let title = m
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or(&ticker)
                    .to_string();
                // strike from "...-T<strike>"; distance from the ATM reference mid.
                let atm_dist = match (atm_mid > 0.0, strike_of(&ticker)) {
                    (true, Some(k)) => (k - atm_mid).abs(),
                    _ => 0.0,
                };
                cands.push((close, atm_dist, ticker, title, open));
            }
            // nearest settlement first; within it, nearest to the money first.
            cands.sort_by(|a, b| {
                a.0.cmp(&b.0)
                    .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            for (i, (close, _dist, ticker, title, open)) in
                cands.into_iter().take(cfg.n_windows).enumerate()
            {
                if markets.contains_key(&ticker) {
                    continue;
                }
                let phase = phase_of(now, open, close);
                if phase == Phase::Expired {
                    continue;
                }
                let role = if i == 0 { "current".to_string() } else { format!("upcoming_{i}") };
                let instrument = format!("kalshi.{ticker}.YES");
                let m = Market { ticker: ticker.clone(), title, open_ns: open, close_ns: close, phase, instrument };
                tracing::info!(
                    "discovered {} [{}] {:?} close in {}s",
                    m.ticker,
                    role,
                    m.title,
                    (close - now) / 1_000_000_000
                );
                publish_catalog(&*bus, &m, &cfg.kind, &mut seq);
                markets.insert(ticker, m);
            }
        }

        // Lifecycle transitions.
        for m in markets.values_mut() {
            match m.phase {
                Phase::Upcoming if now >= m.open_ns => {
                    m.phase = Phase::Live;
                    tracing::info!("LIVE {}", m.ticker);
                    publish_catalog(&*bus, m, &cfg.kind, &mut seq);
                }
                Phase::Live if now >= m.close_ns => {
                    m.phase = Phase::Expired;
                    tracing::info!("EXPIRED {}", m.ticker);
                    publish_catalog(&*bus, m, &cfg.kind, &mut seq);
                }
                _ => {}
            }
        }
        markets.retain(|_, m| !(m.phase == Phase::Expired && now >= m.close_ns + grace_ns));

        let desired = build_desired(&markets);
        if desired != last_desired {
            let _ = desired_tx.send(desired.clone());
            last_desired = desired;
        }

        tokio::time::sleep(Duration::from_secs(cfg.discovery_interval_s.max(1))).await;
    }
}

// ───────────────────────────── Loop 2: WebSocket ────────────────────────

fn build_ws_request(ws_url: &str, signer: &Signer) -> anyhow::Result<Request<()>> {
    let ts_ms = now_ns() / 1_000_000;
    let (ts, sig) = signer.sign("GET", WS_PATH, ts_ms)?;
    let mut req = ws_url.into_client_request()?;
    let h = req.headers_mut();
    h.insert(
        HeaderName::from_static("kalshi-access-key"),
        HeaderValue::from_str(signer.key_id())?,
    );
    h.insert(
        HeaderName::from_static("kalshi-access-timestamp"),
        HeaderValue::from_str(&ts)?,
    );
    h.insert(
        HeaderName::from_static("kalshi-access-signature"),
        HeaderValue::from_str(&sig)?,
    );
    h.insert(
        HeaderName::from_static("user-agent"),
        HeaderValue::from_static("Mozilla/5.0"),
    );
    Ok(req)
}

fn route_ws(
    frame: &Value,
    books: &mut HashMap<String, KalshiBook>,
    ticker_inst: &HashMap<String, String>,
    bus: &dyn Bus,
    top_n: usize,
    seq: &mut u64,
) -> bool {
    let t = frame.get("type").and_then(Value::as_str).unwrap_or("");
    let ticker = frame
        .get("msg")
        .and_then(|m| m.get("market_ticker"))
        .and_then(Value::as_str)
        .unwrap_or("");
    if !ticker_inst.contains_key(ticker) {
        return false; // not a market we track
    }
    let recv = now_ns();
    match t {
        "orderbook_snapshot" => {
            let (yes, no, s) = parse_ws_snapshot(frame);
            let book = books.entry(ticker.to_string()).or_default();
            book.apply_snapshot(&yes, &no, s);
            publish_books(bus, ticker, book, recv, top_n, seq);
            false
        }
        "orderbook_delta" => {
            if let Some((side, p, d, s)) = parse_ws_delta(frame) {
                let book = books.entry(ticker.to_string()).or_default();
                let gap = book.apply_delta(side, p, d, s);
                publish_books(bus, ticker, book, recv, top_n, seq);
                gap
            } else {
                false
            }
        }
        _ => false,
    }
}

async fn ws_task(
    cfg: KalshiCfg,
    signer: Arc<Signer>,
    bus: Arc<dyn Bus>,
    mut desired_rx: watch::Receiver<Desired>,
    ws_streaming: Arc<AtomicBool>,
) {
    let mut seq = 0u64;
    let mut backoff = 1u64;
    loop {
        let desired = desired_rx.borrow().clone();
        if desired.tickers.is_empty() {
            if desired_rx.changed().await.is_err() {
                return;
            }
            continue;
        }
        let ticker_inst: HashMap<String, String> = desired.ticker_inst.iter().cloned().collect();

        let req = match build_ws_request(&cfg.ws_url, &signer) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("kalshi ws request build: {e}");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };
        match connect_async(req).await {
            Ok((mut ws, _)) => {
                backoff = 1;
                let sub = json!({
                    "id": 1, "cmd": "subscribe",
                    "params": {"channels": ["orderbook_delta"], "market_tickers": desired.tickers},
                });
                if ws.send(Message::Text(sub.to_string().into())).await.is_err() {
                    tracing::warn!("kalshi ws subscribe failed");
                    continue;
                }
                tracing::info!("kalshi ws subscribed {} tickers", desired.tickers.len());
                ws_streaming.store(true, Ordering::Relaxed);
                let mut books: HashMap<String, KalshiBook> = HashMap::new();
                loop {
                    tokio::select! {
                        changed = desired_rx.changed() => {
                            if changed.is_err() { ws_streaming.store(false, Ordering::Relaxed); return; }
                            tracing::info!("kalshi desired set changed -> resubscribe");
                            break;
                        }
                        msg = ws.next() => match msg {
                            Some(Ok(Message::Text(t))) => {
                                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                                    if route_ws(&v, &mut books, &ticker_inst, &*bus, cfg.top_n, &mut seq) {
                                        tracing::info!("kalshi seq gap -> reconnect");
                                        break;
                                    }
                                }
                            }
                            Some(Ok(Message::Close(_))) | None => { tracing::info!("kalshi ws closed"); break; }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => { tracing::warn!("kalshi ws error: {e}"); break; }
                        }
                    }
                }
                ws_streaming.store(false, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!("kalshi ws connect failed: {e}");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
            }
        }
    }
}

// ───────────────────────── Loop 3: REST book fallback ───────────────────

async fn poll_task(
    cfg: KalshiCfg,
    http: reqwest::Client,
    bus: Arc<dyn Bus>,
    mut desired_rx: watch::Receiver<Desired>,
    ws_streaming: Arc<AtomicBool>,
    has_key: bool,
) {
    let mut seq = 0u64;
    let mut books: HashMap<String, KalshiBook> = HashMap::new();
    let cadence = Duration::from_secs(cfg.ob_poll_s.max(1));
    loop {
        if desired_rx.borrow().tickers.is_empty() {
            if desired_rx.changed().await.is_err() {
                return;
            }
            continue;
        }
        // REST is a connect-failure fallback: idle while the WS is streaming.
        if has_key && ws_streaming.load(Ordering::Relaxed) {
            tokio::time::sleep(cadence).await;
            continue;
        }
        let desired = desired_rx.borrow().clone();
        for (ticker, _inst) in &desired.ticker_inst {
            match get_orderbook(&http, &cfg.rest_base, ticker).await {
                Ok(ob) => {
                    let recv = now_ns();
                    let (yes, no) = parse_rest_orderbook(&ob);
                    let book = books.entry(ticker.clone()).or_default();
                    book.apply_snapshot(&yes, &no, None);
                    publish_books(&*bus, ticker, book, recv, cfg.top_n, &mut seq);
                }
                Err(e) => tracing::debug!("kalshi ob poll {ticker}: {e}"),
            }
        }
        tokio::time::sleep(cadence).await;
    }
}

// ───────────────────────────── Loop 4: trade tape ───────────────────────

async fn trades_task(
    cfg: KalshiCfg,
    http: reqwest::Client,
    bus: Arc<dyn Bus>,
    mut desired_rx: watch::Receiver<Desired>,
) {
    let mut seq = 0u64;
    let mut watermark: HashMap<String, i64> = HashMap::new();
    let cadence = Duration::from_secs(cfg.trades_poll_s.max(1));
    loop {
        if desired_rx.borrow().tickers.is_empty() {
            if desired_rx.changed().await.is_err() {
                return;
            }
            continue;
        }
        let desired = desired_rx.borrow().clone();
        for (ticker, inst) in &desired.ticker_inst {
            let wm = *watermark.get(ticker).unwrap_or(&0);
            let min_ts = if wm > 0 { Some(wm / 1_000_000_000) } else { None };
            match get_trades(&http, &cfg.rest_base, ticker, min_ts).await {
                Ok(trades) => {
                    let recv = now_ns();
                    let mut max_ts = wm;
                    for tr in &trades {
                        let ts = tr
                            .get("created_time")
                            .and_then(Value::as_str)
                            .and_then(iso_ns)
                            .unwrap_or(recv);
                        if ts <= wm {
                            continue; // already published in a prior poll
                        }
                        if let Some(tick) = parse_trade(tr, inst, ts, recv) {
                            bus.publish(Event::new(
                                format!("market.kalshi.{ticker}.trade"),
                                "kalshi",
                                recv,
                                nxt(&mut seq),
                                Payload::Trade(tick),
                            ));
                            max_ts = max_ts.max(ts);
                        }
                    }
                    watermark.insert(ticker.clone(), max_ts);
                }
                Err(e) => tracing::debug!("kalshi trade poll {ticker}: {e}"),
            }
        }
        tokio::time::sleep(cadence).await;
    }
}

/// Public trade-tape row → `TradeTick` (price in YES/Up terms, [0,1]).
fn parse_trade(t: &Value, inst: &str, exch_ns: i64, recv_ns: i64) -> Option<TradeTick> {
    let qty = t.get("count_fp").and_then(num).or_else(|| t.get("count").and_then(num))?;
    let price = if let Some(p) = t.get("yes_price_dollars").and_then(num) {
        p
    } else if let Some(c) = t.get("yes_price").and_then(num) {
        c / 100.0
    } else if let Some(p) = t.get("no_price_dollars").and_then(num) {
        1.0 - p
    } else {
        1.0 - t.get("no_price").and_then(num)? / 100.0
    };
    let side = match t.get("taker_outcome_side").and_then(Value::as_str).unwrap_or("") {
        "yes" => Side::Buy,
        "no" => Side::Sell,
        _ => Side::Buy,
    };
    Some(TradeTick {
        instrument: inst.to_string(),
        price,
        qty,
        side,
        exch_ts_ns: exch_ns,
        recv_ts_ns: recv_ns,
    })
}

// ───────────────────────────── Module ───────────────────────────────────

pub struct KalshiCollector {
    cfg: KalshiCfg,
    handles: Vec<JoinHandle<()>>,
}

impl KalshiCollector {
    pub fn new(cfg: KalshiCfg) -> Self {
        KalshiCollector { cfg, handles: Vec::new() }
    }
}

#[async_trait]
impl Module for KalshiCollector {
    fn name(&self) -> &'static str {
        "collector-kalshi"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        let http = reqwest::Client::builder()
            .user_agent(self.cfg.user_agent.clone())
            .build()?;

        // Credentials: config first, then env. The PEM path (not the key) is all
        // that lives in config; the secret is the file it points at.
        let key_id = if self.cfg.key_id.is_empty() {
            std::env::var("KALSHI_KEY_ID").unwrap_or_default()
        } else {
            self.cfg.key_id.clone()
        };
        let pem = if self.cfg.private_key_path.is_empty() {
            std::env::var("KALSHI_PRIVATE_KEY_PATH").unwrap_or_default()
        } else {
            self.cfg.private_key_path.clone()
        };
        let signer = if !key_id.is_empty() && !pem.is_empty() {
            match Signer::load(&key_id, &pem) {
                Ok(s) => Some(Arc::new(s)),
                Err(e) => {
                    tracing::warn!("kalshi auth load failed ({e}) — running REST-only");
                    None
                }
            }
        } else {
            tracing::warn!("kalshi: no key_id/private_key_path — running REST-only (no WS book)");
            None
        };
        let has_key = signer.is_some();

        let (tx, rx) = watch::channel(Desired::default());
        let ws_streaming = Arc::new(AtomicBool::new(false));

        // Per-series ATM reference: track each configured instrument's mid (f64 bits)
        // so discovery centres on the at-the-money strikes of each laddered series.
        let mut atm: HashMap<String, Arc<AtomicU64>> = HashMap::new();
        for (series, inst) in &self.cfg.atm_references {
            let cell = Arc::new(AtomicU64::new(0));
            atm.insert(series.clone(), cell.clone());
            let inst = inst.clone();
            let topic = format!("market.{inst}.book");
            tracing::info!("kalshi: ATM reference {series} = {inst}");
            let mut sub = bus.subscribe(&topic, 256, Policy::Conflate(key_by_instrument));
            self.handles.push(tokio::spawn(async move {
                while let Some(ev) = sub.recv().await {
                    if let Payload::Book(b) = &ev.payload {
                        if b.instrument == inst {
                            if let (Some(&(bid, _)), Some(&(ask, _))) =
                                (b.bids.first(), b.asks.first())
                            {
                                cell.store(((bid + ask) / 2.0).to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                }
            }));
        }

        self.handles.push(tokio::spawn(lifecycle_task(
            self.cfg.clone(),
            http.clone(),
            bus.clone(),
            tx,
            atm,
        )));
        self.handles.push(tokio::spawn(poll_task(
            self.cfg.clone(),
            http.clone(),
            bus.clone(),
            rx.clone(),
            ws_streaming.clone(),
            has_key,
        )));
        if let Some(signer) = signer {
            self.handles.push(tokio::spawn(ws_task(
                self.cfg.clone(),
                signer,
                bus.clone(),
                rx.clone(),
                ws_streaming.clone(),
            )));
        }
        if self.cfg.trades_enabled {
            self.handles.push(tokio::spawn(trades_task(
                self.cfg.clone(),
                http,
                bus,
                rx,
            )));
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
