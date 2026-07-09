//! `PolyCollector` — discovers the configured short-term series by slug,
//! tracks the active market set + lifecycle, streams the CLOB `book` channel,
//! and publishes `market.polymarket.catalog` + `.book` / `.trade` (DESIGN §5).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;

use arb_core::bus::Bus;
use arb_core::event::{Event, Payload};
use arb_core::model::{BookUpdate, MarketMeta, MarketStatus, Side, TradeTick};
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::book::{num, parse_levels, parse_price_changes, LevelSide, PolyBook};
use crate::slug::predict_slugs;

const GAMMA: &str = "https://gamma-api.polymarket.com";
const WS_URL: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct PolyCfg {
    pub prefix: String,
    pub window_sec: i64,
    pub n_windows: usize,
    pub kind: String,
    pub discovery_interval_s: u64,
    pub unsub_grace_s: i64,
    pub top_n: usize,
    pub user_agent: String,
}

impl Default for PolyCfg {
    fn default() -> Self {
        PolyCfg {
            prefix: "btc-updown-5m".into(),
            window_sec: 300,
            n_windows: 3,
            kind: "5m_updown".into(),
            discovery_interval_s: 5,
            unsub_grace_s: 10,
            top_n: 10,
            user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                         (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                .into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarketInfo {
    pub cid: String,
    pub slug: String,
    pub up_token: String,
    pub down_token: String,
    pub up_inst: String,
    pub down_inst: String,
    pub start_ns: i64,
    pub expiry_ns: i64,
    /// Per-market economics from Gamma (`orderMinSize`, `orderPriceMinTickSize`,
    /// `feeSchedule.rate`). `None` if the field is absent.
    pub min_order_size: Option<f64>,
    pub tick_size: Option<f64>,
    pub fee_rate: Option<f64>,
}

/// Parse a numeric Gamma field that may be a JSON number or a quoted string.
fn num_value(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn json_str_array(v: &Value) -> Vec<String> {
    match v {
        Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        Value::String(s) => serde_json::from_str::<Vec<String>>(s).unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Parse a Gamma market into a `MarketInfo`. `None` if it lacks a condition id
/// or both CLOB tokens (not yet tradeable → retry later). Expiry is derived from
/// the slug's window START + window length, avoiding the endDateIso midnight bug
/// (DESIGN §5).
pub fn parse_market(m: &Value, slug: &str, start_sec: i64, window_sec: i64) -> Option<MarketInfo> {
    let cid = m
        .get("conditionId")
        .or_else(|| m.get("condition_id"))
        .or_else(|| m.get("id"))
        .and_then(Value::as_str)?
        .to_string();
    let tokens = json_str_array(m.get("clobTokenIds").unwrap_or(&Value::Null));
    if tokens.len() < 2 {
        return None;
    }
    let outcomes = json_str_array(m.get("outcomes").unwrap_or(&Value::Null));
    let up_idx = outcomes
        .iter()
        .position(|o| {
            let l = o.to_ascii_lowercase();
            l == "up" || l == "yes"
        })
        .unwrap_or(0);
    let down_idx = if up_idx == 0 { 1 } else { 0 };
    let up_label = outcomes.get(up_idx).cloned().unwrap_or_else(|| "UP".into());
    let down_label = outcomes.get(down_idx).cloned().unwrap_or_else(|| "DOWN".into());
    let up_token = tokens.get(up_idx).cloned()?;
    let down_token = tokens.get(down_idx).cloned()?;
    let min_order_size = m.get("orderMinSize").and_then(num_value);
    let tick_size = m.get("orderPriceMinTickSize").and_then(num_value);
    let fee_rate = m.get("feeSchedule").and_then(|f| f.get("rate")).and_then(num_value);
    Some(MarketInfo {
        up_inst: format!("polymarket.{cid}.{}", up_label.to_ascii_uppercase()),
        down_inst: format!("polymarket.{cid}.{}", down_label.to_ascii_uppercase()),
        cid,
        slug: slug.to_string(),
        up_token,
        down_token,
        start_ns: start_sec * 1_000_000_000,
        expiry_ns: (start_sec + window_sec) * 1_000_000_000,
        min_order_size,
        tick_size,
        fee_rate,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Upcoming,
    Live,
    Expired,
}

#[derive(Clone, Debug)]
struct Market {
    info: MarketInfo,
    phase: Phase,
}

/// Snapshot of the desired WS asset set, pushed to the WS task on change.
#[derive(Clone, Default, PartialEq)]
struct Desired {
    tokens: Vec<String>,
    /// token_id -> (cid, instrument), sorted for stable equality.
    token_inst: Vec<(String, (String, String))>,
}

fn build_desired(markets: &HashMap<String, Market>) -> Desired {
    let mut tokens = Vec::new();
    let mut token_inst = Vec::new();
    for m in markets.values() {
        if m.phase == Phase::Expired {
            continue;
        }
        for (tok, inst) in [
            (&m.info.up_token, &m.info.up_inst),
            (&m.info.down_token, &m.info.down_inst),
        ] {
            tokens.push(tok.clone());
            token_inst.push((tok.clone(), (m.info.cid.clone(), inst.clone())));
        }
    }
    tokens.sort();
    token_inst.sort();
    Desired { tokens, token_inst }
}

fn nxt(seq: &mut u64) -> u64 {
    let s = *seq;
    *seq += 1;
    s
}

fn status_of(phase: Phase) -> MarketStatus {
    match phase {
        Phase::Upcoming => MarketStatus::Upcoming,
        Phase::Live => MarketStatus::Live,
        Phase::Expired => MarketStatus::Expired,
    }
}

fn publish_catalog(bus: &dyn Bus, m: &Market, kind: &str, seq: &mut u64) {
    let meta = MarketMeta {
        instrument: m.info.up_inst.clone(),
        kind: kind.to_string(),
        status: status_of(m.phase),
        start_ts_ns: Some(m.info.start_ns),
        expiry_ts_ns: Some(m.info.expiry_ns),
        winner: None,
        strike: None,
        min_order_size: m.info.min_order_size,
        tick_size: m.info.tick_size,
        fee_rate: m.info.fee_rate,
    };
    bus.publish(Event::new(
        "market.polymarket.catalog",
        "poly",
        now_ns(),
        nxt(seq),
        Payload::Meta(meta),
    ));
}

async fn fetch_market(
    http: &reqwest::Client,
    slug: &str,
    start_sec: i64,
    window_sec: i64,
) -> Option<MarketInfo> {
    let url = format!("{GAMMA}/markets?slug={slug}");
    let resp = match http.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("gamma fetch {slug}: {e}");
            return None;
        }
    };
    if !resp.status().is_success() {
        return None;
    }
    let data: Value = resp.json().await.ok()?;
    let arr = if data.is_array() {
        data.as_array().cloned().unwrap_or_default()
    } else {
        data.get("markets").and_then(Value::as_array).cloned().unwrap_or_default()
    };
    let m = arr.into_iter().next()?;
    parse_market(&m, slug, start_sec, window_sec)
}

async fn lifecycle_task(
    cfg: PolyCfg,
    http: reqwest::Client,
    bus: Arc<dyn Bus>,
    desired_tx: watch::Sender<Desired>,
) {
    let mut markets: HashMap<String, Market> = HashMap::new(); // keyed by slug
    let mut last_discovery: i64 = 0; // 0 -> first loop runs discovery (now is always > 0)
    let mut last_desired = Desired::default();
    let mut seq = 0u64;
    let disc_ns = (cfg.discovery_interval_s as i64) * 1_000_000_000;
    let grace_ns = cfg.unsub_grace_s * 1_000_000_000;

    loop {
        let now = now_ns();

        if now - last_discovery >= disc_ns {
            last_discovery = now;
            for (slug, start_sec) in
                predict_slugs(&cfg.prefix, cfg.window_sec, cfg.n_windows, now / 1_000_000_000)
            {
                if markets.contains_key(&slug) {
                    continue;
                }
                if let Some(info) = fetch_market(&http, &slug, start_sec, cfg.window_sec).await {
                    let phase = if now < info.start_ns {
                        Phase::Upcoming
                    } else if now < info.expiry_ns {
                        Phase::Live
                    } else {
                        continue; // window already over
                    };
                    let m = Market { info, phase };
                    tracing::info!("discovered {} ({}) phase={:?}", m.info.slug, m.info.cid, phase);
                    publish_catalog(&*bus, &m, &cfg.kind, &mut seq);
                    markets.insert(slug, m);
                }
            }
        }

        // Lifecycle transitions.
        for m in markets.values_mut() {
            match m.phase {
                Phase::Upcoming if now >= m.info.start_ns => {
                    m.phase = Phase::Live;
                    tracing::info!("LIVE {} ({})", m.info.slug, m.info.cid);
                    publish_catalog(&*bus, m, &cfg.kind, &mut seq);
                }
                Phase::Live if now >= m.info.expiry_ns => {
                    m.phase = Phase::Expired;
                    tracing::info!("EXPIRED {} ({})", m.info.slug, m.info.cid);
                    publish_catalog(&*bus, m, &cfg.kind, &mut seq);
                }
                _ => {}
            }
        }

        markets.retain(|_, m| !(m.phase == Phase::Expired && now >= m.info.expiry_ns + grace_ns));

        let desired = build_desired(&markets);
        if desired != last_desired {
            let _ = desired_tx.send(desired.clone());
            last_desired = desired;
        }

        // Sleep until the next *event* — the next discovery tick or the nearest
        // market boundary — rather than polling. Idle cost is ~one wake per
        // discovery interval plus one per Live/Expired transition, while
        // transitions still fire on time (DESIGN §5 sleep_until scheduler).
        let mut next_wake = last_discovery + disc_ns;
        for m in markets.values() {
            let boundary = match m.phase {
                Phase::Upcoming => m.info.start_ns,
                Phase::Live => m.info.expiry_ns,
                Phase::Expired => m.info.expiry_ns + grace_ns,
            };
            if boundary > now && boundary < next_wake {
                next_wake = boundary;
            }
        }
        let dur_ns = (next_wake - now).max(1_000_000) as u64; // >= 1ms guard
        tokio::time::sleep(Duration::from_nanos(dur_ns)).await;
    }
}

fn publish_book(bus: &dyn Bus, cid: &str, inst: &str, book: &PolyBook, recv: i64, top_n: usize, seq: &mut u64) {
    let (bids, asks) = book.top_n(top_n);
    bus.publish(Event::new(
        format!("market.polymarket.{cid}.book"),
        "poly",
        recv,
        nxt(seq),
        Payload::Book(BookUpdate {
            instrument: inst.to_string(),
            bids,
            asks,
            update_id: None,
            exch_ts_ns: recv,
            recv_ts_ns: recv,
        }),
    ));
}

fn route_frame(
    frame: &Value,
    books: &mut HashMap<String, PolyBook>,
    token_inst: &HashMap<String, (String, String)>,
    bus: &dyn Bus,
    top_n: usize,
    seq: &mut u64,
) {
    let items: Vec<&Value> = match frame.as_array() {
        Some(arr) => arr.iter().collect(),
        None => vec![frame],
    };
    for msg in items {
        let ev = msg
            .get("event_type")
            .and_then(Value::as_str)
            .or_else(|| msg.get("type").and_then(Value::as_str))
            .unwrap_or("");
        let recv = now_ns();
        match ev {
            "book" => {
                let asset = msg.get("asset_id").and_then(Value::as_str).unwrap_or("");
                if let Some((cid, inst)) = token_inst.get(asset) {
                    let bids = parse_levels(&msg["bids"], LevelSide::Bid);
                    let asks = parse_levels(&msg["asks"], LevelSide::Ask);
                    let book = books.entry(asset.to_string()).or_default();
                    book.apply_snapshot(&bids, &asks);
                    publish_book(bus, cid, inst, book, recv, top_n, seq);
                }
            }
            "price_change" => {
                let mut by: HashMap<String, Vec<(f64, f64, Side)>> = HashMap::new();
                for (asset, p, s, side) in parse_price_changes(msg) {
                    by.entry(asset).or_default().push((p, s, side));
                }
                for (asset, deltas) in by {
                    if let Some((cid, inst)) = token_inst.get(&asset) {
                        let book = books.entry(asset.clone()).or_default();
                        book.apply_delta(&deltas);
                        publish_book(bus, cid, inst, book, recv, top_n, seq);
                    }
                }
            }
            "last_trade_price" => {
                let asset = msg.get("asset_id").and_then(Value::as_str).unwrap_or("");
                if let Some((cid, inst)) = token_inst.get(asset) {
                    let side = match msg
                        .get("side")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_ascii_uppercase()
                        .as_str()
                    {
                        "BUY" => Side::Buy,
                        _ => Side::Sell,
                    };
                    if let (Some(p), Some(s)) = (num(&msg["price"]), num(&msg["size"])) {
                        bus.publish(Event::new(
                            format!("market.polymarket.{cid}.trade"),
                            "poly",
                            recv,
                            nxt(seq),
                            Payload::Trade(TradeTick {
                                instrument: inst.clone(),
                                price: p,
                                qty: s,
                                side,
                                exch_ts_ns: recv,
                                recv_ts_ns: recv,
                            }),
                        ));
                    }
                }
            }
            _ => {}
        }
    }
}

async fn ws_task(bus: Arc<dyn Bus>, mut desired_rx: watch::Receiver<Desired>, top_n: usize) {
    let mut seq = 0u64;
    loop {
        let desired = desired_rx.borrow().clone();
        if desired.tokens.is_empty() {
            if desired_rx.changed().await.is_err() {
                return;
            }
            continue;
        }
        let token_inst: HashMap<String, (String, String)> = desired.token_inst.iter().cloned().collect();

        match tokio_tungstenite::connect_async(WS_URL).await {
            Ok((mut ws, _)) => {
                let sub = serde_json::json!({
                    "type": "subscribe", "channel": "book", "assets_ids": desired.tokens,
                });
                if ws.send(Message::Text(sub.to_string().into())).await.is_err() {
                    tracing::warn!("ws subscribe failed");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
                tracing::info!("ws subscribed {} tokens", desired.tokens.len());
                let mut books: HashMap<String, PolyBook> = HashMap::new();
                loop {
                    tokio::select! {
                        changed = desired_rx.changed() => {
                            if changed.is_err() { return; }
                            tracing::info!("desired asset set changed -> resubscribe");
                            break;
                        }
                        msg = ws.next() => match msg {
                            Some(Ok(Message::Text(t))) => {
                                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                                    route_frame(&v, &mut books, &token_inst, &*bus, top_n, &mut seq);
                                }
                            }
                            Some(Ok(Message::Close(_))) | None => {
                                tracing::info!("ws closed");
                                break;
                            }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => {
                                tracing::warn!("ws error: {e}");
                                break;
                            }
                        }
                    }
                }
            }
            Err(e) => tracing::warn!("ws connect failed: {e}"),
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub struct PolyCollector {
    cfg: PolyCfg,
    handles: Vec<JoinHandle<()>>,
}

impl PolyCollector {
    pub fn new(cfg: PolyCfg) -> Self {
        PolyCollector { cfg, handles: Vec::new() }
    }
}

#[async_trait]
impl Module for PolyCollector {
    fn name(&self) -> &'static str {
        "collector-polymarket"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        let http = reqwest::Client::builder()
            .user_agent(self.cfg.user_agent.clone())
            .build()?;
        let (tx, rx) = watch::channel(Desired::default());
        self.handles
            .push(tokio::spawn(lifecycle_task(self.cfg.clone(), http, bus.clone(), tx)));
        self.handles.push(tokio::spawn(ws_task(bus, rx, self.cfg.top_n)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_market_basic() {
        let m = json!({
            "conditionId": "0xCID",
            "clobTokenIds": "[\"up_tok\", \"down_tok\"]",
            "outcomes": "[\"Up\", \"Down\"]",
            "slug": "btc-updown-5m-1200",
            "orderMinSize": "5",
            "orderPriceMinTickSize": "0.01",
            "feeSchedule": { "rate": 0.07, "exponent": 1 }
        });
        let mi = parse_market(&m, "btc-updown-5m-1200", 1200, 300).unwrap();
        assert_eq!(mi.cid, "0xCID");
        assert_eq!(mi.up_token, "up_tok");
        assert_eq!(mi.down_token, "down_tok");
        assert_eq!(mi.up_inst, "polymarket.0xCID.UP");
        assert_eq!(mi.down_inst, "polymarket.0xCID.DOWN");
        assert_eq!(mi.start_ns, 1_200_000_000_000);
        assert_eq!(mi.expiry_ns, 1_500_000_000_000);
        assert_eq!(mi.min_order_size, Some(5.0));
        assert_eq!(mi.tick_size, Some(0.01));
        assert_eq!(mi.fee_rate, Some(0.07));
    }

    #[test]
    fn parse_market_down_first_orientation() {
        // outcomes reversed -> up index follows the label, not position 0
        let m = json!({
            "conditionId": "0xCID",
            "clobTokenIds": "[\"down_tok\", \"up_tok\"]",
            "outcomes": "[\"Down\", \"Up\"]",
            "slug": "btc-updown-5m-1200"
        });
        let mi = parse_market(&m, "btc-updown-5m-1200", 1200, 300).unwrap();
        assert_eq!(mi.up_token, "up_tok");
        assert_eq!(mi.down_token, "down_tok");
    }

    #[test]
    fn parse_market_missing_tokens_is_none() {
        let m = json!({ "conditionId": "0xCID", "outcomes": "[\"Up\",\"Down\"]" });
        assert!(parse_market(&m, "s", 1200, 300).is_none());
    }
}
