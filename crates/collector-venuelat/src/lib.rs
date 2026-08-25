//! Venue-latency probe — LOGGING ONLY, never touches the trading path.
//!
//! Answers "which venue/ticker prints a price change first" by subscribing to
//! top-of-book on several venues at once and recording, for every update, both
//! the EXCHANGE timestamp and our RECEIVE timestamp.
//!
//! Two rules from the 2026-08-18 study shape this (memory: venue-latency-geometry):
//!
//! 1. Cross-venue lead-lag MUST use `exch_ts`. On the exchanges own clocks the
//!    venues repriced within ~40ms of each other — inside NTP skew — and no
//!    venue demonstrably led. `recv_ts` ordering is OUR NETWORK GEOMETRY: bybit
//!    looked 8ms fast only because the collector VPS sat in the same Singapore
//!    AZ, while binance (Tokyo) showed 188-315ms and okx 270-630ms.
//! 2. `recv - exch` is deliverability, and it balloons during bursts — exactly
//!    when it matters. Both stamps are logged so the two questions stay apart.
//!
//! Every feed here runs over the SAME box and the SAME proxy, so the network
//! leg is common-mode and `recv` differences are attributable to the venue
//! rather than to our routing.
//!
//! ISOLATION: `spawn()` takes a whole OS thread and runs a private
//! current-thread runtime on it, so this cannot occupy a worker of the trading
//! runtime. Samples cross to the writer over a bounded channel that DROPS when
//! full — a slow disk stalls the log, never a feed, and never the trader.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_socks::tcp::Socks5Stream;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use arb_core::now_ns;

/// One top-of-book feed to probe.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct FeedCfg {
    /// "binance" | "bybit" | "okx" — selects the wire dialect.
    pub venue: String,
    /// "perp" | "spot" — label only; the url/topic already encode it.
    pub market: String,
    /// Venue-native symbol: BTCUSDT, BTC-USDT-SWAP, ...
    pub symbol: String,
    pub url: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct VenueLatCfg {
    pub enabled: bool,
    /// SOCKS5 for the geoblocked box (Tokyo relay: "127.0.0.1:1081").
    pub socks_proxy: Option<String>,
    /// CSV output path; a header is written when the file is created.
    pub out_path: String,
    /// Bounded queue to the writer. Full => sample dropped (counted, logged).
    pub queue: usize,
    pub stale_timeout_s: u64,
    pub reconnect_base_ms: u64,
    pub reconnect_max_ms: u64,
    pub feeds: Vec<FeedCfg>,
}

impl Default for VenueLatCfg {
    fn default() -> Self {
        VenueLatCfg {
            enabled: false,
            socks_proxy: Some("127.0.0.1:1081".into()),
            out_path: "data/venue_latency.csv".into(),
            queue: 16384,
            stale_timeout_s: 60,
            reconnect_base_ms: 500,
            reconnect_max_ms: 30_000,
            feeds: Vec::new(),
        }
    }
}

/// One observed top-of-book update.
struct Sample {
    recv_ts_ns: i64,
    /// Exchange-stamped time (ns). `None` when the venue omits it on this
    /// channel — such rows are still logged, but are NOT usable for lead-lag.
    exch_ts_ns: Option<i64>,
    venue: &'static str,
    market: String,
    symbol: String,
    bid: f64,
    ask: f64,
}

fn sf(v: &Value, k: &str) -> Option<f64> {
    v.get(k)?.as_str()?.parse().ok()
}

/// First level of a `[["px","sz"],...]` ladder.
fn lvl0(v: &Value, k: &str) -> Option<f64> {
    v.get(k)?
        .as_array()?
        .first()?
        .as_array()?
        .first()?
        .as_str()?
        .parse()
        .ok()
}

/// Parse one frame into a top-of-book sample. `None` for control frames
/// (subscribe acks, pongs, partial snapshots).
fn parse(venue: &'static str, symbol: &str, market: &str, txt: &str, recv_ts_ns: i64) -> Option<Sample> {
    let v: Value = serde_json::from_str(txt).ok()?;
    let mk = |exch_ms: Option<f64>, bid: f64, ask: f64| {
        Some(Sample {
            recv_ts_ns,
            exch_ts_ns: exch_ms.map(|m| (m * 1e6) as i64),
            venue,
            market: market.to_string(),
            symbol: symbol.to_string(),
            bid,
            ask,
        })
    };
    match venue {
        // bookTicker: {"u":..,"s":"BTCUSDT","b":"..","a":"..","E":ms,"T":ms}
        // `T` is the matching-engine transaction time, `E` the push time.
        "binance" => {
            let d = v.get("data").unwrap_or(&v);
            let e = d.get("T").or_else(|| d.get("E")).and_then(Value::as_f64);
            mk(e, sf(d, "b")?, sf(d, "a")?)
        }
        // {"topic":"orderbook.1.BTCUSDT","ts":ms,"cts":ms,"data":{"b":[[px,sz]],"a":[..]}}
        // `cts` is the matching-engine time, `ts` the publish time.
        "bybit" => {
            let d = v.get("data")?;
            let e = v.get("cts").or_else(|| v.get("ts")).and_then(Value::as_f64);
            mk(e, lvl0(d, "b")?, lvl0(d, "a")?)
        }
        // {"arg":{..},"data":[{"bids":[[px,sz,..]],"asks":[..],"ts":"ms"}]}
        "okx" => {
            let d = v.get("data")?.as_array()?.first()?;
            let e = d.get("ts").and_then(Value::as_str).and_then(|t| t.parse::<f64>().ok());
            mk(e, lvl0(d, "bids")?, lvl0(d, "asks")?)
        }
        _ => None,
    }
}

/// Subscribe frame for venues that need one after connecting.
fn sub_frame(venue: &str, symbol: &str) -> Option<String> {
    match venue {
        "bybit" => Some(format!("{{\"op\":\"subscribe\",\"args\":[\"orderbook.1.{symbol}\"]}}")),
        // bbo-tbt = tick-by-tick best bid/offer, the lowest-latency OKX channel.
        "okx" => Some(format!(
            "{{\"op\":\"subscribe\",\"args\":[{{\"channel\":\"bbo-tbt\",\"instId\":\"{symbol}\"}}]}}"
        )),
        _ => None, // binance encodes the stream in the URL
    }
}

fn host_port(url: &str) -> (String, u16) {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let hostport = rest.split('/').next().unwrap_or(rest);
    match hostport.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(443)),
        None => (hostport.to_string(), 443),
    }
}

async fn feed_session(
    cfg: &VenueLatCfg,
    f: &FeedCfg,
    venue: &'static str,
    tx: &tokio::sync::mpsc::Sender<Sample>,
    dropped: &Arc<AtomicU64>,
) -> anyhow::Result<()> {
    let req = f.url.as_str().into_client_request()?;
    let mut ws = match &cfg.socks_proxy {
        Some(p) => {
            let (h, port) = host_port(&f.url);
            let sock = Socks5Stream::connect(p.as_str(), (h.as_str(), port)).await?;
            tokio_tungstenite::client_async_tls(req, sock.into_inner()).await?.0
        }
        None => tokio_tungstenite::connect_async(req).await?.0,
    };
    if let Some(s) = sub_frame(venue, &f.symbol) {
        ws.send(Message::Text(s)).await?;
    }
    tracing::info!(target: "venuelat", "{} {} {} connected", venue, f.market, f.symbol);
    let stale = Duration::from_secs(cfg.stale_timeout_s);
    loop {
        let msg = tokio::time::timeout(stale, ws.next())
            .await
            .map_err(|_| anyhow::anyhow!("stale {venue} {} {}", f.market, f.symbol))?;
        match msg {
            Some(Ok(Message::Text(t))) => {
                // Stamp BEFORE parsing so recv_ts measures arrival, not our
                // own JSON cost.
                let recv = now_ns();
                if let Some(smp) = parse(venue, &f.symbol, &f.market, &t, recv) {
                    // try_send: NEVER await the writer. A stalled disk must not
                    // slow the socket read loop — that would corrupt the very
                    // recv_ts we are trying to measure.
                    if tx.try_send(smp).is_err() {
                        dropped.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            Some(Ok(Message::Ping(p))) => {
                let _ = ws.send(Message::Pong(p)).await;
            }
            Some(Ok(Message::Close(_))) | None => anyhow::bail!("{venue} ws closed"),
            Some(Ok(_)) => {}
            Some(Err(e)) => anyhow::bail!("{venue} ws error: {e}"),
        }
    }
}

/// Start the probe on its OWN OS thread. Returns immediately; the thread is
/// detached and lives for the process. Never call from the trading path.
pub fn spawn(cfg: VenueLatCfg) {
    if !cfg.enabled || cfg.feeds.is_empty() {
        return;
    }
    let _ = std::thread::Builder::new().name("venuelat".into()).spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(target: "venuelat", "runtime: {e}");
                return;
            }
        };
        rt.block_on(async move {
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Sample>(cfg.queue);
            let dropped = Arc::new(AtomicU64::new(0));

            // ---- writer ----
            let path = cfg.out_path.clone();
            let dropped_w = dropped.clone();
            tokio::spawn(async move {
                if let Some(p) = std::path::Path::new(&path).parent() {
                    let _ = std::fs::create_dir_all(p);
                }
                let fresh = !std::path::Path::new(&path).exists();
                let mut fh = match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::warn!(target: "venuelat", "open {path}: {e}");
                        return;
                    }
                };
                if fresh {
                    let _ = writeln!(fh, "recv_ts_ns,exch_ts_ns,venue,market,symbol,bid,ask");
                }
                let mut n: u64 = 0;
                while let Some(s) = rx.recv().await {
                    let _ = writeln!(
                        fh,
                        "{},{},{},{},{},{},{}",
                        s.recv_ts_ns,
                        s.exch_ts_ns.map(|v| v.to_string()).unwrap_or_default(),
                        s.venue,
                        s.market,
                        s.symbol,
                        s.bid,
                        s.ask
                    );
                    n += 1;
                    if n % 20_000 == 0 {
                        let _ = fh.flush();
                        tracing::info!(
                            target: "venuelat",
                            "logged {} rows, dropped {}",
                            n,
                            dropped_w.load(Ordering::Relaxed)
                        );
                    }
                }
            });

            // ---- one reconnecting task per feed ----
            for f in cfg.feeds.clone() {
                let venue: &'static str = match f.venue.as_str() {
                    "binance" => "binance",
                    "bybit" => "bybit",
                    "okx" => "okx",
                    other => {
                        tracing::warn!(target: "venuelat", "unknown venue {other}, skipped");
                        continue;
                    }
                };
                let cfg2 = cfg.clone();
                let tx2 = tx.clone();
                let dropped2 = dropped.clone();
                tokio::spawn(async move {
                    let mut backoff = cfg2.reconnect_base_ms;
                    loop {
                        if let Err(e) = feed_session(&cfg2, &f, venue, &tx2, &dropped2).await {
                            tracing::warn!(
                                target: "venuelat", "{} {} {}: {e}", venue, f.market, f.symbol
                            );
                        }
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        backoff = (backoff * 2).min(cfg2.reconnect_max_ms);
                    }
                });
            }
            drop(tx);
            std::future::pending::<()>().await;
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_binance_book_ticker() {
        let t = r#"{"u":1,"s":"BTCUSDT","b":"63000.1","B":"1","a":"63000.2","A":"2","E":1700000000123,"T":1700000000100}"#;
        let s = parse("binance", "BTCUSDT", "perp", t, 42).unwrap();
        assert_eq!(s.bid, 63000.1);
        assert_eq!(s.ask, 63000.2);
        // prefers T (engine time) over E (push time)
        assert_eq!(s.exch_ts_ns, Some(1_700_000_000_100 * 1_000_000));
        assert_eq!(s.recv_ts_ns, 42);
    }

    #[test]
    fn parses_bybit_orderbook_1() {
        let t = r#"{"topic":"orderbook.1.BTCUSDT","ts":1700000000123,"cts":1700000000100,
                    "data":{"b":[["63000.1","1"]],"a":[["63000.2","2"]]}}"#;
        let s = parse("bybit", "BTCUSDT", "perp", t, 7).unwrap();
        assert_eq!((s.bid, s.ask), (63000.1, 63000.2));
        // prefers cts (engine time) over ts (publish time)
        assert_eq!(s.exch_ts_ns, Some(1_700_000_000_100 * 1_000_000));
    }

    #[test]
    fn parses_okx_bbo_tbt() {
        let t = r#"{"arg":{"channel":"bbo-tbt","instId":"BTC-USDT-SWAP"},
                    "data":[{"bids":[["63000.1","1","0","1"]],"asks":[["63000.2","2","0","1"]],"ts":"1700000000100"}]}"#;
        let s = parse("okx", "BTC-USDT-SWAP", "perp", t, 9).unwrap();
        assert_eq!((s.bid, s.ask), (63000.1, 63000.2));
        assert_eq!(s.exch_ts_ns, Some(1_700_000_000_100 * 1_000_000));
    }

    #[test]
    fn control_frames_are_ignored() {
        assert!(parse("bybit", "BTCUSDT", "perp", r#"{"success":true,"op":"subscribe"}"#, 0).is_none());
        assert!(parse("okx", "BTC-USDT-SWAP", "perp", r#"{"event":"subscribe"}"#, 0).is_none());
        assert!(parse("binance", "BTCUSDT", "perp", r#"{"result":null,"id":1}"#, 0).is_none());
    }

    #[test]
    fn host_port_handles_explicit_port() {
        assert_eq!(host_port("wss://ws.okx.com:8443/ws/v5/public"), ("ws.okx.com".into(), 8443));
        assert_eq!(host_port("wss://stream.bybit.com/v5/public/linear"), ("stream.bybit.com".into(), 443));
    }

    #[test]
    fn disabled_or_empty_never_spawns() {
        // must not panic and must not create a thread
        spawn(VenueLatCfg::default());
        spawn(VenueLatCfg { enabled: true, ..Default::default() });
    }
}
