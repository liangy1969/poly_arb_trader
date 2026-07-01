//! `KalshiVenue` — the live Kalshi `TradingVenue` adapter (DESIGN_KALSHI_VENUE).
//!
//! Submits **IOC** (immediate-or-cancel) limit orders to the Kalshi REST trade
//! API, RSA-PSS signed, and returns the synchronous fill — so the trade state
//! machine ([`module`]) drives it exactly like `SimVenue`. v1 uses the **legacy**
//! `/portfolio/orders` request schema, which maps 1:1 to our `.YES`/`.NO` + Buy/Sell
//! model (no bid/ask + complement-price conversion, so a NO trade can't be silently
//! mispriced). Fills come back in the create-order response; precise VWAP/fee
//! reconcile via `GET /portfolio/fills` is P2.
//!
//! Order mapping (legacy schema):
//!   kalshi.<TICKER>.YES + Buy  -> {action:buy,  side:yes, yes_price:<cents>}
//!   kalshi.<TICKER>.YES + Sell -> {action:sell, side:yes, yes_price:<cents>}
//!   kalshi.<TICKER>.NO  + Buy  -> {action:buy,  side:no,  no_price:<cents>}
//!   kalshi.<TICKER>.NO  + Sell -> {action:sell, side:no,  no_price:<cents>}
//! FAK via `time_in_force = "immediate_or_cancel"`. Prices are integer cents
//! (1..=99); count is integer contracts (>=1).
//!
//! NOTE (verify on demo before mainnet): the response field carrying the realized
//! fill cost (`taker_fill_cost`, cents) and the fill-count field name. Until
//! confirmed, VWAP falls back to the order's limit price (conservative for a buy).

use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::SigningKey;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use serde::Deserialize;
use sha2::Sha256;
use tokio::task::JoinHandle;

use arb_core::model::Side;
use arb_core::now_ns;

use crate::types::{Fill, FillStatus, IntentKind, OrderIntent, VenueOutcome};
use crate::venue::{fee_usdc, FillSender, TradingVenue};
use crate::venue_spec::market_id_of;

// Production: external-api.kalshi.com resolves DIRECTLY to the us-east-2 ELB (Kalshi's
// engine region) with no CDN hop — lowest latency from a us-east-2 box, and matches the
// collector. (api.elections.* is CloudFront-fronted: same ~90ms from CA, ~1ms slower
// in-region.) Demo is a SEPARATE account + key on external-api.demo.kalshi.co.
const PROD_BASE: &str = "https://external-api.kalshi.com/trade-api/v2";
const DEMO_BASE: &str = "https://external-api.demo.kalshi.co/trade-api/v2";
/// Path component that is signed (must match the request path exactly).
const ORDERS_PATH: &str = "/trade-api/v2/portfolio/events/orders";

/// RSA-PSS request signer (DESIGN_KALSHI_VENUE §2). Minimal copy of
/// `arb_collector_kalshi::auth::Signer`; TODO factor a shared `kalshi-auth` crate.
struct Signer {
    key_id: String,
    signing: SigningKey<Sha256>,
}

impl Signer {
    fn load(key_id: &str, pem_path: &str) -> Result<Self> {
        let pem = std::fs::read_to_string(pem_path)
            .with_context(|| format!("reading Kalshi private key {pem_path}"))?;
        let key = RsaPrivateKey::from_pkcs8_pem(&pem)
            .or_else(|_| RsaPrivateKey::from_pkcs1_pem(&pem))
            .context("parsing Kalshi RSA private key (PKCS#8/#1 PEM)")?;
        Ok(Signer { key_id: key_id.to_string(), signing: SigningKey::<Sha256>::new(key) })
    }

    /// Sign `{ts_ms}{METHOD}{path}` -> (timestamp, base64 signature).
    fn sign(&self, method: &str, path: &str, ts_ms: i64) -> Result<(String, String)> {
        let ts = ts_ms.to_string();
        let mut rng = rand::rngs::OsRng;
        let sig = self
            .signing
            .try_sign_with_rng(&mut rng, format!("{ts}{method}{path}").as_bytes())
            .context("RSA-PSS signing failed")?;
        Ok((ts, STANDARD.encode(sig.to_bytes())))
    }
}

/// Live Kalshi order adapter.
pub struct KalshiVenue {
    http: reqwest::Client,
    signer: Signer,
    base: String,
    /// Hard per-order USDC cap (validation safety).
    max_order_usdc: f64,
}

impl KalshiVenue {
    /// `network`: "mainnet" -> production, anything else -> demo (safe default).
    pub fn new(key_id: &str, pem_path: &str, network: &str, max_order_usdc: f64) -> Result<Self> {
        if key_id.is_empty() || pem_path.is_empty() {
            anyhow::bail!("kalshi adapter needs venue.key_id + venue.private_key_path");
        }
        let base = if network == "mainnet" { PROD_BASE } else { DEMO_BASE }.to_string();
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .pool_idle_timeout(None)                 // never expire pooled TLS conns
            .tcp_keepalive(Duration::from_secs(15))  // keep the socket alive through NAT/LB
            .build()
            .context("building reqwest client")?;
        tracing::info!("kalshi venue: base={base} max_order_usdc={max_order_usdc}");
        Ok(KalshiVenue { http, signer: Signer::load(key_id, pem_path)?, base, max_order_usdc })
    }
}

/// Kalshi V2 order fields. The V2 schema trades the YES book via bid/ask:
/// bid = buy YES, ask = sell YES (= buy NO at 1-price). A `.NO` trade flips both
/// the side and the price to the YES-book complement.
struct KalshiOrder {
    side: &'static str, // "bid" | "ask"
    cents: i64,         // YES-book price, 1..=99
}

/// Map (instrument suffix, Side, token price 0..1) -> V2 YES-book (side, cents).
///
/// Rounding depends on order intent:
///   - `marketable = true` (IOC taker): round TOWARD the cross (bid ceil / ask
///     floor) so the limit crosses and fills.
///   - `marketable = false` (post-only maker): round AWAY from the cross (bid
///     floor / ask ceil) so the order rests and is NEVER rejected as crossing.
///     Reusing the taker (toward-cross) rounding for a post-only order forces a
///     "post only cross" rejection — the maker-exit bug this guards against.
fn map_order(intent: &OrderIntent, marketable: bool) -> KalshiOrder {
    let is_no = intent.instrument.ends_with(".NO");
    let buy = matches!(intent.side, Side::Buy);
    let yes_bid = (!is_no && buy) || (is_no && !buy); // bid = buy YES / sell NO
    let yes_px = if is_no { 1.0 - intent.price } else { intent.price };
    // Snap sub-cent float error (e.g. 1.0-0.68 = 0.319999.. -> 32.0) BEFORE the
    // directional round, so a clean cent lands exactly on that cent instead of
    // floor'ing a tick too passive (maker) or ceil'ing a tick too aggressive (taker).
    let px100 = (yes_px * 100.0 * 1e6).round() / 1e6;
    // ceil iff (bid AND marketable) OR (ask AND post-only) == (yes_bid == marketable)
    let cents = if yes_bid == marketable { px100.ceil() } else { px100.floor() };
    KalshiOrder {
        side: if yes_bid { "bid" } else { "ask" },
        cents: (cents as i64).clamp(1, 99),
    }
}

/// V2 create-order response (flat; numeric values are fixed-point STRINGS;
/// average_* present only when fill_count > 0).
#[derive(Deserialize)]
struct OrderRespV2 {
    order_id: String,
    #[serde(default)]
    fill_count: String,
    #[serde(default)]
    average_fill_price: String,
    #[serde(default)]
    average_fee_paid: String,
}

/// Pull `order_id` from a create/cancel response, tolerating a `{order:{...}}`
/// wrapper vs a flat body.
fn extract_order_id(v: &serde_json::Value) -> Option<String> {
    let obj = v.get("order").unwrap_or(v);
    obj.get("order_id").and_then(|x| x.as_str()).map(String::from)
}

/// Contracts filled from a create/cancel/get response — tolerant of the wrapper,
/// `fill_count` vs `fill_count_fp`, and string vs numeric fixed-point.
fn extract_fill_count(v: &serde_json::Value) -> f64 {
    let obj = v.get("order").unwrap_or(v);
    for k in ["fill_count", "fill_count_fp"] {
        if let Some(x) = obj.get(k) {
            if let Some(s) = x.as_str() {
                if let Ok(f) = s.parse::<f64>() {
                    return f;
                }
            }
            if let Some(f) = x.as_f64() {
                return f;
            }
        }
    }
    0.0
}

#[async_trait]
impl TradingVenue for KalshiVenue {
    async fn submit(&self, intent: &OrderIntent) -> VenueOutcome {
        if !matches!(intent.kind, IntentKind::TakeNow) {
            return VenueOutcome::Rejected("kalshi adapter v1 supports TakeNow (IOC) only".into());
        }
        let ko = map_order(intent, true); // IOC taker: round toward the cross
        let count = intent.size.floor().max(0.0) as i64; // integer contracts
        if count < 1 {
            return VenueOutcome::Rejected(format!("size {:.2} < 1 contract", intent.size));
        }
        if (ko.cents * count) as f64 / 100.0 > self.max_order_usdc {
            return VenueOutcome::Rejected(format!(
                "notional {:.2} > max_order_usdc {:.2}",
                (ko.cents * count) as f64 / 100.0,
                self.max_order_usdc
            ));
        }
        let Some(ticker) = market_id_of(&intent.instrument) else {
            return VenueOutcome::Rejected(format!("no ticker in {}", intent.instrument));
        };

        // V2 body: bid/ask on the YES book; price + count are fixed-point strings.
        let body = serde_json::json!({
            "ticker": ticker,
            "client_order_id": intent.client_id,
            "side": ko.side,
            "count": format!("{count}.00"),
            "price": format!("{:.2}", ko.cents as f64 / 100.0),
            "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross",
        });

        let ts_ms = now_ns() / 1_000_000;
        let (ts, sig) = match self.signer.sign("POST", ORDERS_PATH, ts_ms) {
            Ok(v) => v,
            Err(e) => return VenueOutcome::Rejected(format!("sign: {e}")),
        };
        let url = format!("{}/portfolio/events/orders", self.base);
        let resp = self
            .http
            .post(&url)
            .header("KALSHI-ACCESS-KEY", self.signer.key_id.as_str())
            .header("KALSHI-ACCESS-TIMESTAMP", ts)
            .header("KALSHI-ACCESS-SIGNATURE", sig)
            .json(&body)
            .send()
            .await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => return VenueOutcome::Rejected(format!("http: {e}")),
        };
        if !resp.status().is_success() {
            let code = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return VenueOutcome::Rejected(format!("kalshi {code}: {txt}"));
        }
        let parsed: OrderRespV2 = match resp.json().await {
            Ok(p) => p,
            Err(e) => return VenueOutcome::Rejected(format!("parse order response: {e}")),
        };
        let filled = parsed.fill_count.parse::<f64>().unwrap_or(0.0);
        if filled <= 0.0 {
            // IOC crossed nothing — acked, nothing resting.
            return VenueOutcome::Acked { order_id: parsed.order_id, fills: Vec::new() };
        }
        // YES-book avg fill price -> the traded token's price (NO is the complement).
        let yes_px = parsed.average_fill_price.parse::<f64>().unwrap_or(ko.cents as f64 / 100.0);
        let token_px = if intent.instrument.ends_with(".NO") { 1.0 - yes_px } else { yes_px };
        let fee = fee_usdc(intent.params.fee_rate, token_px, filled);
        let kalshi_fee: f64 = parsed.average_fee_paid.parse().unwrap_or(f64::NAN);
        tracing::info!(
            target: "executor",
            "kalshi fill: {} {} {:.0}@{:.2}c (yes {:.2}) fee_calc={:.4} kalshi_fee={:.4}",
            ko.side, ticker, filled, token_px * 100.0, yes_px, fee, kalshi_fee
        );
        let fill = Fill {
            venue_trade_id: parsed.order_id.clone(),
            order_id: parsed.order_id.clone(),
            client_id: intent.client_id.clone(),
            instrument: intent.instrument.clone(),
            status: FillStatus::Confirmed,
            side: intent.side,
            qty: filled,
            px: token_px,
            fee,
            ts_ns: now_ns(),
        };
        VenueOutcome::Acked { order_id: parsed.order_id, fills: vec![fill] }
    }

    /// Place a resting GTC **post-only** maker order at `intent.price` (rejected by
    /// Kalshi if it would cross). Returns the venue order_id.
    async fn place_resting(&self, intent: &OrderIntent) -> Result<String, String> {
        let ko = map_order(intent, false); // post-only: round AWAY from the cross so it rests
        let count = intent.size.floor().max(0.0) as i64;
        if count < 1 {
            return Err(format!("size {:.2} < 1 contract", intent.size));
        }
        let Some(ticker) = market_id_of(&intent.instrument) else {
            return Err(format!("no ticker in {}", intent.instrument));
        };
        let body = serde_json::json!({
            "ticker": ticker,
            "client_order_id": intent.client_id,
            "side": ko.side,
            "count": format!("{count}.00"),
            "price": format!("{:.2}", ko.cents as f64 / 100.0),
            "time_in_force": "good_till_canceled",
            "post_only": true,
            "self_trade_prevention_type": "maker",
        });
        let ts_ms = now_ns() / 1_000_000;
        let (ts, sig) = self.signer.sign("POST", ORDERS_PATH, ts_ms).map_err(|e| format!("sign: {e}"))?;
        let url = format!("{}/portfolio/events/orders", self.base);
        let resp = self
            .http
            .post(&url)
            .header("KALSHI-ACCESS-KEY", self.signer.key_id.as_str())
            .header("KALSHI-ACCESS-TIMESTAMP", ts)
            .header("KALSHI-ACCESS-SIGNATURE", sig)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("http: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(format!("kalshi post-only {code}: {}", resp.text().await.unwrap_or_default()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
        extract_order_id(&v).ok_or_else(|| "no order_id in resting response".to_string())
    }

    /// Cancel a resting order; returns contracts filled before cancel took effect.
    async fn cancel_order(&self, order_id: &str) -> Result<f64, String> {
        let path = format!("{ORDERS_PATH}/{order_id}");
        let ts_ms = now_ns() / 1_000_000;
        let (ts, sig) = self.signer.sign("DELETE", &path, ts_ms).map_err(|e| format!("sign: {e}"))?;
        let url = format!("{}/portfolio/events/orders/{order_id}", self.base);
        let resp = self
            .http
            .delete(&url)
            .header("KALSHI-ACCESS-KEY", self.signer.key_id.as_str())
            .header("KALSHI-ACCESS-TIMESTAMP", ts)
            .header("KALSHI-ACCESS-SIGNATURE", sig)
            .send()
            .await
            .map_err(|e| format!("http: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(format!("kalshi cancel {code}: {}", resp.text().await.unwrap_or_default()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
        Ok(extract_fill_count(&v))
    }

    /// Poll contracts filled so far on a resting order.
    async fn order_fill_count(&self, order_id: &str) -> Result<f64, String> {
        let path = format!("/trade-api/v2/portfolio/orders/{order_id}");
        let ts_ms = now_ns() / 1_000_000;
        let (ts, sig) = self.signer.sign("GET", &path, ts_ms).map_err(|e| format!("sign: {e}"))?;
        let url = format!("{}/portfolio/orders/{order_id}", self.base);
        let resp = self
            .http
            .get(&url)
            .header("KALSHI-ACCESS-KEY", self.signer.key_id.as_str())
            .header("KALSHI-ACCESS-TIMESTAMP", ts)
            .header("KALSHI-ACCESS-SIGNATURE", sig)
            .send()
            .await
            .map_err(|e| format!("http: {e}"))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(format!("kalshi get-order {code}: {}", resp.text().await.unwrap_or_default()));
        }
        let v: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
        Ok(extract_fill_count(&v))
    }

    fn start(&self, fills: FillSender) -> Vec<JoinHandle<()>> {
        // Keep the fill sender alive — if it drops, the executor's fill_rx closes
        // and its select! loop breaks on the first poll (no signals ever processed).
        let keeper = tokio::spawn(async move {
            let _hold = fills;
            std::future::pending::<()>().await;
        });
        // Connection warmer: ping a cheap public endpoint every 15s on the SAME
        // client so the pooled TLS connection to Kalshi stays hot. A real order then
        // reuses it and skips the cold-handshake penalty (~tens of ms). Same host as
        // the order POST, so HTTP keep-alive shares the one connection. Critical here
        // because 3bps signals are minutes apart — without this every order is cold.
        let http = self.http.clone();
        let url = format!("{}/exchange/status", self.base);
        let warmer = tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(15));
            loop {
                tick.tick().await;
                let _ = http.get(&url).send().await; // result ignored; just keep it warm
            }
        });
        vec![keeper, warmer]
    }

    fn name(&self) -> &'static str {
        "kalshi"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MarketParams;

    fn intent(inst: &str, side: Side, price: f64, size: f64) -> OrderIntent {
        OrderIntent {
            client_id: "t-1:E:0".into(),
            instrument: inst.into(),
            token_id: String::new(),
            side,
            price,
            size,
            kind: IntentKind::TakeNow,
            params: MarketParams { min_order_size: 1.0, tick_size: 0.01, fee_rate: 0.07 },
            expiry_ns: i64::MAX,
        }
    }

    #[test]
    fn yes_buy_is_bid_ceil() {
        let k = map_order(&intent("kalshi.KXBTC15M-1.YES", Side::Buy, 0.621, 20.0), true);
        assert_eq!(k.side, "bid");
        assert_eq!(k.cents, 63, "buy YES rounds up to cross");
    }

    #[test]
    fn no_buy_is_ask_complement() {
        // buy NO at 0.40 == sell YES at 0.60 -> ask, floor(60)=60
        let k = map_order(&intent("kalshi.KXBTC15M-1.NO", Side::Buy, 0.40, 20.0), true);
        assert_eq!(k.side, "ask");
        assert_eq!(k.cents, 60);
    }

    #[test]
    fn yes_sell_is_ask_floor() {
        let k = map_order(&intent("kalshi.KXBTC15M-1.YES", Side::Sell, 0.589, 20.0), true);
        assert_eq!(k.side, "ask");
        assert_eq!(k.cents, 58, "sell YES rounds down");
    }

    #[test]
    fn no_sell_is_bid_complement() {
        // sell NO at 0.40 == buy YES at 0.60 -> bid, ceil(60)=60
        let k = map_order(&intent("kalshi.KXBTC15M-1.NO", Side::Sell, 0.40, 20.0), true);
        assert_eq!(k.side, "bid");
        assert_eq!(k.cents, 60);
    }

    #[test]
    fn clamps_into_1_99() {
        assert_eq!(map_order(&intent("kalshi.X.YES", Side::Buy, 0.005, 5.0), true).cents, 1);
        assert_eq!(map_order(&intent("kalshi.X.YES", Side::Sell, 0.999, 5.0), true).cents, 99);
    }

    // ── post-only (maker) rounds AWAY from the cross so it never crosses ──

    #[test]
    fn post_only_no_sell_is_bid_floor() {
        // sell NO at 0.685 (maker exit) == buy YES at 0.315 -> bid.
        // taker would ceil(31.5)=32 and could cross; post-only floors to 31 (passive).
        let k = map_order(&intent("kalshi.KXBTC15M-1.NO", Side::Sell, 0.685, 20.0), false);
        assert_eq!(k.side, "bid");
        assert_eq!(k.cents, 31, "post-only bid floors — never crosses");
        // the taker mapping of the same intent WOULD round up into a cross
        assert_eq!(map_order(&intent("kalshi.KXBTC15M-1.NO", Side::Sell, 0.685, 20.0), true).cents, 32);
    }

    #[test]
    fn post_only_yes_sell_is_ask_ceil() {
        // sell YES at 0.582 -> ask. taker floors to 58 (crosses down); post-only ceils to 59.
        let k = map_order(&intent("kalshi.KXBTC15M-1.YES", Side::Sell, 0.582, 20.0), false);
        assert_eq!(k.side, "ask");
        assert_eq!(k.cents, 59, "post-only ask ceils — never crosses");
    }

    #[test]
    fn post_only_integer_cent_is_at_touch() {
        // clean cents: post-only lands exactly at the touch (joins the queue, rests).
        let k = map_order(&intent("kalshi.KXBTC15M-1.NO", Side::Sell, 0.68, 20.0), false);
        assert_eq!(k.side, "bid");
        assert_eq!(k.cents, 32, "1-0.68=0.32 -> 32c exactly, at the passive touch");
    }
}
