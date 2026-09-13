//! Kalshi feed for the order manager: private WebSocket channels (`user_orders`,
//! `fill`, `market_positions`) + periodic REST sync (resting orders, positions,
//! fills gap-fetch, unknown-order resolution). Parsers accept both the WS `msg`
//! objects and the REST objects (same field families, `*_fp` fixed-point strings
//! preferred, legacy integer/cent fields as fallback).

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::task::JoinHandle;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use crate::order_manager::{now_ms, BookSide, FillRec, OmsEvent, OrderManager, OrderMsg};
use crate::venue_kalshi::KalshiVenue;

fn num(v: &Value, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(x) = v.get(*k) {
            if let Some(s) = x.as_str() {
                if let Ok(f) = s.trim().parse::<f64>() {
                    return Some(f);
                }
            }
            if let Some(f) = x.as_f64() {
                return Some(f);
            }
        }
    }
    None
}

fn s<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(*k).and_then(|x| x.as_str()))
}

/// YES price in dollars from `yes_price_dollars` (fixed-point string) or legacy `yes_price` (cents).
fn yes_price(v: &Value) -> f64 {
    if let Some(p) = num(v, &["yes_price_dollars"]) {
        return p;
    }
    if let Some(c) = num(v, &["yes_price"]) {
        return if c > 1.0 { c / 100.0 } else { c };
    }
    if let Some(p) = num(v, &["no_price_dollars"]) {
        return 1.0 - p;
    }
    0.0
}

/// Milliseconds from `*_ts_ms`, `ts` (seconds) or an RFC-3339 `*_time` field.
fn ts_ms(v: &Value, ms_keys: &[&str], s_keys: &[&str], iso_keys: &[&str]) -> i64 {
    if let Some(x) = num(v, ms_keys) {
        return x as i64;
    }
    if let Some(x) = num(v, s_keys) {
        return (x * 1000.0) as i64;
    }
    for k in iso_keys {
        if let Some(t) = v.get(*k).and_then(|x| x.as_str()) {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(t) {
                return dt.timestamp_millis();
            }
        }
    }
    0
}

/// WS `user_order` msg or REST order object -> `OrderMsg`.
pub fn parse_order(v: &Value) -> Option<OrderMsg> {
    let order_id = s(v, &["order_id"])?.to_string();
    let side = BookSide::parse(s(v, &["book_side"]), s(v, &["outcome_side", "side"]), s(v, &["action"]))?;
    let initial = num(v, &["initial_count_fp", "initial_count", "count_fp", "count"]).unwrap_or(0.0);
    let filled = num(v, &["fill_count_fp", "fill_count"]).unwrap_or(0.0);
    let remaining = num(v, &["remaining_count_fp", "remaining_count"]).unwrap_or(if initial > 0.0 { initial - filled } else { -1.0 });
    Some(OrderMsg {
        order_id,
        client_id: s(v, &["client_order_id"]).unwrap_or("").to_string(),
        ticker: s(v, &["ticker", "market_ticker"]).unwrap_or("").to_string(),
        side,
        yes_price: yes_price(v),
        initial,
        filled,
        remaining,
        status: s(v, &["status"]).unwrap_or("").to_string(),
        updated_ms: ts_ms(v, &["last_updated_ts_ms", "created_ts_ms"], &["ts"], &["last_update_time", "created_time"]),
    })
}

/// WS `fill` msg or REST fill object -> `FillRec`.
pub fn parse_fill(v: &Value) -> Option<FillRec> {
    let fill_id = s(v, &["fill_id", "trade_id"])?.to_string();
    let side = BookSide::parse(s(v, &["book_side"]), s(v, &["outcome_side", "purchased_side", "side"]), s(v, &["action"]))?;
    Some(FillRec {
        fill_id,
        order_id: s(v, &["order_id"]).unwrap_or("").to_string(),
        ticker: s(v, &["market_ticker", "ticker"]).unwrap_or("").to_string(),
        side,
        count: num(v, &["count_fp", "count"]).unwrap_or(0.0),
        yes_price: yes_price(v),
        is_taker: v.get("is_taker").and_then(|x| x.as_bool()).unwrap_or(false),
        fee: num(v, &["fee_cost"]).unwrap_or(0.0),
        ts_ms: ts_ms(v, &["ts_ms"], &["ts"], &["created_time"]),
    })
}

/// WS `market_position` msg or REST position row -> (ticker, signed YES position).
pub fn parse_position(v: &Value) -> Option<(String, f64)> {
    let t = s(v, &["market_ticker", "ticker"])?.to_string();
    let p = num(v, &["position_fp", "position"])?;
    Some((t, p))
}

/// Spawn the feed: WS listener (if enabled), REST sync loop, health line.
pub fn spawn(om: OrderManager, kv: Arc<KalshiVenue>) -> Vec<JoinHandle<()>> {
    let cfg = om.cfg().clone();
    let mut handles = Vec::new();
    let resync = Arc::new(tokio::sync::Notify::new());
    if cfg.ws {
        let om2 = om.clone();
        let kv2 = kv.clone();
        let rs = resync.clone();
        handles.push(tokio::spawn(async move { ws_task(om2, kv2, rs).await }));
    }
    {
        let om2 = om.clone();
        let kv2 = kv.clone();
        let rs = resync.clone();
        handles.push(tokio::spawn(async move { sync_task(om2, kv2, rs).await }));
    }
    if cfg.health_log_s > 0 {
        let om2 = om.clone();
        let every = cfg.health_log_s;
        handles.push(tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(every));
            loop {
                tick.tick().await;
                let h = om2.health();
                tracing::info!(target: "oms", "health ws_up={} ws_age_ms={} sync_age_ms={} open={} unknown={} positions={}",
                    h.ws_up, h.ws_age_ms.min(999_999), h.sync_age_ms.min(999_999), h.open_orders, h.unknown_orders, h.positions);
            }
        }));
    }
    handles
}

async fn ws_task(om: OrderManager, kv: Arc<KalshiVenue>, resync: Arc<tokio::sync::Notify>) {
    let mut backoff = 1u64;
    loop {
        let req = match kv.ws_request() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(target: "oms", "ws request build: {e}");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };
        match connect_async(req).await {
            Ok((mut ws, _)) => {
                backoff = 1;
                let sub = json!({"id": 1, "cmd": "subscribe", "params": {"channels": ["user_orders", "fill", "market_positions"]}});
                if ws.send(Message::Text(sub.to_string().into())).await.is_err() {
                    tracing::warn!(target: "oms", "ws subscribe send failed");
                    continue;
                }
                tracing::info!(target: "oms", "ws connected, subscribing user_orders/fill/market_positions");
                om.apply(|st| {
                    st.ws_up = true;
                    st.last_ws_ms = now_ms();
                    vec![OmsEvent::WsUp]
                });
                resync.notify_one(); // full REST sync + fills gap-fetch after every (re)connect
                let mut ping = tokio::time::interval(Duration::from_secs(20));
                loop {
                    tokio::select! {
                        _ = ping.tick() => {
                            if ws.send(Message::Ping(Vec::new().into())).await.is_err() { break; }
                        }
                        msg = ws.next() => match msg {
                            Some(Ok(Message::Text(t))) => {
                                if let Ok(v) = serde_json::from_str::<Value>(t.as_str()) {
                                    route(&om, &v);
                                }
                            }
                            Some(Ok(Message::Close(_))) | None => { tracing::info!(target: "oms", "ws closed"); break; }
                            Some(Ok(_)) => {}
                            Some(Err(e)) => { tracing::warn!(target: "oms", "ws error: {e}"); break; }
                        }
                    }
                }
                om.apply(|st| {
                    st.ws_up = false;
                    vec![OmsEvent::WsDown]
                });
            }
            Err(e) => {
                tracing::warn!(target: "oms", "ws connect failed: {e}");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(60);
            }
        }
    }
}

/// Dispatch one WS envelope `{type, sid, msg}`.
pub fn route(om: &OrderManager, v: &Value) {
    let ty = v.get("type").and_then(|x| x.as_str()).unwrap_or("");
    let msg = v.get("msg").unwrap_or(v);
    let t = now_ms();
    match ty {
        "user_order" => {
            if let Some(m) = parse_order(msg) {
                tracing::info!(target: "oms", "order {} {} {:?} px {:.4} filled {:.2}/{:.2} status {}", m.ticker, m.order_id, m.side, m.yes_price, m.filled, m.initial, m.status);
                om.apply(|st| {
                    st.last_ws_ms = t;
                    st.on_order_msg(&m, false)
                });
            }
        }
        "fill" => {
            if let Some(f) = parse_fill(msg) {
                tracing::info!(target: "oms", "fill {} {} {:?} {:.2} @ {:.4} taker={} order {}", f.ticker, f.fill_id, f.side, f.count, f.yes_price, f.is_taker, f.order_id);
                om.apply(|st| {
                    st.last_ws_ms = t;
                    st.on_fill(&f)
                });
            }
        }
        "market_position" => {
            if let Some((tk, p)) = parse_position(msg) {
                om.apply(|st| {
                    st.last_ws_ms = t;
                    st.on_position(&tk, p, t)
                });
            }
        }
        "subscribed" | "ok" => {
            tracing::info!(target: "oms", "ws {ty}: {}", msg);
            om.apply(|st| {
                st.last_ws_ms = t;
                vec![]
            });
        }
        "error" => tracing::warn!(target: "oms", "ws error msg: {}", v),
        _ => {
            om.apply(|st| {
                st.last_ws_ms = t;
                vec![]
            });
        }
    }
}

async fn full_sync(om: &OrderManager, kv: &KalshiVenue) {
    // resting orders (paginated)
    let mut rows: Vec<OrderMsg> = Vec::new();
    let mut cursor = String::new();
    let mut ok = true;
    for _ in 0..20 {
        let q = if cursor.is_empty() { "?status=resting&limit=200".to_string() } else { format!("?status=resting&limit=200&cursor={cursor}") };
        match kv.signed_get("/portfolio/orders", &q).await {
            Ok(v) => {
                if let Some(arr) = v.get("orders").and_then(|x| x.as_array()) {
                    rows.extend(arr.iter().filter_map(parse_order));
                }
                cursor = v.get("cursor").and_then(|x| x.as_str()).unwrap_or("").to_string();
                if cursor.is_empty() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(target: "oms", "sync orders: {e}");
                ok = false;
                break;
            }
        }
    }
    if ok {
        let grace = om.cfg().unknown_grace_ms as i64;
        om.apply(|st| st.on_rest_orders(&rows, now_ms(), grace));
    }
    // positions
    match kv.signed_get("/portfolio/positions", "?limit=200").await {
        Ok(v) => {
            let rows: Vec<(String, f64)> = v.get("market_positions").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(parse_position).collect()).unwrap_or_default();
            om.apply(|st| st.on_rest_positions(&rows, now_ms()));
        }
        Err(e) => tracing::warn!(target: "oms", "sync positions: {e}"),
    }
}

async fn fills_sync(om: &OrderManager, kv: &KalshiVenue) {
    let since_s = om.with(|st| if st.last_fill_ms > 0 { st.last_fill_ms / 1000 - 10 } else { now_ms() / 1000 - 3600 });
    match kv.signed_get("/portfolio/fills", &format!("?min_ts={since_s}&limit=200")).await {
        Ok(v) => {
            if let Some(arr) = v.get("fills").and_then(|x| x.as_array()) {
                let fills: Vec<FillRec> = arr.iter().filter_map(parse_fill).collect();
                om.apply(|st| fills.iter().flat_map(|f| st.on_fill(f)).collect());
            }
        }
        Err(e) => tracing::warn!(target: "oms", "sync fills: {e}"),
    }
}

async fn resolve_unknown(om: &OrderManager, kv: &KalshiVenue) {
    let ids = om.with(|st| st.unknown_ids());
    for id in ids {
        match kv.signed_get(&format!("/portfolio/orders/{id}"), "").await {
            Ok(v) => {
                let obj = v.get("order").unwrap_or(&v);
                if let Some(m) = parse_order(obj) {
                    tracing::info!(target: "oms", "resolved {} -> {} filled {:.2}/{:.2}", id, m.status, m.filled, m.initial);
                    om.apply(|st| st.on_order_msg(&m, false));
                }
            }
            Err(e) => tracing::warn!(target: "oms", "resolve {id}: {e}"),
        }
    }
}

async fn sync_task(om: OrderManager, kv: Arc<KalshiVenue>, resync: Arc<tokio::sync::Notify>) {
    let cfg = om.cfg().clone();
    let mut tick = tokio::time::interval(Duration::from_millis(cfg.sync_ms.max(200)));
    let mut last_fills = 0i64;
    let mut last_gc = 0i64;
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            _ = resync.notified() => { tracing::info!(target: "oms", "resync after ws (re)connect"); last_fills = 0; }
        }
        full_sync(&om, &kv).await;
        resolve_unknown(&om, &kv).await;
        let t = now_ms();
        if t - last_fills >= cfg.fills_sync_ms as i64 {
            fills_sync(&om, &kv).await;
            last_fills = t;
        }
        if t - last_gc >= 60_000 {
            om.apply(|st| {
                st.gc(t, 15 * 60_000);
                vec![]
            });
            last_gc = t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_ws_fill() {
        let v: Value = serde_json::from_str(r#"{"type":"fill","sid":13,"msg":{"trade_id":"d91bc706","order_id":"ee587a1c","client_order_id":"my-order-1",
            "market_ticker":"HIGHNY-22DEC23-B53.5","is_taker":true,"side":"yes","yes_price_dollars":"0.7500","count_fp":"278.00","fee_cost":"0.010000",
            "action":"buy","ts":1671899397,"ts_ms":1671899397000,"post_position_fp":"500.00","purchased_side":"yes","book_side":"bid"}}"#).unwrap();
        let f = parse_fill(v.get("msg").unwrap()).unwrap();
        assert_eq!(f.fill_id, "d91bc706");
        assert_eq!(f.side, BookSide::Bid);
        assert_eq!(f.count, 278.0);
        assert!((f.yes_price - 0.75).abs() < 1e-9);
        assert!(f.is_taker);
        assert_eq!(f.ts_ms, 1671899397000);
    }

    #[test]
    fn parses_documented_ws_order_and_rest_order() {
        let v: Value = serde_json::from_str(r#"{"order_id":"o-1","user_id":"u","ticker":"KXBTC15M-X","status":"resting","side":"no","book_side":"ask",
            "yes_price_dollars":"0.4200","initial_count_fp":"5.00","fill_count_fp":"2.00","remaining_count_fp":"3.00","client_order_id":"c-1",
            "created_ts_ms":1700000000000,"last_updated_ts_ms":1700000001000}"#).unwrap();
        let m = parse_order(&v).unwrap();
        assert_eq!(m.side, BookSide::Ask);
        assert_eq!((m.initial, m.filled, m.remaining), (5.0, 2.0, 3.0));
        assert_eq!(m.updated_ms, 1700000001000);
        assert_eq!(m.client_id, "c-1");
        // REST shape with ISO times and legacy cents
        let r: Value = serde_json::from_str(r#"{"order_id":"o-2","ticker":"T","status":"executed","side":"yes","action":"buy","yes_price":42,
            "initial_count":"1.00","fill_count":"1.00","remaining_count":"0.00","last_update_time":"2026-09-11T00:00:01Z"}"#).unwrap();
        let m = parse_order(&r).unwrap();
        assert_eq!(m.side, BookSide::Bid);
        assert!((m.yes_price - 0.42).abs() < 1e-9);
        assert_eq!(m.remaining, 0.0);
        assert!(m.updated_ms > 1_700_000_000_000);
    }

    #[test]
    fn parses_documented_position() {
        let v: Value = serde_json::from_str(r#"{"type":"market_position","sid":14,"msg":{"user_id":"u","market_ticker":"FED-23DEC-T3.00","position_fp":"-12.50","volume_fp":"40.00"}}"#).unwrap();
        let (t, p) = parse_position(v.get("msg").unwrap()).unwrap();
        assert_eq!(t, "FED-23DEC-T3.00");
        assert_eq!(p, -12.5);
    }
}
