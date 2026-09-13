//! Order manager (OMS) — the one owner of "what are OUR orders and positions right
//! now" (DESIGN_ORDER_MANAGER.md). Three inputs, one truth:
//!
//!   1. the Kalshi private WebSocket channels `user_orders` (order created /
//!      filled / canceled / executed, with cumulative fill + remaining counts),
//!      `fill` (one message per match: fill id, order id, count, price, taker flag)
//!      and `market_positions` (net position after each change) — milliseconds;
//!   2. the write path itself: `place` registers the intent BEFORE the REST call so
//!      a WS ack that beats the HTTP response is linked by `client_order_id`, and
//!      `cancel` books the engine-synchronous `reduced_by` (exact fill at removal);
//!   3. periodic REST sync: `GET /portfolio/orders?status=resting`,
//!      `GET /portfolio/positions` (the AUTHORITATIVE position — the poll always
//!      wins over the estimate) and `GET /portfolio/fills?min_ts` for gap recovery
//!      after a WS reconnect; a managed order that the sync no longer sees resting
//!      becomes `Unknown` and is resolved by `GET /portfolio/orders/{id}`.
//!
//! Invariants (the 2026-07-02 oversell rules, encoded):
//!   - cumulative fill counts never decrease; fills are deduped by fill id;
//!   - a side with an order in `Pending`/`Canceling`/`Unknown` is LOCKED: the
//!     caller must not post a new order on it until the state resolves;
//!   - `position().auth` is the venue's number; `est` only adds fills seen since
//!     that poll, and is overwritten by every poll.
//!
//! The state machine (`OmsState`) is pure and unit-tested; the Kalshi feed lives in
//! `order_manager_kalshi.rs`. Off by default (`executor.order_manager.enabled`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use arb_core::now_ns;

use crate::types::OrderIntent;
use crate::venue::{CancelOutcome, TradingVenue};

const MS: i64 = 1_000_000;

/// YES-book side. `Bid` = buy YES or sell NO (position +count on fill);
/// `Ask` = sell YES or buy NO (position −count on fill).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BookSide {
    Bid,
    Ask,
}

impl BookSide {
    /// Signed position change for `count` contracts filled on this side.
    pub fn signed(self, count: f64) -> f64 {
        match self {
            BookSide::Bid => count,
            BookSide::Ask => -count,
        }
    }
    /// From Kalshi's `book_side` ("bid"/"ask"), else from (`side`, `action`).
    pub fn parse(book_side: Option<&str>, side: Option<&str>, action: Option<&str>) -> Option<BookSide> {
        match book_side {
            Some("bid") => return Some(BookSide::Bid),
            Some("ask") => return Some(BookSide::Ask),
            _ => {}
        }
        match (side?, action?) {
            ("yes", "buy") | ("no", "sell") => Some(BookSide::Bid),
            ("yes", "sell") | ("no", "buy") => Some(BookSide::Ask),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    /// Sent, no venue ack yet (no `order_id`).
    Pending,
    Resting,
    /// Cancel sent, outcome not yet known.
    Canceling,
    /// Fully filled.
    Executed,
    Canceled,
    /// State could not be established (cancel transport error, 404 on cancel,
    /// vanished from the resting list): terminal-or-not unknown until resolved.
    Unknown,
}

impl OrderStatus {
    pub fn is_open(self) -> bool {
        matches!(self, OrderStatus::Pending | OrderStatus::Resting | OrderStatus::Canceling | OrderStatus::Unknown)
    }
    /// The side is locked while an order's state is not settled.
    pub fn locks_side(self) -> bool {
        matches!(self, OrderStatus::Pending | OrderStatus::Canceling | OrderStatus::Unknown)
    }
}

#[derive(Clone, Debug)]
pub struct OrderRec {
    pub order_id: String,
    pub client_id: String,
    pub ticker: String,
    pub side: BookSide,
    /// YES-book price, dollars (0..1).
    pub yes_price: f64,
    pub initial: f64,
    pub filled: f64,
    pub remaining: f64,
    pub status: OrderStatus,
    pub created_ms: i64,
    pub updated_ms: i64,
    /// Placed through this manager (vs discovered from the feed: the executor's
    /// own IOC entries, manual orders). Only managed orders lock a side.
    pub managed: bool,
    /// Sum of deduped fill messages seen for this order (cross-check for `filled`).
    pub fills_sum: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PositionRec {
    pub ticker: String,
    /// Venue's own number (REST positions poll or `market_positions` push).
    pub auth: f64,
    pub auth_ts_ms: i64,
    /// `auth` + signed fills seen since `auth_ts_ms`.
    pub est: f64,
    pub est_ts_ms: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PositionView {
    pub auth: f64,
    pub auth_age_ms: i64,
    pub est: f64,
}

#[derive(Clone, Debug)]
pub struct FillRec {
    pub fill_id: String,
    pub order_id: String,
    pub ticker: String,
    pub side: BookSide,
    pub count: f64,
    pub yes_price: f64,
    pub is_taker: bool,
    pub fee: f64,
    pub ts_ms: i64,
}

/// One order-state observation (WS `user_order`, REST order object, or a cancel /
/// place outcome mapped into the same shape).
#[derive(Clone, Debug)]
pub struct OrderMsg {
    pub order_id: String,
    pub client_id: String,
    pub ticker: String,
    pub side: BookSide,
    pub yes_price: f64,
    pub initial: f64,
    pub filled: f64,
    pub remaining: f64,
    /// "resting" | "canceled" | "executed" (anything else: keep the current status).
    pub status: String,
    pub updated_ms: i64,
}

#[derive(Clone, Debug)]
pub enum OmsEvent {
    OrderUpdate(OrderRec),
    Fill(FillRec),
    Position { ticker: String, auth: f64, ts_ms: i64 },
    Unknown { order_id: String },
    Resolved(OrderRec),
    WsUp,
    WsDown,
    Synced { resting: usize, positions: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct OmsHealth {
    pub ws_up: bool,
    pub ws_age_ms: i64,
    pub sync_age_ms: i64,
    pub open_orders: usize,
    pub unknown_orders: usize,
    pub positions: usize,
}

const SEEN_FILLS_CAP: usize = 20_000;

/// Pure order/position state. Every `on_*` applies one observation and returns
/// the events it produced; no I/O, so the whole machine is unit-testable.
#[derive(Default)]
pub struct OmsState {
    pub orders: HashMap<String, OrderRec>,
    by_client: HashMap<String, String>,
    pub positions: HashMap<String, PositionRec>,
    seen_fills: HashSet<String>,
    seen_q: VecDeque<String>,
    pub ws_up: bool,
    pub last_ws_ms: i64,
    pub last_sync_ms: i64,
    pub last_fill_ms: i64,
    /// Timestamps of write actions (rate budget).
    actions: VecDeque<i64>,
}

fn map_status(s: &str, current: OrderStatus) -> OrderStatus {
    match s {
        // a resting report while a cancel is in flight does not clear the lock
        "resting" if current == OrderStatus::Canceling => OrderStatus::Canceling,
        "resting" => OrderStatus::Resting,
        "canceled" | "cancelled" => OrderStatus::Canceled,
        "executed" => OrderStatus::Executed,
        _ => current,
    }
}

impl OmsState {
    fn key_for(&self, order_id: &str, client_id: &str) -> Option<String> {
        if !order_id.is_empty() && self.orders.contains_key(order_id) {
            return Some(order_id.to_string());
        }
        if !client_id.is_empty() {
            if let Some(k) = self.by_client.get(client_id) {
                return Some(k.clone());
            }
        }
        None
    }

    fn rekey(&mut self, old: &str, new: &str) {
        if old == new || new.is_empty() {
            return;
        }
        if let Some(mut r) = self.orders.remove(old) {
            r.order_id = new.to_string();
            if !r.client_id.is_empty() {
                self.by_client.insert(r.client_id.clone(), new.to_string());
            }
            self.orders.insert(new.to_string(), r);
        }
    }

    /// Register an order we are about to send (before the REST call).
    pub fn register_pending(&mut self, client_id: &str, ticker: &str, side: BookSide, yes_price: f64, size: f64, now_ms: i64) {
        let key = format!("pending:{client_id}");
        let rec = OrderRec {
            order_id: String::new(),
            client_id: client_id.to_string(),
            ticker: ticker.to_string(),
            side,
            yes_price,
            initial: size,
            filled: 0.0,
            remaining: size,
            status: OrderStatus::Pending,
            created_ms: now_ms,
            updated_ms: now_ms,
            managed: true,
            fills_sum: 0.0,
        };
        self.by_client.insert(client_id.to_string(), key.clone());
        self.orders.insert(key, rec);
    }

    /// The venue acked our placement with `order_id`.
    pub fn on_place_ack(&mut self, client_id: &str, order_id: &str, now_ms: i64) -> Vec<OmsEvent> {
        let Some(key) = self.key_for("", client_id) else { return vec![] };
        self.rekey(&key, order_id);
        let r = self.orders.get_mut(order_id).expect("rekeyed");
        if r.status == OrderStatus::Pending {
            r.status = OrderStatus::Resting;
        }
        r.updated_ms = r.updated_ms.max(now_ms);
        vec![OmsEvent::OrderUpdate(r.clone())]
    }

    /// The placement failed. `definite` = the venue rejected it (4xx, would cross):
    /// the order does not exist. Otherwise (transport/timeout) it MAY exist ->
    /// `Unknown` until the sync sees or clears it.
    pub fn on_place_reject(&mut self, client_id: &str, definite: bool, now_ms: i64) -> Vec<OmsEvent> {
        let Some(key) = self.key_for("", client_id) else { return vec![] };
        if definite {
            self.orders.remove(&key);
            self.by_client.remove(client_id);
            return vec![];
        }
        let r = self.orders.get_mut(&key).expect("key");
        r.status = OrderStatus::Unknown;
        r.updated_ms = now_ms;
        vec![OmsEvent::Unknown { order_id: key }]
    }

    pub fn on_cancel_sent(&mut self, order_id: &str, now_ms: i64) {
        if let Some(r) = self.orders.get_mut(order_id) {
            if r.status.is_open() {
                r.status = OrderStatus::Canceling;
                r.updated_ms = now_ms;
            }
        }
    }

    /// Cancel outcome: `Canceled{reduced_by}` is engine-synchronous truth
    /// (`initial - reduced_by` filled); `Gone` = terminal but the fill count is
    /// not known here -> `Unknown` until resolved; `None` = transport error ->
    /// `Unknown` (never assume it is gone).
    pub fn on_cancel_outcome(&mut self, order_id: &str, outcome: Option<CancelOutcome>, now_ms: i64) -> Vec<OmsEvent> {
        let Some(r) = self.orders.get_mut(order_id) else { return vec![] };
        r.updated_ms = now_ms;
        match outcome {
            Some(CancelOutcome::Canceled { reduced_by }) => {
                let filled = (r.initial - reduced_by).max(r.filled).max(0.0);
                r.filled = filled;
                r.remaining = 0.0;
                r.status = if filled + 1e-9 >= r.initial { OrderStatus::Executed } else { OrderStatus::Canceled };
                vec![OmsEvent::OrderUpdate(r.clone())]
            }
            Some(CancelOutcome::Gone) | None => {
                r.status = OrderStatus::Unknown;
                vec![OmsEvent::Unknown { order_id: order_id.to_string() }]
            }
        }
    }

    /// Apply an order observation (WS `user_order`, REST order object, resolver).
    pub fn on_order_msg(&mut self, m: &OrderMsg, managed_hint: bool) -> Vec<OmsEvent> {
        let key = self.key_for(&m.order_id, &m.client_id);
        let key = match key {
            Some(k) => {
                if k != m.order_id && !m.order_id.is_empty() {
                    self.rekey(&k, &m.order_id);
                    m.order_id.clone()
                } else {
                    k
                }
            }
            None => {
                if m.order_id.is_empty() {
                    return vec![];
                }
                let rec = OrderRec {
                    order_id: m.order_id.clone(),
                    client_id: m.client_id.clone(),
                    ticker: m.ticker.clone(),
                    side: m.side,
                    yes_price: m.yes_price,
                    initial: m.initial,
                    filled: 0.0,
                    remaining: m.remaining,
                    status: OrderStatus::Resting,
                    created_ms: m.updated_ms,
                    updated_ms: 0,
                    managed: managed_hint,
                    fills_sum: 0.0,
                };
                if !m.client_id.is_empty() {
                    self.by_client.insert(m.client_id.clone(), m.order_id.clone());
                }
                self.orders.insert(m.order_id.clone(), rec);
                m.order_id.clone()
            }
        };
        let r = self.orders.get_mut(&key).expect("key");
        if m.updated_ms < r.updated_ms {
            return vec![]; // stale observation
        }
        if m.initial > 0.0 {
            r.initial = m.initial;
        }
        r.filled = r.filled.max(m.filled).max(r.fills_sum);
        r.remaining = if m.remaining >= 0.0 { m.remaining.min((r.initial - r.filled).max(0.0)) } else { (r.initial - r.filled).max(0.0) };
        if m.yes_price > 0.0 {
            r.yes_price = m.yes_price;
        }
        let was_unknown = r.status == OrderStatus::Unknown;
        r.status = map_status(&m.status, r.status);
        if r.status == OrderStatus::Resting && r.remaining <= 1e-9 && r.initial > 0.0 {
            r.status = OrderStatus::Executed;
        }
        r.updated_ms = m.updated_ms;
        let rec = r.clone();
        let mut ev = vec![OmsEvent::OrderUpdate(rec.clone())];
        if was_unknown && !rec.status.locks_side() {
            ev.push(OmsEvent::Resolved(rec));
        }
        ev
    }

    /// Apply one fill (WS `fill` or REST fills). Deduped by fill id.
    pub fn on_fill(&mut self, f: &FillRec) -> Vec<OmsEvent> {
        if f.fill_id.is_empty() || self.seen_fills.contains(&f.fill_id) {
            return vec![];
        }
        self.seen_fills.insert(f.fill_id.clone());
        self.seen_q.push_back(f.fill_id.clone());
        while self.seen_q.len() > SEEN_FILLS_CAP {
            if let Some(old) = self.seen_q.pop_front() {
                self.seen_fills.remove(&old);
            }
        }
        self.last_fill_ms = self.last_fill_ms.max(f.ts_ms);
        let mut ev = vec![OmsEvent::Fill(f.clone())];
        // order side
        if !f.order_id.is_empty() {
            if !self.orders.contains_key(&f.order_id) {
                // a fill on an order we never registered (executor IOC entry, manual):
                // track it unmanaged so its count is visible; initial unknown (0).
                self.orders.insert(
                    f.order_id.clone(),
                    OrderRec {
                        order_id: f.order_id.clone(),
                        client_id: String::new(),
                        ticker: f.ticker.clone(),
                        side: f.side,
                        yes_price: f.yes_price,
                        initial: 0.0,
                        filled: 0.0,
                        remaining: -1.0,
                        status: OrderStatus::Unknown,
                        created_ms: f.ts_ms,
                        updated_ms: 0,
                        managed: false,
                        fills_sum: 0.0,
                    },
                );
            }
            let r = self.orders.get_mut(&f.order_id).expect("order");
            r.fills_sum += f.count;
            r.filled = r.filled.max(r.fills_sum);
            if r.initial > 0.0 {
                r.remaining = (r.initial - r.filled).max(0.0);
                if r.remaining <= 1e-9 && r.status != OrderStatus::Canceled {
                    r.status = OrderStatus::Executed;
                }
            }
            r.updated_ms = r.updated_ms.max(f.ts_ms);
            ev.push(OmsEvent::OrderUpdate(r.clone()));
        }
        // position estimate
        let p = self.positions.entry(f.ticker.clone()).or_insert_with(|| PositionRec { ticker: f.ticker.clone(), ..Default::default() });
        p.est += f.side.signed(f.count);
        p.est_ts_ms = p.est_ts_ms.max(f.ts_ms);
        ev
    }

    /// Authoritative position for one market (REST poll row or `market_positions` push).
    pub fn on_position(&mut self, ticker: &str, position: f64, ts_ms: i64) -> Vec<OmsEvent> {
        let p = self.positions.entry(ticker.to_string()).or_insert_with(|| PositionRec { ticker: ticker.to_string(), ..Default::default() });
        if ts_ms < p.auth_ts_ms {
            return vec![];
        }
        p.auth = position;
        p.auth_ts_ms = ts_ms;
        p.est = position; // the poll wins; fills after this timestamp re-add themselves
        p.est_ts_ms = ts_ms;
        vec![OmsEvent::Position { ticker: ticker.to_string(), auth: position, ts_ms }]
    }

    /// Full REST sync of resting orders: upsert every row, then any MANAGED open
    /// order (with an id) that the venue no longer lists as resting and that has
    /// been quiet for `grace_ms` becomes `Unknown` (to be resolved by id).
    pub fn on_rest_orders(&mut self, rows: &[OrderMsg], now_ms: i64, grace_ms: i64) -> Vec<OmsEvent> {
        let mut ev = Vec::new();
        let mut listed: HashSet<String> = HashSet::new();
        for m in rows {
            listed.insert(m.order_id.clone());
            ev.extend(self.on_order_msg(m, false));
        }
        let mut to_unknown = Vec::new();
        for (k, r) in &self.orders {
            if r.managed && matches!(r.status, OrderStatus::Resting | OrderStatus::Canceling) && !listed.contains(k) && now_ms - r.updated_ms > grace_ms {
                to_unknown.push(k.clone());
            }
            if r.managed && r.status == OrderStatus::Pending && now_ms - r.created_ms > grace_ms.max(5_000) {
                to_unknown.push(k.clone());
            }
        }
        for k in to_unknown {
            if let Some(r) = self.orders.get_mut(&k) {
                r.status = OrderStatus::Unknown;
                r.updated_ms = now_ms;
                ev.push(OmsEvent::Unknown { order_id: k });
            }
        }
        self.last_sync_ms = now_ms;
        ev
    }

    /// Full REST sync of positions: rows are authoritative; known tickers absent
    /// from the response are flat.
    pub fn on_rest_positions(&mut self, rows: &[(String, f64)], now_ms: i64) -> Vec<OmsEvent> {
        let mut ev = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for (t, p) in rows {
            seen.insert(t.as_str());
            ev.extend(self.on_position(t, *p, now_ms));
        }
        let absent: Vec<String> = self.positions.keys().filter(|t| !seen.contains(t.as_str())).cloned().collect();
        for t in absent {
            ev.extend(self.on_position(&t, 0.0, now_ms));
        }
        ev.push(OmsEvent::Synced { resting: self.resting_count(), positions: rows.len() });
        ev
    }

    /// Drop terminal orders older than `keep_ms` (memory hygiene).
    pub fn gc(&mut self, now_ms: i64, keep_ms: i64) {
        let dead: Vec<String> = self
            .orders
            .iter()
            .filter(|(_, r)| !r.status.is_open() && now_ms - r.updated_ms > keep_ms)
            .map(|(k, _)| k.clone())
            .collect();
        for k in dead {
            if let Some(r) = self.orders.remove(&k) {
                self.by_client.remove(&r.client_id);
            }
        }
    }

    pub fn resting(&self, ticker: &str) -> Vec<OrderRec> {
        self.orders.values().filter(|r| r.ticker == ticker && r.managed && r.status == OrderStatus::Resting).cloned().collect()
    }

    pub fn resting_count(&self) -> usize {
        self.orders.values().filter(|r| r.managed && r.status == OrderStatus::Resting).count()
    }

    pub fn unknown_ids(&self) -> Vec<String> {
        self.orders.iter().filter(|(_, r)| r.status == OrderStatus::Unknown && !r.order_id.is_empty()).map(|(k, _)| k.clone()).collect()
    }

    pub fn side_locked(&self, ticker: &str, side: BookSide) -> bool {
        self.orders.values().any(|r| r.managed && r.ticker == ticker && r.side == side && r.status.locks_side())
    }

    pub fn position(&self, ticker: &str, now_ms: i64) -> PositionView {
        match self.positions.get(ticker) {
            Some(p) => PositionView { auth: p.auth, auth_age_ms: if p.auth_ts_ms > 0 { now_ms - p.auth_ts_ms } else { i64::MAX }, est: p.est },
            None => PositionView { auth: 0.0, auth_age_ms: i64::MAX, est: 0.0 },
        }
    }

    /// Token-bucket-ish action budget: true if fewer than `max_per_s` actions in the last second.
    pub fn action_ok(&mut self, now_ms: i64, max_per_s: f64) -> bool {
        while let Some(&t) = self.actions.front() {
            if now_ms - t > 1000 {
                self.actions.pop_front();
            } else {
                break;
            }
        }
        if (self.actions.len() as f64) < max_per_s {
            self.actions.push_back(now_ms);
            true
        } else {
            false
        }
    }

    pub fn health(&self, now_ms: i64) -> OmsHealth {
        OmsHealth {
            ws_up: self.ws_up,
            ws_age_ms: if self.last_ws_ms > 0 { now_ms - self.last_ws_ms } else { i64::MAX },
            sync_age_ms: if self.last_sync_ms > 0 { now_ms - self.last_sync_ms } else { i64::MAX },
            open_orders: self.orders.values().filter(|r| r.managed && r.status.is_open()).count(),
            unknown_orders: self.orders.values().filter(|r| r.status == OrderStatus::Unknown).count(),
            positions: self.positions.values().filter(|p| p.auth.abs() > 1e-9 || p.est.abs() > 1e-9).count(),
        }
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default)]
pub struct OrderManagerCfg {
    /// Off by default: nothing is spawned.
    pub enabled: bool,
    /// Subscribe to the private WS channels (user_orders, fill, market_positions).
    pub ws: bool,
    /// REST sync cadence for resting orders + positions.
    pub sync_ms: u64,
    /// REST fills gap-fetch cadence.
    pub fills_sync_ms: u64,
    /// A managed order missing from the resting list is `Unknown` only after this quiet period.
    pub unknown_grace_ms: u64,
    /// Write-action budget (place + cancel) per second, manager-wide.
    pub max_actions_per_s: f64,
    /// Health line cadence (0 = off).
    pub health_log_s: u64,
}

impl Default for OrderManagerCfg {
    fn default() -> Self {
        OrderManagerCfg { enabled: false, ws: true, sync_ms: 1000, fills_sync_ms: 5000, unknown_grace_ms: 2000, max_actions_per_s: 4.0, health_log_s: 30 }
    }
}

struct Inner {
    st: Mutex<OmsState>,
    tx: broadcast::Sender<OmsEvent>,
    venue: Arc<dyn TradingVenue>,
    cfg: OrderManagerCfg,
}

/// Shared handle: the executor / maker rule reads state and writes orders through it.
#[derive(Clone)]
pub struct OrderManager {
    inner: Arc<Inner>,
}

pub fn now_ms() -> i64 {
    now_ns() / MS
}

impl OrderManager {
    pub fn new(cfg: OrderManagerCfg, venue: Arc<dyn TradingVenue>) -> Self {
        let (tx, _) = broadcast::channel(2048);
        OrderManager { inner: Arc::new(Inner { st: Mutex::new(OmsState::default()), tx, venue, cfg }) }
    }

    pub fn cfg(&self) -> &OrderManagerCfg {
        &self.inner.cfg
    }

    pub fn subscribe(&self) -> broadcast::Receiver<OmsEvent> {
        self.inner.tx.subscribe()
    }

    /// Run `f` against the state and broadcast whatever events it produced.
    pub fn apply<F: FnOnce(&mut OmsState) -> Vec<OmsEvent>>(&self, f: F) {
        let ev = {
            let mut st = self.inner.st.lock().unwrap();
            f(&mut st)
        };
        for e in ev {
            let _ = self.inner.tx.send(e);
        }
    }

    pub fn with<R, F: FnOnce(&OmsState) -> R>(&self, f: F) -> R {
        let st = self.inner.st.lock().unwrap();
        f(&st)
    }

    // ── reads ──
    pub fn resting(&self, ticker: &str) -> Vec<OrderRec> {
        self.with(|s| s.resting(ticker))
    }
    pub fn order(&self, order_id: &str) -> Option<OrderRec> {
        self.with(|s| s.orders.get(order_id).cloned())
    }
    pub fn position(&self, ticker: &str) -> PositionView {
        self.with(|s| s.position(ticker, now_ms()))
    }
    pub fn side_locked(&self, ticker: &str, side: BookSide) -> bool {
        self.with(|s| s.side_locked(ticker, side))
    }
    pub fn health(&self) -> OmsHealth {
        self.with(|s| s.health(now_ms()))
    }

    // ── writes ──
    /// Place a resting post-only order. The intent is registered BEFORE the call
    /// so a WS ack that arrives first is linked by client id. Errors: `Busy` when
    /// the side is locked or the action budget is spent; venue rejects (definite)
    /// remove the pending record; transport errors leave it `Unknown`.
    pub async fn place(&self, intent: &OrderIntent, ticker: &str, side: BookSide) -> Result<String, PlaceError> {
        let t0 = now_ms();
        {
            let mut st = self.inner.st.lock().unwrap();
            if st.side_locked(ticker, side) {
                return Err(PlaceError::SideLocked);
            }
            if !st.action_ok(t0, self.inner.cfg.max_actions_per_s) {
                return Err(PlaceError::Budget);
            }
            st.register_pending(&intent.client_id, ticker, side, intent.price, intent.size, t0);
        }
        match self.inner.venue.place_resting(intent).await {
            Ok(order_id) => {
                self.apply(|s| s.on_place_ack(&intent.client_id, &order_id, now_ms()));
                Ok(order_id)
            }
            Err(e) => {
                // A venue-level reject carries an HTTP status in the message ("kalshi post-only 4xx");
                // anything else is transport: the order may exist.
                let definite = e.contains("post-only 4") || e.contains("< 0.01") || e.contains("no ticker");
                self.apply(|s| s.on_place_reject(&intent.client_id, definite, now_ms()));
                Err(if definite { PlaceError::Rejected(e) } else { PlaceError::Unknown(e) })
            }
        }
    }

    /// Cancel a resting order; books the outcome (see `OmsState::on_cancel_outcome`).
    pub async fn cancel(&self, order_id: &str) -> Result<CancelOutcome, String> {
        let t0 = now_ms();
        {
            let mut st = self.inner.st.lock().unwrap();
            if !st.action_ok(t0, self.inner.cfg.max_actions_per_s) {
                return Err("action budget".into());
            }
            st.on_cancel_sent(order_id, t0);
        }
        let out = self.inner.venue.cancel_order(order_id).await;
        let oc = out.as_ref().ok().copied();
        self.apply(|s| s.on_cancel_outcome(order_id, oc, now_ms()));
        out
    }

    /// Cancel every managed resting order in a market (best effort, sequential).
    pub async fn cancel_all(&self, ticker: &str) -> usize {
        let ids: Vec<String> = self.resting(ticker).into_iter().map(|r| r.order_id).collect();
        let mut n = 0;
        for id in ids {
            if self.cancel(&id).await.is_ok() {
                n += 1;
            }
        }
        n
    }
}

#[derive(Debug)]
pub enum PlaceError {
    SideLocked,
    Budget,
    Rejected(String),
    Unknown(String),
}

impl std::fmt::Display for PlaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaceError::SideLocked => write!(f, "side locked (order in pending/canceling/unknown)"),
            PlaceError::Budget => write!(f, "action budget exhausted"),
            PlaceError::Rejected(e) => write!(f, "rejected: {e}"),
            PlaceError::Unknown(e) => write!(f, "unknown outcome: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(id: &str, cid: &str, status: &str, initial: f64, filled: f64, ts: i64) -> OrderMsg {
        OrderMsg {
            order_id: id.into(),
            client_id: cid.into(),
            ticker: "T".into(),
            side: BookSide::Bid,
            yes_price: 0.42,
            initial,
            filled,
            remaining: initial - filled,
            status: status.into(),
            updated_ms: ts,
        }
    }

    fn fill(fid: &str, oid: &str, side: BookSide, count: f64, ts: i64) -> FillRec {
        FillRec { fill_id: fid.into(), order_id: oid.into(), ticker: "T".into(), side, count, yes_price: 0.42, is_taker: false, fee: 0.0, ts_ms: ts }
    }

    #[test]
    fn place_ack_links_client_id_even_if_ws_arrives_first() {
        let mut s = OmsState::default();
        s.register_pending("c1", "T", BookSide::Bid, 0.42, 5.0, 1000);
        assert!(s.side_locked("T", BookSide::Bid));
        // WS user_order arrives before the HTTP ack
        s.on_order_msg(&msg("o1", "c1", "resting", 5.0, 0.0, 1001), false);
        assert_eq!(s.orders["o1"].status, OrderStatus::Resting);
        assert!(s.orders["o1"].managed);
        // the late HTTP ack is a no-op
        s.on_place_ack("c1", "o1", 1002);
        assert_eq!(s.resting("T").len(), 1);
        assert!(!s.side_locked("T", BookSide::Bid));
    }

    #[test]
    fn fills_dedup_and_never_decrease() {
        let mut s = OmsState::default();
        s.register_pending("c1", "T", BookSide::Bid, 0.42, 5.0, 1000);
        s.on_place_ack("c1", "o1", 1001);
        s.on_fill(&fill("f1", "o1", BookSide::Bid, 2.0, 1500));
        s.on_fill(&fill("f1", "o1", BookSide::Bid, 2.0, 1500)); // duplicate
        assert_eq!(s.orders["o1"].filled, 2.0);
        assert_eq!(s.orders["o1"].remaining, 3.0);
        assert_eq!(s.position("T", 2000).est, 2.0);
        // a stale order report with a lower fill count cannot roll it back
        s.on_order_msg(&msg("o1", "c1", "resting", 5.0, 0.0, 1400), false);
        assert_eq!(s.orders["o1"].filled, 2.0);
        // a newer report with the cumulative count agrees
        s.on_order_msg(&msg("o1", "c1", "resting", 5.0, 2.0, 1600), false);
        assert_eq!(s.orders["o1"].filled, 2.0);
        assert_eq!(s.orders["o1"].status, OrderStatus::Resting);
        // full fill via fills -> executed, side unlocked, position +5
        s.on_fill(&fill("f2", "o1", BookSide::Bid, 3.0, 1700));
        assert_eq!(s.orders["o1"].status, OrderStatus::Executed);
        assert!(s.resting("T").is_empty());
        assert_eq!(s.position("T", 2000).est, 5.0);
    }

    #[test]
    fn cancel_outcomes() {
        let mut s = OmsState::default();
        s.register_pending("c1", "T", BookSide::Ask, 0.42, 5.0, 1000);
        s.on_place_ack("c1", "o1", 1001);
        s.on_cancel_sent("o1", 1100);
        assert!(s.side_locked("T", BookSide::Ask));
        // a resting report during the cancel keeps the lock
        s.on_order_msg(&msg("o1", "c1", "resting", 5.0, 0.0, 1150), false);
        assert_eq!(s.orders["o1"].status, OrderStatus::Canceling);
        // engine says 3 removed -> 2 filled before the cancel landed
        s.on_cancel_outcome("o1", Some(CancelOutcome::Canceled { reduced_by: 3.0 }), 1200);
        assert_eq!(s.orders["o1"].status, OrderStatus::Canceled);
        assert_eq!(s.orders["o1"].filled, 2.0);
        assert!(!s.side_locked("T", BookSide::Ask));
        // Gone -> unknown until resolved
        s.register_pending("c2", "T", BookSide::Ask, 0.42, 1.0, 2000);
        s.on_place_ack("c2", "o2", 2001);
        s.on_cancel_sent("o2", 2100);
        s.on_cancel_outcome("o2", Some(CancelOutcome::Gone), 2200);
        assert_eq!(s.orders["o2"].status, OrderStatus::Unknown);
        assert!(s.side_locked("T", BookSide::Ask));
        // resolver: it had executed
        let ev = s.on_order_msg(&msg("o2", "c2", "executed", 1.0, 1.0, 2300), false);
        assert!(ev.iter().any(|e| matches!(e, OmsEvent::Resolved(_))));
        assert_eq!(s.orders["o2"].status, OrderStatus::Executed);
        assert!(!s.side_locked("T", BookSide::Ask));
        // transport error on cancel -> unknown
        s.register_pending("c3", "T", BookSide::Bid, 0.42, 1.0, 3000);
        s.on_place_ack("c3", "o3", 3001);
        s.on_cancel_sent("o3", 3100);
        s.on_cancel_outcome("o3", None, 3200);
        assert_eq!(s.orders["o3"].status, OrderStatus::Unknown);
    }

    #[test]
    fn rest_sync_marks_vanished_orders_unknown_after_grace() {
        let mut s = OmsState::default();
        s.register_pending("c1", "T", BookSide::Bid, 0.42, 5.0, 1000);
        s.on_place_ack("c1", "o1", 1001);
        // sync inside the grace window: nothing
        s.on_rest_orders(&[], 2000, 2000);
        assert_eq!(s.orders["o1"].status, OrderStatus::Resting);
        // after the grace window: unknown
        let ev = s.on_rest_orders(&[], 4000, 2000);
        assert!(ev.iter().any(|e| matches!(e, OmsEvent::Unknown { .. })));
        assert_eq!(s.orders["o1"].status, OrderStatus::Unknown);
        assert_eq!(s.unknown_ids(), vec!["o1".to_string()]);
        // unmanaged orders discovered from the feed never lock a side
        s.on_order_msg(&msg("x9", "", "resting", 1.0, 0.0, 5000), false);
        assert!(!s.side_locked("T", BookSide::Ask));
        assert!(s.resting("T").is_empty());
    }

    #[test]
    fn position_poll_wins_and_fills_readd() {
        let mut s = OmsState::default();
        s.on_fill(&fill("f1", "", BookSide::Bid, 2.0, 1000));
        assert_eq!(s.position("T", 1000).est, 2.0);
        s.on_rest_positions(&[("T".into(), 1.5)], 1500);
        let p = s.position("T", 1600);
        assert_eq!(p.auth, 1.5);
        assert_eq!(p.est, 1.5);
        assert_eq!(p.auth_age_ms, 100);
        s.on_fill(&fill("f2", "", BookSide::Ask, 0.5, 1700));
        assert_eq!(s.position("T", 1700).est, 1.0);
        // a poll that no longer lists T -> flat
        s.on_rest_positions(&[], 2000);
        assert_eq!(s.position("T", 2000).auth, 0.0);
        // stale position pushes are ignored
        s.on_position("T", 9.0, 1900);
        assert_eq!(s.position("T", 2000).auth, 0.0);
    }

    #[test]
    fn book_side_parsing_and_sign() {
        assert_eq!(BookSide::parse(Some("bid"), None, None), Some(BookSide::Bid));
        assert_eq!(BookSide::parse(None, Some("no"), Some("buy")), Some(BookSide::Ask));
        assert_eq!(BookSide::parse(None, Some("no"), Some("sell")), Some(BookSide::Bid));
        assert_eq!(BookSide::parse(None, Some("yes"), Some("sell")), Some(BookSide::Ask));
        assert_eq!(BookSide::Ask.signed(3.0), -3.0);
    }

    #[test]
    fn action_budget() {
        let mut s = OmsState::default();
        assert!(s.action_ok(0, 2.0));
        assert!(s.action_ok(10, 2.0));
        assert!(!s.action_ok(20, 2.0));
        assert!(s.action_ok(1100, 2.0));
    }
}
