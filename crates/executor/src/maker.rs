//! Passive maker for the 10c–90c band (user 2026-10-04): the exceed taker keeps the
//! tails; in the core band we REST a post-only bid on the side the model favours and
//! pull it when the model turns or the book moves. Policy = the queue simulator's
//! favoured-side cell with hysteresis (tools/nowcast/maker_queue_sim.py
//! `--policy fav --theta θpost --theta-pull θpull`, lone pull, touch-move re-join):
//!
//!   * POST  side YES (bid at the YES best bid) when s > post_theta, side NO (bid at
//!     the NO best bid = YES ask) when −s > post_theta, joining the touch, only when
//!     the displayed size there is ≥ min_touch_size;
//!   * KEEP  while the side's score stays > pull_theta;
//!   * PULL  on a model flip (score ≤ pull_theta), a touch move (best price on our
//!     side ≠ our price: cancel, then re-join at the new touch), a thin touch
//!     (displayed size excluding ours < min_touch_size), leaving the band / tte
//!     window, a stale score, the balance kill, or the order feed going down;
//!   * after a model-flip pull, re-post only once the score has been back above
//!     post_theta for rejoin_hold_ms (the sim's R4).
//!
//! Order truth comes from the order manager (private WS user_orders/fill/
//! market_positions + REST sync); this module keeps one order per market and maps
//! the OMS record into its state. ONE-POSITION RULE shared with the taker through
//! `SharedExposure`: a maker post must not grow |net| past the cap (same rule as
//! the taker's per-event cap), the taker rejects while a maker order is open on the
//! market, and the maker does not post while a taker entry is in flight.
//!
//! `shadow: true` runs the full decision loop and logs every post/cancel as
//! MAKER-SHADOW without sending anything (no order manager needed).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use tokio::sync::broadcast::error::RecvError;

use arb_core::bus::Subscription;
use arb_core::event::Payload;
use arb_core::model::{ModelScore, Side};
use arb_core::now_ns;

use crate::order_manager::{BookSide, OmsEvent, OrderManager, OrderStatus, PlaceError};
use crate::types::{IntentKind, MarketParams, OrderIntent};
use crate::venue_spec::{complement, market_id_of};

const MS: i64 = 1_000_000;
const EPS: f64 = 1e-6;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct MakerCfg {
    /// Off by default.
    pub enabled: bool,
    /// Decide and log only; send nothing.
    pub shadow: bool,
    /// Post the favoured side when its score exceeds this.
    pub post_theta: f64,
    /// Keep a resting order while its side's score stays above this.
    pub pull_theta: f64,
    /// YES-mid band the maker works in (the taker keeps the tails outside it).
    pub px_lo: f64,
    pub px_hi: f64,
    /// Time-to-expiry window (seconds): post/keep only while min < tte ≤ max.
    pub min_tte_s: f64,
    pub max_tte_s: f64,
    /// Displayed size at the touch (excluding ours) below which we do not join / pull.
    pub min_touch_size: f64,
    /// No quoting when the YES spread is wider than this.
    pub max_spread: f64,
    /// A score older than this counts as missing (pull).
    pub score_max_age_ms: i64,
    /// After a model-flip pull: the score must be back above post_theta this long.
    pub rejoin_hold_ms: i64,
    /// Housekeeping cadence (staleness pulls, OMS refresh).
    pub tick_ms: u64,
    /// A placement whose outcome never resolved (no order id seen) is dropped after this.
    pub stuck_unknown_ms: i64,
    /// Per-market state line cadence (0 = off).
    pub log_state_s: u64,
}

impl Default for MakerCfg {
    fn default() -> Self {
        MakerCfg {
            enabled: false,
            shadow: true,
            post_theta: 0.34,
            pull_theta: -0.20,
            px_lo: 0.10,
            px_hi: 0.90,
            min_tte_s: 60.0,
            max_tte_s: 300.0,
            min_touch_size: 100.0,
            max_spread: 0.10,
            score_max_age_ms: 500,
            rejoin_hold_ms: 1000,
            tick_ms: 100,
            stuck_unknown_ms: 10_000,
            log_state_s: 30,
        }
    }
}

// ─────────────────────────────── shared exposure ────────────────────────────────

/// Per-market exposure shared by the taker (Engine) and the maker. Nets are signed
/// YES contracts (+long YES / −long NO).
#[derive(Clone, Copy, Debug, Default)]
pub struct MktExposure {
    /// Taker fills (the executor's position manager), set after every taker entry.
    pub taker_net: f64,
    /// Maker fills (OMS fill events on maker order ids).
    pub maker_net: f64,
    /// The venue's own position (OMS: market_positions push / REST poll).
    pub oms_net: f64,
    /// A maker order is pending / resting / canceling / unresolved on this market.
    pub maker_open: bool,
    /// A taker entry is in flight on this market.
    pub taker_inflight: bool,
    pub touched_ns: i64,
}

impl MktExposure {
    /// Exposure in direction `dir` (+1 YES / −1 NO): the larger of the in-process
    /// sum and the venue's own number (a fill either source has not seen yet still counts).
    pub fn dir_exposure(&self, dir: f64) -> f64 {
        (dir * (self.taker_net + self.maker_net)).max(dir * self.oms_net)
    }
}

/// Per-event position cap shared by maker and taker: an entry may not grow the
/// directional exposure past `cap` (0 = uncapped); an entry that reduces it passes.
pub fn cap_ok(exposure_dir: f64, size: f64, cap: f64) -> bool {
    cap <= 0.0 || exposure_dir + size <= cap + EPS
}

#[derive(Clone, Default)]
pub struct SharedExposure(Arc<Mutex<HashMap<String, MktExposure>>>);

impl SharedExposure {
    pub fn get(&self, market: &str) -> MktExposure {
        self.0.lock().unwrap().get(market).copied().unwrap_or_default()
    }
    pub fn update<F: FnOnce(&mut MktExposure)>(&self, market: &str, f: F) {
        let mut g = self.0.lock().unwrap();
        let e = g.entry(market.to_string()).or_default();
        f(e);
        e.touched_ns = now_ns();
    }
    fn gc(&self, older_than_ns: i64) {
        self.0.lock().unwrap().retain(|_, e| e.maker_open || e.taker_inflight || e.touched_ns >= older_than_ns);
    }
}

// ─────────────────────────────── pure decision ──────────────────────────────────

/// YES-book top of one market.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Top {
    pub bid: f64,
    pub bid_sz: f64,
    pub ask: f64,
    pub ask_sz: f64,
}

/// Our resting order as the decision sees it (YES-book side + price).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestingView {
    pub side: BookSide,
    pub yes_px: f64,
    pub remaining: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Decision {
    Hold,
    Place { side: BookSide, yes_px: f64 },
    Cancel(&'static str),
}

pub struct Inputs<'a> {
    pub now_ns: i64,
    pub score: Option<&'a ModelScore>,
    pub top: Option<Top>,
    pub resting: Option<RestingView>,
    pub exposure: MktExposure,
    pub cap: f64,
    pub size: f64,
    pub halted: bool,
    pub feed_ok: bool,
}

/// The side's own score: s for the YES bid, −s for the NO bid (YES ask).
fn side_score(side: BookSide, s: f64) -> f64 {
    match side {
        BookSide::Bid => s,
        BookSide::Ask => -s,
    }
}

fn touch_of(side: BookSide, t: &Top) -> (f64, f64) {
    match side {
        BookSide::Bid => (t.bid, t.bid_sz),
        BookSide::Ask => (t.ask, t.ask_sz),
    }
}

/// One decision for one market. Resting orders that are not in the `Resting`
/// state are never passed in (in-flight orders are left alone by the caller).
pub fn decide(cfg: &MakerCfg, i: &Inputs) -> Decision {
    let sc = i.score.filter(|s| i.now_ns - s.ts_ns <= cfg.score_max_age_ms * MS);
    let not_ok: Option<&'static str> = if i.halted {
        Some("halted")
    } else if !i.feed_ok {
        Some("feed")
    } else if sc.is_none() {
        Some("stale")
    } else {
        match i.top {
            None => Some("nobook"),
            Some(t) if !(t.bid > 0.0 && t.ask > t.bid) => Some("nobook"),
            Some(t) if t.ask - t.bid > cfg.max_spread + EPS => Some("spread"),
            Some(t) => {
                let mid = 0.5 * (t.bid + t.ask);
                let tte_s = (sc.unwrap().expiry_ns - i.now_ns) as f64 / 1e9;
                if mid < cfg.px_lo - EPS || mid > cfg.px_hi + EPS {
                    Some("region")
                } else if !(tte_s > cfg.min_tte_s && tte_s <= cfg.max_tte_s) {
                    Some("window")
                } else {
                    None
                }
            }
        }
    };
    let s = sc.map(|x| x.s).unwrap_or(0.0);
    match i.resting {
        Some(r) => {
            if let Some(why) = not_ok {
                return Decision::Cancel(why);
            }
            if side_score(r.side, s) <= cfg.pull_theta {
                return Decision::Cancel("flag");
            }
            let t = i.top.expect("checked");
            let (touch, lvl) = touch_of(r.side, &t);
            if (touch - r.yes_px).abs() > EPS {
                return Decision::Cancel("move");
            }
            if lvl - r.remaining < cfg.min_touch_size {
                return Decision::Cancel("thin");
            }
            Decision::Hold
        }
        None => {
            if not_ok.is_some() || i.exposure.taker_inflight {
                return Decision::Hold;
            }
            let side = if s > cfg.post_theta {
                BookSide::Bid
            } else if -s > cfg.post_theta {
                BookSide::Ask
            } else {
                return Decision::Hold;
            };
            let dir = if side == BookSide::Bid { 1.0 } else { -1.0 };
            if !cap_ok(i.exposure.dir_exposure(dir), i.size, i.cap) {
                return Decision::Hold;
            }
            let t = i.top.expect("checked");
            let (touch, lvl) = touch_of(side, &t);
            if lvl < cfg.min_touch_size {
                return Decision::Hold;
            }
            Decision::Place { side, yes_px: touch }
        }
    }
}

// ─────────────────────────────── per-market state ───────────────────────────────

#[derive(Clone, Debug)]
struct MkOrder {
    client_id: String,
    order_id: String,
    side: BookSide,
    yes_px: f64,
    remaining: f64,
    status: OrderStatus,
    placed_ns: i64,
    unknown_since: Option<i64>,
}

#[derive(Default)]
struct Mk {
    inst_yes: String,
    score: Option<ModelScore>,
    top: Option<Top>,
    order: Option<MkOrder>,
    /// Reason + time of our last pull (rejoin hysteresis after "flag").
    last_pull: Option<(&'static str, i64)>,
    clear_since: Option<i64>,
    n_post: u32,
    n_cancel: u32,
    n_fill: f64,
    last_state_log_ns: i64,
}

pub struct MakerCtx {
    pub cfg: MakerCfg,
    pub size: f64,
    pub cap: f64,
    pub params: MarketParams,
    pub om: Option<OrderManager>,
    pub exposure: SharedExposure,
    pub halted: Arc<AtomicBool>,
}

struct Maker {
    ctx: MakerCtx,
    mk: HashMap<String, Mk>,
    /// maker order id -> market (fills can land after the order left our state).
    ids: HashMap<String, (String, i64)>,
    seq: u64,
}

impl Maker {
    fn feed_ok(&self) -> bool {
        if self.ctx.cfg.shadow {
            return true;
        }
        match &self.ctx.om {
            Some(om) => {
                let h = om.health();
                h.ws_up && h.sync_age_ms < 5_000
            }
            None => false,
        }
    }

    /// Pull the OMS record of our order into the local view; drop terminal orders.
    /// Shadow orders exist only here — the OMS never knows them, so skip.
    fn refresh(&mut self, market: &str) {
        if self.ctx.cfg.shadow {
            return;
        }
        let now = now_ns();
        let Some(om) = self.ctx.om.clone() else { return };
        let stuck = self.ctx.cfg.stuck_unknown_ms * MS;
        let Some(m) = self.mk.get_mut(market) else { return };
        let Some(o) = m.order.as_mut() else { return };
        let rec = if o.order_id.is_empty() { om.order_by_client(&o.client_id) } else { om.order(&o.order_id) };
        match rec {
            Some(r) => {
                if !r.order_id.is_empty() && o.order_id != r.order_id {
                    o.order_id = r.order_id.clone();
                    self.ids.insert(r.order_id.clone(), (market.to_string(), now));
                }
                o.status = r.status;
                if r.remaining >= 0.0 {
                    o.remaining = r.remaining;
                }
                if r.status == OrderStatus::Unknown {
                    let since = *o.unknown_since.get_or_insert(now);
                    if o.order_id.is_empty() && now - since > stuck {
                        tracing::warn!(target: "maker", "MAKER {market} placement {} never resolved after {} ms — dropped (position authority covers any fill)", o.client_id, (now - since) / MS);
                        m.order = None;
                    }
                } else {
                    o.unknown_since = None;
                }
                if let Some(o) = m.order.as_ref() {
                    if !o.status.is_open() {
                        tracing::info!(target: "maker", "MAKER done {market} {:?} {} status={:?} filled={:.2}/{:.2}", o.side, o.order_id, o.status, r.filled, r.initial);
                        m.order = None;
                    }
                }
            }
            None => {
                // not (or no longer) known to the OMS: a definite reject removed it, or GC
                if o.status != OrderStatus::Pending || now - o.placed_ns > stuck {
                    tracing::info!(target: "maker", "MAKER gone {market} {} (not in OMS)", o.client_id);
                    m.order = None;
                }
            }
        }
    }

    /// Publish our open-order flag + the venue net into the shared exposure.
    /// Shadow mode never writes: pretend orders must not block the real taker.
    fn sync_exposure(&self, market: &str) {
        if self.ctx.cfg.shadow {
            return;
        }
        let open = self.mk.get(market).map(|m| m.order.is_some()).unwrap_or(false);
        let oms_net = self.ctx.om.as_ref().map(|om| om.position(market).est);
        self.ctx.exposure.update(market, |e| {
            e.maker_open = open;
            if let Some(n) = oms_net {
                e.oms_net = n;
            }
        });
    }

    async fn step(&mut self, market: &str) {
        let now = now_ns();
        self.refresh(market);
        self.sync_exposure(market);
        let cfg = self.ctx.cfg.clone();
        let feed_ok = self.feed_ok();
        let halted = self.ctx.halted.load(Ordering::SeqCst);
        let exposure = self.ctx.exposure.get(market);
        let Some(m) = self.mk.get_mut(market) else { return };
        // in-flight orders (pending / canceling / unknown) are left to the OMS
        if let Some(o) = &m.order {
            if o.status != OrderStatus::Resting {
                return;
            }
        }
        let resting = m.order.as_ref().map(|o| RestingView { side: o.side, yes_px: o.yes_px, remaining: o.remaining });
        let d = decide(
            &cfg,
            &Inputs { now_ns: now, score: m.score.as_ref(), top: m.top, resting, exposure, cap: self.ctx.cap, size: self.ctx.size, halted, feed_ok },
        );
        let s = m.score.as_ref().map(|x| x.s).unwrap_or(f64::NAN);
        let tte = m.score.as_ref().map(|x| (x.expiry_ns - now) as f64 / 1e9).unwrap_or(f64::NAN);
        let topstr = m.top.map(|t| format!("{:.2}x{:.0}/{:.2}x{:.0}", t.bid, t.bid_sz, t.ask, t.ask_sz)).unwrap_or_else(|| "-".into());
        if cfg.log_state_s > 0 && now - m.last_state_log_ns >= cfg.log_state_s as i64 * 1_000_000_000 && (m.order.is_some() || m.n_post > 0) {
            m.last_state_log_ns = now;
            tracing::info!(target: "maker", "MAKER state {market} order={} s={s:+.3} tte={tte:.1} top={topstr} posts={} cancels={} filled={:.2} exp_net={:+.2}/{:+.2}/{:+.2}",
                m.order.as_ref().map(|o| format!("{:?}@{:.2}", o.side, o.yes_px)).unwrap_or_else(|| "none".into()), m.n_post, m.n_cancel, m.n_fill,
                exposure.taker_net, exposure.maker_net, exposure.oms_net);
        }
        match d {
            Decision::Hold => {
                // the rejoin clock only runs while the side is wanted again
                if m.order.is_none() {
                    m.clear_since = None;
                }
            }
            Decision::Place { side, yes_px } => {
                if let Some(("flag", _)) = m.last_pull {
                    let since = *m.clear_since.get_or_insert(now);
                    if now - since < cfg.rejoin_hold_ms * MS {
                        return;
                    }
                }
                m.last_pull = None;
                m.clear_since = None;
                let inst = if side == BookSide::Bid { m.inst_yes.clone() } else { complement(&m.inst_yes) };
                let token_px = if side == BookSide::Bid { yes_px } else { 1.0 - yes_px };
                let expiry_ns = m.score.as_ref().map(|x| x.expiry_ns).unwrap_or(now);
                self.seq += 1;
                let client_id = format!("mk-{}-{}", now / MS, self.seq);
                let intent = OrderIntent {
                    client_id: client_id.clone(),
                    instrument: inst.clone(),
                    token_id: String::new(),
                    side: Side::Buy,
                    price: token_px,
                    size: self.ctx.size,
                    kind: IntentKind::RestUntil { expiry_ns },
                    params: self.ctx.params,
                    expiry_ns,
                };
                m.n_post += 1;
                let tag = if cfg.shadow { "MAKER-SHADOW" } else { "MAKER" };
                tracing::info!(target: "maker", "{tag} post {market} {side:?} buy {inst} @ {token_px:.2} (yes {yes_px:.2}) s={s:+.3} tte={tte:.1} top={topstr} cid={client_id}");
                m.order = Some(MkOrder {
                    client_id: client_id.clone(),
                    order_id: String::new(),
                    side,
                    yes_px,
                    remaining: self.ctx.size,
                    status: if cfg.shadow { OrderStatus::Resting } else { OrderStatus::Pending },
                    placed_ns: now,
                    unknown_since: None,
                });
                if cfg.shadow {
                    return;
                }
                // publish maker_open BEFORE the await so the taker sees it
                self.ctx.exposure.update(market, |e| e.maker_open = true);
                let om = self.ctx.om.clone().expect("live maker requires the order manager");
                let res = om.place(&intent, market, side).await;
                let m = self.mk.get_mut(market).expect("market");
                match res {
                    Ok(id) => {
                        self.ids.insert(id.clone(), (market.to_string(), now_ns()));
                        if let Some(o) = m.order.as_mut() {
                            o.order_id = id.clone();
                            if o.status == OrderStatus::Pending {
                                o.status = OrderStatus::Resting;
                            }
                        }
                        tracing::info!(target: "maker", "MAKER acked {market} {id} in {} ms", (now_ns() - now) / MS);
                    }
                    Err(PlaceError::Unknown(e)) => {
                        tracing::warn!(target: "maker", "MAKER place outcome unknown {market} {client_id}: {e}");
                        if let Some(o) = m.order.as_mut() {
                            o.status = OrderStatus::Unknown;
                            o.unknown_since = Some(now_ns());
                        }
                    }
                    Err(e) => {
                        tracing::info!(target: "maker", "MAKER place failed {market} {client_id}: {e}");
                        m.order = None;
                    }
                }
                self.sync_exposure(market);
            }
            Decision::Cancel(reason) => {
                let o = m.order.clone().expect("cancel implies an order");
                m.n_cancel += 1;
                m.last_pull = Some((reason, now));
                m.clear_since = None;
                let tag = if cfg.shadow { "MAKER-SHADOW" } else { "MAKER" };
                tracing::info!(target: "maker", "{tag} cancel {market} {:?} @ {:.2} reason={reason} s={s:+.3} tte={tte:.1} top={topstr} rested_ms={} id={}",
                    o.side, o.yes_px, (now - o.placed_ns) / MS, if o.order_id.is_empty() { &o.client_id } else { &o.order_id });
                if cfg.shadow {
                    m.order = None;
                    self.sync_exposure(market);
                    return;
                }
                let om = self.ctx.om.clone().expect("live maker requires the order manager");
                if o.order_id.is_empty() {
                    return; // cannot cancel without an id; refresh links it
                }
                if let Some(oo) = m.order.as_mut() {
                    oo.status = OrderStatus::Canceling;
                }
                let out = om.cancel(&o.order_id).await;
                match &out {
                    Ok(oc) => tracing::info!(target: "maker", "MAKER canceled {market} {} -> {oc:?} in {} ms", o.order_id, (now_ns() - now) / MS),
                    Err(e) => tracing::warn!(target: "maker", "MAKER cancel error {market} {}: {e} (OMS resolves)", o.order_id),
                }
                self.refresh(market);
                self.sync_exposure(market);
            }
        }
    }

    fn on_oms(&mut self, ev: &OmsEvent) -> Option<String> {
        match ev {
            OmsEvent::Fill(f) => {
                let (market, _) = self.ids.get(&f.order_id)?.clone();
                let signed = f.side.signed(f.count);
                self.ctx.exposure.update(&market, |e| e.maker_net += signed);
                if let Some(m) = self.mk.get_mut(&market) {
                    m.n_fill += f.count;
                }
                tracing::info!(target: "maker", "MAKER FILL {market} {:?} {:.2} @ {:.2} (yes) taker={} order {} fee {:.4}", f.side, f.count, f.yes_price, f.is_taker, f.order_id, f.fee);
                Some(market)
            }
            OmsEvent::OrderUpdate(r) | OmsEvent::Resolved(r) => {
                if self.ids.contains_key(&r.order_id) || r.client_id.starts_with("mk-") {
                    Some(r.ticker.clone())
                } else {
                    None
                }
            }
            OmsEvent::Position { ticker, auth, .. } => {
                let a = *auth;
                self.ctx.exposure.update(ticker, |e| e.oms_net = a);
                if self.mk.contains_key(ticker) {
                    Some(ticker.clone())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn on_score(&mut self, sc: &ModelScore) -> Option<String> {
        let market = market_id_of(&sc.instrument)?.to_string();
        let m = self.mk.entry(market.clone()).or_insert_with(|| Mk { inst_yes: sc.instrument.clone(), ..Default::default() });
        m.score = Some(sc.clone());
        Some(market)
    }

    fn on_book(&mut self, inst: &str, top: Top) -> Option<String> {
        let market = market_id_of(inst)?;
        let m = self.mk.get_mut(market)?;
        if inst != m.inst_yes {
            return None;
        }
        let changed = m.top != Some(top);
        m.top = Some(top);
        if changed {
            Some(market.to_string())
        } else {
            None
        }
    }

    fn gc(&mut self) {
        let now = now_ns();
        self.mk.retain(|_, m| m.order.is_some() || m.score.as_ref().map(|s| s.expiry_ns > now - 60_000 * MS).unwrap_or(false));
        self.ids.retain(|_, (_, t)| now - *t < 20 * 60_000 * MS);
        self.ctx.exposure.gc(now - 30 * 60_000 * MS);
    }
}

/// Spawn the maker task. The subscriptions are taken by the caller (synchronously,
/// in `Executor::start`) so nothing published before the task runs is missed.
pub fn spawn(ctx: MakerCtx, score_sub: Subscription, book_sub: Subscription) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut oms_rx = ctx.om.as_ref().map(|om| om.subscribe());
        let tick_ms = ctx.cfg.tick_ms.max(20);
        tracing::info!(target: "maker", "maker up: shadow={} post_theta={} pull_theta={} band [{},{}] tte ({},{}] min_touch={} max_spread={} size={} cap={} oms={}",
            ctx.cfg.shadow, ctx.cfg.post_theta, ctx.cfg.pull_theta, ctx.cfg.px_lo, ctx.cfg.px_hi, ctx.cfg.min_tte_s, ctx.cfg.max_tte_s,
            ctx.cfg.min_touch_size, ctx.cfg.max_spread, ctx.size, ctx.cap, ctx.om.is_some());
        let mut mk = Maker { ctx, mk: HashMap::new(), ids: HashMap::new(), seq: 0 };
        let mut score_sub = score_sub;
        let mut book_sub = book_sub;
        let mut tick = tokio::time::interval(Duration::from_millis(tick_ms));
        let mut last_gc = now_ns();
        loop {
            let target: Vec<String> = tokio::select! {
                ev = score_sub.recv() => {
                    let Some(ev) = ev else { break };
                    match &ev.payload { Payload::Score(sc) => mk.on_score(sc).into_iter().collect(), _ => vec![] }
                }
                ev = book_sub.recv() => {
                    let Some(ev) = ev else { break };
                    match &ev.payload {
                        Payload::Book(b) => {
                            let top = Top {
                                bid: b.bids.first().map(|x| x.0).unwrap_or(0.0),
                                bid_sz: b.bids.first().map(|x| x.1).unwrap_or(0.0),
                                ask: b.asks.first().map(|x| x.0).unwrap_or(0.0),
                                ask_sz: b.asks.first().map(|x| x.1).unwrap_or(0.0),
                            };
                            mk.on_book(&b.instrument, top).into_iter().collect()
                        }
                        _ => vec![],
                    }
                }
                ev = async { match oms_rx.as_mut() { Some(rx) => Some(rx.recv().await), None => std::future::pending().await } } => {
                    match ev {
                        Some(Ok(e)) => mk.on_oms(&e).into_iter().collect(),
                        Some(Err(RecvError::Lagged(n))) => { tracing::warn!(target: "maker", "MAKER oms events lagged {n}"); mk.mk.keys().cloned().collect() }
                        Some(Err(RecvError::Closed)) | None => { oms_rx = None; vec![] }
                    }
                }
                _ = tick.tick() => {
                    let now = now_ns();
                    if now - last_gc > 60_000 * MS { mk.gc(); last_gc = now; }
                    mk.mk.iter().filter(|(_, m)| m.order.is_some()).map(|(k, _)| k.clone()).collect()
                }
            };
            for market in target {
                mk.step(&market).await;
            }
        }
        tracing::warn!(target: "maker", "maker task ended");
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MakerCfg {
        MakerCfg { enabled: true, shadow: false, ..Default::default() }
    }

    fn score(s: f64, now: i64) -> ModelScore {
        ModelScore {
            instrument: "kalshi.KXBTC15M-X.YES".into(),
            model: "exceed".into(),
            ts_ns: now,
            expiry_ns: now + 200 * 1_000_000_000,
            s,
            p_up: 0.0,
            p_dn: 0.0,
            mid: 0.5,
            best_bid: 0.49,
            best_ask: 0.51,
        }
    }

    fn top() -> Top {
        Top { bid: 0.49, bid_sz: 500.0, ask: 0.51, ask_sz: 400.0 }
    }

    fn inp<'a>(sc: Option<&'a ModelScore>, t: Option<Top>, r: Option<RestingView>, now: i64) -> Inputs<'a> {
        Inputs { now_ns: now, score: sc, top: t, resting: r, exposure: MktExposure::default(), cap: 1.0, size: 1.0, halted: false, feed_ok: true }
    }

    const NOW: i64 = 1_000_000_000_000;

    #[test]
    fn posts_favoured_side_at_the_touch() {
        let c = cfg();
        let up = score(0.40, NOW);
        assert_eq!(decide(&c, &inp(Some(&up), Some(top()), None, NOW)), Decision::Place { side: BookSide::Bid, yes_px: 0.49 });
        let dn = score(-0.40, NOW);
        assert_eq!(decide(&c, &inp(Some(&dn), Some(top()), None, NOW)), Decision::Place { side: BookSide::Ask, yes_px: 0.51 });
        let flat = score(0.30, NOW);
        assert_eq!(decide(&c, &inp(Some(&flat), Some(top()), None, NOW)), Decision::Hold);
    }

    #[test]
    fn hysteresis_keeps_until_pull_theta() {
        let c = cfg();
        let r = Some(RestingView { side: BookSide::Bid, yes_px: 0.49, remaining: 1.0 });
        // below post, above pull: keep
        let s = score(0.0, NOW);
        assert_eq!(decide(&c, &inp(Some(&s), Some(top()), r, NOW)), Decision::Hold);
        // at/below pull: cancel
        let s = score(-0.20, NOW);
        assert_eq!(decide(&c, &inp(Some(&s), Some(top()), r, NOW)), Decision::Cancel("flag"));
        // the NO side mirrors
        let rn = Some(RestingView { side: BookSide::Ask, yes_px: 0.51, remaining: 1.0 });
        let s = score(0.25, NOW);
        assert_eq!(decide(&c, &inp(Some(&s), Some(top()), rn, NOW)), Decision::Cancel("flag"));
    }

    #[test]
    fn book_moves_and_thin_levels_pull() {
        let c = cfg();
        let s = score(0.40, NOW);
        let r = Some(RestingView { side: BookSide::Bid, yes_px: 0.49, remaining: 1.0 });
        let moved = Top { bid: 0.50, ..top() };
        assert_eq!(decide(&c, &inp(Some(&s), Some(moved), r, NOW)), Decision::Cancel("move"));
        let dropped = Top { bid: 0.48, ..top() };
        assert_eq!(decide(&c, &inp(Some(&s), Some(dropped), r, NOW)), Decision::Cancel("move"));
        // 100 displayed incl. our 1 -> 99 others < 100
        let thin = Top { bid_sz: 100.0, ..top() };
        assert_eq!(decide(&c, &inp(Some(&s), Some(thin), r, NOW)), Decision::Cancel("thin"));
        // and we do not join a thin level
        assert_eq!(decide(&c, &inp(Some(&s), Some(Top { bid_sz: 50.0, ..top() }), None, NOW)), Decision::Hold);
    }

    #[test]
    fn region_window_staleness_and_kill_pull() {
        let c = cfg();
        let r = Some(RestingView { side: BookSide::Bid, yes_px: 0.05, remaining: 1.0 });
        let s = score(0.40, NOW);
        let tails = Top { bid: 0.05, bid_sz: 500.0, ask: 0.06, ask_sz: 500.0 };
        assert_eq!(decide(&c, &inp(Some(&s), Some(tails), r, NOW)), Decision::Cancel("region"));
        let r = Some(RestingView { side: BookSide::Bid, yes_px: 0.49, remaining: 1.0 });
        let late = ModelScore { expiry_ns: NOW + 50 * 1_000_000_000, ..score(0.40, NOW) };
        assert_eq!(decide(&c, &inp(Some(&late), Some(top()), r, NOW)), Decision::Cancel("window"));
        let old = score(0.40, NOW - 600 * MS);
        assert_eq!(decide(&c, &inp(Some(&old), Some(top()), r, NOW)), Decision::Cancel("stale"));
        assert_eq!(decide(&c, &inp(None, Some(top()), r, NOW)), Decision::Cancel("stale"));
        let mut i = inp(Some(&s), Some(top()), r, NOW);
        i.halted = true;
        assert_eq!(decide(&c, &i), Decision::Cancel("halted"));
        let mut i = inp(Some(&s), Some(top()), r, NOW);
        i.feed_ok = false;
        assert_eq!(decide(&c, &i), Decision::Cancel("feed"));
        let wide = Top { bid: 0.40, ask: 0.55, ..top() };
        assert_eq!(decide(&c, &inp(Some(&s), Some(wide), r, NOW)), Decision::Cancel("spread"));
    }

    #[test]
    fn one_position_rule_shared_with_the_taker() {
        let c = cfg();
        let s = score(0.40, NOW);
        // long 1 YES from the taker: no more YES bids
        let mut i = inp(Some(&s), Some(top()), None, NOW);
        i.exposure.taker_net = 1.0;
        assert_eq!(decide(&c, &i), Decision::Hold);
        // the venue already shows the position before our own accounting does
        let mut i = inp(Some(&s), Some(top()), None, NOW);
        i.exposure.oms_net = 1.0;
        assert_eq!(decide(&c, &i), Decision::Hold);
        // long 1 YES and the model favours NO: buying NO reduces exposure -> allowed
        let sn = score(-0.40, NOW);
        let mut i = inp(Some(&sn), Some(top()), None, NOW);
        i.exposure.maker_net = 1.0;
        assert_eq!(decide(&c, &i), Decision::Place { side: BookSide::Ask, yes_px: 0.51 });
        // a taker entry in flight blocks posting
        let mut i = inp(Some(&s), Some(top()), None, NOW);
        i.exposure.taker_inflight = true;
        assert_eq!(decide(&c, &i), Decision::Hold);
        assert!(cap_ok(0.0, 1.0, 1.0));
        assert!(!cap_ok(1.0, 1.0, 1.0));
        assert!(cap_ok(-1.0, 1.0, 1.0));
        assert!(cap_ok(5.0, 1.0, 0.0));
    }
}
