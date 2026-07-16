//! Calibrator — the online (Δb, Δρ) fitting module (DESIGN_FAIR_RIDE §5).
//!
//! `CalibCore` is a synchronous state machine (event in → zero or more
//! `CalibUpdate`s out) so the replay binary can drive it deterministically;
//! `Calibrator` wraps it as a bus `Module` on its own task, publishing
//! `Payload::Calib` on `market.calib.<market_id>` (under `market.#` so every
//! existing subscriber pattern sees it).
//!
//! Protocol (replicates the sim's `FIT_MODE=expand` exactly):
//! - 1s samples of (tte_s, reference px mid, kalshi YES mid), with the sim's
//!   row filters (both sides fresh ≤ stale_ms, spread ≤ max_spread, two-sided).
//! - First fit when tte ≤ first_tte_s with ≥ min_rows rows (else keep waiting);
//!   warm-started refits at each further refit_every_s boundary down to
//!   entry_min (60s). All fits use ALL rows collected so far.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use tokio::task::JoinHandle;

use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::model::CalibUpdate;
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::fair::{FairSurface, FeatureState, FitRow, MAX_EXTRA};

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct CalibCfg {
    pub enabled: bool,
    /// Reference PRICE instrument. cb model: `coinbase.BTC`. px2imb: the perp
    /// (`binance.usdt_perp.BTCUSDT`) — its book also supplies imb1 sizes.
    pub reference: String,
    /// BASIS reference instrument (coinbase mid for `basis = cb − perp`);
    /// only used when the surface has extras. Default `coinbase.BTC`.
    pub basis_reference: String,
    /// Market kind to calibrate (matches MarketMeta.kind).
    pub kind: String,
    pub model_path: String,
    pub first_tte_s: f64,
    pub refit_every_s: f64,
    pub last_tte_s: f64,
    pub steps_first: u32,
    pub steps_refit: u32,
    pub lr: f64,
    pub min_rows: usize,
    pub sample_ms: i64,
    pub max_spread: f64,
    pub stale_ms: i64,
    /// Rolling fit window (seconds): fit each boundary B on rows with tte in
    /// (B, B+fit_window_s]. 0 = expanding (all history since open). The frozen
    /// analysis config uses 120 — matches the sim's `FIT_WINDOW_S`.
    pub fit_window_s: f64,
}

impl Default for CalibCfg {
    fn default() -> Self {
        CalibCfg {
            enabled: false,
            reference: "coinbase.BTC".into(),
            basis_reference: "coinbase.BTC".into(),
            kind: "15m_updown".into(),
            model_path: "models/fair-cb-x60.json".into(),
            first_tte_s: 300.0,
            refit_every_s: 60.0,
            last_tte_s: 60.0,
            steps_first: 150,
            steps_refit: 60,
            lr: 0.05,
            min_rows: 60,
            sample_ms: 1000,
            max_spread: 0.15,
            stale_ms: 1500,
            fit_window_s: 0.0,
        }
    }
}

struct EvState {
    strike: f64,
    expiry_ns: i64,
    rows: Vec<FitRow>,
    last_sample_ns: i64,
    ybid: f64,
    yask: f64,
    y_ns: i64,
    d_b: f64,
    d_rho: f64,
    fitted: bool,
    /// Next tte boundary (seconds) at which to (re)fit; decreases by
    /// refit_every_s down to last_tte_s.
    next_boundary_s: f64,
    seq: u64,
}

/// Synchronous calibration state machine. Fits run inline (few ms, once per
/// minute per event) — acceptable on the Calibrator's own task and required
/// for deterministic replay.
pub struct CalibCore {
    pub cfg: CalibCfg,
    pub surface: Arc<FairSurface>,
    pub model_hash: u64,
    ref_px: f64,
    ref_ns: i64,
    /// px2imb feature reconstruction (basis/dbasis/imb1); inactive for cb.
    feats: FeatureState,
    events: HashMap<String, EvState>,
}

impl CalibCore {
    pub fn new(cfg: CalibCfg, surface: Arc<FairSurface>, model_hash: u64) -> Self {
        let feats = FeatureState::new(surface.extras.clone());
        CalibCore { cfg, surface, model_hash, ref_px: f64::NAN, ref_ns: 0, feats, events: HashMap::new() }
    }

    /// Feed one bus event; returns any calibration updates produced.
    pub fn on_event(&mut self, ev: &Event) -> Vec<CalibUpdate> {
        match &ev.payload {
            Payload::Book(b) if b.instrument == self.cfg.reference => {
                if let (Some(&(bid, bsz)), Some(&(ask, asz))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        let mid = 0.5 * (bid + ask);
                        self.ref_px = mid;
                        self.ref_ns = b.recv_ts_ns;
                        // the price reference (perp for px2imb) also carries imb1 sizes
                        if self.feats.active() {
                            self.feats.on_perp(b.recv_ts_ns, mid, bsz, asz);
                        }
                    }
                }
            }
            Payload::Book(b) if self.feats.active() && b.instrument == self.cfg.basis_reference => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.feats.on_cb(b.recv_ts_ns, 0.5 * (bid + ask));
                    }
                }
            }
            Payload::Book(b) if b.instrument.ends_with(".YES") => {
                if let Some(st) = self.events.get_mut(&b.instrument) {
                    st.ybid = b.bids.first().map(|&(p, _)| p).unwrap_or(0.0);
                    st.yask = b.asks.first().map(|&(p, _)| p).unwrap_or(0.0);
                    st.y_ns = b.recv_ts_ns;
                }
            }
            Payload::Meta(m) if m.kind == self.cfg.kind => {
                use arb_core::model::MarketStatus::*;
                match m.status {
                    Upcoming | Live => {
                        if let (Some(strike), Some(exp)) = (m.strike, m.expiry_ts_ns) {
                            self.events.entry(m.instrument.clone()).or_insert(EvState {
                                strike,
                                expiry_ns: exp,
                                rows: Vec::new(),
                                last_sample_ns: 0,
                                ybid: 0.0,
                                yask: 0.0,
                                y_ns: 0,
                                d_b: 0.0,
                                d_rho: 0.0,
                                fitted: false,
                                next_boundary_s: self.cfg.first_tte_s,
                                seq: 0,
                            });
                        } else if m.strike.is_none() {
                            tracing::warn!(target: "fit", "meta without strike: {}", m.instrument);
                        }
                    }
                    Expired | Resolved => {
                        self.events.remove(&m.instrument);
                    }
                }
            }
            _ => {}
        }
        self.tick(ev.ts_ns)
    }

    /// Sample + boundary-fit pass at time `now` (also callable from a timer).
    pub fn tick(&mut self, now: i64) -> Vec<CalibUpdate> {
        let mut out = Vec::new();
        let cfg = &self.cfg;
        for (inst, st) in self.events.iter_mut() {
            let tte_s = (st.expiry_ns - now) as f64 / 1e9;
            if tte_s <= 0.0 {
                continue;
            }
            // 1s sampling with the sim's row filters
            if now - st.last_sample_ns >= cfg.sample_ms * 1_000_000 {
                let ref_fresh = self.ref_ns > 0 && now - self.ref_ns <= cfg.stale_ms * 1_000_000;
                let y_fresh = st.y_ns > 0 && now - st.y_ns <= cfg.stale_ms * 1_000_000;
                let two_sided = st.ybid > 0.0 && st.yask > 0.0 && st.yask > st.ybid;
                if ref_fresh && y_fresh && two_sided && (st.yask - st.ybid) <= cfg.max_spread {
                    // px2imb: reconstruct features; None (no lookback / stale cb)
                    // ⇒ drop this sample, exactly as the sim drops NaN rows (but
                    // still consume the 1s slot and fall through to the boundary
                    // fit, which uses previously-collected rows).
                    let feats = if self.feats.active() { self.feats.feats(now) } else { Some([0.0; MAX_EXTRA]) };
                    if let Some(feats) = feats {
                        st.rows.push(FitRow { tte_s, px: self.ref_px, mid: 0.5 * (st.ybid + st.yask), feats });
                    }
                    st.last_sample_ns = now;
                }
            }
            // boundary fit
            if tte_s <= st.next_boundary_s && st.next_boundary_s >= cfg.last_tte_s {
                let boundary = st.next_boundary_s;
                // Rolling window: fit on rows with tte in (B, B+window]. Expanding
                // (window 0) uses all collected rows. Borrow-free: filter into a
                // scratch Vec since `self.surface` and `st` are both needed.
                let fit_rows: Vec<FitRow> = if cfg.fit_window_s > 0.0 {
                    st.rows
                        .iter()
                        .filter(|r| r.tte_s > boundary && r.tte_s <= boundary + cfg.fit_window_s)
                        .copied()
                        .collect()
                } else {
                    st.rows.clone()
                };
                let min_fit = if cfg.fit_window_s > 0.0 { 30 } else { cfg.min_rows };
                if fit_rows.len() < min_fit {
                    continue; // keep waiting at this boundary until enough rows
                }
                let (steps, init) = if st.fitted {
                    (cfg.steps_refit, (st.d_b, st.d_rho))
                } else {
                    (cfg.steps_first, (0.0, 0.0))
                };
                let (db, dr) = self.surface.fit(&fit_rows, st.strike, init, steps, cfg.lr);
                let bce = self.surface.fit_loss(&fit_rows, st.strike, db, dr);
                st.d_b = db;
                st.d_rho = dr;
                st.fitted = true;
                st.seq += 1;
                st.next_boundary_s -= cfg.refit_every_s;
                out.push(CalibUpdate {
                    instrument: inst.clone(),
                    reference: cfg.reference.clone(),
                    seq: st.seq,
                    ts_ns: now,
                    fitted_at_tte_s: boundary,
                    rows: fit_rows.len() as u32,
                    d_b: db,
                    d_rho: dr,
                    bce,
                    model_hash: self.model_hash,
                });
            }
        }
        out
    }
}

/// FNV-1a over the model file bytes — cheap identity check between the
/// calibrator's surface and the rule's.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct Calibrator {
    cfg: CalibCfg,
    handle: Option<JoinHandle<()>>,
}

impl Calibrator {
    pub fn new(cfg: CalibCfg) -> Self {
        Calibrator { cfg, handle: None }
    }
}

#[async_trait]
impl Module for Calibrator {
    fn name(&self) -> &'static str {
        "calibrator"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        if !self.cfg.enabled {
            return Ok(());
        }
        let bytes = std::fs::read(&self.cfg.model_path)?;
        let surface = Arc::new(FairSurface::from_json(std::str::from_utf8(&bytes)?)?);
        let hash = fnv1a(&bytes);
        let mut core = CalibCore::new(self.cfg.clone(), surface, hash);
        let sub = bus.subscribe("market.#", 1024, Policy::Conflate(key_by_instrument));
        tracing::info!(
            "calibrator up: ref={} model={} hash={:016x}",
            self.cfg.reference,
            self.cfg.model_path,
            hash
        );
        let mut seq = 0u64;
        self.handle = Some(tokio::spawn(async move {
            let mut sub = sub;
            let mut tick = tokio::time::interval(std::time::Duration::from_millis(250));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                let ups = tokio::select! {
                    ev = sub.recv() => {
                        let Some(ev) = ev else { break };
                        core.on_event(&ev)
                    }
                    _ = tick.tick() => core.tick(now_ns()),
                };
                for u in ups {
                    seq += 1;
                    tracing::info!(
                        target: "fit",
                        "calib {} tte={}s rows={} db={:+.4} dr={:+.4} bce={:.4}",
                        u.instrument, u.fitted_at_tte_s, u.rows, u.d_b, u.d_rho, u.bce
                    );
                    bus.publish(Event::new(
                        format!("market.calib.{}", u.instrument),
                        "calibrator",
                        u.ts_ns,
                        seq,
                        Payload::Calib(u),
                    ));
                }
            }
        }));
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
