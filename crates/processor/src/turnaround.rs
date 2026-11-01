//! Turnaround shadow scorer: the `up30` cheap-side delta head (tools/nowcast/train_turnaround.py --export)
//! evaluated live on tail rows, SHADOW-ONLY — it emits no TradeSignals, only `TURN-SIG` log lines (one per
//! event per tier) plus periodic stats, so live fill reality can be measured before any arming decision.
//!
//! Feature parity with the offline builder (build_v10 + train_turnaround.load_day):
//!   grid        200 ms; a value at lag j = the last update at/before now − j·200 ms
//!   hist        m · (lg(mid_now) − lg(mid_lag))            for lag in LAGS
//!   perp        m · (perp_lag − perp_now)                  raw dollars
//!   kz          m·dp, m·dc, m·dp/sc, m·dc/sc, m·(Φ(dc/sc) − mid), m·(Φ(dp/sc) − mid)
//!               dp = perp − K, dc = cb − K, sc = 150·√(max(tte,1)/900)
//!   ctx         ln(max(tte,1)), σ, 100·spread, lg(p_cheap), mid·(1−mid), lo
//!   m = +1 on the low tail (cheap side = YES), −1 on the high tail; σ = exceed's σ (same construction).
//! Rows are only scored where the offline builder kept them: tails, tte in window, sane book (spr ≤ 0.06).

use std::collections::{HashMap, VecDeque};

use arb_core::event::{Event, Payload};
use serde::Deserialize;

use crate::exceed::sigmoid;
use arb_core::model::TradeSignal;

use crate::rule::Rule;
use crate::state::MarketState;

const GRID_NS: i64 = 200_000_000;
const KEEP_NS: i64 = 62_000_000_000; // 300 lags · 200 ms + margin
const SIG_WIN: usize = 600;
const SIG_MIN: usize = 200;
const SIG_FLOOR: f64 = 1.25;
const EPS: f64 = 1e-4;

fn lg(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

/// Standard normal CDF via Abramowitz–Stegun 26.2.17 (|err| < 7.5e-8).
fn ndtr(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let p = 1.0 - pdf * poly;
    if x >= 0.0 { p } else { 1.0 - p }
}

fn prune(ring: &mut VecDeque<(i64, f64)>, now: i64) {
    while ring.len() >= 2 && ring[1].0 <= now - KEEP_NS {
        ring.pop_front();
    }
}

fn at(ring: &VecDeque<(i64, f64)>, t: i64) -> Option<f64> {
    ring.iter().rev().find(|&&(ts, _)| ts <= t).map(|&(_, x)| x)
}

// ───────────────────────────── model ─────────────────────────────

#[derive(Deserialize)]
struct LayerJson {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Deserialize)]
struct MemberJson {
    layers: Vec<LayerJson>,
}

#[derive(Deserialize)]
struct TurnJson {
    kind: String,
    target: String,
    feat: String,
    lags: Vec<usize>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    thresholds: HashMap<String, f64>,
    members: Vec<MemberJson>,
    #[serde(default)]
    test_vectors: serde_json::Value,
}

struct Mlp {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<(usize, usize)>,
}

impl Mlp {
    fn forward(&self, x: &[f64]) -> f64 {
        let mut cur = x.to_vec();
        let last = self.dims.len() - 1;
        for (li, &(ind, outd)) in self.dims.iter().enumerate() {
            let mut nxt = vec![0.0; outd];
            for o in 0..outd {
                let row = &self.w[li][o * ind..(o + 1) * ind];
                let mut acc = self.b[li][o];
                for (a_, b_) in row.iter().zip(cur.iter()) {
                    acc += a_ * b_;
                }
                nxt[o] = if li < last { acc.max(0.0) } else { acc };
            }
            cur = nxt;
        }
        cur[0]
    }
}

pub struct TurnModel {
    pub lags: Vec<usize>,
    pub nf: usize,
    pub target: String,
    pub thresholds: HashMap<String, f64>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    members: Vec<Mlp>,
}

impl TurnModel {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let js: TurnJson = serde_json::from_str(text)?;
        anyhow::ensure!(js.kind == "turnaround", "not a turnaround model (kind {:?})", js.kind);
        anyhow::ensure!(js.feat == "hist,perp,kz", "turnaround: unsupported feature set {}", js.feat);
        let nf = js.mu.len();
        anyhow::ensure!(nf == 2 * js.lags.len() + 6 + 6, "turnaround: nf {} != feature spec", nf);
        anyhow::ensure!(js.sd.len() == nf && !js.members.is_empty());
        let mut members = Vec::new();
        for m in &js.members {
            let (mut w, mut b, mut dims) = (Vec::new(), Vec::new(), Vec::new());
            for l in &m.layers {
                let outd = l.w.len();
                let ind = l.w[0].len();
                anyhow::ensure!(l.b.len() == outd);
                let mut flat = Vec::with_capacity(outd * ind);
                for row in &l.w {
                    anyhow::ensure!(row.len() == ind, "turnaround: ragged weights");
                    flat.extend_from_slice(row);
                }
                w.push(flat);
                b.push(l.b.clone());
                dims.push((ind, outd));
            }
            anyhow::ensure!(dims.first().map(|d| d.0) == Some(nf) && dims.last().map(|d| d.1) == Some(1));
            members.push(Mlp { w, b, dims });
        }
        Ok(TurnModel { lags: js.lags, nf, target: js.target, thresholds: js.thresholds, mu: js.mu, sd: js.sd, members })
    }

    /// Ensemble probability from a RAW (pre-standardization) feature vector.
    pub fn score(&self, x: &[f64]) -> f64 {
        let z: Vec<f64> = x.iter().zip(self.mu.iter().zip(self.sd.iter())).map(|(v, (m, s))| (v - m) / s).collect();
        self.members.iter().map(|m| sigmoid(m.forward(&z))).sum::<f64>() / self.members.len() as f64
    }
}

// ───────────────────────────── config ─────────────────────────────

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct TurnCfg {
    pub enabled: bool,
    pub model_path: String,
    /// Perp reference (bookTicker mid) and the coinbase mid instrument (kz needs both).
    pub reference: String,
    pub cb_instrument: String,
    pub px_lo: f64,
    pub px_hi: f64,
    pub entry_min_tte_s: f64,
    pub entry_max_tte_s: f64,
    /// Shadow tiers: log TURN-SIG when p ≥ tier (json threshold keys, e.g. "q95"/"q98").
    pub tier_lo: String,
    pub tier_hi: String,
    /// Max cheap-side ask for a signal (the sim's cask cap).
    pub max_cheap_ask: f64,
    /// Book sanity for scoring (the builder kept spr ≤ 0.06 only).
    pub max_spread: f64,
    pub cb_max_age_ms: i64,
    pub eval_min_ms: i64,
    pub log_every_s: f64,
}

impl Default for TurnCfg {
    fn default() -> Self {
        TurnCfg {
            enabled: false,
            model_path: "models/turnaround-up30-rs-btc.json".into(),
            reference: "binance.usdt_perp.BTCUSDT".into(),
            cb_instrument: "coinbase.BTC".into(),
            px_lo: 0.10,
            px_hi: 0.90,
            entry_min_tte_s: 60.0,
            entry_max_tte_s: 300.0,
            tier_lo: "q95".into(),
            tier_hi: "q98".into(),
            max_cheap_ask: 0.12,
            max_spread: 0.06,
            cb_max_age_ms: 60_000,
            eval_min_ms: 200,
            log_every_s: 60.0,
        }
    }
}

// ───────────────────────────── rule ─────────────────────────────

#[derive(Default)]
struct EvState {
    fired_lo: bool,
    fired_hi: bool,
    last_eval_ns: i64,
}

pub struct TurnRule {
    cfg: TurnCfg,
    model: TurnModel,
    thr_lo: f64,
    thr_hi: f64,
    perp: VecDeque<(i64, f64)>,
    perp_last: f64,
    grid_next: i64,
    grid: VecDeque<f64>,
    c1: VecDeque<f64>,
    cb_mid: f64,
    cb_ts: i64,
    yes: HashMap<String, VecDeque<(i64, f64)>>,
    evs: HashMap<String, EvState>,
    n_eval: u64,
    n_scored: u64,
    n_sig: u64,
    last_stat_ns: i64,
}

impl TurnRule {
    pub fn new(cfg: TurnCfg, model: TurnModel) -> Self {
        let thr_lo = *model.thresholds.get(&cfg.tier_lo).unwrap_or(&1.1);
        let thr_hi = *model.thresholds.get(&cfg.tier_hi).unwrap_or(&1.1);
        tracing::info!(target: "turn", "turnaround shadow up: target={} tiers {}={:.4} {}={:.4} band [{},{}] tte ({},{}]",
            model.target, cfg.tier_lo, thr_lo, cfg.tier_hi, thr_hi, cfg.px_lo, cfg.px_hi, cfg.entry_min_tte_s, cfg.entry_max_tte_s);
        TurnRule {
            cfg, model, thr_lo, thr_hi,
            perp: VecDeque::new(), perp_last: f64::NAN, grid_next: 0, grid: VecDeque::new(), c1: VecDeque::new(),
            cb_mid: f64::NAN, cb_ts: 0, yes: HashMap::new(), evs: HashMap::new(),
            n_eval: 0, n_scored: 0, n_sig: 0, last_stat_ns: 0,
        }
    }

    fn on_perp(&mut self, ts_ns: i64, mid: f64) {
        if !(mid > 0.0) {
            return;
        }
        if self.grid_next == 0 {
            self.grid_next = ts_ns + 1_000_000_000;
            self.perp_last = mid;
        }
        self.grid_fill(ts_ns, false);
        self.perp_last = mid;
        self.perp.push_back((ts_ns, mid));
        prune(&mut self.perp, ts_ns);
    }

    fn grid_fill(&mut self, t_ns: i64, inclusive: bool) {
        let v = self.perp_last;
        while if inclusive { self.grid_next <= t_ns } else { self.grid_next < t_ns } {
            self.grid.push_back(v);
            if self.grid.len() > SIG_WIN + 6 {
                self.grid.pop_front();
            }
            let n = self.grid.len();
            if n >= 6 {
                self.c1.push_back(self.grid[n - 1] - self.grid[n - 6]);
                if self.c1.len() > SIG_WIN {
                    self.c1.pop_front();
                }
            }
            self.grid_next += GRID_NS;
        }
    }

    fn sigma(&self) -> f64 {
        let n = self.c1.len();
        if n < SIG_MIN {
            return SIG_FLOOR;
        }
        let mean = self.c1.iter().sum::<f64>() / n as f64;
        let var = self.c1.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
        var.sqrt().max(SIG_FLOOR)
    }

    fn eval(&mut self, inst: &str, state: &MarketState, now: i64) {
        let Some(t) = state.get(inst) else { return };
        let Some(expiry) = t.expiry_ts_ns else { return };
        let tte_s = (expiry - now) as f64 / 1e9;
        if tte_s <= 0.0 {
            self.evs.remove(inst);
            self.yes.remove(inst);
            return;
        }
        if !(tte_s > self.cfg.entry_min_tte_s && tte_s <= self.cfg.entry_max_tte_s) {
            return;
        }
        let Some(strike) = t.strike else { return };
        let (ybid, yask) = (t.best_bid, t.best_ask);
        let spr = yask - ybid;
        if !(ybid > 0.01 && yask < 0.99 && yask > ybid && spr <= self.cfg.max_spread) {
            return;
        }
        let mid = 0.5 * (ybid + yask);
        if !(mid < self.cfg.px_lo || mid > self.cfg.px_hi) {
            return;
        }
        let lo = mid < 0.5;
        {
            let st = self.evs.entry(inst.to_string()).or_default();
            if now - st.last_eval_ns < self.cfg.eval_min_ms * 1_000_000 {
                return;
            }
            st.last_eval_ns = now;
            if (lo && st.fired_lo) || (!lo && st.fired_hi) {
                return;
            }
        }
        self.n_eval += 1;
        if self.grid_next == 0 || !(self.perp_last > 0.0) {
            return;
        }
        if now - self.cb_ts > self.cfg.cb_max_age_ms * 1_000_000 || !(self.cb_mid > 0.0) {
            return;
        }
        self.grid_fill(now, true);
        let perp_now = self.perp_last;
        let m = if lo { 1.0 } else { -1.0 };
        let p_cheap = if lo { mid } else { 1.0 - mid };
        let cheap_ask = if lo { yask } else { 1.0 - ybid };
        if cheap_ask > self.cfg.max_cheap_ask {
            return;
        }
        let mut x = Vec::with_capacity(self.model.nf);
        let l0 = lg(mid);
        let yring = match self.yes.get(inst) {
            Some(r) => r,
            None => return,
        };
        for &j in &self.model.lags {
            match at(yring, now - GRID_NS * j as i64) {
                Some(v) => x.push(m * (l0 - lg(v))),
                None => return, // history not yet covering the lag window
            }
        }
        for &j in &self.model.lags {
            match at(&self.perp, now - GRID_NS * j as i64) {
                Some(v) => x.push(m * (v - perp_now)),
                None => return,
            }
        }
        let sig = self.sigma();
        let dp = perp_now - strike;
        let dc = self.cb_mid - strike;
        let sc = 150.0 * (tte_s.max(1.0) / 900.0).sqrt();
        x.extend_from_slice(&[m * dp, m * dc, m * dp / sc, m * dc / sc,
                              m * (ndtr(dc / sc) - mid), m * (ndtr(dp / sc) - mid)]);
        x.extend_from_slice(&[tte_s.max(1.0).ln(), sig, 100.0 * spr, lg(p_cheap), mid * (1.0 - mid), if lo { 1.0 } else { 0.0 }]);
        if x.len() != self.model.nf || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        self.n_scored += 1;
        let p = self.model.score(&x);
        if p >= self.thr_lo {
            let tier = if p >= self.thr_hi { "HI" } else { "LO" };
            let st = self.evs.entry(inst.to_string()).or_default();
            if lo {
                st.fired_lo = true;
            } else {
                st.fired_hi = true;
            }
            self.n_sig += 1;
            tracing::info!(
                target: "turn",
                "{} TURN-SIG tier={} p={:.4} side={} cheap_ask={:.3} mid={:.4} tte={:.1} spr={:.3} sig={:.2} dp={:+.1} dc={:+.1} cb_age_ms={}",
                inst, tier, p, if lo { "YES" } else { "NO" }, cheap_ask, mid, tte_s, spr, sig, dp, dc,
                (now - self.cb_ts) / 1_000_000
            );
        }
        if now - self.last_stat_ns >= (self.cfg.log_every_s * 1e9) as i64 {
            if self.last_stat_ns != 0 {
                tracing::info!(target: "turn", "stats evals={} scored={} sigs={} sigma={:.2} tracked={}",
                    self.n_eval, self.n_scored, self.n_sig, self.sigma(), self.evs.len());
            }
            self.last_stat_ns = now;
        }
    }
}

impl Rule for TurnRule {
    fn id(&self) -> &str {
        "turnaround"
    }

    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal> {
        match &ev.payload {
            Payload::Book(b) if b.instrument == self.cfg.reference => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.on_perp(b.recv_ts_ns, 0.5 * (bid + ask));
                    }
                }
            }
            Payload::Book(b) if b.instrument == self.cfg.cb_instrument => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.cb_mid = 0.5 * (bid + ask);
                        self.cb_ts = b.recv_ts_ns;
                    }
                }
            }
            Payload::Trade(t) if t.instrument == self.cfg.cb_instrument => {
                if t.price > 0.0 {
                    self.cb_mid = t.price;
                    self.cb_ts = t.recv_ts_ns;
                }
            }
            Payload::Book(b) if b.instrument.ends_with(".YES") => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        let r = self.yes.entry(b.instrument.clone()).or_default();
                        r.push_back((b.recv_ts_ns, 0.5 * (bid + ask)));
                        prune(r, b.recv_ts_ns);
                    }
                }
                let inst = b.instrument.clone();
                self.eval(&inst, state, b.recv_ts_ns);
            }
            Payload::Meta(m) => {
                use arb_core::model::MarketStatus::*;
                if matches!(m.status, Expired | Resolved) {
                    self.evs.remove(&m.instrument);
                    self.yes.remove(&m.instrument);
                }
            }
            _ => {}
        }
        Vec::new() // SHADOW: never a trade signal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ensemble forward vs the exporter's torch test vectors (raw features -> standardized -> MLPs -> mean σ).
    #[test]
    fn turnaround_score_matches_torch() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/turnaround-up30-rs-btc.json");
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => return, // model not exported in this checkout
        };
        let model = TurnModel::from_json(&text).unwrap();
        let js: serde_json::Value = serde_json::from_str(&text).unwrap();
        for tv in js["test_vectors"].as_array().unwrap() {
            let x: Vec<f64> = tv["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let want = tv["p"].as_f64().unwrap();
            let got = model.score(&x);
            assert!((got - want).abs() < 5e-5, "p {got} vs torch {want}");
        }
    }

    #[test]
    fn ndtr_sane() {
        assert!((ndtr(0.0) - 0.5).abs() < 1e-7);
        assert!((ndtr(1.959964) - 0.975).abs() < 1e-5);
        assert!((ndtr(-1.959964) - 0.025).abs() < 1e-5);
    }
}
