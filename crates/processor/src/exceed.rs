//! ExceedRule — the exceedance classifier (tools/nowcast/train_exceed.py; model JSON `kind: "exceed"`,
//! exported by tools/nowcast/export_exceed_rs.py) as a DIRECTIONAL entry signal.
//!
//!   (p_up, p_dn) = mean_k sigmoid( MLP_k( (x − mu) / sd ) )      s = p_up − p_dn
//!
//! Fire YES when s ≥ cut, NO when s ≤ −cut (inside the configured price region), one fire per
//! episode (re-arm when s·dir ≤ rearm_eps). No fair value, no calibration: the harness twin is
//! `analyze_online.py --model x=<json> --gate none --deltas <cut> --px-tails lo,hi --rearm-eps e`
//! (kind=exceed).
//!
//! x = the STANDARD 43-input nowcast vector rebuilt from the live feeds, in the trainer's order:
//!   hist : lg(mid at lag j) − lg(mid), (perp at lag j − perp now)/σ, j ∈ lags (ticks of 200 ms;
//!          "at lag" = the last update at/before now − 200j ms)
//!   book : single columns among kbs=ln1p(ybid_sz), kas=ln1p(yask_sz), kim=imb, kmi=100·micro,
//!          pbs=ln1p(perp_bid_sz), pas=ln1p(perp_ask_sz) (canonical order)
//!   lake : binance-perp print/depth features on 100 ms bars (tools/nowcast/build_lake.py), rebuilt
//!          from the trader's own `.vol`/`.volb` cumulative trade counters and the @depth@100ms book:
//!          flow_tfi_{1,5,30}s, flow_vol_1s, flow_n_{1,5}s, depth_spread_bps, depth_depth10,
//!          depth_band{5,10,25}, depth_dimb5_{1,5}s, depth_dband10_1s. The offline bar is the one
//!          of (ts − 50 ms), complete; live the window ends at the CURRENT (partial) bar and the
//!          depth stats are the latest snapshot ≤ now (ffill ≤ 20 bars, else 0 — the builder's
//!          NaN→0). Trade counters arrive on ~100–200 ms publish throttles (±1 bar smear).
//!   ctx  : ln(max(tte,1)), σ, 100·spread, lg(mid), mid(1−mid)
//! σ = std (ddof 1) of the 1 s perp move on the 200 ms perp grid over the last 600 grid points
//! (min 200, else 1.25), floored at 1.25 — the training builder's definition.
//!
//! Log streams (data/trader-events.log): `exceed` (per-market state every log_every_s + signals +
//! 60 s stats), `exceedfeat` (the full input vector every feat_log_every_s) — the offline parity
//! check joins them to the harness rows by (ticker, ts).

use std::collections::{HashMap, VecDeque};

use serde::Deserialize;

use arb_core::event::{Event, Payload};
use arb_core::model::{TradeSignal, Trigger};

use crate::rule::Rule;
use crate::state::MarketState;

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
struct ExceedJson {
    kind: String,
    feat: String,
    lags: Vec<usize>,
    #[serde(default)]
    lake: Vec<String>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    members: Vec<MemberJson>,
}

struct Mlp {
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<(usize, usize)>,
}

impl Mlp {
    fn forward(&self, z: &[f64]) -> Vec<f64> {
        let mut h: Vec<f64> = z.to_vec();
        let last = self.dims.len() - 1;
        for l in 0..=last {
            let (in_dim, out_dim) = self.dims[l];
            let mut out = vec![0.0f64; out_dim];
            for (o, slot) in out.iter_mut().enumerate() {
                let row = &self.w[l][o * in_dim..(o + 1) * in_dim];
                let mut acc = self.b[l][o];
                for i in 0..in_dim {
                    acc += row[i] * h[i];
                }
                *slot = if l < last { acc.max(0.0) } else { acc };
            }
            h = out;
        }
        h
    }
}

const BOOK_ORDER: [&str; 6] = ["kbs", "kas", "kim", "kmi", "pbs", "pas"];
const LAKE_SUPPORTED: [&str; 14] = [
    "flow_tfi_1s", "flow_tfi_5s", "flow_tfi_30s", "flow_vol_1s", "flow_n_1s", "flow_n_5s",
    "depth_spread_bps", "depth_depth10", "depth_band5", "depth_band10", "depth_band25",
    "depth_dimb5_1s", "depth_dimb5_5s", "depth_dband10_1s",
];

/// Frozen exceedance classifier: feature spec + standardization + member MLPs (2 outputs each).
pub struct ExceedModel {
    pub lags: Vec<usize>,
    /// single book columns present, in canonical order
    pub book: Vec<String>,
    /// lake features in the model's input order (the bars-file column order)
    pub lake: Vec<String>,
    pub nf: usize,
    mu: Vec<f64>,
    sd: Vec<f64>,
    members: Vec<Mlp>,
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl ExceedModel {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let js: ExceedJson = serde_json::from_str(text)?;
        anyhow::ensure!(js.kind == "exceed", "not an exceed model (kind {:?})", js.kind);
        let groups: Vec<String> = js.feat.split(',').map(|s| s.trim().to_string()).collect();
        anyhow::ensure!(groups.iter().any(|g| g == "hist"), "exceed: the 'hist' group is required");
        for g in &groups {
            anyhow::ensure!(
                g == "hist" || BOOK_ORDER.contains(&g.as_str()) || g.starts_with("lf_"),
                "exceed: unsupported feature group {g}"
            );
        }
        let book: Vec<String> = BOOK_ORDER
            .iter()
            .filter(|c| groups.iter().any(|g| g == *c))
            .map(|s| s.to_string())
            .collect();
        let lf: Vec<String> = groups.iter().filter(|g| g.starts_with("lf_")).map(|g| g[3..].to_string()).collect();
        anyhow::ensure!(
            js.lake.len() == lf.len() && lf.iter().all(|n| js.lake.contains(n)),
            "exceed: json `lake` order list must hold exactly the lf_ groups"
        );
        for n in &js.lake {
            anyhow::ensure!(LAKE_SUPPORTED.contains(&n.as_str()), "exceed: lake feature {n} not reproducible live");
        }
        let nf = js.mu.len();
        let expect = 2 * js.lags.len() + book.len() + js.lake.len() + 5;
        anyhow::ensure!(expect == nf, "exceed: feature count {expect} != mu length {nf}");
        anyhow::ensure!(js.sd.len() == nf, "exceed: mu/sd length mismatch");
        anyhow::ensure!(!js.members.is_empty(), "exceed: no members");
        let mut members = Vec::new();
        for m in &js.members {
            let mut w = Vec::new();
            let mut b = Vec::new();
            let mut dims = Vec::new();
            for l in &m.layers {
                let out_dim = l.w.len();
                anyhow::ensure!(out_dim > 0, "exceed: empty layer");
                let in_dim = l.w[0].len();
                anyhow::ensure!(l.b.len() == out_dim, "exceed: bias dim mismatch");
                let mut flat = Vec::with_capacity(out_dim * in_dim);
                for row in &l.w {
                    anyhow::ensure!(row.len() == in_dim, "exceed: ragged weight row");
                    flat.extend_from_slice(row);
                }
                w.push(flat);
                b.push(l.b.clone());
                dims.push((in_dim, out_dim));
            }
            anyhow::ensure!(dims.first().map(|d| d.0) == Some(nf), "exceed: member input dim != nf");
            anyhow::ensure!(dims.last().map(|d| d.1) == Some(2), "exceed: member output dim != 2");
            members.push(Mlp { w, b, dims });
        }
        Ok(ExceedModel { lags: js.lags, book, lake: js.lake, nf, mu: js.mu, sd: js.sd, members })
    }

    /// (P(up ≥ X), P(down ≥ X)) on the RAW feature vector: z-score, member-mean of the sigmoids
    /// (the harness's kind=exceed averaging).
    pub fn score(&self, x: &[f64]) -> (f64, f64) {
        let z: Vec<f64> = (0..self.nf).map(|i| (x[i] - self.mu[i]) / self.sd[i]).collect();
        let (mut up, mut dn) = (0.0, 0.0);
        for m in &self.members {
            let o = m.forward(&z);
            up += sigmoid(o[0]);
            dn += sigmoid(o[1]);
        }
        let n = self.members.len() as f64;
        (up / n, dn / n)
    }
}

// ───────────────────────────── histories (hist + σ) ─────────────────────────────

#[inline]
fn lg(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

const HIST_KEEP_NS: i64 = 8_000_000_000; // 6 s lookback + margin
const GRID_NS: i64 = 200_000_000;
const SIG_WIN: usize = 600; // 120 s of 200 ms grid points
const SIG_MIN: usize = 200;
const SIG_FLOOR: f64 = 1.25;

/// A 200 ms grid of a piecewise-constant series: value at grid time g = the last update at/before g.
struct Grid {
    next_ns: i64,
    vals: VecDeque<f64>,
    keep: usize,
}

impl Grid {
    fn fill(&mut self, t_ns: i64, v: f64, inclusive: bool) -> usize {
        let mut added = 0;
        while if inclusive { self.next_ns <= t_ns } else { self.next_ns < t_ns } {
            self.vals.push_back(v);
            if self.vals.len() > self.keep {
                self.vals.pop_front();
            }
            self.next_ns += GRID_NS;
            added += 1;
        }
        added
    }
}

fn prune(ring: &mut VecDeque<(i64, f64)>, now: i64) {
    // keep the newest sample older than the horizon so "state at t" stays defined
    while ring.len() >= 2 && ring[1].0 <= now - HIST_KEEP_NS {
        ring.pop_front();
    }
}

/// Last update at/before `t` (None before the first update).
fn at(ring: &VecDeque<(i64, f64)>, t: i64) -> Option<f64> {
    ring.iter().rev().find(|&&(ts, _)| ts <= t).map(|&(_, x)| x)
}

/// Perp (global) + YES (per market) mid histories and the σ window.
pub struct Hist {
    perp: VecDeque<(i64, f64)>,
    perp_last: f64,
    pgrid: Option<Grid>,
    /// 1 s perp move on the grid: pgrid[i] − pgrid[i−5]; last SIG_WIN values
    c1: VecDeque<f64>,
    yes: HashMap<String, VecDeque<(i64, f64)>>,
}

impl Hist {
    pub fn new() -> Self {
        Hist { perp: VecDeque::new(), perp_last: f64::NAN, pgrid: None, c1: VecDeque::new(), yes: HashMap::new() }
    }

    fn pgrid_fill(&mut self, t_ns: i64, inclusive: bool) {
        let v = self.perp_last;
        if let Some(g) = self.pgrid.as_mut() {
            let added = g.fill(t_ns, v, inclusive);
            let n = g.vals.len();
            for k in (n - added)..n {
                if k >= 5 {
                    self.c1.push_back(g.vals[k] - g.vals[k - 5]);
                    if self.c1.len() > SIG_WIN {
                        self.c1.pop_front();
                    }
                }
            }
        }
    }

    pub fn on_perp(&mut self, ts_ns: i64, mid: f64) {
        if !(mid > 0.0) {
            return;
        }
        if self.pgrid.is_none() {
            self.pgrid = Some(Grid { next_ns: ts_ns + 1_000_000_000, vals: VecDeque::new(), keep: SIG_WIN + 6 });
            self.perp_last = mid;
        }
        self.pgrid_fill(ts_ns, false);
        self.perp_last = mid;
        self.perp.push_back((ts_ns, mid));
        prune(&mut self.perp, ts_ns);
    }

    pub fn on_yes(&mut self, inst: &str, ts_ns: i64, mid: f64) {
        if !(mid > 0.0) {
            return;
        }
        let r = self.yes.entry(inst.to_string()).or_default();
        r.push_back((ts_ns, mid));
        prune(r, ts_ns);
    }

    pub fn remove(&mut self, inst: &str) {
        self.yes.remove(inst);
    }

    pub fn sigma(&self) -> f64 {
        let n = self.c1.len();
        if n < SIG_MIN {
            return SIG_FLOOR;
        }
        let mean = self.c1.iter().sum::<f64>() / n as f64;
        let var = self.c1.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
        var.sqrt().max(SIG_FLOOR)
    }

    /// σ + the 2·lags hist block, or None while the histories are incomplete.
    fn hist_block(&mut self, inst: &str, now: i64, lags: &[usize], l0: f64, perp_now: f64, out: &mut Vec<f64>) -> Option<f64> {
        if self.pgrid.is_none() || !(perp_now > 0.0) {
            return None;
        }
        self.pgrid_fill(now, true);
        let sig = self.sigma();
        let y = self.yes.get(inst)?;
        for &j in lags {
            out.push(lg(at(y, now - GRID_NS * j as i64)?) - l0);
        }
        for &j in lags {
            out.push((at(&self.perp, now - GRID_NS * j as i64)? - perp_now) / sig);
        }
        Some(sig)
    }
}

impl Default for Hist {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────── lake bars (100 ms) ─────────────────────────────

const BAR_NS: i64 = 100_000_000;
const LAKE_KEEP: usize = 340; // 30 s window + slack
const FF_BARS: usize = 20; // builder ffill limit (2 s)
const BANDS_BPS: [f64; 3] = [5.0, 10.0, 25.0];
const DEPTH_LEVELS: usize = 60; // builder: top-60 levels per side

#[derive(Clone, Copy, Debug, Default)]
struct DepthStats {
    spread_bps: f64,
    imb5: f64,
    band5: f64,
    band10: f64,
    band25: f64,
    depth10: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bar {
    buy: f64,
    tot: f64,
    cnt: f64,
    depth: Option<DepthStats>,
}

pub struct LakeFeats {
    first_bar: i64,
    bars: VecDeque<Bar>,
    last_tot: Option<f64>,
    last_buy: Option<f64>,
    last_cnt: Option<f64>,
    pub n_depth: u64,
    pub n_vol: u64,
}

fn depth_stats(bids: &[(f64, f64)], asks: &[(f64, f64)]) -> Option<DepthStats> {
    let (bp0, _) = *bids.first()?;
    let (ap0, _) = *asks.first()?;
    if !(bp0 > 0.0 && ap0 > bp0) {
        return None;
    }
    let mid = 0.5 * (bp0 + ap0);
    let b = &bids[..bids.len().min(DEPTH_LEVELS)];
    let a = &asks[..asks.len().min(DEPTH_LEVELS)];
    let sum5 = |lv: &[(f64, f64)]| lv.iter().take(5).map(|&(_, q)| q).sum::<f64>();
    let (b5, a5) = (sum5(b), sum5(a));
    let mut st = DepthStats { spread_bps: 1e4 * (ap0 - bp0) / mid, imb5: (b5 - a5) / (b5 + a5).max(1e-9), ..Default::default() };
    for w in BANDS_BPS {
        let sb: f64 = b.iter().filter(|&&(p, _)| p >= mid * (1.0 - w * 1e-4)).map(|&(_, q)| q).sum();
        let sa: f64 = a.iter().filter(|&&(p, _)| p <= mid * (1.0 + w * 1e-4)).map(|&(_, q)| q).sum();
        let imb = (sb - sa) / (sb + sa).max(1e-9);
        if w == 5.0 {
            st.band5 = imb;
        } else if w == 10.0 {
            st.band10 = imb;
            st.depth10 = (sb + sa).ln_1p();
        } else {
            st.band25 = imb;
        }
    }
    Some(st)
}

impl LakeFeats {
    pub fn new() -> Self {
        LakeFeats { first_bar: 0, bars: VecDeque::new(), last_tot: None, last_buy: None, last_cnt: None, n_depth: 0, n_vol: 0 }
    }

    #[inline]
    fn bar_of(ts_ns: i64) -> i64 {
        ts_ns.div_euclid(BAR_NS)
    }

    /// Index of `bar` in the ring, extending it forward with empty bars (None if the bar is older
    /// than the ring — a late/out-of-order sample is dropped).
    fn advance(&mut self, bar: i64) -> Option<usize> {
        if self.bars.is_empty() {
            self.first_bar = bar;
            self.bars.push_back(Bar::default());
            return Some(0);
        }
        if bar < self.first_bar {
            return None;
        }
        let mut last = self.first_bar + self.bars.len() as i64 - 1;
        while last < bar {
            self.bars.push_back(Bar::default());
            last += 1;
            if self.bars.len() > LAKE_KEEP {
                self.bars.pop_front();
                self.first_bar += 1;
            }
        }
        Some((bar - self.first_bar) as usize)
    }

    pub fn on_depth(&mut self, ts_ns: i64, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        if let Some(st) = depth_stats(bids, asks) {
            if let Some(i) = self.advance(Self::bar_of(ts_ns)) {
                self.bars[i].depth = Some(st);
                self.n_depth += 1;
            }
        }
    }

    fn delta(last: &mut Option<f64>, cum: f64) -> f64 {
        let d = match *last {
            Some(p) if cum >= p => cum - p,
            _ => 0.0, // first sample or a counter reset (collector restart)
        };
        *last = Some(cum);
        d
    }

    /// `.vol` feed: cumulative TOTAL taker volume.
    pub fn on_vol(&mut self, ts_ns: i64, cum_tot: f64) {
        let d = Self::delta(&mut self.last_tot, cum_tot);
        if let Some(i) = self.advance(Self::bar_of(ts_ns)) {
            self.bars[i].tot += d;
            self.n_vol += 1;
        }
    }

    /// `.volb` feed: cumulative BUY volume + cumulative print count.
    pub fn on_volb(&mut self, ts_ns: i64, cum_buy: f64, cum_cnt: f64) {
        let db = Self::delta(&mut self.last_buy, cum_buy);
        let dc = Self::delta(&mut self.last_cnt, cum_cnt);
        if let Some(i) = self.advance(Self::bar_of(ts_ns)) {
            self.bars[i].buy += db;
            self.bars[i].cnt += dc;
        }
    }

    fn depth_at(&self, idx: usize) -> Option<DepthStats> {
        (idx.saturating_sub(FF_BARS)..=idx).rev().find_map(|k| self.bars[k].depth)
    }

    /// (buy, sell, count) over the `w` bars ending at `idx` (inclusive).
    fn sums(&self, idx: usize, w: usize) -> (f64, f64, f64) {
        let lo = idx + 1 - w.min(idx + 1);
        let (mut b, mut t, mut c) = (0.0, 0.0, 0.0);
        for k in lo..=idx {
            let br = &self.bars[k];
            b += br.buy;
            t += br.tot;
            c += br.cnt;
        }
        (b, (t - b).max(0.0), c)
    }

    /// Push the named lake features at `now` (builder semantics, NaN → 0). false if `now` is
    /// older than the ring or a name is unknown.
    pub fn features(&mut self, now_ns: i64, names: &[String], out: &mut Vec<f64>) -> bool {
        let Some(idx) = self.advance(Self::bar_of(now_ns)) else { return false };
        let tfi = |w: usize| {
            let (b, s, _) = self.sums(idx, w);
            (b - s) / (b + s).max(1e-6)
        };
        let d0 = self.depth_at(idx);
        let d10 = if idx >= 10 { self.depth_at(idx - 10) } else { None };
        let d50 = if idx >= 50 { self.depth_at(idx - 50) } else { None };
        let g = |d: Option<DepthStats>, f: fn(&DepthStats) -> f64| d.map(|d| f(&d)).unwrap_or(0.0);
        let dd = |a: Option<DepthStats>, b: Option<DepthStats>, f: fn(&DepthStats) -> f64| match (a, b) {
            (Some(a), Some(b)) => f(&a) - f(&b),
            _ => 0.0,
        };
        for n in names {
            let v = match n.as_str() {
                "flow_tfi_1s" => tfi(10),
                "flow_tfi_5s" => tfi(50),
                "flow_tfi_30s" => tfi(300),
                "flow_vol_1s" => {
                    let (b, s, _) = self.sums(idx, 10);
                    (b + s).ln_1p()
                }
                "flow_n_1s" => self.sums(idx, 10).2,
                "flow_n_5s" => self.sums(idx, 50).2,
                "depth_spread_bps" => g(d0, |d| d.spread_bps),
                "depth_depth10" => g(d0, |d| d.depth10),
                "depth_band5" => g(d0, |d| d.band5),
                "depth_band10" => g(d0, |d| d.band10),
                "depth_band25" => g(d0, |d| d.band25),
                "depth_dimb5_1s" => dd(d0, d10, |d| d.imb5),
                "depth_dimb5_5s" => dd(d0, d50, |d| d.imb5),
                "depth_dband10_1s" => dd(d0, d10, |d| d.band10),
                _ => return false,
            };
            out.push(v);
        }
        true
    }
}

impl Default for LakeFeats {
    fn default() -> Self {
        Self::new()
    }
}

// ───────────────────────────── rule ─────────────────────────────

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct ExceedCfg {
    pub model_path: String,
    /// Reference perp instrument (bookTicker: mid + top sizes).
    pub reference: String,
    /// Full-depth perp instrument (`binance_depth` collector) for the lake depth features.
    pub depth_instrument: String,
    /// |s| entry threshold (the harness `--deltas`).
    pub cut: f64,
    /// Re-arm when s·dir ≤ rearm_eps after a fire (the harness `--rearm-eps`).
    pub rearm_eps: f64,
    /// Price region: "tails" (mid < px_lo or mid > px_hi — the harness `--px-tails`), "band"
    /// (px_lo ≤ mid ≤ px_hi), anything else = all prices.
    pub px_mode: String,
    pub px_lo: f64,
    pub px_hi: f64,
    pub entry_min_tte_s: f64,
    pub entry_max_tte_s: f64,
    pub max_entries_per_event: u32,
    pub stale_ms: i64,
    pub max_spread: f64,
    pub ref_max_age_ms: i64,
    pub hold_ms: u64,
    pub ttl_ms: u64,
    /// Minimum spacing between two evaluations of the same market (the harness scans a 50 ms grid).
    pub eval_min_ms: i64,
    pub log_every_s: f64,
    pub feat_log_every_s: f64,
}

impl Default for ExceedCfg {
    fn default() -> Self {
        ExceedCfg {
            model_path: "models/exceed-1s-x1-rs-btc.json".into(),
            reference: "binance.usdt_perp.BTCUSDT".into(),
            depth_instrument: "binance.usdt_perp.BTCUSDT.depth".into(),
            cut: 0.48,
            rearm_eps: 0.2,
            px_mode: "tails".into(),
            px_lo: 0.10,
            px_hi: 0.90,
            entry_min_tte_s: 60.0,
            entry_max_tte_s: 300.0,
            max_entries_per_event: 255,
            stale_ms: 1500,
            max_spread: 0.15,
            ref_max_age_ms: 5000,
            hold_ms: 0,
            ttl_ms: 500,
            eval_min_ms: 50,
            log_every_s: 1.0,
            feat_log_every_s: 10.0,
        }
    }
}

#[derive(Default)]
struct EvState {
    armed: bool,
    disarm_dir: f64,
    entries: u32,
    last_eval_ns: i64,
    last_log_ns: i64,
    last_feat_log_ns: i64,
}

pub struct ExceedRule {
    cfg: ExceedCfg,
    model: ExceedModel,
    hist: Hist,
    lake: LakeFeats,
    evs: HashMap<String, EvState>,
    vol_inst: String,
    volb_inst: String,
    n_eval: u64,
    n_invalid: u64,
    n_sig: u64,
    last_stat_ns: i64,
}

impl ExceedRule {
    pub fn new(cfg: ExceedCfg, model: ExceedModel) -> Self {
        let vol_inst = format!("{}.vol", cfg.reference);
        let volb_inst = format!("{}.volb", cfg.reference);
        ExceedRule {
            cfg,
            model,
            hist: Hist::new(),
            lake: LakeFeats::new(),
            evs: HashMap::new(),
            vol_inst,
            volb_inst,
            n_eval: 0,
            n_invalid: 0,
            n_sig: 0,
            last_stat_ns: 0,
        }
    }

    /// The RAW 43-vector at `now`, or None while a history is incomplete (the harness drops such rows).
    #[allow(clippy::too_many_arguments)]
    fn features(
        &mut self,
        inst: &str,
        now: i64,
        tte_s: f64,
        ybid: f64,
        yask: f64,
        ybs: f64,
        yas: f64,
        perp_now: f64,
        pbs: f64,
        pas: f64,
    ) -> Option<Vec<f64>> {
        let mid = 0.5 * (ybid + yask);
        let l0 = lg(mid);
        let mut x = Vec::with_capacity(self.model.nf);
        let sig = self.hist.hist_block(inst, now, &self.model.lags, l0, perp_now, &mut x)?;
        let ssum = (ybs + yas).max(1e-6);
        for c in &self.model.book {
            x.push(match c.as_str() {
                "kbs" => ybs.ln_1p(),
                "kas" => yas.ln_1p(),
                "kim" => (ybs - yas) / ssum,
                "kmi" => 100.0 * ((ybid * yas + yask * ybs) / ssum - mid),
                "pbs" => pbs.ln_1p(),
                "pas" => pas.ln_1p(),
                _ => return None,
            });
        }
        if !self.lake.features(now, &self.model.lake, &mut x) {
            return None;
        }
        x.extend_from_slice(&[tte_s.max(1.0).ln(), sig, 100.0 * (yask - ybid), l0, mid * (1.0 - mid)]);
        if x.len() != self.model.nf || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(x)
    }

    fn in_region(&self, mid: f64) -> bool {
        match self.cfg.px_mode.as_str() {
            "tails" => mid < self.cfg.px_lo || mid > self.cfg.px_hi,
            "band" => mid >= self.cfg.px_lo && mid <= self.cfg.px_hi,
            _ => true,
        }
    }

    fn eval(&mut self, inst: &str, state: &MarketState, now: i64) -> Option<TradeSignal> {
        let t = state.get(inst)?;
        let r = state.get(&self.cfg.reference)?;
        let expiry = t.expiry_ts_ns?;
        let tte_s = (expiry - now) as f64 / 1e9;
        if tte_s <= 0.0 {
            self.evs.remove(inst);
            self.hist.remove(inst);
            return None;
        }
        if !self.evs.contains_key(inst) {
            return None; // not a tracked event (no strike/meta yet)
        }
        if !(tte_s > self.cfg.entry_min_tte_s && tte_s <= self.cfg.entry_max_tte_s) {
            return None;
        }
        let perp_now = r.mid;
        if !(perp_now > 0.0) || now - r.recv_ts_ns > self.cfg.ref_max_age_ms * 1_000_000 {
            return None;
        }
        let (ybid, yask) = (t.best_bid, t.best_ask);
        if !(ybid > 0.0 && yask > ybid && yask - ybid <= self.cfg.max_spread) || now - t.recv_ts_ns > self.cfg.stale_ms * 1_000_000 {
            return None;
        }
        let (ybs, yas, pbs, pas) = (t.bid_sz, t.ask_sz, r.bid_sz, r.ask_sz);
        let eval_min_ns = self.cfg.eval_min_ms * 1_000_000;
        {
            let st = self.evs.get_mut(inst)?;
            if now - st.last_eval_ns < eval_min_ns {
                return None;
            }
            st.last_eval_ns = now;
        }
        self.n_eval += 1;
        let mid = 0.5 * (ybid + yask);
        let x = match self.features(inst, now, tte_s, ybid, yask, ybs, yas, perp_now, pbs, pas) {
            Some(x) => x,
            None => {
                self.n_invalid += 1;
                return None;
            }
        };
        let (pup, pdn) = self.model.score(&x);
        let s = pup - pdn;
        let sig = if self.in_region(mid) { s } else { 0.0 }; // harness: signal zeroed outside the region

        // periodic stats (60 s): evaluation/validity counts, σ, lake feed counters
        if now - self.last_stat_ns >= 60_000_000_000 {
            if self.last_stat_ns != 0 {
                tracing::info!(
                    target: "exceed",
                    "stats evals={} invalid={} signals={} sigma={:.3} lake_depth={} lake_vol={} tracked={}",
                    self.n_eval, self.n_invalid, self.n_sig, self.hist.sigma(), self.lake.n_depth, self.lake.n_vol, self.evs.len()
                );
            }
            self.last_stat_ns = now;
        }
        let cfg_log_ns = (self.cfg.log_every_s * 1e9) as i64;
        let cfg_feat_ns = (self.cfg.feat_log_every_s * 1e9) as i64;
        let cut = self.cfg.cut;
        let rearm_eps = self.cfg.rearm_eps;
        let max_entries = self.cfg.max_entries_per_event;
        let st = self.evs.get_mut(inst)?;
        if now - st.last_log_ns >= cfg_log_ns {
            st.last_log_ns = now;
            tracing::info!(
                target: "exceed",
                "{} ts={} tte={:.1} mid={:.4} s={:+.4} pup={:.4} pdn={:.4} armed={} px={:.2}",
                inst, now / 1_000_000, tte_s, mid, s, pup, pdn, st.armed as u8, perp_now
            );
        }
        if now - st.last_feat_log_ns >= cfg_feat_ns {
            st.last_feat_log_ns = now;
            let xs: Vec<String> = x.iter().map(|v| format!("{v:.5}")).collect();
            tracing::info!(target: "exceedfeat", "{} ts={} tte={:.1} x={}", inst, now / 1_000_000, tte_s, xs.join(","));
        }

        if !st.armed {
            if sig * st.disarm_dir <= rearm_eps {
                st.armed = true;
                tracing::info!(target: "exceed", "{} RE-ARM tte={:.1} s={:+.4} mid={:.4}", inst, tte_s, sig, mid);
            }
            return None;
        }
        if st.entries >= max_entries || sig.abs() < cut {
            return None;
        }
        st.entries += 1;
        st.armed = false;
        st.disarm_dir = if sig > 0.0 { 1.0 } else { -1.0 };
        let entry_no = st.entries;
        self.n_sig += 1;
        let up = sig > 0.0;
        Some(TradeSignal {
            strategy: "exceed".into(),
            ts_ns: now,
            reason: format!(
                "s={sig:+.4} pup={pup:.4} pdn={pdn:.4} mid={mid:.4} tte={tte_s:.1}s px={perp_now:.2} entry#{entry_no}"
            ),
            reference: self.cfg.reference.clone(),
            target: inst.to_string(),
            direction: if up { 1 } else { -1 },
            trigger: Trigger {
                move_bps: 100.0 * sig, // the direction score, in "cents" of probability
                window_ms: 1000,
                yes_price: if up { yask } else { 1.0 - ybid },
                target_move_c: f64::NAN,
            },
            hold_ms: self.cfg.hold_ms,
            ttl_ms: self.cfg.ttl_ms,
        })
    }
}

impl Rule for ExceedRule {
    fn id(&self) -> &str {
        "exceed"
    }

    fn on_event(&mut self, ev: &Event, state: &MarketState) -> Vec<TradeSignal> {
        match &ev.payload {
            Payload::Book(b) if b.instrument == self.cfg.reference => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.hist.on_perp(b.recv_ts_ns, 0.5 * (bid + ask));
                    }
                }
                let insts: Vec<String> = self.evs.keys().cloned().collect();
                let now = b.recv_ts_ns;
                insts.iter().filter_map(|i| self.eval(i, state, now)).collect()
            }
            Payload::Book(b) if b.instrument == self.cfg.depth_instrument => {
                self.lake.on_depth(b.recv_ts_ns, &b.bids, &b.asks);
                Vec::new()
            }
            Payload::Trade(t) if t.instrument == self.vol_inst => {
                self.lake.on_vol(t.recv_ts_ns, t.qty);
                Vec::new()
            }
            Payload::Trade(t) if t.instrument == self.volb_inst => {
                self.lake.on_volb(t.recv_ts_ns, t.qty, t.price);
                Vec::new()
            }
            Payload::Book(b) if b.instrument.ends_with(".YES") => {
                if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                    if bid > 0.0 && ask > 0.0 {
                        self.hist.on_yes(&b.instrument, b.recv_ts_ns, 0.5 * (bid + ask));
                    }
                }
                let inst = b.instrument.clone();
                self.eval(&inst, state, b.recv_ts_ns).into_iter().collect()
            }
            Payload::Meta(m) => {
                use arb_core::model::MarketStatus::*;
                if matches!(m.status, Expired | Resolved) {
                    self.evs.remove(&m.instrument);
                    self.hist.remove(&m.instrument);
                } else if m.strike.is_some() && !self.evs.contains_key(&m.instrument) {
                    self.evs.insert(m.instrument.clone(), EvState { armed: true, ..Default::default() });
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Member forward + sigmoid mean vs the exporter's torch test vectors (models/exceed-*-rs-btc.json
    /// carries `test_vectors: [{x, p_up, p_dn}]`).
    #[test]
    fn exceed_score_matches_torch() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/exceed-1s-x1-rs-btc.json");
        let Ok(text) = std::fs::read_to_string(path) else { return }; // model not exported yet
        let js: serde_json::Value = serde_json::from_str(&text).unwrap();
        let m = ExceedModel::from_json(&text).unwrap();
        assert_eq!(m.nf, 43);
        let mut n = 0;
        for tv in js["test_vectors"].as_array().unwrap() {
            let x: Vec<f64> = tv["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let (up, dn) = m.score(&x);
            let (wu, wd) = (tv["p_up"].as_f64().unwrap(), tv["p_dn"].as_f64().unwrap());
            assert!((up - wu).abs() < 1e-5 && (dn - wd).abs() < 1e-5, "({up},{dn}) vs torch ({wu},{wd})");
            n += 1;
        }
        assert!(n > 0);
    }

    #[test]
    fn hist_lags_and_sigma() {
        let mut h = Hist::new();
        let t0 = 1_000_000_000_000i64;
        // perp: 100 at t0, 110 at t0+3s ; YES: 0.50 at t0, 0.60 at t0+2s
        h.on_perp(t0, 100.0);
        h.on_yes("k", t0, 0.50);
        h.on_yes("k", t0 + 2_000_000_000, 0.60);
        h.on_perp(t0 + 3_000_000_000, 110.0);
        let now = t0 + 4_000_000_000;
        let mut x = Vec::new();
        let sig = h.hist_block("k", now, &[5, 10, 15], lg(0.60), 110.0, &mut x).unwrap();
        assert_eq!(sig, SIG_FLOOR); // < SIG_MIN grid points
        // mid lags: 1s ago = 0.60, 2s ago = 0.60 (at t0+2s exactly), 3s ago = 0.50
        assert!((x[0] - 0.0).abs() < 1e-12);
        assert!((x[1] - 0.0).abs() < 1e-12);
        assert!((x[2] - (lg(0.50) - lg(0.60))).abs() < 1e-12);
        // perp lags: 1s ago = 110 (t0+3s), 2s ago = 100, 3s ago = 100 → (100-110)/1.25 = -8
        assert!((x[3] - 0.0).abs() < 1e-12);
        assert!((x[4] + 8.0).abs() < 1e-12);
        assert!((x[5] + 8.0).abs() < 1e-12);
        // lag beyond the first update → None
        assert!(h.hist_block("k", now, &[30], lg(0.60), 110.0, &mut Vec::new()).is_none());
    }

    #[test]
    fn lake_bars_flow_and_depth() {
        let mut l = LakeFeats::new();
        let t0 = 1_700_000_000_000_000_000i64; // bar-aligned (multiple of 100 ms)
        // trades: cumulative counters published every 100 ms; each bar 2 BTC total, 1.5 buy, 3 prints
        l.on_vol(t0, 0.0);
        l.on_volb(t0, 0.0, 0.0);
        for i in 1..=12 {
            let t = t0 + i * BAR_NS;
            l.on_vol(t, 2.0 * i as f64);
            l.on_volb(t, 1.5 * i as f64, 3.0 * i as f64);
        }
        // depth: mid 100, spread 0.02 (2 bps), levels 0.011 apart (4 per side within 5 bps, 9 within 10 bps),
        // bids twice as heavy as asks
        let bids: Vec<(f64, f64)> = (0..10).map(|k| (99.99 - 0.011 * k as f64, 2.0)).collect();
        let asks: Vec<(f64, f64)> = (0..10).map(|k| (100.01 + 0.011 * k as f64, 1.0)).collect();
        l.on_depth(t0 + 12 * BAR_NS, &bids, &asks);
        let names: Vec<String> = ["flow_tfi_1s", "flow_vol_1s", "flow_n_1s", "flow_n_5s", "depth_spread_bps", "depth_imb5_x", "depth_band5", "depth_depth10", "depth_dimb5_1s"]
            .iter().map(|s| s.to_string()).collect();
        let mut x = Vec::new();
        // unknown name → false
        assert!(!l.features(t0 + 12 * BAR_NS, &names, &mut x));
        let names: Vec<String> = ["flow_tfi_1s", "flow_vol_1s", "flow_n_1s", "flow_n_5s", "depth_spread_bps", "depth_band5", "depth_depth10", "depth_dimb5_1s", "depth_band10"]
            .iter().map(|s| s.to_string()).collect();
        let mut x = Vec::new();
        assert!(l.features(t0 + 12 * BAR_NS + 30_000_000, &names, &mut x));
        // last 10 bars (bars 3..=12): buy 15, sell 5 → tfi 0.5; vol ln1p(20); n 30; n_5s (50 bars) = all 12 → 36
        assert!((x[0] - 0.5).abs() < 1e-9, "tfi {}", x[0]);
        assert!((x[1] - 20f64.ln_1p()).abs() < 1e-9);
        assert!((x[2] - 30.0).abs() < 1e-9);
        assert!((x[3] - 36.0).abs() < 1e-9, "n_5s {}", x[3]);
        assert!((x[4] - 2.0).abs() < 1e-6, "spread bps {}", x[4]);
        // band5: bids ≥ 99.95 → 4 levels (99.99..99.957) = 8; asks ≤ 100.05 → 4 levels = 4 → (8-4)/12
        assert!((x[5] - 4.0 / 12.0).abs() < 1e-9, "band5 {}", x[5]);
        // depth10: bids ≥ 99.90 → 9 levels = 18; asks ≤ 100.10 → 9 levels = 9 → ln1p(27)
        assert!((x[6] - 27f64.ln_1p()).abs() < 1e-9, "depth10 {}", x[6]);
        assert_eq!(x[7], 0.0); // no depth 10 bars ago → 0 (builder NaN → 0)
        assert!((x[8] - 9.0 / 27.0).abs() < 1e-9, "band10 {}", x[8]);
        // ffill: 15 bars later the same depth stats still apply; 25 bars later they are gone (→ 0)
        let mut x = Vec::new();
        assert!(l.features(t0 + 27 * BAR_NS, &names, &mut x));
        assert!((x[4] - 2.0).abs() < 1e-6);
        let mut x = Vec::new();
        assert!(l.features(t0 + 37 * BAR_NS, &names, &mut x));
        assert_eq!(x[4], 0.0);
    }
}
