//! ObFeats — live reconstruction of the OB_DIST_MODEL `flow_vol` features
//! (E:/crypto/collector, doc/OB_DIST_MODEL.md) on 100ms bars, from the
//! trader's own feeds: perp bookTicker (top-of-book transitions → CKS OFI,
//! upd, mid), the @depth stream (top-5 queue mass → ofl5) and the signed
//! cumulative trade feed `.volb` (tfi/tv/nt).
//!
//! 18 of the 19 mirror features — `big_tr` (max trade in bar) is dropped:
//! it cannot be cumulative-encoded through a conflating bus.
//!
//! delta_min extension (2026-08-19, collector doc §9): `ob_dofi` (+EWMAs) is
//! the CKS best-quote OFI computed from the DEPTH-DIFF book state (the same
//! Binance @depth@100ms source the remediated lake `d_ofi` is built from,
//! same increment formula as `ob_delta_feats.py`), and `ob_bigtr` is a
//! best-effort max-trade proxy: the largest per-publish cumulative-volume
//! delta in the bar (= the true max when one trade dominates a 100ms window;
//! an overestimate when several conflate). Appended AFTER the original 18 so
//! existing consumers/parity joins are unaffected.
//!
//! Parity caveats vs the mirror-built dataset (documented, deliberate):
//! - `upd` counts bookTicker EVENTS (mirror: ~7Hz l2 snapshots) → different
//!   scale; any consumer must be (re)normalized on THESE columns.
//! - OFI over event-cadence transitions ≈ but ≠ snapshot-diff OFI.
//! - trade deltas smear ±1 bar (the .volb feed is 100ms-throttled).
//! Purpose: accumulate live-cadence columns for the parity check / retrain —
//! not to feed the mirror-trained model blind.

use std::collections::VecDeque;

const BAR_NS: i64 = 100_000_000; // 100ms
const STALE_BARS: i64 = 20; // ff at most 2s, as the mirror builder
const HL_BARS: [f64; 3] = [20.0, 100.0, 600.0]; // EWMA half-lives 2s/10s/60s

/// §11 windows, in BARS (100ms): 1s / 10s / 60s. Must match the trainer's
/// `STD` list in `scripts/ob_dist_w.py`.
pub const W_WINDOWS: [(usize, &str); 3] = [(10, "1s"), (100, "10s"), (600, "60s")];
/// Large-print threshold for `i_big`, in BTC. Must match `ob_wfeats.py::BIG_BTC`.
pub const BIG_BTC: f64 = 1.0;
/// Counter order inside the ring: ofi, r2, absm, qupd, vol, buy, cnt, big.
const C_OFI: usize = 0;
const C_R2: usize = 1;
const C_ABSM: usize = 2;
const C_QUPD: usize = 3;
const C_VOL: usize = 4;
const C_BUY: usize = 5;
const C_CNT: usize = 6;
const C_BIG: usize = 7;

/// Design-B (§11.6) feature names, in the trainer's exact emission order:
/// for each window, then `spread_bps` last.
/// §11.10 `spnorm` reference window, in BARS: 30 min. Must match
/// `ob_dist_w.py::REF_SP`. This is the B4 warmup — 30 minutes after a
/// restart before any logit publishes (design B needed only 60s).
pub const REF_SP_BARS: usize = 18000;
/// BTCUSDT tick size, for expressing the spread in ticks (scale-free).
pub const TICK_BTC: f64 = 0.1;

/// Design-B4 (§11.10b) feature names. Identical to `OB_W_FEATS` except the
/// trailing point-in-time `spread_bps` is replaced by the normalised
/// `spnorm` = log(spread_ticks / 30-min mean spread).
pub const OB_W4_FEATS: [&str; 22] = [
    "tfifrac_1s", "ofin_1s", "retstd_1s", "ltv_1s", "lnt_1s", "bigfrac_1s", "lqupd_1s",
    "tfifrac_10s", "ofin_10s", "retstd_10s", "ltv_10s", "lnt_10s", "bigfrac_10s", "lqupd_10s",
    "tfifrac_60s", "ofin_60s", "retstd_60s", "ltv_60s", "lnt_60s", "bigfrac_60s", "lqupd_60s",
    "spnorm",
];

pub const OB_W_FEATS: [&str; 22] = [
    "tfifrac_1s", "ofin_1s", "retstd_1s", "ltv_1s", "lnt_1s", "bigfrac_1s", "lqupd_1s",
    "tfifrac_10s", "ofin_10s", "retstd_10s", "ltv_10s", "lnt_10s", "bigfrac_10s", "lqupd_10s",
    "tfifrac_60s", "ofin_60s", "retstd_60s", "ltv_60s", "lnt_60s", "bigfrac_60s", "lqupd_60s",
    "spread_bps",
];

/// Number of logit bins the OB distribution model emits.
pub const OB_NLOGIT: usize = 11;

/// Column names for the published OB logits (sampler header + downstream).
pub const OB_LOGIT_COLS: [&str; OB_NLOGIT] = [
    "oblg0", "oblg1", "oblg2", "oblg3", "oblg4", "oblg5",
    "oblg6", "oblg7", "oblg8", "oblg9", "oblg10",
];

#[derive(serde::Deserialize)]
struct ObLayerJson {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
    act: String,
}

#[derive(serde::Deserialize)]
struct ObNetJson {
    /// LIVE column names, in the order the net consumes them. Each must be a
    /// member of `OB_FEATS`; the loader resolves them to indices once.
    live_cols: Vec<String>,
    #[serde(rename = "K")]
    k: usize,
    #[serde(rename = "T")]
    t: f64,
    lo: Vec<f64>,
    hi: Vec<f64>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    layers: Vec<ObLayerJson>,
}

/// The frozen order-book distribution model: 17 delta_min features -> K
/// temperature-scaled logits. Mirrors `scripts/ob_gen_logits.py::fwd`:
/// `z = (clip(x, lo, hi) - mu) / sd`, two ReLU layers, a linear head, then
/// divide by T (so `softmax(stored)` is the calibrated distribution).
/// Which feature space a net consumes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ObNetKind {
    /// legacy bar-bucketed `OB_FEATS` columns (pre-§11)
    Legacy,
    /// §11.10b B4 window features, consumed in `OB_W4_FEATS` order
    Window4,
    /// §11 window features, consumed in `OB_W_FEATS` order
    Window,
}

pub struct ObNet {
    pub kind: ObNetKind,
    idx: Vec<usize>, // OB_FEATS index per net input (Legacy only)
    lo: Vec<f64>,
    hi: Vec<f64>,
    mu: Vec<f64>,
    sd: Vec<f64>,
    w: Vec<Vec<f64>>, // row-major per layer
    b: Vec<Vec<f64>>,
    dims: Vec<(usize, usize)>,
    relu: Vec<bool>,
    t: f64,
    pub k: usize,
}

impl ObNet {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let js: ObNetJson = serde_json::from_str(text)?;
        anyhow::ensure!(js.k == OB_NLOGIT, "net K={} but OB_NLOGIT={}", js.k, OB_NLOGIT);
        let n_in = js.live_cols.len();
        anyhow::ensure!(
            js.lo.len() == n_in && js.hi.len() == n_in && js.mu.len() == n_in && js.sd.len() == n_in,
            "lo/hi/mu/sd must all be {n_in} long"
        );
        // Window nets declare exactly the OB_W_FEATS names, in order.
        let matches = |names: &[&str]| {
            js.live_cols.len() == names.len()
                && js.live_cols.iter().zip(names.iter()).all(|(a, b)| a == b)
        };
        let kind = if matches(&OB_W4_FEATS) {
            ObNetKind::Window4
        } else if matches(&OB_W_FEATS) {
            ObNetKind::Window
        } else {
            ObNetKind::Legacy
        };
        let mut idx = Vec::with_capacity(n_in);
        if kind == ObNetKind::Legacy {
            for c in &js.live_cols {
                let i = OB_FEATS.iter().position(|f| f == c).ok_or_else(|| {
                    anyhow::anyhow!(
                        "net input {c:?} is neither an OB_FEATS column nor a valid                          OB_W_FEATS vector (window nets must declare all {} names in order)",
                        OB_W_FEATS.len())
                })?;
                idx.push(i);
            }
        }
        let (mut w, mut b, mut dims, mut relu) = (vec![], vec![], vec![], vec![]);
        for l in &js.layers {
            let out_dim = l.w.len();
            let in_dim = l.w[0].len();
            anyhow::ensure!(l.b.len() == out_dim, "bias dim mismatch");
            let mut flat = Vec::with_capacity(out_dim * in_dim);
            for row in &l.w {
                anyhow::ensure!(row.len() == in_dim, "ragged weight row");
                flat.extend_from_slice(row);
            }
            w.push(flat);
            b.push(l.b.clone());
            dims.push((in_dim, out_dim));
            relu.push(l.act == "relu");
        }
        anyhow::ensure!(dims[0].0 == n_in, "layer0 in_dim != {n_in}");
        anyhow::ensure!(dims[dims.len() - 1].1 == js.k, "head out_dim != K");
        Ok(ObNet { kind, idx, lo: js.lo, hi: js.hi, mu: js.mu, sd: js.sd, w, b, dims, relu, t: js.t, k: js.k })
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Logits for one CLOSED bar's feature vector. `None` if any input the net
    /// needs is NaN (warmup / missing feed) — never publish a partial vector.
    pub fn logits(&self, feats: &[f64; OB_FEATS.len()]) -> Option<Vec<f64>> {
        debug_assert_eq!(self.kind, ObNetKind::Legacy);
        let mut raw = Vec::with_capacity(self.idx.len());
        for &i in self.idx.iter() {
            raw.push(feats[i]);
        }
        self.forward(&raw)
    }

    /// §11 window path: the vector is already in the net's own order.
    pub fn logits_window(&self, x: &[f64]) -> Option<Vec<f64>> {
        debug_assert!(matches!(self.kind, ObNetKind::Window | ObNetKind::Window4));
        self.forward(x)
    }

    fn forward(&self, raw: &[f64]) -> Option<Vec<f64>> {
        let mut x = Vec::with_capacity(raw.len());
        for (j, &v) in raw.iter().enumerate() {
            if !v.is_finite() {
                return None;
            }
            let c = v.clamp(self.lo[j], self.hi[j]);
            x.push((c - self.mu[j]) / self.sd[j]);
        }
        for (l, (in_dim, out_dim)) in self.dims.iter().enumerate() {
            let mut y = vec![0.0f64; *out_dim];
            for o in 0..*out_dim {
                let mut acc = self.b[l][o];
                let row = &self.w[l][o * in_dim..(o + 1) * in_dim];
                for i in 0..*in_dim {
                    acc += row[i] * x[i];
                }
                y[o] = if self.relu[l] { acc.max(0.0) } else { acc };
            }
            x = y;
        }
        for v in x.iter_mut() {
            *v /= self.t;
        }
        Some(x)
    }
}

pub const OB_FEATS: [&str; 23] = [
    "ob_ofi", "ob_ofi_e2", "ob_ofi_e10", "ob_ofi_e60", "ob_ofl5",
    "ob_tfi", "ob_tfi_e2", "ob_tfi_e10", "ob_tfi_e60", "ob_tv_e10",
    "ob_nt", "ob_upd", "ob_ret1", "ob_ret10", "ob_ret100", "ob_ret600",
    "ob_rv100", "ob_tslc",
    // delta_min extension: event-cadence OFI from the depth-diff book + big_tr proxy
    "ob_dofi", "ob_dofi_e2", "ob_dofi_e10", "ob_dofi_e60", "ob_bigtr",
];

pub struct ObFeats {
    // previous top-of-book (for CKS OFI transitions)
    bb: f64,
    ba: f64,
    bq: f64,
    aq: f64,
    have_tob: bool,
    // previous top-5 queue sums (for ofl5)
    b5: f64,
    a5: f64,
    have_d5: bool,
    // previous DEPTH-book best quotes (for d_ofi — the delta_min OFI)
    dbb: f64,
    dba: f64,
    dbq: f64,
    daq: f64,
    have_dbest: bool,
    // latest cumulative trade counters (.vol total, .volb buy/count)
    cum_vol: f64,
    cum_buy: f64,
    cum_cnt: f64,
    have_tr: bool,
    // cumulative counters at the last bar close
    bar_vol0: f64,
    bar_buy0: f64,
    bar_cnt0: f64,
    // current bar accumulators
    cur_bar: i64,
    ofi_acc: f64,
    ofl5_acc: f64,
    upd_acc: f64,
    dofi_acc: f64,
    // big_tr proxy: max per-publish cum-volume delta this bar
    last_vol_evt: f64,
    bigtr_acc: f64,
    stale_bars: i64,
    // EWMA state
    ofi_e: [f64; 3],
    tfi_e: [f64; 3],
    dofi_e: [f64; 3],
    tv_e10: f64,
    // bar-close log-mid ring (601 = enough for ret_600) and rv ring (100 r1²)
    lm_ring: VecDeque<f64>,
    r2_ring: VecDeque<f64>,
    r2_sum: f64,
    tslc: f64,
    last_mid: f64,
    out: [f64; 23],
    ready: bool,
    // ---- §11 WINDOW features (the remediated definition) ----
    // Eight ADDITIVE per-bar increments, ring-buffered over the longest
    // window. Features are `cum(t) - cum(t-W)` = a plain sum over the ring,
    // which is what makes them reproducible by any consumer at any cadence
    // (OB_DIST_MODEL §10.7 class A). `i_big` is a large-print VOLUME counter
    // (>= BIG_BTC), not a max — that is the class-C -> class-A conversion.
    w_ring: std::collections::VecDeque<[f64; 8]>,
    /// running sums per window, indexed [w][counter]
    w_sum: [[f64; 8]; W_WINDOWS.len()],
    /// log-mid at each bar edge, for `ret_W = logmid(t) - logmid(t-W)`
    w_logmid: std::collections::VecDeque<f64>,
    /// per-bar accumulators (reset at each close)
    wi_r2: f64,
    wi_absm: f64,
    wi_qupd: f64,
    wi_big: f64,
    /// last mid seen on a quote change (for the between-change return)
    w_last_lm: f64,
    /// bars since the last best-quote change (point-in-time state)
    w_quote_age: f64,
    /// spread in bps at the bar edge
    w_spread_bps: f64,
    // ---- §11.10 B4: time-weighted spread integral (ticks x seconds) ----
    // Spread is a step function between quote changes, so sum(prev_spread x
    // elapsed) is EXACT and the 30-min mean is a class-A cumulative
    // difference — not a bar-based rolling mean, which would violate §10.7.
    /// per-bar `i_spdt` increment
    wi_spdt: f64,
    /// 30-min ring of `i_spdt` increments, and its running sum
    sp_ring: std::collections::VecDeque<f64>,
    sp_sum: f64,
    /// spread in TICKS as of the last depth event (bar-edge value)
    w_sp_ticks: f64,
    /// last depth-event timestamp, for the time weighting
    w_prev_ts_ns: i64,
    /// spread in ticks at `w_prev_ts_ns` (the step being integrated)
    w_prev_sp_ticks: f64,
    /// bars observed since start (explicit warmup, mirrors WARMUP_BARS)
    w_bars: i64,
    /// Optional OB distribution model. When present, `roll()` evaluates it on
    /// every CLOSED bar and republishes `logits`.
    net: Option<ObNet>,
    /// Latest published logits (None until a bar closes with all net inputs
    /// finite). Consumers read `latest_logits()`.
    logits: Option<Vec<f64>>,
    /// Bar index the published logits belong to (for staleness checks/logs).
    logits_bar: i64,
}

impl Default for ObFeats {
    fn default() -> Self {
        Self::new()
    }
}

impl ObFeats {
    pub fn new() -> Self {
        ObFeats {
            bb: f64::NAN,
            ba: f64::NAN,
            bq: 0.0,
            aq: 0.0,
            have_tob: false,
            b5: f64::NAN,
            a5: f64::NAN,
            have_d5: false,
            dbb: f64::NAN,
            dba: f64::NAN,
            dbq: 0.0,
            daq: 0.0,
            have_dbest: false,
            cum_vol: f64::NAN,
            cum_buy: f64::NAN,
            cum_cnt: f64::NAN,
            have_tr: false,
            bar_vol0: f64::NAN,
            bar_buy0: f64::NAN,
            bar_cnt0: f64::NAN,
            cur_bar: i64::MIN,
            ofi_acc: 0.0,
            ofl5_acc: 0.0,
            upd_acc: 0.0,
            dofi_acc: 0.0,
            last_vol_evt: f64::NAN,
            bigtr_acc: 0.0,
            stale_bars: 0,
            ofi_e: [0.0; 3],
            tfi_e: [0.0; 3],
            dofi_e: [0.0; 3],
            tv_e10: 0.0,
            lm_ring: VecDeque::with_capacity(602),
            r2_ring: VecDeque::with_capacity(101),
            r2_sum: 0.0,
            tslc: 0.0,
            last_mid: f64::NAN,
            out: [f64::NAN; 23],
            ready: false,
            w_ring: std::collections::VecDeque::new(),
            w_sum: [[0.0; 8]; W_WINDOWS.len()],
            w_logmid: std::collections::VecDeque::new(),
            wi_r2: 0.0,
            wi_absm: 0.0,
            wi_qupd: 0.0,
            wi_big: 0.0,
            w_last_lm: f64::NAN,
            w_quote_age: 0.0,
            w_spread_bps: f64::NAN,
            wi_spdt: 0.0,
            sp_ring: std::collections::VecDeque::new(),
            sp_sum: 0.0,
            w_sp_ticks: f64::NAN,
            w_prev_ts_ns: i64::MIN,
            w_prev_sp_ticks: f64::NAN,
            w_bars: 0,
            net: None,
            logits: None,
            logits_bar: i64::MIN,
        }
    }

    /// perp bookTicker update → CKS best-level OFI transition + activity.
    pub fn on_tob(&mut self, ts_ns: i64, bb: f64, ba: f64, bq: f64, aq: f64) {
        if !(bb > 0.0 && ba > 0.0) {
            return;
        }
        self.roll(ts_ns);
        if self.have_tob {
            // e = ΔW_bid − ΔW_ask (Cont–Kukanov–Stoikov best-level flow)
            let e_bid = if bb > self.bb {
                bq
            } else if bb < self.bb {
                -self.bq
            } else {
                bq - self.bq
            };
            let e_ask = if ba < self.ba {
                aq
            } else if ba > self.ba {
                -self.aq
            } else {
                aq - self.aq
            };
            self.ofi_acc += e_bid - e_ask;
        }
        self.bb = bb;
        self.ba = ba;
        self.bq = bq;
        self.aq = aq;
        self.have_tob = true;
        self.upd_acc += 1.0;
    }

    /// depth stream update → top-5 queue-mass flow (coarse Kolm-style).
    pub fn on_depth5(&mut self, ts_ns: i64, b5: f64, a5: f64) {
        self.roll(ts_ns);
        if self.have_d5 {
            self.ofl5_acc += (b5 - self.b5) - (a5 - self.a5);
        }
        self.b5 = b5;
        self.a5 = a5;
        self.have_d5 = true;
    }

    /// depth-diff book best quotes → the delta_min CKS OFI (`d_ofi`).
    /// Same increment as `ob_delta_feats.py` (state-transition based, so a
    /// conflated A→C transition is still well-defined), same source stream.
    pub fn on_depth_best(&mut self, ts_ns: i64, bb: f64, bq: f64, ba: f64, aq: f64) {
        if !(bb > 0.0 && ba > 0.0 && bb < ba) {
            return;
        }
        self.roll(ts_ns);
        if self.have_dbest {
            let e_bid = if bb > self.dbb {
                bq
            } else if bb < self.dbb {
                -self.dbq
            } else {
                bq - self.dbq
            };
            let e_ask = if ba < self.dba {
                aq
            } else if ba > self.dba {
                -self.daq
            } else {
                aq - self.daq
            };
            self.dofi_acc += e_bid - e_ask;
            // §11 counters accumulated AT THE EVENT SOURCE (class C -> A):
            // squared and absolute log-mid return BETWEEN CONSECUTIVE
            // best-quote changes, plus the change count.
            // §11.10 i_spdt: integrate the PREVIOUS spread over the elapsed
            // interval, then latch the new one. Accumulated on EVERY book
            // event (not only quote moves), matching ob_wfeats.py — `roll`
            // has already advanced the bar, so the interval is attributed to
            // the bar containing `ts_ns`, exactly as the lake does.
            if self.w_prev_ts_ns != i64::MIN && self.w_prev_sp_ticks.is_finite() {
                let dt_s = (ts_ns - self.w_prev_ts_ns) as f64 / 1e9;
                if dt_s > 0.0 {
                    self.wi_spdt += self.w_prev_sp_ticks * dt_s;
                }
            }
            self.w_prev_ts_ns = ts_ns;
            self.w_sp_ticks = (ba - bb) / TICK_BTC;
            self.w_prev_sp_ticks = self.w_sp_ticks;

            let m = 0.5 * (bb + ba);
            if m > 0.0 {
                let lm = m.ln();
                // PRICE-only, matching ob_wfeats.py:
                //   quote_moved = (best_b != prev_bb_px) or (best_a != prev_ba_px)
                // Counting SIZE changes here made i_qupd ~25x the lake's rate
                // (live lqupd_1s 2.20 => 8 changes/s vs a training mean of
                // 0.31/s), saturating every lqupd_W input at its clip ceiling.
                let changed = bb != self.dbb || ba != self.dba;
                if changed {
                    if self.w_last_lm.is_finite() {
                        let d = lm - self.w_last_lm;
                        self.wi_r2 += d * d;
                        self.wi_absm += d.abs();
                    }
                    self.w_last_lm = lm;
                    self.wi_qupd += 1.0;
                    self.w_quote_age = 0.0;
                }
                self.w_spread_bps = (ba - bb) / m * 1e4;
            }
        }
        if !self.w_last_lm.is_finite() {
            let m = 0.5 * (bb + ba);
            if m > 0.0 {
                self.w_last_lm = m.ln();
                self.w_spread_bps = (ba - bb) / m * 1e4;
            }
        }
        self.dbb = bb;
        self.dba = ba;
        self.dbq = bq;
        self.daq = aq;
        self.have_dbest = true;
    }

    /// `.vol` feed (cumulative TOTAL taker volume).
    pub fn on_vol(&mut self, ts_ns: i64, cum_vol: f64) {
        self.roll(ts_ns);
        if self.last_vol_evt.is_finite() && cum_vol > self.last_vol_evt {
            let d = cum_vol - self.last_vol_evt;
            if d > self.bigtr_acc {
                self.bigtr_acc = d;
            }
            // §11 `i_big`: VOLUME of large prints (additive), not a max.
            if d >= BIG_BTC {
                self.wi_big += d;
            }
        }
        self.last_vol_evt = cum_vol;
        self.cum_vol = cum_vol;
    }

    /// `.volb` feed (qty = cumulative BUY volume, price = cumulative count).
    pub fn on_volb(&mut self, ts_ns: i64, cum_buy: f64, cum_cnt: f64) {
        self.roll(ts_ns);
        self.cum_buy = cum_buy;
        self.cum_cnt = cum_cnt;
        self.have_tr = true;
    }

    /// Advance the bar clock; closes every bar boundary crossed.
    pub fn roll(&mut self, now_ns: i64) {
        let bar = now_ns / BAR_NS;
        if self.cur_bar == i64::MIN {
            self.cur_bar = bar;
            return;
        }
        while self.cur_bar < bar {
            self.close_bar();
            self.cur_bar += 1;
        }
    }

    fn close_bar(&mut self) {
        // staleness: a bar without any book event extends the ff run
        if self.upd_acc == 0.0 {
            self.stale_bars += 1;
        } else {
            self.stale_bars = 0;
        }
        let valid = self.have_tob && self.stale_bars <= STALE_BARS;
        let mid = 0.5 * (self.bb + self.ba);

        // per-bar trade deltas from the cumulative feeds (sell = vol − buy)
        let (tfi_b, tv_b, nt_b) = if self.have_tr && self.cum_vol.is_finite() {
            if self.bar_vol0.is_finite() {
                let dv = self.cum_vol - self.bar_vol0;
                let db = self.cum_buy - self.bar_buy0;
                let dn = self.cum_cnt - self.bar_cnt0;
                (2.0 * db - dv, dv, dn)
            } else {
                (0.0, 0.0, 0.0)
            }
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        self.bar_vol0 = self.cum_vol;
        self.bar_buy0 = self.cum_buy;
        self.bar_cnt0 = self.cum_cnt;

        // EWMAs (mirror: acc = (1−a)acc + a·x, a = 1 − exp(−ln2/hl))
        for (k, hl) in HL_BARS.iter().enumerate() {
            let a = 1.0 - (-std::f64::consts::LN_2 / hl).exp();
            self.ofi_e[k] = (1.0 - a) * self.ofi_e[k] + a * self.ofi_acc;
            if tfi_b.is_finite() {
                self.tfi_e[k] = (1.0 - a) * self.tfi_e[k] + a * tfi_b;
            }
            self.dofi_e[k] = (1.0 - a) * self.dofi_e[k] + a * self.dofi_acc;
        }
        let a10 = 1.0 - (-std::f64::consts::LN_2 / 100.0).exp();
        if tv_b.is_finite() {
            self.tv_e10 = (1.0 - a10) * self.tv_e10 + a10 * tv_b;
        }

        // log-mid ring (NaN while invalid — rets propagate NaN, as the mirror)
        let lm = if valid { mid.ln() } else { f64::NAN };
        self.lm_ring.push_back(lm);
        if self.lm_ring.len() > 601 {
            self.lm_ring.pop_front();
        }
        let ret = |k: usize| -> f64 {
            let n = self.lm_ring.len();
            if n > k {
                lm - self.lm_ring[n - 1 - k]
            } else {
                f64::NAN
            }
        };
        let r1 = ret(1);
        // rv: mirror zero-fills NaN r1 before squaring
        let r2 = if r1.is_finite() { r1 * r1 } else { 0.0 };
        self.r2_ring.push_back(r2);
        self.r2_sum += r2;
        if self.r2_ring.len() > 100 {
            self.r2_sum -= self.r2_ring.pop_front().unwrap_or(0.0);
        }
        let rv = self.r2_sum.max(0.0).sqrt() * 1e4;

        // tslc on bar-close mids, capped 600
        if valid && (self.last_mid.is_nan() || mid != self.last_mid) {
            self.tslc = 0.0;
            self.last_mid = mid;
        } else {
            self.tslc = (self.tslc + 1.0).min(600.0);
        }

        let nanv = f64::NAN;
        self.out = if valid {
            [
                self.ofi_acc,
                self.ofi_e[0],
                self.ofi_e[1],
                self.ofi_e[2],
                if self.have_d5 { self.ofl5_acc } else { nanv },
                tfi_b,
                self.tfi_e[0],
                self.tfi_e[1],
                self.tfi_e[2],
                self.tv_e10,
                nt_b,
                self.upd_acc,
                r1,
                ret(10),
                ret(100),
                ret(600),
                rv,
                self.tslc,
                if self.have_dbest { self.dofi_acc } else { nanv },
                if self.have_dbest { self.dofi_e[0] } else { nanv },
                if self.have_dbest { self.dofi_e[1] } else { nanv },
                if self.have_dbest { self.dofi_e[2] } else { nanv },
                if self.have_tr { self.bigtr_acc } else { nanv },
            ]
        } else {
            [nanv; 23]
        };
        self.ready = self.ready || valid;

        // ---- §11 WINDOW counters: push this bar's increments, roll sums ----
        // tfi/tv/nt come from the SAME cumulative trade feeds as the legacy
        // columns, but they are now consumed only as window differences, so
        // the 100ms/200ms publish throttle costs endpoint precision instead
        // of corrupting the value (OB_DIST_MODEL §10.2).
        let inc: [f64; 8] = [
            if self.have_dbest { self.dofi_acc } else { 0.0 },
            self.wi_r2,
            self.wi_absm,
            self.wi_qupd,
            if tv_b.is_finite() { tv_b } else { 0.0 },
            if tv_b.is_finite() { 0.5 * (tv_b + tfi_b) } else { 0.0 }, // buy = (vol+tfi)/2
            if nt_b.is_finite() { nt_b } else { 0.0 },
            self.wi_big,
        ];
        self.w_ring.push_back(inc);
        for (wi, (w, _)) in W_WINDOWS.iter().enumerate() {
            for c in 0..8 {
                self.w_sum[wi][c] += inc[c];
            }
            if self.w_ring.len() > *w {
                let old = self.w_ring[self.w_ring.len() - 1 - *w];
                for c in 0..8 {
                    self.w_sum[wi][c] -= old[c];
                }
            }
        }
        let maxw = W_WINDOWS[W_WINDOWS.len() - 1].0;
        while self.w_ring.len() > maxw + 1 {
            self.w_ring.pop_front();
        }
        // ret_W must use the SAME event mid as i_r2 / i_absm / i_qupd (the
        // depth best-quote stream), not the bookTicker mid — §11 derives every
        // book quantity from one source, and mixing them would reintroduce a
        // source inconsistency of exactly the §10.6 kind.
        self.w_logmid.push_back(self.w_last_lm);
        while self.w_logmid.len() > maxw + 1 {
            self.w_logmid.pop_front();
        }
        // 30-min spdt ring for B4's `spnorm` reference
        self.sp_ring.push_back(self.wi_spdt);
        self.sp_sum += self.wi_spdt;
        while self.sp_ring.len() > REF_SP_BARS {
            self.sp_sum -= self.sp_ring.pop_front().unwrap_or(0.0);
        }

        self.w_quote_age += 1.0;
        self.w_bars += 1;
        self.wi_spdt = 0.0;
        self.wi_r2 = 0.0;
        self.wi_absm = 0.0;
        self.wi_qupd = 0.0;
        self.wi_big = 0.0;

        // OB model: one forward per CLOSED bar (a "feature update"), so the
        // published logits always describe the bar that just ended. A NaN in
        // any net input (warmup, dead feed) republishes None rather than a
        // partial vector.
        if let Some(net) = self.net.as_ref() {
            let lg = match net.kind {
                ObNetKind::Window4 => {
                    self.window_feats_b4().and_then(|f| net.logits_window(&f))
                }
                ObNetKind::Window => self.window_feats().and_then(|f| net.logits_window(&f)),
                ObNetKind::Legacy => {
                    if valid { net.logits(&self.out) } else { None }
                }
            };
            if lg.is_some() {
                self.logits_bar = self.cur_bar;
            }
            self.logits = lg;
        }

        self.ofi_acc = 0.0;
        self.ofl5_acc = 0.0;
        self.upd_acc = 0.0;
        self.dofi_acc = 0.0;
        self.bigtr_acc = 0.0;
    }

    /// Latest CLOSED bar's features (call `roll(now)` first).
    pub fn latest(&self) -> &[f64; 23] {
        &self.out
    }

    /// Design-B (§11.6) window features for the bar that just closed, in the
    /// trainer's emission order (`OB_W_FEATS`). `None` until the longest
    /// window has filled, or if any input is not finite — never a partial
    /// vector. Every term is a ratio of window differences of cumulative
    /// counters, so it is cadence-invariant by construction (§10.7).
    pub fn window_feats(&self) -> Option<[f64; OB_W_FEATS.len()]> {
        let maxw = W_WINDOWS[W_WINDOWS.len() - 1].0;
        if self.w_ring.len() <= maxw || self.w_logmid.len() <= maxw {
            return None; // explicit warmup
        }
        const EPS: f64 = 1e-9;
        let lm_now = *self.w_logmid.back()?;
        let mut out = [0.0f64; OB_W_FEATS.len()];
        let mut k = 0usize;
        for (wi, (w, _)) in W_WINDOWS.iter().enumerate() {
            let sm = &self.w_sum[wi];
            let (ofi, r2, _absm, qupd) = (sm[C_OFI], sm[C_R2].max(0.0), sm[C_ABSM], sm[C_QUPD]);
            let (vol, buy, cnt, big) = (sm[C_VOL], sm[C_BUY], sm[C_CNT], sm[C_BIG]);
            let tfi = 2.0 * buy - vol;
            let rv = r2.sqrt() * 1e4;
            let lm_then = self.w_logmid[self.w_logmid.len() - 1 - *w];
            let ret_bp = (lm_now - lm_then) * 1e4;
            out[k] = tfi / (vol + EPS); k += 1;             // tfifrac_W
            out[k] = ofi / (vol + EPS); k += 1;             // ofin_W
            out[k] = ret_bp / (rv + EPS); k += 1;           // retstd_W
            out[k] = vol.max(0.0).ln_1p(); k += 1;          // ltv_W
            out[k] = cnt.max(0.0).ln_1p(); k += 1;          // lnt_W
            out[k] = big / (vol + EPS); k += 1;             // bigfrac_W
            out[k] = qupd.max(0.0).ln_1p(); k += 1;         // lqupd_W
        }
        out[k] = self.w_spread_bps;
        if out.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(out)
    }

    /// Design-B4 (§11.10b) window features for the bar that just closed, in
    /// the trainer's emission order (`OB_W4_FEATS`). Differs from
    /// `window_feats` in exactly the three ways §11.9/§11.10 specify:
    ///
    /// 1. the volume denominator is FLOORED at 10% of the volume expected in
    ///    this window from the trailing 60s rate — `ofin_1s` reached 1.3e11
    ///    on the 15.0% of 1s windows that contain no trades at all;
    /// 2. the rv denominator is floored the same way (scaled sqrt of the
    ///    window ratio, since rv ~ sqrt(t));
    /// 3. the trailing `spread_bps` — a function of PRICE LEVEL, since BTC's
    ///    spread is pinned at one tick — becomes `spnorm`, the log of the
    ///    spread over its own 30-minute mean.
    ///
    /// Returns `None` on a book-event gap (`spref <= 0`), where the lake
    /// drops the bar rather than fabricate a reference.
    pub fn window_feats_b4(&self) -> Option<[f64; OB_W4_FEATS.len()]> {
        let maxw = W_WINDOWS[W_WINDOWS.len() - 1].0;
        if self.w_ring.len() <= maxw || self.w_logmid.len() <= maxw {
            return None; // 60s counter warmup
        }
        if self.sp_ring.len() < REF_SP_BARS {
            return None; // 30-min spnorm reference warmup
        }
        const EPS: f64 = 1e-9;
        let lm_now = *self.w_logmid.back()?;
        // 60s reference rates that floor the ratio denominators
        let s60 = &self.w_sum[W_WINDOWS.len() - 1];
        let vol60 = s60[C_VOL];
        let rv60 = s60[C_R2].max(0.0).sqrt() * 1e4;
        let wmax = maxw as f64;

        let mut out = [0.0f64; OB_W4_FEATS.len()];
        let mut k = 0usize;
        for (wi, (w, _)) in W_WINDOWS.iter().enumerate() {
            let sm = &self.w_sum[wi];
            let (ofi, r2, qupd) = (sm[C_OFI], sm[C_R2].max(0.0), sm[C_QUPD]);
            let (vol, buy, cnt, big) = (sm[C_VOL], sm[C_BUY], sm[C_CNT], sm[C_BIG]);
            let tfi = 2.0 * buy - vol;
            let rv = r2.sqrt() * 1e4;
            let frac = *w as f64 / wmax;
            let den = vol.max(0.1 * vol60 * frac) + EPS;
            let drv = rv.max(0.1 * rv60 * frac.sqrt()) + EPS;
            let lm_then = self.w_logmid[self.w_logmid.len() - 1 - *w];
            let ret_bp = (lm_now - lm_then) * 1e4;
            out[k] = tfi / den; k += 1;                     // tfifrac_W
            out[k] = ofi / den; k += 1;                     // ofin_W
            out[k] = ret_bp / drv; k += 1;                  // retstd_W
            out[k] = vol.max(0.0).ln_1p(); k += 1;          // ltv_W
            out[k] = cnt.max(0.0).ln_1p(); k += 1;          // lnt_W
            out[k] = big / den; k += 1;                     // bigfrac_W
            out[k] = qupd.max(0.0).ln_1p(); k += 1;         // lqupd_W
        }
        // spnorm = log(spread_ticks / mean spread over REF_SP). The ring
        // holds ticks x seconds, so the mean is sum / (REF_SP bars x 0.1 s).
        let spref = self.sp_sum / (REF_SP_BARS as f64 * 0.1);
        if !(spref > 0.0) {
            return None; // book-event gap: no reference, drop the bar
        }
        out[k] = (self.w_sp_ticks.max(1e-9) / spref).ln();
        if out.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(out)
    }

    /// Attach the OB distribution model. Once set, `roll()` republishes
    /// logits on every closed bar.
    pub fn set_net(&mut self, net: ObNet) {
        self.net = Some(net);
    }

    pub fn has_net(&self) -> bool {
        self.net.is_some()
    }

    /// Latest published OB logits (temperature-applied; softmax => calibrated
    /// distribution). `None` during warmup or on a bar with missing inputs.
    pub fn latest_logits(&self) -> Option<&[f64]> {
        self.logits.as_deref()
    }

    /// Bar index the published logits belong to.
    pub fn logits_bar(&self) -> i64 {
        self.logits_bar
    }

    pub fn ready(&self) -> bool {
        self.ready
    }
}

#[cfg(test)]
mod obnet_tests {
    //! Parity for the live OB model forward vs `scripts/ob_gen_logits.py`.
    //! Vectors are the Python reference's own output on drawn feature rows.
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        feats: Vec<f64>,
        logits: Vec<f64>,
    }
    #[derive(Deserialize)]
    struct Cases {
        cases: Vec<Case>,
    }

    fn net() -> ObNet {
        ObNet::load(&format!("{}/../../models/ob-h100-delta-min.json", env!("CARGO_MANIFEST_DIR")))
            .unwrap()
    }

    /// Build a full OB_FEATS row that carries the net's 17 inputs in the right
    /// slots (everything else NaN — the net must ignore unused columns).
    fn row(net: &ObNet, feats: &[f64]) -> [f64; OB_FEATS.len()] {
        let mut r = [f64::NAN; OB_FEATS.len()];
        for (j, &i) in net.idx.iter().enumerate() {
            r[i] = feats[j];
        }
        r
    }

    #[test]
    fn ob_net_parity_vs_python() {
        let n = net();
        let c: Cases = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/testdata/ob_net_case.json",
                                             env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let mut worst: f64 = 0.0;
        for case in &c.cases {
            let got = n.logits(&row(&n, &case.feats)).expect("finite inputs must produce logits");
            assert_eq!(got.len(), case.logits.len());
            for (g, e) in got.iter().zip(case.logits.iter()) {
                worst = worst.max((g - e).abs());
            }
        }
        // python reference runs the forward in f32; rust is f64 throughout
        assert!(worst < 1e-4, "max |dlogit| = {worst}");
    }

    #[test]
    fn nan_input_publishes_nothing() {
        let n = net();
        let c: Cases = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/testdata/ob_net_case.json",
                                             env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let mut r = row(&n, &c.cases[0].feats);
        r[n.idx[3]] = f64::NAN; // one required input missing (e.g. feed warmup)
        assert!(n.logits(&r).is_none(), "a NaN input must suppress publication");
    }

    #[test]
    fn feature_mapping_is_exact() {
        // every net input must resolve to a real OB_FEATS column, and the
        // delta_min ordering must match the trained net's feature order.
        let n = net();
        assert_eq!(n.idx.len(), 17);
        let names: Vec<&str> = n.idx.iter().map(|&i| OB_FEATS[i]).collect();
        assert_eq!(names, vec![
            "ob_tfi", "ob_tfi_e2", "ob_tfi_e10", "ob_tfi_e60", "ob_tv_e10",
            "ob_nt", "ob_bigtr", "ob_tslc", "ob_ret1", "ob_ret10", "ob_ret100",
            "ob_ret600", "ob_rv100", "ob_dofi", "ob_dofi_e2", "ob_dofi_e10", "ob_dofi_e60",
        ]);
    }
}

#[cfg(test)]
mod obwnet_tests {
    //! §11 design-B (window) net: parity vs the Python reference, and the
    //! feature-space detection that keeps the two net kinds apart.
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case { feats: Vec<f64>, logits: Vec<f64> }
    #[derive(Deserialize)]
    struct Cases { cases: Vec<Case> }

    fn wnet() -> ObNet {
        ObNet::load(&format!("{}/../../models/ob-h100-wB.json", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[test]
    fn window_net_is_detected_as_window_kind() {
        let n = wnet();
        assert_eq!(n.kind, ObNetKind::Window);
        // the legacy net must still resolve as legacy
        let l = ObNet::load(&format!("{}/../../models/ob-h100-delta-min.json",
                                     env!("CARGO_MANIFEST_DIR"))).unwrap();
        assert_eq!(l.kind, ObNetKind::Legacy);
    }

    #[test]
    fn window_net_parity_vs_python() {
        let n = wnet();
        let c: Cases = serde_json::from_str(&std::fs::read_to_string(
            format!("{}/testdata/ob_wnet_case.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let mut worst: f64 = 0.0;
        for case in &c.cases {
            let got = n.logits_window(&case.feats).expect("finite inputs -> logits");
            assert_eq!(got.len(), case.logits.len());
            for (g, e) in got.iter().zip(case.logits.iter()) {
                worst = worst.max((g - e).abs());
            }
        }
        assert!(worst < 1e-4, "max |dlogit| = {worst}");
    }

    #[test]
    fn window_feats_warm_up_then_publish() {
        // Feed a synthetic tape: features must stay None until the longest
        // window (60s = 600 bars) has filled, then become finite.
        let mut f = ObFeats::new();
        let mut t = 0i64;
        let mut px = 100_000.0f64;
        f.on_vol(t, 0.0);
        f.on_volb(t, 0.0, 0.0);
        let mut cum = 0.0;
        let mut cum_buy = 0.0;
        for i in 0..900 {
            t += 100_000_000; // one 100ms bar
            px += if i % 3 == 0 { 1.0 } else { -0.5 };
            f.on_depth_best(t, px - 0.5, 3.0, px + 0.5, 4.0);
            cum += 0.4;
            cum_buy += 0.25;
            f.on_vol(t, cum);
            f.on_volb(t, cum_buy, (i + 1) as f64);
            if i == 300 {
                assert!(f.window_feats().is_none(), "must not publish before 600 bars");
            }
        }
        f.roll(t + 100_000_000);
        let w = f.window_feats().expect("should publish after warmup");
        assert!(w.iter().all(|v| v.is_finite()));
        // spread 1.0 on a ~100k mid ~= 0.1bp
        assert!((w[OB_W_FEATS.len() - 1] - 0.1).abs() < 0.05, "spread_bps {}", w[OB_W_FEATS.len() - 1]);
    }
}
