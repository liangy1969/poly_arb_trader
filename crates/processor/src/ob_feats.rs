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
pub struct ObNet {
    idx: Vec<usize>, // OB_FEATS index per net input
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
        let mut idx = Vec::with_capacity(n_in);
        for c in &js.live_cols {
            let i = OB_FEATS
                .iter()
                .position(|f| f == c)
                .ok_or_else(|| anyhow::anyhow!("net input {c:?} is not an OB_FEATS column"))?;
            idx.push(i);
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
        Ok(ObNet { idx, lo: js.lo, hi: js.hi, mu: js.mu, sd: js.sd, w, b, dims, relu, t: js.t, k: js.k })
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Logits for one CLOSED bar's feature vector. `None` if any input the net
    /// needs is NaN (warmup / missing feed) — never publish a partial vector.
    pub fn logits(&self, feats: &[f64; OB_FEATS.len()]) -> Option<Vec<f64>> {
        let mut x = Vec::with_capacity(self.idx.len());
        for (j, &i) in self.idx.iter().enumerate() {
            let v = feats[i];
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

        // OB model: one forward per CLOSED bar (a "feature update"), so the
        // published logits always describe the bar that just ended. A NaN in
        // any net input (warmup, dead feed) republishes None rather than a
        // partial vector.
        if let Some(net) = self.net.as_ref() {
            let lg = if valid { net.logits(&self.out) } else { None };
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
