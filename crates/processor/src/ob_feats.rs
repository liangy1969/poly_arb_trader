//! ObFeats — live reconstruction of the OB_DIST_MODEL `flow_vol` features
//! (E:/crypto/collector, doc/OB_DIST_MODEL.md) on 100ms bars, from the
//! trader's own feeds: perp bookTicker (top-of-book transitions → CKS OFI,
//! upd, mid), the @depth stream (top-5 queue mass → ofl5) and the signed
//! cumulative trade feed `.volb` (tfi/tv/nt).
//!
//! 18 of the 19 mirror features — `big_tr` (max trade in bar) is dropped:
//! it cannot be cumulative-encoded through a conflating bus.
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

pub const OB_FEATS: [&str; 18] = [
    "ob_ofi", "ob_ofi_e2", "ob_ofi_e10", "ob_ofi_e60", "ob_ofl5",
    "ob_tfi", "ob_tfi_e2", "ob_tfi_e10", "ob_tfi_e60", "ob_tv_e10",
    "ob_nt", "ob_upd", "ob_ret1", "ob_ret10", "ob_ret100", "ob_ret600",
    "ob_rv100", "ob_tslc",
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
    stale_bars: i64,
    // EWMA state
    ofi_e: [f64; 3],
    tfi_e: [f64; 3],
    tv_e10: f64,
    // bar-close log-mid ring (601 = enough for ret_600) and rv ring (100 r1²)
    lm_ring: VecDeque<f64>,
    r2_ring: VecDeque<f64>,
    r2_sum: f64,
    tslc: f64,
    last_mid: f64,
    out: [f64; 18],
    ready: bool,
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
            stale_bars: 0,
            ofi_e: [0.0; 3],
            tfi_e: [0.0; 3],
            tv_e10: 0.0,
            lm_ring: VecDeque::with_capacity(602),
            r2_ring: VecDeque::with_capacity(101),
            r2_sum: 0.0,
            tslc: 0.0,
            last_mid: f64::NAN,
            out: [f64::NAN; 18],
            ready: false,
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

    /// `.vol` feed (cumulative TOTAL taker volume).
    pub fn on_vol(&mut self, ts_ns: i64, cum_vol: f64) {
        self.roll(ts_ns);
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
            ]
        } else {
            [nanv; 18]
        };
        self.ready = self.ready || valid;

        self.ofi_acc = 0.0;
        self.ofl5_acc = 0.0;
        self.upd_acc = 0.0;
    }

    /// Latest CLOSED bar's features (call `roll(now)` first).
    pub fn latest(&self) -> &[f64; 18] {
        &self.out
    }

    pub fn ready(&self) -> bool {
        self.ready
    }
}
