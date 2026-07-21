//! FairSurface (frozen offline MLP) + the online (Δb, Δρ) fit — the numeric
//! core of the FairRide strategy (DESIGN_FAIR_RIDE §4/§5/§10).
//!
//! - Forward: τ = clamp(tte/900, 1e-4, 1); z′ = ((px − b)/s)/√τ;
//!   logit = MLP([z′, ln τ, x₀…x_{k−1}]) (2+k→32→32→1 tanh, "direct" mode);
//!   fair = σ(logit). The k EXTRA features (px2imb: basis, dbasis15, dbasis60,
//!   imb1) are z-scored with the model's `mu`/`sd` and clipped to ±5 — exactly
//!   the sim's `NATIVE_MOM` reconstruction. k = 0 is the plain cb surface.
//! - Fit: full-batch Adam (lr .05, fresh moments per fit, fixed steps) on
//!   BCE(σ(logit), clip(mid, .01, .99)) over the 2 params, gradients by
//!   central finite difference (ε=1e-4). b = strike + b_scale·Δb,
//!   s = exp(rho_bar + Δρ). Features are held fixed per row during the fit.
//!
//! Everything is f64; parity vs PyTorch is asserted in output space
//! (tests below: |Δlogit| < 1e-4 on exported vectors; |Δfair| < 0.005 across
//! the trade window on a canned real event).

use serde::Deserialize;

const P_CLIP: f64 = 0.01;
const FD_EPS: f64 = 1e-4;
/// Compile-time cap on extra features (px2imb uses 4). Keeps `FitRow` `Copy`.
pub const MAX_EXTRA: usize = 8;

#[derive(Deserialize)]
struct LayerJson {
    w: Vec<Vec<f64>>,
    b: Vec<f64>,
}

#[derive(Deserialize)]
struct ArchJson {
    hidden: usize,
    mode: String,
}

#[derive(Deserialize)]
struct ModelJson {
    arch: ArchJson,
    rho_bar: f64,
    b_scale: f64,
    layers: Vec<LayerJson>,
    #[serde(default)]
    extras: Vec<String>,
    #[serde(default)]
    mu: Vec<f64>,
    #[serde(default)]
    sd: Vec<f64>,
}

/// The frozen fair-probability surface.
pub struct FairSurface {
    /// Row-major weights per layer; `w[l][o * in_dim + i]`.
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<(usize, usize)>, // (in, out) per layer
    pub rho_bar: f64,
    pub b_scale: f64,
    /// Extra-feature names in input order (after [z′, ln τ]); empty for cb.
    pub extras: Vec<String>,
    mu: Vec<f64>,
    sd: Vec<f64>,
}

/// One calibration sample: tte seconds, reference price, kalshi YES mid, and
/// the RAW extra features (un-z-scored; the surface z-scores with mu/sd). Only
/// the first `surface.n_extra()` entries of `feats` are read.
#[derive(Clone, Copy, Debug)]
pub struct FitRow {
    pub tte_s: f64,
    pub px: f64,
    pub mid: f64,
    pub feats: [f64; MAX_EXTRA],
}

impl FitRow {
    /// Convenience constructor for the cb (no-extra) path.
    pub fn new(tte_s: f64, px: f64, mid: f64) -> Self {
        FitRow { tte_s, px, mid, feats: [0.0; MAX_EXTRA] }
    }
}

impl FairSurface {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let js: ModelJson = serde_json::from_str(text)?;
        anyhow::ensure!(js.arch.mode == "direct", "unsupported model mode {}", js.arch.mode);
        anyhow::ensure!(js.layers.len() == 3, "expected 3 layers");
        anyhow::ensure!(js.layers[0].b.len() == js.arch.hidden, "hidden dim mismatch");
        let n_extra = js.extras.len();
        anyhow::ensure!(n_extra <= MAX_EXTRA, "too many extras ({n_extra} > {MAX_EXTRA})");
        anyhow::ensure!(
            js.mu.len() == n_extra && js.sd.len() == n_extra,
            "mu/sd length must match extras ({n_extra})"
        );
        anyhow::ensure!(
            js.layers[0].w[0].len() == 2 + n_extra,
            "input dim {} != 2 + {n_extra} extras",
            js.layers[0].w[0].len()
        );
        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut dims = Vec::new();
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
        }
        Ok(FairSurface {
            w,
            b,
            dims,
            rho_bar: js.rho_bar,
            b_scale: js.b_scale,
            extras: js.extras,
            mu: js.mu,
            sd: js.sd,
        })
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Number of extra features this surface consumes (0 for cb).
    pub fn n_extra(&self) -> usize {
        self.extras.len()
    }

    /// Raw model logit for (px, tte) under event params (b, s). `feats` are the
    /// RAW extra features (only the first `n_extra` are read; z-scored here).
    pub fn logit(&self, px: f64, tte_s: f64, b: f64, s: f64, feats: &[f64]) -> f64 {
        let tau = (tte_s / 900.0).clamp(1e-4, 1.0);
        let zp = ((px - b) / s) / tau.sqrt();
        let mut h = Vec::with_capacity(2 + self.extras.len());
        h.push(zp);
        h.push(tau.ln());
        for i in 0..self.extras.len() {
            // z-score + clip to ±5, matching the sim's np.clip((X-mu)/sd,-5,5)
            h.push(((feats[i] - self.mu[i]) / self.sd[i]).clamp(-5.0, 5.0));
        }
        for l in 0..3 {
            let (in_dim, out_dim) = self.dims[l];
            let mut out = vec![0.0f64; out_dim];
            for o in 0..out_dim {
                let mut acc = self.b[l][o];
                let row = &self.w[l][o * in_dim..(o + 1) * in_dim];
                for i in 0..in_dim {
                    acc += row[i] * h[i];
                }
                out[o] = if l < 2 { acc.tanh() } else { acc };
            }
            h = out;
        }
        h[0]
    }

    /// Fair probability under event params (Δb, Δρ) with the strike prior.
    /// `feats` are the RAW extra features (empty slice for cb).
    pub fn fair(&self, px: f64, tte_s: f64, strike: f64, d_b: f64, d_rho: f64, feats: &[f64]) -> f64 {
        let b = strike + self.b_scale * d_b;
        let s = (self.rho_bar + d_rho).exp();
        sigmoid(self.logit(px, tte_s, b, s, feats))
    }

    /// Mean BCE(σ(logit), clip(mid)) over rows for params (Δb, Δρ). Each row's
    /// fixed features feed the surface.
    pub fn fit_loss(&self, rows: &[FitRow], strike: f64, d_b: f64, d_rho: f64) -> f64 {
        let b = strike + self.b_scale * d_b;
        let s = (self.rho_bar + d_rho).exp();
        let mut acc = 0.0;
        for r in rows {
            let l = self.logit(r.px, r.tte_s, b, s, &r.feats);
            let y = r.mid.clamp(P_CLIP, 1.0 - P_CLIP);
            // stable BCE-with-logits: max(l,0) − l·y + ln(1 + e^{−|l|})
            acc += l.max(0.0) - l * y + (-l.abs()).exp().ln_1p();
        }
        acc / rows.len() as f64
    }

    /// Full-batch Adam on (Δb, Δρ), central-FD gradients. Fresh moment state
    /// (matches the sim's per-call `torch.optim.Adam`); params warm-started
    /// from `init`. Deterministic.
    pub fn fit(&self, rows: &[FitRow], strike: f64, init: (f64, f64), steps: u32, lr: f64) -> (f64, f64) {
        let (mut db, mut dr) = init;
        let (b1, b2, eps) = (0.9, 0.999, 1e-8);
        let (mut m_b, mut v_b, mut m_r, mut v_r) = (0.0, 0.0, 0.0, 0.0);
        for t in 1..=steps {
            let g_b = (self.fit_loss(rows, strike, db + FD_EPS, dr)
                - self.fit_loss(rows, strike, db - FD_EPS, dr))
                / (2.0 * FD_EPS);
            let g_r = (self.fit_loss(rows, strike, db, dr + FD_EPS)
                - self.fit_loss(rows, strike, db, dr - FD_EPS))
                / (2.0 * FD_EPS);
            m_b = b1 * m_b + (1.0 - b1) * g_b;
            v_b = b2 * v_b + (1.0 - b2) * g_b * g_b;
            m_r = b1 * m_r + (1.0 - b1) * g_r;
            v_r = b2 * v_r + (1.0 - b2) * g_r * g_r;
            let bc1 = 1.0 - b1.powi(t as i32);
            let bc2 = 1.0 - b2.powi(t as i32);
            db -= lr * (m_b / bc1) / ((v_b / bc2).sqrt() + eps);
            dr -= lr * (m_r / bc1) / ((v_r / bc2).sqrt() + eps);
        }
        (db, dr)
    }
}

#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Native reconstruction of the px2imb extra features from the live feed,
/// replicating the sim's `NATIVE_MOM` path exactly:
///   basis     = cb_mid − perp_mid            (both = latest snapshot)
///   dbasis{k} = basis − basis(now − k·1000ms) (lag from the basis history)
///   imb1      = (perp_bid_sz − perp_ask_sz) / (perp_bid_sz + perp_ask_sz)
/// The lag accepts the most recent basis sample with ts ≤ now − k·1000 and
/// ts ≥ now − k·1000 − 2000 (the sim's ±2s tolerance); if absent, `feats`
/// returns `None` so the caller DROPS the row/skips the tick — matching the
/// sim dropping NaN-feature rows. `extras` empty ⇒ inactive (the cb surface).
#[derive(Clone)]
pub struct FeatureState {
    extras: Vec<String>,
    perp_mid: f64,
    perp_bsz: f64,
    perp_asz: f64,
    cb_mid: f64,
    cb_ts: i64,
    /// (ts_ns, basis) history; pruned to `horizon_ns`.
    basis_ring: std::collections::VecDeque<(i64, f64)>,
    /// (ts_ns, perp_mid) history for momentum lags; pruned to `horizon_ns`.
    mid_ring: std::collections::VecDeque<(i64, f64)>,
    /// (ts_ns, qty) perp trade volume for surge windows; pruned to `VOL_HORIZON_NS`.
    trade_ring: std::collections::VecDeque<(i64, f64)>,
    horizon_ns: i64,
}

impl FeatureState {
    pub fn new(extras: Vec<String>) -> Self {
        FeatureState {
            extras,
            perp_mid: f64::NAN,
            perp_bsz: 0.0,
            perp_asz: 0.0,
            cb_mid: f64::NAN,
            cb_ts: 0,
            basis_ring: std::collections::VecDeque::new(),
            mid_ring: std::collections::VecDeque::new(),
            trade_ring: std::collections::VecDeque::new(),
            horizon_ns: 130_000_000_000, // 130s — covers dbasis60 + mom120 + slack
        }
    }

    /// Coinbase quote staleness gate (the sim only uses cb aged ≤ 5s).
    const CB_STALE_NS: i64 = 5_000_000_000;
    /// Trade-volume ring horizon — covers vsurge's 600s window + slack.
    const VOL_HORIZON_NS: i64 = 620_000_000_000;

    /// Any extra features to build? (false for the cb surface.)
    pub fn active(&self) -> bool {
        !self.extras.is_empty()
    }

    pub fn on_perp(&mut self, ts_ns: i64, mid: f64, bid_sz: f64, ask_sz: f64) {
        if mid > 0.0 {
            self.perp_mid = mid;
            self.perp_bsz = bid_sz;
            self.perp_asz = ask_sz;
            self.push_basis(ts_ns);
            self.push_mid(ts_ns, mid);
        }
    }

    pub fn on_cb(&mut self, ts_ns: i64, mid: f64) {
        if mid > 0.0 {
            self.cb_mid = mid;
            self.cb_ts = ts_ns;
            self.push_basis(ts_ns);
        }
    }

    fn push_basis(&mut self, ts_ns: i64) {
        if self.perp_mid > 0.0 && self.cb_mid > 0.0 {
            self.basis_ring.push_back((ts_ns, self.cb_mid - self.perp_mid));
            let cutoff = ts_ns - self.horizon_ns;
            while self.basis_ring.front().map_or(false, |&(t, _)| t < cutoff) {
                self.basis_ring.pop_front();
            }
        }
    }

    /// Most recent basis with ts ≤ now − k·1000ms, accepted iff also ≥ that − 2s.
    fn basis_lag(&self, now_ns: i64, k_s: i64) -> Option<f64> {
        let target = now_ns - k_s * 1_000_000_000;
        let lo = target - 2_000_000_000;
        for &(t, b) in self.basis_ring.iter().rev() {
            if t <= target {
                return if t >= lo { Some(b) } else { None };
            }
        }
        None
    }

    fn push_mid(&mut self, ts_ns: i64, mid: f64) {
        self.mid_ring.push_back((ts_ns, mid));
        let cutoff = ts_ns - self.horizon_ns;
        while self.mid_ring.front().map_or(false, |&(t, _)| t < cutoff) {
            self.mid_ring.pop_front();
        }
    }

    /// Most recent perp mid with ts ≤ now − k·1000ms, accepted iff also ≥ that − 2s.
    fn mid_lag(&self, now_ns: i64, k_s: i64) -> Option<f64> {
        let target = now_ns - k_s * 1_000_000_000;
        let lo = target - 2_000_000_000;
        for &(t, m) in self.mid_ring.iter().rev() {
            if t <= target {
                return if t >= lo { Some(m) } else { None };
            }
        }
        None
    }

    /// Feed the perp CUMULATIVE traded volume (monotone). Conflation-safe: the
    /// latest cumulative captures every trade even if intermediate events drop.
    pub fn on_perp_trade(&mut self, ts_ns: i64, cum_vol: f64) {
        self.trade_ring.push_back((ts_ns, cum_vol));
        let cutoff = ts_ns - Self::VOL_HORIZON_NS;
        while self.trade_ring.front().map_or(false, |&(t, _)| t < cutoff) {
            self.trade_ring.pop_front();
        }
    }

    /// Cumulative volume as of ≤ now − w_s (accepted iff also ≥ that − 5s).
    fn cum_lag(&self, now_ns: i64, w_s: i64) -> Option<f64> {
        let target = now_ns - w_s * 1_000_000_000;
        let lo = target - 5_000_000_000;
        for &(t, c) in self.trade_ring.iter().rev() {
            if t <= target {
                return if t >= lo { Some(c) } else { None };
            }
        }
        None
    }

    /// Raw extra features in `extras` order, or `None` if any lag/size missing
    /// or the coinbase quote is stale (> 5s) — each maps to a dropped sim row.
    pub fn feats(&self, now_ns: i64) -> Option<[f64; MAX_EXTRA]> {
        if !(self.perp_mid > 0.0 && self.cb_mid > 0.0)
            || now_ns - self.cb_ts > Self::CB_STALE_NS
        {
            return None;
        }
        let basis = self.cb_mid - self.perp_mid;
        let mut out = [0.0; MAX_EXTRA];
        for (i, name) in self.extras.iter().enumerate() {
            out[i] = match name.as_str() {
                "basis" => basis,
                "imb1" => {
                    let s = self.perp_bsz + self.perp_asz;
                    if s <= 0.0 {
                        return None;
                    }
                    (self.perp_bsz - self.perp_asz) / s
                }
                n if n.starts_with("dbasis") => {
                    let k: i64 = n[6..].parse().ok()?;
                    basis - self.basis_lag(now_ns, k)?
                }
                n if n.starts_with("mom") => {
                    // perp mid change over k seconds (matches the trainer's mom{k})
                    let k: i64 = n[3..].parse().ok()?;
                    self.perp_mid - self.mid_lag(now_ns, k)?
                }
                "vsurge" => {
                    // 60s / 600s taker volume from the cumulative counter:
                    // vol(W) = cum(now) − cum(now − W). None until 600s of history.
                    let cum_now = self.trade_ring.back().map(|&(_, c)| c)?;
                    let v60 = cum_now - self.cum_lag(now_ns, 60)?;
                    let v600 = cum_now - self.cum_lag(now_ns, 600)?;
                    if v600 <= 0.0 {
                        return None;
                    }
                    v60 / v600
                }
                _ => return None, // unknown feature name — fail loud via caller
            };
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn testdata(name: &str) -> String {
        std::fs::read_to_string(format!("{}/testdata/{}", env!("CARGO_MANIFEST_DIR"), name)).unwrap()
    }

    fn model() -> FairSurface {
        FairSurface::load(&format!("{}/../../models/fair-cb-x60.json", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[derive(Deserialize)]
    struct Vecs {
        rows: Vec<VecRow>,
    }
    #[derive(Deserialize)]
    struct VecRow {
        px: f64,
        tte_s: f64,
        b: f64,
        s: f64,
        logit: f64,
    }

    /// Gate 1 (DESIGN_FAIR_RIDE §7): Rust forward == torch forward.
    #[test]
    fn surface_parity_vs_torch() {
        let m = model();
        let v: Vecs = serde_json::from_str(&testdata("surface_vectors.json")).unwrap();
        assert!(v.rows.len() >= 500);
        let mut worst = 0.0f64;
        for r in &v.rows {
            let l = m.logit(r.px, r.tte_s, r.b, r.s, &[]);
            worst = worst.max((l - r.logit).abs());
        }
        assert!(worst < 1e-4, "max |Δlogit| = {worst}");
    }

    #[derive(Deserialize)]
    struct XVecs {
        rows: Vec<XVecRow>,
    }
    #[derive(Deserialize)]
    struct XVecRow {
        px: f64,
        tte_s: f64,
        b: f64,
        s: f64,
        feats: Vec<f64>,
        logit: f64,
    }

    /// Gate 1 for px2imb: the 6-input forward + mu/sd z-scoring == torch. The
    /// vectors pass RAW features; the surface z-scores internally.
    #[test]
    fn surface_parity_px2imb_vs_torch() {
        let m = FairSurface::load(&format!(
            "{}/../../models/fair-px2imb-btc.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        assert_eq!(m.n_extra(), 4);
        let v: XVecs = serde_json::from_str(&testdata("surface_vectors_px2imb.json")).unwrap();
        assert!(v.rows.len() >= 500);
        let mut worst = 0.0f64;
        for r in &v.rows {
            let l = m.logit(r.px, r.tte_s, r.b, r.s, &r.feats);
            worst = worst.max((l - r.logit).abs());
        }
        assert!(worst < 1e-4, "px2imb max |Δlogit| = {worst}");
    }

    #[derive(Deserialize)]
    struct FitCase {
        strike: f64,
        rows: Vec<CaseRow>,
        core_rows: Vec<CaseRow>,
        stage1: Stage,
        stage2: Stage,
    }
    #[derive(Deserialize)]
    struct CaseRow {
        tte_s: f64,
        px: f64,
        mid: f64,
    }
    #[derive(Deserialize)]
    struct Stage {
        fit_tte_gt: f64,
        steps: u32,
        init: (f64, f64),
        db: f64,
        dr: f64,
        fair_core: Vec<f64>,
    }

    fn max_fair_diff(m: &FairSurface, case: &FitCase, db: f64, dr: f64, torch_curve: &[f64]) -> f64 {
        let mut worst = 0.0f64;
        for (r, tf) in case.core_rows.iter().zip(torch_curve) {
            let f = m.fair(r.px, r.tte_s, case.strike, db, dr, &[]);
            worst = worst.max((f - tf).abs());
        }
        worst
    }

    /// Gate 2: the FD-Adam fit reproduces the torch fit's fair curve.
    #[test]
    fn fit_parity_vs_torch() {
        let m = model();
        let case: FitCase = serde_json::from_str(&testdata("fit_case.json")).unwrap();
        let sub = |gt: f64| -> Vec<FitRow> {
            case.rows
                .iter()
                .filter(|r| r.tte_s > gt)
                .map(|r| FitRow::new(r.tte_s, r.px, r.mid))
                .collect()
        };
        // stage 1: cold fit
        let rows1 = sub(case.stage1.fit_tte_gt);
        let (db1, dr1) = m.fit(&rows1, case.strike, case.stage1.init, case.stage1.steps, 0.05);
        let d1 = max_fair_diff(&m, &case, db1, dr1, &case.stage1.fair_core);
        assert!(d1 < 0.005, "stage1 max |Δfair| = {d1} (db {db1} vs {}, dr {dr1} vs {})", case.stage1.db, case.stage1.dr);
        // stage 2: warm refit, init pinned to the torch stage-1 params to
        // isolate the refit step itself
        let rows2 = sub(case.stage2.fit_tte_gt);
        let (db2, dr2) = m.fit(&rows2, case.strike, case.stage2.init, case.stage2.steps, 0.05);
        let d2 = max_fair_diff(&m, &case, db2, dr2, &case.stage2.fair_core);
        assert!(d2 < 0.005, "stage2 max |Δfair| = {d2}");
        // full own-chain (stage2 warm-started from OUR stage1) must also land
        // inside tolerance — this is how it runs live.
        let (db2b, dr2b) = m.fit(&rows2, case.strike, (db1, dr1), case.stage2.steps, 0.05);
        let d2b = max_fair_diff(&m, &case, db2b, dr2b, &case.stage2.fair_core);
        assert!(d2b < 0.005, "own-chain max |Δfair| = {d2b}");
    }
}
