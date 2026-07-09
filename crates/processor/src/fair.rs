//! FairSurface (frozen offline MLP) + the online (Δb, Δρ) fit — the numeric
//! core of the FairRide strategy (DESIGN_FAIR_RIDE §4/§5/§10).
//!
//! - Forward: τ = clamp(tte/900, 1e-4, 1); z′ = ((px − b)/s)/√τ;
//!   logit = MLP([z′, ln τ]) (2→32→32→1 tanh, "direct" mode); fair = σ(logit).
//! - Fit: full-batch Adam (lr .05, fresh moments per fit, fixed steps) on
//!   BCE(σ(logit), clip(mid, .01, .99)) over the 2 params, gradients by
//!   central finite difference (ε=1e-4). b = strike + b_scale·Δb,
//!   s = exp(rho_bar + Δρ).
//!
//! Everything is f64; parity vs PyTorch is asserted in output space
//! (tests below: |Δlogit| < 1e-4 on exported vectors; |Δfair| < 0.005 across
//! the trade window on a canned real event).

use serde::Deserialize;

const P_CLIP: f64 = 0.01;
const FD_EPS: f64 = 1e-4;

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
}

/// The frozen fair-probability surface.
pub struct FairSurface {
    /// Row-major weights per layer; `w[l][o * in_dim + i]`.
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
    dims: Vec<(usize, usize)>, // (in, out) per layer
    pub rho_bar: f64,
    pub b_scale: f64,
}

/// One calibration sample: (tte seconds, reference price, kalshi YES mid).
#[derive(Clone, Copy, Debug)]
pub struct FitRow {
    pub tte_s: f64,
    pub px: f64,
    pub mid: f64,
}

impl FairSurface {
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let js: ModelJson = serde_json::from_str(text)?;
        anyhow::ensure!(js.arch.mode == "direct", "unsupported model mode {}", js.arch.mode);
        anyhow::ensure!(js.layers.len() == 3, "expected 3 layers");
        anyhow::ensure!(js.layers[0].b.len() == js.arch.hidden, "hidden dim mismatch");
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
        Ok(FairSurface { w, b, dims, rho_bar: js.rho_bar, b_scale: js.b_scale })
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Raw model logit for (px, tte) under event params (b, s).
    pub fn logit(&self, px: f64, tte_s: f64, b: f64, s: f64) -> f64 {
        let tau = (tte_s / 900.0).clamp(1e-4, 1.0);
        let zp = ((px - b) / s) / tau.sqrt();
        let mut h = vec![zp, tau.ln()];
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
    pub fn fair(&self, px: f64, tte_s: f64, strike: f64, d_b: f64, d_rho: f64) -> f64 {
        let b = strike + self.b_scale * d_b;
        let s = (self.rho_bar + d_rho).exp();
        sigmoid(self.logit(px, tte_s, b, s))
    }

    /// Mean BCE(σ(logit), clip(mid)) over rows for params (Δb, Δρ).
    pub fn fit_loss(&self, rows: &[FitRow], strike: f64, d_b: f64, d_rho: f64) -> f64 {
        let b = strike + self.b_scale * d_b;
        let s = (self.rho_bar + d_rho).exp();
        let mut acc = 0.0;
        for r in rows {
            let l = self.logit(r.px, r.tte_s, b, s);
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
            let l = m.logit(r.px, r.tte_s, r.b, r.s);
            worst = worst.max((l - r.logit).abs());
        }
        assert!(worst < 1e-4, "max |Δlogit| = {worst}");
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
            let f = m.fair(r.px, r.tte_s, case.strike, db, dr);
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
                .map(|r| FitRow { tte_s: r.tte_s, px: r.px, mid: r.mid })
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
