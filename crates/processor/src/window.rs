//! Fixed-horizon ring of `(ts_ns, mid)` samples; `move_bps` is what the rule
//! reads (DESIGN §8). Samples are pushed in time order, so lookups assume
//! ascending `ts`.

use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct RollingWindow {
    buf: VecDeque<(i64, f64)>,
    cap: usize,
    horizon_ns: i64,
}

impl RollingWindow {
    pub fn new(cap: usize, horizon_ms: u64) -> Self {
        RollingWindow {
            buf: VecDeque::with_capacity(cap),
            cap,
            horizon_ns: (horizon_ms as i64) * 1_000_000,
        }
    }

    /// Append a sample; evict anything older than `horizon` or beyond `cap`.
    pub fn push(&mut self, ts_ns: i64, mid: f64) {
        self.buf.push_back((ts_ns, mid));
        let cutoff = ts_ns - self.horizon_ns;
        while let Some(&(t, _)) = self.buf.front() {
            if t < cutoff || self.buf.len() > self.cap {
                self.buf.pop_front();
            } else {
                break;
            }
        }
    }

    /// Latest sample at or before `ts_ns`.
    pub fn asof(&self, ts_ns: i64) -> Option<f64> {
        let mut found = None;
        for &(t, m) in self.buf.iter() {
            if t <= ts_ns {
                found = Some(m);
            } else {
                break;
            }
        }
        found
    }

    /// `(mid_now / mid_asof(now - window) - 1) * 1e4`. `None` if no sample spans
    /// the window.
    pub fn move_bps(&self, now_ns: i64, window_ms: u64) -> Option<f64> {
        let cur = self.asof(now_ns)?;
        let past = self.asof(now_ns - (window_ms as i64) * 1_000_000)?;
        if past.abs() > 1e-12 {
            Some((cur / past - 1.0) * 1e4)
        } else {
            None
        }
    }

    /// Raw mid change `mid_now - mid_asof(now - window)` (price units, not bps).
    /// `None` if no sample spans the window. For a [0,1] token, ×100 = cents.
    pub fn move_abs(&self, now_ns: i64, window_ms: u64) -> Option<f64> {
        let cur = self.asof(now_ns)?;
        let past = self.asof(now_ns - (window_ms as i64) * 1_000_000)?;
        Some(cur - past)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_bps_spans_window() {
        let mut w = RollingWindow::new(64, 5000);
        let base = 1_000_000_000_000i64;
        // +1 bps per 100ms step over 1s
        for k in 0..=10i64 {
            let mid = 100.0 * (1.0 + (k as f64) * 1.0 / 10000.0);
            w.push(base + k * 100_000_000, mid);
        }
        // not enough history before this point
        assert!(w.move_bps(base + 500_000_000, 1000).is_none());
        // full 1s window: ~10 bps
        let m = w.move_bps(base + 1_000_000_000, 1000).unwrap();
        assert!((m - 10.0).abs() < 0.1, "got {m}");
    }
}
