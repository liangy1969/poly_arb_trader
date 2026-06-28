//! Short-term slug prediction (DESIGN §5). For btc-updown the slug timestamp is
//! the window **START** (verified in the Python reference `_WINDOW_REGISTRY`),
//! so the current window's slug is `(now // window) * window`.

/// Predict `{prefix}-{start}` slugs for the current + upcoming `n-1` windows.
/// Returns `(slug, window_start_secs)` pairs.
pub fn predict_slugs(prefix: &str, window_sec: i64, n: usize, now_sec: i64) -> Vec<(String, i64)> {
    let first = (now_sec / window_sec) * window_sec;
    (0..n as i64)
        .map(|i| {
            let start = first + i * window_sec;
            (format!("{prefix}-{start}"), start)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_window_is_floor_of_now() {
        // now=1000s, window=300 -> floor = 900; slugs 900, 1200, 1500
        let s = predict_slugs("btc-updown-5m", 300, 3, 1000);
        assert_eq!(s[0], ("btc-updown-5m-900".into(), 900));
        assert_eq!(s[1], ("btc-updown-5m-1200".into(), 1200));
        assert_eq!(s[2], ("btc-updown-5m-1500".into(), 1500));
    }

    #[test]
    fn exact_boundary() {
        let s = predict_slugs("btc-updown-5m", 300, 1, 1200);
        assert_eq!(s[0].1, 1200);
    }
}
