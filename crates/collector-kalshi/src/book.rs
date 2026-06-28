//! Stateful reflected-YES L2 book for one Kalshi market + WS/REST frame parsing
//! (ports the Python reference's `kalshi/book.py` + `kalshi/normalizer.py`).
//!
//! Kalshi rests **bids on two sides** — resting yes-bids and resting no-bids. By
//! the binary identity (Yes + No = $1), a NO bid @ `q` ≡ a YES ask @ `1 - q`. This
//! book keeps both raw bid maps and emits the canonical **reflected YES (= Up)**
//! ladder. Prices are kept in **[0,1]** (probability / dollars-per-share) — the
//! same canonical unit as the Polymarket book, *not* the reference's cents.

use std::collections::BTreeMap;

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KSide {
    Yes,
    No,
}

/// Parse a number Kalshi may send as a JSON number or a quoted string.
pub fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Parse a level array `[[price, size], …]` (or `[{price,size}]`) into
/// `(price[0,1], size>0)`. `cents == true` → the price field is integer cents
/// (÷100); `false` → already dollars.
pub fn parse_levels(v: &Value, cents: bool) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    if let Some(arr) = v.as_array() {
        for lvl in arr {
            let (p, s) = if lvl.is_object() {
                (num(&lvl["price"]), num(&lvl["size"]))
            } else if let Some(a) = lvl.as_array() {
                if a.len() >= 2 {
                    (num(&a[0]), num(&a[1]))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
            if let (Some(mut p), Some(s)) = (p, s) {
                if cents {
                    p /= 100.0;
                }
                if s > 0.0 {
                    out.push((p, s));
                }
            }
        }
    }
    out
}

/// Prefer the fixed-point `*_dollars*` field (already [0,1]); fall back to the
/// plain integer-cents field (÷100). Absent → empty.
fn levels_from(m: &Value, dollar_key: &str, cent_key: &str) -> Vec<(f64, f64)> {
    if let Some(a) = m.get(dollar_key) {
        parse_levels(a, false)
    } else if let Some(a) = m.get(cent_key) {
        parse_levels(a, true)
    } else {
        Vec::new()
    }
}

/// REST `GET /markets/{ticker}/orderbook` → `(yes_bids, no_bids)` in [0,1].
pub fn parse_rest_orderbook(ob: &Value) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let null = Value::Null;
    let book = ob
        .get("orderbook_fp")
        .or_else(|| ob.get("orderbook"))
        .unwrap_or(&null);
    (
        levels_from(book, "yes_dollars", "yes"),
        levels_from(book, "no_dollars", "no"),
    )
}

/// WS `orderbook_snapshot` → `(yes_bids, no_bids, seq)`.
pub fn parse_ws_snapshot(msg: &Value) -> (Vec<(f64, f64)>, Vec<(f64, f64)>, Option<i64>) {
    let null = Value::Null;
    let m = msg.get("msg").unwrap_or(&null);
    (
        levels_from(m, "yes_dollars_fp", "yes"),
        levels_from(m, "no_dollars_fp", "no"),
        msg.get("seq").and_then(Value::as_i64),
    )
}

/// WS `orderbook_delta` → `(side, price[0,1], signed size delta, seq)`. The
/// `delta_fp` is a *signed* change to the resting size at that price level.
pub fn parse_ws_delta(msg: &Value) -> Option<(KSide, f64, f64, Option<i64>)> {
    let m = msg.get("msg")?;
    let side = match m
        .get("side")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "yes" => KSide::Yes,
        "no" => KSide::No,
        _ => return None,
    };
    let price = if let Some(p) = m.get("price_dollars").and_then(num) {
        p
    } else {
        m.get("price").and_then(num)? / 100.0
    };
    let delta = m
        .get("delta_fp")
        .and_then(num)
        .or_else(|| m.get("delta").and_then(num))?;
    Some((side, price, delta, msg.get("seq").and_then(Value::as_i64)))
}

fn micro(p: f64) -> i64 {
    (p * 1_000_000.0).round() as i64
}

fn round6(x: f64) -> f64 {
    (x * 1_000_000.0).round() / 1_000_000.0
}

/// Reflected-YES book for one Kalshi market. `yes`/`no` hold the raw resting bid
/// ladders keyed by micro-price; the public ladder is derived (asks = reflected
/// no-bids).
#[derive(Default, Clone, Debug)]
pub struct KalshiBook {
    yes: BTreeMap<i64, f64>,
    no: BTreeMap<i64, f64>,
    seq: Option<i64>,
    initialized: bool,
}

impl KalshiBook {
    pub fn apply_snapshot(&mut self, yes: &[(f64, f64)], no: &[(f64, f64)], seq: Option<i64>) {
        self.yes.clear();
        self.no.clear();
        for &(p, s) in yes {
            if s > 0.0 {
                self.yes.insert(micro(p), s);
            }
        }
        for &(p, s) in no {
            if s > 0.0 {
                self.no.insert(micro(p), s);
            }
        }
        self.seq = seq;
        self.initialized = true;
    }

    /// Apply a signed delta. Returns `true` if a `seq` gap was detected (caller
    /// should resync — i.e. reconnect for a fresh snapshot).
    pub fn apply_delta(
        &mut self,
        side: KSide,
        price: f64,
        size_delta: f64,
        seq: Option<i64>,
    ) -> bool {
        if !self.initialized {
            return false; // wait for a base snapshot
        }
        let gap = matches!((self.seq, seq), (Some(prev), Some(cur)) if cur != prev + 1);
        self.seq = seq;
        let book = match side {
            KSide::Yes => &mut self.yes,
            KSide::No => &mut self.no,
        };
        let m = micro(price);
        // round6 cancels float-accumulation dust so a fully-consumed level lands
        // at exactly 0 and is removed (else ~1e-14 residue lingers and can cross).
        let new = round6(book.get(&m).copied().unwrap_or(0.0) + size_delta);
        if new <= 0.0 {
            book.remove(&m);
        } else {
            book.insert(m, new);
        }
        gap
    }

    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Reflected **YES** ladder (the Up instrument), prices in [0,1], best-first,
    /// top-`n`. `bids` = raw yes-bids (highest first); `asks` = `1 - no_bid`
    /// (lowest first, from the highest no-bids).
    pub fn top_n(&self, n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        (
            ladder(&self.yes, n),
            reflected(&self.no, n),
        )
    }

    /// Reflected **NO** ladder (the Down/complement instrument) — the mirror of
    /// `top_n`. `bids` = raw no-bids (highest first); `asks` = `1 - yes_bid`
    /// (lowest first). The NO **asks** are the buy-NO liquidity a long-only
    /// executor lifts to open a DOWN position (= the same orders that sit as raw
    /// yes-bids, reframed as a buyable ask).
    pub fn top_n_no(&self, n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        (
            ladder(&self.no, n),
            reflected(&self.yes, n),
        )
    }
}

/// Raw bid side as a best-first ladder in [0,1].
fn ladder(side: &BTreeMap<i64, f64>, n: usize) -> Vec<(f64, f64)> {
    side.iter()
        .rev()
        .take(n)
        .map(|(k, v)| (*k as f64 / 1_000_000.0, *v))
        .collect()
}

/// Opposite side reflected into ask prices `1 - q`, lowest-first (from the
/// highest opposite-bids).
fn reflected(opp: &BTreeMap<i64, f64>, n: usize) -> Vec<(f64, f64)> {
    opp.iter()
        .rev()
        .take(n)
        .map(|(k, v)| (round6(1.0 - (*k as f64 / 1_000_000.0)), *v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn snapshot_reflects_no_bids_into_yes_asks() {
        let mut b = KalshiBook::default();
        // yes bids @ .48/.47 ; no bids @ .51/.50  → yes asks @ .49/.50
        b.apply_snapshot(&[(0.48, 10.0), (0.47, 20.0)], &[(0.51, 5.0), (0.50, 8.0)], Some(1));
        let (bids, asks) = b.top_n(10);
        assert_eq!(bids[0], (0.48, 10.0)); // highest yes bid first
        assert_eq!(asks[0], (0.49, 5.0)); // 1 - 0.51 = 0.49, lowest ask first
        assert_eq!(asks[1], (0.50, 8.0));
    }

    #[test]
    fn no_ladder_mirrors_yes() {
        let mut b = KalshiBook::default();
        // yes bids @ .48/.47 ; no bids @ .51/.50
        b.apply_snapshot(&[(0.48, 10.0), (0.47, 20.0)], &[(0.51, 5.0), (0.50, 8.0)], Some(1));
        let (nb, na) = b.top_n_no(10);
        // NO bids = raw no bids, highest first
        assert_eq!(nb[0], (0.51, 5.0));
        assert_eq!(nb[1], (0.50, 8.0));
        // NO asks = 1 - yes_bid (buy-NO liquidity), lowest first → from highest yes bid
        assert_eq!(na[0], (0.52, 10.0)); // 1 - 0.48
        assert_eq!(na[1], (0.53, 20.0)); // 1 - 0.47
        // YES ask and NO ask complement: best yes ask + best no bid ... and the
        // best NO ask (0.52) = 1 - best yes bid (0.48). ✓
    }

    #[test]
    fn delta_signed_and_seq_gap() {
        let mut b = KalshiBook::default();
        b.apply_snapshot(&[(0.48, 10.0)], &[(0.51, 5.0)], Some(1));
        // +3 to yes @ .48 (seq 2, contiguous → no gap)
        assert!(!b.apply_delta(KSide::Yes, 0.48, 3.0, Some(2)));
        let (bids, _) = b.top_n(10);
        assert_eq!(bids[0], (0.48, 13.0));
        // remove yes @ .48 fully
        assert!(!b.apply_delta(KSide::Yes, 0.48, -13.0, Some(3)));
        assert!(b.top_n(10).0.is_empty());
        // seq jump 3 → 5 is a gap
        assert!(b.apply_delta(KSide::No, 0.51, 1.0, Some(5)));
    }

    #[test]
    fn ws_frames_parse() {
        let snap = json!({
            "type": "orderbook_snapshot", "seq": 7,
            "msg": {"market_ticker": "KXBTC15M-Y", "yes_dollars_fp": [["0.48","10"]], "no_dollars_fp": [["0.51","5"]]}
        });
        let (yes, no, seq) = parse_ws_snapshot(&snap);
        assert_eq!(yes, vec![(0.48, 10.0)]);
        assert_eq!(no, vec![(0.51, 5.0)]);
        assert_eq!(seq, Some(7));

        let delta = json!({
            "type": "orderbook_delta", "seq": 8,
            "msg": {"market_ticker": "KXBTC15M-Y", "side": "yes", "price_dollars": "0.48", "delta_fp": "-4"}
        });
        let (side, p, d, s) = parse_ws_delta(&delta).unwrap();
        assert_eq!(side, KSide::Yes);
        assert_eq!((p, d, s), (0.48, -4.0, Some(8)));
    }

    #[test]
    fn rest_orderbook_cents_fallback() {
        // no *_dollars fields → integer cents path
        let ob = json!({"orderbook": {"yes": [[48, 10]], "no": [[51, 5]]}});
        let (yes, no) = parse_rest_orderbook(&ob);
        assert_eq!(yes, vec![(0.48, 10.0)]);
        assert_eq!(no, vec![(0.51, 5.0)]);
    }
}
