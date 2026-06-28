//! Stateful per-token L2 book + WS frame parsing (ports `book.py` /
//! `normalizer.py`). Prices are kept in **[0,1]** (USDC/share) — our canonical
//! unit — not the Python reference's cents.

use std::collections::BTreeMap;

use arb_core::model::Side;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelSide {
    Bid,
    Ask,
}

/// Parse a number that Polymarket may send as a JSON number or string.
pub fn num(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
}

/// Parse a raw level array into `(price[0,1], size)` tuples, **best-first**
/// (bids highest-first, asks lowest-first). Polymarket sends them worst-first.
pub fn parse_levels(v: &Value, side: LevelSide) -> Vec<(f64, f64)> {
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
            if let (Some(p), Some(s)) = (p, s) {
                if s > 0.0 {
                    out.push((p, s));
                }
            }
        }
    }
    match side {
        LevelSide::Bid => out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)),
        LevelSide::Ask => out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)),
    }
    out
}

/// One `price_change` level update: `(price[0,1], size, side)`; `size==0` removes.
pub fn parse_price_changes(msg: &Value) -> Vec<(String, f64, f64, Side)> {
    let changes = msg
        .get("price_changes")
        .or_else(|| msg.get("changes"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for ch in &changes {
        let asset = ch.get("asset_id").and_then(Value::as_str).unwrap_or("");
        let (p, s) = (num(&ch["price"]), num(&ch["size"]));
        let side = match ch.get("side").and_then(Value::as_str).unwrap_or("").to_ascii_uppercase().as_str() {
            "BUY" => Some(Side::Buy),
            "SELL" => Some(Side::Sell),
            _ => None,
        };
        if let (false, Some(p), Some(s), Some(side)) = (asset.is_empty(), p, s, side) {
            out.push((asset.to_string(), p, s, side));
        }
    }
    out
}

fn micro(p: f64) -> i64 {
    (p * 1_000_000.0).round() as i64
}

/// L2 book for one outcome token. `Side::Buy` = bid, `Side::Sell` = ask.
#[derive(Default, Clone, Debug)]
pub struct PolyBook {
    bids: BTreeMap<i64, f64>,
    asks: BTreeMap<i64, f64>,
}

impl PolyBook {
    pub fn apply_snapshot(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        self.bids.clear();
        self.asks.clear();
        for &(p, s) in bids {
            if s > 0.0 {
                self.bids.insert(micro(p), s);
            }
        }
        for &(p, s) in asks {
            if s > 0.0 {
                self.asks.insert(micro(p), s);
            }
        }
    }

    pub fn apply_delta(&mut self, changes: &[(f64, f64, Side)]) {
        for &(p, s, side) in changes {
            let m = micro(p);
            let book = match side {
                Side::Buy => &mut self.bids,
                Side::Sell => &mut self.asks,
            };
            if s == 0.0 {
                book.remove(&m);
            } else {
                book.insert(m, s);
            }
        }
    }

    /// Top-N levels, best-first, prices back in [0,1].
    pub fn top_n(&self, n: usize) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
        let bids = self
            .bids
            .iter()
            .rev()
            .take(n)
            .map(|(k, v)| (*k as f64 / 1_000_000.0, *v))
            .collect();
        let asks = self
            .asks
            .iter()
            .take(n)
            .map(|(k, v)| (*k as f64 / 1_000_000.0, *v))
            .collect();
        (bids, asks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn levels_sorted_best_first_in_unit_prices() {
        let bids = json!([{"price": "0.49", "size": "10"}, {"price": "0.50", "size": "5"}]);
        let asks = json!([{"price": "0.52", "size": "8"}, {"price": "0.51", "size": "3"}]);
        let b = parse_levels(&bids, LevelSide::Bid);
        let a = parse_levels(&asks, LevelSide::Ask);
        assert_eq!(b[0], (0.50, 5.0)); // highest bid first
        assert_eq!(a[0], (0.51, 3.0)); // lowest ask first
    }

    #[test]
    fn snapshot_then_delta() {
        let mut book = PolyBook::default();
        book.apply_snapshot(&[(0.49, 10.0), (0.48, 20.0)], &[(0.51, 8.0), (0.52, 4.0)]);
        let (b, a) = book.top_n(10);
        assert_eq!(b[0], (0.49, 10.0));
        assert_eq!(a[0], (0.51, 8.0));

        // bump best bid to 0.50, remove ask @0.51
        book.apply_delta(&[(0.50, 7.0, Side::Buy), (0.51, 0.0, Side::Sell)]);
        let (b, a) = book.top_n(10);
        assert_eq!(b[0], (0.50, 7.0));
        assert_eq!(a[0], (0.52, 4.0));
    }

    #[test]
    fn price_changes_parsed_by_asset() {
        let msg = json!({
            "event_type": "price_change",
            "market": "0xabc",
            "price_changes": [
                {"asset_id": "tok_up", "price": "0.50", "size": "7", "side": "BUY"},
                {"asset_id": "tok_up", "price": "0.51", "size": "0", "side": "SELL"}
            ]
        });
        let chs = parse_price_changes(&msg);
        assert_eq!(chs.len(), 2);
        assert_eq!(chs[0].0, "tok_up");
        assert_eq!(chs[0].3, Side::Buy);
        assert_eq!(chs[1].2, 0.0);
    }
}
