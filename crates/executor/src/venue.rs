//! `TradingVenue` (DESIGN_EXECUTION §10) — the one boundary the executor talks
//! to a market through. v1 ships `SimVenue`; the real `PolymarketClob` (EIP-712
//! signing, L2 auth, FAK/GTD, reconcile) is P2 and reads account secrets per
//! `SETUP_POLYMARKET.md`.
//!
//! Fills arrive **asynchronously** on a channel — `submit` returns the immediate
//! FAK result, but a venue also produces fills out-of-band via `start`: the sim
//! runs a **liquidator** that sells any open position at the bid near expiry; a
//! live adapter would run the user-channel WS listener + `GET /data/trades` pull.
//! Same channel, same consumer (§7/§8.6).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use tokio::task::JoinHandle;

use arb_core::model::Side;
use arb_core::now_ns;

use crate::types::{BookSource, Fill, FillStatus, IntentKind, OrderIntent, VenueOutcome};

const EPS: f64 = 1e-9;
const MS: i64 = 1_000_000;

/// Async fill stream from a venue to the executor (the §8.6 ingestion path).
pub type FillSender = tokio::sync::mpsc::UnboundedSender<Fill>;

#[async_trait]
pub trait TradingVenue: Send + Sync {
    /// Submit one order; returns the immediate FAK result. The venue holds its
    /// own book/connection — fills against the **latest** book.
    async fn submit(&self, intent: &OrderIntent) -> VenueOutcome;
    /// Spawn the venue's background fill producers, pushing onto `fills`. Sim: a
    /// near-expiry liquidator. Live: the user-channel WS listener + REST pull.
    /// Returns their join handles for the caller to manage.
    fn start(&self, fills: FillSender) -> Vec<JoinHandle<()>>;
    fn name(&self) -> &'static str;

    // ── Maker-exit primitives (default: unsupported) ──
    /// Place a resting (GTC, post-only) maker limit order at `intent.price`.
    /// Returns the venue `order_id`, or an error (e.g. rejected because it would
    /// cross). Only the live adapters implement this.
    async fn place_resting(&self, _intent: &OrderIntent) -> Result<String, String> {
        Err("venue does not support resting maker orders".into())
    }
    /// Cancel a resting order; returns the contracts filled before the cancel took
    /// effect (so a maker exit never loses a last-moment fill on cancel).
    async fn cancel_order(&self, _order_id: &str) -> Result<f64, String> {
        Err("venue does not support cancel".into())
    }
    /// Contracts filled so far on a resting order (poll for maker-exit fills).
    async fn order_fill_count(&self, _order_id: &str) -> Result<f64, String> {
        Err("venue does not support order status".into())
    }
}

/// Taker fee in USDC, Polymarket's current formula (docs.polymarket.com/trading/
/// fees + the market's `feeSchedule`): `fee = feeRate · p · (1−p) · qty`.
pub fn fee_usdc(rate: f64, px: f64, qty: f64) -> f64 {
    rate * px * (1.0 - px) * qty
}

/// Sweep a bid ladder (highest-first) to SELL `size`, taking every level at or
/// above `floor`. Returns `(filled_qty, vwap)` — the realized exit price with
/// multi-level slippage. Used by the venue and the hold-period probe.
pub fn sweep_sell(bids: &[(f64, f64)], size: f64, floor: f64) -> (f64, f64) {
    let mut remaining = size;
    let mut qty = 0.0;
    let mut notional = 0.0;
    for &(px, sz) in bids {
        if remaining <= EPS || px <= 0.0 || sz <= 0.0 || px < floor - EPS {
            break;
        }
        let take = remaining.min(sz);
        qty += take;
        notional += take * px;
        remaining -= take;
    }
    (qty, if qty > EPS { notional / qty } else { 0.0 })
}

/// Per-token net open position the venue tracks (it's the market — it knows what
/// it filled), so the liquidator can clean up near expiry.
#[derive(Clone, Copy)]
struct NetPos {
    qty: f64,
    expiry_ns: i64,
    fee_rate: f64,
}

/// In-process fill simulator. `submit` is an honest FAK against the **post-delay**
/// book (250 ms taker-delay floor, multi-level sweep, taker fee). It does **not**
/// fabricate liquidity. The "you can always get out before it resolves" model is
/// the separate `run_liquidator` task (DESIGN sim model): near each market's
/// expiry it sells the remaining position at the current best bid, delivered as
/// an async fill — the sim's stand-in for a live venue's WS fill stream.
pub struct SimVenue {
    taker_delay_ms: u64,
    force_liquidity: bool,
    force_window_ms: u64,
    force_check_ms: u64,
    source: Arc<dyn BookSource>,
    net: Arc<Mutex<HashMap<String, NetPos>>>,
    seq: Arc<AtomicU64>,
}

impl SimVenue {
    pub fn new(
        taker_delay_ms: u64,
        force_liquidity: bool,
        force_window_ms: u64,
        force_check_ms: u64,
        source: Arc<dyn BookSource>,
    ) -> Self {
        SimVenue {
            taker_delay_ms,
            force_liquidity,
            force_window_ms,
            force_check_ms,
            source,
            net: Arc::new(Mutex::new(HashMap::new())),
            seq: Arc::new(AtomicU64::new(0)),
        }
    }

    fn next_id(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Net open qty the venue believes it holds for `instrument` (test/diagnostic).
    pub fn net_qty(&self, instrument: &str) -> f64 {
        self.net.lock().unwrap().get(instrument).map(|n| n.qty).unwrap_or(0.0)
    }
}

#[async_trait]
impl TradingVenue for SimVenue {
    async fn submit(&self, intent: &OrderIntent) -> VenueOutcome {
        if !matches!(intent.kind, IntentKind::TakeNow) {
            return VenueOutcome::Rejected("sim v1 supports TakeNow (FAK) only".into());
        }
        if intent.size < intent.params.min_order_size {
            return VenueOutcome::Rejected(format!(
                "size {:.2} < min {:.2}",
                intent.size, intent.params.min_order_size
            ));
        }

        // Mandatory taker-delay floor: held before the order resolves.
        tokio::time::sleep(Duration::from_millis(self.taker_delay_ms)).await;

        let order_id = format!("sim-ord-{}", self.next_id());
        // Match against the LATEST book (re-read after the delay), sweeping every
        // level the cap/floor allows — a marketable FAK takes all displayed
        // liquidity up to its size. No fabricated liquidity here.
        let Some(book) = self.source.depth(&intent.instrument) else {
            return VenueOutcome::Acked { order_id, fills: Vec::new() };
        };
        let levels = match intent.side {
            Side::Buy => &book.asks,  // lowest-first; take while px <= cap
            Side::Sell => &book.bids, // highest-first; take while px >= floor
        };
        let mut remaining = intent.size;
        let mut qty = 0.0;
        let mut notional = 0.0;
        for &(px, sz) in levels {
            if remaining <= EPS || px <= 0.0 || sz <= 0.0 {
                break;
            }
            let within = match intent.side {
                Side::Buy => px <= intent.price + EPS,
                Side::Sell => px >= intent.price - EPS,
            };
            if !within {
                break; // levels are best-first; no deeper level can qualify
            }
            let take = remaining.min(sz);
            qty += take;
            notional += take * px;
            remaining -= take;
        }

        let fills = if qty > EPS {
            let vwap = notional / qty;
            vec![Fill {
                venue_trade_id: format!("sim-trd-{}", self.next_id()),
                order_id: order_id.clone(),
                client_id: intent.client_id.clone(),
                instrument: intent.instrument.clone(),
                status: FillStatus::Confirmed, // sim settles immediately
                side: intent.side,
                qty,
                px: vwap,
                fee: fee_usdc(intent.params.fee_rate, vwap, qty),
                ts_ns: now_ns(),
            }]
        } else {
            Vec::new() // FAK with no cross: acked, nothing resting
        };

        // Track net position so the liquidator can clean it up near expiry.
        if let Some(f) = fills.first() {
            let mut net = self.net.lock().unwrap();
            let e = net.entry(intent.instrument.clone()).or_insert(NetPos {
                qty: 0.0,
                expiry_ns: intent.expiry_ns,
                fee_rate: intent.params.fee_rate,
            });
            e.expiry_ns = intent.expiry_ns;
            e.fee_rate = intent.params.fee_rate;
            match f.side {
                Side::Buy => e.qty += f.qty,
                Side::Sell => e.qty -= f.qty,
            }
        }

        VenueOutcome::Acked { order_id, fills }
    }

    fn start(&self, fills: FillSender) -> Vec<JoinHandle<()>> {
        if !self.force_liquidity {
            return Vec::new();
        }
        let net = self.net.clone();
        let source = self.source.clone();
        let seq = self.seq.clone();
        let window_ns = self.force_window_ms as i64 * MS;
        let check = Duration::from_millis(self.force_check_ms.max(1));

        // Liquidator: near each market's expiry, sell the remaining position at
        // the current best bid and emit the fill async (the venue's fill stream).
        let handle = tokio::spawn(async move {
            let mut tick = tokio::time::interval(check);
            loop {
                tick.tick().await;
                let now = now_ns();
                let due: Vec<String> = {
                    let g = net.lock().unwrap();
                    g.iter()
                        .filter(|(_, np)| np.qty > EPS && np.expiry_ns > 0 && now >= np.expiry_ns - window_ns)
                        .map(|(inst, _)| inst.clone())
                        .collect()
                };
                for inst in due {
                    let bid = source.top(&inst).map(|b| b.best_bid).unwrap_or(0.0);
                    if bid <= 0.0 {
                        continue; // no bid to fill against → retry next tick
                    }
                    // Atomically claim the CURRENT net (not the stale snapshot): if a
                    // concurrent normal exit already sold it, qty is 0 here → skip, so
                    // we never emit a phantom liquidation for an already-flat position.
                    let (qty, fee_rate) = {
                        let mut g = net.lock().unwrap();
                        match g.get_mut(&inst) {
                            Some(np) if np.qty > EPS => {
                                let q = np.qty;
                                let fr = np.fee_rate;
                                np.qty = 0.0;
                                (q, fr)
                            }
                            _ => continue,
                        }
                    };
                    let id = seq.fetch_add(1, Ordering::Relaxed);
                    let fill = Fill {
                        venue_trade_id: format!("sim-liq-{id}"),
                        order_id: format!("sim-liq-{id}"),
                        client_id: format!("liq:{inst}"),
                        instrument: inst.clone(),
                        status: FillStatus::Confirmed,
                        side: Side::Sell,
                        qty,
                        px: bid,
                        fee: fee_usdc(fee_rate, bid, qty),
                        ts_ns: now_ns(),
                    };
                    if fills.send(fill).is_err() {
                        return; // consumer gone
                    }
                    // net already zeroed atomically in the claim above.
                }
            }
        });
        vec![handle]
    }

    fn name(&self) -> &'static str {
        "sim"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BookDepth, BookTop, MarketParams, OrderIntent};
    use tokio::sync::mpsc::unbounded_channel;

    fn params(fee_rate: f64) -> MarketParams {
        MarketParams { min_order_size: 5.0, tick_size: 0.01, fee_rate }
    }

    /// Fixed book source (same book for any instrument).
    struct Fixed(Option<BookDepth>);
    impl BookSource for Fixed {
        fn top(&self, _: &str) -> Option<BookTop> {
            self.0.as_ref().map(|d| BookTop {
                best_bid: d.bids.first().map(|&(p, _)| p).unwrap_or(0.0),
                bid_sz: d.bids.first().map(|&(_, s)| s).unwrap_or(0.0),
                best_ask: d.asks.first().map(|&(p, _)| p).unwrap_or(0.0),
                ask_sz: d.asks.first().map(|&(_, s)| s).unwrap_or(0.0),
                recv_ts_ns: d.recv_ts_ns,
            })
        }
        fn depth(&self, _: &str) -> Option<BookDepth> {
            self.0.clone()
        }
    }

    /// A sim venue over a fixed book (liquidator off unless `force`).
    fn sim(book: Option<BookDepth>, force: bool) -> SimVenue {
        SimVenue::new(0, force, 30_000, 5, Arc::new(Fixed(book)))
    }

    fn intent(side: Side, price: f64, size: f64) -> OrderIntent {
        intent_fee(side, price, size, 0.0)
    }

    fn intent_fee(side: Side, price: f64, size: f64, fee_rate: f64) -> OrderIntent {
        OrderIntent {
            client_id: "t-1:E:0".into(),
            instrument: "polymarket.0xabc.UP".into(),
            token_id: "tok".into(),
            side,
            price,
            size,
            kind: IntentKind::TakeNow,
            params: params(fee_rate),
            expiry_ns: i64::MAX,
        }
    }

    fn book() -> BookDepth {
        BookDepth { bids: vec![(0.40, 50.0)], asks: vec![(0.42, 30.0)], recv_ts_ns: 0 }
    }

    #[tokio::test]
    async fn buy_fills_at_ask_capped_by_depth() {
        let v = sim(Some(book()), false);
        match v.submit(&intent(Side::Buy, 0.42, 20.0)).await {
            VenueOutcome::Acked { fills, .. } => {
                assert_eq!(fills.len(), 1);
                assert!((fills[0].px - 0.42).abs() < 1e-9);
                assert!((fills[0].qty - 20.0).abs() < 1e-9);
            }
            o => panic!("expected Acked fill, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn buy_above_cap_does_not_fill() {
        let v = sim(Some(book()), false);
        match v.submit(&intent(Side::Buy, 0.41, 20.0)).await {
            VenueOutcome::Acked { fills, .. } => assert!(fills.is_empty()),
            o => panic!("expected Acked no-fill, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn sell_fills_at_bid() {
        let v = sim(Some(book()), false);
        match v.submit(&intent(Side::Sell, 0.40, 20.0)).await {
            VenueOutcome::Acked { fills, .. } => assert!((fills[0].px - 0.40).abs() < 1e-9),
            o => panic!("expected Acked fill, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn fill_uses_post_delay_book() {
        // The venue holds the (post-delay) book; ask 0.45 > cap 0.42 -> no fill.
        let moved = BookDepth { bids: vec![(0.43, 50.0)], asks: vec![(0.45, 30.0)], recv_ts_ns: 0 };
        let v = sim(Some(moved), false);
        match v.submit(&intent(Side::Buy, 0.42, 20.0)).await {
            VenueOutcome::Acked { fills, .. } => assert!(fills.is_empty(), "should miss: ask past cap"),
            o => panic!("expected Acked no-fill, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn sweep_fills_multiple_bid_levels() {
        // 25 across three bid levels: 10@0.40 + 10@0.39 + 5@0.38 -> vwap 0.392.
        let deep = BookDepth {
            bids: vec![(0.40, 10.0), (0.39, 10.0), (0.38, 10.0)],
            asks: vec![(0.42, 30.0)],
            recv_ts_ns: 0,
        };
        let v = sim(Some(deep), false);
        match v.submit(&intent(Side::Sell, 0.01, 25.0)).await {
            VenueOutcome::Acked { fills, .. } => {
                assert!((fills[0].qty - 25.0).abs() < 1e-9);
                assert!((fills[0].px - 0.392).abs() < 1e-6, "vwap was {}", fills[0].px);
            }
            o => panic!("expected Acked sweep fill, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn submit_is_book_only_no_fabricated_liquidity() {
        // Even with the liquidator ON, submit only fills real book depth (15),
        // leaving a residual — fabrication is the liquidator's job, not submit's.
        let thin = BookDepth { bids: vec![(0.40, 10.0), (0.39, 5.0)], asks: vec![(0.42, 30.0)], recv_ts_ns: 0 };
        let v = sim(Some(thin), true);
        let mut it = intent(Side::Sell, 0.01, 25.0);
        it.expiry_ns = i64::MAX;
        match v.submit(&it).await {
            VenueOutcome::Acked { fills, .. } => assert!((fills[0].qty - 15.0).abs() < 1e-9),
            o => panic!("expected book-only partial, got {o:?}"),
        }
    }

    #[tokio::test]
    async fn submit_tracks_net_position() {
        let v = sim(Some(book()), false);
        let inst = "polymarket.0xabc.UP";
        v.submit(&intent(Side::Buy, 0.42, 20.0)).await; // +20
        assert!((v.net_qty(inst) - 20.0).abs() < 1e-9);
        v.submit(&intent(Side::Sell, 0.40, 4.0)).await; // -4 (4 >= min 5? no — use 6)
        // (sell 4 is below min 5 → rejected, net unchanged)
        assert!((v.net_qty(inst) - 20.0).abs() < 1e-9);
        v.submit(&intent(Side::Sell, 0.40, 6.0)).await; // -6
        assert!((v.net_qty(inst) - 14.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn liquidator_sells_residual_at_bid_near_expiry() {
        let bk = BookDepth { bids: vec![(0.40, 5.0)], asks: vec![(0.42, 50.0)], recv_ts_ns: 0 };
        let v = sim(Some(bk), true);
        // buy 20 to set the net, with an already-passed expiry (within the window)
        let mut it = intent(Side::Buy, 0.42, 20.0);
        it.expiry_ns = 1;
        v.submit(&it).await;
        assert!((v.net_qty(&it.instrument) - 20.0).abs() < 1e-9);

        let (tx, mut rx) = unbounded_channel();
        let handles = v.start(tx);
        let fill = tokio::time::timeout(Duration::from_millis(300), rx.recv())
            .await
            .expect("liquidator timeout")
            .expect("channel closed");
        assert!(matches!(fill.side, Side::Sell));
        assert!((fill.qty - 20.0).abs() < 1e-9);
        assert!((fill.px - 0.40).abs() < 1e-9, "sold at best bid");
        for h in handles {
            h.abort();
        }
    }

    #[tokio::test]
    async fn liquidator_skips_when_normal_exit_already_flat() {
        // Buy 20 then sell 20 via submit (the normal exit) → net flat. The
        // liquidator must find nothing to claim and emit NO phantom fill.
        let bk = BookDepth { bids: vec![(0.40, 50.0)], asks: vec![(0.42, 50.0)], recv_ts_ns: 0 };
        let v = sim(Some(bk), true);
        let mut buy = intent(Side::Buy, 0.42, 20.0);
        buy.expiry_ns = 1; // already within the liquidator window
        v.submit(&buy).await;
        let mut sell = intent(Side::Sell, 0.40, 20.0);
        sell.expiry_ns = 1;
        v.submit(&sell).await;
        assert!(v.net_qty(&buy.instrument).abs() < 1e-9, "normal exit flattened the net");

        let (tx, mut rx) = unbounded_channel();
        let handles = v.start(tx);
        let got = tokio::time::timeout(Duration::from_millis(60), rx.recv()).await;
        assert!(got.is_err(), "liquidator must not emit for an already-flat position");
        for h in handles {
            h.abort();
        }
    }

    #[test]
    fn sweep_sell_walks_the_book_with_slippage() {
        let bids = [(0.40, 10.0), (0.39, 10.0), (0.38, 10.0)];
        // 25 shares: 10@0.40 + 10@0.39 + 5@0.38 = vwap 0.392 (slippage below the bid)
        let (q, vwap) = sweep_sell(&bids, 25.0, 0.0);
        assert!((q - 25.0).abs() < 1e-9);
        assert!((vwap - 0.392).abs() < 1e-6, "vwap {vwap}");
        // capped by total depth (30) when asking for more
        let (q2, _) = sweep_sell(&bids, 40.0, 0.0);
        assert!((q2 - 30.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn crypto_fee_matches_polymarket_formula() {
        // feeRate=0.07, p=0.40: fee = 0.07 * 0.40 * 0.60 * 20 = 0.336 USDC.
        let v = sim(Some(book()), false);
        match v.submit(&intent_fee(Side::Sell, 0.40, 20.0, 0.07)).await {
            VenueOutcome::Acked { fills, .. } => {
                assert!((fills[0].fee - 0.336).abs() < 1e-9, "fee was {}", fills[0].fee);
            }
            o => panic!("expected Acked fill, got {o:?}"),
        }
    }
}
