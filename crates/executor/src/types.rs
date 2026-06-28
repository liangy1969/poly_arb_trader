//! Executor domain types (DESIGN_EXECUTION §3–§4). Plans, order intents, and
//! fills — the vocabulary the trade state machine, venue, and PM exchange.

use arb_core::model::{Side, Trigger};

/// ns-per-ms — the one place duration knobs (`_ms`) cross into the `_ns` clock.
pub const MS: i64 = 1_000_000;

// Instrument↔venue helpers (`traded_instrument`, `market_id_of`, `VenueSpec`) live
// in `venue_spec` so the trade path carries no venue string literal.

/// Per-market economics (DESIGN_EXECUTION §13/§15), resolved from the catalog
/// meta with config fallback. Threaded into orders/plan so sizing, pricing, and
/// fees all use the real per-market values rather than globals.
#[derive(Clone, Copy, Debug)]
pub struct MarketParams {
    pub min_order_size: f64,
    pub tick_size: f64,
    pub fee_rate: f64,
}

/// The three CLOB intent shapes (DESIGN_EXECUTION §4.1). v1 uses `TakeNow` only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntentKind {
    /// Marketable limit, FAK — immediate, accepts partials, rests nothing.
    TakeNow,
    /// Passive limit, GTD (passive exit mode, P3).
    RestUntil { expiry_ns: i64 },
    /// Cancel by order id.
    Cancel,
}

/// One venue order attempt.
#[derive(Clone, Debug)]
pub struct OrderIntent {
    /// `{trade_id}:{leg}:{attempt}` — our handle (DESIGN_EXECUTION §4.2).
    pub client_id: String,
    pub instrument: String,
    pub token_id: String,
    pub side: Side,
    /// Worst acceptable price: cap on BUY, floor on SELL.
    pub price: f64,
    pub size: f64,
    pub kind: IntentKind,
    /// Per-market economics the venue charges/enforces (fee, min size).
    pub params: MarketParams,
    /// Market resolution time — the sim venue provides forced sell liquidity
    /// only while `now < expiry_ns` (you can always get out before it resolves).
    pub expiry_ns: i64,
}

/// On-chain settlement status of a fill (DESIGN_EXECUTION §8.2). The sim venue
/// emits `Confirmed` directly; live distinguishes matched→confirmed→failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillStatus {
    Matched,
    Confirmed,
    Failed,
}

/// One match/fill event. `qty` is shares **received** (fee-reduced on BUY).
#[derive(Clone, Debug)]
pub struct Fill {
    pub venue_trade_id: String,
    pub order_id: String,
    pub client_id: String,
    pub instrument: String,
    pub status: FillStatus,
    pub side: Side,
    pub qty: f64,
    pub px: f64,
    pub fee: f64,
    pub ts_ns: i64,
}

/// Top-of-book snapshot a live order is priced/sized against (the ExecBookMirror
/// projection, DESIGN_EXECUTION §2.2/§3.2).
#[derive(Clone, Copy, Debug)]
pub struct BookTop {
    pub best_bid: f64,
    pub bid_sz: f64,
    pub best_ask: f64,
    pub ask_sz: f64,
    pub recv_ts_ns: i64,
}

/// Full top-N ladder for sweeping fills. `bids` highest-first, `asks` lowest-first.
#[derive(Clone, Debug, Default)]
pub struct BookDepth {
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub recv_ts_ns: i64,
}

/// A live view the venue can re-read **after** its internal taker delay, so the
/// fill matches against the latest book (not the submit-time snapshot). The real
/// adapter matches against the exchange book and ignores this; the sim reads the
/// `ExecBookMirror` through it. `depth` lets a marketable order sweep multiple
/// levels (entry caps within top depth; exit sell-out sweeps the ladder).
pub trait BookSource: Send + Sync {
    fn top(&self, instrument: &str) -> Option<BookTop>;
    fn depth(&self, instrument: &str) -> Option<BookDepth>;
}

/// Outcome of a venue `submit` (DESIGN_EXECUTION §5/§7).
#[derive(Clone, Debug)]
pub enum VenueOutcome {
    /// Order acked; zero or more fills (FAK: full / partial / none-but-acked).
    Acked { order_id: String, fills: Vec<Fill> },
    /// Pre-trade / venue reject (balance, min-size, rate-limit, …).
    Rejected(String),
}

/// The execution plan built from a gated signal (DESIGN_EXECUTION §3.1).
#[derive(Clone, Debug)]
pub struct TradePlan {
    pub trade_id: String,
    pub instrument: String,
    pub token_id: String,
    pub direction: i8,
    pub size_shares: f64,
    pub hold_ms: u64,
    pub exit_deadline_ns: i64,
    pub signal_ts_ns: i64,
    pub trigger: Trigger,
    /// Traded-token best ask at decision time (entry-slippage baseline).
    pub signal_ask: f64,
    /// Per-market economics resolved from catalog meta (config fallback).
    pub params: MarketParams,
    /// Market resolution time (catalog meta), for the venue's force-liquidity gate.
    pub expiry_ns: i64,
}

// Instrument-parsing tests live in `venue_spec` (the home of those functions).
