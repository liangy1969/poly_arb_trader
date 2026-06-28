//! Canonical market/signal payloads that cross the bus (DESIGN §8).

use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Prediction-market lifecycle phase (catalog events, DESIGN §5/§8.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MarketStatus {
    Upcoming,
    Live,
    Expired,
    Resolved,
}

/// Top-of-book / top-N snapshot. `bids` highest-first, `asks` lowest-first.
#[derive(Clone, Debug, Serialize)]
pub struct BookUpdate {
    pub instrument: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub update_id: Option<u64>,
    pub exch_ts_ns: i64,
    pub recv_ts_ns: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TradeTick {
    pub instrument: String,
    pub price: f64,
    pub qty: f64,
    pub side: Side,
    pub exch_ts_ns: i64,
    pub recv_ts_ns: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Liquidation {
    pub instrument: String,
    pub price: f64,
    pub qty: f64,
    pub side: Side,
    pub recv_ts_ns: i64,
}

/// Catalog/lifecycle event for a prediction market (drives the InstrumentLinker).
#[derive(Clone, Debug, Serialize)]
pub struct MarketMeta {
    pub instrument: String,
    pub kind: String,
    pub status: MarketStatus,
    pub start_ts_ns: Option<i64>,
    pub expiry_ts_ns: Option<i64>,
    pub winner: Option<String>,
    /// Per-market economics from the venue catalog (DESIGN §13/§15). `None` when
    /// the source didn't supply them → consumers fall back to config defaults.
    pub min_order_size: Option<f64>,
    pub tick_size: Option<f64>,
    pub fee_rate: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Trigger {
    pub move_bps: f64,
    pub window_ms: u64,
    pub yes_price: f64,
    /// The target (YES) token's mid change over the SAME trigger window, in cents.
    /// Measures whether the prediction token already reacted to the perp move by
    /// the time the signal fired (NaN if the token book didn't span the window).
    pub target_move_c: f64,
}

/// Emitted by the rule engine on `signal.<strategy>`; consumed by the executor
/// (later) — for now it is logged. `plan` fields kept minimal until the
/// executor lands (DESIGN_EXECUTION §3).
#[derive(Clone, Debug, Serialize)]
pub struct TradeSignal {
    pub strategy: String,
    pub ts_ns: i64,
    pub reason: String,
    pub reference: String,
    pub target: String,
    pub direction: i8,
    pub trigger: Trigger,
    pub hold_ms: u64,
    pub ttl_ms: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct LastTrade {
    pub price: f64,
    pub qty: f64,
    pub side: Side,
    pub ts_ns: i64,
}

/// One raw latency sample, published per trade signal on
/// `latency.<source>.<name>` and persisted by the recorder like any other
/// event. `origin_ts_ns` is the triggering book tick's collector recv time
/// (== the signal's `ts_ns`), so a sample joins back to its signal.
#[derive(Clone, Debug, Serialize)]
pub struct LatencySample {
    /// Which measurement, e.g. `book_to_signal`.
    pub name: String,
    /// Measured latency in microseconds.
    pub latency_us: f64,
    /// Collector recv time of the originating book tick (== signal `ts_ns`).
    pub origin_ts_ns: i64,
    pub strategy: String,
    pub target: String,
}

// ───────────────────────── execution-side records (DESIGN_EXECUTION §11) ─────

/// Terminal disposition of a Trade (one round trip).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TradeOutcome {
    /// Entered and fully exited.
    Closed,
    /// Entry never filled; no position ever existed.
    Abandoned,
    /// Held to market resolution and settled by payout.
    Settled,
    /// Still long at the exit deadline; awaiting resolution (incident-grade).
    PendingResolution,
}

/// Aggregate of one leg's fills (entry or exit), for the `TradeRecord`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct LegSummary {
    pub qty: f64,
    pub vwap: f64,
    pub fees: f64,
    pub n_fills: u32,
    pub first_ts_ns: i64,
    pub last_ts_ns: i64,
}

/// The backtest row, measured live — column-compatible (DESIGN_EXECUTION §11).
/// Published on `exec.trade` at every terminal trade state.
#[derive(Clone, Debug, Serialize)]
pub struct TradeRecord {
    pub trade_id: String,
    pub outcome: TradeOutcome,
    pub direction: i8,
    pub instrument: String,
    pub signal_ts_ns: i64,
    pub trigger: Trigger,
    pub entry: LegSummary,
    pub exit: LegSummary,
    pub hold_actual_ms: u64,
    pub pnl_gross: f64,
    pub pnl_net: f64,
    pub slippage_entry_c: f64,
    pub slippage_exit_c: f64,
    pub lat_signal_to_submit_ms: f64,
    pub lat_submit_to_ack_ms: f64,
    pub lat_ack_to_fill_ms: f64,
}

/// Per-order / per-transition report (DESIGN_EXECUTION §11), on `exec.report`.
/// `state` is the trade- or order-state name; `detail` is human context.
#[derive(Clone, Debug, Serialize)]
pub struct ExecReport {
    pub trade_id: String,
    pub instrument: String,
    pub state: String,
    pub detail: String,
    pub ts_ns: i64,
}

/// Position Manager snapshot (DESIGN_EXECUTION §8), on `exec.position`.
#[derive(Clone, Debug, Serialize)]
pub struct PositionSnapshot {
    pub instrument: String,
    pub qty: f64,
    pub avg_cost: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub mark_px: f64,
    pub upnl: f64,
    pub status: String,
    pub cash_usdc: f64,
}
