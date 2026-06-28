//! arb-executor — turns `signal.<strategy>` into trades (DESIGN_EXECUTION).
//!
//! v1: full trade state machine (Entering→Holding→Exiting→Closed/Abandoned),
//! one-trade gate, idempotent Position Manager, risk gate, and an
//! `ExecBookMirror`, driven by a **simulated** `TradingVenue` (no funds/secrets).
//! The real `PolymarketClob` adapter (EIP-712 signing + L2 auth) is P2 — see
//! `SETUP_POLYMARKET.md`.

pub mod config;
pub mod mirror;
pub mod module;
pub mod position;
pub mod risk;
pub mod types;
pub mod venue;
pub mod venue_kalshi;
pub mod venue_spec;

pub use config::ExecutorCfg;
pub use module::Executor;
pub use venue::{SimVenue, TradingVenue};
pub use venue_kalshi::KalshiVenue;
