//! arb-collector-polymarket — tradeable book + catalog for the configured
//! short-term series (DESIGN §5). Reimplements the Python reference's
//! slug-prediction discovery, stateful CLOB book, and lifecycle.

pub mod book;
pub mod collector;
pub mod slug;

pub use collector::{MarketInfo, PolyCfg, PolyCollector};
