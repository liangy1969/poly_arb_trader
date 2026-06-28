//! arb-collector-binance — perp depth (order book only) signal feed, routed
//! through the SOCKS5 proxy (DESIGN §5). Ports the Python reference's sequenced
//! L2 book + resync.

pub mod book;
pub mod collector;

pub use collector::{BinanceCfg, BinanceCollector};
