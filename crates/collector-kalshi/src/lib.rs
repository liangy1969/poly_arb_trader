//! arb-collector-kalshi — reflected-YES book + catalog + trade tape for the
//! configured Kalshi crypto series (e.g. `KXBTC15M`). Ports the Python
//! reference's Kalshi collector: REST discovery, RSA-PSS-signed `orderbook_delta`
//! WS (canonical book), REST `/orderbook` fallback, and public trade-tape poll.
//! Publishes `market.kalshi.catalog` + `market.kalshi.{ticker}.book` / `.trade`,
//! mirroring the Polymarket collector's bus contract (DESIGN §5).

pub mod auth;
pub mod book;
pub mod collector;

pub use collector::{KalshiCfg, KalshiCollector};
