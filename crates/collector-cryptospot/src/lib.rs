//! arb-collector-cryptospot — free, public, 24/7 US spot BTC reference feeds for the
//! lead-lag study: Coinbase Exchange `ticker` (no auth) and Binance.US `bookTicker`
//! (no auth). Each publishes the standard top-of-book `market.<venue>.BTC.book`
//! (`coinbase.BTC`, `binanceus.BTC`) so the processor/recorder consume them like any
//! other reference. No proxy (US-direct); these are the likeliest predators of Kalshi.

pub mod collector;

pub use collector::{CryptoSpotCfg, CryptoSpotCollector};
