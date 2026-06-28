//! arb-collector-databento — CME BTC futures reference feed via the Databento Live
//! API (US-direct, sub-ms exchange→gateway latency). Signal-only; publishes the
//! standard `market.databento.<root>.book` so the processor consumes it identically
//! to any other reference. See DESIGN_DATABENTO_COLLECTOR.md.

pub mod collector;

pub use collector::{DatabentoCfg, DatabentoCollector};
