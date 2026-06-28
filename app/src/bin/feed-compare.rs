//! Collect the Binance perp top-of-book from BOTH sources simultaneously —
//! `bookTicker` (real-time) and `@depth@100ms` (L2 diff) — to a full JSONL
//! record, for offline equivalence + latency analysis.
//!
//!   (tunnel up) cargo run -p arb-app --bin feed-compare [seconds]
//!
//! Both publish single-level `BookUpdate`s on the same `.book` topic but with
//! distinct instruments (`.tick` vs `.depth`), so they're separable downstream.

use std::sync::Arc;
use std::time::Duration;

use arb_bus::InProcBus;
use arb_collector_binance::{BinanceCfg, BinanceCollector};
use arb_core::bus::Bus;
use arb_core::module::Module;
use arb_recorder::{Recorder, RecorderCfg};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let secs: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(120);

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());

    // Full record (no sampling) of every binance book event.
    let mut rec = Recorder::new(RecorderCfg {
        enabled: true,
        dir: "data/feedcompare".into(),
        pattern: "market.binance.#".into(),
        sample_interval_ms: 0,
        sample_key: "instrument".into(),
        always_keep: vec![],
    });
    rec.start(bus.clone()).await?;

    let mut tick = BinanceCollector::new(BinanceCfg {
        instrument: "binance.usdt_perp.BTCUSDT.tick".into(),
        stream: "btcusdt@bookTicker".into(),
        ..Default::default()
    });
    let mut depth = BinanceCollector::new(BinanceCfg {
        instrument: "binance.usdt_perp.BTCUSDT.depth".into(),
        stream: "btcusdt@depth@100ms".into(),
        ..Default::default()
    });

    tick.start(bus.clone()).await?;
    depth.start(bus.clone()).await?;
    tracing::info!("collecting bookTicker + depth@100ms for {secs}s -> data/feedcompare/");

    tokio::time::sleep(Duration::from_secs(secs)).await;

    tick.stop().await?;
    depth.stop().await?;
    rec.stop().await?;
    tracing::info!("done");
    Ok(())
}
