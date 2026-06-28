//! Live probe: run the Polymarket collector for a bit and log the catalog +
//! book events it publishes. Proves the collector works end-to-end against the
//! live public feed (no trading).
//!
//!   cargo run -p arb-collector-polymarket --bin poly-probe

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arb_bus::InProcBus;
use arb_core::bus::{Bus, Policy};
use arb_core::event::Payload;
use arb_core::module::Module;
use arb_core::now_ns;
use arb_collector_polymarket::{PolyCfg, PolyCollector};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());

    let mut sub = bus.subscribe("market.polymarket.#", 4096, Policy::Block);
    let logger = tokio::spawn(async move {
        let mut last_book: HashMap<String, i64> = HashMap::new();
        let mut books = 0u64;
        while let Some(ev) = sub.recv().await {
            match &ev.payload {
                Payload::Meta(m) => tracing::info!(
                    "CATALOG {} status={:?} start={:?} expiry={:?}",
                    m.instrument, m.status, m.start_ts_ns, m.expiry_ts_ns
                ),
                Payload::Book(b) => {
                    books += 1;
                    // throttle book logging to ~1/s per instrument
                    let now = now_ns();
                    let last = last_book.entry(b.instrument.clone()).or_insert(0);
                    if now - *last > 1_000_000_000 {
                        *last = now;
                        tracing::info!(
                            "BOOK {} bid={:?} ask={:?} (total books={books})",
                            b.instrument,
                            b.bids.first(),
                            b.asks.first()
                        );
                    }
                }
                Payload::Trade(t) => {
                    tracing::info!("TRADE {} px={} qty={} {:?}", t.instrument, t.price, t.qty, t.side)
                }
                _ => {}
            }
        }
    });

    let mut collector = PolyCollector::new(PolyCfg::default());
    collector.start(bus.clone()).await?;
    tracing::info!("collector started; observing for 45s…");

    tokio::time::sleep(Duration::from_secs(45)).await;
    collector.stop().await?;
    logger.abort();
    tracing::info!("probe done");
    Ok(())
}
