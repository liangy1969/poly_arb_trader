//! Live probe: run ONLY the Kalshi collector for a bit and log the catalog +
//! book + trade events it publishes. Verifies the collector end-to-end against
//! the live feed (read-only, no trading) — in particular whether the RSA-PSS
//! **WS auth** is accepted (look for `kalshi ws subscribed …` + a rising book
//! count; a `connect failed` / REST-only fallback means auth was rejected).
//!
//! Reads the `kalshi:` section from `config/local.yaml` (so credentials are not
//! passed on the command line). Run from the repo root:
//!
//!   cargo run -p arb-collector-kalshi --bin kalshi-probe

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arb_bus::InProcBus;
use arb_collector_kalshi::{KalshiCfg, KalshiCollector};
use arb_core::bus::{Bus, Policy};
use arb_core::event::Payload;
use arb_core::module::Module;
use arb_core::now_ns;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Load just the kalshi section from the app config (reuses key_id + PEM path).
    let path = std::env::args().nth(1).unwrap_or_else(|| "config/local.yaml".into());
    let cfg: KalshiCfg = match std::fs::read_to_string(&path) {
        Ok(text) => serde_yaml::from_str::<serde_yaml::Value>(&text)?
            .get("kalshi")
            .cloned()
            .map(serde_yaml::from_value)
            .transpose()?
            .unwrap_or_default(),
        Err(e) => {
            tracing::warn!("no config at {path} ({e}); using defaults + env creds");
            KalshiCfg::default()
        }
    };

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());

    let mut sub = bus.subscribe("market.kalshi.#", 4096, Policy::Block);
    let logger = tokio::spawn(async move {
        let mut last_book: HashMap<String, i64> = HashMap::new();
        let mut books = 0u64;
        let mut trades = 0u64;
        while let Some(ev) = sub.recv().await {
            match &ev.payload {
                Payload::Meta(m) => tracing::info!(
                    "CATALOG {} status={:?} start={:?} expiry={:?}",
                    m.instrument, m.status, m.start_ts_ns, m.expiry_ts_ns
                ),
                Payload::Book(b) => {
                    books += 1;
                    let now = now_ns();
                    let last = last_book.entry(b.instrument.clone()).or_insert(0);
                    if now - *last > 1_000_000_000 {
                        *last = now;
                        tracing::info!(
                            "BOOK {} bid={:?} ask={:?} (total books={books})",
                            b.instrument, b.bids.first(), b.asks.first()
                        );
                    }
                }
                Payload::Trade(t) => {
                    trades += 1;
                    tracing::info!("TRADE {} px={} qty={} {:?} (total trades={trades})",
                        t.instrument, t.price, t.qty, t.side);
                }
                _ => {}
            }
        }
    });

    let mut collector = KalshiCollector::new(cfg);
    collector.start(bus.clone()).await?;
    tracing::info!("kalshi collector started; observing for 25s…");

    tokio::time::sleep(Duration::from_secs(25)).await;
    collector.stop().await?;
    logger.abort();
    tracing::info!("probe done");
    Ok(())
}
