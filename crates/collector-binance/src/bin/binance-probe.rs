//! Live probe: run the Binance perp collector and log the top-of-book it
//! publishes. Uses the SOCKS5 proxy by default (set BINANCE_DIRECT=1 to try a
//! direct connection).
//!
//!   cargo run -p arb-collector-binance --bin binance-probe

use std::sync::Arc;
use std::time::Duration;

use arb_bus::InProcBus;
use arb_collector_binance::{BinanceCfg, BinanceCollector};
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

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());

    let mut sub = bus.subscribe("market.binance.#", 4096, Policy::Block);
    let logger = tokio::spawn(async move {
        // d1 = recv - exch (collector server→client; offset-dominated → report jitter)
        // d2 = now  - recv (bus delivery to a consumer; clean, no clock offset)
        let mut d1: Vec<i64> = Vec::new();
        let mut d2: Vec<i64> = Vec::new();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            tokio::select! {
                ev = sub.recv() => {
                    let Some(ev) = ev else { break };
                    if let Payload::Book(b) = &ev.payload {
                        let now = now_ns();
                        d1.push((b.recv_ts_ns - b.exch_ts_ns) / 1_000_000); // ms
                        d2.push((now - b.recv_ts_ns) / 1000); // µs
                    }
                }
                _ = tokio::time::sleep_until(deadline) => break,
            }
        }
        let pct = |v: &mut Vec<i64>, p: f64| -> i64 {
            if v.is_empty() { return 0; }
            v.sort_unstable();
            v[((v.len() as f64 - 1.0) * p) as usize]
        };
        let (mn, p50, p99) = (pct(&mut d1, 0.0), pct(&mut d1, 0.5), pct(&mut d1, 0.99));
        tracing::info!(
            "ISOLATED n={} | recv-exch ms: min={mn} p50={p50} p99={p99} (jitter p50-min={} p99-min={}) | bus-deliver µs: p50={} p99={}",
            d1.len(), p50 - mn, p99 - mn, pct(&mut d2, 0.5), pct(&mut d2, 0.99),
        );
    });

    let mut cfg = BinanceCfg::default();
    if std::env::var("BINANCE_DIRECT").is_ok() {
        cfg.socks_proxy = None;
        tracing::info!("BINANCE_DIRECT set -> connecting directly (no proxy)");
    } else {
        tracing::info!("using SOCKS5 proxy {:?}", cfg.socks_proxy);
    }

    let mut collector = BinanceCollector::new(cfg);
    collector.start(bus.clone()).await?;
    tracing::info!("collector started; observing for 30s…");

    let _ = logger.await; // self-timed 30s; prints the latency summary
    collector.stop().await?;
    tracing::info!("probe done");
    Ok(())
}
