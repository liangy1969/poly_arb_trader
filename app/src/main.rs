//! Bring-up binary: wire the bus + processor, drive a synthetic perp move
//! through a Live prediction market, and log the resulting trade signal.
//! Proves the signal path end-to-end before real collectors land.

use std::sync::Arc;
use std::time::Duration;

use arb_bus::InProcBus;
use arb_core::bus::{Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::model::*;
use arb_core::module::Module;
use arb_processor::{ProcCfg, Processor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());

    // Signal logger — subscribes signal.# and prints each TradeSignal as JSON.
    let mut sig = bus.subscribe("signal.#", 256, Policy::Block);
    let logger = tokio::spawn(async move {
        let mut n = 0u64;
        while let Some(ev) = sig.recv().await {
            if let Payload::Signal(s) = &ev.payload {
                n += 1;
                let json = serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}"));
                tracing::info!(target: "signal", "SIGNAL #{n} {json}");
            }
        }
    });

    // Processor — defaults mirror the backtest's validated cell.
    let mut processor = Processor::new(ProcCfg {
        reference: "binance.usdt_perp.BTCUSDT".into(),
        strategy: "perp_move".into(),
        link_kind: "5m_updown".into(),
        window_ms: 1000,
        threshold_bps: 3.0,
        yes_bucket: (0.05, 0.95),
        cooldown_ms: 2000,
        min_tte_ms: 15000,
        hold_ms: 1000,
        ttl_ms: 1000,
        ring_cap: 512,
        ring_horizon_ms: 5000,
    });
    processor.start(bus.clone()).await?;

    tracing::info!("running synthetic feed (perp +~6bps over 1.2s, Live 5m market)…");
    feed(bus.clone()).await;

    // Let the async pipeline drain, then shut down.
    tokio::time::sleep(Duration::from_millis(300)).await;
    processor.stop().await?;
    logger.abort();
    tracing::info!("done");
    Ok(())
}

/// Publish a Live catalog event + poly book (mid 0.50) + a rising perp book.
/// All timestamps are synthetic so the run is deterministic.
async fn feed(bus: Arc<dyn Bus>) {
    let base: i64 = 1_700_000_000_000_000_000;
    let cid = "polymarket.0xabc.UP";
    let perp = "binance.usdt_perp.BTCUSDT";
    let mut seq = 0u64;

    bus.publish(Event::new(
        "market.polymarket.0xabc.catalog",
        "feed",
        base,
        next(&mut seq),
        Payload::Meta(MarketMeta {
            instrument: cid.into(),
            kind: "5m_updown".into(),
            status: MarketStatus::Live,
            start_ts_ns: Some(base),
            expiry_ts_ns: Some(base + 300_000_000_000),
            winner: None,
            min_order_size: None,
            tick_size: None,
            fee_rate: None,
        }),
    ));

    bus.publish(Event::new(
        "market.polymarket.0xabc.book",
        "feed",
        base,
        next(&mut seq),
        Payload::Book(BookUpdate {
            instrument: cid.into(),
            bids: vec![(0.49, 100.0)],
            asks: vec![(0.51, 100.0)],
            update_id: None,
            exch_ts_ns: base,
            recv_ts_ns: base,
        }),
    ));

    for k in 0..=12i64 {
        let ts = base + k * 100_000_000; // 100ms steps
        let px = 50000.0 * (1.0 + (k as f64) * 0.5 / 10000.0); // +0.5 bps per step
        let (bid, ask) = (px - 1.0, px + 1.0);
        bus.publish(Event::new(
            "market.binance.usdt_perp.BTCUSDT.book",
            "feed",
            ts,
            next(&mut seq),
            Payload::Book(BookUpdate {
                instrument: perp.into(),
                bids: vec![(bid, 5.0)],
                asks: vec![(ask, 5.0)],
                update_id: Some(k as u64),
                exch_ts_ns: ts,
                recv_ts_ns: ts,
            }),
        ));
        tokio::task::yield_now().await; // let the processor interleave
    }
}

fn next(seq: &mut u64) -> u64 {
    let s = *seq;
    *seq += 1;
    s
}
