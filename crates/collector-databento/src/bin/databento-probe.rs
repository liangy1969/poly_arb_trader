//! Rust-path validation for the Databento BTC feed — mirrors
//! `scripts/databento_validate.py`. Confirms the `databento` crate streams
//! GLBX.MDP3 / BTC.v.0 (volume-roll continuous front) / mbp-1 with our key, so the
//! full collector can be built on this exact stack.
//!
//!   DATABENTO_API_KEY=db-...  cargo run -p arb-collector-databento --bin databento-probe

use std::time::Instant;

use databento::dbn::{Dataset, Mbp1Msg, SType, Schema, FIXED_PRICE_SCALE, UNDEF_PRICE};
use databento::live::Subscription;
use databento::LiveClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut client = LiveClient::builder()
        .key_from_env()? // reads DATABENTO_API_KEY
        .dataset(Dataset::GlbxMdp3)
        .build()
        .await?;

    client
        .subscribe(
            Subscription::builder()
                .symbols("BTC.v.0")
                .schema(Schema::Mbp1)
                .stype_in(SType::Continuous)
                .build(),
        )
        .await?;
    client.start().await?;
    tracing::info!("subscribed GLBX.MDP3 / BTC.v.0 / mbp-1; streaming (CME weekday hours)…");

    let scale = FIXED_PRICE_SCALE as f64;
    let mut n: u32 = 0;
    let mut lat_sum = 0f64;
    let t0 = Instant::now();
    while let Some(rec) = client.next_record().await? {
        if let Some(m) = rec.get::<Mbp1Msg>() {
            let l = &m.levels[0];
            if l.bid_px == UNDEF_PRICE || l.ask_px == UNDEF_PRICE {
                continue;
            }
            let bid = l.bid_px as f64 / scale;
            let ask = l.ask_px as f64 / scale;
            let lat_ms = (m.ts_recv as i64 - m.hd.ts_event as i64) as f64 / 1e6;
            lat_sum += lat_ms;
            if n < 6 {
                tracing::info!(
                    "[inst {}] bid={:.1} ask={:.1} mid={:.1} sz={}x{} exch->gw={:.3}ms",
                    m.hd.instrument_id, bid, ask, (bid + ask) / 2.0, l.bid_sz, l.ask_sz, lat_ms
                );
            }
            n += 1;
            if n >= 40 {
                break;
            }
        }
        if t0.elapsed().as_secs() > 45 {
            tracing::warn!("timeout (CME break / off-hours / entitlement?)");
            break;
        }
    }

    if n > 0 {
        tracing::info!("PASS: {n} mbp-1 records, mean exch->gateway latency {:.3}ms", lat_sum / n as f64);
    } else {
        tracing::error!("FAIL: no data — check CME hours, live entitlement for GLBX.MDP3, symbol BTC.v.0");
    }
    Ok(())
}
