//! Live end-to-end: both collectors (Polymarket direct, Binance via the SOCKS
//! proxy) + the processor, logging real `PerpMoveRule` signals. All parameters
//! come from a YAML config (default `config/local.yaml`).
//!
//!   (start the tunnel first: scripts/tunnel.ps1)
//!   cargo run -p arb-app --bin live [config/local.yaml]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arb_app::config::AppConfig;
use arb_bus::InProcBus;
use arb_collector_binance::BinanceCollector;
use arb_collector_cryptospot::CryptoSpotCollector;
use arb_collector_databento::DatabentoCollector;
use arb_collector_kalshi::KalshiCollector;
use arb_collector_polymarket::PolyCollector;
use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::Payload;
use arb_core::model::MarketStatus;
use arb_core::module::Module;
use arb_executor::Executor;
use arb_processor::Processor;
use arb_recorder::Recorder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let path = std::env::args().nth(1).unwrap_or_else(|| "config/local.yaml".into());
    let cfg = AppConfig::load(&path)?;
    tracing::info!("loaded config from {path}");

    let bus: Arc<dyn Bus> = Arc::new(InProcBus::new());
    let sig_count = Arc::new(AtomicU64::new(0));

    // --- signal logger (the validation target) ---
    let mut sig = bus.subscribe("signal.#", 256, Policy::Block);
    let sc = sig_count.clone();
    let signal_logger = tokio::spawn(async move {
        while let Some(ev) = sig.recv().await {
            if let Payload::Signal(s) = &ev.payload {
                let n = sc.fetch_add(1, Ordering::Relaxed) + 1;
                let json = serde_json::to_string(s).unwrap_or_else(|_| format!("{s:?}"));
                tracing::info!(target: "signal", "*** SIGNAL #{n} *** {json}");
            }
        }
    });

    // --- pipeline heartbeat: perp mid + current Live target ---
    let mut mkt = bus.subscribe("market.#", 8192, Policy::Conflate(key_by_instrument));
    let heartbeat = tokio::spawn(async move {
        let mut perp_mid = f64::NAN;
        let mut live_target: Option<String> = None;
        let mut bn = 0u64; // binance book events since last tick
        let mut pm = 0u64; // polymarket book events since last tick
        let mut km = 0u64; // kalshi book events since last tick
        let mut tick = tokio::time::interval(Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = tick.tick() => {
                    tracing::info!(
                        "heartbeat: perp_mid={:.1} target={} | last 5s: binance_books={bn} poly_books={pm} kalshi_books={km}",
                        perp_mid,
                        live_target.as_deref().unwrap_or("(none)")
                    );
                    bn = 0;
                    pm = 0;
                    km = 0;
                }
                ev = mkt.recv() => {
                    let Some(ev) = ev else { break };
                    match &ev.payload {
                        Payload::Book(b) if b.instrument == "binance.usdt_perp.BTCUSDT" => {
                            bn += 1;
                            if let (Some(&(bid, _)), Some(&(ask, _))) = (b.bids.first(), b.asks.first()) {
                                perp_mid = (bid + ask) / 2.0;
                            }
                        }
                        Payload::Book(b) if b.instrument.starts_with("polymarket.") => {
                            pm += 1;
                        }
                        Payload::Book(b) if b.instrument.starts_with("kalshi.") => {
                            km += 1;
                        }
                        Payload::Meta(m) => match m.status {
                            MarketStatus::Live => {
                                tracing::info!("LINK target -> {}", m.instrument);
                                live_target = Some(m.instrument.clone());
                            }
                            MarketStatus::Expired => {
                                if live_target.as_deref() == Some(m.instrument.as_str()) {
                                    live_target = None;
                                }
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }
        }
    });

    // --- exec event logger (trades / reports / positions) ---
    let mut execlog = bus.subscribe("exec.#", 256, Policy::Block);
    let exec_logger = tokio::spawn(async move {
        while let Some(ev) = execlog.recv().await {
            match &ev.payload {
                Payload::TradeRecord(t) => {
                    let json = serde_json::to_string(t).unwrap_or_else(|_| format!("{t:?}"));
                    tracing::info!(target: "exec", "TRADE {json}");
                }
                Payload::ExecReport(r) => {
                    tracing::info!(target: "exec", "[{}] {} {} {}", r.trade_id, r.state, r.instrument, r.detail);
                }
                _ => {}
            }
        }
    });

    // --- modules (all from config) ---
    let mut recorder = Recorder::new(cfg.recorder.clone());
    let mut poly = PolyCollector::new(cfg.polymarket.clone());
    let mut kalshi = KalshiCollector::new(cfg.kalshi.clone());
    let mut binance = BinanceCollector::new(cfg.binance.clone());
    let mut databento = DatabentoCollector::new(cfg.databento.clone());
    let mut cryptospot = CryptoSpotCollector::new(cfg.cryptospot.clone());
    let mut processor = Processor::new(cfg.processor.clone());
    let mut executor = Executor::new(cfg.executor.clone());

    // Single active prediction venue (DESIGN_MULTI_VENUE): start only its
    // collector. The other is constructed but never connected.
    let active_venue = cfg.executor.venue.market.clone();
    recorder.start(bus.clone()).await?; // first, so it captures the full stream
    if active_venue == "polymarket" {
        poly.start(bus.clone()).await?;
    }
    if active_venue == "kalshi" {
        kalshi.start(bus.clone()).await?;
    }
    binance.start(bus.clone()).await?;
    if cfg.databento.enabled {
        databento.start(bus.clone()).await?;
    }
    if cfg.cryptospot.enabled {
        cryptospot.start(bus.clone()).await?;
    }
    processor.start(bus.clone()).await?;
    executor.start(bus.clone()).await?;
    tracing::info!("active prediction venue = {active_venue}");
    tracing::info!(
        "live pipeline up (threshold={}bps/{}ms, proxy={:?})",
        cfg.processor.threshold_bps,
        cfg.processor.window_ms,
        cfg.binance.socks_proxy
    );

    // Run for the configured duration, or until Ctrl-C (duration_secs = 0).
    if cfg.run.duration_secs == 0 {
        tracing::info!("running until Ctrl-C…");
        let _ = tokio::signal::ctrl_c().await;
    } else {
        tracing::info!("observing {}s…", cfg.run.duration_secs);
        tokio::time::sleep(Duration::from_secs(cfg.run.duration_secs)).await;
    }

    executor.stop().await?;
    processor.stop().await?;
    if cfg.cryptospot.enabled {
        cryptospot.stop().await?;
    }
    if cfg.databento.enabled {
        databento.stop().await?;
    }
    binance.stop().await?;
    if active_venue == "kalshi" {
        kalshi.stop().await?;
    }
    if active_venue == "polymarket" {
        poly.stop().await?;
    }
    recorder.stop().await?;
    signal_logger.abort();
    heartbeat.abort();
    exec_logger.abort();

    let total = sig_count.load(Ordering::Relaxed);
    tracing::info!("done: {total} signal(s) emitted", total = total);
    Ok(())
}
