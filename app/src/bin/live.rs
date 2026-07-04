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
    // Two sinks:
    //  - stdout (the run log, redirected to /tmp/live-trade.log; truncated per
    //    relaunch — heartbeats/collector noise, disposable)
    //  - data/trader-events.log: APPEND-ONLY, survives relaunches. Analysis
    //    targets only (probes, chase, exit reconciler, trade records, signals) —
    //    low volume, so it's disk-safe on the small box. Added after per-run log
    //    truncation destroyed the PXPROBE dataset for a filled live trade.
    use tracing_subscriber::filter::{EnvFilter, LevelFilter, Targets};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer as _;
    let stdout_layer = tracing_subscriber::fmt::layer().with_filter(
        EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    );
    let _ = std::fs::create_dir_all("data");
    let events_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("data/trader-events.log");
    match events_file {
        Ok(f) => {
            let events_layer = tracing_subscriber::fmt::layer()
                .with_writer(std::sync::Arc::new(f))
                .with_ansi(false)
                .with_filter(
                    Targets::new()
                        .with_target("pxprobe", LevelFilter::INFO)
                        .with_target("chase", LevelFilter::INFO)
                        .with_target("exit", LevelFilter::INFO)
                        .with_target("executor", LevelFilter::INFO)
                        .with_target("exec", LevelFilter::INFO)
                        .with_target("signal", LevelFilter::INFO)
                        .with_target("hold", LevelFilter::INFO)
                        .with_target("maker", LevelFilter::INFO),
                );
            tracing_subscriber::registry().with(stdout_layer).with(events_layer).init();
            tracing::info!("trader events -> data/trader-events.log (append; survives relaunches)");
        }
        Err(e) => {
            tracing_subscriber::registry().with(stdout_layer).init();
            tracing::warn!("events log unavailable ({e}); stdout only");
        }
    }

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

    // --- 50ms aligned sampler (model-training data, run.sample_dir) ---
    // One clock, both sources: every sample_ms tick, write a row per active
    // Kalshi market with the CURRENT perp top + YES book top + tte. This is the
    // time-aligned high-resolution series the REST backfill cannot provide
    // (1s spot bars vs trade prints with unknown cross-source clock skew).
    // Daily-rotated CSVs: <sample_dir>/YYYY-MM-DD.csv (gzip old days via cron).
    let _sampler = if cfg.run.sample_dir.is_empty() {
        None
    } else {
        let dir = cfg.run.sample_dir.clone();
        let period = Duration::from_millis(cfg.run.sample_ms.max(10));
        let mut sub = bus.subscribe("market.#", 8192, Policy::Conflate(key_by_instrument));
        Some(tokio::spawn(async move {
            use std::collections::HashMap;
            use std::io::Write as _;
            #[derive(Default, Clone, Copy)]
            struct KTop {
                ybid: f64,
                yask: f64,
                ybsz: f64,
                yasz: f64,
                expiry_ns: i64,
                book_ns: i64,
            }
            std::fs::create_dir_all(&dir).ok();
            let mid_of = |inst: &str| -> Option<String> {
                inst.strip_prefix("kalshi.")?.rsplit_once('.').map(|(m, _)| m.to_string())
            };
            let mut perp: Option<(f64, f64)> = None;
            let mut books: HashMap<String, KTop> = HashMap::new();
            let mut expiry: HashMap<String, i64> = HashMap::new();
            let mut out: Option<(String, std::io::BufWriter<std::fs::File>)> = None;
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            let mut n_since_flush = 0u32;
            tracing::info!("sampler up -> {dir} @ {}ms grid", period.as_millis());
            loop {
                tokio::select! {
                    ev = sub.recv() => {
                        let Some(ev) = ev else { break };
                        match &ev.payload {
                            Payload::Book(b) if b.instrument == "binance.usdt_perp.BTCUSDT" => {
                                if let (Some(&(pb, _)), Some(&(pa, _))) = (b.bids.first(), b.asks.first()) {
                                    perp = Some((pb, pa));
                                }
                            }
                            Payload::Book(b) if b.instrument.starts_with("kalshi.") && b.instrument.ends_with(".YES") => {
                                if let Some(mid) = mid_of(&b.instrument) {
                                    let e = books.entry(mid.clone()).or_default();
                                    e.ybid = b.bids.first().map(|&(p, _)| p).unwrap_or(0.0);
                                    e.ybsz = b.bids.first().map(|&(_, s)| s).unwrap_or(0.0);
                                    e.yask = b.asks.first().map(|&(p, _)| p).unwrap_or(0.0);
                                    e.yasz = b.asks.first().map(|&(_, s)| s).unwrap_or(0.0);
                                    e.book_ns = b.recv_ts_ns;
                                    if e.expiry_ns == 0 {
                                        if let Some(&x) = expiry.get(&mid) { e.expiry_ns = x; }
                                    }
                                }
                            }
                            Payload::Meta(m) if m.instrument.starts_with("kalshi.") => {
                                if let (Some(mid), Some(x)) = (mid_of(&m.instrument), m.expiry_ts_ns) {
                                    expiry.insert(mid.clone(), x);
                                    if let Some(e) = books.get_mut(&mid) { e.expiry_ns = x; }
                                }
                            }
                            _ => {}
                        }
                    }
                    _ = tick.tick() => {
                        let Some((pb, pa)) = perp else { continue };
                        let now = arb_core::now_ns();
                        // rotate the daily file
                        let day = {
                            let secs = now / 1_000_000_000;
                            let days = secs / 86_400;
                            // civil date from unix days (valid 2000-2099)
                            let (mut y, mut doy) = (1970i64, days);
                            loop {
                                let len = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 366 } else { 365 };
                                if doy < len { break; }
                                doy -= len; y += 1;
                            }
                            let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
                            let ml = [31, if leap {29} else {28}, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
                            let mut m = 0usize;
                            while doy >= ml[m] { doy -= ml[m]; m += 1; }
                            format!("{y:04}-{:02}-{:02}", m + 1, doy + 1)
                        };
                        if out.as_ref().map(|(d, _)| d != &day).unwrap_or(true) {
                            if let Some((_, mut w)) = out.take() { let _ = w.flush(); }
                            let path = format!("{dir}/{day}.csv");
                            let fresh = !std::path::Path::new(&path).exists();
                            match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                                Ok(f) => {
                                    let mut w = std::io::BufWriter::new(f);
                                    if fresh {
                                        let _ = writeln!(w, "ts_ms,ticker,tte_ms,perp_bid,perp_ask,ybid,yask,ybid_sz,yask_sz");
                                    }
                                    out = Some((day, w));
                                }
                                Err(e) => { tracing::warn!("sampler open {path}: {e}"); continue; }
                            }
                        }
                        let Some((_, w)) = out.as_mut() else { continue };
                        for (mid, k) in &books {
                            if k.expiry_ns == 0 || (k.yask <= 0.0 && k.ybid <= 0.0) {
                                continue;
                            }
                            let tte_ms = (k.expiry_ns - now) / 1_000_000;
                            if !(-10_000..=1_200_000).contains(&tte_ms) {
                                continue; // only the active window (+/- a little)
                            }
                            let _ = writeln!(
                                w,
                                "{},{},{},{:.2},{:.2},{:.3},{:.3},{:.1},{:.1}",
                                now / 1_000_000, mid, tte_ms, pb, pa, k.ybid, k.yask, k.ybsz, k.yasz
                            );
                        }
                        n_since_flush += 1;
                        if n_since_flush >= 20 { // ~1s
                            let _ = w.flush();
                            n_since_flush = 0;
                            // drop books long past settle so the map stays small
                            books.retain(|_, k| k.expiry_ns == 0 || k.expiry_ns > now - 3_600_000_000_000);
                        }
                    }
                }
            }
        }))
    };

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
