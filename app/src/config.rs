//! YAML config for the trading system. Each module's config struct is reused
//! directly (they derive `Deserialize` with `#[serde(default)]`), so any
//! section or field omitted falls back to the built-in default.

use serde::Deserialize;

use arb_collector_binance::BinanceCfg;
use arb_collector_cryptospot::CryptoSpotCfg;
use arb_collector_venuelat::VenueLatCfg;
use arb_collector_databento::DatabentoCfg;
use arb_collector_kalshi::KalshiCfg;
use arb_collector_polymarket::PolyCfg;
use arb_executor::ExecutorCfg;
use arb_processor::{CalibCfg, ProcCfg};
use arb_recorder::RecorderCfg;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct RunCfg {
    /// Seconds to run; `0` = until Ctrl-C.
    pub duration_secs: u64,
    /// Model-training sampler: when non-empty, write 50ms-grid aligned rows
    /// (perp top + each active Kalshi YES book + tte) to daily CSVs in this
    /// directory. One clock, both sources — the resolution the REST backfill
    /// can't provide. Empty = off.
    pub sample_dir: String,
    /// Sampler grid period (ms).
    pub sample_ms: u64,
    /// Perp bus instrument the sampler/heartbeat track (per-asset instances).
    pub perp_instrument: String,
    /// Coinbase product for the sampler's settlement-chain quote feed.
    pub cb_product: String,
    /// Extra features the sampler reconstructs and appends to every 50ms row
    /// (e.g. ["band5", "vsurge120_1200"]); names as understood by
    /// `FeatureState::feats`. Empty = no extra columns.
    pub feature_extras: Vec<String>,
    /// OB flow_vol probe columns (18 ob_* sampler columns; needs .volb feed).
    pub ob_features: bool,
    /// Optional OB distribution model (delta_min -> K logits). When set AND
    /// `ob_features` is on, the sampler module evaluates it once per closed
    /// 1s bar and PUBLISHES the logits (sampler columns `oblg0..` + a
    /// throttled log line). Nothing consumes them yet — this is the
    /// verification stage before a trade model reads them.
    pub ob_model_path: String,
    /// Deep-book bus instrument for band{k} features (the `binance_depth`
    /// collector's instrument). Empty = no depth routing.
    pub depth_instrument: String,
}

impl Default for RunCfg {
    fn default() -> Self {
        RunCfg {
            duration_secs: 0,
            sample_dir: String::new(),
            sample_ms: 50,
            perp_instrument: "binance.usdt_perp.BTCUSDT".into(),
            cb_product: "BTC-USD".into(),
            feature_extras: Vec::new(),
            ob_features: false,
            ob_model_path: String::new(),
            depth_instrument: String::new(),
        }
    }
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub binance: BinanceCfg,
    /// Optional second Binance feed (e.g. SPOT bookTicker) for the sampler's
    /// settlement-chain columns. None = not spawned.
    pub binance_spot: Option<BinanceCfg>,
    /// Optional DEEP perp book feed (`@depth@100ms` + snapshot resync, top-N
    /// levels) for the band{k} features. Publish under a distinct instrument
    /// (e.g. `binance.usdt_perp.BTCUSDT.depth`). None = not spawned.
    pub binance_depth: Option<BinanceCfg>,
    pub databento: DatabentoCfg,
    pub cryptospot: CryptoSpotCfg,
    pub polymarket: PolyCfg,
    pub kalshi: KalshiCfg,
    pub processor: ProcCfg,
    pub calibrator: CalibCfg,
    pub recorder: RecorderCfg,
    /// LOGGING-ONLY cross-venue top-of-book probe (bybit/okx/binance) used to
    /// rank venue+ticker speed. Runs on its OWN OS thread and publishes
    /// NOTHING to the bus, so it cannot affect trading latency. Disabled by
    /// default; see `crates/collector-venuelat`.
    pub venue_latency: VenueLatCfg,
    pub executor: ExecutorCfg,
    pub run: RunCfg,
}

impl AppConfig {
    pub fn load(path: &str) -> anyhow::Result<AppConfig> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading config {path}: {e}"))?;
        serde_yaml::from_str(&text).map_err(|e| anyhow::anyhow!("parsing config {path}: {e}"))
    }
}
