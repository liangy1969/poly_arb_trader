//! YAML config for the trading system. Each module's config struct is reused
//! directly (they derive `Deserialize` with `#[serde(default)]`), so any
//! section or field omitted falls back to the built-in default.

use serde::Deserialize;

use arb_collector_binance::BinanceCfg;
use arb_collector_cryptospot::CryptoSpotCfg;
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
}

impl Default for RunCfg {
    fn default() -> Self {
        RunCfg {
            duration_secs: 0,
            sample_dir: String::new(),
            sample_ms: 50,
            perp_instrument: "binance.usdt_perp.BTCUSDT".into(),
            cb_product: "BTC-USD".into(),
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
    pub databento: DatabentoCfg,
    pub cryptospot: CryptoSpotCfg,
    pub polymarket: PolyCfg,
    pub kalshi: KalshiCfg,
    pub processor: ProcCfg,
    pub calibrator: CalibCfg,
    pub recorder: RecorderCfg,
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
