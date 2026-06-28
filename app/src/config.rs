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
use arb_processor::ProcCfg;
use arb_recorder::RecorderCfg;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct RunCfg {
    /// Seconds to run; `0` = until Ctrl-C.
    pub duration_secs: u64,
}

impl Default for RunCfg {
    fn default() -> Self {
        RunCfg { duration_secs: 0 }
    }
}

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub binance: BinanceCfg,
    pub databento: DatabentoCfg,
    pub cryptospot: CryptoSpotCfg,
    pub polymarket: PolyCfg,
    pub kalshi: KalshiCfg,
    pub processor: ProcCfg,
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
