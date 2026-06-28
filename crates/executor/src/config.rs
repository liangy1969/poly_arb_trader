//! Executor config (DESIGN_EXECUTION §13). Every field has a default mirroring
//! the backtest's validated cell, so any omitted section/field falls back.
//!
//! v1 ships the `sim` venue (no secrets, no network) — the real `polymarket_clob`
//! adapter (EIP-712 signing + L2 auth) is P2 and reads account secrets per
//! `SETUP_POLYMARKET.md`. Fields the simulated slice does not yet consult are
//! carried for forward-compatibility and marked below.

use serde::Deserialize;

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct ExecutorCfg {
    /// Opt-in: when false the module is inert (like the recorder).
    pub enabled: bool,
    /// Shadow / dry-run: process signals and fire the price probe, but place NO
    /// order (no real money). Requires `enabled: true` so signals still flow.
    pub dry_run: bool,
    pub venue: VenueCfg,
    pub sizing: SizingCfg,
    pub entry: EntryCfg,
    pub exit: ExitCfg,
    pub hold_ms: u64,
    pub risk: RiskCfg,
    /// Simulated-venue knobs (adapter = "sim").
    pub sim: SimCfg,
    /// Hold-period study: shadow-sample each entry's realized exit (book sweep
    /// for the held size + taker fee) at these hold offsets (ms). Empty = off.
    /// One run yields the return-vs-hold curve without changing the real exit.
    pub hold_probe_ms: Vec<u64>,
    /// Maker-exit study: shadow-evaluate a Post-Only maker exit (post at the ask,
    /// rest, fill if the bid lifts it, else fall through to a taker cross) vs the
    /// current taker cross. Measures fill rate + per-trade gain. Off by default.
    pub maker_probe: MakerProbeCfg,
    /// Post-signal price-trajectory probe: on every triggered trade (filled OR
    /// abandoned), sample the traded-token book every `step_ms` for `window_ms`
    /// and log bid/ask/mid + delta from the signal price. Tells us whether the
    /// signal was directionally right independent of whether our order filled.
    pub price_probe: PriceProbeCfg,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct MakerProbeCfg {
    pub enabled: bool,
    /// Passive rest window before falling through to a taker cross.
    pub rest_ms: u64,
    /// Offer price: join the best ask (false) or improve by one tick (true).
    pub improve_tick: bool,
}

impl Default for MakerProbeCfg {
    fn default() -> Self {
        MakerProbeCfg { enabled: false, rest_ms: 1500, improve_tick: false }
    }
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct PriceProbeCfg {
    pub enabled: bool,
    /// Total trajectory window after the signal (ms).
    pub window_ms: u64,
    /// Sample interval (ms).
    pub step_ms: u64,
    /// Pre-signal lookback (ms): how far back to measure the Kalshi ask move that
    /// preceded the trigger (logged in PXMETA). 0 disables the pre-move.
    pub pre_window_ms: u64,
}

impl Default for PriceProbeCfg {
    fn default() -> Self {
        PriceProbeCfg { enabled: false, window_ms: 1000, step_ms: 50, pre_window_ms: 500 }
    }
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct VenueCfg {
    /// Active prediction venue (DESIGN_MULTI_VENUE): "polymarket" | "kalshi".
    /// Drives the bus topics the executor reads, the outcome model (UP/DOWN vs
    /// YES/NO), and the sim taker-delay floor.
    pub market: String,
    /// "sim" (default) | "kalshi" (live REST IOC) | "polymarket_clob" (P2).
    pub adapter: String,
    /// Polymarket: "testnet"/"mainnet". Kalshi: "demo" (default) / "mainnet".
    pub network: String,
    /// Hard per-order USDC cap while validating on mainnet.
    pub max_order_usdc: f64,
    /// Kalshi adapter: access-key id (or empty → unused for sim).
    pub key_id: String,
    /// Kalshi adapter: RSA private-key PEM path (PKCS#8/#1).
    pub private_key_path: String,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct SizingCfg {
    pub size_shares: f64,
    pub depth_frac: f64,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct EntryCfg {
    pub chase_c: f64,
    pub max_chase_total_c: f64,
    pub max_attempts: u32,
    pub retry_delay_ms: u64,
    pub ttl_ms: u64,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct ExitCfg {
    /// "cross" (default, backtest-validated) | "passive" (P3).
    pub mode: String,
    pub retry_interval_ms: u64,
    pub step_c: f64,
    pub max_slip_c: f64,
    pub deadline_buffer_ms: u64,
    /// Safety cap on exit ladder rungs (sim/live).
    pub max_attempts: u32,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct RiskCfg {
    pub yes_bucket: (f64, f64),
    pub min_tte_ms: i64,
    pub max_order_notional: f64,
    pub max_trades_per_min: u32,
    pub cooldown_ms: u64,
    pub stale_ms: i64,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
pub struct SimCfg {
    // The taker-delay floor is now a per-venue fact (DESIGN_MULTI_VENUE) supplied
    // by the active VenueSpec (Polymarket 250 ms; Kalshi none), not a sim knob.
    // fee_rate / tick_size / min_order_size are now read per-market from the
    // catalog meta (collector parses Gamma's feeSchedule.rate / orderMinSize /
    // orderPriceMinTickSize). These config values are only the FALLBACK used when
    // a market's catalog didn't carry them. For BTC up/down they equal the live
    // values (0.07 / 0.01 / 5).
    /// Fallback taker `feeRate` in `fee = feeRate·p·(1−p)·qty` (taker-only).
    pub fee_rate: f64,
    pub tick_size: f64,
    pub min_order_size: f64,
    /// Notional USDC the simulated PM starts with.
    pub start_cash_usdc: f64,
    /// Run the venue liquidator: near each market's expiry, sell any remaining
    /// position at the current best bid (delivered as an async fill). Off → a
    /// residual is simply held (the gate ignores it).
    pub force_liquidity: bool,
    /// How long before expiry the liquidator starts cleaning up a position.
    pub force_window_ms: u64,
    /// Liquidator scan cadence.
    pub force_check_ms: u64,
}

impl Default for ExecutorCfg {
    fn default() -> Self {
        ExecutorCfg {
            enabled: false,
            dry_run: false,
            venue: VenueCfg::default(),
            sizing: SizingCfg::default(),
            entry: EntryCfg::default(),
            exit: ExitCfg::default(),
            hold_ms: 1000,
            risk: RiskCfg::default(),
            sim: SimCfg::default(),
            hold_probe_ms: Vec::new(),
            maker_probe: MakerProbeCfg::default(),
            price_probe: PriceProbeCfg::default(),
        }
    }
}

impl Default for VenueCfg {
    fn default() -> Self {
        VenueCfg {
            market: "polymarket".into(),
            adapter: "sim".into(),
            network: "testnet".into(),
            max_order_usdc: 1.0,
            key_id: String::new(),
            private_key_path: String::new(),
        }
    }
}

impl Default for SizingCfg {
    fn default() -> Self {
        SizingCfg { size_shares: 20.0, depth_frac: 0.5 }
    }
}

impl Default for EntryCfg {
    fn default() -> Self {
        EntryCfg { chase_c: 0.0, max_chase_total_c: 0.01, max_attempts: 3, retry_delay_ms: 100, ttl_ms: 1000 }
    }
}

impl Default for ExitCfg {
    fn default() -> Self {
        ExitCfg {
            mode: "cross".into(),
            retry_interval_ms: 250,
            step_c: 0.01,
            max_slip_c: 0.05,
            deadline_buffer_ms: 5000,
            max_attempts: 4,
        }
    }
}

impl Default for RiskCfg {
    fn default() -> Self {
        RiskCfg {
            yes_bucket: (0.05, 0.95),
            min_tte_ms: 15000,
            max_order_notional: 25.0,
            max_trades_per_min: 6,
            cooldown_ms: 2000,
            stale_ms: 1500,
        }
    }
}

impl Default for SimCfg {
    fn default() -> Self {
        SimCfg {
            fee_rate: 0.07, // Polymarket crypto-market taker rate (docs, 2026)
            tick_size: 0.01,
            min_order_size: 5.0,
            start_cash_usdc: 1000.0,
            force_liquidity: true,
            force_window_ms: 30_000,
            force_check_ms: 2000,
        }
    }
}
