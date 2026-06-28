//! `RiskGate` (DESIGN_EXECUTION §9) — pre-trade checks, no I/O. v1 implements the
//! checks feasible without a funded account; the cash/allowance, net-edge/fee,
//! and daily-loss rules are wired once the real venue + market-meta fee land (P2,
//! marked inline). The one-trade gate (rule 3) is the PM's `try_begin_trade`.

use std::collections::VecDeque;

use arb_core::model::TradeSignal;

use crate::config::RiskCfg;
use crate::types::{BookTop, MS};

pub struct RiskGate {
    cfg: RiskCfg,
    kill_switch: bool,
    trade_times_ns: VecDeque<i64>,
}

impl RiskGate {
    pub fn new(cfg: RiskCfg) -> Self {
        RiskGate { cfg, kill_switch: false, trade_times_ns: VecDeque::new() }
    }

    pub fn killed(&self) -> bool {
        self.kill_switch
    }

    /// Trip the kill switch (manual or the long-only-invariant safety, §8.2).
    /// NOTE: the consecutive-venue-reject auto-trip (§5/§9) is intentionally
    /// removed for v1 simplicity — re-add once we collect live trading data.
    pub fn trip(&mut self, reason: &str) {
        if !self.kill_switch {
            tracing::error!(target: "executor", "KILL SWITCH tripped: {reason}");
        }
        self.kill_switch = true;
    }

    pub fn resume(&mut self) {
        self.kill_switch = false;
    }

    /// Record a trade start for the per-minute rate limit.
    pub fn note_trade(&mut self, now_ns: i64) {
        self.trade_times_ns.push_back(now_ns);
        self.prune_rate_window(now_ns);
    }

    fn prune_rate_window(&mut self, now_ns: i64) {
        let cutoff = now_ns - 60_000 * MS;
        while self.trade_times_ns.front().is_some_and(|&t| t < cutoff) {
            self.trade_times_ns.pop_front();
        }
    }

    /// Pre-trade checks (DESIGN_EXECUTION §9, feasible subset, in order). `price`
    /// is the traded-token best ask; `size` the planned shares.
    pub fn check(
        &mut self,
        sig: &TradeSignal,
        book: BookTop,
        tte_ms: Option<i64>,
        now_ns: i64,
        size: f64,
        price: f64,
    ) -> Result<(), String> {
        // 1 — kill switch.
        if self.kill_switch {
            return Err("kill switch engaged".into());
        }
        // 2 — mirror freshness.
        if now_ns - book.recv_ts_ns > self.cfg.stale_ms * MS {
            return Err(format!("stale book ({}ms)", (now_ns - book.recv_ts_ns) / MS));
        }
        // 4 — yes-price bucket (symmetric; applies to the traded token's own ask).
        if !(price.is_finite() && price >= self.cfg.yes_bucket.0 && price <= self.cfg.yes_bucket.1) {
            return Err(format!("ask {price:.3} outside bucket {:?}", self.cfg.yes_bucket));
        }
        // 5 — time to expiry (a position must never be designed to straddle resolution).
        match tte_ms {
            Some(t) if t >= self.cfg.min_tte_ms => {}
            Some(t) => return Err(format!("tte {t}ms < min {}ms", self.cfg.min_tte_ms)),
            None => return Err("no expiry meta".into()),
        }
        // 6 — signal not stale by queueing.
        if now_ns - sig.ts_ns > sig.ttl_ms as i64 * MS {
            return Err(format!("signal stale ({}ms)", (now_ns - sig.ts_ns) / MS));
        }
        // 6b — order notional cap.
        let notional = size * price;
        if notional > self.cfg.max_order_notional {
            return Err(format!("notional {notional:.2} > max {:.2}", self.cfg.max_order_notional));
        }
        // 8 — per-minute rate limit.
        self.prune_rate_window(now_ns);
        if self.trade_times_ns.len() as u32 >= self.cfg.max_trades_per_min {
            return Err(format!("rate limit {} trades/min", self.cfg.max_trades_per_min));
        }
        // (3 one-trade gate: PM.try_begin_trade, in the engine.)
        // (7 cash/allowance, 5 net-edge/fee, 9 daily-loss: P2 — need funded
        //  account + per-market fee meta; see SETUP_POLYMARKET.md.)
        Ok(())
    }
}
