//! `PositionManager` (DESIGN_EXECUTION §8) — single authority on what we hold
//! and what it's worth. Idempotent, status-aware, **long-only** fill application;
//! the one-trade gate + cooldown; resolution settlement.

use std::collections::HashMap;

use arb_core::model::{PositionSnapshot, Side};

use crate::types::{Fill, FillStatus, MS};

const EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosStatus {
    Open,
    PendingResolution,
    Settled,
}

impl PosStatus {
    fn as_str(&self) -> &'static str {
        match self {
            PosStatus::Open => "Open",
            PosStatus::PendingResolution => "PendingResolution",
            PosStatus::Settled => "Settled",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Position {
    pub instrument: String,
    pub qty: f64,
    pub avg_cost: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub mark_px: f64,
    pub status: PosStatus,
}

impl Position {
    fn new(instrument: String) -> Self {
        Position {
            instrument,
            qty: 0.0,
            avg_cost: 0.0,
            realized_pnl: 0.0,
            fees_paid: 0.0,
            mark_px: f64::NAN,
            status: PosStatus::Open,
        }
    }

    fn upnl(&self) -> f64 {
        if self.qty > 0.0 && self.mark_px.is_finite() {
            self.qty * (self.mark_px - self.avg_cost)
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DailyStats {
    pub realized: f64,
    pub fees: f64,
    pub n_trades: u32,
}

pub struct PositionManager {
    cash_usdc: f64,
    positions: HashMap<String, Position>,
    active_trade: Option<String>,
    cooldown_until_ns: i64,
    daily: DailyStats,
    seen_fills: HashMap<String, FillStatus>,
}

impl PositionManager {
    pub fn new(cash_usdc: f64) -> Self {
        PositionManager {
            cash_usdc,
            positions: HashMap::new(),
            active_trade: None,
            cooldown_until_ns: 0,
            daily: DailyStats::default(),
            seen_fills: HashMap::new(),
        }
    }

    pub fn cash(&self) -> f64 {
        self.cash_usdc
    }

    pub fn daily(&self) -> &DailyStats {
        &self.daily
    }

    pub fn qty(&self, instrument: &str) -> f64 {
        self.positions.get(instrument).map(|p| p.qty).unwrap_or(0.0)
    }

    pub fn position(&self, instrument: &str) -> Option<&Position> {
        self.positions.get(instrument)
    }

    /// One **active** trade at a time + cooldown (DESIGN_EXECUTION §8.4, relaxed):
    /// the flat check is intentionally dropped so a residual held to expiry
    /// (PendingResolution, cleaned up by the venue liquidator) doesn't block new
    /// trades. Positions are keyed by token, so a re-entry merges and the next
    /// exit sells the whole.
    pub fn try_begin_trade(&mut self, trade_id: &str, now_ns: i64) -> bool {
        if self.active_trade.is_none() && now_ns >= self.cooldown_until_ns {
            self.active_trade = Some(trade_id.to_string());
            true
        } else {
            false
        }
    }

    /// Free the slot and arm the cooldown (Closed/Abandoned/Settled).
    pub fn end_trade(&mut self, now_ns: i64, cooldown_ms: u64) {
        self.active_trade = None;
        self.cooldown_until_ns = now_ns + cooldown_ms as i64 * MS;
    }

    pub fn active_trade(&self) -> Option<&str> {
        self.active_trade.as_deref()
    }

    /// Apply one fill — idempotent on `venue_trade_id`, reversible on `failed`
    /// (DESIGN_EXECUTION §8.2). `f.qty` is shares **received**. Errs on a
    /// long-only violation (a phantom/double fill) — caller trips the kill switch.
    pub fn apply_fill(&mut self, f: &Fill) -> anyhow::Result<()> {
        let prev = self.seen_fills.get(&f.venue_trade_id).copied();
        if prev == Some(f.status) {
            return Ok(()); // exact replay → no-op
        }
        let applied = matches!(prev, Some(FillStatus::Matched) | Some(FillStatus::Confirmed));

        match (f.status, applied) {
            (FillStatus::Matched | FillStatus::Confirmed, false) => self.apply_math(f, 1.0)?,
            (FillStatus::Confirmed, true) => {} // already applied at matched → just finalize status
            (FillStatus::Failed, true) => self.apply_math(f, -1.0)?, // reverse the provisional fill
            (FillStatus::Failed, false) => {}                        // never applied → nothing to undo
            (FillStatus::Matched, true) => {}                        // out-of-order; already applied
        }
        self.seen_fills.insert(f.venue_trade_id.clone(), f.status);
        Ok(())
    }

    /// `sign = +1` applies, `-1` reverses. Keeps cash, cost basis, realized P&L,
    /// and the long-only invariant consistent in both directions.
    fn apply_math(&mut self, f: &Fill, sign: f64) -> anyhow::Result<()> {
        let pos = self
            .positions
            .entry(f.instrument.clone())
            .or_insert_with(|| Position::new(f.instrument.clone()));
        let q = sign * f.qty;
        match f.side {
            Side::Buy => {
                let new_qty = pos.qty + q;
                if new_qty < -EPS {
                    anyhow::bail!("long-only violation: buy reverse would drive qty {new_qty:.4} < 0");
                }
                if new_qty > EPS {
                    // weighted-average cost, fee folded into basis
                    pos.avg_cost = (pos.qty * pos.avg_cost + q * f.px + sign * f.fee) / new_qty;
                }
                pos.qty = new_qty.max(0.0);
                self.cash_usdc -= sign * (f.qty * f.px + f.fee);
            }
            Side::Sell => {
                let new_qty = pos.qty - q;
                if new_qty < -EPS {
                    anyhow::bail!("long-only violation: sell {:.4} exceeds held {:.4}", f.qty, pos.qty);
                }
                pos.realized_pnl += sign * (f.qty * (f.px - pos.avg_cost) - f.fee);
                pos.qty = new_qty.max(0.0);
                self.cash_usdc += sign * (f.qty * f.px - f.fee);
                self.daily.realized += sign * (f.qty * (f.px - pos.avg_cost) - f.fee);
            }
        }
        pos.fees_paid += sign * f.fee;
        self.daily.fees += sign * f.fee;
        Ok(())
    }

    /// Mark a held token to the conservative exit-side best bid (DESIGN_EXECUTION §8.5).
    pub fn mark(&mut self, instrument: &str, best_bid: f64) {
        if let Some(p) = self.positions.get_mut(instrument) {
            p.mark_px = best_bid;
        }
    }

    /// Resolution settlement (DESIGN_EXECUTION §8.3): pay 1.0 if the instrument
    /// won else 0.0; realize, zero the position, mark Settled.
    pub fn settle(&mut self, instrument: &str, winner: &str) {
        if let Some(p) = self.positions.get_mut(instrument) {
            if p.qty.abs() < EPS {
                return;
            }
            let payout = if instrument == winner { 1.0 } else { 0.0 };
            let pnl = p.qty * (payout - p.avg_cost);
            p.realized_pnl += pnl;
            self.daily.realized += pnl;
            self.cash_usdc += p.qty * payout;
            p.qty = 0.0;
            p.status = PosStatus::Settled;
        }
    }

    pub fn set_status(&mut self, instrument: &str, status: PosStatus) {
        if let Some(p) = self.positions.get_mut(instrument) {
            p.status = status;
        }
    }

    pub fn snapshot(&self, instrument: &str) -> Option<PositionSnapshot> {
        let p = self.positions.get(instrument)?;
        Some(PositionSnapshot {
            instrument: p.instrument.clone(),
            qty: p.qty,
            avg_cost: p.avg_cost,
            realized_pnl: p.realized_pnl,
            fees_paid: p.fees_paid,
            mark_px: p.mark_px,
            upnl: p.upnl(),
            status: p.status.as_str().to_string(),
            cash_usdc: self.cash_usdc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(inst: &str, side: Side, qty: f64, px: f64, vtid: &str, status: FillStatus) -> Fill {
        Fill {
            venue_trade_id: vtid.into(),
            order_id: "o".into(),
            client_id: "t-1:E:0".into(),
            instrument: inst.into(),
            status,
            side,
            qty,
            px,
            fee: 0.0,
            ts_ns: 0,
        }
    }

    #[test]
    fn buy_then_sell_realizes_pnl_and_flattens() {
        let inst = "polymarket.0xabc.UP";
        let mut pm = PositionManager::new(1000.0);
        pm.apply_fill(&fill(inst, Side::Buy, 20.0, 0.40, "v1", FillStatus::Confirmed)).unwrap();
        assert!((pm.qty(inst) - 20.0).abs() < 1e-9);
        assert!((pm.cash() - (1000.0 - 8.0)).abs() < 1e-9); // spent 20*0.40
        pm.apply_fill(&fill(inst, Side::Sell, 20.0, 0.45, "v2", FillStatus::Confirmed)).unwrap();
        assert!(pm.qty(inst).abs() < 1e-9);
        let p = pm.position(inst).unwrap();
        assert!((p.realized_pnl - (20.0 * (0.45 - 0.40))).abs() < 1e-9); // +1.0
        assert!((pm.cash() - 1001.0).abs() < 1e-9);
    }

    #[test]
    fn apply_fill_is_idempotent_on_venue_trade_id() {
        let inst = "polymarket.0xabc.UP";
        let mut pm = PositionManager::new(1000.0);
        let f = fill(inst, Side::Buy, 20.0, 0.40, "v1", FillStatus::Confirmed);
        pm.apply_fill(&f).unwrap();
        pm.apply_fill(&f).unwrap(); // replay
        assert!((pm.qty(inst) - 20.0).abs() < 1e-9); // applied once
    }

    #[test]
    fn matched_then_failed_reverses() {
        let inst = "polymarket.0xabc.UP";
        let mut pm = PositionManager::new(1000.0);
        pm.apply_fill(&fill(inst, Side::Buy, 20.0, 0.40, "v1", FillStatus::Matched)).unwrap();
        assert!((pm.qty(inst) - 20.0).abs() < 1e-9);
        pm.apply_fill(&fill(inst, Side::Buy, 20.0, 0.40, "v1", FillStatus::Failed)).unwrap();
        assert!(pm.qty(inst).abs() < 1e-9); // reversed
        assert!((pm.cash() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn oversell_violates_long_only() {
        let inst = "polymarket.0xabc.UP";
        let mut pm = PositionManager::new(1000.0);
        pm.apply_fill(&fill(inst, Side::Buy, 10.0, 0.40, "v1", FillStatus::Confirmed)).unwrap();
        let err = pm.apply_fill(&fill(inst, Side::Sell, 20.0, 0.45, "v2", FillStatus::Confirmed));
        assert!(err.is_err());
    }

    #[test]
    fn one_trade_gate_and_cooldown() {
        let mut pm = PositionManager::new(1000.0);
        assert!(pm.try_begin_trade("t-1", 0));
        assert!(!pm.try_begin_trade("t-2", 0)); // slot taken
        pm.end_trade(0, 2000);
        assert!(!pm.try_begin_trade("t-3", 1_000_000_000)); // 1s < 2s cooldown
        assert!(pm.try_begin_trade("t-3", 2_000_000_000)); // cooldown elapsed
    }

    #[test]
    fn settlement_pays_winner() {
        let inst = "polymarket.0xabc.UP";
        let mut pm = PositionManager::new(1000.0);
        pm.apply_fill(&fill(inst, Side::Buy, 20.0, 0.40, "v1", FillStatus::Confirmed)).unwrap();
        pm.settle(inst, inst); // UP won
        assert!(pm.qty(inst).abs() < 1e-9);
        assert!((pm.cash() - (1000.0 - 8.0 + 20.0)).abs() < 1e-9); // got 20*1.0
    }
}
