//! The bus envelope. One `Event` per published message; `Payload` is a closed
//! enum so the hot path never does dynamic dispatch (DESIGN §3).

use serde::Serialize;

use crate::model::*;

/// Dotted topic key, e.g. `market.binance.usdt_perp.BTCUSDT.book`.
/// (Plain `String` for now; interning is a later optimization.)
pub type Topic = String;

#[derive(Clone, Debug, Serialize)]
pub struct Event {
    pub topic: Topic,
    pub source: &'static str,
    pub ts_ns: i64,
    pub seq: u64,
    pub payload: Payload,
}

#[derive(Clone, Debug, Serialize)]
pub enum Payload {
    Book(BookUpdate),
    Trade(TradeTick),
    Liq(Liquidation),
    Meta(MarketMeta),
    Signal(TradeSignal),
    Latency(LatencySample),
    ExecReport(ExecReport),
    TradeRecord(TradeRecord),
    Position(PositionSnapshot),
    /// Online per-event (Δb,Δρ) calibration (FairRide; topic `market.calib.<id>`).
    Calib(CalibUpdate),
}

impl Event {
    pub fn new(
        topic: impl Into<Topic>,
        source: &'static str,
        ts_ns: i64,
        seq: u64,
        payload: Payload,
    ) -> Self {
        Event { topic: topic.into(), source, ts_ns, seq, payload }
    }
}
