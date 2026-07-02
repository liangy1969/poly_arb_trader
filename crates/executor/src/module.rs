//! `Executor` module (DESIGN_EXECUTION §2) — wires the bus subscriptions to the
//! trade state machine and the (simulated) venue.
//!
//! Two tasks share an `ExecBookMirror` behind a mutex (a derived projection, not
//! cross-module shared memory): a **mirror task** folds `market.polymarket.#`
//! into the book/meta view; a **trade task** drains `signal.#`, runs one Trade at
//! a time through Entering→Holding→Exiting→Closed (§2.1), and settles on catalog
//! `resolved`. The trade task owns the `PositionManager` and `RiskGate` (single
//! owner — no locks). Order pricing/sizing read a fresh `BookTop` from the mirror
//! at submit time; the venue is abstracted by `TradingVenue` (sim in v1).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use arb_core::bus::{key_by_instrument, Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::model::{
    ExecReport, LegSummary, MarketStatus, Side, TradeOutcome, TradeRecord, TradeSignal,
};
use arb_core::module::{Health, Module};
use arb_core::now_ns;

use crate::config::ExecutorCfg;
use crate::mirror::{ExecBookMirror, MirrorSource};
use crate::position::{PosStatus, PositionManager};
use crate::risk::RiskGate;
use crate::types::{
    BookSource, BookTop, Fill, FillStatus, IntentKind, MarketParams, OrderIntent, TradePlan,
    VenueOutcome, MS,
};
use crate::venue::{CancelOutcome, FillSender, SimVenue, TradingVenue};
use crate::venue_spec::{self, complement, market_id_of, traded_instrument};

const EPS: f64 = 1e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Decoupled exit (exit.mode = "reconcile") — per-market state machine + slot.
//
// Design (2026-07-02, after the oversell incident): the ENTRY path stays
// latency-critical and does zero venue reads; a per-market EXIT RECONCILER task
// owns position truth (authoritative venue polls) and all close orders. A close
// only ever BUYS the complement token (buy NO to close long YES) — a buy
// structurally cannot oversell; worst case is an over-hedge into a matched pair
// (~breakeven at settle), never a naked short.
// ─────────────────────────────────────────────────────────────────────────────

/// Per-market trade lifecycle. Transitions have a SINGLE writer each: the entry
/// path drives Idle/Rest→Submitted→Filled; the reconciler drives Filled→Rest
/// (after `transit_ms`) and Rest→Idle (confirmed flat).
#[derive(Clone, Copy, Debug, PartialEq)]
enum TradeState {
    /// Flat, no cycle.
    Idle,
    /// Entry in flight (IOC outcome unknown) — reconciler frozen.
    Submitted { since_ns: i64 },
    /// Entry filled; reprice + read-replication lag settling — reconciler frozen.
    Filled { fill_ns: i64 },
    /// Steady state — reconciler may manage close orders (subject to the hold).
    Rest,
}

/// Our one resting close order on a market (the reconciler's).
#[derive(Clone, Debug)]
struct RestingClose {
    order_id: String,
    /// Token the close BUYS (the complement of the held side).
    instrument: String,
    px: f64,
    count: f64,
}

/// Shared per-market slot: the entry path (engine task) and the market's exit
/// reconciler coordinate through this. INVARIANT: a resting close may only exist
/// with the slot's knowledge — a reference is dropped ONLY once the venue
/// confirms the order is off the book (Canceled/Gone). Lock is held for field
/// access only, never across an await.
struct MarketSlot {
    state: TradeState,
    /// In-memory net estimate (+YES/−NO): entry fills apply immediately; the
    /// reconciler's authoritative polls refresh it. Used for the entry exposure
    /// cap only — close sizing always uses the fresh authoritative poll.
    net_est: f64,
    hold_until_ns: i64,
    resting: Option<RestingClose>,
    reconciler_alive: bool,
    /// YES-side instrument of this market (complement() gives the other side).
    inst_yes: String,
    params: MarketParams,
    expiry_ns: i64,
    trade_id: String,
}

impl MarketSlot {
    fn new_idle(instrument: &str, params: MarketParams, expiry_ns: i64) -> Self {
        MarketSlot {
            state: TradeState::Idle,
            net_est: 0.0,
            hold_until_ns: 0,
            resting: None,
            reconciler_alive: false,
            inst_yes: yes_side_of(instrument),
            params,
            expiry_ns,
            trade_id: String::new(),
        }
    }
}

type Slots = Arc<Mutex<HashMap<String, MarketSlot>>>;

/// Reconciler timing knobs (from `ExitCfg`), all pre-converted.
#[derive(Clone, Copy)]
struct ReconCfg {
    cadence: Duration,
    transit_ns: i64,
    submitted_timeout_ns: i64,
    settle_wait: Duration,
}

fn recon_cfg(e: &crate::config::ExitCfg) -> ReconCfg {
    ReconCfg {
        cadence: Duration::from_millis(e.cadence_ms.max(50)),
        transit_ns: e.transit_ms as i64 * MS,
        submitted_timeout_ns: e.submitted_timeout_ms as i64 * MS,
        settle_wait: Duration::from_millis(e.settle_wait_ms),
    }
}

/// The YES-side token of an instrument's market (identity if already YES-side).
fn yes_side_of(instrument: &str) -> String {
    if let Some(spec) = venue_spec::venue_of(instrument) {
        if instrument.ends_with(&format!(".{}", spec.yes_label)) {
            return instrument.to_string();
        }
    }
    complement(instrument)
}

/// Entry gate against the market's state machine: reject while an entry is in
/// flight or within the Filled→Rest transit window; stale Submitted (entry task
/// died) falls through so the market can't deadlock.
fn entry_gate(state: TradeState, now_ns_: i64, rc: &ReconCfg) -> Result<(), &'static str> {
    match state {
        TradeState::Submitted { since_ns } if now_ns_ - since_ns <= rc.submitted_timeout_ns => {
            Err("entry in flight")
        }
        TradeState::Filled { fill_ns } if now_ns_ - fill_ns < rc.transit_ns => {
            Err("entry transit gate")
        }
        _ => Ok(()),
    }
}

/// Per-market exposure cap: reject entries that GROW |net| past the cap;
/// always allow entries that reduce it (an opposite-direction entry nets out).
fn exposure_ok(net_est: f64, dir_sign: f64, size: f64, cap: f64) -> bool {
    let after = net_est + dir_sign * size;
    after.abs() <= cap + EPS || after.abs() <= net_est.abs() + EPS
}

/// A close order BUYS the complement token at `close_px`; economically that is a
/// SELL of the held token at `1 − close_px` (Kalshi auto-nets the pair). Translate
/// so the long-only PositionManager books the exit + realized P&L directly.
fn translated_close_fill(held_inst: &str, order_id: &str, seq: u64, qty: f64, close_px: f64) -> Fill {
    Fill {
        venue_trade_id: format!("{order_id}-cl{seq}"),
        order_id: order_id.to_string(),
        client_id: format!("close:{held_inst}"),
        instrument: held_inst.to_string(),
        status: FillStatus::Confirmed,
        side: Side::Sell,
        qty,
        px: 1.0 - close_px,
        fee: 0.0, // post-only maker fill: fee-free on Kalshi
        ts_ns: now_ns(),
    }
}

pub struct Executor {
    cfg: ExecutorCfg,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl Executor {
    pub fn new(cfg: ExecutorCfg) -> Self {
        Executor { cfg, handles: Vec::new() }
    }
}

#[async_trait]
impl Module for Executor {
    fn name(&self) -> &'static str {
        "executor"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        if !self.cfg.enabled {
            tracing::info!("executor disabled");
            return Ok(());
        }

        // Active prediction venue (DESIGN_MULTI_VENUE): drives the bus topics, the
        // instrument outcome model, and the sim's taker-delay floor.
        let spec = venue_spec::by_name(&self.cfg.venue.market).unwrap_or_else(|| {
            tracing::warn!("unknown venue.market '{}', defaulting to polymarket", self.cfg.venue.market);
            venue_spec::POLYMARKET
        });
        tracing::info!("active prediction venue = {} (taker_delay={}ms)", spec.prefix, spec.taker_delay_ms);

        let mirror = Arc::new(Mutex::new(ExecBookMirror::new()));

        // Mirror task: fold market.<venue>.# (books + catalog meta) into the
        // projection. Conflated so a book flood can't back it up.
        let msub = bus.subscribe(&format!("market.{}.#", spec.prefix), 4096, Policy::Conflate(key_by_instrument));
        let mmirror = mirror.clone();
        self.handles.push(tokio::spawn(async move {
            let mut sub = msub;
            while let Some(ev) = sub.recv().await {
                mmirror.lock().unwrap().on_event(&ev);
            }
        }));

        // Trade-tape task: a separate BLOCK subscription (NOT conflated, so trade
        // bursts/sweeps aren't collapsed) feeds the mirror's rolling trade tape.
        // The price probe reads it to tell a sweep (volume prints) from an MM
        // cancel-and-reprice (no prints) when depth drops.
        let tsub = bus.subscribe(&format!("market.{}.#", spec.prefix), 8192, Policy::Block);
        let tmirror = mirror.clone();
        self.handles.push(tokio::spawn(async move {
            let mut sub = tsub;
            while let Some(ev) = sub.recv().await {
                if let Payload::Trade(t) = &ev.payload {
                    tmirror.lock().unwrap().on_trade(&t.instrument, t.exch_ts_ns, t.qty);
                }
            }
        }));

        // Trade task: signals + settlement + the state machine.
        let sigsub = bus.subscribe("signal.#", 256, Policy::Block);
        let catsub = bus.subscribe(&format!("market.{}.catalog", spec.prefix), 256, Policy::Block);

        // Venue holds the book source; its background producers (sim: the
        // near-expiry liquidator) push async fills onto `fill_tx` — the §8.6
        // fill-ingestion path a live adapter would feed from the user WS + pull.
        let source: Arc<dyn BookSource> = Arc::new(MirrorSource(mirror.clone()));
        let venue: Arc<dyn TradingVenue> = match self.cfg.venue.adapter.as_str() {
            "sim" => Arc::new(SimVenue::new(
                spec.taker_delay_ms,
                self.cfg.sim.force_liquidity,
                self.cfg.sim.force_window_ms,
                self.cfg.sim.force_check_ms,
                source,
            )),
            "kalshi" => Arc::new(crate::venue_kalshi::KalshiVenue::new(
                &self.cfg.venue.key_id,
                &self.cfg.venue.private_key_path,
                &self.cfg.venue.network,
                self.cfg.venue.max_order_usdc,
            )?),
            other => anyhow::bail!(
                "venue adapter '{other}' not implemented (have: sim, kalshi; polymarket_clob is P2)"
            ),
        };
        // The removed "maker" exit stranded/oversold positions (2026-07-02
        // incident) — refuse to start rather than silently fall back.
        if self.cfg.exit.mode == "maker" {
            anyhow::bail!(
                "exit.mode 'maker' was removed after the 2026-07-02 oversell incident; use 'reconcile'"
            );
        }

        let (fill_tx, fill_rx) = tokio::sync::mpsc::unbounded_channel();
        self.handles.extend(venue.start(fill_tx.clone()));

        let slots: Slots = Arc::new(Mutex::new(HashMap::new()));
        let entries_halted = Arc::new(AtomicBool::new(false));

        // Real-balance kill switch: poll the venue's ACTUAL balance (never the
        // executor's self-reported P&L — that hid the incident's losses) and
        // latch a halt on new entries below the floor. Exits keep flattening.
        if self.cfg.risk.min_balance_usd > 0.0 {
            let v = venue.clone();
            let halted = entries_halted.clone();
            let floor = self.cfg.risk.min_balance_usd;
            self.handles.push(tokio::spawn(async move {
                let mut tick = tokio::time::interval(Duration::from_secs(60));
                loop {
                    tick.tick().await;
                    match v.balance().await {
                        Ok(b) if b < floor => {
                            if !halted.swap(true, Ordering::SeqCst) {
                                tracing::error!(
                                    target: "executor",
                                    "REAL-BALANCE KILL: ${b:.2} < floor ${floor:.2} — new entries halted (exits keep flattening)"
                                );
                            }
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!(target: "executor", "balance poll: {e}"),
                    }
                }
            }));
        }

        // Crash recovery (reconcile mode): anything still open at the venue gets
        // a reconciler immediately — a restart must never orphan a position.
        if self.cfg.exit.mode == "reconcile" {
            let v = venue.clone();
            let m = mirror.clone();
            let sl = slots.clone();
            let ftx = fill_tx.clone();
            let rc = recon_cfg(&self.cfg.exit);
            let fallback = MarketParams {
                min_order_size: self.cfg.sim.min_order_size,
                tick_size: self.cfg.sim.tick_size,
                fee_rate: self.cfg.sim.fee_rate,
            };
            let prefix = spec.prefix;
            let yes_label = spec.yes_label;
            self.handles.push(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(3)).await; // let the mirror warm
                match v.open_positions().await {
                    Ok(list) => {
                        for (mid, net) in list {
                            tracing::warn!(
                                target: "exit",
                                "recovered open position {mid} net={net:+.1} — spawning reconciler"
                            );
                            {
                                let mut g = sl.lock().unwrap();
                                let s = g.entry(mid.clone()).or_insert_with(|| {
                                    MarketSlot::new_idle(
                                        &format!("{prefix}.{mid}.{yes_label}"),
                                        fallback,
                                        i64::MAX,
                                    )
                                });
                                if s.reconciler_alive {
                                    continue; // an entry already spawned one
                                }
                                s.state = TradeState::Rest;
                                s.net_est = net;
                                s.hold_until_ns = 0; // recovered: close ASAP
                                s.reconciler_alive = true;
                                s.trade_id = format!("recovered-{mid}");
                            }
                            tokio::spawn(exit_reconciler(
                                mid,
                                sl.clone(),
                                v.clone(),
                                m.clone(),
                                ftx.clone(),
                                rc,
                            ));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(target: "exit", "startup position sweep: {e} (no crash recovery)")
                    }
                }
            }));
        }

        // Hold-period study sampler (optional): shadow-samples each entry's
        // realized exit at a ladder of hold offsets. Pure observation — doesn't
        // touch the real trade.
        let hold_tx = if self.cfg.hold_probe_ms.is_empty() {
            None
        } else {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<HoldProbe>();
            let m = mirror.clone();
            let ladder = self.cfg.hold_probe_ms.clone();
            self.handles.push(tokio::spawn(hold_sampler(m, rx, ladder)));
            Some(tx)
        };

        let maker_tx = if self.cfg.maker_probe.enabled {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<MakerProbe>();
            let m = mirror.clone();
            let cfg = self.cfg.maker_probe.clone();
            self.handles.push(tokio::spawn(maker_sampler(m, rx, cfg)));
            Some(tx)
        } else {
            None
        };

        // Post-signal price-trajectory sampler (optional): on every triggered
        // trade, log the traded-token book at a step_ms ladder over window_ms.
        let px_tx = if self.cfg.price_probe.enabled {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<PxProbe>();
            let m = mirror.clone();
            let step = self.cfg.price_probe.step_ms.max(1);
            let ladder: Vec<u64> = (0..=self.cfg.price_probe.window_ms).step_by(step as usize).collect();
            self.handles.push(tokio::spawn(price_sampler(m, rx, ladder, step)));
            Some(tx)
        } else {
            None
        };

        let mut engine = Engine {
            cfg: self.cfg.clone(),
            bus: bus.clone(),
            mirror,
            venue,
            pm: PositionManager::new(self.cfg.sim.start_cash_usdc),
            risk: RiskGate::new(self.cfg.risk.clone()),
            seq: 0,
            hold_tx,
            maker_tx,
            px_tx,
            slots,
            fill_tx,
            entries_halted,
        };
        self.handles.push(tokio::spawn(async move {
            let mut sigsub = sigsub;
            let mut catsub = catsub;
            let mut fill_rx = fill_rx;
            tracing::info!(
                "executor up (venue={}, size={} shares, hold={}ms)",
                engine.venue.name(),
                engine.cfg.sizing.size_shares,
                engine.cfg.hold_ms,
            );
            loop {
                tokio::select! {
                    sig = sigsub.recv() => {
                        let Some(ev) = sig else { break };
                        if let Payload::Signal(s) = &ev.payload {
                            engine.on_signal(s).await;
                        }
                    }
                    cat = catsub.recv() => {
                        let Some(ev) = cat else { break };
                        if let Payload::Meta(m) = &ev.payload {
                            engine.on_catalog(m.status, &m.instrument, m.winner.as_deref());
                        }
                    }
                    // Async fills: sim liquidations + reconciler close fills
                    // (routed by the "close:" client_id prefix).
                    fill = fill_rx.recv() => {
                        let Some(f) = fill else { break };
                        if f.client_id.starts_with("close:") {
                            engine.on_close_fill(&f);
                        } else {
                            engine.on_liquidation_fill(&f);
                        }
                    }
                }
            }
        }));
        Ok(())
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        for h in self.handles.drain(..) {
            h.abort();
        }
        Ok(())
    }

    fn health(&self) -> Health {
        Health::Ok
    }
}

/// Owns the active trade, PM, risk gate, and venue. Single-threaded (one task).
struct Engine {
    cfg: ExecutorCfg,
    bus: Arc<dyn Bus>,
    mirror: Arc<Mutex<ExecBookMirror>>,
    venue: Arc<dyn TradingVenue>,
    pm: PositionManager,
    risk: RiskGate,
    seq: u64,
    /// Hold-period study probe (None = off).
    hold_tx: Option<tokio::sync::mpsc::UnboundedSender<HoldProbe>>,
    /// Maker-exit study probe (None = off).
    maker_tx: Option<tokio::sync::mpsc::UnboundedSender<MakerProbe>>,
    /// Post-signal price-trajectory probe (None = off).
    px_tx: Option<tokio::sync::mpsc::UnboundedSender<PxProbe>>,
    /// Per-market trade-lifecycle slots shared with the exit reconcilers.
    slots: Slots,
    /// Sender the reconcilers push translated close fills through (same channel
    /// as the venue's async fills; routed by the "close:" client_id prefix).
    fill_tx: FillSender,
    /// Latched by the real-balance watchdog: halt NEW entries (exits keep running).
    entries_halted: Arc<AtomicBool>,
}

/// One entry registered with the hold-period sampler: it samples the realized
/// exit (book sweep for `qty` + fee) at a ladder of hold offsets from `t0_ns`.
struct HoldProbe {
    trade_id: String,
    inst: String,
    t0_ns: i64,
    qty: f64,
    entry_vwap: f64,
    fee_rate: f64,
}

/// One exit registered with the maker-exit sampler: at exit-decision time the
/// Post-Only sell would rest at `offer_px`; the bid then was `bid0`. The sampler
/// resolves the outcome (reject / maker-fill / adverse / timeout) over the rest
/// window and compares the maker exit to the taker cross at `bid0`.
struct MakerProbe {
    trade_id: String,
    inst: String,
    t0_ns: i64,
    offer_px: f64,
    bid0: f64,
    fee_rate: f64,
}

/// One triggered signal registered with the price-trajectory sampler: it samples
/// the traded-token top book at a ladder of offsets from `t0_ns` and logs the
/// reprice path (bid/ask/mid + ¢ delta from the signal reference). Fires whether
/// or not our order fills, so it validates signal direction on abandoned trades.
struct PxProbe {
    trade_id: String,
    inst: String,
    t0_ns: i64,
    ref_ask: f64,
    ref_mid: f64,
}

impl Engine {
    fn nxt(&mut self) -> u64 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    fn top(&self, instrument: &str) -> Option<BookTop> {
        self.mirror.lock().unwrap().top(instrument)
    }

    fn report(&mut self, trade_id: &str, instrument: &str, state: &str, detail: &str) {
        tracing::info!(target: "executor", "[{trade_id}] {state} {instrument} {detail}");
        let seq = self.nxt();
        self.bus.publish(Event::new(
            "exec.report",
            "executor",
            now_ns(),
            seq,
            Payload::ExecReport(ExecReport {
                trade_id: trade_id.into(),
                instrument: instrument.into(),
                state: state.into(),
                detail: detail.into(),
                ts_ns: now_ns(),
            }),
        ));
    }

    fn emit_position(&mut self, instrument: &str) {
        if let Some(snap) = self.pm.snapshot(instrument) {
            let seq = self.nxt();
            self.bus.publish(Event::new(
                "exec.position",
                "executor",
                now_ns(),
                seq,
                Payload::Position(snap),
            ));
        }
    }

    /// Apply an async fill from the venue (the near-expiry liquidator, and live
    /// fills later). The liquidator flattens whatever a market still holds near
    /// expiry — but a position entered close to expiry is *also* exited by the
    /// normal ladder, which runs inline and applies first (fills queue on `fill_rx`
    /// until `on_signal` returns). So a liquidation fill often lands on an
    /// already-flat (or smaller) PM. **Clamp the sell to what's actually held** —
    /// the liquidator can only sell remaining inventory; selling more is the benign
    /// exit↔liquidator race, not a phantom fill, so it must not trip the kill.
    /// (A live adapter's real oversell is a separate P2 reconcile concern.)
    fn on_liquidation_fill(&mut self, f: &Fill) {
        let Some(clamped) = clamp_liquidation(self.pm.qty(&f.instrument), f) else {
            self.report(
                "liquidation",
                &f.instrument,
                "Skip",
                &format!("{:.2} @ {:.3}: already flat (normal exit beat the liquidator)", f.qty, f.px),
            );
            return;
        };
        match self.pm.apply_fill(&clamped) {
            Ok(()) => {
                self.report(
                    "liquidation",
                    &f.instrument,
                    "Liquidated",
                    &format!("{:.2} @ {:.3} (near expiry, sim)", clamped.qty, clamped.px),
                );
                self.pm.set_status(&f.instrument, PosStatus::Settled);
                self.emit_position(&f.instrument);
            }
            Err(e) => {
                // Clamped to held, so this should not fire — keep the safety net.
                self.report("liquidation", &f.instrument, "Halt", &format!("fill error: {e}"));
                self.risk.trip("liquidation fill long-only violation");
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_trade(&mut self, plan: &TradePlan, outcome: TradeOutcome, entry: &LegSummary, exit: &LegSummary, hold_actual_ms: u64, exit_ref_bid: f64) {
        // pnl on the round-tripped shares (full exit: entry.qty == exit.qty).
        let pnl_gross = exit.qty * exit.vwap - entry.qty * entry.vwap;
        let pnl_net = (exit.qty * exit.vwap - exit.fees) - (entry.qty * entry.vwap + entry.fees);
        let rec = TradeRecord {
            trade_id: plan.trade_id.clone(),
            outcome,
            direction: plan.direction,
            instrument: plan.instrument.clone(),
            signal_ts_ns: plan.signal_ts_ns,
            trigger: plan.trigger.clone(),
            entry: entry.clone(),
            exit: exit.clone(),
            hold_actual_ms,
            pnl_gross,
            pnl_net,
            slippage_entry_c: (entry.vwap - plan.signal_ask) * 100.0,
            slippage_exit_c: if exit.qty > 0.0 { (exit_ref_bid - exit.vwap) * 100.0 } else { 0.0 },
            lat_signal_to_submit_ms: (entry.first_ts_ns.max(plan.signal_ts_ns) - plan.signal_ts_ns) as f64 / 1e6,
            lat_submit_to_ack_ms: 0.0, // sim: folded into the taker-delay floor
            lat_ack_to_fill_ms: 0.0,
        };
        tracing::info!(
            target: "executor",
            "*** TRADE {} {:?} *** pnl_net={:.4} gross={:.4} entry={:.2}@{:.3} exit={:.2}@{:.3} hold={}ms",
            rec.trade_id, rec.outcome, rec.pnl_net, rec.pnl_gross,
            entry.qty, entry.vwap, exit.qty, exit.vwap, hold_actual_ms,
        );
        let seq = self.nxt();
        self.bus.publish(Event::new("exec.trade", "executor", now_ns(), seq, Payload::TradeRecord(rec)));
    }

    /// Settlement on catalog `resolved` (DESIGN_EXECUTION §8.3).
    fn on_catalog(&mut self, status: MarketStatus, instrument: &str, winner: Option<&str>) {
        if status != MarketStatus::Resolved {
            return;
        }
        let Some(winner) = winner else { return };
        // The catalog keys meta on the UP token; settle both outcomes of the cid.
        for inst in [instrument.to_string(), traded_instrument(instrument, -1)] {
            if self.pm.qty(&inst) > EPS {
                self.pm.settle(&inst, winner);
                self.report("settlement", &inst, "Settled", &format!("winner={winner}"));
                self.emit_position(&inst);
                self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            }
        }
    }

    /// A signal arrived: build + validate the plan, claim the one-trade slot, and
    /// drive the trade to a terminal state (Entering→…→Closed/Abandoned).
    async fn on_signal(&mut self, s: &TradeSignal) {
        let now = now_ns();
        let instrument = traded_instrument(&s.target, s.direction);
        let trade_id = format!("{}-{}", s.strategy, s.ts_ns);

        // Real-balance kill switch (latched by the watchdog): no new entries.
        if self.entries_halted.load(Ordering::SeqCst) {
            self.report(&trade_id, &instrument, "Rejected", "real-balance kill switch");
            return;
        }

        // Traded-token book + tte from the mirror.
        let Some(book) = self.top(&instrument) else {
            self.report(&trade_id, &instrument, "Rejected", "no book in mirror");
            return;
        };
        let tte_ms = self.mirror.lock().unwrap().tte_ms(&instrument, now);
        // Per-market economics (min size / tick / fee) from catalog meta, with
        // config fallback when a market's catalog didn't carry them.
        let params = self.market_params(&instrument);

        // Sizing (§3.2): fixed target capped by displayed top-of-book depth.
        let size = self.cfg.sizing.size_shares.min(self.cfg.sizing.depth_frac * book.ask_sz);
        if size < params.min_order_size {
            self.report(&trade_id, &instrument, "Rejected", &format!("size {size:.2} < min"));
            return;
        }

        // Risk gate (§9 feasible subset).
        if let Err(reason) = self.risk.check(s, book, tte_ms, now, size, book.best_ask) {
            self.report(&trade_id, &instrument, "Rejected", &reason);
            return;
        }

        let expiry_ns =
            self.mirror.lock().unwrap().meta_of(&plan_instrument(s)).map(|m| m.expiry_ns).unwrap_or(now);
        let plan = TradePlan {
            trade_id: trade_id.clone(),
            instrument: instrument.clone(),
            token_id: String::new(), // sim: not needed; real adapter reads it from catalog meta (P2)
            direction: s.direction,
            size_shares: size,
            hold_ms: if s.hold_ms > 0 { s.hold_ms } else { self.cfg.hold_ms },
            exit_deadline_ns: expiry_ns - self.cfg.exit.deadline_buffer_ms as i64 * MS,
            signal_ts_ns: s.ts_ns,
            trigger: s.trigger.clone(),
            signal_ask: book.best_ask,
            params,
            expiry_ns,
        };

        // Decoupled path: entry only; the exit reconciler owns the close.
        if self.cfg.exit.mode == "reconcile" {
            self.run_entry_reconcile(plan).await;
            return;
        }

        // Legacy inline path (cross): one-trade gate + entry→hold→exit.
        if !self.pm.try_begin_trade(&trade_id, now) {
            self.report(&trade_id, &instrument, "Rejected", "active trade / cooldown");
            return;
        }
        self.risk.note_trade(now);
        self.run_trade(plan).await;
    }

    /// Resolve per-market economics from the catalog meta, falling back to the
    /// sim config when a market didn't supply them.
    fn market_params(&self, instrument: &str) -> MarketParams {
        let m = self.mirror.lock().unwrap().meta_of(instrument);
        MarketParams {
            min_order_size: m.and_then(|x| x.min_order_size).unwrap_or(self.cfg.sim.min_order_size),
            tick_size: m.and_then(|x| x.tick_size).unwrap_or(self.cfg.sim.tick_size),
            fee_rate: m.and_then(|x| x.fee_rate).unwrap_or(self.cfg.sim.fee_rate),
        }
    }

    async fn run_trade(&mut self, plan: TradePlan) {
        let inst = plan.instrument.clone();
        self.report(&plan.trade_id, &inst, "Entering", &format!("size={:.2}", plan.size_shares));
        self.fire_probe(&plan);

        // Shadow / dry-run: the probe has fired; place NO order, free the
        // one-trade slot + arm the cooldown, and we're done. No real money.
        if self.cfg.dry_run {
            self.report(&plan.trade_id, &inst, "Shadow", "dry_run: probe only, no order");
            self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            return;
        }

        // ── ENTRY: bounded FAK attempt loop (§5) ──
        let Ok(entry) = self.entry_chase(&plan).await else {
            return; // kill switch tripped in fill application (abort_kill reported)
        };

        if entry.qty <= EPS {
            // ── ABANDONED: entry never filled; no position ever existed (§5) ──
            self.report(&plan.trade_id, &inst, "Abandoned", "no entry fill");
            self.emit_trade(&plan, TradeOutcome::Abandoned, &entry, &LegSummary::default(), 0, 0.0);
            self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            return;
        }

        // ── HOLDING: arm the exit timer off the first fill (§6.1) ──
        self.emit_position(&inst);
        self.report(&plan.trade_id, &inst, "Holding", &format!("qty={:.2} @ {:.3}", entry.qty, entry.vwap));
        // Register the entry with the hold-period sampler (study only).
        if let Some(tx) = &self.hold_tx {
            let _ = tx.send(HoldProbe {
                trade_id: plan.trade_id.clone(),
                inst: inst.clone(),
                t0_ns: entry.first_ts_ns,
                qty: entry.qty,
                entry_vwap: entry.vwap,
                fee_rate: plan.params.fee_rate,
            });
        }
        // Cap the hold at the exit deadline: with min_tte low/zero a trade can enter
        // near settle, and a full hold_ms would sleep PAST the deadline (even past
        // settlement), stranding the position. min() makes the hold deadline-aware —
        // a no-op for normal-TTE trades (hold ends well before the deadline).
        let exit_due = (entry.first_ts_ns + plan.hold_ms as i64 * MS).min(plan.exit_deadline_ns);
        sleep_until(exit_due).await;

        // ── EXITING: cross ladder, deepening only on misses (§6.2) ──
        self.report(&plan.trade_id, &inst, "Exiting", "");
        // Register the maker-exit shadow probe (study only): what a Post-Only
        // maker exit would realize here vs the taker cross we're about to run.
        if let Some(tx) = &self.maker_tx {
            if let Some(b) = self.top(&inst) {
                let offer_px = if self.cfg.maker_probe.improve_tick {
                    (b.best_ask - plan.params.tick_size).max(plan.params.tick_size)
                } else {
                    b.best_ask
                };
                let _ = tx.send(MakerProbe {
                    trade_id: plan.trade_id.clone(),
                    inst: inst.clone(),
                    t0_ns: now_ns(),
                    offer_px,
                    bid0: b.best_bid,
                    fee_rate: plan.params.fee_rate,
                });
            }
        }
        let mut exit = LegSummary::default();
        let mut exit_ref_bid = self.top(&inst).map(|b| b.best_bid).unwrap_or(0.0);
        self.cross_exit(&plan, &inst, &mut exit, &mut exit_ref_bid, plan.exit_deadline_ns).await;

        // ── terminal ──
        let hold_actual_ms = ((exit.last_ts_ns.max(entry.first_ts_ns) - entry.first_ts_ns).max(0) / MS) as u64;
        let outcome = if self.pm.qty(&inst) <= EPS {
            self.report(&plan.trade_id, &inst, "Closed", &format!("exit qty={:.2} @ {:.3}", exit.qty, exit.vwap));
            TradeOutcome::Closed
        } else {
            // Still long after the exit ladder — the venue gave no liquidity (book
            // empty / after expiry; force_liquidity normally clears it). Hold to
            // resolution (§6.4): the position stays PendingResolution, holding the
            // one-trade slot until settlement clears it (§8.3 redemption, P2).
            // No synthetic fill here — the executor never fabricates a trade.
            let resid = self.pm.qty(&inst);
            self.pm.set_status(&inst, PosStatus::PendingResolution);
            self.report(
                &plan.trade_id,
                &inst,
                "PendingResolution",
                &format!("ALERT: long {resid:.2} at exit deadline; awaiting resolution"),
            );
            TradeOutcome::PendingResolution
        };
        self.emit_position(&inst);
        self.emit_trade(&plan, outcome, &entry, &exit, hold_actual_ms, exit_ref_bid);
        self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
    }

    fn intent(&self, plan: &TradePlan, leg: &str, attempt: u32, side: Side, price: f64, size: f64) -> OrderIntent {
        OrderIntent {
            client_id: format!("{}:{}:{}", plan.trade_id, leg, attempt),
            instrument: plan.instrument.clone(),
            token_id: plan.token_id.clone(),
            side,
            price,
            size,
            kind: IntentKind::TakeNow,
            params: plan.params,
            expiry_ns: plan.expiry_ns,
        }
    }

    /// Taker cross exit (§6.2): a deepening IOC cross ladder into the bid, bounded
    /// by `deadline_ns`. The primary (mode="cross") path passes the soft
    /// `exit_deadline_ns`; the maker fallback passes the hard settle `expiry_ns` so
    /// the ladder actually runs in the deadline→settle buffer instead of no-opping.
    async fn cross_exit(&mut self, plan: &TradePlan, inst: &str, exit: &mut LegSummary, exit_ref_bid: &mut f64, deadline_ns: i64) {
        let mut k = 0u32;
        loop {
            if self.pm.qty(inst) <= EPS {
                break;
            }
            if now_ns() >= deadline_ns {
                break; // → PendingResolution
            }
            let held = self.pm.qty(inst);
            let Some(book) = self.top(inst) else { break };
            if k == 0 {
                *exit_ref_bid = book.best_bid;
            }
            let floor = (book.best_bid - k as f64 * self.cfg.exit.step_c)
                .max(book.best_bid - self.cfg.exit.max_slip_c)
                .max(plan.params.tick_size);
            let intent = self.intent(plan, "X", k, Side::Sell, floor, held);
            let venue = self.venue.clone();
            match venue.submit(&intent).await {
                VenueOutcome::Acked { fills, .. } => {
                    if !fills.is_empty() && self.ingest_fills(&fills, exit).is_err() {
                        self.report(&plan.trade_id, inst, "Reject", "exit fill-apply error");
                        break;
                    }
                }
                VenueOutcome::Rejected(r) => self.report(&plan.trade_id, inst, "Reject", &r),
            }
            k += 1;
            if k >= self.cfg.exit.max_attempts {
                break;
            }
            tokio::time::sleep(Duration::from_millis(self.cfg.exit.retry_interval_ms)).await;
        }
    }

    /// Fire the post-signal price-trajectory probe (PXMETA + trajectory ladder),
    /// anchored at the signal moment. Fires whether or not the entry fills.
    fn fire_probe(&mut self, plan: &TradePlan) {
        let inst = plan.instrument.clone();
        if let Some(tx) = &self.px_tx {
            if let Some(b) = self.top(&inst) {
                // Per-trigger metadata: tte + the pre-signal Kalshi ask move + the
                // trigger. dry_run skips emit_trade, so this is the only record.
                let (tte, pre) = {
                    let m = self.mirror.lock().unwrap();
                    let pre_w = self.cfg.price_probe.pre_window_ms as i64 * MS;
                    (
                        m.tte_ms(&inst, now_ns()),
                        m.ask_move_c(&inst, plan.signal_ts_ns - pre_w, plan.signal_ts_ns),
                    )
                };
                tracing::info!(
                    target: "pxprobe",
                    "PXMETA trade={} inst={} tte_ms={} pre_move_c={} signal_ask={:.3} bps={:+.2} yes_px={:.3} tgt_c={:+.1}",
                    plan.trade_id, inst,
                    tte.map(|t| t.to_string()).unwrap_or_else(|| "?".into()),
                    pre.map(|v| format!("{:+.2}", v)).unwrap_or_else(|| "?".into()),
                    plan.signal_ask, plan.trigger.move_bps, plan.trigger.yes_price, plan.trigger.target_move_c,
                );
                let _ = tx.send(PxProbe {
                    trade_id: plan.trade_id.clone(),
                    inst: inst.clone(),
                    t0_ns: plan.signal_ts_ns,
                    ref_ask: plan.signal_ask,
                    ref_mid: 0.5 * (b.best_bid + b.best_ask),
                });
            }
        }
    }

    /// Bounded FAK entry attempt loop (§5), shared by both exit modes. Returns
    /// the entry leg (possibly empty = no fill), or Err(()) after a kill-switch
    /// trip (fill application violated the long-only invariant; already reported).
    async fn entry_chase(&mut self, plan: &TradePlan) -> Result<LegSummary, ()> {
        let inst = plan.instrument.clone();
        let mut entry = LegSummary::default();
        let mut attempt = 0u32;
        loop {
            if now_ns() - plan.signal_ts_ns >= self.cfg.entry.ttl_ms as i64 * MS {
                self.report(&plan.trade_id, &inst, "Entering", "entry ttl elapsed");
                break;
            }
            let Some(book) = self.top(&inst) else { break };
            let cap = book.best_ask + self.cfg.entry.chase_c;
            if cap - plan.signal_ask > self.cfg.entry.max_chase_total_c {
                self.report(&plan.trade_id, &inst, "Entering", "ask ran past chase cap");
                break;
            }
            let remaining = (plan.size_shares - entry.qty).max(0.0);
            let intent = self.intent(plan, "E", attempt, Side::Buy, cap, remaining);
            let venue = self.venue.clone();
            match venue.submit(&intent).await {
                VenueOutcome::Acked { fills, .. } => {
                    // Chase-tradeoff probe: drift over the taker delay is the
                    // chase a fill needed. `fill_ask` is the exact fill price (a
                    // fill) or the post-delay ask re-read (a miss). One line per
                    // signal's first attempt → fill-rate-vs-chase curve offline.
                    if attempt == 0 {
                        let (filled, fill_ask) = match fills.first() {
                            Some(f) => (true, f.px),
                            None => (false, self.top(&inst).map(|b| b.best_ask).unwrap_or(f64::NAN)),
                        };
                        tracing::info!(
                            target: "chase",
                            "CHASE trade={} signal_ask={:.3} fill_ask={:.3} drift_c={:.2} ask_sz={:.1} filled={}",
                            plan.trade_id, plan.signal_ask, fill_ask,
                            (fill_ask - plan.signal_ask) * 100.0, book.ask_sz, filled,
                        );
                    }
                    if fills.is_empty() {
                        // FAK no-cross: retry per loop.
                    } else {
                        if self.ingest_fills(&fills, &mut entry).is_err() {
                            self.abort_kill(plan, &inst);
                            return Err(());
                        }
                        break; // partial or full → proceed
                    }
                }
                VenueOutcome::Rejected(r) => {
                    self.report(&plan.trade_id, &inst, "Reject", &r);
                    break;
                }
            }
            attempt += 1;
            if attempt >= self.cfg.entry.max_attempts {
                break;
            }
            tokio::time::sleep(Duration::from_millis(self.cfg.entry.retry_delay_ms)).await;
        }
        Ok(entry)
    }

    /// Decoupled entry (exit.mode = "reconcile"): gate on the per-market state
    /// machine + exposure cap, cancel our resting close (self-cross guard), fire
    /// the taker entry, then hand the position to the market's exit reconciler.
    /// NO hold/exit here — the reconciler owns the close.
    async fn run_entry_reconcile(&mut self, plan: TradePlan) {
        let now = now_ns();
        let inst = plan.instrument.clone();
        let Some(mid) = market_id_of(&inst).map(str::to_string) else {
            self.report(&plan.trade_id, &inst, "Rejected", "no market id");
            return;
        };
        // Exposure sign of BUYING this token: YES-side +1, NO-side −1.
        let dir_sign = if inst == yes_side_of(&inst) { 1.0 } else { -1.0 };
        let rc = recon_cfg(&self.cfg.exit);

        // Global entry spacing (cooldown): begin/end brackets the ENTRY only.
        if !self.pm.try_begin_trade(&plan.trade_id, now) {
            self.report(&plan.trade_id, &inst, "Rejected", "cooldown");
            return;
        }

        // ── claim the market slot: state machine + exposure cap (in-memory) ──
        let claim: Result<Option<RestingClose>, &'static str> = {
            let mut g = self.slots.lock().unwrap();
            let s = g
                .entry(mid.clone())
                .or_insert_with(|| MarketSlot::new_idle(&inst, plan.params, plan.expiry_ns));
            match entry_gate(s.state, now, &rc) {
                Err(r) => Err(r),
                Ok(()) => {
                    if !exposure_ok(
                        s.net_est,
                        dir_sign,
                        plan.size_shares,
                        self.cfg.risk.max_net_per_market,
                    ) {
                        Err("exposure cap")
                    } else {
                        s.state = TradeState::Submitted { since_ns: now };
                        s.inst_yes = yes_side_of(&inst);
                        s.params = plan.params;
                        s.expiry_ns = plan.expiry_ns;
                        s.trade_id = plan.trade_id.clone();
                        Ok(s.resting.take())
                    }
                }
            }
        };
        let resting = match claim {
            Err(r) => {
                self.report(&plan.trade_id, &inst, "Rejected", r);
                self.pm.end_trade(now, 0); // free the slot; no cooldown burn on a reject
                return;
            }
            Ok(r) => r,
        };
        self.risk.note_trade(now);

        // ── cancel our resting close BEFORE the taker (self-cross guard) ──
        if let Some(r) = resting {
            match self.venue.clone().cancel_order(&r.order_id).await {
                Ok(CancelOutcome::Canceled { reduced_by }) => {
                    let filled = (r.count - reduced_by).max(0.0);
                    if filled > EPS {
                        // Last-moment close fill: account it before entering.
                        let held_inst = complement(&r.instrument);
                        let f = translated_close_fill(&held_inst, &r.order_id, 0, filled, r.px);
                        self.on_close_fill(&f);
                        let mut g = self.slots.lock().unwrap();
                        if let Some(s) = g.get_mut(&mid) {
                            s.net_est -= s.net_est.signum() * filled;
                        }
                    }
                }
                Ok(CancelOutcome::Gone) => {
                    tracing::warn!(
                        target: "exit",
                        "[{mid}] resting close already terminal at entry; position poll reconciles"
                    );
                }
                Err(e) => {
                    // Order state UNKNOWN — restore the reference and abort the
                    // entry rather than risk an orphan (the incident's bug).
                    self.report(
                        &plan.trade_id,
                        &inst,
                        "Rejected",
                        &format!("cancel resting close failed: {e}"),
                    );
                    let mut g = self.slots.lock().unwrap();
                    if let Some(s) = g.get_mut(&mid) {
                        s.resting = Some(r);
                        s.state = TradeState::Rest;
                    }
                    self.pm.end_trade(now, 0);
                    return;
                }
            }
        }

        // ── probe + dry-run + taker entry ──
        self.report(&plan.trade_id, &inst, "Entering", &format!("size={:.2}", plan.size_shares));
        self.fire_probe(&plan);
        if self.cfg.dry_run {
            self.report(&plan.trade_id, &inst, "Shadow", "dry_run: probe only, no order");
            self.release_submitted(&mid);
            self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            return;
        }
        let entry = match self.entry_chase(&plan).await {
            Ok(e) => e,
            Err(()) => {
                self.release_submitted(&mid);
                return; // abort_kill already reported + freed the trade slot
            }
        };

        if entry.qty <= EPS {
            // No fill → release the gate immediately (no needless 1s freeze).
            self.report(&plan.trade_id, &inst, "Abandoned", "no entry fill");
            self.emit_trade(&plan, TradeOutcome::Abandoned, &entry, &LegSummary::default(), 0, 0.0);
            self.release_submitted(&mid);
            self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            return;
        }

        // ── Filled → transit state; hand off to the reconciler ──
        self.emit_position(&inst);
        self.report(
            &plan.trade_id,
            &inst,
            "Holding",
            &format!("qty={:.2} @ {:.3} (exit: reconciler)", entry.qty, entry.vwap),
        );
        // TradeRecord at entry: PendingResolution = "open — exit decoupled".
        self.emit_trade(&plan, TradeOutcome::PendingResolution, &entry, &LegSummary::default(), 0, 0.0);
        let fill_ns = if entry.first_ts_ns > 0 { entry.first_ts_ns } else { now_ns() };
        let need_spawn = {
            let mut g = self.slots.lock().unwrap();
            let s = g.get_mut(&mid).expect("slot claimed above");
            s.state = TradeState::Filled { fill_ns };
            s.net_est += dir_sign * entry.qty;
            // Each fill extends the hold for the whole (merged) position.
            s.hold_until_ns = s.hold_until_ns.max(fill_ns + plan.hold_ms as i64 * MS);
            let need = !s.reconciler_alive;
            if need {
                s.reconciler_alive = true;
            }
            need
        };
        if need_spawn {
            tokio::spawn(exit_reconciler(
                mid,
                self.slots.clone(),
                self.venue.clone(),
                self.mirror.clone(),
                self.fill_tx.clone(),
                rc,
            ));
        }
        self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
    }

    /// Release a Submitted claim without a fill: back to Rest if the market still
    /// holds something (the reconciler resumes), else Idle.
    fn release_submitted(&mut self, mid: &str) {
        let mut g = self.slots.lock().unwrap();
        if let Some(s) = g.get_mut(mid) {
            s.state = if s.net_est.abs() > EPS { TradeState::Rest } else { TradeState::Idle };
        }
    }

    /// Apply a translated close fill from a reconciler (or the entry path's
    /// pre-entry cancel): a fee-free SELL of the held token at `1 − close_px`.
    /// Clamped to held — the venue's authoritative position drives order sizing,
    /// so a clamp here is an accounting-race artifact, not a phantom trade.
    fn on_close_fill(&mut self, f: &Fill) {
        let Some(clamped) = clamp_liquidation(self.pm.qty(&f.instrument), f) else {
            self.report("close", &f.instrument, "Skip", "close fill on already-flat PM (accounting)");
            return;
        };
        match self.pm.apply_fill(&clamped) {
            Ok(()) => {
                let held = self.pm.qty(&f.instrument);
                self.report(
                    "close",
                    &f.instrument,
                    "Exiting",
                    &format!("close fill {:.0} @ {:.3} (held {:.0})", clamped.qty, clamped.px, held),
                );
                if held <= EPS {
                    let realized =
                        self.pm.position(&f.instrument).map(|p| p.realized_pnl).unwrap_or(0.0);
                    tracing::info!(
                        target: "executor",
                        "*** CLOSE {} flat *** realized_total={realized:.4}",
                        f.instrument
                    );
                    self.report("close", &f.instrument, "Closed", &format!("flat; realized_total={realized:.4}"));
                }
                self.emit_position(&f.instrument);
            }
            Err(e) => {
                self.report("close", &f.instrument, "Halt", &format!("close fill: {e}"));
                self.risk.trip("close fill long-only violation");
            }
        }
    }

    /// Apply fills to the PM and accumulate the leg summary. Errs (→ kill) on a
    /// long-only violation.
    fn ingest_fills(&mut self, fills: &[Fill], leg: &mut LegSummary) -> anyhow::Result<()> {
        for f in fills {
            self.pm.apply_fill(f)?;
            let prev_notional = leg.vwap * leg.qty;
            leg.qty += f.qty;
            leg.vwap = if leg.qty > 0.0 { (prev_notional + f.qty * f.px) / leg.qty } else { 0.0 };
            leg.fees += f.fee;
            leg.n_fills += 1;
            if leg.first_ts_ns == 0 {
                leg.first_ts_ns = f.ts_ns;
            }
            leg.last_ts_ns = f.ts_ns;
        }
        Ok(())
    }

    fn abort_kill(&mut self, plan: &TradePlan, inst: &str) {
        self.risk.trip("long-only invariant violation in fill application");
        self.report(&plan.trade_id, inst, "Halt", "long-only violation — see logs");
        self.emit_position(inst);
        self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
    }
}

/// Resolve the UP instrument for meta lookup (catalog keys meta on the UP token).
fn plan_instrument(s: &TradeSignal) -> String {
    s.target.clone()
}

async fn sleep_until(ts_ns: i64) {
    let dur = (ts_ns - now_ns()).max(0) as u64;
    tokio::time::sleep(Duration::from_nanos(dur)).await;
}

/// Cancel a resting close and account its fills authoritatively (`reduced_by`
/// from the engine-synchronous cancel response; `Gone` resolved via the status
/// endpoint after the read replica settles). Returns Ok(filled) ⇒ the order is
/// confirmed OFF the book (caller may drop its reference); Err(()) ⇒ transport
/// failure, order state UNKNOWN — the caller MUST keep the reference and retry.
/// Dropping an unconfirmed reference is the incident's orphan bug.
async fn cancel_and_account(
    venue: &Arc<dyn TradingVenue>,
    r: &RestingClose,
    trade_id: &str,
    seq: &mut u64,
    fills: &FillSender,
    settle_wait: Duration,
) -> Result<f64, ()> {
    let filled = match venue.cancel_order(&r.order_id).await {
        Ok(CancelOutcome::Canceled { reduced_by }) => (r.count - reduced_by).max(0.0),
        Ok(CancelOutcome::Gone) => {
            // Terminal (fully filled or already canceled). Give the query replica
            // its ~150ms, then read the definitive fill count.
            tokio::time::sleep(settle_wait).await;
            match venue.order_fill_count(&r.order_id).await {
                Ok(Some(fc)) => fc.min(r.count),
                Ok(None) => {
                    tracing::warn!(
                        target: "exit",
                        "[{trade_id}] close {} terminal but not visible; fills unknown (position poll heals sizing)",
                        r.order_id
                    );
                    0.0
                }
                Err(e) => {
                    tracing::warn!(target: "exit", "[{trade_id}] close {} status: {e}", r.order_id);
                    0.0
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                target: "exit",
                "[{trade_id}] cancel {} FAILED: {e} — order state unknown, keeping reference",
                r.order_id
            );
            return Err(());
        }
    };
    if filled > EPS {
        *seq += 1;
        let held_inst = complement(&r.instrument);
        let _ = fills.send(translated_close_fill(&held_inst, &r.order_id, *seq, filled, r.px));
        tracing::info!(
            target: "exit",
            "[{trade_id}] close fill {filled:.0} @ {:.3} (cancel accounting)",
            r.px
        );
    }
    Ok(filled)
}

/// Per-market exit reconciler (exit.mode = "reconcile") — the independent loop
/// that owns position truth and all close orders for one market:
///
///   1. state gate: frozen during Submitted/Filled (promotes Filled→Rest after
///      `transit_ms` > the venue's read-replication lag, so the first post-gate
///      poll reliably includes the entry's fill; force-promotes a stuck Submitted)
///   2. AUTHORITATIVE position poll (venue records, never internal accounting)
///   3. flat → cancel any resting close, park (Rest→Idle, task ends)
///   4. past the hold → desired close = BUY |net| of the complement at its best
///      bid (post-only; joins the held side's away queue — never crosses)
///   5. resting order matches desired → leave it; else cancel (fills learned
///      exactly from `reduced_by`) and RE-POLL before re-posting — the
///      anti-oversell recheck: a fill during the cancel changes the size
///
/// Maker-only by design: no taker fallback. If the close never fills the
/// position rides to settlement (accepted trade-off; cap is 1 net contract).
async fn exit_reconciler(
    market_id: String,
    slots: Slots,
    venue: Arc<dyn TradingVenue>,
    mirror: Arc<Mutex<ExecBookMirror>>,
    fills: FillSender,
    rc: ReconCfg,
) {
    tracing::info!(target: "exit", "[{market_id}] reconciler up");
    let mut close_seq: u64 = 0;
    loop {
        tokio::time::sleep(rc.cadence).await;
        let now = now_ns();

        // ── 1. state gate (promote the transitions this task owns) ──
        let (params, expiry_ns, trade_id, hold_until, resting) = {
            let mut g = slots.lock().unwrap();
            let Some(s) = g.get_mut(&market_id) else { break };
            match s.state {
                TradeState::Submitted { since_ns } => {
                    if now - since_ns > rc.submitted_timeout_ns {
                        tracing::warn!(
                            target: "exit",
                            "[{market_id}] Submitted stuck {}ms — force → Rest (healing from venue position)",
                            (now - since_ns) / MS
                        );
                        s.state = TradeState::Rest;
                    } else {
                        continue; // frozen: entry in flight
                    }
                }
                TradeState::Filled { fill_ns } => {
                    if now - fill_ns >= rc.transit_ns {
                        s.state = TradeState::Rest;
                    } else {
                        continue; // frozen: transit window
                    }
                }
                TradeState::Idle | TradeState::Rest => {}
            }
            (s.params, s.expiry_ns, s.trade_id.clone(), s.hold_until_ns, s.resting.clone())
        };

        // ── 2. authoritative position ──
        let net = match venue.market_position(&market_id).await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(target: "exit", "[{market_id}] position poll: {e}");
                continue;
            }
        };
        if let Some(s) = slots.lock().unwrap().get_mut(&market_id) {
            s.net_est = net;
        }

        // ── 3. flat → clean up + park ──
        if net.abs() < EPS {
            if let Some(r) = &resting {
                if cancel_and_account(&venue, r, &trade_id, &mut close_seq, &fills, rc.settle_wait)
                    .await
                    .is_err()
                {
                    continue; // cancel unconfirmed: keep the reference, retry next cycle
                }
                if let Some(s) = slots.lock().unwrap().get_mut(&market_id) {
                    s.resting = None;
                }
            }
            let mut g = slots.lock().unwrap();
            if let Some(s) = g.get_mut(&market_id) {
                // Re-check under the lock: an entry may have claimed the slot
                // while we were polling — if so, stay alive and frozen.
                if matches!(s.state, TradeState::Rest | TradeState::Idle) && s.resting.is_none() {
                    s.state = TradeState::Idle;
                    s.reconciler_alive = false;
                    tracing::info!(target: "exit", "[{market_id}] flat → Idle; reconciler parked");
                    break;
                }
            }
            continue;
        }

        // ── 4. hold: capture the reprice before starting to close ──
        if now < hold_until {
            continue;
        }

        // Desired close: BUY the complement of the held side at its best bid.
        let inst_yes = {
            let g = slots.lock().unwrap();
            match g.get(&market_id) {
                Some(s) => s.inst_yes.clone(),
                None => break,
            }
        };
        let held_inst = if net > 0.0 { inst_yes.clone() } else { complement(&inst_yes) };
        let close_inst = complement(&held_inst);
        let Some(top) = mirror.lock().unwrap().top(&close_inst) else { continue };
        if top.best_bid <= 0.0 {
            continue;
        }
        let want_px = top.best_bid;
        let want_count = net.abs().floor();
        if want_count < 1.0 {
            continue; // sub-contract residual can't be closed by order; settles
        }

        // ── 5. compare with the resting order ──
        if let Some(r) = &resting {
            if r.instrument == close_inst
                && (r.px - want_px).abs() < EPS
                && (r.count - want_count).abs() < EPS
            {
                continue; // matches desired: leave it (queue priority preserved)
            }
            // Stale: cancel (learning exact fills from reduced_by), then RE-POLL
            // before re-posting — the anti-oversell recheck. Recompute next cycle.
            if cancel_and_account(&venue, r, &trade_id, &mut close_seq, &fills, rc.settle_wait)
                .await
                .is_ok()
            {
                if let Some(s) = slots.lock().unwrap().get_mut(&market_id) {
                    s.resting = None;
                }
            }
            continue;
        }

        // No resting order → post the desired close. Gate re-check first (an
        // entry may have claimed the slot during our awaits).
        {
            let g = slots.lock().unwrap();
            match g.get(&market_id) {
                Some(s) if matches!(s.state, TradeState::Rest) => {}
                _ => continue,
            }
        }
        close_seq += 1;
        let intent = OrderIntent {
            client_id: format!("close:{trade_id}:{close_seq}"),
            instrument: close_inst.clone(),
            token_id: String::new(),
            side: Side::Buy,
            price: want_px,
            size: want_count,
            kind: IntentKind::RestUntil { expiry_ns },
            params,
            expiry_ns,
        };
        match venue.place_resting(&intent).await {
            Ok(order_id) => {
                let placed = RestingClose {
                    order_id: order_id.clone(),
                    instrument: close_inst.clone(),
                    px: want_px,
                    count: want_count,
                };
                // Record under the lock ONLY if still in Rest; if an entry raced
                // us while the POST was in flight, it couldn't have seen this
                // order — roll it back ourselves (never leave an untracked order).
                let entry_raced = {
                    let mut g = slots.lock().unwrap();
                    match g.get_mut(&market_id) {
                        Some(s) if matches!(s.state, TradeState::Rest) => {
                            s.resting = Some(placed.clone());
                            false
                        }
                        _ => true,
                    }
                };
                if entry_raced {
                    tracing::warn!(
                        target: "exit",
                        "[{market_id}] entry raced the close POST — rolling back {order_id}"
                    );
                    // Best-effort: on transport failure the next reconcile cycle
                    // re-discovers via the authoritative position + re-cancel.
                    let _ = cancel_and_account(
                        &venue, &placed, &trade_id, &mut close_seq, &fills, rc.settle_wait,
                    )
                    .await;
                } else {
                    tracing::info!(
                        target: "exit",
                        "[{market_id}] close rest: buy {want_count:.0} {close_inst} @ {want_px:.3} (net {net:+.0})"
                    );
                }
            }
            Err(e) => tracing::warn!(target: "exit", "[{market_id}] close post: {e}"),
        }
    }
}

/// Hold-period study sampler. For each registered entry, at every hold offset it
/// computes the **realized** exit return — sweeps the bid book for the held size
/// (slippage) and charges the taker fee on both legs — and logs it. One run
/// yields the full return-vs-hold curve without touching the real exit.
async fn hold_sampler(
    mirror: Arc<Mutex<ExecBookMirror>>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<HoldProbe>,
    ladder: Vec<u64>,
) {
    let mut probes: Vec<(HoldProbe, Vec<u64>)> = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    loop {
        tokio::select! {
            p = rx.recv() => match p {
                Some(p) => probes.push((p, ladder.clone())),
                None => break,
            },
            _ = tick.tick() => {
                let now = now_ns();
                for (probe, pending) in probes.iter_mut() {
                    let (t0, qty, entry, fee) = (probe.t0_ns, probe.qty, probe.entry_vwap, probe.fee_rate);
                    let (inst, tid) = (probe.inst.clone(), probe.trade_id.clone());
                    pending.retain(|&h| {
                        if now < t0 + h as i64 * MS {
                            return true; // offset not reached yet
                        }
                        let bids = mirror.lock().unwrap().depth(&inst).map(|d| d.bids).unwrap_or_default();
                        // Realized exit: market-sell the held size into the bids.
                        let (filled, exit_vwap) = crate::venue::sweep_sell(&bids, qty, 0.0);
                        if filled > 1e-6 {
                            let fee_ex = fee * exit_vwap * (1.0 - exit_vwap); // per-share, exit leg
                            let fee_en = fee * entry * (1.0 - entry);         // per-share, entry leg
                            let gross_c = (exit_vwap - entry) * 100.0;
                            let net_c = ((exit_vwap - entry) - fee_ex - fee_en) * 100.0;
                            tracing::info!(
                                target: "hold",
                                "HOLD trade={tid} inst={inst} h={h} qty={qty:.2} filled={filled:.2} \
                                 entry={entry:.3} exit={exit_vwap:.3} gross_c={gross_c:.2} net_c={net_c:.2}",
                            );
                        }
                        false // offset sampled → drop it
                    });
                }
                probes.retain(|(_, pending)| !pending.is_empty());
            }
        }
    }
}

/// Post-signal price-trajectory sampler. For each triggered trade, at every
/// offset in the ladder it reads the traded-token top book and logs bid/ask/mid
/// plus the ¢ delta from the signal reference — the reprice path, independent of
/// whether our order filled. This is the ground-truth "was the signal right?"
/// curve: if the mid walks our way over the next ~100-300ms the signal is real
/// even when the order abandons.
async fn price_sampler(
    mirror: Arc<Mutex<ExecBookMirror>>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<PxProbe>,
    ladder: Vec<u64>,
    step_ms: u64,
) {
    let window_ms = *ladder.last().unwrap_or(&1000) as i64;
    let vol_delay_ms: i64 = 5000; // wait for the ~3s REST trade poll + latency to catch up
    let mut probes: Vec<(PxProbe, Vec<u64>)> = Vec::new();
    let mut vol_q: Vec<(PxProbe, i64)> = Vec::new(); // completed probes awaiting the delayed volume pass
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    loop {
        tokio::select! {
            p = rx.recv() => match p {
                Some(p) => probes.push((p, ladder.clone())),
                None => break,
            },
            _ = tick.tick() => {
                let now = now_ns();
                // ── price + depth pass (real-time) ──
                for (probe, pending) in probes.iter_mut() {
                    pending.retain(|&h| {
                        if now < probe.t0_ns + h as i64 * MS {
                            return true; // offset not reached yet
                        }
                        // Top + ladder depth under one lock. Depth tells an MM reprice
                        // (levels shift, depth replenished) from a sweep (depth eaten).
                        let snap = {
                            let m = mirror.lock().unwrap();
                            m.top(&probe.inst).map(|b| {
                                let (bdep, adep, nb, na) = match m.depth(&probe.inst) {
                                    Some(d) => (
                                        d.bids.iter().map(|&(_, s)| s).sum::<f64>(),
                                        d.asks.iter().map(|&(_, s)| s).sum::<f64>(),
                                        d.bids.len(),
                                        d.asks.len(),
                                    ),
                                    None => (0.0, 0.0, 0, 0),
                                };
                                (b, bdep, adep, nb, na)
                            })
                        };
                        if let Some((b, bdep, adep, nb, na)) = snap {
                            let mid = 0.5 * (b.best_bid + b.best_ask);
                            tracing::info!(
                                target: "pxprobe",
                                "PXPROBE trade={} inst={} t_ms={h} bid={:.3} ask={:.3} mid={:.3} \
                                 dmid_c={:.2} dask_c={:.2} bsz={:.0} asz={:.0} bdep={:.0}({}) adep={:.0}({})",
                                probe.trade_id, probe.inst, b.best_bid, b.best_ask, mid,
                                (mid - probe.ref_mid) * 100.0, (b.best_ask - probe.ref_ask) * 100.0,
                                b.bid_sz, b.ask_sz, bdep, nb, adep, na,
                            );
                        }
                        false // sampled → drop this offset
                    });
                }
                // completed price-passes → queue for the delayed volume pass
                let mut i = 0;
                while i < probes.len() {
                    if probes[i].1.is_empty() {
                        let (probe, _) = probes.remove(i);
                        let fire = probe.t0_ns + (window_ms + vol_delay_ms) * MS;
                        vol_q.push((probe, fire));
                    } else {
                        i += 1;
                    }
                }
                // ── delayed volume pass: trades are REST-polled ~1-3s late, so wait,
                // then attribute by exchange time. tvol>0 at a depth drop = a sweep;
                // tvol~0 = an MM cancel-and-reprice. One PXVOL line per offset. ──
                vol_q.retain(|(probe, fire)| {
                    if now < *fire {
                        return true;
                    }
                    let m = mirror.lock().unwrap();
                    for &h in ladder.iter() {
                        let lo = probe.t0_ns + (h as i64 - step_ms as i64) * MS;
                        let hi = probe.t0_ns + h as i64 * MS;
                        let tvol = m.traded_between(&probe.inst, lo, hi);
                        tracing::info!(target: "pxprobe", "PXVOL trade={} t_ms={h} tvol={:.0}", probe.trade_id, tvol);
                    }
                    false // logged → drop
                });
            }
        }
    }
}

/// Maker-exit study sampler. At each exit it shadow-resolves a Post-Only maker
/// SELL (post at `offer_px`, rest `rest_ms`) and compares it to the taker cross
/// at the exit-time bid. Outcomes:
///   - reject      : bid already ≥ offer at post-time → Post-Only would cross →
///                   rejected → fall through to a taker cross (at that bid).
///   - maker_fill  : the bid lifts to the offer while resting → filled as MAKER,
///                   fee-free, at the ask (the win).
///   - timeout     : rest elapsed unfilled → cancel-and-cross as a taker.
/// `gain_c` = maker net price − taker net price (¢/share); >0 means maker beat taker.
/// Clamp a liquidation fill to the held qty — the near-expiry liquidator can only
/// sell remaining inventory. Returns `None` when already flat (the normal exit
/// beat the liquidator: a benign race, skip it — never trip the kill switch).
/// Otherwise returns the fill with `qty`/`fee` scaled down to the residual.
fn clamp_liquidation(held: f64, f: &Fill) -> Option<Fill> {
    if held <= EPS {
        return None;
    }
    let qty = f.qty.min(held);
    Some(Fill {
        qty,
        fee: if f.qty > EPS { f.fee * qty / f.qty } else { 0.0 },
        ..f.clone()
    })
}

/// Resolve a maker-exit outcome: `offer_px` the resting offer, `bid0` the bid at
/// post-time, `bid_now` the current bid, `expired` whether the rest elapsed.
/// Returns (outcome, cross_px); "rest" means keep waiting (no early bail — the
/// offer simply rests until it fills or the timer expires). The reject branch is
/// the Post-Only "would cross on arrival" case (bid0 already ≥ offer).
fn maker_outcome(offer_px: f64, bid0: f64, bid_now: f64, expired: bool) -> (&'static str, f64) {
    if bid0 >= offer_px - 1e-9 {
        ("reject", bid0) // would cross at post-time → Post-Only rejected → cross
    } else if bid_now >= offer_px - 1e-9 {
        ("maker_fill", offer_px) // bid lifted to the offer while resting
    } else if expired {
        ("timeout", bid_now) // rest elapsed unfilled → cancel-and-cross
    } else {
        ("rest", 0.0)
    }
}

async fn maker_sampler(
    mirror: Arc<Mutex<ExecBookMirror>>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<MakerProbe>,
    cfg: crate::config::MakerProbeCfg,
) {
    fn emit(p: &MakerProbe, outcome: &str, cross_px: f64) {
        let fee = |px: f64| p.fee_rate * px * (1.0 - px);
        // maker fill is fee-free at the offer; any cross fallback pays the taker fee.
        let maker_net = if outcome == "maker_fill" { p.offer_px } else { cross_px - fee(cross_px) };
        let taker_net = p.bid0 - fee(p.bid0);
        let gain_c = (maker_net - taker_net) * 100.0;
        tracing::info!(
            target: "maker",
            "MAKER trade={} inst={} outcome={outcome} offer={:.3} bid0={:.3} cross={:.3} \
             maker_net={:.4} taker_net={:.4} gain_c={:.2}",
            p.trade_id, p.inst, p.offer_px, p.bid0, cross_px, maker_net, taker_net, gain_c,
        );
    }

    let mut active: Vec<(MakerProbe, i64)> = Vec::new(); // (probe, deadline_ns)
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    loop {
        tokio::select! {
            p = rx.recv() => match p {
                None => break,
                Some(p) => {
                    // At post-time bid_now == bid0; the only non-rest outcome is reject.
                    let (outcome, cross) = maker_outcome(p.offer_px, p.bid0, p.bid0, false);
                    if outcome == "rest" {
                        let deadline = p.t0_ns + cfg.rest_ms as i64 * MS;
                        active.push((p, deadline));
                    } else {
                        emit(&p, outcome, cross);
                    }
                }
            },
            _ = tick.tick() => {
                let now = now_ns();
                active.retain(|(p, deadline)| {
                    let bid = mirror.lock().unwrap().top(&p.inst).map(|b| b.best_bid).unwrap_or(0.0);
                    let (outcome, cross) = maker_outcome(p.offer_px, p.bid0, bid, now >= *deadline);
                    if outcome == "rest" {
                        true
                    } else {
                        emit(p, outcome, cross);
                        false
                    }
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        clamp_liquidation, entry_gate, exposure_ok, maker_outcome, translated_close_fill,
        yes_side_of, ReconCfg, TradeState,
    };
    use std::time::Duration;

    fn rc() -> ReconCfg {
        ReconCfg {
            cadence: Duration::from_millis(200),
            transit_ns: 1_000 * super::MS,
            submitted_timeout_ns: 3_000 * super::MS,
            settle_wait: Duration::from_millis(150),
        }
    }

    #[test]
    fn entry_gate_freezes_in_flight_and_transit() {
        let r = rc();
        let now = 10_000 * super::MS;
        // in-flight entry blocks
        assert!(entry_gate(TradeState::Submitted { since_ns: now - 100 * super::MS }, now, &r).is_err());
        // stale Submitted (entry task died) falls through — no deadlock
        assert!(entry_gate(TradeState::Submitted { since_ns: now - 5_000 * super::MS }, now, &r).is_ok());
        // inside the 1s Filled→Rest transit blocks
        assert!(entry_gate(TradeState::Filled { fill_ns: now - 500 * super::MS }, now, &r).is_err());
        // past the transit allows
        assert!(entry_gate(TradeState::Filled { fill_ns: now - 1_500 * super::MS }, now, &r).is_ok());
        assert!(entry_gate(TradeState::Rest, now, &r).is_ok());
        assert!(entry_gate(TradeState::Idle, now, &r).is_ok());
    }

    #[test]
    fn exposure_cap_blocks_growth_allows_reduction() {
        // fresh entry within the cap
        assert!(exposure_ok(0.0, 1.0, 1.0, 1.0));
        // stacking past the cap blocked
        assert!(!exposure_ok(1.0, 1.0, 1.0, 1.0));
        // opposite-direction entry that REDUCES |net| always allowed (nets to 0)
        assert!(exposure_ok(1.0, -1.0, 1.0, 1.0));
        // reducing from beyond the cap is allowed too (recovery)
        assert!(exposure_ok(-2.0, 1.0, 1.0, 1.0));
        // overshooting through zero past the cap blocked
        assert!(!exposure_ok(1.0, -1.0, 3.0, 1.0));
    }

    #[test]
    fn close_fill_translates_to_held_side_sell() {
        // close = buy NO at 0.43 to exit long YES ⇒ sell YES at 0.57, fee-free.
        let f = translated_close_fill("kalshi.KXBTC15M-1.YES", "ord-1", 2, 1.0, 0.43);
        assert_eq!(f.instrument, "kalshi.KXBTC15M-1.YES");
        assert!(matches!(f.side, arb_core::model::Side::Sell));
        assert!((f.px - 0.57).abs() < 1e-9);
        assert_eq!(f.fee, 0.0);
        assert!(f.client_id.starts_with("close:"), "routes via the close: prefix");
        assert_eq!(f.venue_trade_id, "ord-1-cl2", "unique per accounting event");
    }

    #[test]
    fn yes_side_resolves_from_either_token() {
        assert_eq!(yes_side_of("kalshi.KXBTC15M-1.YES"), "kalshi.KXBTC15M-1.YES");
        assert_eq!(yes_side_of("kalshi.KXBTC15M-1.NO"), "kalshi.KXBTC15M-1.YES");
        assert_eq!(yes_side_of("polymarket.0xabc.DOWN"), "polymarket.0xabc.UP");
    }
    use crate::types::{Fill, FillStatus};
    use arb_core::model::Side;

    fn liq_fill(qty: f64, fee: f64) -> Fill {
        Fill {
            venue_trade_id: "sim-liq-1".into(),
            order_id: "sim-liq-1".into(),
            client_id: "liq:x".into(),
            instrument: "polymarket.0xabc.UP".into(),
            status: FillStatus::Confirmed,
            side: Side::Sell,
            qty,
            px: 0.40,
            fee,
            ts_ns: 0,
        }
    }

    #[test]
    fn clamp_liquidation_handles_the_exit_race() {
        // already flat (normal exit won the race) → skip, no fill, no kill
        assert!(clamp_liquidation(0.0, &liq_fill(20.0, 0.34)).is_none());
        // full residual → sells the lot unchanged
        let full = clamp_liquidation(20.0, &liq_fill(20.0, 0.336)).unwrap();
        assert!((full.qty - 20.0).abs() < 1e-9);
        assert!((full.fee - 0.336).abs() < 1e-9);
        // partial residual (venue thinks 20, PM holds 5) → clamp qty + scale fee
        let part = clamp_liquidation(5.0, &liq_fill(20.0, 0.40)).unwrap();
        assert!((part.qty - 5.0).abs() < 1e-9);
        assert!((part.fee - 0.10).abs() < 1e-9, "fee scaled to qty: 0.40*5/20");
    }

    #[test]
    fn maker_outcome_covers_all_branches() {
        // reject — bid already at/above the offer on arrival (would cross)
        assert_eq!(maker_outcome(0.50, 0.50, 0.50, false).0, "reject");
        assert_eq!(maker_outcome(0.50, 0.51, 0.51, false).0, "reject");
        // rest — bid below offer, not expired → keep waiting (no early bail)
        assert_eq!(maker_outcome(0.50, 0.48, 0.48, false).0, "rest");
        // a bid drop while resting is NOT a bail anymore — still rests
        assert_eq!(maker_outcome(0.50, 0.48, 0.45, false).0, "rest");
        // maker_fill — bid lifts to the offer while resting (fee-free at offer)
        let (o, px) = maker_outcome(0.50, 0.48, 0.50, false);
        assert_eq!(o, "maker_fill");
        assert!((px - 0.50).abs() < 1e-9);
        // timeout — rest elapsed, no fill → cross at the then-bid (whatever it is)
        let (o, px) = maker_outcome(0.50, 0.48, 0.45, true);
        assert_eq!(o, "timeout");
        assert!((px - 0.45).abs() < 1e-9);
    }
}
