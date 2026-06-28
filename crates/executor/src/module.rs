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
    BookSource, BookTop, Fill, IntentKind, MarketParams, OrderIntent, TradePlan, VenueOutcome, MS,
};
use crate::venue::{SimVenue, TradingVenue};
use crate::venue_spec::{self, traded_instrument};

const EPS: f64 = 1e-6;

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
        let (fill_tx, fill_rx) = tokio::sync::mpsc::unbounded_channel();
        self.handles.extend(venue.start(fill_tx));

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
            self.handles.push(tokio::spawn(price_sampler(m, rx, ladder)));
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
                    // Async fills from the venue (liquidations, and live fills later).
                    fill = fill_rx.recv() => {
                        let Some(f) = fill else { break };
                        engine.on_liquidation_fill(&f);
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
        // One-trade gate (§8.4) — the structural one-position guarantee.
        if !self.pm.try_begin_trade(&trade_id, now) {
            self.report(&trade_id, &instrument, "Rejected", "active trade / cooldown");
            return;
        }
        self.risk.note_trade(now);

        let expiry_ns =
            self.mirror.lock().unwrap().meta_of(&plan_instrument(s)).map(|m| m.expiry_ns).unwrap_or(now);
        let plan = TradePlan {
            trade_id,
            instrument,
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

        // Post-signal price-trajectory probe: fire BEFORE the entry loop so it
        // captures the reprice path even if we abandon (no fill). Anchored at the
        // signal moment; deltas are ¢ vs the signal book.
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

        // Shadow / dry-run: the probe has fired; place NO order, free the
        // one-trade slot + arm the cooldown, and we're done. No real money.
        if self.cfg.dry_run {
            self.report(&plan.trade_id, &inst, "Shadow", "dry_run: probe only, no order");
            self.pm.end_trade(now_ns(), self.cfg.risk.cooldown_ms);
            return;
        }

        // ── ENTRY: bounded FAK attempt loop (§5) ──
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
            let intent = self.intent(&plan, "E", attempt, Side::Buy, cap, remaining);
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
                            return self.abort_kill(&plan, &inst);
                        }
                        break; // partial or full → proceed to HOLDING
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
        let exit_due = entry.first_ts_ns + plan.hold_ms as i64 * MS;
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
        let mut exit_ref_bid = 0.0;
        let mut k = 0u32;
        loop {
            if self.pm.qty(&inst) <= EPS {
                break;
            }
            if now_ns() >= plan.exit_deadline_ns {
                break; // → PendingResolution
            }
            let held = self.pm.qty(&inst);
            let Some(book) = self.top(&inst) else { break };
            if k == 0 {
                exit_ref_bid = book.best_bid;
            }
            let floor = (book.best_bid - k as f64 * self.cfg.exit.step_c)
                .max(book.best_bid - self.cfg.exit.max_slip_c)
                .max(plan.params.tick_size);
            let intent = self.intent(&plan, "X", k, Side::Sell, floor, held);
            let venue = self.venue.clone();
            match venue.submit(&intent).await {
                VenueOutcome::Acked { fills, .. } => {
                    if !fills.is_empty() {
                        if self.ingest_fills(&fills, &mut exit).is_err() {
                            return self.abort_kill(&plan, &inst);
                        }
                    }
                }
                VenueOutcome::Rejected(r) => {
                    self.report(&plan.trade_id, &inst, "Reject", &r);
                }
            }
            k += 1;
            if k >= self.cfg.exit.max_attempts {
                break;
            }
            tokio::time::sleep(Duration::from_millis(self.cfg.exit.retry_interval_ms)).await;
        }

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
) {
    let mut probes: Vec<(PxProbe, Vec<u64>)> = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_millis(10));
    loop {
        tokio::select! {
            p = rx.recv() => match p {
                Some(p) => probes.push((p, ladder.clone())),
                None => break,
            },
            _ = tick.tick() => {
                let now = now_ns();
                for (probe, pending) in probes.iter_mut() {
                    pending.retain(|&h| {
                        if now < probe.t0_ns + h as i64 * MS {
                            return true; // offset not reached yet
                        }
                        if let Some(b) = mirror.lock().unwrap().top(&probe.inst) {
                            let mid = 0.5 * (b.best_bid + b.best_ask);
                            tracing::info!(
                                target: "pxprobe",
                                "PXPROBE trade={} inst={} t_ms={h} bid={:.3} ask={:.3} mid={:.3} \
                                 dmid_c={:.2} dask_c={:.2}",
                                probe.trade_id, probe.inst, b.best_bid, b.best_ask, mid,
                                (mid - probe.ref_mid) * 100.0, (b.best_ask - probe.ref_ask) * 100.0,
                            );
                        }
                        false // sampled → drop this offset
                    });
                }
                probes.retain(|(_, pending)| !pending.is_empty());
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
    use super::{clamp_liquidation, maker_outcome};
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
