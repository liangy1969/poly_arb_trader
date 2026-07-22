//! Replay-parity harness (DESIGN_FAIR_RIDE §7 gate 3): stream online-collected
//! 50ms sampler rows through the REAL FairRide pipeline (CalibCore →
//! MarketState → FairRideRule) and dump signals + per-row fair values for
//! equivalence checks against the PyTorch sim.
//!
//! Mode is auto-detected from the surface:
//!   - cb (0 extras): price reference = coinbase.BTC; feed coinbase + YES.
//!   - px2imb (extras): price reference = the perp (also supplies imb1 sizes);
//!     basis reference = coinbase.BTC; feed perp + coinbase + YES. Features
//!     (basis, dbasis15, dbasis60, imb1) are reconstructed inside the pipeline.
//!
//! Usage: replay_fair_ride <samples.csv> <meta_cache.json> <model.json> <out_prefix>
//! (plain CSV — gunzip first). Emits <out_prefix>_signals.csv and
//! <out_prefix>_fair.csv (1s-grid fair series per event, post-first-fit).

use std::collections::HashMap;
use std::io::{BufRead, Write};

use arb_core::event::{Event, Payload};
use arb_core::model::{BookUpdate, MarketMeta, MarketStatus};
use arb_processor::{CalibCfg, CalibCore, FairRideCfg, FairRideRule, FeatureState, MarketState, Rule};

const PERP: &str = "binance.usdt_perp.BTCUSDT";
const CB: &str = "coinbase.BTC";

fn ev(topic: String, ts_ns: i64, seq: u64, payload: Payload) -> Event {
    Event { topic, source: "replay", ts_ns, seq, payload }
}

fn book(inst: &str, ts_ns: i64, seq: u64, bid: f64, bsz: f64, ask: f64, asz: f64) -> Event {
    ev(
        format!("market.{inst}.book"),
        ts_ns,
        seq,
        Payload::Book(BookUpdate {
            instrument: inst.into(),
            bids: vec![(bid, bsz)],
            asks: vec![(ask, asz)],
            update_id: None,
            exch_ts_ns: ts_ns,
            recv_ts_ns: ts_ns,
        }),
    )
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (csv_path, meta_path, model_path, out_prefix) = (&args[1], &args[2], &args[3], &args[4]);

    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
    let bytes = std::fs::read(model_path)?;
    let surface = std::sync::Arc::new(arb_processor::FairSurface::from_json(std::str::from_utf8(&bytes)?)?);
    let hash = arb_processor::calib::fnv1a(&bytes);
    let px2imb = surface.n_extra() > 0;
    let reference = if px2imb { PERP } else { CB };
    eprintln!(
        "mode={} reference={reference} extras={:?} fit_window=120 delta=0.03 tte=300-60",
        if px2imb { "px2imb" } else { "cb" },
        surface.extras
    );

    // Match the frozen analysis config: rolling 120s fit, δ=0.03, entries
    // 300→60s, uncapped (so episode accounting compares trade-by-trade).
    let calib_cfg = CalibCfg {
        enabled: true,
        model_path: model_path.clone(),
        reference: reference.into(),
        basis_reference: CB.into(),
        fit_window_s: 120.0,
        first_tte_s: 300.0,
        last_tte_s: 60.0,
        ..Default::default()
    };
    let mut core = CalibCore::new(calib_cfg, surface.clone(), hash);
    let ride_cfg = FairRideCfg {
        reference: reference.into(),
        basis_reference: CB.into(),
        delta: 0.03,
        entry_min_tte_s: 60.0,
        entry_max_tte_s: 300.0,
        max_entries_per_event: 255,
        ..Default::default()
    };
    let mut rule = FairRideRule::new(ride_cfg.clone(), surface.clone(), hash);
    let mut state = MarketState::new(reference.into(), "15m_updown".into(), 256, 10_000);
    // Local mirror of the pipeline's feature reconstruction, only for the fair
    // diagnostic series (the rule/core own their own copies internally).
    let mut feat_diag = FeatureState::new(surface.extras.clone());

    let f = std::io::BufReader::new(std::fs::File::open(csv_path)?);
    let mut lines = f.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty csv"))??
        .split(',')
        .map(str::to_string)
        .collect();
    let col: HashMap<&str, usize> = header.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
    let idx = |name: &str| -> usize { *col.get(name).unwrap_or_else(|| panic!("missing col {name}")) };
    let (i_ts, i_tick, i_tte) = (idx("ts_ms"), idx("ticker"), idx("tte_ms"));
    let (i_ybid, i_yask) = (idx("ybid"), idx("yask"));
    let (i_cbb, i_cba, i_cbage) = (idx("cb_bid"), idx("cb_ask"), idx("cb_age_ms"));
    let (i_pb, i_pa) = (idx("perp_bid"), idx("perp_ask"));
    let (i_pbs, i_pas) = (idx("perp_bid_sz"), idx("perp_ask_sz"));
    let i_page = *col.get("perp_age_ms").unwrap_or(&usize::MAX);

    let mut sigs = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}_signals.csv"))?);
    writeln!(sigs, "ts_ms,target,direction,yes_price,reason")?;
    let mut fairs = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}_fair.csv"))?);
    writeln!(fairs, "ts_ms,ticker,tte_s,fair,mid,d_b,d_rho,fit_seq")?;

    let mut seq = 0u64;
    let mut known: HashMap<String, (f64, i64)> = HashMap::new(); // ticker -> (strike, expiry_ns)
    let mut calibs: HashMap<String, arb_core::model::CalibUpdate> = HashMap::new();
    let mut last_cb_recv_ns: i64 = i64::MIN;
    let mut last_perp_feed_ms: i64 = i64::MIN;
    let mut last_perp: (f64, f64) = (0.0, 0.0);
    let mut n_cb_feeds: u64 = 0;
    let mut last_cb_feed_ts: i64 = 0;
    let debug = std::env::var("REPLAY_DEBUG").is_ok();
    let mut perp_mid = f64::NAN;
    let mut last_fair_ms: HashMap<String, i64> = HashMap::new();
    let mut n_rows = 0u64;
    let mut n_sigs = 0u64;

    macro_rules! feed {
        ($e:expr) => {{
            let e = $e;
            state.on_event(&e);
            for u in core.on_event(&e) {
                push_calib(&mut state, &mut rule, &mut calibs, u, &mut seq, e.ts_ns);
            }
            for s in rule.on_event(&e, &state) {
                n_sigs += 1;
                writeln!(sigs, "{},{},{},{:.3},\"{}\"", e.ts_ns / 1_000_000, s.target, s.direction, s.trigger.yes_price, s.reason)?;
            }
        }};
    }

    for line in lines {
        let line = line?;
        let f: Vec<&str> = line.split(',').collect();
        let ts_ms: i64 = f[i_ts].parse()?;
        let ts_ns = ts_ms * 1_000_000;
        let ticker = f[i_tick];
        let tte_ms: i64 = f[i_tte].parse()?;
        if tte_ms <= 0 {
            continue;
        }
        n_rows += 1;

        // one-time meta per ticker (strike from the cache; expiry from tte)
        if !known.contains_key(ticker) {
            let strike = meta.get(ticker).and_then(|m| m.get("strike")).and_then(|v| v.as_f64());
            let Some(strike) = strike else { continue };
            let expiry_ns = ts_ns + tte_ms * 1_000_000;
            known.insert(ticker.to_string(), (strike, expiry_ns));
            let inst = format!("kalshi.{ticker}.YES");
            seq += 1;
            let m = ev(
                "market.kalshi.catalog".into(),
                ts_ns,
                seq,
                Payload::Meta(MarketMeta {
                    instrument: inst,
                    kind: "15m_updown".into(),
                    status: MarketStatus::Live,
                    start_ts_ns: Some(expiry_ns - 900_000_000_000),
                    expiry_ts_ns: Some(expiry_ns),
                    winner: None,
                    min_order_size: None,
                    tick_size: None,
                    fee_rate: None,
                    strike: Some(strike),
                }),
            );
            feed!(m);
        }

        // perp book (px2imb price reference + imb1 sizes). Arrival semantics:
        // deliver at recv (staleness shows as an unchanged value → no event).
        let (pb, pa): (f64, f64) = (f[i_pb].parse().unwrap_or(0.0), f[i_pa].parse().unwrap_or(0.0));
        let (pbs, pas): (f64, f64) = (f[i_pbs].parse().unwrap_or(0.0), f[i_pas].parse().unwrap_or(0.0));
        if px2imb && pb > 0.0 && pa > 0.0 {
            perp_mid = 0.5 * (pb + pa);
            feat_diag.on_perp(ts_ns, perp_mid, pbs, pas);
            // Keep the pipeline's perp reference FRESH regardless of value
            // change. In a flat market the best bid/ask sit unchanged for tens
            // of seconds while binance messages keep arriving; feeding only on
            // value change would let ref_ns go stale (>1.5s) so the calibrator
            // drops most samples -> no fit (the flat-market analog of the
            // coinbase bug). Feed immediately on a real price change (ride-gate
            // responsiveness) AND at least every 250ms (freshness) — the 250ms
            // throttle bounds the reference-book eval cost (it scans all events)
            // while staying well inside the 1.5s stale gate. `perp_age_ms` isn't
            // needed once we heartbeat; kept for reference only.
            let _ = i_page;
            if (pb, pa) != last_perp || ts_ms - last_perp_feed_ms >= 250 {
                last_perp = (pb, pa);
                last_perp_feed_ms = ts_ms;
                seq += 1;
                feed!(book(PERP, ts_ns, seq, pb, pbs, pa, pas));
            }
        }

        // coinbase book: price reference for cb, basis reference for px2imb.
        // The ticker channel resets its age on EVERY message (even when bid/ask
        // are unchanged), so feed on a NEW MESSAGE — detected by the implied
        // receive time (ts − age) advancing — not on value change. Otherwise a
        // flat-price stretch would look stale in the pipeline while the sim
        // (per-row cb_age ≤ 5s gate) still uses it. Fresh (age ≤ 5s) only.
        let (cb_b, cb_a): (f64, f64) = (f[i_cbb].parse().unwrap_or(0.0), f[i_cba].parse().unwrap_or(0.0));
        let cb_age_ms: i64 = f[i_cbage].parse().unwrap_or(-1);
        if cb_b > 0.0 && cb_a > 0.0 && (0..=5000).contains(&cb_age_ms) {
            let cb_recv_ns = ts_ns - cb_age_ms * 1_000_000;
            if cb_recv_ns > last_cb_recv_ns {
                last_cb_recv_ns = cb_recv_ns;
                feat_diag.on_cb(ts_ns, 0.5 * (cb_b + cb_a));
                seq += 1;
                n_cb_feeds += 1;
                last_cb_feed_ts = ts_ms;
                feed!(book(CB, ts_ns, seq, cb_b, 0.0, cb_a, 0.0));
            }
        }

        // kalshi YES book
        let (ybid, yask): (f64, f64) = (f[i_ybid].parse().unwrap_or(0.0), f[i_yask].parse().unwrap_or(0.0));
        if ybid > 0.0 && yask > 0.0 {
            let inst = format!("kalshi.{ticker}.YES");
            seq += 1;
            feed!(book(&inst, ts_ns, seq, ybid, 0.0, yask, 0.0));

            // 1s fair diagnostics (post-fit)
            if let Some(u) = calibs.get(&inst) {
                let last = last_fair_ms.entry(inst.clone()).or_insert(0);
                if ts_ms - *last >= 1000 {
                    let (strike, expiry_ns) = known[ticker];
                    let tte_s = (expiry_ns - ts_ns) as f64 / 1e9;
                    let (price, feats_opt) = if px2imb {
                        (perp_mid, feat_diag.feats(ts_ns))
                    } else {
                        (0.5 * (cb_b + cb_a), Some([0.0; arb_processor::MAX_EXTRA]))
                    };
                    if tte_s > 0.0 && price > 0.0 {
                        if let Some(feats) = feats_opt {
                            *last = ts_ms;
                            // two-price surfaces: cb mid as px2 (NAN otherwise)
                            let px2 = if surface.two_price() { 0.5 * (cb_b + cb_a) } else { f64::NAN };
                            let fair = surface.fair(price, px2, tte_s, strike, u.d_b, u.d_rho, &feats);
                            writeln!(
                                fairs,
                                "{},{},{:.3},{:.6},{:.4},{:.6},{:.6},{}",
                                ts_ms, ticker, tte_s, fair, 0.5 * (ybid + yask), u.d_b, u.d_rho, u.seq
                            )?;
                        }
                    }
                }
            }
        }
    }
    eprintln!("replay done: rows={n_rows} signals={n_sigs} events={}", known.len());
    if debug {
        eprintln!("cb feeds={n_cb_feeds}, last cb feed at ts_ms={last_cb_feed_ts}");
    }
    Ok(())
}

fn push_calib(
    state: &mut MarketState,
    rule: &mut FairRideRule,
    calibs: &mut HashMap<String, arb_core::model::CalibUpdate>,
    u: arb_core::model::CalibUpdate,
    seq: &mut u64,
    ts_ns: i64,
) {
    if std::env::var("REPLAY_DEBUG").is_ok() {
        eprintln!("CALIB {} tte={} rows={}", u.instrument, u.fitted_at_tte_s, u.rows);
    }
    calibs.insert(u.instrument.clone(), u.clone());
    *seq += 1;
    let e = ev(format!("market.calib.{}", u.instrument), ts_ns, *seq, Payload::Calib(u));
    state.on_event(&e);
    rule.on_event(&e, state);
}
