//! Replay-parity harness (DESIGN_FAIR_RIDE §7 gate 3): stream online-collected
//! 50ms sampler rows through the REAL FairRide pipeline (CalibCore →
//! MarketState → FairRideRule) and dump signals + per-row fair values for
//! equivalence checks against the PyTorch reference.
//!
//! Usage: replay_fair_ride <samples.csv> <meta_cache.json> <model.json> <out_prefix>
//! (plain CSV — gunzip first). Emits <out_prefix>_signals.csv and
//! <out_prefix>_fair.csv (1s-grid fair series per event, post-first-fit).

use std::collections::HashMap;
use std::io::{BufRead, Write};

use arb_core::event::{Event, Payload};
use arb_core::model::{BookUpdate, MarketMeta, MarketStatus};
use arb_processor::{CalibCfg, CalibCore, FairRideCfg, FairRideRule, MarketState, Rule};

fn ev(topic: String, ts_ns: i64, seq: u64, payload: Payload) -> Event {
    Event { topic, source: "replay", ts_ns, seq, payload }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (csv_path, meta_path, model_path, out_prefix) = (&args[1], &args[2], &args[3], &args[4]);

    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
    let bytes = std::fs::read(model_path)?;
    let surface = std::sync::Arc::new(arb_processor::FairSurface::from_json(std::str::from_utf8(&bytes)?)?);
    let hash = arb_processor::calib::fnv1a(&bytes);

    let calib_cfg = CalibCfg { enabled: true, model_path: model_path.clone(), ..Default::default() };
    let mut core = CalibCore::new(calib_cfg, surface.clone(), hash);
    // Equivalence mode: uncap entries so episode accounting can be compared
    // trade-by-trade against the reference; the live cap is a spec constant.
    let ride_cfg = FairRideCfg { max_entries_per_event: 255, ..Default::default() };
    let mut rule = FairRideRule::new(ride_cfg.clone(), surface.clone(), hash);
    let mut state = MarketState::new(ride_cfg.reference.clone(), "15m_updown".into(), 256, 10_000);

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

    let mut sigs = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}_signals.csv"))?);
    writeln!(sigs, "ts_ms,target,direction,yes_price,reason")?;
    let mut fairs = std::io::BufWriter::new(std::fs::File::create(format!("{out_prefix}_fair.csv"))?);
    writeln!(fairs, "ts_ms,ticker,tte_s,fair,mid,d_b,d_rho,fit_seq")?;

    let mut seq = 0u64;
    let mut known: HashMap<String, (f64, i64)> = HashMap::new(); // ticker -> (strike, expiry_ns)
    let mut calibs: HashMap<String, arb_core::model::CalibUpdate> = HashMap::new();
    let mut last_cb: (f64, f64) = (0.0, 0.0);
    let mut last_fair_ms: HashMap<String, i64> = HashMap::new();
    let mut n_rows = 0u64;
    let mut n_sigs = 0u64;

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
            let strike = meta
                .get(ticker)
                .and_then(|m| m.get("strike"))
                .and_then(|v| v.as_f64());
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
            state.on_event(&m);
            for u in core.on_event(&m) {
                calibs.insert(u.instrument.clone(), u);
            }
            rule.on_event(&m, &state);
        }

        // coinbase reference book (recv stamped back by its age, as live)
        let (cb_b, cb_a): (f64, f64) = (f[i_cbb].parse().unwrap_or(0.0), f[i_cba].parse().unwrap_or(0.0));
        let cb_age_ms: i64 = f[i_cbage].parse().unwrap_or(-1);
        if cb_b > 0.0 && cb_a > 0.0 && cb_age_ms >= 0 && (cb_b, cb_a) != last_cb {
            last_cb = (cb_b, cb_a);
            seq += 1;
            let b = ev(
                "market.coinbase.BTC.book".into(),
                ts_ns,
                seq,
                Payload::Book(BookUpdate {
                    instrument: "coinbase.BTC".into(),
                    bids: vec![(cb_b, 0.0)],
                    asks: vec![(cb_a, 0.0)],
                    update_id: None,
                    // arrival semantics: the live collector delivers at recv
                    // time; the venue-print staleness shows up as the VALUE
                    // not changing (no event), matching this change-detect.
                    exch_ts_ns: ts_ns,
                    recv_ts_ns: ts_ns,
                }),
            );
            state.on_event(&b);
            for u in core.on_event(&b) {
                push_calib(&mut state, &mut rule, &mut calibs, u, &mut seq, ts_ns);
            }
            for s in rule.on_event(&b, &state) {
                n_sigs += 1;
                writeln!(sigs, "{},{},{},{:.3},\"{}\"", ts_ms, s.target, s.direction, s.trigger.yes_price, s.reason)?;
            }
        }

        // kalshi YES book
        let (ybid, yask): (f64, f64) = (f[i_ybid].parse().unwrap_or(0.0), f[i_yask].parse().unwrap_or(0.0));
        if ybid > 0.0 && yask > 0.0 {
            let inst = format!("kalshi.{ticker}.YES");
            seq += 1;
            let b = ev(
                format!("market.kalshi.{ticker}.book"),
                ts_ns,
                seq,
                Payload::Book(BookUpdate {
                    instrument: inst.clone(),
                    bids: vec![(ybid, 0.0)],
                    asks: vec![(yask, 0.0)],
                    update_id: None,
                    exch_ts_ns: ts_ns,
                    recv_ts_ns: ts_ns,
                }),
            );
            state.on_event(&b);
            for u in core.on_event(&b) {
                push_calib(&mut state, &mut rule, &mut calibs, u, &mut seq, ts_ns);
            }
            for s in rule.on_event(&b, &state) {
                n_sigs += 1;
                writeln!(sigs, "{},{},{},{:.3},\"{}\"", ts_ms, s.target, s.direction, s.trigger.yes_price, s.reason)?;
            }
            // 1s fair diagnostics (post-fit)
            if let Some(u) = calibs.get(&inst) {
                let last = last_fair_ms.entry(inst.clone()).or_insert(0);
                if ts_ms - *last >= 1000 {
                    *last = ts_ms;
                    let (strike, expiry_ns) = known[ticker];
                    let tte_s = (expiry_ns - ts_ns) as f64 / 1e9;
                    if tte_s > 0.0 && cb_b > 0.0 {
                        let fair = surface.fair(0.5 * (cb_b + cb_a), tte_s, strike, u.d_b, u.d_rho);
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
    eprintln!("replay done: rows={n_rows} signals={n_sigs} events={}", known.len());
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
    calibs.insert(u.instrument.clone(), u.clone());
    *seq += 1;
    let e = ev(format!("market.calib.{}", u.instrument), ts_ns, *seq, Payload::Calib(u));
    state.on_event(&e);
    rule.on_event(&e, state);
}
