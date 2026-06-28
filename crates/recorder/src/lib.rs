//! Recorder — persists the bus event stream to **Hive-partitioned JSONL** for
//! audit + replay (DESIGN §3/§11). Layout:
//!
//!   <dir>/stream=<book|trade|catalog|signal|…>/venue=<binance|polymarket|…>/date=<YYYY-MM-DD>/events.jsonl
//!
//! Partition columns are derived per event (stream + venue from the topic,
//! date from `ts_ns` in UTC), enabling partition-pruned reads and date-based
//! retention (drop old `date=` dirs). Subscribes with `Block` (full stream);
//! optional per-key time-cadence downsampling bounds hot-instrument volume.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::task::JoinHandle;

use arb_core::bus::{Bus, Policy};
use arb_core::event::{Event, Payload};
use arb_core::module::{Health, Module};

#[derive(Clone, serde::Deserialize)]
#[serde(default)]
pub struct RecorderCfg {
    pub enabled: bool,
    /// Base directory; Hive partitions are created underneath it.
    pub dir: String,
    /// Topic pattern to record (`#` = all, `signal.#`, `market.binance.#`, …).
    pub pattern: String,
    /// Keep at most one event per sample-key per this interval. `0` = every event.
    pub sample_interval_ms: u64,
    /// Sample-key granularity: `"instrument"` (default) or `"topic"`.
    pub sample_key: String,
    /// Topic prefixes never downsampled — always recorded in full.
    pub always_keep: Vec<String>,
}

impl Default for RecorderCfg {
    fn default() -> Self {
        RecorderCfg {
            enabled: false,
            dir: "data/events".into(),
            pattern: "#".into(),
            sample_interval_ms: 0,
            sample_key: "instrument".into(),
            always_keep: vec![
                "signal.".into(),
                "exec.".into(),
                "latency.".into(),
                "market.polymarket.catalog".into(),
            ],
        }
    }
}

fn instrument_of(ev: &Event) -> &str {
    match &ev.payload {
        Payload::Book(b) => &b.instrument,
        Payload::Trade(t) => &t.instrument,
        Payload::Liq(l) => &l.instrument,
        Payload::Meta(m) => &m.instrument,
        Payload::Signal(s) => &s.target,
        Payload::Latency(l) => &l.name,
        Payload::ExecReport(r) => &r.instrument,
        Payload::TradeRecord(t) => &t.instrument,
        Payload::Position(p) => &p.instrument,
    }
}

/// `(stream, venue)` partition columns from the topic. `market.<venue>.….<stream>`
/// → (last segment, venue); otherwise (first segment, event source).
fn stream_venue(ev: &Event) -> (&str, &str) {
    let mut segs = ev.topic.split('.');
    let first = segs.next().unwrap_or("event");
    if first == "market" {
        let venue = ev.topic.split('.').nth(1).unwrap_or(ev.source);
        let stream = ev.topic.rsplit('.').next().unwrap_or("event");
        (stream, venue)
    } else {
        (first, ev.source)
    }
}

/// UTC `YYYY-MM-DD` from nanoseconds since the unix epoch (Hinnant civil-from-days).
fn date_utc(ts_ns: i64) -> String {
    let days = ts_ns.div_euclid(86_400_000_000_000);
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }).div_euclid(146_097);
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02}")
}

pub struct Recorder {
    cfg: RecorderCfg,
    handle: Option<JoinHandle<()>>,
}

impl Recorder {
    pub fn new(cfg: RecorderCfg) -> Self {
        Recorder { cfg, handle: None }
    }
}

#[async_trait]
impl Module for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }

    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()> {
        if !self.cfg.enabled {
            tracing::info!("recorder disabled");
            return Ok(());
        }
        let mut sub = bus.subscribe(&self.cfg.pattern, 65_536, Policy::Block);
        let base = self.cfg.dir.clone();
        let interval_ns = (self.cfg.sample_interval_ms as i64) * 1_000_000;
        let by_topic = self.cfg.sample_key == "topic";
        let always_keep = self.cfg.always_keep.clone();
        tracing::info!(
            "recording '{}' -> {}/stream=*/venue=*/date=* (sample={}ms by {})",
            self.cfg.pattern, base, self.cfg.sample_interval_ms, self.cfg.sample_key,
        );

        let handle = tokio::spawn(async move {
            // one buffered writer per (stream, venue, date) partition
            let mut writers: HashMap<String, BufWriter<File>> = HashMap::new();
            let mut last: HashMap<String, i64> = HashMap::new();
            let mut written = 0u64;
            let mut dropped = 0u64;
            let mut flush = tokio::time::interval(Duration::from_secs(1));
            loop {
                tokio::select! {
                    ev = sub.recv() => {
                        let Some(ev) = ev else { break };
                        // downsample (uses event ts_ns for determinism)
                        let keep = if interval_ns == 0 {
                            true
                        } else if always_keep.iter().any(|p| ev.topic.starts_with(p.as_str())) {
                            true
                        } else {
                            let key = if by_topic { ev.topic.clone() } else { instrument_of(&ev).to_string() };
                            match last.get(&key) {
                                Some(&t) if ev.ts_ns - t < interval_ns => false,
                                _ => { last.insert(key, ev.ts_ns); true }
                            }
                        };
                        if !keep { dropped += 1; continue; }

                        let (stream, venue) = stream_venue(&ev);
                        let date = date_utc(ev.ts_ns);
                        let dir = format!("{base}/stream={stream}/venue={venue}/date={date}");
                        if !writers.contains_key(&dir) {
                            let _ = tokio::fs::create_dir_all(&dir).await;
                            match tokio::fs::OpenOptions::new()
                                .create(true).append(true)
                                .open(format!("{dir}/events.jsonl")).await
                            {
                                Ok(f) => { writers.insert(dir.clone(), BufWriter::new(f)); }
                                Err(e) => { tracing::warn!("recorder open {dir}: {e}"); continue; }
                            }
                        }
                        if let (Some(w), Ok(mut line)) = (writers.get_mut(&dir), serde_json::to_string(&*ev)) {
                            line.push('\n');
                            if w.write_all(line.as_bytes()).await.is_ok() { written += 1; }
                        }
                    }
                    _ = flush.tick() => {
                        for w in writers.values_mut() { let _ = w.flush().await; }
                    }
                }
            }
            for w in writers.values_mut() { let _ = w.flush().await; }
            tracing::info!("recorder stopped: {written} written, {dropped} downsampled, {} partitions", writers.len());
        });
        self.handle = Some(handle);
        Ok(())
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(h) = self.handle.take() {
            h.abort();
        }
        Ok(())
    }

    fn health(&self) -> Health {
        Health::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_from_ns() {
        // ts from a recorded event on 2026-06-14
        assert_eq!(date_utc(1_781_416_438_838_654_500), "2026-06-14");
        assert_eq!(date_utc(0), "1970-01-01");
    }

    #[test]
    fn latency_sample_routes_and_serializes() {
        use arb_core::model::LatencySample;
        let ev = Event::new(
            "latency.processor.book_to_signal",
            "processor",
            1_781_460_000_000_000_000,
            7,
            Payload::Latency(LatencySample {
                name: "book_to_signal".into(),
                latency_us: 42.5,
                origin_ts_ns: 1_781_460_000_000_000_000,
                strategy: "perp_move".into(),
                target: "polymarket.0xabc.UP".into(),
            }),
        );
        // Partition routing: non-`market.` topic -> (first segment, source).
        assert_eq!(stream_venue(&ev), ("latency", "processor"));
        assert_eq!(instrument_of(&ev), "book_to_signal");
        // Recorded line shape (same serde the writer uses).
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains(r#""topic":"latency.processor.book_to_signal""#), "{json}");
        assert!(
            json.contains(r#""Latency":{"name":"book_to_signal","latency_us":42.5"#),
            "{json}"
        );
    }
}
