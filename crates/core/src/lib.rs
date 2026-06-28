//! arb-core — shared event model, data types, and the Bus/Module traits.
//! No I/O; depended on by every other crate.

pub mod bus;
pub mod event;
pub mod model;
pub mod module;

pub use bus::{key_by_instrument, Bus, ConflateChan, Policy, PortInbox, Subscription};
pub use event::{Event, Payload, Topic};
pub use model::*;
pub use module::{Health, Module};

/// Wall-clock nanoseconds since the unix epoch. Live code stamps events with
/// this; tests/replay pass explicit `ts_ns` so logic never reads the clock.
pub fn now_ns() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos() as i64
}
