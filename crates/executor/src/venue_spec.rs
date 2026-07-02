//! Venue descriptors (DESIGN_MULTI_VENUE §4) — the per-venue structural facts the
//! executor needs, resolved from the instrument prefix, so no venue string literal
//! lives in the trade path. The collectors emit the uniform
//! `<venue>.<market_id>.<outcome>` instrument convention; this is the executor's
//! view of it. Selecting the active venue is config (`executor.venue.market`).

/// One prediction venue's structural facts.
#[derive(Clone, Copy, Debug)]
pub struct VenueSpec {
    /// Bus topic segment + instrument prefix, e.g. "polymarket" | "kalshi".
    pub prefix: &'static str,
    /// Outcome label bought on a +1 (up) signal — the linker target's suffix.
    pub yes_label: &'static str,
    /// Outcome label bought on a -1 (down) signal.
    pub no_label: &'static str,
    /// Mandatory delay the venue applies to marketable orders (Polymarket 250 ms;
    /// Kalshi none). Drives the sim's taker-delay floor.
    pub taker_delay_ms: u64,
}

pub const POLYMARKET: VenueSpec = VenueSpec {
    prefix: "polymarket",
    yes_label: "UP",
    no_label: "DOWN",
    taker_delay_ms: 250,
};

pub const KALSHI: VenueSpec = VenueSpec {
    prefix: "kalshi",
    yes_label: "YES",
    no_label: "NO",
    taker_delay_ms: 0,
};

const REGISTRY: &[VenueSpec] = &[POLYMARKET, KALSHI];

/// Resolve a venue by its config name ("polymarket" | "kalshi").
pub fn by_name(name: &str) -> Option<VenueSpec> {
    REGISTRY.iter().find(|v| v.prefix == name).copied()
}

/// Resolve a venue from an instrument's prefix (`<venue>.…`).
pub fn venue_of(instrument: &str) -> Option<VenueSpec> {
    by_name(instrument.split('.').next()?)
}

/// `<venue>.<market_id>.<outcome>` → `<market_id>` (the middle segment), for a
/// known prediction venue only. Generic across venues: Polymarket condition-ids
/// (hex) and Kalshi tickers (`KXBTC15M-240621-14C`, dashes) hold no internal '.'.
/// Used to join a traded outcome back to its market metadata (expiry/status/winner).
pub fn market_id_of(instrument: &str) -> Option<&str> {
    venue_of(instrument)?; // only for recognised prediction venues (not e.g. binance.)
    let rest = instrument.split_once('.')?.1; // drop "<venue>."
    rest.rsplit_once('.').map(|(mid, _)| mid) // drop ".<outcome>"
}

/// Map signal `direction` to the instrument we BUY (DESIGN_EXECUTION §0). The
/// linker target is the YES outcome (UP/YES); `-1` trades the complementary NO
/// outcome (DOWN/NO) — buying NO ≡ selling YES.
pub fn traded_instrument(target_yes: &str, direction: i8) -> String {
    if direction >= 0 {
        return target_yes.to_string();
    }
    if let Some(spec) = venue_of(target_yes) {
        if let Some(stem) = target_yes.strip_suffix(&format!(".{}", spec.yes_label)) {
            return format!("{stem}.{}", spec.no_label);
        }
    }
    target_yes.to_string() // unknown orientation: trade the target as-is
}

/// The other outcome token of the same market (YES↔NO, UP↔DOWN). Works from
/// EITHER side (unlike `traded_instrument`, which assumes a YES-side input).
/// The exit reconciler closes a position by BUYING this complement.
pub fn complement(instrument: &str) -> String {
    if let Some(spec) = venue_of(instrument) {
        if let Some(stem) = instrument.strip_suffix(&format!(".{}", spec.yes_label)) {
            return format!("{stem}.{}", spec.no_label);
        }
        if let Some(stem) = instrument.strip_suffix(&format!(".{}", spec.no_label)) {
            return format!("{stem}.{}", spec.yes_label);
        }
    }
    instrument.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_venue_by_prefix() {
        assert_eq!(venue_of("polymarket.0xabc.UP").unwrap().prefix, "polymarket");
        assert_eq!(venue_of("kalshi.KXBTC15M-240621-14C.YES").unwrap().prefix, "kalshi");
        assert!(venue_of("binance.usdt_perp.BTCUSDT").is_none());
        assert_eq!(by_name("kalshi").unwrap().taker_delay_ms, 0);
        assert_eq!(by_name("polymarket").unwrap().taker_delay_ms, 250);
        assert!(by_name("nope").is_none());
    }

    #[test]
    fn market_id_parses_both_venues() {
        assert_eq!(market_id_of("polymarket.0xabc.UP"), Some("0xabc"));
        assert_eq!(market_id_of("polymarket.0xabc.DOWN"), Some("0xabc"));
        assert_eq!(market_id_of("kalshi.KXBTC15M-240621-14C.YES"), Some("KXBTC15M-240621-14C"));
        assert_eq!(market_id_of("kalshi.KXBTC15M-240621-14C.NO"), Some("KXBTC15M-240621-14C"));
        assert_eq!(market_id_of("binance.usdt_perp.BTCUSDT"), None); // not a prediction venue
        assert_eq!(market_id_of("nope"), None);
    }

    #[test]
    fn traded_instrument_maps_direction_per_venue() {
        // Polymarket UP/DOWN
        assert_eq!(traded_instrument("polymarket.0xabc.UP", 1), "polymarket.0xabc.UP");
        assert_eq!(traded_instrument("polymarket.0xabc.UP", -1), "polymarket.0xabc.DOWN");
        // Kalshi YES/NO
        assert_eq!(traded_instrument("kalshi.KXBTC15M-1.YES", 1), "kalshi.KXBTC15M-1.YES");
        assert_eq!(traded_instrument("kalshi.KXBTC15M-1.YES", -1), "kalshi.KXBTC15M-1.NO");
    }

    #[test]
    fn complement_flips_either_side() {
        assert_eq!(complement("kalshi.KXBTC15M-1.YES"), "kalshi.KXBTC15M-1.NO");
        assert_eq!(complement("kalshi.KXBTC15M-1.NO"), "kalshi.KXBTC15M-1.YES");
        assert_eq!(complement("polymarket.0xabc.UP"), "polymarket.0xabc.DOWN");
        assert_eq!(complement("polymarket.0xabc.DOWN"), "polymarket.0xabc.UP");
        assert_eq!(complement("binance.usdt_perp.BTCUSDT"), "binance.usdt_perp.BTCUSDT");
    }
}
