//! arb-processor — merges market.* into a derived `MarketState`, runs the rule
//! engine, and emits `signal.<strategy>` (DESIGN §6).

pub mod calib;
pub mod fair;
pub mod module;
pub mod rule;
pub mod state;
pub mod window;

pub use calib::{CalibCfg, CalibCore, Calibrator};
pub use fair::{FairSurface, FitRow};
pub use module::{ProcCfg, Processor};
pub use rule::{FairRideCfg, FairRideRule, PerpMoveRule, Rule, RuleEngine};
pub use state::{InstrumentKind, InstrumentState, MarketState};
pub use window::RollingWindow;
