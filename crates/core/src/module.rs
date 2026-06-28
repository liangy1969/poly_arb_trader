//! Module lifecycle trait (DESIGN §4). The Supervisor (later) start/stops these.

use std::sync::Arc;

use async_trait::async_trait;

use crate::bus::Bus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Health {
    Ok,
    Degraded,
    Down,
}

#[async_trait]
pub trait Module: Send {
    fn name(&self) -> &'static str;
    async fn start(&mut self, bus: Arc<dyn Bus>) -> anyhow::Result<()>;
    async fn stop(&mut self) -> anyhow::Result<()>;
    fn health(&self) -> Health;
}
