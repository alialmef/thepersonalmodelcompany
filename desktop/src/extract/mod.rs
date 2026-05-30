//! Source-specific extractors. Each module reads one local Mac data
//! source, normalizes the rows into typed graph entities, and writes
//! them through the shared `GraphStore`.
//!
//! Cadence is decided in `schedule::config`. Each extractor exposes a
//! single `run(...)` entrypoint that takes the store + watermarks and
//! advances the watermark on completion.

pub mod contacts;
pub mod imessage_enrich;
pub mod calendar;
pub mod photos;
pub mod safari;
pub mod call_history;
pub mod music;
pub mod files;
pub mod mail_enrich;
pub mod notes_enrich;
pub mod reminders;
// Phase 2 — deeper ingest for the chief-of-staff agent
pub mod chrome;
pub mod screen_time;
pub mod shell;
pub mod locations;

use crate::graph::{GraphStore, Watermarks};
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("source not found at {0}")]
    SourceNotFound(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("graph store: {0}")]
    Store(#[from] crate::graph::store::GraphStoreError),
    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ExtractSummary {
    pub source: String,
    pub items_processed: u64,
    pub entities_written: u64,
    pub duration_ms: u64,
    pub skipped: bool,
    pub skip_reason: Option<String>,
}

pub struct ExtractCtx {
    pub store: Arc<GraphStore>,
    pub watermarks: std::sync::Mutex<Watermarks>,
}

impl ExtractCtx {
    pub fn new(store: Arc<GraphStore>, watermarks: Watermarks) -> Self {
        Self { store, watermarks: std::sync::Mutex::new(watermarks) }
    }

    pub fn save_watermarks(&self) {
        if let Ok(w) = self.watermarks.lock() {
            let _ = w.save();
        }
    }
}
