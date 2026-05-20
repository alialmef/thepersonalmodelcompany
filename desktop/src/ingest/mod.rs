//! Native ingestion modules — read local Mac data sources directly.
//!
//! Each submodule handles one data source (iMessage, Apple Notes, Apple Mail,
//! WhatsApp). They produce `RawItem`-shaped JSON that's pushed to the FastAPI
//! backend's `/v1/users/{user_id}/sources/items` endpoint.

pub mod imessage;
pub mod mail;
pub mod notes;

use serde::Serialize;
use std::collections::HashMap;

/// Wire-format RawItem that matches the Pydantic `pmc.ingest.base.RawItem`
/// schema on the backend. Each ingestion module produces these.
#[derive(Debug, Serialize)]
pub struct RawItemJson {
    pub source_type: &'static str,
    pub source_id: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_user: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, String>,
}

/// Errors any ingestion module can produce, serialized to JSON for the Tauri
/// command boundary.
#[derive(Debug, thiserror::Error, Serialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
pub enum IngestError {
    #[error("source not found at the expected location")]
    NotFound,
    #[error(
        "permission denied — grant Full Disk Access in System Settings → \
         Privacy & Security → Full Disk Access"
    )]
    PermissionDenied,
    #[error("read error: {0}")]
    ReadError(String),
    #[error("http error: {0}")]
    HttpError(String),
    #[error("internal: {0}")]
    Internal(String),
}

/// Summary returned to the frontend after a successful native ingestion run.
#[derive(Debug, Serialize)]
pub struct IngestSummary {
    pub source: &'static str,
    pub source_id: String,
    pub items_ingested: usize,
}
