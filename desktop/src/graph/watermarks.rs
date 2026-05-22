//! Per-source watermarks: the last successfully-processed cursor for each
//! extractor. Lets continuous-build runs read only what's new since last
//! pass without re-scanning the full source.
//!
//! For SQLite-backed sources (iMessage, Contacts, Calendar, Photos,
//! Safari, etc.) the watermark is a row identifier or timestamp. For
//! filesystem-backed sources (Notes, Mail, file walks) the watermark is
//! an mtime cutoff. The format is intentionally opaque — each extractor
//! interprets its own watermark string.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WatermarkEntry {
    pub cursor: String,
    pub last_run_at: Option<DateTime<Utc>>,
    pub items_seen: u64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Watermarks {
    #[serde(default)]
    by_source: BTreeMap<String, WatermarkEntry>,
    #[serde(skip)]
    path: PathBuf,
}

impl Watermarks {
    pub fn load(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(mut w) = serde_json::from_str::<Watermarks>(&text) {
                w.path = path;
                return w;
            }
        }
        Self { path, ..Default::default() }
    }

    pub fn get(&self, source: &str) -> WatermarkEntry {
        self.by_source.get(source).cloned().unwrap_or_default()
    }

    pub fn set(&mut self, source: &str, cursor: impl Into<String>, items_seen: u64) {
        self.by_source.insert(source.to_string(), WatermarkEntry {
            cursor: cursor.into(),
            last_run_at: Some(Utc::now()),
            items_seen,
        });
    }

    pub fn save(&self) -> Result<(), std::io::Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let text = serde_json::to_string_pretty(self).unwrap_or_default();
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }

    pub fn path(&self) -> &Path { &self.path }
}
