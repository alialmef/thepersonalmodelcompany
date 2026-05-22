//! Append-on-write JSONL store with per-entity-type files.
//!
//! Idempotency model: each extractor passes a `key` (the stable id of
//! the entity); the store overwrites any prior entry with that key on
//! flush rather than appending duplicates. We achieve "append-only with
//! upsert semantics" by buffering writes in-memory keyed by id, then
//! atomically rewriting the file on `flush_kind`. For tiny tables this
//! is fine; we'll add a compaction-on-disk path when individual files
//! grow past ~10 MB.
//!
//! Concurrency: a single `GraphStore` instance per user is the contract.
//! Multiple extractors call into it from different async tasks; an
//! internal `Mutex` serializes writes. Reading is currently lockless
//! (re-reads the file each time) — fine for our scale.

use crate::graph::schema::EntityKind;
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, thiserror::Error)]
pub enum GraphStoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("lock poisoned")]
    Poisoned,
}

pub struct GraphStore {
    root: PathBuf,
    /// In-memory buffers, keyed by entity kind then by id. We rewrite the
    /// file from this map on flush. Persisted on disk on `flush_kind`.
    buffers: Mutex<BTreeMap<&'static str, BTreeMap<String, serde_json::Value>>>,
}

impl GraphStore {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self, GraphStoreError> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let store = Self { root, buffers: Mutex::new(BTreeMap::new()) };
        store.hydrate()?;
        Ok(store)
    }

    /// Read existing JSONL files back into the in-memory map so partial
    /// extractor runs are additive rather than destructive.
    fn hydrate(&self) -> Result<(), GraphStoreError> {
        let kinds = [
            EntityKind::Person, EntityKind::Place, EntityKind::Event,
            EntityKind::Episode, EntityKind::Project, EntityKind::Theme,
            EntityKind::OpenLoop, EntityKind::TasteItem,
            EntityKind::FileSignal, EntityKind::CodeRepo,
            EntityKind::WebSignal, EntityKind::Edge,
        ];
        let mut buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        for kind in kinds {
            let path = self.root.join(kind.filename());
            let mut by_id: BTreeMap<String, serde_json::Value> = BTreeMap::new();
            if path.is_file() {
                let text = std::fs::read_to_string(&path)?;
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };
                    let id = value
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !id.is_empty() {
                        by_id.insert(id, value);
                    }
                }
            }
            buffers.insert(kind.filename(), by_id);
        }
        Ok(())
    }

    /// Upsert one entity into the in-memory buffer (call `flush_kind` to
    /// persist).
    pub fn upsert<T: Serialize>(
        &self,
        kind: EntityKind,
        id: &str,
        entity: &T,
    ) -> Result<(), GraphStoreError> {
        let value = serde_json::to_value(entity)?;
        let mut buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        let by_id = buffers
            .entry(kind.filename())
            .or_insert_with(BTreeMap::new);
        by_id.insert(id.to_string(), value);
        Ok(())
    }

    /// Bulk variant: upsert many entries before flushing.
    pub fn upsert_many<T: Serialize, F: Fn(&T) -> String>(
        &self,
        kind: EntityKind,
        entities: &[T],
        id_of: F,
    ) -> Result<(), GraphStoreError> {
        let mut buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        let by_id = buffers
            .entry(kind.filename())
            .or_insert_with(BTreeMap::new);
        for e in entities {
            let id = id_of(e);
            let value = serde_json::to_value(e)?;
            by_id.insert(id, value);
        }
        Ok(())
    }

    /// Drop every entry of one entity kind. Useful for synthesis passes
    /// that recompute from scratch (themes, episodes) and shouldn't
    /// accumulate stale rows from prior runs.
    pub fn clear_kind(&self, kind: EntityKind) -> Result<(), GraphStoreError> {
        let mut buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        buffers.entry(kind.filename()).or_insert_with(BTreeMap::new).clear();
        Ok(())
    }

    /// Persist the in-memory buffer for one entity kind to disk.
    /// Uses write-temp-then-rename for atomicity.
    pub fn flush_kind(&self, kind: EntityKind) -> Result<usize, GraphStoreError> {
        let buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        let empty = BTreeMap::new();
        let by_id = buffers.get(kind.filename()).unwrap_or(&empty);
        let path = self.root.join(kind.filename());
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            use std::io::Write;
            for value in by_id.values() {
                writeln!(f, "{}", serde_json::to_string(value)?)?;
            }
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(by_id.len())
    }

    /// Flush all entity kinds.
    pub fn flush_all(&self) -> Result<(), GraphStoreError> {
        let kinds = [
            EntityKind::Person, EntityKind::Place, EntityKind::Event,
            EntityKind::Episode, EntityKind::Project, EntityKind::Theme,
            EntityKind::OpenLoop, EntityKind::TasteItem,
            EntityKind::FileSignal, EntityKind::CodeRepo,
            EntityKind::WebSignal, EntityKind::Edge,
        ];
        for kind in kinds {
            self.flush_kind(kind)?;
        }
        Ok(())
    }

    pub fn load<T: DeserializeOwned>(
        &self,
        kind: EntityKind,
    ) -> Result<Vec<T>, GraphStoreError> {
        let buffers = self.buffers.lock().map_err(|_| GraphStoreError::Poisoned)?;
        let empty = BTreeMap::new();
        let by_id = buffers.get(kind.filename()).unwrap_or(&empty);
        let mut out = Vec::with_capacity(by_id.len());
        for value in by_id.values() {
            out.push(serde_json::from_value(value.clone())?);
        }
        Ok(out)
    }

    pub fn count(&self, kind: EntityKind) -> usize {
        let Ok(buffers) = self.buffers.lock() else { return 0 };
        buffers.get(kind.filename()).map(|m| m.len()).unwrap_or(0)
    }

    pub fn root(&self) -> &Path { &self.root }
}

// ---------------------------------------------------------------------------
// Tiny helper: deterministic id from arbitrary string. Used by extractors
// to derive stable IDs from raw source identifiers without dragging in a
// UUID dependency.
// ---------------------------------------------------------------------------

pub fn stable_id(parts: &[&str]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for p in parts { p.hash(&mut h); "\u{0}".hash(&mut h); }
    format!("{:016x}", h.finish())
}

#[allow(dead_code)]
pub fn iso_now() -> DateTime<Utc> { Utc::now() }
