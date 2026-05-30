//! Editor recent-files / recent-projects extractor.
//!
//! Reads VS Code and Cursor's recently-opened paths from each editor's
//! global storage. These are the projects + files the user has had
//! *in their head* recently — a leading signal for the project-momentum
//! and time-allocation agents.
//!
//! We emit one `FileSignal` per recent path with `kind="recent_project"`
//! (for folders) or `kind="recent_file"` (for individual files). The
//! synthesis layer joins these against `CodeRepo` entries to enrich
//! the work-in-flight picture.
//!
//! No file contents are read. Only the path + name + mtime are
//! captured from filesystem metadata.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, FileSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const SOURCE: &str = "editor_state";

struct Editor {
    name: &'static str,
    storage_relpath: &'static str,
}

const EDITORS: &[Editor] = &[
    Editor {
        name: "vscode",
        storage_relpath: "Library/Application Support/Code/storage.json",
    },
    Editor {
        name: "cursor",
        storage_relpath: "Library/Application Support/Cursor/storage.json",
    },
    Editor {
        name: "windsurf",
        storage_relpath: "Library/Application Support/Windsurf/storage.json",
    },
];

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Ok(skipped("HOME unset"));
    };

    // Dedup across editors — the same path may be in VS Code + Cursor
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut signals: Vec<FileSignal> = Vec::new();
    let mut any_present = false;

    for ed in EDITORS {
        let path = home.join(ed.storage_relpath);
        if !path.is_file() {
            continue;
        }
        any_present = true;
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let Ok(json) = serde_json::from_str::<Value>(&text) else { continue };
        for p in collect_recent_paths(&json) {
            if !seen.insert(p.clone()) {
                continue;
            }
            if let Some(fs) = path_to_signal(&p, ed.name) {
                signals.push(fs);
            }
            if signals.len() >= 1000 {
                break;
            }
        }
    }

    if !any_present {
        return Ok(skipped("no VS Code / Cursor / Windsurf storage found"));
    }

    let n = signals.len();
    let total = seen.len() as u64;
    ctx.store
        .upsert_many(EntityKind::FileSignal, &signals, |s| s.id.clone())?;
    ctx.store.flush_kind(EntityKind::FileSignal)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", total);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

/// Pull out every recently-opened folder + file path the editor's
/// storage.json exposes. VS Code's shape has rotated a few times so
/// we walk known keys defensively and accept either flat-list or
/// object-wrapped forms.
fn collect_recent_paths(root: &Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    walk_for_paths(root, &mut out, 0);
    // Dedup while preserving order
    let mut seen: BTreeSet<String> = BTreeSet::new();
    out.retain(|p| seen.insert(p.clone()));
    out
}

fn walk_for_paths(v: &Value, out: &mut Vec<String>, depth: u32) {
    if depth > 12 {
        return;
    }
    match v {
        Value::Object(map) => {
            for (k, child) in map {
                let lk = k.to_lowercase();
                // Direct path-bearing keys
                if matches!(
                    lk.as_str(),
                    "folderuri" | "fileuri" | "configpath" | "workspace" | "path" | "uri"
                ) {
                    if let Some(s) = child.as_str() {
                        if let Some(p) = uri_to_path(s) {
                            out.push(p);
                        }
                    }
                }
                walk_for_paths(child, out, depth + 1);
            }
        }
        Value::Array(arr) => {
            for child in arr {
                walk_for_paths(child, out, depth + 1);
            }
        }
        Value::String(s) => {
            // Some shapes flat-store URIs in arrays we already walked
            if let Some(p) = uri_to_path(s) {
                out.push(p);
            }
        }
        _ => {}
    }
}

fn uri_to_path(s: &str) -> Option<String> {
    if let Some(rest) = s.strip_prefix("file://") {
        // URL-decode the most common escapes (spaces). Good enough for
        // the local-paths-only space we care about.
        let decoded = rest.replace("%20", " ");
        if decoded.starts_with('/') {
            return Some(decoded);
        }
    }
    if s.starts_with('/') && s.len() > 1 {
        return Some(s.to_string());
    }
    None
}

fn path_to_signal(path_str: &str, editor: &str) -> Option<FileSignal> {
    let p = Path::new(path_str);
    let name = p.file_name()?.to_string_lossy().to_string();
    let (kind, ext) = if p.is_dir() {
        ("recent_project".to_string(), None)
    } else if p.is_file() {
        let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase());
        ("recent_file".to_string(), ext)
    } else {
        // Stale entry — path no longer exists. Still useful as a "the
        // user thought about this recently" signal, but mark it.
        ("recent_missing".to_string(), None)
    };

    let modified: Option<DateTime<Utc>> = std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos()));
    let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);

    Some(FileSignal {
        id: stable_id(&["editor_recent", editor, path_str]),
        path: path_str.to_string(),
        name,
        extension: ext,
        modified,
        size_bytes: size,
        kind: Some(kind),
    })
}

fn skipped(reason: &str) -> ExtractSummary {
    ExtractSummary {
        source: SOURCE.into(),
        items_processed: 0,
        entities_written: 0,
        duration_ms: 0,
        skipped: true,
        skip_reason: Some(reason.into()),
    }
}
