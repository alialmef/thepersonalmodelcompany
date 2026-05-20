//! Notes / personal text-file ingestion.
//!
//! Apple Notes' own database (`NoteStore.sqlite` + `NotesV*` CoreData) is
//! encrypted-ish — the body is a binary plist of an Apple proprietary
//! Protobuf, decoding it well is a project unto itself.
//!
//! For V0 we cover the much more common case: **the user's plain-text
//! writing scattered across their disk** — Obsidian vaults, Bear export
//! folders, freeform `.md` and `.txt` files in Documents and Desktop, etc.
//!
//! Scope:
//!   - ~/Documents (recursive, depth-limited)
//!   - ~/Desktop (recursive, depth-limited)
//!   - ~/Notes (if present — common Obsidian vault location)
//! Extensions: `.txt`, `.md`, `.markdown`, `.mdx`, `.rtf`
//! File-size cap: 1 MB per file (skip novels; we want notes-sized writing)
//! File count cap: 5000 files (defensive against pathological scans)
//!
//! Apple Notes proper can be added later via a NoteStore parser; this
//! covers the realistic 80% for tonight.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::ingest::{IngestError, RawItemJson};

const MAX_FILE_SIZE: u64 = 1_000_000; // 1 MB
const MAX_FILES: usize = 5000;
const MAX_DEPTH: u32 = 6;

const EXTENSIONS: &[&str] = &["txt", "md", "markdown", "mdx", "rtf"];

// Directories we deliberately don't walk into — they're full of code / vendor
// stuff that isn't the user's writing.
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "target",
    "dist",
    "build",
    ".next",
    ".cache",
    "Library",       // Mac library — too noisy
    ".Trash",
    ".DS_Store",
];

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn scan_roots() -> Vec<PathBuf> {
    let h = match home() {
        Some(h) => h,
        None => return Vec::new(),
    };
    let mut roots = Vec::new();
    for name in &["Documents", "Desktop", "Notes"] {
        let p = h.join(name);
        if p.exists() {
            roots.push(p);
        }
    }
    roots
}

#[derive(Debug, serde::Serialize)]
pub struct NotesStatus {
    pub exists: bool,
    pub can_read: bool,
    pub message_count: Option<i64>,
    pub error: Option<String>,
}

pub fn status() -> NotesStatus {
    let roots = scan_roots();
    if roots.is_empty() {
        return NotesStatus {
            exists: false,
            can_read: false,
            message_count: None,
            error: Some("not_found".to_string()),
        };
    }
    // Quick count — just walk for paths, don't read contents.
    let paths = find_text_files(&roots, MAX_FILES);
    NotesStatus {
        exists: true,
        can_read: true,
        message_count: Some(paths.len() as i64),
        error: None,
    }
}

fn find_text_files(roots: &[PathBuf], limit: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for root in roots {
        let mut stack: Vec<(PathBuf, u32)> = vec![(root.clone(), 0)];
        while let Some((dir, depth)) = stack.pop() {
            if depth > MAX_DEPTH || out.len() >= limit {
                continue;
            }
            let entries = match fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                // Skip hidden files and known-noisy dirs.
                if name_str.starts_with('.') {
                    continue;
                }
                if SKIP_DIRS.iter().any(|s| s.eq_ignore_ascii_case(&name_str)) {
                    continue;
                }

                if path.is_dir() {
                    stack.push((path, depth + 1));
                } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ext_lower = ext.to_ascii_lowercase();
                    if EXTENSIONS.iter().any(|e| *e == ext_lower) {
                        if let Ok(meta) = path.metadata() {
                            if meta.len() > 0 && meta.len() <= MAX_FILE_SIZE {
                                out.push(path);
                                if out.len() >= limit {
                                    return out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Read every text file found under our roots, return RawItems for the
/// content. RTF files get a crude tag-strip; markdown/text pass through as-is.
pub fn read_notes(limit: Option<usize>) -> Result<Vec<RawItemJson>, IngestError> {
    let roots = scan_roots();
    if roots.is_empty() {
        return Err(IngestError::NotFound);
    }

    let cap = limit.unwrap_or(MAX_FILES).min(MAX_FILES);
    let paths = find_text_files(&roots, cap);

    let mut items = Vec::with_capacity(paths.len());
    for path in paths {
        let raw = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue, // skip unreadable / non-utf8
        };
        let content = if path.extension().and_then(|s| s.to_str()) == Some("rtf") {
            strip_rtf(&raw)
        } else {
            raw
        };
        let content = content.trim();
        if content.is_empty() {
            continue;
        }

        let mut metadata = HashMap::new();
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            metadata.insert("filename".to_string(), name.to_string());
        }

        let timestamp = path
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| {
                t.duration_since(std::time::UNIX_EPOCH).ok().map(|d| {
                    chrono::DateTime::<chrono::Utc>::from_timestamp(d.as_secs() as i64, 0)
                        .map(|dt| dt.to_rfc3339())
                })
            })
            .flatten();

        items.push(RawItemJson {
            source_type: "text",
            source_id: path.to_string_lossy().to_string(),
            content: content.to_string(),
            timestamp,
            thread_id: None,
            author_identifier: None,
            is_user: Some(true), // files in user's own dirs are theirs by default
            subject: path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string()),
            metadata,
        });
    }

    Ok(items)
}

/// Strip RTF control words. Crude but adequate — gives us the text content
/// without needing a full RTF parser.
fn strip_rtf(s: &str) -> String {
    let mut out = String::with_capacity(s.len() / 2);
    let mut chars = s.chars().peekable();
    let mut depth = 0u32;
    while let Some(c) = chars.next() {
        match c {
            '{' => depth = depth.saturating_add(1),
            '}' => depth = depth.saturating_sub(1),
            '\\' => {
                // Skip the control word (letters) and any optional numeric arg.
                while let Some(&nc) = chars.peek() {
                    if nc.is_ascii_alphabetic() {
                        chars.next();
                    } else {
                        break;
                    }
                }
                while let Some(&nc) = chars.peek() {
                    if nc.is_ascii_digit() || nc == '-' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                if let Some(&' ') = chars.peek() {
                    chars.next();
                }
            }
            _ if depth > 0 => out.push(c),
            _ => out.push(c),
        }
    }
    // Collapse runs of whitespace.
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}
