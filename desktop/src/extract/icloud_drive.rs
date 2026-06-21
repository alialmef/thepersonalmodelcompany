//! iCloud Drive walker.
//!
//! Reads `~/Library/Mobile Documents/com~apple~CloudDocs` — every file
//! the user has saved to iCloud Drive. Filenames + paths + mtimes are
//! enormous signal even without opening the files: bank statements,
//! contracts, receipts, voice recordings, photos all carry intent in
//! their names.
//!
//! We never read file *contents* here (PDFs, docs would need parsers
//! we don't ship). A separate Whisper / OCR / parser pass can opt-in
//! to read content for specific files later.
//!
//! What we capture per file:
//!   - path, name, extension, size, mtime
//!   - kind classification (document, code, image, archive, etc.)
//!   - whether it's a "highlight" — flagged with a heuristic that
//!     catches names like "PASSPORT.pdf", "rent statement.pdf",
//!     "Dad's Interview.m4a"
//!
//! Each top-level directory in iCloud Drive becomes a Project so the
//! agent can reason about your "filing cabinet" structure (you have
//! a Documents folder, a PDFs folder, etc.).

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, FileSignal, Project};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};

const SOURCE: &str = "icloud_drive";

const ROOT_REL: &str = "Library/Mobile Documents/com~apple~CloudDocs";

// Skip these top-level dirs — they're either app-managed or noise.
const SKIP_TOP_DIRS: &[&str] = &[
    ".com.apple.mobile_container_manager.metadata.plist",
    ".DS_Store",
];

// File walk caps so a 30k-file iCloud doesn't take a minute. We sort
// dirs by recency-of-modification first so the file budget surfaces
// recent activity rather than 5-year-old archives.
const FILE_LIMIT: usize = 5_000;
const PER_DIR_RECURSE_CAP: usize = 800;

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Ok(skipped("HOME unset"));
    };
    let root = home.join(ROOT_REL);
    if !root.is_dir() {
        return Ok(skipped("iCloud Drive root not present"));
    }

    let mut files: Vec<FileSignal> = Vec::new();
    let mut projects: Vec<Project> = Vec::new();
    let mut walked = 0u64;

    // First pass — enumerate top-level dirs as Projects so the agent
    // can see "the filing cabinet" structure even when the per-file
    // budget gets exhausted.
    let read = match std::fs::read_dir(&root) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "iCloud Drive (Full Disk Access)".into(),
            ));
        }
        Err(_) => return Ok(skipped("couldn't read iCloud Drive root")),
    };

    let mut top_entries: Vec<(PathBuf, std::fs::Metadata)> = Vec::new();
    for entry in read.flatten() {
        let p = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if SKIP_TOP_DIRS.contains(&name) { continue; }
        top_entries.push((p, meta));
    }
    // Sort by mtime desc so the recent stuff gets the file budget
    top_entries.sort_by_key(|(_, m)| {
        std::cmp::Reverse(m.modified().ok().map(|t| t.duration_since(std::time::UNIX_EPOCH).ok()))
    });

    for (p, meta) in &top_entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        if meta.is_dir() {
            // Project entity for the dir
            let last_activity: Option<DateTime<Utc>> = meta.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos()));
            projects.push(Project {
                id: stable_id(&[SOURCE, "dir", name.as_str()]),
                name: format!("iCloud · {}", name),
                state: Some("active".into()),
                people_ids: Vec::new(),
                last_activity,
                summary: Some(format!("iCloud Drive folder: {}", name)),
                sources: vec![SOURCE.into()],
            });
        }
    }

    // Second pass — walk the tree, capturing FileSignals up to FILE_LIMIT.
    // Walk top-level files (loose under root) first, then descend into
    // dirs in mtime-desc order.
    for (p, meta) in &top_entries {
        if files.len() >= FILE_LIMIT { break; }
        if meta.is_file() {
            push_file(p, meta, &mut files, &mut walked);
        }
    }
    for (p, meta) in &top_entries {
        if files.len() >= FILE_LIMIT { break; }
        if meta.is_dir() {
            walk_dir(p, &mut files, &mut walked, 0);
        }
    }

    let n_files = files.len();
    let n_projects = projects.len();
    ctx.store.upsert_many(EntityKind::FileSignal, &files, |f| f.id.clone())?;
    ctx.store.upsert_many(EntityKind::Project, &projects, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::FileSignal)?;
    ctx.store.flush_kind(EntityKind::Project)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", walked);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: walked,
        entities_written: (n_files + n_projects) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn walk_dir(
    dir: &Path,
    files: &mut Vec<FileSignal>,
    walked: &mut u64,
    depth: u32,
) {
    if depth > 6 { return; }
    if files.len() >= FILE_LIMIT { return; }

    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return,
    };
    let mut entries: Vec<(PathBuf, std::fs::Metadata)> = Vec::new();
    for entry in read.flatten() {
        let p = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') { continue; }
        entries.push((p, meta));
        if entries.len() > PER_DIR_RECURSE_CAP { break; }
    }
    // mtime-desc so recent files surface first
    entries.sort_by_key(|(_, m)| {
        std::cmp::Reverse(m.modified().ok().map(|t| t.duration_since(std::time::UNIX_EPOCH).ok()))
    });

    for (p, meta) in &entries {
        if files.len() >= FILE_LIMIT { break; }
        if meta.is_file() {
            push_file(p, meta, files, walked);
        }
    }
    for (p, meta) in &entries {
        if files.len() >= FILE_LIMIT { break; }
        if meta.is_dir() {
            walk_dir(p, files, walked, depth + 1);
        }
    }
}

fn push_file(
    p: &Path,
    meta: &std::fs::Metadata,
    out: &mut Vec<FileSignal>,
    walked: &mut u64,
) {
    *walked += 1;
    let Some(name) = p.file_name().and_then(|n| n.to_str()) else { return };
    let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase());
    let kind = classify(name, ext.as_deref());
    let modified: Option<DateTime<Utc>> = meta.modified().ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .and_then(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos()));
    out.push(FileSignal {
        id: stable_id(&[SOURCE, &p.to_string_lossy()]),
        path: p.to_string_lossy().to_string(),
        name: name.to_string(),
        extension: ext,
        modified,
        size_bytes: meta.len(),
        kind: Some(kind),
    });
}

fn classify(name: &str, ext: Option<&str>) -> String {
    let lower = name.to_lowercase();
    // High-priority semantic flags first
    if lower.contains("passport") { return "passport".into(); }
    if lower.contains("license") || lower.contains("licence") { return "license".into(); }
    if lower.contains("receipt") { return "receipt".into(); }
    if lower.contains("rent") { return "rent".into(); }
    if lower.contains("statement") && (lower.contains("bank") || lower.contains("rent") || lower.contains("card")) {
        return "statement".into();
    }
    if lower.contains("invoice") { return "invoice".into(); }
    if lower.contains("contract") || lower.contains("sow") || lower.contains("nda") {
        return "contract".into();
    }
    if lower.contains("interview") { return "interview".into(); }
    if lower.contains("resume") || lower.contains("cv") || lower.contains("bio") {
        return "resume".into();
    }

    match ext {
        Some("pdf") => "document".into(),
        Some("md") | Some("txt") | Some("rtf") | Some("pages") | Some("doc") | Some("docx") => "document".into(),
        Some("rs") | Some("ts") | Some("tsx") | Some("js") | Some("jsx") | Some("py") | Some("go")
        | Some("rb") | Some("swift") | Some("java") | Some("c") | Some("h") | Some("cpp") | Some("cs") => "code".into(),
        Some("jpg") | Some("jpeg") | Some("png") | Some("heic") | Some("gif") | Some("webp") | Some("tiff") => "image".into(),
        Some("mov") | Some("mp4") | Some("m4v") | Some("avi") | Some("webm") => "video".into(),
        Some("mp3") | Some("m4a") | Some("wav") | Some("flac") | Some("caf") => "audio".into(),
        Some("sketch") | Some("fig") | Some("psd") | Some("ai") | Some("key") => "design".into(),
        Some("zip") | Some("tar") | Some("gz") | Some("tgz") | Some("7z") | Some("rar") => "archive".into(),
        Some("csv") | Some("tsv") | Some("xlsx") | Some("numbers") => "spreadsheet".into(),
        _ => "other".into(),
    }
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
