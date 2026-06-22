//! Voice Memos extractor (metadata only).
//!
//! Enumerates audio files in the Voice Memos library directories and
//! emits one `FileSignal` per recording. We capture filename, path,
//! mtime, and size — *not* audio content or transcripts.
//!
//! Whisper transcription is deferred to a later phase. When that
//! lands, the transcript becomes a separate entity (text content)
//! linked to the FileSignal via an Edge, so the user can independently
//! redact transcripts without losing the existence-of-recording signal.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, FileSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use std::path::PathBuf;

const SOURCE: &str = "voice_memos";

const REL_DIRS: &[&str] = &[
    // Catalina+ on-device library
    "Library/Application Support/com.apple.voicememos/Recordings",
    // iCloud-synced shared group container (universal Voice Memos)
    "Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings",
];

const AUDIO_EXTS: &[&str] = &["m4a", "caf", "wav", "mp3", "aac"];

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Ok(skipped("HOME unset"));
    };

    let mut signals: Vec<FileSignal> = Vec::new();
    let mut walked = 0u64;
    let mut any_dir_present = false;

    for rel in REL_DIRS {
        let dir = home.join(rel);
        if !dir.is_dir() {
            continue;
        }
        any_dir_present = true;
        let read = match std::fs::read_dir(&dir) {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return Err(ExtractError::PermissionDenied(
                    "Voice Memos (Full Disk Access)".into(),
                ));
            }
            Err(_) => continue,
        };
        for entry in read.flatten() {
            let path = entry.path();
            walked += 1;
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else { continue };
            let ext_lc = ext.to_lowercase();
            if !AUDIO_EXTS.contains(&ext_lc.as_str()) {
                continue;
            }

            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            // Voice Memos filenames embed the original recording date:
            //   "20141124 191401-45EFC2CD.m4a" → 2014-11-24 19:14:01
            // iCloud sync routinely bumps the file mtime, so the
            // on-disk mtime is unreliable for "when did this memo
            // happen." Prefer the filename-derived date; fall back to
            // mtime if the filename doesn't parse.
            let modified_from_name = parse_date_from_filename(name);
            let modified_from_mtime: Option<DateTime<Utc>> = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|d| {
                    DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos())
                });
            let modified = modified_from_name.or(modified_from_mtime);
            signals.push(FileSignal {
                id: stable_id(&["voice_memo", &path.to_string_lossy()]),
                path: path.to_string_lossy().to_string(),
                name: name.to_string(),
                extension: Some(ext_lc),
                modified,
                size_bytes: meta.len(),
                kind: Some("voice_memo".into()),
            });
        }
    }

    if !any_dir_present {
        return Ok(skipped("voice memos directory not present"));
    }

    let n = signals.len();
    ctx.store
        .upsert_many(EntityKind::FileSignal, &signals, |s| s.id.clone())?;
    ctx.store.flush_kind(EntityKind::FileSignal)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", walked);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: walked,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

/// Parse Voice Memos filename into a UTC datetime when possible.
/// Apple's Voice Memos app names files like:
///   "20141124 191401-45EFC2CD.m4a"
///   "20260615 114425.m4a"  (newer iCloud-synced layout, no UUID suffix)
/// The leading 14 digits are YYYYMMDD HHMMSS in *local time*, which
/// we treat as UTC for stable ordering — losing TZ accuracy is much
/// less destructive than losing the year (which is what the bumped
/// mtime does).
fn parse_date_from_filename(name: &str) -> Option<DateTime<Utc>> {
    use chrono::TimeZone;
    // Strip trailing extension and UUID suffix (if any). The format is:
    //   "YYYYMMDD HHMMSS[-<UUID>].<ext>"
    let stem = name.split('.').next()?;
    let head = stem.split('-').next()?;
    let mut chars = head.chars();
    // Must look like "YYYYMMDD HHMMSS"
    let mut digits = String::new();
    let mut saw_space = false;
    for c in chars.by_ref() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if c == ' ' && !saw_space {
            saw_space = true;
        } else {
            return None;
        }
    }
    if digits.len() != 14 || !saw_space {
        return None;
    }
    let year: i32 = digits[0..4].parse().ok()?;
    let month: u32 = digits[4..6].parse().ok()?;
    let day: u32 = digits[6..8].parse().ok()?;
    let hour: u32 = digits[8..10].parse().ok()?;
    let minute: u32 = digits[10..12].parse().ok()?;
    let second: u32 = digits[12..14].parse().ok()?;
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second).single()
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
