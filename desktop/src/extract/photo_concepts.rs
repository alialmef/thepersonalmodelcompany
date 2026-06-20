//! Photos.app concept index extractor.
//!
//! Reads `~/Pictures/Photos Library.photoslibrary/database/search/psi.sqlite`
//! — Apple's machine-learned index over every photo in the library.
//! The `groups` table holds recognized concepts (scenes, objects,
//! OCR'd text, place names, captions) categorized by an integer
//! `category` column. Each `category` corresponds to a different
//! recognition pipeline:
//!
//!   1, 2, 3       — street / locality / region names recognized in the photo
//!   1500          — scene/object concepts (acoustic guitar, hawk, advertising)
//!   1501          — broad people / clothing classes (Humans, Apparel)
//!   1203          — short OCR'd tokens (mostly digits — low signal)
//!   2100          — internal filenames (junk, filtered out)
//!   1400          — date / size metadata (filtered out)
//!   ... many more
//!
//! For V1 we surface the categories that carry *content*: scenes
//! (1500), people-classes (1501), and place names (1, 2, 3). Each
//! becomes a Theme entity in the graph — the synthesis layer can later
//! group them ("you take a lot of photos of hawks").
//!
//! NOTE: This is the agent learning about you from your *visual* life
//! without anything you typed. The richest single signal a Mac
//! produces, after iMessage.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Theme};
use crate::graph::store::stable_id;
use chrono::Utc;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "photo_concepts";

fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Pictures/Photos Library.photoslibrary/database/search/psi.sqlite");
        p
    })
}

/// Apple PSITokenizer categories we capture. Each becomes a labelled
/// Theme source — the label format makes them legible in the graph
/// without us having to ship an Apple-internal category dictionary.
const KEPT_CATEGORIES: &[(i64, &str)] = &[
    (1500, "scene"),       // "Acoustic Guitar", "Advertising", "Hawk"
    (1501, "people_class"),// "Humans", "Apparel"
    (1, "place_street"),   // "14 St/8 Av", "168 St"
    (2, "place_locality"), // "10th Ave", "12th St NE"
    (3, "place_region"),   // "Al Amrania", "Al Haram Al Sharif"
];

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_db_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped("Photos psi.sqlite not present"));
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    // Build (category, concept) → count by joining groups → ga → assets
    // for an asset count per concept. This tells us "you have 47 photos
    // of guitars" instead of just "you have a photo of a guitar".
    let mut by_concept: HashMap<(i64, String), u64> = HashMap::new();
    let mut by_concept_first_seen: HashMap<(i64, String), i64> = HashMap::new();
    let mut by_concept_last_seen: HashMap<(i64, String), i64> = HashMap::new();
    let mut scanned = 0u64;

    let sql = r#"
        SELECT g.category, g.content_string,
               MIN(a.creationDate) AS first_seen,
               MAX(a.creationDate) AS last_seen,
               COUNT(*) AS hits
        FROM groups g
        JOIN ga    ON ga.groupid = g.rowid
        JOIN assets a ON a.rowid = ga.assetid
        WHERE g.category IN (1, 2, 3, 1500, 1501)
          AND g.content_string IS NOT NULL
          AND LENGTH(g.content_string) > 0
        GROUP BY g.category, g.content_string
    "#;
    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(_) => return Ok(skipped("psi.sqlite schema unexpected")),
    };
    let rows = stmt.query_map([], |row| {
        let cat: i64 = row.get(0)?;
        let content: String = row.get(1)?;
        let first: Option<i64> = row.get(2).ok();
        let last: Option<i64> = row.get(3).ok();
        let hits: i64 = row.get(4)?;
        Ok((cat, content, first, last, hits))
    })?;

    for r in rows {
        let Ok((cat, content, first, last, hits)) = r else { continue };
        scanned += 1;
        let key = (cat, content.clone());
        *by_concept.entry(key.clone()).or_insert(0) += hits.max(0) as u64;
        if let Some(f) = first {
            let e = by_concept_first_seen.entry(key.clone()).or_insert(i64::MAX);
            if f < *e { *e = f; }
        }
        if let Some(l) = last {
            let e = by_concept_last_seen.entry(key).or_insert(i64::MIN);
            if l > *e { *e = l; }
        }
    }

    // Build Theme entities. We drop concepts with fewer than 2 hits —
    // a single photo of something isn't a theme; a dozen is.
    let mut themes: Vec<Theme> = Vec::new();
    let category_label: HashMap<i64, &str> = KEPT_CATEGORIES.iter().copied().collect();
    for ((cat, content), hits) in &by_concept {
        if *hits < 2 { continue; }
        let category = category_label.get(cat).copied().unwrap_or("photo_concept");
        let label = content.clone();
        let first_unix = by_concept_first_seen
            .get(&(*cat, content.clone()))
            .copied()
            .unwrap_or(0);
        let last_unix = by_concept_last_seen
            .get(&(*cat, content.clone()))
            .copied()
            .unwrap_or(0);
        // Photos.app stores creationDate as Apple absolute time (sec
        // since 2001-01-01). Convert.
        let first = apple_unix_to_utc(first_unix);
        let last = apple_unix_to_utc(last_unix);
        themes.push(Theme {
            id: stable_id(&[SOURCE, category, &label]),
            label: label.clone(),
            keywords: vec![category.to_string()],
            mentions_30d: 0,  // we don't have a 30d window from psi.sqlite
            mentions_180d: *hits,
            trajectory: None,
            first_seen: first,
            last_seen: last,
            source_kinds: vec![SOURCE.into()],
            example_quotes: Vec::new(),
        });
    }

    let n = themes.len();
    ctx.store.upsert_many(EntityKind::Theme, &themes, |t| t.id.clone())?;
    ctx.store.flush_kind(EntityKind::Theme)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", scanned);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: scanned,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn apple_unix_to_utc(secs: i64) -> Option<chrono::DateTime<Utc>> {
    if secs <= 0 { return None; }
    // Apple absolute time = 2001-01-01 epoch
    use chrono::TimeZone;
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    Some(epoch + chrono::Duration::seconds(secs))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-psi-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("psi.sqlite");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Photos psi.sqlite (Full Disk Access)".into(),
            ));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["sqlite-wal", "sqlite-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("psi.{ext}")));
        }
    }
    Ok(dst)
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
