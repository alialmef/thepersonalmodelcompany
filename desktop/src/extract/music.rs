//! Apple Music / iTunes listening taste extractor.
//!
//! Reads `~/Music/Music/Music Library.musiclibrary/Application.musicdb`
//! (or the legacy `iTunes Music Library.xml` if present). We emit
//! TasteItem rows for: top tracks by play count, top artists, top
//! albums, plus podcasts when surfaced.
//!
//! Music taste isn't fluff. The voice you imagine in your head when you
//! write is shaped by what you listen to. We surface the top-N items
//! as evidence of register, mood, and what's currently in heavy
//! rotation.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, TasteItem};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "music";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Music/Music/Music Library.musiclibrary/Application.musicdb");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let path = default_db_path().ok_or_else(|| ExtractError::Other("HOME unset".into()))?;
    if !path.is_file() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Music library not present".into()),
            ..Default::default()
        });
    }

    // Modern Apple Music stores the library in `Library.musicdb` — a
    // proprietary binary format, NOT sqlite. Parsing it cleanly needs
    // MusicKit (Swift) or a dedicated reverse-engineered reader. We
    // ship a stub that detects the format and reports an honest skip
    // until we wire in a native MusicKit shim.
    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = match Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    ) {
        Ok(c) => c,
        Err(_) => {
            return Ok(ExtractSummary {
                source: SOURCE.into(),
                skipped: true,
                skip_reason: Some("Music library uses Apple's binary .musicdb — needs MusicKit shim".into()),
                ..Default::default()
            });
        }
    };

    // Apple keeps swapping the Music DB schema across versions, so we
    // discover. We look for a tracks-like table.
    let tracks_table = detect_tracks_table(&conn);
    let Some(table) = tracks_table else {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Music uses Apple binary .musicdb — MusicKit shim needed".into()),
            ..Default::default()
        });
    };

    // Discover relevant columns.
    let cols = list_columns(&conn, &table).unwrap_or_default();
    let name_col   = pick(&cols, &["name", "title", "ZTITLE", "ZNAME"]);
    let artist_col = pick(&cols, &["artist", "ZARTIST", "ZARTISTNAME"]);
    let album_col  = pick(&cols, &["album", "ZALBUM", "ZALBUMTITLE"]);
    let plays_col  = pick(&cols, &["play_count", "playcount", "ZPLAYCOUNT"]);
    let last_col   = pick(&cols, &["last_played_date", "ZLASTPLAYEDDATE", "ZLASTPLAYED"]);
    let kind_col   = pick(&cols, &["kind", "ZKIND"]);

    if name_col.is_none() || artist_col.is_none() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("required columns missing".into()),
            ..Default::default()
        });
    }

    let sql = format!(
        "SELECT {n}, {a}, {al}, {pc}, {lp}, {k} FROM {t}",
        n  = name_col.unwrap(),
        a  = artist_col.unwrap(),
        al = album_col.unwrap_or("NULL"),
        pc = plays_col.unwrap_or("0"),
        lp = last_col.unwrap_or("NULL"),
        k  = kind_col.unwrap_or("NULL"),
        t  = table,
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let name:   Option<String> = row.get(0).ok().flatten();
        let artist: Option<String> = row.get(1).ok().flatten();
        let album:  Option<String> = row.get(2).ok().flatten();
        let plays:  Option<i64>    = row.get(3).ok().flatten();
        let last:   Option<f64>    = row.get(4).ok().flatten();
        let kind:   Option<String> = row.get(5).ok().flatten();
        Ok((name, artist, album, plays, last, kind))
    })?;

    let mut track_items: Vec<TasteItem> = Vec::new();
    let mut artist_agg: HashMap<String, (u64, Option<DateTime<Utc>>)> = HashMap::new();
    let mut album_agg:  HashMap<(String, Option<String>), (u64, Option<DateTime<Utc>>)> = HashMap::new();
    let mut podcasts:   Vec<TasteItem> = Vec::new();

    let mut total = 0u64;
    for r in rows {
        let (name, artist, album, plays, last, kind) = r?;
        let (Some(name), Some(artist)) = (name, artist) else { continue };
        let plays = plays.unwrap_or(0).max(0) as u64;
        if plays == 0 { continue; }
        total += 1;
        let last_dt = last.and_then(apple_seconds_to_utc);

        let is_podcast = kind.as_deref().map(|k| k.to_lowercase().contains("podcast")).unwrap_or(false);
        let item = TasteItem {
            id: stable_id(&[if is_podcast { "podcast" } else { "track" }, &artist, &name]),
            kind: if is_podcast { "podcast".into() } else { "track".into() },
            name: name.clone(),
            creator: Some(artist.clone()),
            play_count: plays,
            last_played: last_dt,
            source: SOURCE.into(),
        };
        if is_podcast { podcasts.push(item); } else { track_items.push(item); }

        // Aggregate artists + albums.
        let (ac, al) = artist_agg.entry(artist.clone()).or_insert((0, None));
        *ac += plays;
        if last_dt.map(|t| al.map(|a| t > a).unwrap_or(true)).unwrap_or(false) {
            *al = last_dt;
        }
        let (bc, bl) = album_agg.entry((artist.clone(), album.clone())).or_insert((0, None));
        *bc += plays;
        if last_dt.map(|t| bl.map(|a| t > a).unwrap_or(true)).unwrap_or(false) {
            *bl = last_dt;
        }
    }

    // Keep top 100 tracks by play count to bound the graph size.
    track_items.sort_by(|a, b| b.play_count.cmp(&a.play_count));
    track_items.truncate(100);
    podcasts.sort_by(|a, b| b.play_count.cmp(&a.play_count));
    podcasts.truncate(50);

    let mut artists: Vec<TasteItem> = artist_agg.into_iter().map(|(name, (count, last))| TasteItem {
        id: stable_id(&["artist", &name]),
        kind: "artist".into(),
        name,
        creator: None,
        play_count: count,
        last_played: last,
        source: SOURCE.into(),
    }).collect();
    artists.sort_by(|a, b| b.play_count.cmp(&a.play_count));
    artists.truncate(50);

    let mut albums: Vec<TasteItem> = album_agg.into_iter().filter_map(|((artist, album), (count, last))| {
        let album = album?;
        Some(TasteItem {
            id: stable_id(&["album", &artist, &album]),
            kind: "album".into(),
            name: album,
            creator: Some(artist),
            play_count: count,
            last_played: last,
            source: SOURCE.into(),
        })
    }).collect();
    albums.sort_by(|a, b| b.play_count.cmp(&a.play_count));
    albums.truncate(50);

    let n = track_items.len() + artists.len() + albums.len() + podcasts.len();
    ctx.store.upsert_many(EntityKind::TasteItem, &track_items, |t| t.id.clone())?;
    ctx.store.upsert_many(EntityKind::TasteItem, &artists,     |t| t.id.clone())?;
    ctx.store.upsert_many(EntityKind::TasteItem, &albums,      |t| t.id.clone())?;
    ctx.store.upsert_many(EntityKind::TasteItem, &podcasts,    |t| t.id.clone())?;
    ctx.store.flush_kind(EntityKind::TasteItem)?;

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

fn detect_tracks_table(conn: &Connection) -> Option<String> {
    let names: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table'")
        .ok()?
        .query_map([], |row| row.get::<_, String>(0))
        .ok()?
        .flatten()
        .collect();
    for cand in ["tracks", "ZMUSICTRACK", "ZTRACK", "MZTRACK"] {
        if names.iter().any(|n| n.eq_ignore_ascii_case(cand)) {
            return Some(cand.to_string());
        }
    }
    // Fallback: any table whose name ends with "TRACK".
    names.into_iter().find(|n| n.to_uppercase().ends_with("TRACK"))
}

fn list_columns(conn: &Connection, table: &str) -> Result<Vec<String>, ExtractError> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let name: String = row.get(1)?;
        Ok(name)
    })?;
    Ok(rows.flatten().collect())
}

fn pick<'a>(cols: &'a [String], candidates: &[&'a str]) -> Option<&'a str> {
    for cand in candidates {
        if cols.iter().any(|c| c.eq_ignore_ascii_case(cand)) {
            return Some(cand);
        }
    }
    None
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    Some(epoch + Duration::seconds(s as i64))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-music-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("Application.musicdb");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied("Music (Full Disk Access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    Ok(dst)
}
