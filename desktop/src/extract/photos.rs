//! Apple Photos extractor (metadata only — no pixel data, no OCR).
//!
//! Reads `~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite`.
//!
//! What we pull and why:
//!   * **Per-asset metadata**: timestamp, GPS, place name, person count,
//!     favorite flag. Lets us identify trips (date+place clusters),
//!     gatherings (multi-person photos at one place on one date),
//!     "memorable" moments (favorites).
//!   * **Detected persons** (ZPERSON table when present): face IDs Apple
//!     has clustered, with their user-assigned names. These resolve
//!     against Contacts during synthesis.
//!   * **Places** are aggregated from per-asset lat/lon — cluster within
//!     ~5 km on the same day = one Place visit.
//!
//! We never read pixels. We never OCR. The graph stays metadata-only;
//! the photos themselves stay in the library.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Event, Person, Place};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, NaiveDate, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "photos";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Pictures/Photos Library.photoslibrary/database/Photos.sqlite");
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
            skip_reason: Some("Photos library not present".into()),
            ..Default::default()
        });
    }

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
                skip_reason: Some("Photos sqlite locked or unreadable (TCC?)".into()),
                ..Default::default()
            });
        }
    };

    // Photos uses ZASSET as the main asset table; ZPLACEINFO for reverse
    // geocoded place; ZPERSON / Z_2ASSETS for face clusters. Column names
    // shift across macOS versions — we discover by introspection.
    let cols = list_columns(&conn, "ZASSET").unwrap_or_default();
    let date_col = pick_col(&cols, &["ZDATECREATED", "ZADDEDDATE", "ZSORTTOKEN"]);
    let lat_col  = pick_col(&cols, &["ZLATITUDE"]);
    let lon_col  = pick_col(&cols, &["ZLONGITUDE"]);
    let fav_col  = pick_col(&cols, &["ZFAVORITE"]);
    if date_col.is_none() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("ZASSET schema unrecognized".into()),
            ..Default::default()
        });
    }

    let sql = format!(
        "SELECT Z_PK, {date_c}, {lat_c}, {lon_c}, {fav_c} FROM ZASSET",
        date_c = date_col.unwrap(),
        lat_c  = lat_col.unwrap_or("NULL"),
        lon_c  = lon_col.unwrap_or("NULL"),
        fav_c  = fav_col.unwrap_or("0"),
    );

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let pk: i64 = row.get(0)?;
        let ts: Option<f64> = row.get(1).ok().flatten();
        let lat: Option<f64> = row.get(2).ok().flatten();
        let lon: Option<f64> = row.get(3).ok().flatten();
        let fav: Option<i64> = row.get(4).ok().flatten();
        Ok((pk, ts, lat, lon, fav))
    })?;

    // Aggregate by (date, place-bucket).
    #[derive(Default)]
    struct DayPlace {
        first: Option<DateTime<Utc>>,
        last:  Option<DateTime<Utc>>,
        count: u64,
        favorites: u64,
        lat: f64,
        lon: f64,
        lat_sum: f64,
        lon_sum: f64,
    }
    let mut clusters: HashMap<(NaiveDate, i32, i32), DayPlace> = HashMap::new();
    let mut total = 0u64;
    for r in rows {
        let (_pk, ts, lat, lon, fav) = r?;
        let Some(dt) = ts.and_then(apple_seconds_to_utc) else { continue };
        total += 1;
        let date = dt.date_naive();
        let (lat_b, lon_b) = match (lat, lon) {
            (Some(la), Some(lo)) => (((la * 20.0).round() as i32), ((lo * 20.0).round() as i32)),
            _ => (i32::MIN, i32::MIN),
        };
        let key = (date, lat_b, lon_b);
        let c = clusters.entry(key).or_default();
        c.count += 1;
        if matches!(fav, Some(1)) { c.favorites += 1; }
        if c.first.map(|f| dt < f).unwrap_or(true) { c.first = Some(dt); }
        if c.last.map(|l| dt > l).unwrap_or(true)  { c.last  = Some(dt); }
        if let (Some(la), Some(lo)) = (lat, lon) {
            c.lat_sum += la; c.lon_sum += lo;
            c.lat = c.lat_sum / c.count as f64;
            c.lon = c.lon_sum / c.count as f64;
        }
    }

    let mut places: Vec<Place> = Vec::new();
    let mut events: Vec<Event> = Vec::new();

    for ((date, lat_b, lon_b), c) in clusters {
        if c.count < 2 { continue; } // single-photo days aren't a place
        let has_geo = lat_b != i32::MIN && (c.lat.abs() <= 90.0) && (c.lon.abs() <= 180.0)
            && !(c.lat == 0.0 && c.lon == 0.0);
        let place_label = if has_geo {
            format!("({:.3}, {:.3})", c.lat, c.lon)
        } else {
            format!("Photo cluster — {date}")
        };
        let pid = stable_id(&[
            "photos_place",
            &lat_b.to_string(),
            &lon_b.to_string(),
        ]);
        places.push(Place {
            id: pid.clone(),
            label: place_label.clone(),
            lat: if has_geo { Some(c.lat) } else { None },
            lon: if has_geo { Some(c.lon) } else { None },
            kind: Some(if c.count > 30 { "frequent".into() } else { "trip".into() }),
            visit_count: 1,
            first_seen: c.first,
            last_seen: c.last,
            sources: vec![SOURCE.into()],
        });

        let event_label = if c.favorites > 0 {
            format!("Photo moment — {} ({} favorites)", date, c.favorites)
        } else {
            format!("Photo cluster — {} ({})", date, c.count)
        };
        let eid = stable_id(&["photos_cluster", &date.to_string(), &lat_b.to_string(), &lon_b.to_string()]);
        events.push(Event {
            id: eid,
            title: event_label,
            start: c.first,
            end: c.last,
            kind: Some(if c.count > 20 { "trip".into() } else { "gathering".into() }),
            place_id: Some(pid),
            attendee_ids: vec![],
            notes: None,
            sources: vec![SOURCE.into()],
        });
    }

    // Faces — best effort, schema-tolerant.
    let mut faces: Vec<Person> = Vec::new();
    if let Ok(mut stmt) = conn.prepare(
        "SELECT Z_PK, ZFULLNAME FROM ZPERSON \
         WHERE ZFULLNAME IS NOT NULL AND LENGTH(TRIM(ZFULLNAME)) > 0",
    ) {
        if let Ok(iter) = stmt.query_map([], |row| {
            let pk: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            Ok((pk, name))
        }) {
            for r in iter.flatten() {
                let (pk, name) = r;
                // Belt-and-suspenders — also reject names that are only
                // whitespace at the Rust layer, since Apple's collation
                // can sometimes admit zero-width characters through the
                // LENGTH check.
                let trimmed = name.trim().to_string();
                if trimmed.is_empty() { continue; }
                let pid = stable_id(&["photos_face", &pk.to_string()]);
                faces.push(Person {
                    id: pid,
                    display_name: Some(trimmed.clone()),
                    aliases: vec![trimmed],
                    phones: vec![],
                    emails: vec![],
                    relationship: None,
                    inferred_role: Some("photo_face".into()),
                    temperature: None,
                    channel_counts: HashMap::from([(
                        "photos".to_string(), 1u64,
                    )]),
                    first_seen: None,
                    last_seen: None,
                    organizations: vec![],
                    birthday: None,
                    sources: vec![SOURCE.into()],
                });
            }
        }
    }

    let n_p = places.len();
    let n_e = events.len();
    let n_f = faces.len();

    ctx.store.upsert_many(EntityKind::Place,  &places, |p| p.id.clone())?;
    ctx.store.upsert_many(EntityKind::Event,  &events, |e| e.id.clone())?;
    ctx.store.upsert_many(EntityKind::Person, &faces,  |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Place)?;
    ctx.store.flush_kind(EntityKind::Event)?;
    ctx.store.flush_kind(EntityKind::Person)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", total);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total,
        entities_written: (n_p + n_e + n_f) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
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

fn pick_col<'a>(cols: &'a [String], candidates: &[&'a str]) -> Option<&'a str> {
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
        "pmc-photos-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("Photos.sqlite");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied("Photos (grant Photos access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["sqlite-wal", "sqlite-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("Photos.{ext}")));
        }
    }
    Ok(dst)
}
