//! Significant Locations extractor.
//!
//! macOS's "Significant Locations" feature (sister to the iOS one) lives
//! in `~/Library/Caches/com.apple.routined/Cache.sqlite`. It records
//! every cluster of meaningful presence — home, work, regular coffee
//! shop, that gym you keep going to. Each visit has a start + end
//! timestamp and a lat/long.
//!
//! We aggregate visits per location cluster (rounded to 4 decimal
//! degrees ≈ 11m precision) into Place entities with frequency-by-
//! window and the last-seen timestamp. No individual visit timestamps
//! are persisted.
//!
//! The actual table layout varies a bit across macOS versions; this
//! extractor probes a few candidate queries and falls back to skipping
//! rather than failing the whole ingest if the schema doesn't match.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Place};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Timelike, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "locations";

fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Caches/com.apple.routined/Cache.sqlite");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_db_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped(
            "Significant Locations cache not present (Location Services may be disabled)",
        ));
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    // The routined Cache.sqlite has shifted shape a few times. Probe
    // for each known table name in order of recency. Each variant has
    // its own column names, so we keep the queries explicit per shape.
    let rows = read_visits(&conn);
    if rows.is_empty() {
        return Ok(skipped("routined cache schema unknown or empty"));
    }

    let now = Utc::now();
    let cutoff_30 = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        visits_30d: u64,
        visits_180d: u64,
        total_minutes_180d: u64,
        last: Option<DateTime<Utc>>,
        by_hour: [u64; 24],
        sample_lat: f64,
        sample_lon: f64,
    }
    let mut by_cluster: HashMap<(i64, i64), Agg> = HashMap::new();
    let mut total = 0u64;

    for v in &rows {
        if v.end < cutoff_180 {
            continue;
        }
        total += 1;
        // Round to ~11m precision so clusters collapse.
        let key = (
            (v.lat * 10_000.0).round() as i64,
            (v.lon * 10_000.0).round() as i64,
        );
        let dur_min = v
            .end
            .signed_duration_since(v.start)
            .num_minutes()
            .max(0) as u64;
        let h = v.start.hour() as usize;
        let agg = by_cluster.entry(key).or_insert(Agg {
            visits_30d: 0,
            visits_180d: 0,
            total_minutes_180d: 0,
            last: None,
            by_hour: [0; 24],
            sample_lat: v.lat,
            sample_lon: v.lon,
        });
        agg.visits_180d += 1;
        agg.total_minutes_180d += dur_min;
        if v.end >= cutoff_30 {
            agg.visits_30d += 1;
        }
        if agg.last.map(|l| v.end > l).unwrap_or(true) {
            agg.last = Some(v.end);
        }
        if h < 24 {
            agg.by_hour[h] = agg.by_hour[h].saturating_add(1);
        }
    }

    // Drop clusters seen only once over 180 days — those are flyovers,
    // not "places."
    let mut places: Vec<Place> = Vec::with_capacity(by_cluster.len());
    for ((klat, klon), agg) in by_cluster {
        if agg.visits_180d < 2 {
            continue;
        }
        let label = label_for(klat, klon);
        places.push(Place {
            id: stable_id(&[
                "significant_location",
                &format!("{},{}", klat, klon),
            ]),
            label,
            lat: Some(agg.sample_lat),
            lon: Some(agg.sample_lon),
            kind: Some(infer_place_kind(&agg.by_hour, agg.total_minutes_180d)),
            // We persist trailing-180d total here. The synthesis layer
            // can recompute 30d windows from raw audit events when it
            // needs to; storing both was overkill in the JSONL row.
            visit_count: agg.visits_180d,
            first_seen: None,
            last_seen: agg.last,
            sources: vec!["significant_locations".into()],
        });
    }

    let n = places.len();
    ctx.store
        .upsert_many(EntityKind::Place, &places, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Place)?;

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

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct VisitRow {
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    lat: f64,
    lon: f64,
}

fn read_visits(conn: &Connection) -> Vec<VisitRow> {
    // Try shapes in newest-first order. Each query returns
    // (start_seconds_since_apple_epoch, end_seconds_since_apple_epoch,
    //  lat, lon). The probe is wrapped in defensive option chains so a
    // schema shift just rolls to the next candidate rather than
    // erroring the whole extractor.
    const CANDIDATES: &[&str] = &[
        "SELECT ZSTART, ZEND, ZLATITUDE, ZLONGITUDE FROM ZRTVISITMO",
        "SELECT ZENTRYDATE, ZEXITDATE, ZLATITUDE, ZLONGITUDE FROM ZRTVISITMO",
        // older naming
        "SELECT ZARRIVALDATE, ZDEPARTUREDATE, ZLATITUDE, ZLONGITUDE FROM ZRTVISIT",
        // iCloud-synced location fingerprints sometimes live here
        "SELECT ZARRIVALDATE, ZDEPARTUREDATE, ZLATITUDE, ZLONGITUDE FROM ZRTLOCATIONMO",
    ];
    for sql in CANDIDATES {
        if let Ok(mut stmt) = conn.prepare(sql) {
            let it = stmt.query_map([], |row| {
                let s: f64 = row.get(0)?;
                let e: f64 = row.get(1)?;
                let lat: f64 = row.get(2)?;
                let lon: f64 = row.get(3)?;
                Ok((s, e, lat, lon))
            });
            if let Ok(rows) = it {
                let mut out: Vec<VisitRow> = Vec::new();
                for r in rows.flatten() {
                    let (s, e, lat, lon) = r;
                    if !(lat.is_finite() && lon.is_finite()) { continue; }
                    let (Some(sdt), Some(edt)) = (apple_seconds_to_utc(s), apple_seconds_to_utc(e))
                    else { continue };
                    if edt <= sdt { continue; }
                    out.push(VisitRow { start: sdt, end: edt, lat, lon });
                }
                if !out.is_empty() {
                    return out;
                }
            }
        }
    }
    Vec::new()
}

fn label_for(klat: i64, klon: i64) -> String {
    // We don't reverse-geocode at this layer — the synthesis pass
    // upstairs joins these to Photos clusters and Calendar locations
    // for naming. For raw extraction, surface a stable lat/lon tag.
    format!("place@{:.4},{:.4}", klat as f64 / 10_000.0, klon as f64 / 10_000.0)
}

fn infer_place_kind(by_hour: &[u64; 24], total_minutes_180d: u64) -> String {
    // Quick heuristic: when you're mostly there overnight → home;
    // mostly during business hours → work; otherwise → recurring spot.
    if total_minutes_180d == 0 {
        return "place".into();
    }
    let night_minutes: u64 = (0..7).chain(22..24).map(|h| by_hour[h]).sum();
    let work_minutes: u64 = (9..18).map(|h| by_hour[h]).sum();
    let total: u64 = by_hour.iter().sum();
    if total == 0 {
        return "place".into();
    }
    if night_minutes * 2 > total {
        return "home".into();
    }
    if work_minutes * 2 > total {
        return "work".into();
    }
    "recurring".into()
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    if !s.is_finite() {
        return None;
    }
    Some(epoch + Duration::seconds(s as i64))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-routined-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("Cache.sqlite");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Significant Locations (Full Disk Access)".into(),
            ));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["sqlite-wal", "sqlite-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("Cache.{ext}")));
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
