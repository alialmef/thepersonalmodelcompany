//! Apple Calendar.app extractor.
//!
//! Reads `~/Library/Group Containers/group.com.apple.calendar/Calendar.sqlitedb`.
//!
//! Schema (CalendarStore SQLite, simplified):
//!   * `CalendarItem`  — event rows (start_date, end_date, summary,
//!     location_id, calendar_id, status)
//!   * `Calendar`      — calendar metadata (title, color)
//!   * `Location`      — geo (title, address)
//!   * `Participant`   — attendees per item (email, role)
//!
//! We pull both past and future events, classify each into a coarse
//! `kind`, attach a `Place` when location resolves, and write
//! Participant emails as Person hints for synthesis to merge with
//! Contacts.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Event, Person, Place};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "calendar";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Group Containers/group.com.apple.calendar/Calendar.sqlitedb");
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
            skip_reason: Some("Calendar.sqlitedb not present".into()),
            ..Default::default()
        });
    }

    let snapshot = snapshot_db(&path, "calendar")?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    let locations = load_locations(&conn).unwrap_or_default();
    let participants = load_participants(&conn).unwrap_or_default();

    // CalendarItem stores start/end as Mac absolute (seconds since 2001).
    // status: 0 = none, 1 = canceled, 2 = confirmed, 3 = tentative.
    let mut stmt = match conn.prepare(
        r#"
        SELECT
            ROWID,
            summary,
            start_date,
            end_date,
            location_id,
            status
        FROM CalendarItem
        WHERE summary IS NOT NULL
        "#,
    ) {
        Ok(s) => s,
        Err(_) => {
            return Ok(ExtractSummary {
                source: SOURCE.into(),
                skipped: true,
                skip_reason: Some("CalendarItem schema unexpected".into()),
                ..Default::default()
            });
        }
    };

    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let summary: String = row.get(1)?;
        let start: Option<f64> = row.get(2)?;
        let end: Option<f64> = row.get(3)?;
        let location_id: Option<i64> = row.get(4)?;
        let status: Option<i64> = row.get(5)?;
        Ok((id, summary, start, end, location_id, status))
    })?;

    let mut events: Vec<Event> = Vec::new();
    let mut places: Vec<Place> = Vec::new();
    let mut people: Vec<Person> = Vec::new();

    for r in rows {
        let (id, summary, start, end, loc_id, status) = r?;
        if matches!(status, Some(1)) { continue; } // canceled
        let start_dt = start.and_then(apple_seconds_to_utc);
        let end_dt = end.and_then(apple_seconds_to_utc);

        let place_id = if let Some(lid) = loc_id {
            if let Some(loc) = locations.get(&lid) {
                let pid = stable_id(&["calendar_location", &lid.to_string()]);
                places.push(Place {
                    id: pid.clone(),
                    label: loc.title.clone(),
                    lat: loc.lat,
                    lon: loc.lon,
                    kind: Some("venue".into()),
                    visit_count: 1,
                    first_seen: start_dt,
                    last_seen: start_dt,
                    sources: vec![SOURCE.into()],
                });
                Some(pid)
            } else { None }
        } else { None };

        let attendee_emails = participants.get(&id).cloned().unwrap_or_default();
        let mut attendee_ids = Vec::new();
        for email in &attendee_emails {
            let pid = stable_id(&["calendar_participant", email]);
            attendee_ids.push(pid.clone());
            people.push(Person {
                id: pid,
                display_name: None,
                aliases: vec![email.clone()],
                phones: vec![],
                emails: vec![email.clone()],
                relationship: None,
                inferred_role: Some("calendar_attendee".into()),
                temperature: None,
                channel_counts: std::collections::HashMap::from([(
                    "calendar".to_string(), 1u64,
                )]),
                first_seen: start_dt,
                last_seen: start_dt,
                organizations: vec![],
                birthday: None,
                sources: vec![SOURCE.into()],
            });
        }

        let kind = classify_event(&summary, attendee_emails.len(), start_dt, end_dt);
        let event_id = stable_id(&["calendar_event", &id.to_string()]);
        events.push(Event {
            id: event_id,
            title: summary,
            start: start_dt,
            end: end_dt,
            kind: Some(kind),
            place_id,
            attendee_ids,
            notes: None,
            sources: vec![SOURCE.into()],
        });
    }

    let n_events = events.len();
    let n_places = places.len();
    let n_people = people.len();

    ctx.store.upsert_many(EntityKind::Event,  &events, |e| e.id.clone())?;
    ctx.store.upsert_many(EntityKind::Place,  &places, |p| p.id.clone())?;
    ctx.store.upsert_many(EntityKind::Person, &people, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Event)?;
    ctx.store.flush_kind(EntityKind::Place)?;
    ctx.store.flush_kind(EntityKind::Person)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", n_events as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: n_events as u64,
        entities_written: (n_events + n_places + n_people) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

#[derive(Default, Clone)]
struct LocationRow {
    title: String,
    lat: Option<f64>,
    lon: Option<f64>,
}

fn load_locations(conn: &Connection) -> Result<HashMap<i64, LocationRow>, ExtractError> {
    let stmt_res = conn.prepare("SELECT ROWID, title, latitude, longitude FROM Location");
    let mut stmt = match stmt_res {
        Ok(s) => s,
        Err(_) => return Ok(HashMap::new()),
    };
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let title: Option<String> = row.get(1)?;
        let lat: Option<f64> = row.get(2)?;
        let lon: Option<f64> = row.get(3)?;
        Ok((id, title.unwrap_or_default(), lat, lon))
    })?;
    let mut out = HashMap::new();
    for r in rows {
        let (id, title, lat, lon) = r?;
        if !title.is_empty() {
            out.insert(id, LocationRow { title, lat, lon });
        }
    }
    Ok(out)
}

fn load_participants(conn: &Connection) -> Result<HashMap<i64, Vec<String>>, ExtractError> {
    let stmt_res = conn.prepare(
        "SELECT owner_id, email FROM Participant WHERE owner_id IS NOT NULL AND email IS NOT NULL",
    );
    let mut stmt = match stmt_res {
        Ok(s) => s,
        Err(_) => return Ok(HashMap::new()),
    };
    let rows = stmt.query_map([], |row| {
        let owner: i64 = row.get(0)?;
        let email: String = row.get(1)?;
        Ok((owner, email.to_lowercase()))
    })?;
    let mut out: HashMap<i64, Vec<String>> = HashMap::new();
    for r in rows {
        let (owner, email) = r?;
        out.entry(owner).or_default().push(email);
    }
    Ok(out)
}

fn classify_event(
    summary: &str,
    attendee_count: usize,
    start: Option<DateTime<Utc>>,
    end: Option<DateTime<Utc>>,
) -> String {
    let lower = summary.to_lowercase();
    let dur = match (start, end) {
        (Some(s), Some(e)) => (e - s).num_minutes(),
        _ => 0,
    };
    if lower.contains("dinner") || lower.contains("lunch") || lower.contains("breakfast")
        || lower.contains("drinks") || lower.contains("coffee")
    {
        return "meal".into();
    }
    if lower.contains("flight") || lower.contains("trip") || lower.contains("vacation") || dur > 12 * 60 {
        return "trip".into();
    }
    if lower.contains("birthday") || lower.contains("anniversary") || lower.contains("wedding") {
        return "milestone".into();
    }
    if lower.contains("call") || lower.contains("zoom") || lower.contains("meet") {
        return if attendee_count >= 2 { "meeting".into() } else { "call".into() };
    }
    if attendee_count >= 2 { "meeting".into() } else { "other".into() }
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    Some(epoch + Duration::seconds(s as i64))
}

fn snapshot_db(src: &std::path::Path, label: &str) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-{label}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join(src.file_name().unwrap_or_default());
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(format!("{label} (Full Disk Access)")));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("db");
    for ext in ["sqlitedb-wal", "sqlitedb-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("{stem}.{ext}")));
        }
    }
    Ok(dst)
}
