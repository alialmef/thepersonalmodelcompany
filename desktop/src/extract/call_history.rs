//! FaceTime / phone call history.
//!
//! Reads `~/Library/Application Support/CallHistoryDB/CallHistory.storedata`
//! — a CoreData SQLite store. The interesting table is `ZCALLRECORD`
//! with fields:
//!   * `ZADDRESS`     — phone number / email called
//!   * `ZDATE`        — Mac absolute seconds
//!   * `ZDURATION`    — call duration in seconds
//!   * `ZORIGINATED`  — 1 if user initiated, 0 if inbound
//!   * `ZANSWERED`    — 1 if connected
//!
//! Output: enriches Person entities with `facetime` channel counts and
//! refreshes `last_seen` from call dates. Honest signal because call
//! frequency is a different relationship indicator than text frequency
//! (people you call vs. people you text are not the same set).

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Person};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "call_history";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Application Support/CallHistoryDB/CallHistory.storedata");
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
            skip_reason: Some("CallHistory.storedata not present".into()),
            ..Default::default()
        });
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    let stmt_res = conn.prepare(
        "SELECT ZADDRESS, ZDATE, ZDURATION, ZORIGINATED FROM ZCALLRECORD WHERE ZADDRESS IS NOT NULL"
    );
    let mut stmt = match stmt_res {
        Ok(s) => s,
        Err(_) => {
            return Ok(ExtractSummary {
                source: SOURCE.into(),
                skipped: true,
                skip_reason: Some("ZCALLRECORD missing".into()),
                ..Default::default()
            });
        }
    };

    struct Agg {
        count: u64,
        outbound: u64,
        duration_sec: u64,
        first: Option<DateTime<Utc>>,
        last:  Option<DateTime<Utc>>,
    }
    let mut by_addr: HashMap<String, Agg> = HashMap::new();
    let mut total = 0u64;
    let rows = stmt.query_map([], |row| {
        let addr: String = row.get(0)?;
        let date: Option<f64> = row.get(1)?;
        let dur: Option<f64>  = row.get(2)?;
        let originated: Option<i64> = row.get(3)?;
        Ok((addr, date, dur, originated))
    })?;
    for r in rows {
        let (addr, date, dur, originated) = r?;
        total += 1;
        let dt = date.and_then(apple_seconds_to_utc);
        let a = by_addr.entry(normalize(&addr)).or_insert(Agg {
            count: 0, outbound: 0, duration_sec: 0, first: None, last: None,
        });
        a.count += 1;
        if matches!(originated, Some(1)) { a.outbound += 1; }
        if let Some(d) = dur { a.duration_sec += d.max(0.0) as u64; }
        if let Some(t) = dt {
            if a.first.map(|f| t < f).unwrap_or(true) { a.first = Some(t); }
            if a.last.map(|l| t > l).unwrap_or(true)  { a.last  = Some(t); }
        }
    }

    let mut people: Vec<Person> = Vec::with_capacity(by_addr.len());
    for (addr, agg) in by_addr {
        let id = stable_id(&["facetime_address", &addr]);
        let phones = if addr.starts_with('+') || addr.chars().all(|c| c.is_ascii_digit() || c == '+') {
            vec![addr.clone()] } else { vec![] };
        let emails = if addr.contains('@') { vec![addr.to_lowercase()] } else { vec![] };

        let mut channel_counts = HashMap::new();
        channel_counts.insert("facetime_total".to_string(),     agg.count);
        channel_counts.insert("facetime_outbound".to_string(),  agg.outbound);
        channel_counts.insert("facetime_seconds".to_string(),   agg.duration_sec);

        people.push(Person {
            id,
            display_name: None,
            aliases: vec![addr],
            phones,
            emails,
            relationship: None,
            inferred_role: None,
            temperature: None,
            channel_counts,
            first_seen: agg.first,
            last_seen: agg.last,
            organizations: vec![],
            birthday: None,
            sources: vec![SOURCE.into()],
        });
    }

    let n = people.len();
    ctx.store.upsert_many(EntityKind::Person, &people, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Person)?;

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

fn normalize(addr: &str) -> String {
    if addr.contains('@') { return addr.to_lowercase(); }
    let mut out = String::with_capacity(addr.len());
    let mut first = true;
    for c in addr.chars() {
        if first && c == '+' { out.push('+'); first = false; continue; }
        if c.is_ascii_digit() { out.push(c); }
        first = false;
    }
    out
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    Some(epoch + Duration::seconds(s as i64))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-callhist-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("CallHistory.storedata");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied("Call History (Full Disk Access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["storedata-wal", "storedata-shm"] {
        let s = src.with_extension(ext);
        if s.exists() { let _ = std::fs::copy(&s, dir.join(format!("CallHistory.{ext}"))); }
    }
    Ok(dst)
}
