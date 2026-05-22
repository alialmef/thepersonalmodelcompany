//! Apple Reminders extractor.
//!
//! Reminders live in CloudKit-backed CoreData. The store is rooted at
//! `~/Library/Group Containers/group.com.apple.reminders/Container_v1`
//! with mboxes and metadata, and a SQLite store at
//! `…/Container_v1/Stores/Data-local.sqlite` on modern macOS.
//!
//! Every uncompleted reminder is an `OpenLoop`. Each list is a coarse
//! `Project` (groceries don't really count as a project, but the
//! synthesis pass downgrades trivial ones).

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, OpenLoop, Project};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::path::PathBuf;

const SOURCE: &str = "reminders";

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let home = std::env::var_os("HOME").map(PathBuf::from)
        .ok_or_else(|| ExtractError::Other("HOME unset".into()))?;
    let base = home.join("Library/Group Containers/group.com.apple.reminders/Container_v1/Stores");
    if !base.is_dir() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Reminders store not present".into()),
            ..Default::default()
        });
    }

    // Find any *.sqlite file under Stores/.
    let Some(sqlite) = first_sqlite_under(&base) else {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("no sqlite file in Reminders store".into()),
            ..Default::default()
        });
    };

    let snapshot = snapshot_db(&sqlite)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = match Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    ) {
        Ok(c) => c,
        Err(_) => return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Reminders sqlite unreadable".into()),
            ..Default::default()
        }),
    };

    // CoreData tables typically: ZREMCDREMINDER (or similar) with
    // ZTITLE, ZDUEDATE, ZCOMPLETED, ZLIST. Discover.
    let table = first_table_with(&conn, "REMINDER").unwrap_or_default();
    if table.is_empty() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("no Reminder table found".into()),
            ..Default::default()
        });
    }

    let cols = pragma_cols(&conn, &table).unwrap_or_default();
    let title_col = pick(&cols, &["ZTITLE", "ZNAME", "title"]);
    let due_col = pick(&cols, &["ZDUEDATE", "ZTRIGGERDATE", "due_date"]);
    let completed_col = pick(&cols, &["ZCOMPLETED", "ZISCOMPLETED", "completed"]);
    let list_col = pick(&cols, &["ZLIST", "ZCONTAINER"]);

    if title_col.is_none() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Reminder.title column missing".into()),
            ..Default::default()
        });
    }

    let sql = format!(
        "SELECT {t}, {d}, {c}, {l} FROM {tbl}",
        t = title_col.unwrap(),
        d = due_col.unwrap_or("NULL"),
        c = completed_col.unwrap_or("0"),
        l = list_col.unwrap_or("NULL"),
        tbl = table,
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let title:     Option<String> = row.get(0).ok().flatten();
        let due:       Option<f64>    = row.get(1).ok().flatten();
        let completed: Option<i64>    = row.get(2).ok().flatten();
        let list:      Option<i64>    = row.get(3).ok().flatten();
        Ok((title, due, completed, list))
    })?;

    let now = Utc::now();
    let mut open_loops: Vec<OpenLoop> = Vec::new();
    let mut projects: Vec<Project> = Vec::new();
    let mut total = 0u64;
    let mut by_list: std::collections::HashMap<i64, u64> = std::collections::HashMap::new();

    for r in rows {
        let (title, due, completed, list) = r?;
        let Some(title) = title.filter(|t| !t.is_empty()) else { continue };
        total += 1;
        let is_done = matches!(completed, Some(1));
        if let Some(lid) = list { *by_list.entry(lid).or_default() += 1; }
        if is_done { continue; }
        let due_dt = due.and_then(apple_seconds_to_utc);
        let opened = due_dt.unwrap_or(now);
        let days_since = (now - opened).num_days();
        let liveness = if days_since < 0 {
            1.0
        } else if days_since > 90 {
            (1.0 - (days_since as f32 / 365.0)).clamp(0.0, 1.0)
        } else {
            1.0 - (days_since as f32 / 180.0).clamp(0.0, 0.5)
        };
        open_loops.push(OpenLoop {
            id: stable_id(&["reminder", &title]),
            kind: "planned_unscheduled".into(),
            summary: title.clone(),
            related_person_ids: vec![],
            related_theme_ids: vec![],
            excerpt: Some(title),
            opened_at: opened,
            last_touched: due_dt,
            liveness,
            source: SOURCE.into(),
        });
    }

    for (lid, count) in by_list {
        if count < 3 { continue; }
        projects.push(Project {
            id: stable_id(&["reminders_list", &lid.to_string()]),
            name: format!("Reminders list #{}", lid),
            state: Some("active".into()),
            people_ids: vec![],
            last_activity: Some(now),
            summary: Some(format!("{count} items")),
            sources: vec![SOURCE.into()],
        });
    }

    let n_l = open_loops.len();
    let n_p = projects.len();
    ctx.store.upsert_many(EntityKind::OpenLoop, &open_loops, |o| o.id.clone())?;
    ctx.store.upsert_many(EntityKind::Project,  &projects,   |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::OpenLoop)?;
    ctx.store.flush_kind(EntityKind::Project)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", total);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total,
        entities_written: (n_l + n_p) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn first_sqlite_under(dir: &std::path::Path) -> Option<PathBuf> {
    let read = std::fs::read_dir(dir).ok()?;
    for entry in read.flatten() {
        let p = entry.path();
        if p.is_dir() {
            if let Some(inner) = first_sqlite_under(&p) { return Some(inner); }
        } else if p.extension().and_then(|s| s.to_str()).map(|s| s.eq_ignore_ascii_case("sqlite")).unwrap_or(false) {
            return Some(p);
        }
    }
    None
}

fn first_table_with(conn: &Connection, substr: &str) -> Option<String> {
    conn.prepare("SELECT name FROM sqlite_master WHERE type='table'").ok()?
        .query_map([], |row| row.get::<_, String>(0)).ok()?
        .flatten()
        .find(|n| n.to_uppercase().contains(&substr.to_uppercase()))
}

fn pragma_cols(conn: &Connection, table: &str) -> Result<Vec<String>, ExtractError> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
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
        "pmc-reminders-{}",
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
            return Err(ExtractError::PermissionDenied("Reminders (Full Disk Access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    Ok(dst)
}
