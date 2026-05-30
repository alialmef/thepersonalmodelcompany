//! Notifications extractor.
//!
//! Reads `~/Library/DoNotDisturb/DB/UserNotifications.db` — macOS's
//! sqlite store of every notification ever delivered. We aggregate
//! per-app counts over 30 / 180 day windows and the last-received
//! timestamp. Notification *bodies* are never persisted (only
//! aggregate metadata) — the goal is "which apps are pulling at the
//! user's attention and how much," not surveillance of message content.
//!
//! High-signal for the attention-allocation agent: chronic
//! notification load from one app is a leading indicator of
//! "you're being managed by your phone rather than the other way around."

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, NotificationSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "notifications";

fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/DoNotDisturb/DB/UserNotifications.db");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_db_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped("UserNotifications.db not present"));
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    // Schema (modern macOS): RECORD table has app_id (FK to APP) and
    // delivered_date (Apple absolute seconds since 2001).
    // The APP table maps app_id → bundle_id.
    let sql = r#"
        SELECT APP.identifier AS bundle_id,
               RECORD.delivered_date AS delivered
        FROM RECORD
        JOIN APP ON RECORD.app_id = APP.app_id
        WHERE APP.identifier IS NOT NULL
    "#;
    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(_) => return Ok(skipped("UserNotifications schema unexpected")),
    };

    let now = Utc::now();
    let cutoff_30 = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        count_30d: u64,
        count_180d: u64,
        last: Option<DateTime<Utc>>,
    }
    let mut by_app: HashMap<String, Agg> = HashMap::new();
    let mut total = 0u64;

    let rows = stmt.query_map([], |row| {
        let bundle: String = row.get(0)?;
        let delivered: f64 = row.get(1)?;
        Ok((bundle, delivered))
    })?;
    for r in rows {
        let Ok((bundle, ts)) = r else { continue };
        let Some(dt) = apple_seconds_to_utc(ts) else { continue };
        if dt < cutoff_180 {
            continue;
        }
        total += 1;
        let agg = by_app.entry(bundle).or_insert(Agg {
            count_30d: 0,
            count_180d: 0,
            last: None,
        });
        agg.count_180d += 1;
        if dt >= cutoff_30 {
            agg.count_30d += 1;
        }
        if agg.last.map(|l| dt > l).unwrap_or(true) {
            agg.last = Some(dt);
        }
    }

    let mut entries: Vec<NotificationSignal> = Vec::with_capacity(by_app.len());
    for (bundle, agg) in by_app {
        if agg.count_180d < 3 {
            continue; // noise floor — a handful in 6 months isn't a pattern
        }
        entries.push(NotificationSignal {
            id: stable_id(&["notification", &bundle]),
            display_name: friendly_name(&bundle),
            category: Some(categorize_bundle(&bundle)),
            bundle_id: bundle,
            count_30d: agg.count_30d,
            count_180d: agg.count_180d,
            last_received: agg.last,
        });
    }

    let n = entries.len();
    ctx.store
        .upsert_many(EntityKind::NotificationSignal, &entries, |e| e.id.clone())?;
    ctx.store.flush_kind(EntityKind::NotificationSignal)?;

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

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    if !s.is_finite() {
        return None;
    }
    Some(epoch + Duration::seconds(s as i64))
}

fn friendly_name(bundle: &str) -> Option<String> {
    if let Some(last) = bundle.rsplit('.').next() {
        if !last.is_empty() {
            let mut chars = last.chars();
            let first = chars.next()?.to_uppercase().to_string();
            return Some(format!("{}{}", first, chars.as_str()));
        }
    }
    None
}

fn categorize_bundle(bundle: &str) -> String {
    // Reuse the AppUsage categorizer's buckets so the synthesis layer
    // can correlate app-time + notification-volume per category.
    let b = bundle.to_lowercase();
    let social = ["slack", "discord", "twitter", "instagram", "tiktok", "messenger",
                  "whatsapp", "telegram", "signal", "imessage", "facebook", "reddit"];
    let communication = ["mail", "outlook", "gmail", "spark", "superhuman", "zoom"];
    let work = ["notion", "linear", "asana", "jira", "figma", "monday"];
    let productivity = ["calendar", "fantastical", "things", "omnifocus", "todoist", "reminders"];
    let entertainment = ["spotify", "music", "podcasts", "youtube", "netflix"];

    if social.iter().any(|s| b.contains(s)) { return "social".into(); }
    if communication.iter().any(|s| b.contains(s)) { return "communication".into(); }
    if work.iter().any(|s| b.contains(s)) { return "work".into(); }
    if productivity.iter().any(|s| b.contains(s)) { return "productivity".into(); }
    if entertainment.iter().any(|s| b.contains(s)) { return "entertainment".into(); }
    "other".into()
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-notifications-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("UserNotifications.db");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Notifications (Full Disk Access)".into(),
            ));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["db-wal", "db-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("UserNotifications.{ext}")));
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
