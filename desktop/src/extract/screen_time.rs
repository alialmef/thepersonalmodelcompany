//! Screen Time extractor.
//!
//! Reads `knowledgeC.db` — Apple's on-device Screen Time database
//! (`~/Library/Application Support/Knowledge/knowledgeC.db`). It records
//! every app foreground period (the `/app/usage` stream): start
//! timestamp, duration, bundle id, and a couple thousand other event
//! streams we ignore.
//!
//! We aggregate per-app foreground minutes over the trailing 30 / 180
//! days, the hour-of-day distribution, and the day-of-week distribution.
//! No timestamps of individual sessions are persisted — only the
//! aggregated `AppUsage` row per bundle id.
//!
//! Of every signal on a Mac, this one is closest to "how does this
//! person actually spend their time." It's the single richest input
//! for the chief-of-staff agent's time-allocation reasoning.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{AppUsage, EntityKind};
use crate::graph::store::stable_id;
use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "screen_time";

fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Application Support/Knowledge/knowledgeC.db");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_db_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped("knowledgeC.db not present"));
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    // knowledgeC stores events keyed by Core Data string id. The
    // /app/usage stream's ZVALUESTRING column carries the app bundle id;
    // ZSTARTDATE / ZENDDATE are Mac absolute seconds (since 2001-01-01 UTC).
    //
    // Some events are zero-length pings (background fetches, etc.). We
    // filter to durations >= 1 second.
    let sql = r#"
        SELECT
            ZOBJECT.ZVALUESTRING                AS bundle_id,
            ZOBJECT.ZSTARTDATE                  AS start_s,
            ZOBJECT.ZENDDATE                    AS end_s
        FROM ZOBJECT
        LEFT JOIN ZSTRUCTUREDMETADATA
                ON ZOBJECT.ZSTRUCTUREDMETADATA = ZSTRUCTUREDMETADATA.Z_PK
        WHERE ZOBJECT.ZSTREAMNAME = '/app/usage'
          AND ZOBJECT.ZVALUESTRING IS NOT NULL
          AND ZOBJECT.ZENDDATE > ZOBJECT.ZSTARTDATE
    "#;
    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(_) => return Ok(skipped("knowledgeC schema unexpected")),
    };

    let now = Utc::now();
    let cutoff_30 = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        seconds_30d: u64,
        seconds_180d: u64,
        last: Option<DateTime<Utc>>,
        by_hour: [u64; 24],   // seconds
        by_dow: [u64; 7],     // seconds
    }
    let mut by_app: HashMap<String, Agg> = HashMap::new();
    let mut total_sessions = 0u64;

    let rows = stmt.query_map([], |row| {
        let bundle: String = row.get(0)?;
        let start: f64 = row.get(1)?;
        let end: f64 = row.get(2)?;
        Ok((bundle, start, end))
    })?;

    for r in rows {
        let (bundle, start, end) = match r {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (Some(start_dt), Some(end_dt)) =
            (apple_seconds_to_utc(start), apple_seconds_to_utc(end))
        else {
            continue;
        };
        if end_dt < cutoff_180 {
            continue;
        }
        let duration = end_dt.signed_duration_since(start_dt).num_seconds().max(0) as u64;
        if duration < 1 {
            continue;
        }
        total_sessions += 1;

        let a = by_app.entry(bundle).or_insert(Agg {
            seconds_30d: 0,
            seconds_180d: 0,
            last: None,
            by_hour: [0; 24],
            by_dow: [0; 7],
        });
        a.seconds_180d += duration;
        if end_dt >= cutoff_30 {
            a.seconds_30d += duration;
        }
        if a.last.map(|l| end_dt > l).unwrap_or(true) {
            a.last = Some(end_dt);
        }
        let h = start_dt.hour() as usize;
        let dow = start_dt.weekday().num_days_from_monday() as usize;
        if h < 24 {
            a.by_hour[h] = a.by_hour[h].saturating_add(duration);
        }
        if dow < 7 {
            a.by_dow[dow] = a.by_dow[dow].saturating_add(duration);
        }
    }

    let mut entries: Vec<AppUsage> = Vec::with_capacity(by_app.len());
    for (bundle, agg) in by_app {
        let minutes_30d = agg.seconds_30d / 60;
        let minutes_180d = agg.seconds_180d / 60;
        // Drop apps with <5 minutes over 180 days — noise floor.
        if minutes_180d < 5 {
            continue;
        }
        let by_hour = arr_to_minutes(&agg.by_hour);
        let by_dow = arr_to_minutes_7(&agg.by_dow);
        entries.push(AppUsage {
            id: stable_id(&["app_usage", &bundle]),
            display_name: friendly_name(&bundle),
            category: Some(categorize_bundle(&bundle)),
            bundle_id: bundle,
            minutes_30d,
            minutes_180d,
            last_used: agg.last,
            by_hour,
            by_dow,
        });
    }

    let n = entries.len();
    ctx.store.upsert_many(EntityKind::AppUsage, &entries, |e| e.id.clone())?;
    ctx.store.flush_kind(EntityKind::AppUsage)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", total_sessions);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total_sessions,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn arr_to_minutes(secs: &[u64; 24]) -> [u64; 24] {
    let mut out = [0u64; 24];
    for i in 0..24 {
        out[i] = secs[i] / 60;
    }
    out
}

fn arr_to_minutes_7(secs: &[u64; 7]) -> [u64; 7] {
    let mut out = [0u64; 7];
    for i in 0..7 {
        out[i] = secs[i] / 60;
    }
    out
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    if !s.is_finite() {
        return None;
    }
    Some(epoch + Duration::seconds(s as i64))
}

fn friendly_name(bundle: &str) -> Option<String> {
    // A bundle id like "com.tinyspeck.slackmacgap" doesn't read well in
    // a UI — try to surface the last meaningful segment.
    if let Some(last) = bundle.rsplit('.').next() {
        if !last.is_empty() {
            // capitalize first letter
            let mut chars = last.chars();
            let first = chars.next()?.to_uppercase().to_string();
            return Some(format!("{}{}", first, chars.as_str()));
        }
    }
    None
}

fn categorize_bundle(bundle: &str) -> String {
    // Heuristic categorization. Same buckets as Safari/Chrome so the
    // synthesis layer can correlate web + app activity per category.
    let b = bundle.to_lowercase();
    let social = [
        "slack", "discord", "twitter", "tweetbot", "instagram", "tiktok",
        "messenger", "whatsapp", "telegram", "signal", "imessage", "facebook",
        "reddit", "apollo", "ivory", "threads", "bluesky",
    ];
    let communication = ["mail", "outlook", "gmail", "spark", "superhuman", "zoom", "facetime"];
    let developer = [
        "code", "cursor", "claude", "xcode", "intellij", "pycharm", "webstorm",
        "sublime", "atom", "vim", "neovim", "iterm", "terminal", "ghostty",
        "wezterm", "kitty", "docker", "tower", "fork", "sourcetree",
    ];
    let work = [
        "notion", "linear", "asana", "monday", "jira", "figma", "sketch",
        "miro", "google.docs", "office", "word", "excel", "powerpoint",
        "keynote", "pages", "numbers",
    ];
    let entertainment = [
        "spotify", "music", "podcasts", "youtube", "netflix", "tv", "vlc",
        "infuse", "plex", "twitch", "steam", "playstation", "xbox",
    ];
    let reference = ["safari", "chrome", "arc", "brave", "edge", "firefox", "books", "kindle"];
    let productivity = [
        "calendar", "fantastical", "things", "omnifocus", "todoist", "reminders",
        "obsidian", "bear", "drafts", "ulysses", "craft", "logseq",
    ];

    if social.iter().any(|s| b.contains(s)) { return "social".into(); }
    if communication.iter().any(|s| b.contains(s)) { return "communication".into(); }
    if developer.iter().any(|s| b.contains(s)) { return "developer".into(); }
    if work.iter().any(|s| b.contains(s)) { return "work".into(); }
    if entertainment.iter().any(|s| b.contains(s)) { return "entertainment".into(); }
    if reference.iter().any(|s| b.contains(s)) { return "reference".into(); }
    if productivity.iter().any(|s| b.contains(s)) { return "productivity".into(); }
    "other".into()
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-screen-time-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("knowledgeC.db");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Screen Time (Full Disk Access required for knowledgeC.db)".into(),
            ));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["db-wal", "db-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("knowledgeC.{ext}")));
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
