//! Safari history extractor.
//!
//! Reads `~/Library/Safari/History.db` — Safari's SQLite store of every
//! page visit. We collapse the per-visit firehose into one `WebSignal`
//! per domain with 30-day / 180-day visit counts, last-visit timestamp,
//! and a heuristic category ("research" / "social" / "news" / "work" /
//! "reference" / "shopping" / "entertainment").
//!
//! No URL paths leak into the graph — only domains. Path-level history
//! has too much noise and too many embarrassing private contexts for
//! the graph to be safe.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, WebSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "safari";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Safari/History.db");
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
            skip_reason: Some("Safari history not present".into()),
            ..Default::default()
        });
    }

    let snapshot = snapshot_db(&path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    // history_items has 'url' and 'visit_count'.
    // history_visits has 'history_item' (FK) and 'visit_time' (mac
    // absolute seconds).
    let stmt_res = conn.prepare(
        r#"
        SELECT
            hi.url,
            hv.visit_time
        FROM history_visits hv
        JOIN history_items hi ON hv.history_item = hi.id
        "#,
    );
    let mut stmt = match stmt_res {
        Ok(s) => s,
        Err(_) => {
            return Ok(ExtractSummary {
                source: SOURCE.into(),
                skipped: true,
                skip_reason: Some("history schema unexpected".into()),
                ..Default::default()
            });
        }
    };

    let now = Utc::now();
    let cutoff_30  = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        visits_30d: u64,
        visits_180d: u64,
        last: Option<DateTime<Utc>>,
    }
    let mut by_domain: HashMap<String, Agg> = HashMap::new();
    let mut total = 0u64;
    let rows = stmt.query_map([], |row| {
        let url: String = row.get(0)?;
        let ts: f64 = row.get(1)?;
        Ok((url, ts))
    })?;
    for r in rows {
        let (url, ts) = r?;
        let Some(dt) = apple_seconds_to_utc(ts) else { continue };
        if dt < cutoff_180 { continue; }
        total += 1;
        let domain = extract_domain(&url).to_lowercase();
        if domain.is_empty() { continue; }
        let a = by_domain.entry(domain).or_insert(Agg { visits_30d: 0, visits_180d: 0, last: None });
        a.visits_180d += 1;
        if dt >= cutoff_30 { a.visits_30d += 1; }
        if a.last.map(|l| dt > l).unwrap_or(true) { a.last = Some(dt); }
    }

    let mut signals: Vec<WebSignal> = Vec::with_capacity(by_domain.len());
    for (domain, agg) in by_domain {
        if agg.visits_180d < 2 { continue; } // drop noise
        signals.push(WebSignal {
            id: stable_id(&["safari_domain", &domain]),
            domain: domain.clone(),
            visits_30d: agg.visits_30d,
            visits_180d: agg.visits_180d,
            last_visit: agg.last,
            category: Some(categorize_domain(&domain)),
            browser: Some("safari".into()),
        });
    }

    let n = signals.len();
    ctx.store.upsert_many(EntityKind::WebSignal, &signals, |s| s.id.clone())?;
    ctx.store.flush_kind(EntityKind::WebSignal)?;

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

fn extract_domain(url: &str) -> String {
    let s = url.trim_start_matches("https://").trim_start_matches("http://");
    let host = s.split('/').next().unwrap_or("");
    let host = host.split('?').next().unwrap_or(host);
    host.trim_start_matches("www.").to_string()
}

fn categorize_domain(d: &str) -> String {
    let social   = ["twitter.com","x.com","instagram.com","tiktok.com","facebook.com","reddit.com","linkedin.com","threads.net","bsky.app","mastodon"];
    let news     = ["nytimes.com","wsj.com","ft.com","bloomberg.com","theatlantic.com","economist.com","newyorker.com","politico.com","theverge.com","arstechnica.com"];
    let work     = ["github.com","gitlab.com","notion.so","figma.com","linear.app","slack.com","atlassian.net","vercel.com","railway.app","cloudflare.com"];
    let ref_     = ["wikipedia.org","wolframalpha.com","stackoverflow.com","stackexchange.com","developer.mozilla.org"];
    let shop     = ["amazon.com","ebay.com","etsy.com","shopify.com","walmart.com","target.com","airbnb.com","booking.com"];
    let ent      = ["youtube.com","netflix.com","spotify.com","twitch.tv","music.apple.com","hulu.com","apple.com/tv"];
    let research_ai = ["openai.com","anthropic.com","ai.google","huggingface.co","arxiv.org","together.ai","groq.com"];

    if social.iter().any(|s| d.ends_with(s)) { return "social".into(); }
    if news.iter().any(|s| d.ends_with(s)) { return "news".into(); }
    if work.iter().any(|s| d.ends_with(s)) { return "work".into(); }
    if ref_.iter().any(|s| d.ends_with(s)) { return "reference".into(); }
    if shop.iter().any(|s| d.ends_with(s)) { return "shopping".into(); }
    if ent.iter().any(|s| d.ends_with(s)) { return "entertainment".into(); }
    if research_ai.iter().any(|s| d.ends_with(s)) { return "research".into(); }
    "other".into()
}

fn apple_seconds_to_utc(s: f64) -> Option<DateTime<Utc>> {
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    Some(epoch + Duration::seconds(s as i64))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-safari-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("History.db");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied("Safari (Full Disk Access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["db-wal", "db-shm"] {
        let s = src.with_extension(ext);
        if s.exists() { let _ = std::fs::copy(&s, dir.join(format!("History.{ext}"))); }
    }
    Ok(dst)
}
