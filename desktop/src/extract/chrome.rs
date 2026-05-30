//! Chromium-family history extractor.
//!
//! Covers Chrome, Arc, Brave, Edge, Vivaldi — all share the same
//! `History` SQLite shape because they share Chromium's HistoryService.
//! We probe each common install location, aggregate per-domain visits
//! into `WebSignal` entities, and tag them with the browser they came
//! from so the agent can reason cross-browser without conflating signals.
//!
//! Same privacy posture as Safari: only domains, never full URLs.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, WebSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SOURCE: &str = "chrome";

struct Browser {
    name: &'static str,
    relpath: &'static str,
}

const BROWSERS: &[Browser] = &[
    Browser { name: "chrome",   relpath: "Library/Application Support/Google/Chrome/Default/History" },
    Browser { name: "arc",      relpath: "Library/Application Support/Arc/User Data/Default/History" },
    Browser { name: "brave",    relpath: "Library/Application Support/BraveSoftware/Brave-Browser/Default/History" },
    Browser { name: "edge",     relpath: "Library/Application Support/Microsoft Edge/Default/History" },
    Browser { name: "vivaldi",  relpath: "Library/Application Support/Vivaldi/Default/History" },
];

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Ok(skipped("HOME unset"));
    };

    let now = Utc::now();
    let cutoff_30 = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        visits_30d: u64,
        visits_180d: u64,
        last: Option<DateTime<Utc>>,
    }
    // Keyed by (browser, domain) — one WebSignal per (browser, domain).
    // Cross-browser merging is left for the synthesis layer to decide.
    let mut by_key: HashMap<(String, String), Agg> = HashMap::new();
    let mut total = 0u64;
    let mut any_present = false;

    for b in BROWSERS {
        let path = home.join(b.relpath);
        if !path.is_file() {
            continue;
        }
        any_present = true;
        let snapshot = match snapshot_db(&path, b.name) {
            Ok(s) => s,
            Err(ExtractError::PermissionDenied(_)) => continue,  // skip this browser, keep others
            Err(e) => return Err(e),
        };
        let uri = format!("file:{}?mode=ro", snapshot.display());
        let conn = match Connection::open_with_flags(
            &uri,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
        ) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Chromium history: urls.last_visit_time is webkit-time
        // (microseconds since 1601-01-01 UTC).
        let mut stmt = match conn.prepare(
            "SELECT url, visit_count, last_visit_time FROM urls WHERE visit_count > 0",
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let rows = stmt.query_map([], |row| {
            let url: String = row.get(0)?;
            let visits: i64 = row.get(1)?;
            let ts: i64 = row.get(2)?;
            Ok((url, visits, ts))
        });
        let rows = match rows {
            Ok(r) => r,
            Err(_) => continue,
        };
        for r in rows {
            let Ok((url, visits, ts)) = r else { continue };
            let Some(dt) = webkit_micros_to_utc(ts) else { continue };
            if dt < cutoff_180 {
                continue;
            }
            let domain = extract_domain(&url);
            if domain.is_empty() {
                continue;
            }
            total += visits.max(0) as u64;
            let entry = by_key
                .entry((b.name.to_string(), domain))
                .or_insert(Agg { visits_30d: 0, visits_180d: 0, last: None });
            let v = visits.max(0) as u64;
            entry.visits_180d += v;
            if dt >= cutoff_30 {
                entry.visits_30d += v;
            }
            if entry.last.map(|l| dt > l).unwrap_or(true) {
                entry.last = Some(dt);
            }
        }
    }

    if !any_present {
        return Ok(skipped("no Chromium browsers found"));
    }

    let mut signals: Vec<WebSignal> = Vec::with_capacity(by_key.len());
    for ((browser, domain), agg) in by_key {
        if agg.visits_180d < 2 {
            continue;
        }
        signals.push(WebSignal {
            id: stable_id(&["web_signal", &browser, &domain]),
            domain: domain.clone(),
            visits_30d: agg.visits_30d,
            visits_180d: agg.visits_180d,
            last_visit: agg.last,
            category: Some(categorize_domain(&domain)),
            browser: Some(browser),
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

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn extract_domain(url: &str) -> String {
    let s = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("chrome://")
        .trim_start_matches("chrome-extension://");
    let host = s.split('/').next().unwrap_or("");
    let host = host.split('?').next().unwrap_or(host);
    let host = host.split('#').next().unwrap_or(host);
    let host = host.trim_start_matches("www.");
    // Skip noise: empty hosts, IP-ish, in-browser chrome:// pages
    if host.is_empty() || host.starts_with("newtab") || host.starts_with("new-tab") {
        return String::new();
    }
    host.to_lowercase()
}

fn categorize_domain(d: &str) -> String {
    // Same buckets as safari.rs so the synthesis layer can aggregate
    // across browsers without re-translating.
    let social = ["twitter.com","x.com","instagram.com","tiktok.com","facebook.com","reddit.com","linkedin.com","threads.net","bsky.app","mastodon"];
    let news = ["nytimes.com","wsj.com","ft.com","bloomberg.com","theatlantic.com","economist.com","newyorker.com","politico.com","theverge.com","arstechnica.com"];
    let work = ["github.com","gitlab.com","notion.so","figma.com","linear.app","slack.com","atlassian.net","vercel.com","railway.app","cloudflare.com"];
    let ref_ = ["wikipedia.org","wolframalpha.com","stackoverflow.com","stackexchange.com","developer.mozilla.org"];
    let shop = ["amazon.com","ebay.com","etsy.com","shopify.com","walmart.com","target.com","airbnb.com","booking.com"];
    let ent = ["youtube.com","netflix.com","spotify.com","twitch.tv","music.apple.com","hulu.com","apple.com/tv"];
    let research_ai = ["openai.com","anthropic.com","ai.google","huggingface.co","arxiv.org","together.ai","groq.com","claude.ai","gemini.google.com"];

    if social.iter().any(|s| d.ends_with(s)) { return "social".into(); }
    if news.iter().any(|s| d.ends_with(s)) { return "news".into(); }
    if work.iter().any(|s| d.ends_with(s)) { return "work".into(); }
    if ref_.iter().any(|s| d.ends_with(s)) { return "reference".into(); }
    if shop.iter().any(|s| d.ends_with(s)) { return "shopping".into(); }
    if ent.iter().any(|s| d.ends_with(s)) { return "entertainment".into(); }
    if research_ai.iter().any(|s| d.ends_with(s)) { return "research".into(); }
    "other".into()
}

fn webkit_micros_to_utc(usec: i64) -> Option<DateTime<Utc>> {
    if usec <= 0 {
        return None;
    }
    // Webkit epoch: 1601-01-01 UTC
    let epoch = Utc.with_ymd_and_hms(1601, 1, 1, 0, 0, 0).single()?;
    Some(epoch + Duration::microseconds(usec))
}

fn snapshot_db(src: &Path, browser: &str) -> Result<PathBuf, ExtractError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-{}-{}",
        browser,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("History");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(format!(
                "{} (Full Disk Access)",
                browser
            )));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    for ext in ["wal", "shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("History-{}", ext)));
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
