//! Theme detection across sources.
//!
//! "Theme" = a recurring topic the user keeps returning to. We don't
//! cluster with embeddings yet (would need a model on the Mac); we use
//! a fast lexical pipeline that's been good enough for surfacing
//! "what's actually on your mind" in V0:
//!
//!   1. Pull text from raw sources we already have on disk
//!      (`pmc-dev/storage/users/<uid>/raw/*.jsonl`).
//!   2. Token-frequency analysis with English stopwords stripped.
//!   3. Bigram boost — multi-word phrases often carry more signal than
//!      single tokens for theme work ("lake house" beats "lake").
//!   4. Recency weighting: count mentions in last 30 days vs last 180
//!      days, derive trajectory.
//!
//! Real embedding-based theme detection is a Wave-2 upgrade.

use crate::extract::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Theme};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "synthesis.themes";

// English stopwords + a small list of high-frequency conversational
// tokens that aren't themes by themselves.
const STOPWORDS: &[&str] = &[
    "the","a","an","and","or","but","of","in","on","at","to","for","by","with","from","as",
    "is","are","was","were","be","been","being","am","do","does","did","done","doing",
    "have","has","had","having","not","no","yes","yeah","yep","nope","ok","okay",
    "this","that","these","those","there","here","i","im","ive","id","ill","you","youre",
    "youve","youll","he","she","we","they","them","us","me","my","your","his","her","our",
    "their","its","just","very","really","much","more","most","also","one","two","get",
    "got","go","going","gone","come","came","know","knew","think","thought","want","need",
    "make","made","take","took","say","said","tell","told","see","saw","feel","felt",
    "good","bad","right","wrong","new","old","up","down","out","over","under","back","off",
    "on","then","than","when","what","who","where","why","how","because","so","if",
    "would","could","should","will","can","cant","wont","dont","didnt","doesnt","wasnt",
    "werent","arent","isnt","havent","hasnt","hadnt","yet","still","like","love","hate",
    "thats","its","theyre","were","weve","well","ive","ill","im","gonna","wanna","kinda",
    "sorta","lol","haha","ha","yo","ya","u","ur","r","n","s","t","m","ok","oh","ah","eh",
    "hi","hey","bye","thanks","thx","please","sure","fine","cool","nice","wait","whatever",
    "anything","something","nothing","everything","one","two","three","time","day","week",
    "year","tonight","tomorrow","today","yesterday","morning","night","later","soon",
    "now","ever","never","always","sometimes","maybe","probably","actually","basically",
];

#[derive(Default)]
struct Counts {
    total_30d: u64,
    total_180d: u64,
    first: Option<DateTime<Utc>>,
    last: Option<DateTime<Utc>>,
    source_kinds: std::collections::BTreeSet<String>,
    sample_quotes: Vec<String>,
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let raw_dir = ctx.store.root().parent().map(|p| p.join("raw"))
        .or_else(|| Some(PathBuf::from("./raw")));
    let Some(raw_dir) = raw_dir else {
        return Ok(ExtractSummary { source: SOURCE.into(), ..Default::default() });
    };
    if !raw_dir.is_dir() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("no raw dir".into()),
            ..Default::default()
        });
    }

    let now = Utc::now();
    let cutoff_30  = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    let mut term_counts: HashMap<String, Counts> = HashMap::new();
    let mut docs_scanned = 0u64;

    let Ok(read) = std::fs::read_dir(&raw_dir) else {
        return Ok(ExtractSummary { source: SOURCE.into(), ..Default::default() });
    };
    for entry in read.flatten() {
        let p = entry.path();
        if !p.extension().map(|e| e == "jsonl").unwrap_or(false) { continue; }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let content = v.get("content").and_then(|v| v.as_str()).unwrap_or("");
            if content.is_empty() { continue; }
            let ts = v.get("timestamp").and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|d| d.with_timezone(&Utc));
            if let Some(t) = ts { if t < cutoff_180 { continue; } }
            docs_scanned += 1;

            // Unigrams + bigrams. We reject tokens that:
            //   * are pure digits or look like serialized JSON noise
            //     ("null", "---", "undefined", etc.)
            //   * begin/end with non-alphabetic chars (catches "—-" runs)
            //   * are in the stopword list
            let tokens: Vec<String> = content
                .split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
                .filter(|s| !s.is_empty())
                .map(|s| s.to_lowercase())
                .filter(|s| {
                    s.len() > 2
                        && s.len() < 32
                        && s.chars().any(|c| c.is_alphabetic())
                        && !STOPWORDS.contains(&s.as_str())
                        && !is_junk_token(s)
                })
                .collect();

            for tok in &tokens {
                let c = term_counts.entry(tok.clone()).or_default();
                c.total_180d += 1;
                if let Some(t) = ts {
                    if t >= cutoff_30 { c.total_30d += 1; }
                    if c.first.map(|f| t < f).unwrap_or(true) { c.first = Some(t); }
                    if c.last.map(|l| t > l).unwrap_or(true)  { c.last  = Some(t); }
                }
                c.source_kinds.insert(stem.split('-').next().unwrap_or(&stem).to_string());
            }
            for w in tokens.windows(2) {
                if w[0] == w[1] { continue; }
                let bi = format!("{} {}", w[0], w[1]);
                let c = term_counts.entry(bi).or_default();
                c.total_180d += 1;
                if let Some(t) = ts {
                    if t >= cutoff_30 { c.total_30d += 1; }
                    if c.first.map(|f| t < f).unwrap_or(true) { c.first = Some(t); }
                    if c.last.map(|l| t > l).unwrap_or(true)  { c.last  = Some(t); }
                }
                c.source_kinds.insert(stem.split('-').next().unwrap_or(&stem).to_string());
                if c.sample_quotes.len() < 3 {
                    let q: String = content.chars().take(160).collect();
                    if q.to_lowercase().contains(&w[0].to_lowercase()) {
                        c.sample_quotes.push(q);
                    }
                }
            }
        }
    }

    // Rank: prefer bigrams that appear in multiple source kinds, with
    // sufficient frequency.
    let mut themes: Vec<Theme> = term_counts.into_iter()
        .filter(|(t, c)| {
            if t.contains(' ') { c.total_180d >= 5 } else { c.total_180d >= 25 }
        })
        .map(|(label, c)| {
            let trajectory = trajectory(c.total_30d, c.total_180d);
            Theme {
                id: stable_id(&["theme", &label]),
                label: label.clone(),
                keywords: label.split_whitespace().map(String::from).collect(),
                mentions_30d: c.total_30d,
                mentions_180d: c.total_180d,
                trajectory: Some(trajectory),
                first_seen: c.first,
                last_seen: c.last,
                source_kinds: c.source_kinds.into_iter().collect(),
                example_quotes: c.sample_quotes,
            }
        })
        .collect();

    // Keep top 200 most cross-source-and-recent. Otherwise the graph
    // floods.
    themes.sort_by(|a, b| {
        let score_a = a.mentions_30d * 4 + a.mentions_180d + (a.source_kinds.len() as u64) * 10;
        let score_b = b.mentions_30d * 4 + b.mentions_180d + (b.source_kinds.len() as u64) * 10;
        score_b.cmp(&score_a)
    });
    themes.truncate(200);

    let n = themes.len();
    // Themes are recomputed from scratch each pass — clear stale rows
    // so old junk tokens don't persist after filter improvements.
    ctx.store.clear_kind(EntityKind::Theme)?;
    ctx.store.upsert_many(EntityKind::Theme, &themes, |t| t.id.clone())?;
    ctx.store.flush_kind(EntityKind::Theme)?;

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: docs_scanned,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

/// Tokens that survived the splitter but are noise rather than themes —
/// JSON sentinels, common boilerplate words, generic placeholders.
fn is_junk_token(s: &str) -> bool {
    matches!(s, "null" | "none" | "undefined" | "nan" | "true" | "false"
        | "http" | "https" | "www" | "com" | "html" | "css" | "json" | "img"
        | "src" | "href" | "url" | "href=" | "data" | "type" | "value"
        | "name" | "key" | "id" | "row" | "col" | "item" | "list"
        | "test" | "todo" | "fixme" | "xxx" | "wip"
        | "subject" | "from" | "to" | "cc" | "bcc" | "re" | "fwd"
        | "att" | "tel" | "ext" | "vcf" | "ical")
}

fn trajectory(c30: u64, c180: u64) -> String {
    if c180 == 0 { return "dormant".into(); }
    // Recent-share ratio: what fraction of 180-day mentions were in the last 30 days?
    // ~16.7% (30/180) is the steady-state expectation.
    let recent_share = c30 as f64 / c180 as f64;
    if c30 == 0 { return "dormant".into(); }
    if recent_share > 0.30 { return "rising".into(); }
    if recent_share < 0.05 { return "falling".into(); }
    "stable".into()
}
