//! Notes typing + project + theme extractor.
//!
//! `ingest::notes` already pulls raw note bodies for training. This
//! pass re-reads the same store and adds structural signal:
//!
//!   * **Typed classification** — each note is labeled as one of:
//!     `draft` (longform prose, unfinished), `list` (mostly bullets /
//!     short lines), `project` (titled + recurring updates),
//!     `journal` (dated entry), or `other`.
//!   * **Project entities** — notes that look like ongoing projects
//!     produce a `Project` with `state` (active/dormant/done).
//!   * **Open loops** — notes that end mid-thought, contain explicit
//!     `TODO:` markers, or end in "?" lines become OpenLoop entries.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, OpenLoop, Project};
use crate::graph::store::stable_id;
use crate::ingest::notes;
use chrono::{DateTime, Utc};

const SOURCE: &str = "notes_enrich";

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    // Reuse the existing reader. No need to re-do schema discovery here.
    let items = match notes::read_notes(None) {
        Ok(v) => v,
        Err(_) => {
            return Ok(ExtractSummary {
                source: SOURCE.into(),
                skipped: true,
                skip_reason: Some("notes read failed".into()),
                ..Default::default()
            });
        }
    };

    let now = Utc::now();
    let mut projects: Vec<Project> = Vec::new();
    let mut open_loops: Vec<OpenLoop> = Vec::new();

    for item in &items {
        let kind = classify(&item.content);
        let ts_iso = item.timestamp.as_deref().and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|d| d.with_timezone(&Utc));

        // Title heuristic: first non-empty line.
        let title = item.content.lines().find(|l| !l.trim().is_empty()).unwrap_or("Untitled").trim().to_string();

        if kind == "project" {
            let pid = stable_id(&["notes_project", &item.source_id]);
            let state = if let Some(t) = ts_iso {
                let days = (now - t).num_days();
                if days < 14 { "active" } else if days < 90 { "dormant" } else if has_done_marker(&item.content) { "done" } else { "abandoned" }
            } else { "unknown" };
            projects.push(Project {
                id: pid,
                name: truncate(&title, 80),
                state: Some(state.into()),
                people_ids: vec![],
                last_activity: ts_iso,
                summary: Some(first_paragraph(&item.content, 240)),
                sources: vec!["notes".into()],
            });
        }

        // Open-loop scan inside any note.
        for (loop_kind, excerpt) in scan_open_loops(&item.content) {
            let lid = stable_id(&["notes_loop", &item.source_id, &loop_kind, &excerpt]);
            let opened = ts_iso.unwrap_or(now);
            let days = (now - opened).num_days().max(0) as f32;
            let liveness = (1.0 - (days / 60.0)).clamp(0.0, 1.0);
            open_loops.push(OpenLoop {
                id: lid,
                kind: loop_kind,
                summary: format!("Note: {}", truncate(&title, 60)),
                related_person_ids: vec![],
                related_theme_ids: vec![],
                excerpt: Some(truncate(&excerpt, 240)),
                opened_at: opened,
                last_touched: ts_iso,
                liveness,
                source: "notes".into(),
            });
        }
    }

    let n_p = projects.len();
    let n_l = open_loops.len();
    ctx.store.upsert_many(EntityKind::Project,   &projects,   |p| p.id.clone())?;
    ctx.store.upsert_many(EntityKind::OpenLoop,  &open_loops, |o| o.id.clone())?;
    ctx.store.flush_kind(EntityKind::Project)?;
    ctx.store.flush_kind(EntityKind::OpenLoop)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", items.len() as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: items.len() as u64,
        entities_written: (n_p + n_l) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn classify(body: &str) -> &'static str {
    let trimmed = body.trim();
    if trimmed.is_empty() { return "other"; }
    let lines: Vec<&str> = trimmed.lines().filter(|l| !l.trim().is_empty()).collect();
    let n = lines.len();
    if n == 0 { return "other"; }
    let bullet_share = lines.iter().filter(|l| {
        let t = l.trim();
        t.starts_with('-') || t.starts_with('*') || t.starts_with('•') ||
        t.starts_with("[ ]") || t.starts_with("[x]") || t.starts_with("[X]") ||
        t.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
    }).count() as f32 / n as f32;
    if bullet_share > 0.55 { return "list"; }

    let total_words: usize = lines.iter().map(|l| l.split_whitespace().count()).sum();
    let avg_words = total_words as f32 / n as f32;

    // Journal heuristic: leading line looks like a date.
    let first = lines[0].trim();
    if looks_like_date(first) { return "journal"; }

    // Project heuristic: title-cased short first line + later structure.
    if first.split_whitespace().count() <= 8
        && first.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
        && n >= 3
        && (body.contains("TODO") || body.contains("- [ ]") || body.contains("Next:") || body.contains("plan"))
    { return "project"; }

    if avg_words > 12.0 && n >= 3 { return "draft"; }
    "other"
}

fn looks_like_date(s: &str) -> bool {
    // Catches "2026-05-21", "May 21, 2026", "5/21/26", etc., loosely.
    let lc = s.to_lowercase();
    let months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"];
    if months.iter().any(|m| lc.contains(m)) && lc.chars().any(|c| c.is_ascii_digit()) { return true; }
    let digit_count = s.chars().filter(|c| c.is_ascii_digit()).count();
    let punct_count = s.chars().filter(|c| matches!(c, '-' | '/' | '.')).count();
    digit_count >= 4 && punct_count >= 1
}

fn has_done_marker(body: &str) -> bool {
    let lower = body.to_lowercase();
    lower.contains("[x]") || lower.contains("done") || lower.contains("✓") || lower.contains("✅")
}

fn scan_open_loops(body: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("- [ ]") || t.starts_with("[ ]") {
            out.push(("planned_unscheduled".into(), t.to_string()));
        } else if t.to_uppercase().contains("TODO") {
            out.push(("planned_unscheduled".into(), t.to_string()));
        } else if t.ends_with('?') && t.split_whitespace().count() > 3 {
            out.push(("undecided".into(), t.to_string()));
        }
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { return s.to_string(); }
    let mut out: String = s.chars().take(n).collect();
    out.push('…');
    out
}

fn first_paragraph(body: &str, n: usize) -> String {
    let p = body.split("\n\n").next().unwrap_or(body);
    truncate(p, n)
}
