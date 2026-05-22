//! Open-loop re-scoring + theme/person linking.
//!
//! Open loops are written by individual extractors (iMessage,
//! Reminders, Notes). This pass:
//!
//!   * **Decays liveness** based on age — a question from 6 months ago
//!     is mostly noise.
//!   * **Links to themes** — if an open loop's excerpt matches a known
//!     theme's keywords, attach that theme id to `related_theme_ids`.
//!   * **Prunes** loops with liveness below a small floor — they aren't
//!     deleted but they fall off the "live" list the agent surfaces.

use crate::extract::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, OpenLoop, Theme};
use chrono::Utc;
use std::collections::HashMap;

const SOURCE: &str = "synthesis.open_loops";

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let mut loops: Vec<OpenLoop> = ctx.store.load(EntityKind::OpenLoop)?;
    if loops.is_empty() {
        return Ok(ExtractSummary { source: SOURCE.into(), ..Default::default() });
    }
    let themes: Vec<Theme> = ctx.store.load(EntityKind::Theme).unwrap_or_default();

    // Build keyword index for theme lookup.
    let mut by_kw: HashMap<String, String> = HashMap::new();
    for t in &themes {
        for kw in &t.keywords { by_kw.insert(kw.to_lowercase(), t.id.clone()); }
    }

    let now = Utc::now();
    for loop_ in &mut loops {
        // Re-compute liveness from age.
        let days = (now - loop_.opened_at).num_days().max(0) as f32;
        let decayed = (1.0 - (days / 90.0)).clamp(0.0, 1.0);
        loop_.liveness = (loop_.liveness * 0.5 + decayed * 0.5).clamp(0.0, 1.0);

        // Theme tagging from excerpt.
        if let Some(ex) = &loop_.excerpt {
            let lower = ex.to_lowercase();
            let mut tags = Vec::new();
            for (kw, tid) in by_kw.iter() {
                if lower.contains(kw) { tags.push(tid.clone()); }
            }
            tags.sort(); tags.dedup();
            loop_.related_theme_ids = tags;
        }
    }

    loops.sort_by(|a, b| b.liveness.partial_cmp(&a.liveness).unwrap_or(std::cmp::Ordering::Equal));
    let n = loops.len();
    ctx.store.upsert_many(EntityKind::OpenLoop, &loops, |o| o.id.clone())?;
    ctx.store.flush_kind(EntityKind::OpenLoop)?;

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: n as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}
