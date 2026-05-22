//! Episode synthesis — the highest-level structure in the graph.
//!
//! An episode is a slice of life that binds multiple events + entities
//! across time: "the Vermont weekend," "the LA decision arc," "the
//! month I was reading about Rome." We surface them by looking for:
//!
//!   * **Trip clusters** — consecutive multi-day photo-cluster events
//!     at the same place, or a calendar event tagged `trip`. Groups
//!     anchored on place + date proximity.
//!   * **Decision arcs** — open loops that have stayed alive >= 14 days
//!     and accumulated multiple related touchpoints. (Wave-2: detect
//!     theme-trajectory inflections.)
//!
//! For V0 we ship trip clustering; arc detection follows once embeddings
//! land.

use crate::extract::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Episode, Event, Place};
use crate::graph::store::stable_id;
use chrono::Duration;
use std::collections::HashMap;

const SOURCE: &str = "synthesis.episodes";

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let mut events: Vec<Event> = ctx.store.load(EntityKind::Event)?;
    if events.is_empty() {
        return Ok(ExtractSummary { source: SOURCE.into(), ..Default::default() });
    }

    // Episodes are about lived moments, not recurring calendar
    // placeholders. Drop:
    //   * events more than 60 days in the future (recurring holidays,
    //     birthdays repeating forever, etc.)
    //   * "milestone" calendar events with no attendees and no place
    //     (almost always Apple's preloaded holiday calendar)
    let now = chrono::Utc::now();
    let future_cutoff = now + Duration::days(60);
    events.retain(|e| {
        let start = match e.start { Some(s) => s, None => return false };
        if start > future_cutoff { return false; }
        let is_calendar_only = e.sources.iter().any(|s| s == "calendar")
            && e.attendee_ids.is_empty()
            && e.place_id.is_none();
        let is_holidayish = matches!(e.kind.as_deref(), Some("milestone")) && is_calendar_only;
        !is_holidayish
    });

    // Build place-id → label map for nicer episode names.
    let places: Vec<Place> = ctx.store.load(EntityKind::Place).unwrap_or_default();
    let place_label: HashMap<String, String> = places.into_iter()
        .map(|p| (p.id, p.label)).collect();

    // Sort chronologically.
    events.sort_by_key(|e| e.start.map(|t| t.timestamp()).unwrap_or(0));

    // Group consecutive events (within 36h gap) at the same place
    // into a trip-like episode. If no place, still group by date
    // proximity but with a tighter window (24h).
    let mut episodes: Vec<Episode> = Vec::new();
    let mut current_group: Vec<&Event> = Vec::new();
    let mut current_place: Option<String> = None;
    let mut current_last_end: Option<chrono::DateTime<chrono::Utc>> = None;

    let close_group = |group: &Vec<&Event>, place: &Option<String>, out: &mut Vec<Episode>| {
        if group.len() < 2 { return; }
        let start = group.iter().filter_map(|e| e.start).min();
        let end   = group.iter().filter_map(|e| e.end.or(e.start)).max();
        if let (Some(s), Some(e)) = (start, end) {
            let dur_days = (e - s).num_days();
            if dur_days < 1 && group.len() < 3 { return; } // not really an episode
            let label = make_label(group, place, &place_label);
            let place_ids: Vec<String> = group.iter().filter_map(|ev| ev.place_id.clone())
                .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            let people_ids: Vec<String> = group.iter().flat_map(|ev| ev.attendee_ids.clone())
                .collect::<std::collections::BTreeSet<_>>().into_iter().collect();
            let event_ids: Vec<String> = group.iter().map(|ev| ev.id.clone()).collect();
            out.push(Episode {
                id: stable_id(&["episode", &label, &s.to_rfc3339()]),
                label,
                start: s,
                end: e,
                event_ids,
                people_ids,
                place_ids,
                summary: None,
            });
        }
    };

    for ev in &events {
        let Some(start) = ev.start else { continue };
        let place = ev.place_id.clone();
        let gap_ok = match current_last_end {
            Some(last) => (start - last) < Duration::hours(36),
            None => true,
        };
        let place_ok = current_place == place || current_place.is_none() || place.is_none();
        if gap_ok && place_ok {
            current_group.push(ev);
            if current_place.is_none() { current_place = place.clone(); }
            current_last_end = ev.end.or(Some(start));
        } else {
            close_group(&current_group, &current_place, &mut episodes);
            current_group = vec![ev];
            current_place = place;
            current_last_end = ev.end.or(Some(start));
        }
    }
    close_group(&current_group, &current_place, &mut episodes);

    // Keep most-recent 100 episodes — anything older is rarely
    // surfaced by the agent.
    episodes.sort_by(|a, b| b.start.cmp(&a.start));
    episodes.truncate(100);

    let n = episodes.len();
    // Episodes are recomputed from scratch — clear so we don't keep
    // bogus episodes built from earlier (buggier) event data.
    ctx.store.clear_kind(EntityKind::Episode)?;
    ctx.store.upsert_many(EntityKind::Episode, &episodes, |e| e.id.clone())?;
    ctx.store.flush_kind(EntityKind::Episode)?;

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: events.len() as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn make_label(group: &[&Event], place: &Option<String>, place_labels: &HashMap<String, String>) -> String {
    // Prefer a non-photo-cluster title from the underlying events
    // (these are usually Calendar event names — "Coffee with Sarah"
    // beats "Photo cluster — 2026-05-09").
    let preferred: Option<&str> = group.iter()
        .find(|e| !e.title.starts_with("Photo "))
        .map(|e| e.title.as_str());
    if let Some(t) = preferred { return t.to_string(); }

    // Photo-only episode — use the resolved place label, not its hash.
    if let Some(p) = place {
        if let Some(lbl) = place_labels.get(p) {
            let start_date = group.iter().filter_map(|e| e.start).min()
                .map(|t| t.format("%b %d").to_string());
            return match start_date {
                Some(d) => format!("{d} — {lbl}"),
                None => lbl.clone(),
            };
        }
    }
    let dates: Vec<String> = group.iter().filter_map(|e| e.start).take(1)
        .map(|t| t.format("%Y-%m-%d").to_string()).collect();
    if let Some(d) = dates.first() {
        return format!("Episode — {d}");
    }
    "Episode".into()
}
