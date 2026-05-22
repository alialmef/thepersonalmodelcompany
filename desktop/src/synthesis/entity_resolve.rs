//! Cross-source Person resolution.
//!
//! Strategy: build a union-find over Person rows where two rows are
//! linked if they share any of (normalized phone, lowercased email,
//! display name + relationship). Then emit `Edge` records linking each
//! non-canonical Person to a chosen canonical (the one with the most
//! evidence: most aliases × highest channel count). Edges are
//! reversible by design — a future "wait, that's not the same person"
//! correction just deletes the edge; the underlying entities are
//! untouched.
//!
//! We don't *merge in place* because each source-derived Person is
//! independently meaningful — Contacts gives us the label, iMessage
//! gives us cadence, Photos gives us face. Collapsing them destroys
//! provenance; linking them preserves it.

use crate::extract::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{Edge, EntityKind, Person};
use crate::graph::store::stable_id;
use chrono::Utc;
use std::collections::HashMap;

const SOURCE: &str = "synthesis.entity_resolve";

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let people: Vec<Person> = ctx.store.load(EntityKind::Person)?;
    if people.is_empty() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            ..Default::default()
        });
    }

    // Index by normalized handles.
    let mut by_phone: HashMap<String, Vec<usize>> = HashMap::new();
    let mut by_email: HashMap<String, Vec<usize>> = HashMap::new();
    let mut by_name:  HashMap<String, Vec<usize>> = HashMap::new();
    for (i, p) in people.iter().enumerate() {
        for ph in &p.phones { by_phone.entry(normalize_phone(ph)).or_default().push(i); }
        for em in &p.emails { by_email.entry(em.to_lowercase()).or_default().push(i); }
        if let Some(name) = &p.display_name {
            let k = name.trim().to_lowercase();
            if !k.is_empty() { by_name.entry(k).or_default().push(i); }
        }
        for alias in &p.aliases {
            let k = alias.trim().to_lowercase();
            if !k.is_empty() && k.chars().any(|c| c.is_alphabetic()) {
                by_name.entry(k).or_default().push(i);
            }
        }
    }

    // Union-find.
    let mut parent: Vec<usize> = (0..people.len()).collect();
    fn find(p: &mut [usize], x: usize) -> usize {
        if p[x] == x { return x; }
        let r = find(p, p[x]);
        p[x] = r;
        r
    }
    fn union(p: &mut [usize], a: usize, b: usize) {
        let ra = find(p, a);
        let rb = find(p, b);
        if ra != rb { p[ra] = rb; }
    }
    for group in by_phone.values().chain(by_email.values()).chain(by_name.values()) {
        if group.len() < 2 { continue; }
        let first = group[0];
        for &other in &group[1..] { union(&mut parent, first, other); }
    }

    // For each cluster, pick a canonical = highest "evidence score".
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..people.len() {
        let r = find(&mut parent, i);
        clusters.entry(r).or_default().push(i);
    }

    let now = Utc::now();
    let mut edges: Vec<Edge> = Vec::new();
    for (_root, members) in clusters {
        if members.len() < 2 { continue; }
        // Score by aliases + total channel counts.
        let canonical = *members.iter().max_by_key(|&&i| {
            let p = &people[i];
            let ch: u64 = p.channel_counts.values().sum();
            (p.display_name.is_some() as u64) * 100_000
                + (p.aliases.len() as u64) * 1_000
                + ch
        }).unwrap();
        let from = &people[canonical].id;
        for &m in &members {
            if m == canonical { continue; }
            let to = &people[m].id;
            edges.push(Edge {
                id: stable_id(&["same_as", from, to]),
                from_type: "person".into(),
                from_id: from.clone(),
                to_type: "person".into(),
                to_id: to.clone(),
                kind: "same_as".into(),
                confidence: 0.8,
                created_at: now,
            });
        }
    }

    let n = edges.len();
    // Edges are recomputed from scratch — clear stale ones so
    // corrections (a "same_as" we removed by hand) don't reappear.
    ctx.store.clear_kind(EntityKind::Edge)?;
    ctx.store.upsert_many(EntityKind::Edge, &edges, |e| e.id.clone())?;
    ctx.store.flush_kind(EntityKind::Edge)?;

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: people.len() as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn normalize_phone(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut first = true;
    for c in raw.chars() {
        if first && c == '+' { out.push('+'); first = false; continue; }
        if c.is_ascii_digit() { out.push(c); }
        first = false;
    }
    out
}
