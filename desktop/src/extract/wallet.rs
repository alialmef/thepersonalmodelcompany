//! Apple Wallet passes extractor.
//!
//! Reads `~/Library/Passes/Cards/<uuid>.pkpass/pass.json` — every
//! boarding pass, event ticket, loyalty card, store card, and generic
//! pass the user has ever added. Apple keeps these even after they
//! expire (boarding passes from 2018, etc.), which makes the Cards
//! directory a multi-year travel + commerce ledger.
//!
//! Each pass type carries different structured fields:
//!   boardingPass  → flight #, depart/dest IATA, passenger, dates
//!   eventTicket   → venue, performer, date, seat
//!   storeCard     → merchant, balance, transactions
//!   coupon        → merchant, offer, expiry
//!   generic       → free-form fields
//!
//! We don't try to canonicalize across kinds — each pass becomes a
//! Project entity (Wallet passes ARE projects in the loose sense:
//! durable commitments with a time arc). The synthesis layer can
//! group them by merchant / destination / event series later.
//!
//! We also emit a Place node for each unique location (airport,
//! venue) so locations.rs and this extractor cross-reference cleanly.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Place, Project};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "wallet";

fn cards_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Passes/Cards");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(dir) = cards_dir() else {
        return Ok(skipped("HOME unset"));
    };
    if !dir.is_dir() {
        return Ok(skipped("Wallet Cards directory not present"));
    }

    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Apple Wallet (Full Disk Access)".into(),
            ));
        }
        Err(_) => return Ok(skipped("couldn't read Wallet Cards directory")),
    };

    let mut projects: Vec<Project> = Vec::new();
    let mut places_by_key: HashMap<(i64, i64), Place> = HashMap::new();
    let mut scanned = 0u64;

    for entry in read.flatten() {
        let entry_path = entry.path();
        if !entry_path.is_dir() { continue; }
        // .pkpass directory or just a uuid dir
        let pass_path = entry_path.join("pass.json");
        if !pass_path.is_file() { continue; }
        scanned += 1;

        let Ok(text) = std::fs::read_to_string(&pass_path) else { continue };
        let Ok(json) = serde_json::from_str::<Value>(&text) else { continue };

        let dict = match json.as_object() {
            Some(d) => d,
            None => continue,
        };

        let serial = dict
            .get("serialNumber")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let org = dict
            .get("organizationName")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let description = dict
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Determine pass kind by which top-level key holds the body.
        let (kind, body) = if let Some(v) = dict.get("boardingPass") {
            ("boardingPass", v)
        } else if let Some(v) = dict.get("eventTicket") {
            ("eventTicket", v)
        } else if let Some(v) = dict.get("coupon") {
            ("coupon", v)
        } else if let Some(v) = dict.get("storeCard") {
            ("storeCard", v)
        } else if let Some(v) = dict.get("generic") {
            ("generic", v)
        } else {
            ("unknown", &Value::Null)
        };

        let relevant_date = dict
            .get("relevantDate")
            .and_then(|v| v.as_str())
            .and_then(parse_iso8601);
        let expiration = dict
            .get("expirationDate")
            .and_then(|v| v.as_str())
            .and_then(parse_iso8601);

        // Try to pull a short "primary value" from the body — for
        // boarding passes that's depart→destination; for events that's
        // the venue or performer; for store cards that's the merchant.
        let mut headline = String::new();
        if let Some(b) = body.as_object() {
            for section in ["primaryFields", "secondaryFields", "auxiliaryFields", "headerFields"] {
                if let Some(arr) = b.get(section).and_then(|v| v.as_array()) {
                    for field in arr {
                        if let Some(val) = field.get("value").and_then(|v| v.as_str()) {
                            if !headline.is_empty() {
                                headline.push_str(" · ");
                            }
                            headline.push_str(val);
                            if headline.len() > 200 { break; }
                        }
                    }
                    if !headline.is_empty() { break; }
                }
            }
        }

        // For boarding passes, prefer the depart→destination form.
        if kind == "boardingPass" {
            if let Some(b) = body.as_object() {
                if let Some(arr) = b.get("primaryFields").and_then(|v| v.as_array()) {
                    let mut depart = None;
                    let mut dest = None;
                    for field in arr {
                        let key = field.get("key").and_then(|v| v.as_str()).unwrap_or("");
                        let val = field.get("value").and_then(|v| v.as_str()).unwrap_or("");
                        if key.contains("depart") || key == "origin" { depart = Some(val); }
                        if key.contains("destination") || key == "arrive" { dest = Some(val); }
                    }
                    if let (Some(d), Some(a)) = (depart, dest) {
                        headline = format!("{d} → {a}");
                    }
                }
            }
        }

        let name = if !headline.is_empty() {
            format!("{org} · {headline}")
        } else if !description.is_empty() {
            format!("{org} · {description}")
        } else if !org.is_empty() {
            org.clone()
        } else {
            kind.to_string()
        };

        let last_activity = relevant_date.or(expiration);

        // Tag state by recency: if relevant_date is in the past → done,
        // future → active, no date → unknown.
        let state = if let Some(d) = relevant_date {
            if d < Utc::now() { "done" } else { "active" }
        } else {
            "unknown"
        };

        let mut summary_parts: Vec<String> = vec![format!("kind={kind}")];
        if !org.is_empty() { summary_parts.push(format!("org={org}")); }
        if let Some(d) = relevant_date {
            summary_parts.push(format!("relevant={}", d.format("%Y-%m-%d")));
        }
        if let Some(d) = expiration {
            summary_parts.push(format!("expires={}", d.format("%Y-%m-%d")));
        }

        projects.push(Project {
            id: stable_id(&["wallet_pass", &serial, &org]),
            name,
            state: Some(state.to_string()),
            people_ids: Vec::new(),
            last_activity,
            summary: Some(summary_parts.join(" · ")),
            sources: vec![format!("{SOURCE}:{kind}")],
        });

        // Locations — each pass.json may have a `locations` array.
        if let Some(locs) = dict.get("locations").and_then(|v| v.as_array()) {
            for loc in locs {
                let lat = loc.get("latitude").and_then(|v| v.as_f64());
                let lon = loc.get("longitude").and_then(|v| v.as_f64());
                let rel_text = loc
                    .get("relevantText")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if let (Some(lat), Some(lon)) = (lat, lon) {
                    let key = ((lat * 10_000.0).round() as i64,
                               (lon * 10_000.0).round() as i64);
                    let label = if !rel_text.is_empty() {
                        rel_text.to_string()
                    } else if !org.is_empty() {
                        format!("{org} location")
                    } else {
                        format!("wallet@{:.4},{:.4}", lat, lon)
                    };
                    places_by_key.entry(key).or_insert(Place {
                        id: stable_id(&[SOURCE, "loc", &format!("{},{}", key.0, key.1)]),
                        label,
                        lat: Some(lat),
                        lon: Some(lon),
                        kind: Some("wallet".into()),
                        visit_count: 1,
                        first_seen: relevant_date,
                        last_seen: relevant_date,
                        sources: vec![SOURCE.into()],
                    });
                }
            }
        }
    }

    let n_projects = projects.len();
    let n_places = places_by_key.len();
    ctx.store.upsert_many(EntityKind::Project, &projects, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Project)?;
    let places: Vec<Place> = places_by_key.into_values().collect();
    ctx.store.upsert_many(EntityKind::Place, &places, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Place)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", scanned);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: scanned,
        entities_written: (n_projects + n_places) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn parse_iso8601(s: &str) -> Option<DateTime<Utc>> {
    // Try RFC3339 first (most pass.json dates are this form).
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    // Fallback: drop trailing "Z" or numeric offset, parse without.
    None
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
