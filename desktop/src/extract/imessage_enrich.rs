//! iMessage relationship-enrichment extractor.
//!
//! Where `ingest::imessage` produces raw text rows for training,
//! `extract::imessage_enrich` produces graph signal: per-handle activity
//! counts, recency, topics, response cadence, and unanswered questions.
//!
//! The two run on the same chat.db. We open a fresh snapshot here so we
//! don't share state with the raw-ingest pass.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, OpenLoop, Person};
use crate::graph::store::stable_id;
use crate::ingest::imessage;
use chrono::{DateTime, Duration, TimeZone, Utc};
use std::collections::HashMap;

const SOURCE: &str = "imessage_enrich";

#[derive(Default, Debug)]
struct HandleStats {
    inbound: u64,
    outbound: u64,
    first_seen: Option<DateTime<Utc>>,
    last_seen: Option<DateTime<Utc>>,
    last_inbound_was_question: bool,
    last_inbound_text: Option<String>,
    last_inbound_at: Option<DateTime<Utc>>,
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let path = imessage::default_chat_db_path()
        .ok_or_else(|| ExtractError::Other("HOME unset".into()))?;
    if !path.is_file() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("no chat.db".into()),
            ..Default::default()
        });
    }

    let conn = imessage::open_chat_db(&path)
        .map_err(|e| ExtractError::Other(format!("open chat.db: {e:?}")))?;

    // Per-handle aggregation.
    let mut stats: HashMap<String, HandleStats> = HashMap::new();

    // We don't need attributedBody decode here — handle + ts + direction
    // is enough for cadence. We DO need text when scanning for open
    // questions, so we fall back to the decoder for those.
    let mut stmt = conn.prepare(
        r#"
        SELECT
            COALESCE(h.id, '') AS handle,
            m.is_from_me      AS is_from_me,
            m.date            AS date_ns,
            COALESCE(m.text, '') AS text,
            m.attributedBody  AS ab
        FROM message m
        LEFT JOIN handle h ON m.handle_id = h.ROWID
        WHERE h.id IS NOT NULL
        "#,
    )?;

    let rows = stmt.query_map([], |row| {
        let handle: String = row.get(0)?;
        let is_from_me: i32 = row.get(1)?;
        let date_ns: i64 = row.get::<_, Option<i64>>(2)?.unwrap_or(0);
        let text: String = row.get(3)?;
        let ab: Option<Vec<u8>> = row.get(4)?;
        Ok((handle, is_from_me, date_ns, text, ab))
    })?;

    for r in rows {
        let (handle, is_from_me, date_ns, text, ab) = r?;
        if handle.is_empty() { continue; }
        let ts = apple_ns_to_utc(date_ns);

        let body = if !text.is_empty() {
            text
        } else if let Some(blob) = ab {
            decode_text_lite(&blob).unwrap_or_default()
        } else {
            String::new()
        };

        let s = stats.entry(handle.clone()).or_default();
        if is_from_me == 1 { s.outbound += 1; } else { s.inbound += 1; }

        if let Some(t) = ts {
            if s.first_seen.map(|f| t < f).unwrap_or(true) { s.first_seen = Some(t); }
            if s.last_seen.map(|l| t > l).unwrap_or(true)  { s.last_seen  = Some(t); }
            if is_from_me == 0 {
                s.last_inbound_at = Some(t);
                s.last_inbound_was_question = looks_like_open_question(&body);
                if !body.is_empty() { s.last_inbound_text = Some(truncate(&body, 200)); }
            } else if s.last_inbound_was_question {
                // The user did reply after the inbound question; close it.
                s.last_inbound_was_question = false;
            }
        }
    }

    // Build Person entries from handle stats. We upsert by handle id so
    // they merge cleanly with Contacts during synthesis (synthesis links
    // these handle-derived persons to contact-derived persons via Edge).
    let now = Utc::now();
    let mut people: Vec<Person> = Vec::with_capacity(stats.len());
    let mut open_loops: Vec<OpenLoop> = Vec::new();

    for (handle, s) in stats {
        let id = stable_id(&["imessage_handle", &handle]);
        let temp = temperature(now, s.last_seen);
        let role = inferred_role(s.inbound + s.outbound, &handle);

        let phones = if handle.starts_with('+') || handle.chars().all(|c| c.is_ascii_digit() || c == '+') {
            vec![handle.clone()]
        } else { vec![] };
        let emails = if handle.contains('@') { vec![handle.to_lowercase()] } else { vec![] };

        let mut channel_counts = HashMap::new();
        channel_counts.insert("imessage_inbound".to_string(),  s.inbound);
        channel_counts.insert("imessage_outbound".to_string(), s.outbound);

        people.push(Person {
            id: id.clone(),
            display_name: None,
            aliases: vec![handle.clone()],
            phones,
            emails,
            relationship: None,
            inferred_role: Some(role),
            temperature: Some(temp),
            channel_counts,
            first_seen: s.first_seen,
            last_seen: s.last_seen,
            organizations: vec![],
            birthday: None,
            sources: vec!["imessage".into()],
        });

        if s.last_inbound_was_question {
            if let (Some(opened), Some(excerpt)) = (s.last_inbound_at, s.last_inbound_text) {
                // Liveness decays linearly over 30 days.
                let days = (now - opened).num_days().max(0) as f32;
                let liveness = (1.0 - (days / 30.0)).clamp(0.0, 1.0);
                let loop_id = stable_id(&["imessage_question", &handle, &opened.to_rfc3339()]);
                open_loops.push(OpenLoop {
                    id: loop_id,
                    kind: "unanswered_question".into(),
                    summary: format!("Unanswered question from {handle}"),
                    related_person_ids: vec![id.clone()],
                    related_theme_ids: vec![],
                    excerpt: Some(excerpt),
                    opened_at: opened,
                    last_touched: Some(opened),
                    liveness,
                    source: "imessage".into(),
                });
            }
        }
    }

    ctx.store.upsert_many(EntityKind::Person, &people, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Person)?;
    let loop_count = open_loops.len();
    ctx.store.upsert_many(EntityKind::OpenLoop, &open_loops, |o| o.id.clone())?;
    ctx.store.flush_kind(EntityKind::OpenLoop)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", people.len() as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: people.len() as u64,
        entities_written: (people.len() + loop_count) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn apple_ns_to_utc(date_ns: i64) -> Option<DateTime<Utc>> {
    if date_ns == 0 { return None; }
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    if date_ns > 10i64.pow(12) {
        Some(epoch + Duration::nanoseconds(date_ns))
    } else {
        Some(epoch + Duration::seconds(date_ns))
    }
}

fn temperature(now: DateTime<Utc>, last_seen: Option<DateTime<Utc>>) -> String {
    let Some(last) = last_seen else { return "unknown".into() };
    let days = (now - last).num_days();
    match days {
        d if d <= 7   => "hot",
        d if d <= 30  => "warm",
        d if d <= 90  => "cool",
        _              => "dormant",
    }.into()
}

fn inferred_role(total: u64, handle: &str) -> String {
    let is_email = handle.contains('@');
    match total {
        t if t > 2000 => if is_email { "professional".into() } else { "close-friend".into() },
        t if t > 500  => if is_email { "professional".into() } else { "friend".into() },
        t if t > 50   => "acquaintance".into(),
        _              => "occasional".into(),
    }
}

fn looks_like_open_question(body: &str) -> bool {
    let trimmed = body.trim();
    if trimmed.is_empty() { return false; }
    let last = trimmed.chars().last().unwrap_or(' ');
    if last == '?' { return true; }
    // Lead phrases that often imply a question even without trailing ?
    let lower = trimmed.to_lowercase();
    [ "are you ", "can you ", "could you ", "do you ", "would you ", "should we ",
      "when do ", "what time", "wanna ", "want to ", "you up", "you down" ]
        .iter()
        .any(|p| lower.starts_with(p) || lower.contains(p))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { return s.to_string(); }
    let mut out: String = s.chars().take(n).collect();
    out.push('…');
    out
}

/// Lightweight NSAttributedString decoder for the open-question scan.
/// Mirrors the logic in `ingest::imessage::decode_attributed_body` but is
/// inlined here so this file doesn't depend on the ingester's internals.
fn decode_text_lite(blob: &[u8]) -> Option<String> {
    let marker = b"NSString";
    let mut idx_search = 0usize;
    while let Some(rel) = blob[idx_search..].windows(marker.len()).position(|w| w == marker) {
        let after = idx_search + rel + marker.len();
        let window_end = (after + 12).min(blob.len());
        if let Some(plus_rel) = blob[after..window_end].iter().position(|&b| b == 0x2b) {
            let len_pos = after + plus_rel + 1;
            if len_pos < blob.len() {
                let b = blob[len_pos];
                let (start, len) = if b == 0x81 && len_pos + 3 <= blob.len() {
                    (len_pos + 3, u16::from_le_bytes([blob[len_pos+1], blob[len_pos+2]]) as usize)
                } else if b == 0x82 && len_pos + 5 <= blob.len() {
                    let n = u32::from_le_bytes([blob[len_pos+1], blob[len_pos+2], blob[len_pos+3], blob[len_pos+4]]) as usize;
                    if n < 10_000_000 { (len_pos + 5, n) } else { return None; }
                } else if b > 0 && b < 0x80 {
                    (len_pos + 1, b as usize)
                } else {
                    idx_search = after + 1;
                    continue;
                };
                if start + len <= blob.len() {
                    if let Ok(s) = std::str::from_utf8(&blob[start..start+len]) {
                        return Some(s.to_string());
                    }
                }
            }
        }
        idx_search = after + 1;
    }
    None
}
