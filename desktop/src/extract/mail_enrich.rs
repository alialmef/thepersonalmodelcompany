//! Mail-correspondent extractor.
//!
//! Where `ingest::mail` reads sent-mail bodies for training, this pass
//! walks *all* mailboxes (sent + received) for headers only — From/To/
//! Date — and turns the address graph into Person hints with email-
//! channel counts. We never read message bodies here.
//!
//! Apple Mail stores `.emlx` files per message under
//! `~/Library/Mail/V*/<account-uuid>/<mailbox>/Data/Messages`. We walk
//! the tree, peek the small header prefix on each file, and increment
//! correspondent counters.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Person};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "mail_enrich";

// The full mail tree can be enormous; cap at this many file probes
// per run to keep one extraction pass under a few seconds.
const FILE_LIMIT: usize = 50_000;

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let home = std::env::var_os("HOME").map(PathBuf::from)
        .ok_or_else(|| ExtractError::Other("HOME unset".into()))?;
    let mail_root = home.join("Library/Mail");
    if !mail_root.is_dir() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("no Apple Mail data on this Mac".into()),
            ..Default::default()
        });
    }

    let mut by_addr: HashMap<String, Agg> = HashMap::new();
    let mut scanned = 0usize;

    let mut stack: Vec<PathBuf> = vec![mail_root];
    while let Some(d) = stack.pop() {
        if scanned >= FILE_LIMIT { break; }
        let Ok(read) = std::fs::read_dir(&d) else { continue };
        for entry in read.flatten() {
            if scanned >= FILE_LIMIT { break; }
            let p = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(p);
                continue;
            }
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !ext.eq_ignore_ascii_case("emlx") { continue; }
            scanned += 1;
            if let Some(parsed) = peek_emlx(&p) {
                for addr in &parsed.from { by_addr.entry(addr.clone()).or_default().mark_from(parsed.date); }
                for addr in &parsed.to { by_addr.entry(addr.clone()).or_default().mark_to(parsed.date); }
            }
        }
    }

    let people: Vec<Person> = by_addr.into_iter()
        .filter(|(_, a)| a.total() >= 2)
        .map(|(addr, agg)| {
            let id = stable_id(&["mail_address", &addr]);
            let mut channel_counts = HashMap::new();
            channel_counts.insert("mail_received".to_string(), agg.received);
            channel_counts.insert("mail_sent".to_string(),     agg.sent);
            let org = email_domain(&addr).filter(|d| !is_personal_domain(d));
            Person {
                id,
                display_name: None,
                aliases: vec![addr.clone()],
                phones: vec![],
                emails: vec![addr.to_lowercase()],
                relationship: None,
                inferred_role: Some(if agg.sent > 0 && agg.received > 0 { "correspondent".into() } else { "contact_email".into() }),
                temperature: None,
                channel_counts,
                first_seen: agg.first,
                last_seen: agg.last,
                organizations: org.into_iter().collect(),
                birthday: None,
                sources: vec![SOURCE.into()],
            }
        })
        .collect();

    let n = people.len();
    ctx.store.upsert_many(EntityKind::Person, &people, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Person)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "scanned", scanned as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: scanned as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

#[derive(Default)]
struct Agg {
    received: u64,
    sent: u64,
    first: Option<DateTime<Utc>>,
    last:  Option<DateTime<Utc>>,
}
impl Agg {
    fn total(&self) -> u64 { self.received + self.sent }
    fn mark_from(&mut self, dt: Option<DateTime<Utc>>) {
        self.received += 1;
        self.bump(dt);
    }
    fn mark_to(&mut self, dt: Option<DateTime<Utc>>) {
        self.sent += 1;
        self.bump(dt);
    }
    fn bump(&mut self, dt: Option<DateTime<Utc>>) {
        if let Some(t) = dt {
            if self.first.map(|f| t < f).unwrap_or(true) { self.first = Some(t); }
            if self.last.map(|l| t > l).unwrap_or(true)  { self.last  = Some(t); }
        }
    }
}

#[derive(Default)]
struct Headers {
    from: Vec<String>,
    to:   Vec<String>,
    date: Option<DateTime<Utc>>,
}

fn peek_emlx(path: &std::path::Path) -> Option<Headers> {
    use std::io::Read;
    // .emlx files start with a byte count then an rfc822 message. We
    // only need the headers — read at most 8 KB.
    let mut f = std::fs::File::open(path).ok()?;
    let mut buf = [0u8; 8192];
    let n = f.read(&mut buf).ok()?;
    let s = std::str::from_utf8(&buf[..n]).ok()?;
    // Skip the leading byte count line.
    let body = s.find('\n').map(|i| &s[i+1..]).unwrap_or(s);
    let mut h = Headers::default();
    for line in body.lines() {
        if line.is_empty() { break; } // end of headers
        if let Some(rest) = line.strip_prefix("From:")  { h.from = parse_addresses(rest); }
        else if let Some(rest) = line.strip_prefix("To:") { h.to = parse_addresses(rest); }
        else if let Some(rest) = line.strip_prefix("Date:") {
            h.date = parse_rfc2822(rest.trim());
        }
    }
    Some(h)
}

fn parse_addresses(field: &str) -> Vec<String> {
    field.split(',').filter_map(|part| {
        let p = part.trim();
        if let (Some(lt), Some(gt)) = (p.find('<'), p.rfind('>')) {
            if lt < gt {
                return Some(p[lt+1..gt].to_lowercase());
            }
        }
        if p.contains('@') { Some(p.to_lowercase()) } else { None }
    }).collect()
}

fn parse_rfc2822(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc2822(s).ok().map(|d| d.with_timezone(&Utc))
}

fn email_domain(addr: &str) -> Option<String> {
    addr.split('@').nth(1).map(|s| s.to_lowercase())
}

fn is_personal_domain(domain: &str) -> bool {
    matches!(
        domain,
        "gmail.com" | "icloud.com" | "me.com" | "mac.com" | "yahoo.com" |
        "hotmail.com" | "outlook.com" | "live.com" | "proton.me" | "protonmail.com" |
        "fastmail.com" | "aol.com" | "msn.com"
    )
}
