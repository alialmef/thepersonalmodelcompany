//! Apple Contacts.app extractor.
//!
//! Reads from `~/Library/Application Support/AddressBook/AddressBook-v22.abcddb`,
//! the CoreData SQLite store the system Contacts app writes through.
//!
//! Key tables:
//!   * `ZABCDRECORD`        — one row per contact (and per group)
//!   * `ZABCDPHONENUMBER`   — phones (ZOWNER -> ZABCDRECORD.Z_PK)
//!   * `ZABCDEMAILADDRESS`  — emails
//!   * `ZABCDRELATEDNAME`   — "sister", "mother", custom relation labels
//!   * `ZABCDPOSTALADDRESS` — postal (not extracted here yet)
//!
//! Output: a `Person` per real contact, with phones / emails / relation
//! label / org / birthday. This is the resolver every other extractor
//! depends on — every iMessage handle, every email, every photo-detected
//! face gets reconciled against these IDs in the synthesis pass.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Person};
use crate::graph::store::stable_id;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

const SOURCE: &str = "contacts";

pub fn default_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Application Support/AddressBook/AddressBook-v22.abcddb");
        p
    })
}

/// Discover every AddressBook-v22.abcddb the user has — the top-level
/// store plus every per-account source under `Sources/<UUID>/`. Real
/// contacts often live entirely in the per-source dbs (iCloud accounts,
/// Google sync, etc.).
fn all_address_books() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Some(home) = std::env::var_os("HOME") else { return out };
    let base = PathBuf::from(&home).join("Library/Application Support/AddressBook");
    let top = base.join("AddressBook-v22.abcddb");
    if top.is_file() { out.push(top); }
    let sources_dir = base.join("Sources");
    if let Ok(read) = std::fs::read_dir(&sources_dir) {
        for entry in read.flatten() {
            let candidate = entry.path().join("AddressBook-v22.abcddb");
            if candidate.is_file() { out.push(candidate); }
        }
    }
    out
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let dbs = all_address_books();
    if dbs.is_empty() {
        return Ok(ExtractSummary {
            source: SOURCE.into(),
            skipped: true,
            skip_reason: Some("Contacts.app not yet used on this Mac".into()),
            ..Default::default()
        });
    }

    let mut all_people: Vec<Person> = Vec::new();
    let mut total_input = 0u64;
    for (idx, path) in dbs.iter().enumerate() {
        match read_one(path, idx) {
            Ok((people, n)) => {
                total_input += n;
                all_people.extend(people);
            }
            Err(_) => continue, // a corrupt or empty source shouldn't kill the whole run
        }
    }

    let written = all_people.len();
    ctx.store.upsert_many(EntityKind::Person, &all_people, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Person)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", written as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total_input,
        entities_written: written as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn read_one(path: &std::path::Path, source_idx: usize) -> Result<(Vec<Person>, u64), ExtractError> {
    let snapshot = snapshot_db(path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    let conn = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;

    let phones = collect_phones(&conn).unwrap_or_default();
    let emails = collect_emails(&conn).unwrap_or_default();
    let relations = collect_relations(&conn).unwrap_or_default();
    let orgs = collect_org_labels(&conn).unwrap_or_default();

    // Filter widened: any record with a non-null name field, nickname,
    // org, or that has at least one phone/email. Apple often stores the
    // canonical name on related rows we don't have schema for, so we
    // accept "has any handle" as evidence the row is a real person.
    let mut stmt = conn.prepare(
        r#"
        SELECT
            Z_PK,
            ZFIRSTNAME,
            ZLASTNAME,
            ZNICKNAME,
            ZORGANIZATION,
            ZBIRTHDAY
        FROM ZABCDRECORD
        "#,
    )?;

    let mut people = Vec::new();
    let mut input_rows: u64 = 0;
    let rows = stmt.query_map([], |row| {
        let pk: i64 = row.get(0)?;
        let first: Option<String> = row.get(1)?;
        let last: Option<String> = row.get(2)?;
        let nick: Option<String> = row.get(3)?;
        let org: Option<String> = row.get(4)?;
        let birthday: Option<f64> = row.get(5)?;
        Ok((pk, first, last, nick, org, birthday))
    })?;

    for r in rows {
        let (pk, first, last, nick, org, birthday) = r?;
        input_rows += 1;

        let phones_v = phones.get(&pk).cloned().unwrap_or_default();
        let emails_v = emails.get(&pk).cloned().unwrap_or_default();

        let display_name = compose_display_name(first.as_deref(), last.as_deref(), nick.as_deref(), org.as_deref());
        // Skip records with no useful info at all.
        if display_name.is_none() && phones_v.is_empty() && emails_v.is_empty() { continue; }

        let mut aliases = Vec::new();
        if let Some(n) = &nick { if !n.is_empty() { aliases.push(n.clone()); } }
        if let Some(o) = &org { if !o.is_empty() && display_name.as_deref() != Some(o) { aliases.push(o.clone()); } }

        let relation_label = relations.get(&pk).cloned();
        let org_v: Vec<String> = orgs.get(&pk).cloned().unwrap_or_default();
        let bday_str = birthday.and_then(apple_date_to_md);

        // Source index in the id keeps per-source contacts distinct
        // until synthesis links them — Sarah in iCloud and Sarah in
        // Google are merged via shared phone/email, not by collapsing
        // the pks.
        let id = stable_id(&[SOURCE, &source_idx.to_string(), &pk.to_string()]);

        people.push(Person {
            id,
            display_name,
            aliases,
            phones: phones_v,
            emails: emails_v,
            relationship: relation_label,
            inferred_role: None,
            temperature: None,
            channel_counts: HashMap::new(),
            first_seen: None,
            last_seen: None,
            organizations: org_v,
            birthday: bday_str,
            sources: vec![SOURCE.into()],
        });
    }
    Ok((people, input_rows))
}

fn snapshot_db(src: &std::path::Path) -> Result<PathBuf, ExtractError> {
    // Same pattern as iMessage — snapshot to /tmp so we don't fight a
    // live writer.
    let dir = std::env::temp_dir().join(format!(
        "pmc-contacts-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir)?;
    let dst = dir.join("AddressBook-v22.abcddb");
    match std::fs::copy(src, &dst) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied("Contacts (grant Full Disk Access)".into()));
        }
        Err(e) => return Err(ExtractError::Io(e)),
    }
    // Copy WAL sidecars if present.
    for ext in ["abcddb-wal", "abcddb-shm"] {
        let s = src.with_extension(ext);
        if s.exists() {
            let _ = std::fs::copy(&s, dir.join(format!("AddressBook-v22.{ext}")));
        }
    }
    Ok(dst)
}

fn collect_phones(conn: &Connection) -> Result<HashMap<i64, Vec<String>>, ExtractError> {
    let mut stmt = conn.prepare("SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZOWNER IS NOT NULL AND ZFULLNUMBER IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        let owner: i64 = row.get(0)?;
        let number: String = row.get(1)?;
        Ok((owner, number))
    })?;
    let mut by_owner: HashMap<i64, Vec<String>> = HashMap::new();
    for r in rows {
        let (owner, num) = r?;
        by_owner.entry(owner).or_default().push(normalize_phone(&num));
    }
    Ok(by_owner)
}

fn collect_emails(conn: &Connection) -> Result<HashMap<i64, Vec<String>>, ExtractError> {
    let mut stmt = conn.prepare("SELECT ZOWNER, ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZOWNER IS NOT NULL AND ZADDRESS IS NOT NULL")?;
    let rows = stmt.query_map([], |row| {
        let owner: i64 = row.get(0)?;
        let addr: String = row.get(1)?;
        Ok((owner, addr))
    })?;
    let mut by_owner: HashMap<i64, Vec<String>> = HashMap::new();
    for r in rows {
        let (owner, a) = r?;
        by_owner.entry(owner).or_default().push(a.to_lowercase());
    }
    Ok(by_owner)
}

fn collect_relations(conn: &Connection) -> Result<HashMap<i64, String>, ExtractError> {
    let stmt_res = conn.prepare("SELECT ZOWNER, ZLABEL FROM ZABCDRELATEDNAME WHERE ZOWNER IS NOT NULL");
    let mut stmt = match stmt_res {
        Ok(s) => s,
        Err(_) => return Ok(HashMap::new()), // table may not exist on older macOS
    };
    let rows = stmt.query_map([], |row| {
        let owner: i64 = row.get(0)?;
        let label: Option<String> = row.get(1)?;
        Ok((owner, label))
    })?;
    let mut out: HashMap<i64, String> = HashMap::new();
    for r in rows {
        let (owner, label) = r?;
        if let Some(l) = label {
            // Prefer the first non-empty label per contact.
            out.entry(owner).or_insert(l);
        }
    }
    Ok(out)
}

fn collect_org_labels(conn: &Connection) -> Result<HashMap<i64, Vec<String>>, ExtractError> {
    // Contacts records the org both as a top-level field on ZABCDRECORD
    // and as an entity_type=2 record for company-only contacts.
    let mut stmt = conn.prepare(
        "SELECT Z_PK, ZORGANIZATION FROM ZABCDRECORD WHERE ZORGANIZATION IS NOT NULL",
    )?;
    let rows = stmt.query_map([], |row| {
        let pk: i64 = row.get(0)?;
        let org: Option<String> = row.get(1)?;
        Ok((pk, org))
    })?;
    let mut by_owner: HashMap<i64, Vec<String>> = HashMap::new();
    for r in rows {
        let (pk, org) = r?;
        if let Some(o) = org {
            if !o.is_empty() {
                by_owner.entry(pk).or_default().push(o);
            }
        }
    }
    Ok(by_owner)
}

fn compose_display_name(first: Option<&str>, last: Option<&str>, nick: Option<&str>, org: Option<&str>) -> Option<String> {
    let f = first.map(str::trim).filter(|s| !s.is_empty());
    let l = last.map(str::trim).filter(|s| !s.is_empty());
    if let (Some(f), Some(l)) = (f, l) {
        return Some(format!("{f} {l}"));
    }
    if let Some(f) = f { return Some(f.to_string()); }
    if let Some(l) = l { return Some(l.to_string()); }
    if let Some(n) = nick.map(str::trim).filter(|s| !s.is_empty()) {
        return Some(n.to_string());
    }
    org.map(str::trim).filter(|s| !s.is_empty()).map(String::from)
}

fn normalize_phone(raw: &str) -> String {
    // Keep digits and a leading '+'. Strip the rest.
    let mut out = String::with_capacity(raw.len());
    let mut first = true;
    for c in raw.chars() {
        if first && c == '+' { out.push('+'); first = false; continue; }
        if c.is_ascii_digit() { out.push(c); }
        first = false;
    }
    out
}

/// Convert Apple's `ZBIRTHDAY` (Mac absolute seconds since 2001-01-01) to
/// a "MM-DD" string. We drop the year because user-entered birthdays
/// often omit it.
fn apple_date_to_md(seconds_since_2001: f64) -> Option<String> {
    use chrono::{Datelike, Duration, TimeZone, Utc};
    let epoch = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    let dt = epoch + Duration::seconds(seconds_since_2001 as i64);
    Some(format!("{:02}-{:02}", dt.month(), dt.day()))
}
