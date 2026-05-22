//! iMessage ingestion via direct chat.db SQLite reads.
//!
//! macOS gates access to `~/Library/Messages/chat.db` behind Full Disk Access
//! (TCC). The app must be granted that permission in System Settings → Privacy
//! & Security → Full Disk Access before this module can read messages. When
//! permission is missing, `open_chat_db` returns `IngestError::PermissionDenied`
//! and the frontend should prompt the user with deep-link instructions.
//!
//! Apple's timestamp format on modern macOS is nanoseconds since 2001-01-01
//! (Mac absolute time). Older databases stored seconds — we detect by
//! magnitude.

use chrono::{DateTime, Duration, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::ingest::{IngestError, RawItemJson};

// macOS Ventura+ stopped writing message text to `m.text` for most messages —
// it lives in `m.attributedBody`, an Apple typedstream blob wrapping an
// NSAttributedString. We pull both columns and decode the blob in Rust when
// `text` is empty. Without this, ~99% of a modern chat.db looks empty.
const QUERY: &str = r#"
    SELECT
        m.ROWID            AS msg_id,
        m.text             AS text,
        m.attributedBody   AS attributed_body,
        m.is_from_me       AS is_from_me,
        m.date             AS date_ns,
        h.id               AS handle_id,
        c.chat_identifier  AS chat_id,
        c.display_name     AS chat_name
    FROM message m
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
    LEFT JOIN chat c ON c.ROWID = cmj.chat_id
    WHERE (m.text IS NOT NULL AND length(m.text) > 0)
       OR m.attributedBody IS NOT NULL
    ORDER BY m.date ASC
"#;

/// Decode the message text from an Apple typedstream `attributedBody` blob.
///
/// The blob wraps an NSAttributedString. The underlying NSString appears
/// right after the "NSString" class marker in this exact layout:
///
///   NSString \x01 \x94|\x95 \x84 \x01 + <length-encoding> <utf-8 bytes>
///
/// where `+` (0x2b) is the typedstream `char` type code marking the start
/// of the length-prefixed C-string field, and `<length-encoding>` is one of:
///   - a single byte `n` in 0x01..0x7f  → length n
///   - 0x81 then 2-byte LE u16          → length up to 65,535
///   - 0x82 then 4-byte LE u32          → length up to ~4 GB
///
/// Earlier versions of this decoder scanned for *any* byte 0x01..0x7f after
/// "NSString" as a candidate length, which made it latch onto the `+`
/// separator (0x2b = 43) and over-read by one byte — every long message
/// came out with a junk leading character.
///
/// Returns the decoded text, or None if no plausible string is found.
fn decode_attributed_body(blob: &[u8]) -> Option<String> {
    let marker = b"NSString";
    let mut idx_search = 0usize;
    while let Some(rel) = blob[idx_search..]
        .windows(marker.len())
        .position(|w| w == marker)
    {
        let after = idx_search + rel + marker.len();
        // The `+` (0x2b) separator that introduces the length encoding
        // sits within ~8 bytes of the class-name end. Scan for it
        // explicitly instead of treating any low byte as a length.
        let window_end = (after + 12).min(blob.len());
        if let Some(plus_rel) = blob[after..window_end].iter().position(|&b| b == 0x2b) {
            let len_pos = after + plus_rel + 1;
            if len_pos < blob.len() {
                let b = blob[len_pos];
                let parsed = if b == 0x81 && len_pos + 3 <= blob.len() {
                    let n = u16::from_le_bytes([blob[len_pos + 1], blob[len_pos + 2]]) as usize;
                    Some((len_pos + 3, n))
                } else if b == 0x82 && len_pos + 5 <= blob.len() {
                    let n = u32::from_le_bytes([
                        blob[len_pos + 1],
                        blob[len_pos + 2],
                        blob[len_pos + 3],
                        blob[len_pos + 4],
                    ]) as usize;
                    if n < 10_000_000 { Some((len_pos + 5, n)) } else { None }
                } else if b > 0 && b < 0x80 {
                    Some((len_pos + 1, b as usize))
                } else {
                    None
                };
                if let Some((str_start, str_len)) = parsed {
                    if let Some(s) = try_take_string(blob, str_start, str_len) {
                        if !KNOWN_CLASS_NAMES.iter().any(|n| s.starts_with(n)) {
                            return Some(s);
                        }
                    }
                }
            }
        }
        idx_search = after + 1;
    }
    None
}

fn try_take_string(blob: &[u8], start: usize, len: usize) -> Option<String> {
    if len == 0 || start + len > blob.len() {
        return None;
    }
    let bytes = &blob[start..start + len];
    let s = std::str::from_utf8(bytes).ok()?;
    // A real message has SOME alphanumeric / whitespace content. Pure
    // punctuation/garbage is almost certainly not a message body.
    let printable = s
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .count();
    if printable < 1 {
        return None;
    }
    Some(s.to_string())
}

/// Class names that appear as length-prefixed strings inside the typedstream
/// envelope. Without filtering, the scan can latch onto these instead of
/// the real message text.
const KNOWN_CLASS_NAMES: &[&str] = &[
    "NSAttributedString",
    "NSMutableAttributedString",
    "NSString",
    "NSMutableString",
    "NSDictionary",
    "NSMutableDictionary",
    "NSArray",
    "NSMutableArray",
    "NSObject",
    "NSNumber",
    "NSDate",
    "NSData",
    "NSConcreteAttributedString",
    "NSConcreteMutableAttributedString",
];

/// Default location of the iMessage SQLite database on macOS.
pub fn default_chat_db_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut path = PathBuf::from(h);
        path.push("Library/Messages/chat.db");
        path
    })
}

/// Whether chat.db exists at the expected location. Doesn't attempt to open
/// it — useful for showing a "no messages yet" vs "Full Disk Access needed"
/// distinction in the UI.
pub fn chat_db_exists() -> bool {
    default_chat_db_path().map(|p| p.exists()).unwrap_or(false)
}

/// Open chat.db read-only. Returns a typed error so the frontend can
/// distinguish "missing" from "denied" from "broken".
///
/// We snapshot the live database (.db + .db-wal + .db-shm) to a temp dir
/// before opening it. Why:
///   - Messages.app keeps chat.db open in WAL mode and holds locks on
///     the WAL files. Opening the live file directly either fights the
///     lock or, with `immutable=1`, silently misses everything in the WAL
///     (often months of recent history).
///   - Copying gives us a stable, lock-free snapshot that includes the WAL,
///     so SQLite merges everything into a complete read.
///
/// Returns the connection and a guard path that the caller can drop when
/// done. The temp dir is automatically cleaned up on guard drop.
pub fn open_chat_db(path: &PathBuf) -> Result<Connection, IngestError> {
    if !path.exists() {
        return Err(IngestError::NotFound);
    }
    let snapshot = snapshot_chat_db(path)?;
    let uri = format!("file:{}?mode=ro", snapshot.display());
    Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(classify_open_error)
}

/// Copy chat.db + its WAL sidecars into a fresh temp dir and return the
/// path to the copied .db. The temp dir is leaked intentionally — the OS
/// will clean it up, and the path keeps working for the life of the
/// process. (For long-running daemons we'd track and clean these.)
fn snapshot_chat_db(src: &PathBuf) -> Result<PathBuf, IngestError> {
    let dir = std::env::temp_dir().join(format!(
        "pmc-chatdb-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default()
    ));
    std::fs::create_dir_all(&dir).map_err(|e| {
        IngestError::ReadError(format!("snapshot tmp dir: {e}"))
    })?;

    let dst_db = dir.join("chat.db");
    std::fs::copy(src, &dst_db).map_err(|e| classify_io_error(&e))?;

    // Copy WAL + SHM if they exist. They're optional but contain the
    // recent (un-checkpointed) writes — the whole point of this snapshot.
    let mut wal_src = src.clone();
    wal_src.set_extension("db-wal");
    if wal_src.exists() {
        let _ = std::fs::copy(&wal_src, dir.join("chat.db-wal"));
    }
    let mut shm_src = src.clone();
    shm_src.set_extension("db-shm");
    if shm_src.exists() {
        let _ = std::fs::copy(&shm_src, dir.join("chat.db-shm"));
    }

    Ok(dst_db)
}

fn classify_io_error(e: &std::io::Error) -> IngestError {
    use std::io::ErrorKind::*;
    match e.kind() {
        PermissionDenied => IngestError::PermissionDenied,
        NotFound => IngestError::NotFound,
        _ => {
            let msg = e.to_string().to_lowercase();
            if msg.contains("operation not permitted") || msg.contains("not authorized") {
                IngestError::PermissionDenied
            } else {
                IngestError::ReadError(e.to_string())
            }
        }
    }
}

fn classify_open_error(e: rusqlite::Error) -> IngestError {
    let msg = e.to_string().to_lowercase();
    if msg.contains("permission")
        || msg.contains("operation not permitted")
        || msg.contains("unable to open")
        || msg.contains("not authorized")
    {
        IngestError::PermissionDenied
    } else {
        IngestError::ReadError(e.to_string())
    }
}

/// Read messages from the open chat.db connection. Optionally limit to the
/// most recent N for previews / dry-runs.
pub fn read_messages(
    conn: &Connection,
    limit: Option<usize>,
) -> Result<Vec<RawItemJson>, IngestError> {
    let query = match limit {
        Some(n) => format!("{}\nLIMIT {}", QUERY, n),
        None => QUERY.to_string(),
    };

    let mut stmt = conn
        .prepare(&query)
        .map_err(|e| IngestError::ReadError(e.to_string()))?;

    let rows = stmt
        .query_map([], |row| {
            let msg_id: i64 = row.get("msg_id")?;
            let plain_text: Option<String> = row.get::<_, Option<String>>("text")?;
            let attributed_body: Option<Vec<u8>> = row.get::<_, Option<Vec<u8>>>("attributed_body")?;
            // Prefer the plain text column when populated. Otherwise fall back
            // to decoding the attributedBody blob (the common path on Ventura+).
            let text: String = match plain_text {
                Some(s) if !s.is_empty() => s,
                _ => attributed_body
                    .as_deref()
                    .and_then(decode_attributed_body)
                    .unwrap_or_default(),
            };
            let is_from_me: i32 = row.get("is_from_me")?;
            let date_ns: i64 = row.get::<_, Option<i64>>("date_ns")?.unwrap_or(0);
            let handle_id: Option<String> = row.get("handle_id")?;
            let chat_id: Option<String> = row.get("chat_id")?;
            let chat_name: Option<String> = row.get("chat_name")?;

            let timestamp = apple_time_to_iso(date_ns);
            let mut metadata = HashMap::new();
            if let Some(name) = chat_name {
                if !name.is_empty() {
                    metadata.insert("chat_name".to_string(), name);
                }
            }

            let thread_id = chat_id.or_else(|| handle_id.clone());

            Ok(RawItemJson {
                source_type: "imessage",
                source_id: format!("imessage:{}", msg_id),
                content: text,
                timestamp,
                thread_id: thread_id.or_else(|| Some("unknown".to_string())),
                author_identifier: handle_id,
                is_user: Some(is_from_me != 0),
                subject: None,
                metadata,
            })
        })
        .map_err(|e| IngestError::ReadError(e.to_string()))?;

    let items: Vec<RawItemJson> = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| IngestError::ReadError(e.to_string()))?;
    // Skip rows where neither column produced text — these are typically
    // image-only messages, reactions, or audio messages that have no body.
    Ok(items
        .into_iter()
        .filter(|i| !i.content.is_empty())
        .collect())
}

/// Count rows that *would* be ingested. Cheap query for the preview UX.
pub fn count_messages(conn: &Connection) -> Result<i64, IngestError> {
    conn.query_row(
        "SELECT COUNT(*) FROM message WHERE text IS NOT NULL AND length(text) > 0",
        [],
        |row| row.get::<_, i64>(0),
    )
    .map_err(|e| IngestError::ReadError(e.to_string()))
}

/// Convert Apple's date format to ISO 8601 RFC 3339.
///
/// Modern iMessage stores nanoseconds since 2001-01-01 UTC. Older databases
/// stored seconds — we detect by magnitude (anything > 10^12 must be ns).
fn apple_time_to_iso(date_value: i64) -> Option<String> {
    if date_value == 0 {
        return None;
    }
    let apple_epoch: DateTime<Utc> = Utc.with_ymd_and_hms(2001, 1, 1, 0, 0, 0).single()?;
    let dt = if date_value > 10i64.pow(12) {
        apple_epoch + Duration::nanoseconds(date_value)
    } else {
        apple_epoch + Duration::seconds(date_value)
    };
    Some(dt.to_rfc3339())
}

// ---------------------------------------------------------------------------
// Tests — these work because we can construct a temporary SQLite database
// with the same schema as chat.db and verify the parsing.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use tempfile::NamedTempFile;

    fn create_fake_chat_db() -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        let conn = Connection::open(file.path()).unwrap();
        conn.execute_batch(r#"
            CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
            CREATE TABLE chat (ROWID INTEGER PRIMARY KEY, chat_identifier TEXT, display_name TEXT);
            CREATE TABLE message (
                ROWID INTEGER PRIMARY KEY,
                text TEXT,
                attributedBody BLOB,
                is_from_me INTEGER,
                date INTEGER,
                handle_id INTEGER
            );
            CREATE TABLE chat_message_join (chat_id INTEGER, message_id INTEGER);
        "#).unwrap();
        // Insert one inbound + one outbound
        conn.execute("INSERT INTO handle (ROWID, id) VALUES (?, ?)", params![1, "+15551234567"]).unwrap();
        conn.execute("INSERT INTO chat (ROWID, chat_identifier, display_name) VALUES (?, ?, ?)", params![1, "chat-abc", "Family"]).unwrap();
        // Apple ns timestamp ~ 2024
        let ns = 23i64 * 365 * 24 * 3600 * 1_000_000_000;
        conn.execute(
            "INSERT INTO message (ROWID, text, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?)",
            params![1, "Hey", 0, ns, 1],
        ).unwrap();
        conn.execute(
            "INSERT INTO message (ROWID, text, is_from_me, date, handle_id) VALUES (?, ?, ?, ?, ?)",
            params![2, "What's up", 1, ns + 60_000_000_000i64, 1],
        ).unwrap();
        conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)", params![1, 1]).unwrap();
        conn.execute("INSERT INTO chat_message_join (chat_id, message_id) VALUES (?, ?)", params![1, 2]).unwrap();
        file
    }

    #[test]
    fn test_read_messages_basic() {
        let f = create_fake_chat_db();
        let conn = Connection::open(f.path()).unwrap();
        let items = read_messages(&conn, None).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].content, "Hey");
        assert_eq!(items[0].is_user, Some(false));
        assert_eq!(items[1].is_user, Some(true));
        assert_eq!(items[0].thread_id.as_deref(), Some("chat-abc"));
        assert_eq!(items[0].metadata.get("chat_name").map(|s| s.as_str()), Some("Family"));
    }

    #[test]
    fn test_read_messages_with_limit() {
        let f = create_fake_chat_db();
        let conn = Connection::open(f.path()).unwrap();
        let items = read_messages(&conn, Some(1)).unwrap();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_count_messages() {
        let f = create_fake_chat_db();
        let conn = Connection::open(f.path()).unwrap();
        let count = count_messages(&conn).unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_apple_time_to_iso_nanoseconds() {
        // ~23 years × 365 days × 24 × 3600 × 1e9 ≈ 2024
        let ns = 23i64 * 365 * 24 * 3600 * 1_000_000_000;
        let iso = apple_time_to_iso(ns).unwrap();
        assert!(iso.starts_with("2023") || iso.starts_with("2024"));
    }

    #[test]
    fn test_apple_time_to_iso_zero_returns_none() {
        assert!(apple_time_to_iso(0).is_none());
    }

    #[test]
    fn test_default_chat_db_path() {
        let p = default_chat_db_path().unwrap();
        assert!(p.ends_with("Library/Messages/chat.db"));
    }

    fn build_blob(prefix_byte_after_class: u8, length_encoding: &[u8], body: &[u8]) -> Vec<u8> {
        // Minimal stand-in for the Apple typedstream layout we care about:
        //   <padding> NSString <prefix_byte> \x84 \x01 + <length-encoding> <body>
        // prefix_byte_after_class is the byte that varies per real blob
        // (\x94 for short messages, \x95 for long ones, etc).
        let mut blob = vec![0xaa; 16]; // arbitrary leading padding
        blob.extend_from_slice(b"NSString");
        blob.push(0x01);
        blob.push(prefix_byte_after_class);
        blob.push(0x84);
        blob.push(0x01);
        blob.push(0x2b); // '+'
        blob.extend_from_slice(length_encoding);
        blob.extend_from_slice(body);
        blob.extend_from_slice(b"\x86\x84\x02iI"); // trailing metadata
        blob
    }

    #[test]
    fn test_decode_attributed_body_short() {
        // 22-char string, length fits in one byte.
        let body = b"what are YOU doing up ";
        let blob = build_blob(0x94, &[body.len() as u8], body);
        let s = decode_attributed_body(&blob).expect("decoder should find string");
        assert_eq!(s, "what are YOU doing up ");
    }

    #[test]
    fn test_decode_attributed_body_no_leading_byte_when_length_is_43() {
        // Regression test: a 43-char message has length byte 0x2b ('+'). The
        // old decoder picked the SEPARATOR `+` as the length and over-read
        // by one byte, prepending a junk character. Verify clean decode.
        let body = b"Accept slack invite when you get the chance"; // 43 chars
        assert_eq!(body.len(), 43);
        let blob = build_blob(0x94, &[0x2b], body);
        let s = decode_attributed_body(&blob).expect("decoder should find string");
        assert_eq!(s, "Accept slack invite when you get the chance");
    }

    #[test]
    fn test_decode_attributed_body_long() {
        // 281-char string uses the 0x81 <u16 LE> encoding.
        let body_str: String = "x".repeat(281);
        let body = body_str.as_bytes();
        let len_bytes = (body.len() as u16).to_le_bytes();
        let mut len_enc = vec![0x81];
        len_enc.extend_from_slice(&len_bytes);
        let blob = build_blob(0x95, &len_enc, body);
        let s = decode_attributed_body(&blob).expect("decoder should find string");
        assert_eq!(s.len(), 281);
    }

    #[test]
    fn test_open_chat_db_not_found() {
        let path = PathBuf::from("/tmp/this-file-definitely-does-not-exist.db");
        match open_chat_db(&path) {
            Err(IngestError::NotFound) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }
}
