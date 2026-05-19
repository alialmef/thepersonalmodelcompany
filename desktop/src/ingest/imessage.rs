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

const QUERY: &str = r#"
    SELECT
        m.ROWID            AS msg_id,
        m.text             AS text,
        m.is_from_me       AS is_from_me,
        m.date             AS date_ns,
        h.id               AS handle_id,
        c.chat_identifier  AS chat_id,
        c.display_name     AS chat_name
    FROM message m
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
    LEFT JOIN chat c ON c.ROWID = cmj.chat_id
    WHERE m.text IS NOT NULL AND length(m.text) > 0
    ORDER BY m.date ASC
"#;

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
pub fn open_chat_db(path: &PathBuf) -> Result<Connection, IngestError> {
    if !path.exists() {
        return Err(IngestError::NotFound);
    }
    let uri = format!("file:{}?mode=ro&immutable=1", path.display());
    Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .map_err(classify_open_error)
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
            let text: String = row
                .get::<_, Option<String>>("text")?
                .unwrap_or_default();
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

    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| IngestError::ReadError(e.to_string()))
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

    #[test]
    fn test_open_chat_db_not_found() {
        let path = PathBuf::from("/tmp/this-file-definitely-does-not-exist.db");
        match open_chat_db(&path) {
            Err(IngestError::NotFound) => {}
            other => panic!("expected NotFound, got {:?}", other),
        }
    }
}
