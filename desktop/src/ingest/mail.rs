//! Apple Mail native ingestion.
//!
//! Mail stores each message as a single `.emlx` file under
//! `~/Library/Mail/V<n>/<Account>/<Mailbox>/Messages/<NNNNNN>.emlx`. The
//! `.emlx` format is:
//!
//!   <length>\n
//!   <RFC 2822 message of that length>
//!   <Apple-specific binary plist trailer — we ignore>
//!
//! We focus on the user's *sent* mail (the writing they actually produced)
//! by preferring "Sent" mailbox names. This matches the principle baked into
//! `pmc/train/formatter.py`: the user's writing is the training target.
//!
//! Reading the Mail directory requires Full Disk Access on modern macOS —
//! same permission gate as iMessage.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::ingest::{IngestError, RawItemJson};

/// The Mail data root. Apple bumps the version directory (V8, V9, V10…)
/// every few macOS releases; we glob for the highest-numbered one we find.
fn mail_root() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let mail_dir = home.join("Library").join("Mail");
    if !mail_dir.exists() {
        return None;
    }

    // Pick the highest V<N> directory.
    let mut best: Option<(u32, PathBuf)> = None;
    for entry in fs::read_dir(&mail_dir).ok()?.flatten() {
        let name = entry.file_name();
        let name_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if let Some(rest) = name_str.strip_prefix('V') {
            if let Ok(version) = rest.parse::<u32>() {
                if best.as_ref().is_none_or(|(b, _)| version > *b) {
                    best = Some((version, entry.path()));
                }
            }
        }
    }
    best.map(|(_, path)| path)
}

/// Quick pre-flight: does Mail look readable? Counts approximate sent messages.
pub fn status() -> MailStatus {
    let root = match mail_root() {
        Some(p) => p,
        None => {
            return MailStatus {
                exists: false,
                can_read: false,
                message_count: None,
                error: Some("not_found".to_string()),
            };
        }
    };

    // Probe a single directory entry to detect EACCES vs OK.
    match fs::read_dir(&root) {
        Ok(_) => {
            // Try to count .emlx files in Sent mailboxes — best-effort.
            let count = count_sent_messages(&root);
            MailStatus {
                exists: true,
                can_read: true,
                message_count: Some(count),
                error: None,
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => MailStatus {
            exists: true,
            can_read: false,
            message_count: None,
            error: Some("permission_denied".to_string()),
        },
        Err(e) => MailStatus {
            exists: true,
            can_read: false,
            message_count: None,
            error: Some(format!("error:{}", e)),
        },
    }
}

#[derive(Debug, serde::Serialize)]
pub struct MailStatus {
    pub exists: bool,
    pub can_read: bool,
    pub message_count: Option<i64>,
    pub error: Option<String>,
}

/// Walk the Mail root, collecting paths to .emlx files inside Sent mailboxes.
/// Bounded depth: ~5 levels (V<n>/<account>/<mailbox>.mbox/Data/<bucket>/Messages).
fn find_sent_emlx(root: &Path, limit: Option<usize>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack: Vec<(PathBuf, u32)> = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = stack.pop() {
        if depth > 8 {
            continue;
        }
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Prune obviously non-mail dirs once we're past depth 0 to save IO.
                stack.push((path, depth + 1));
            } else if path.extension().and_then(|s| s.to_str()) == Some("emlx") {
                // Only include if the parent path mentions Sent (case-insensitive).
                let p_str = path.to_string_lossy().to_lowercase();
                if p_str.contains("sent") {
                    out.push(path);
                    if let Some(l) = limit {
                        if out.len() >= l {
                            return out;
                        }
                    }
                }
            }
        }
    }
    out
}

fn count_sent_messages(root: &Path) -> i64 {
    // For speed, walk once with a high cap. We just want an order-of-magnitude.
    find_sent_emlx(root, Some(20_000)).len() as i64
}

/// Parse a single .emlx file: read the length prefix, take that many bytes
/// of RFC 2822, drop the plist trailer.
fn parse_emlx(path: &Path) -> Result<EmlxMessage, IngestError> {
    let mut file = fs::File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            IngestError::PermissionDenied
        } else {
            IngestError::ReadError(e.to_string())
        }
    })?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| IngestError::ReadError(e.to_string()))?;

    // First line: byte length of the RFC 2822 portion.
    let newline = bytes.iter().position(|&b| b == b'\n').ok_or_else(|| {
        IngestError::ReadError(".emlx missing length-prefix newline".to_string())
    })?;
    let len_str = std::str::from_utf8(&bytes[..newline])
        .map_err(|_| IngestError::ReadError(".emlx length not utf8".to_string()))?;
    let msg_len: usize = len_str
        .trim()
        .parse()
        .map_err(|_| IngestError::ReadError(format!(".emlx length not a number: {:?}", len_str)))?;

    let start = newline + 1;
    let end = (start + msg_len).min(bytes.len());
    let msg_bytes = &bytes[start..end];

    // Crude RFC 2822 split: headers, blank line, body.
    let msg_str = String::from_utf8_lossy(msg_bytes);
    let mut subject = None;
    let mut date_hdr = None;
    let mut from = None;
    let mut to = None;
    let mut header_end = msg_str.len();
    let mut lines = msg_str.split_inclusive('\n');
    let mut byte_pos = 0;
    let mut last_header_value = String::new();
    let mut last_header_name: Option<String> = None;

    while let Some(line) = lines.next() {
        let trim = line.trim_end_matches(|c: char| c == '\r' || c == '\n');
        if trim.is_empty() {
            header_end = byte_pos + line.len();
            break;
        }
        // RFC 2822 continuation: lines starting with whitespace continue the
        // previous header value.
        if let Some(rest) = trim.strip_prefix(|c: char| c == ' ' || c == '\t') {
            last_header_value.push(' ');
            last_header_value.push_str(rest.trim_start());
        } else {
            // Flush previous header.
            if let Some(name) = last_header_name.take() {
                set_header(&name, &last_header_value, &mut subject, &mut date_hdr, &mut from, &mut to);
                last_header_value.clear();
            }
            if let Some((name, value)) = trim.split_once(':') {
                last_header_name = Some(name.trim().to_ascii_lowercase());
                last_header_value.push_str(value.trim());
            }
        }
        byte_pos += line.len();
    }
    // Flush last header.
    if let Some(name) = last_header_name.take() {
        set_header(&name, &last_header_value, &mut subject, &mut date_hdr, &mut from, &mut to);
    }

    let body = msg_str[header_end..].trim().to_string();

    Ok(EmlxMessage {
        subject,
        date: date_hdr.and_then(|s| parse_rfc2822_date(&s)),
        from,
        to,
        body: clean_body(&body),
    })
}

fn set_header(
    name: &str,
    value: &str,
    subject: &mut Option<String>,
    date: &mut Option<String>,
    from: &mut Option<String>,
    to: &mut Option<String>,
) {
    match name {
        "subject" => *subject = Some(value.to_string()),
        "date" => *date = Some(value.to_string()),
        "from" => *from = Some(value.to_string()),
        "to" => *to = Some(value.to_string()),
        _ => {}
    }
}

fn parse_rfc2822_date(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc2822(s).ok().map(|dt| dt.with_timezone(&Utc))
}

/// Strip quoted-reply lines (those starting with '>') and signatures so the
/// trained model focuses on what the user actually wrote in this message.
fn clean_body(body: &str) -> String {
    let mut out_lines = Vec::new();
    for raw in body.lines() {
        let line = raw.trim_end();
        // Drop quoted replies — common when forwarding/replying.
        if line.starts_with('>') {
            continue;
        }
        // Stop at canonical signature delimiter.
        if line == "-- " {
            break;
        }
        // Stop at "On <date>, <person> wrote:" lines from Apple Mail replies.
        if line.starts_with("On ") && line.contains("wrote:") {
            break;
        }
        out_lines.push(line);
    }
    out_lines.join("\n").trim().to_string()
}

#[derive(Debug)]
struct EmlxMessage {
    subject: Option<String>,
    date: Option<DateTime<Utc>>,
    from: Option<String>,
    to: Option<String>,
    body: String,
}

/// Read the Mac's Mail directory and return RawItems for all sent messages.
pub fn read_sent_mail(limit: Option<usize>) -> Result<Vec<RawItemJson>, IngestError> {
    let root = mail_root().ok_or(IngestError::NotFound)?;
    let paths = find_sent_emlx(&root, limit);

    let mut items = Vec::with_capacity(paths.len());
    for path in paths {
        // Soft-fail per file: a corrupt .emlx shouldn't kill the whole pass.
        let msg = match parse_emlx(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        // Skip empties — no point training on a one-line "thanks!" we can't
        // even tell is from the user.
        if msg.body.trim().is_empty() {
            continue;
        }
        let mut metadata = HashMap::new();
        if let Some(t) = &msg.to {
            metadata.insert("to".to_string(), t.clone());
        }
        if let Some(f) = &msg.from {
            metadata.insert("from".to_string(), f.clone());
        }
        let item = RawItemJson {
            source_type: "email",
            source_id: path.to_string_lossy().to_string(),
            content: msg.body,
            timestamp: msg.date.map(|d| d.to_rfc3339()),
            thread_id: None,
            author_identifier: msg.from.clone(),
            is_user: Some(true), // sent = user-authored by definition
            subject: msg.subject,
            metadata,
        };
        items.push(item);
    }
    Ok(items)
}
