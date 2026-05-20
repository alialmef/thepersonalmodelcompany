//! Personal Model Company — desktop app entry point.
//!
//! Tauri commands exposed to the webview live in this crate. Native data
//! ingestion modules (iMessage, Apple Notes, Mail, WhatsApp) live in
//! `ingest::*` and are called via the commands defined here.

pub mod ingest;

use serde::Serialize;

use crate::ingest::{
    imessage,
    mail,
    notes,
    IngestError, IngestSummary, RawItemJson,
};

#[derive(Serialize)]
struct AppInfo {
    name: String,
    version: String,
    platform: String,
    backend_url: String,
}

fn backend_url() -> String {
    std::env::var("PMC_API_URL").unwrap_or_else(|_| "http://localhost:8000".to_string())
}

/// Identity info for the webview. Frontend uses this to detect Tauri mode
/// and to know which backend URL to hit.
#[tauri::command]
fn app_info() -> AppInfo {
    AppInfo {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        platform: std::env::consts::OS.to_string(),
        backend_url: backend_url(),
    }
}

#[tauri::command]
fn ping() -> &'static str {
    "pong"
}

// ---------------------------------------------------------------------------
// iMessage ingestion
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct IMessageStatus {
    chat_db_exists: bool,
    can_read: bool,
    message_count: Option<i64>,
    /// Error to surface in the UI ("PermissionDenied" → show Full Disk Access prompt)
    error: Option<String>,
}

/// Pre-flight check: can we read chat.db at all? Used by the Connect screen
/// to show "Ready to ingest" vs "Grant Full Disk Access first".
#[tauri::command]
fn imessage_status() -> IMessageStatus {
    let exists = imessage::chat_db_exists();
    if !exists {
        return IMessageStatus {
            chat_db_exists: false,
            can_read: false,
            message_count: None,
            error: Some("not_found".to_string()),
        };
    }
    let path = match imessage::default_chat_db_path() {
        Some(p) => p,
        None => {
            return IMessageStatus {
                chat_db_exists: false,
                can_read: false,
                message_count: None,
                error: Some("not_found".to_string()),
            };
        }
    };
    match imessage::open_chat_db(&path) {
        Ok(conn) => {
            let count = imessage::count_messages(&conn).ok();
            IMessageStatus {
                chat_db_exists: true,
                can_read: true,
                message_count: count,
                error: None,
            }
        }
        Err(IngestError::PermissionDenied) => IMessageStatus {
            chat_db_exists: true,
            can_read: false,
            message_count: None,
            error: Some("permission_denied".to_string()),
        },
        Err(e) => IMessageStatus {
            chat_db_exists: true,
            can_read: false,
            message_count: None,
            error: Some(format!("error:{}", e)),
        },
    }
}

/// Open the macOS Full Disk Access settings panel. Called from the UI when
/// `imessage_status` reports `permission_denied`.
#[tauri::command]
fn open_full_disk_access_settings() -> Result<(), String> {
    // x-apple.systempreferences URL scheme deep-links to the right panel.
    let url = "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles";
    std::process::Command::new("open")
        .arg(url)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Ingest iMessage messages. Reads chat.db, batches the RawItems, POSTs to
/// the backend's `/v1/users/{user_id}/sources/items` endpoint.
#[tauri::command]
async fn ingest_imessage(
    user_id: String,
    limit: Option<usize>,
) -> Result<IngestSummary, IngestError> {
    let path = imessage::default_chat_db_path().ok_or(IngestError::NotFound)?;
    let items = tokio::task::spawn_blocking(move || -> Result<Vec<RawItemJson>, IngestError> {
        let conn = imessage::open_chat_db(&path)?;
        imessage::read_messages(&conn, limit)
    })
    .await
    .map_err(|e| IngestError::Internal(format!("task join: {e}")))??;

    let count = items.len();
    if count == 0 {
        return Ok(IngestSummary {
            source: "imessage",
            source_id: "imessage-empty".to_string(),
            items_ingested: 0,
        });
    }

    let now = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let source_id = format!("imessage-{now}");

    let body = serde_json::json!({
        "kind": "imessage",
        "source_id": source_id,
        "items": items,
    });

    let url = format!(
        "{}/v1/users/{}/sources/items",
        backend_url(),
        urlencoding::encode(&user_id),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| IngestError::HttpError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(IngestError::HttpError(format!(
            "backend {status}: {text}"
        )));
    }

    Ok(IngestSummary {
        source: "imessage",
        source_id,
        items_ingested: count,
    })
}

// ---------------------------------------------------------------------------
// Mail ingestion (Apple Mail .emlx walker — sent folders only)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct MailStatus {
    chat_db_exists: bool,
    can_read: bool,
    message_count: Option<i64>,
    error: Option<String>,
}

/// Pre-flight check for Apple Mail. Returns the same shape as iMessage so
/// the JS bridge can use a single status type for all native sources.
#[tauri::command]
fn mail_status() -> MailStatus {
    let s = mail::status();
    MailStatus {
        chat_db_exists: s.exists,
        can_read: s.can_read,
        message_count: s.message_count,
        error: s.error,
    }
}

#[tauri::command]
async fn ingest_mail(
    user_id: String,
    limit: Option<usize>,
) -> Result<IngestSummary, IngestError> {
    let items = tokio::task::spawn_blocking(move || -> Result<Vec<RawItemJson>, IngestError> {
        mail::read_sent_mail(limit)
    })
    .await
    .map_err(|e| IngestError::Internal(format!("task join: {e}")))??;

    let count = items.len();
    if count == 0 {
        return Ok(IngestSummary {
            source: "email",
            source_id: "email-empty".to_string(),
            items_ingested: 0,
        });
    }

    let now = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let source_id = format!("email-{now}");
    post_items(&user_id, "email_mbox", &source_id, items).await?;
    Ok(IngestSummary {
        source: "email",
        source_id,
        items_ingested: count,
    })
}

// ---------------------------------------------------------------------------
// Notes / personal text files ingestion
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct NotesStatus {
    chat_db_exists: bool,
    can_read: bool,
    message_count: Option<i64>,
    error: Option<String>,
}

#[tauri::command]
fn notes_status() -> NotesStatus {
    let s = notes::status();
    NotesStatus {
        chat_db_exists: s.exists,
        can_read: s.can_read,
        message_count: s.message_count,
        error: s.error,
    }
}

#[tauri::command]
async fn ingest_notes(
    user_id: String,
    limit: Option<usize>,
) -> Result<IngestSummary, IngestError> {
    let items = tokio::task::spawn_blocking(move || -> Result<Vec<RawItemJson>, IngestError> {
        notes::read_notes(limit)
    })
    .await
    .map_err(|e| IngestError::Internal(format!("task join: {e}")))??;

    let count = items.len();
    if count == 0 {
        return Ok(IngestSummary {
            source: "text",
            source_id: "notes-empty".to_string(),
            items_ingested: 0,
        });
    }

    let now = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let source_id = format!("notes-{now}");
    post_items(&user_id, "text", &source_id, items).await?;
    Ok(IngestSummary {
        source: "text",
        source_id,
        items_ingested: count,
    })
}

/// Shared POST helper — every native ingester drops items at the same
/// backend endpoint via the same JSON shape.
async fn post_items(
    user_id: &str,
    kind: &str,
    source_id: &str,
    items: Vec<RawItemJson>,
) -> Result<(), IngestError> {
    let body = serde_json::json!({
        "kind": kind,
        "source_id": source_id,
        "items": items,
    });
    let url = format!(
        "{}/v1/users/{}/sources/items",
        backend_url(),
        urlencoding::encode(user_id),
    );
    let response = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| IngestError::HttpError(e.to_string()))?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(IngestError::HttpError(format!(
            "backend {status}: {text}"
        )));
    }
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            app_info,
            ping,
            imessage_status,
            open_full_disk_access_settings,
            ingest_imessage,
            mail_status,
            ingest_mail,
            notes_status,
            ingest_notes,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
