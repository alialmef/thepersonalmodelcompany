//! Personal Model Company — desktop app entry point.
//!
//! Tauri commands exposed to the webview live in this crate. Native data
//! ingestion modules (iMessage, Apple Notes, Mail, WhatsApp) live in
//! `ingest::*` and are called via the commands defined here.
//!
//! Personal knowledge graph: `graph::*` defines the typed entities and
//! `store`; `extract::*` writes to it from each local Mac source;
//! `synthesis::*` cross-resolves entities and detects themes;
//! `schedule::*` runs the whole pipeline continuously in the background.

pub mod ingest;
pub mod graph;
pub mod extract;
pub mod synthesis;
pub mod schedule;

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
            source_id: "imessage".to_string(),
            items_ingested: 0,
        });
    }

    // Stable per-kind source_id: re-ingesting overwrites the canonical
    // raw file (UserStore.save_raw_items defaults to write mode) instead
    // of appending a new timestamped file every time. Without this the
    // raw/ directory accumulates duplicates and the curate pipeline
    // re-processes the same content N times.
    let source_id = "imessage".to_string();

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
            source_id: "email".to_string(),
            items_ingested: 0,
        });
    }

    // Stable per-kind source_id — see ingest_imessage for rationale.
    let source_id = "email".to_string();
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
            source_id: "notes".to_string(),
            items_ingested: 0,
        });
    }

    // Stable per-kind source_id — see ingest_imessage for rationale.
    let source_id = "notes".to_string();
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

// ---------------------------------------------------------------------------
// Documents — multipart-upload each picked file to /sources/upload so the
// backend can extract text (PDFs, DOCX) via its own parsers. .txt / .md go
// through as kind="text" so they end up as plain RawItems.
// ---------------------------------------------------------------------------

#[tauri::command]
async fn ingest_documents(
    user_id: String,
    paths: Vec<String>,
) -> Result<IngestSummary, IngestError> {
    use std::path::Path;
    let client = reqwest::Client::new();
    let now = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let source_id = format!("documents-{now}");
    let mut total: usize = 0;

    for path_str in &paths {
        let path = Path::new(path_str);
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| IngestError::ReadError(format!("{}: {e}", path.display())))?;
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("upload")
            .to_string();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        let kind = match ext.as_str() {
            "txt" | "md" | "markdown" | "mdx" | "rtf" => "text",
            _ => "document",
        };

        let part = reqwest::multipart::Part::bytes(bytes).file_name(filename.clone());
        let form = reqwest::multipart::Form::new()
            .text("kind", kind)
            .text("source_id", source_id.clone())
            .part("file", part);
        let url = format!(
            "{}/v1/users/{}/sources/upload",
            backend_url(),
            urlencoding::encode(&user_id),
        );
        let response = client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| IngestError::HttpError(e.to_string()))?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(IngestError::HttpError(format!(
                "backend {status} uploading {filename}: {text}"
            )));
        }
        let parsed: serde_json::Value = response
            .json()
            .await
            .map_err(|e| IngestError::HttpError(e.to_string()))?;
        if let Some(n) = parsed.get("raw_items_ingested").and_then(|v| v.as_u64()) {
            total += n as usize;
        }
    }

    Ok(IngestSummary {
        source: "document",
        source_id,
        items_ingested: total,
    })
}

// ---------------------------------------------------------------------------
// Personal knowledge graph commands
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct GraphSummary {
    root: String,
    counts: std::collections::HashMap<String, usize>,
}

fn graph_root_for_user(user_id: &str) -> std::path::PathBuf {
    // The local dev backend is at `$HOME/.pmc-dev/storage/users/<uid>/graph/`.
    // We mirror that path here so the agent and the extractors share a store.
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    home.join(".pmc-dev/storage/users").join(user_id).join("graph")
}

fn watermarks_path_for_user(user_id: &str) -> std::path::PathBuf {
    graph_root_for_user(user_id).join("_watermarks.json")
}

/// Open a `GraphStore` for the given user (creating the dir if needed).
fn open_store(user_id: &str) -> Result<std::sync::Arc<graph::GraphStore>, String> {
    let root = graph_root_for_user(user_id);
    graph::GraphStore::new(&root)
        .map(std::sync::Arc::new)
        .map_err(|e| format!("graph store: {e}"))
}

/// Run every extractor + synthesis once. Returns per-source summaries.
#[tauri::command]
async fn graph_run_full(user_id: String) -> Result<Vec<extract::ExtractSummary>, String> {
    let store = open_store(&user_id)?;
    let watermarks = graph::Watermarks::load(watermarks_path_for_user(&user_id));
    let (scheduler, _events) = schedule::Scheduler::new(store, watermarks);
    Ok(scheduler.run_full().await)
}

/// Wipe every trace of this user — raw data, graph, memory, training
/// bundles, registered adapter — and return so the webview can clear
/// its own state and route back to /welcome. The user comes through
/// the connect flow again as if it were first launch. Irreversible.
#[tauri::command]
async fn reset_user(user_id: String) -> Result<(), String> {
    let url = format!(
        "{}/v1/users/{}/reset",
        backend_url(),
        urlencoding::encode(&user_id),
    );
    let response = reqwest::Client::new()
        .post(&url)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("backend {status}: {text}"));
    }
    Ok(())
}

/// Fire-and-forget graph extraction for the /connect → /reading
/// handoff. The webview calls this once after the user has connected
/// their sources; it spawns the full extraction in the background and
/// returns immediately so the user can advance to /reading while the
/// graph (contacts, calendar, photos metadata, etc.) populates. The
/// running scheduler picks up incremental updates from there on.
///
/// MUST be `async fn` so we're inside a Tokio runtime context when we
/// call `tokio::spawn`. A synchronous Tauri command runs on the
/// webview thread which has no tokio runtime bound — `tokio::spawn`
/// from there panics → SIGABRT → the whole app dies. Found this the
/// hard way: app crashed cleanly on /connect → /reading transition.
#[tauri::command]
async fn graph_kickoff(user_id: String) -> Result<(), String> {
    use std::sync::OnceLock;
    static IN_FLIGHT: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let set = IN_FLIGHT.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    {
        let mut guard = set.lock().map_err(|_| "lock poisoned".to_string())?;
        if !guard.insert(user_id.clone()) { return Ok(()); }
    }
    let user_id_for_task = user_id.clone();
    tokio::spawn(async move {
        let store = match open_store(&user_id_for_task) { Ok(s) => s, Err(_) => return };
        let watermarks = graph::Watermarks::load(watermarks_path_for_user(&user_id_for_task));
        let (scheduler, _events) = schedule::Scheduler::new(store, watermarks);
        let _ = scheduler.run_full().await;
        if let Some(set) = IN_FLIGHT.get() {
            if let Ok(mut guard) = set.lock() {
                guard.remove(&user_id_for_task);
            }
        }
    });
    Ok(())
}

/// Counts per entity-kind file in the user's graph store. Cheap; used by
/// the UI to show "what's in the graph" without loading everything.
#[tauri::command]
fn graph_counts(user_id: String) -> Result<GraphSummary, String> {
    let store = open_store(&user_id)?;
    let mut counts = std::collections::HashMap::new();
    for kind in [
        graph::EntityKind::Person, graph::EntityKind::Place,
        graph::EntityKind::Event, graph::EntityKind::Episode,
        graph::EntityKind::Project, graph::EntityKind::Theme,
        graph::EntityKind::OpenLoop, graph::EntityKind::TasteItem,
        graph::EntityKind::FileSignal, graph::EntityKind::CodeRepo,
        graph::EntityKind::WebSignal, graph::EntityKind::Edge,
    ] {
        counts.insert(kind.filename().to_string(), store.count(kind));
    }
    Ok(GraphSummary {
        root: store.root().to_string_lossy().to_string(),
        counts,
    })
}

/// Sample N entries of one entity kind — for the "let's look at it"
/// inspection step. Returns raw JSON values for client-side rendering.
#[tauri::command]
fn graph_sample(
    user_id: String,
    kind: String,
    limit: usize,
) -> Result<Vec<serde_json::Value>, String> {
    let store = open_store(&user_id)?;
    let ek = match kind.as_str() {
        "people"     | "person"     => graph::EntityKind::Person,
        "places"     | "place"      => graph::EntityKind::Place,
        "events"     | "event"      => graph::EntityKind::Event,
        "episodes"   | "episode"    => graph::EntityKind::Episode,
        "projects"   | "project"    => graph::EntityKind::Project,
        "themes"     | "theme"      => graph::EntityKind::Theme,
        "open_loops" | "openloop"   => graph::EntityKind::OpenLoop,
        "taste"      | "tasteitem"  => graph::EntityKind::TasteItem,
        "files"      | "filesignal" => graph::EntityKind::FileSignal,
        "repos"      | "coderepo"   => graph::EntityKind::CodeRepo,
        "web"        | "websignal"  => graph::EntityKind::WebSignal,
        "edges"      | "edge"       => graph::EntityKind::Edge,
        _ => return Err(format!("unknown entity kind: {kind}")),
    };
    let values: Vec<serde_json::Value> = store
        .load::<serde_json::Value>(ek)
        .map_err(|e| format!("load: {e}"))?;
    Ok(values.into_iter().take(limit.max(1)).collect())
}

/// Start the always-on background scheduler for this user. Safe to
/// call multiple times — second-and-later calls are a no-op.
#[tauri::command]
fn graph_start_scheduler(user_id: String) -> Result<(), String> {
    use std::sync::OnceLock;
    static STARTED: OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let started = STARTED.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    {
        let mut s = started.lock().map_err(|_| "lock poisoned".to_string())?;
        if !s.insert(user_id.clone()) { return Ok(()); }
    }
    let store = open_store(&user_id)?;
    let watermarks = graph::Watermarks::load(watermarks_path_for_user(&user_id));
    let (scheduler, mut events) = schedule::Scheduler::new(store, watermarks);
    scheduler.start();
    // Drain the event stream into stdout for debugging; a future
    // tauri::AppHandle wire-up will forward to the frontend.
    tokio::spawn(async move {
        while let Some(ev) = events.recv().await {
            let _ = serde_json::to_string(&ev).map(|s| eprintln!("[graph] {s}"));
        }
    });
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
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
            ingest_documents,
            graph_run_full,
            graph_kickoff,
            graph_counts,
            graph_sample,
            graph_start_scheduler,
            reset_user,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
