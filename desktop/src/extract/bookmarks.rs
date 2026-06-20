//! Safari bookmarks + Reading List extractor.
//!
//! Reads `~/Library/Safari/Bookmarks.plist`. The file is a nested tree
//! of bookmark folders and leaves (`WebBookmarkTypeProxy` for system
//! folders, `WebBookmarkTypeList` for user folders,
//! `WebBookmarkTypeLeaf` for actual bookmarks).
//!
//! We walk the whole tree, capture each leaf's title + URL + folder
//! path, and emit a `TasteItem` per bookmark. The Reading List is one
//! of Safari's top-level folders (titled "com.apple.ReadingList" by
//! identifier); we tag those with `kind="reading_list"` so the
//! synthesis layer can treat saved-to-read articles separately from
//! permanent bookmarks.
//!
//! Bookmarks are *explicit curation* — the user actively chose to
//! save this. That makes them much higher-signal than browser
//! history, which is just whatever the user happened to visit.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, TasteItem};
use crate::graph::store::stable_id;
use plist::Value;
use std::path::PathBuf;

const SOURCE: &str = "safari_bookmarks";

fn default_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Safari/Bookmarks.plist");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped("Safari Bookmarks.plist not present"));
    }

    let root: Value = match plist::from_file(&path) {
        Ok(v) => v,
        Err(e) => {
            // Permission errors come through here on macOS when FDA
            // isn't granted — propagate as PermissionDenied so the
            // caller can surface the right prompt.
            let s = e.to_string();
            if s.contains("Permission denied") || s.contains("Operation not permitted") {
                return Err(ExtractError::PermissionDenied(
                    "Safari Bookmarks (Full Disk Access)".into(),
                ));
            }
            return Ok(skipped(&format!("Bookmarks.plist parse failed: {s}")));
        }
    };

    let mut items: Vec<TasteItem> = Vec::new();
    walk(&root, &Vec::new(), &mut items);

    let n = items.len();
    ctx.store.upsert_many(EntityKind::TasteItem, &items, |t| t.id.clone())?;
    ctx.store.flush_kind(EntityKind::TasteItem)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", n as u64);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: n as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn walk(node: &Value, folder_path: &Vec<String>, out: &mut Vec<TasteItem>) {
    let Value::Dictionary(dict) = node else { return };

    let kind = dict
        .get("WebBookmarkType")
        .and_then(|v| v.as_string())
        .unwrap_or("");
    let title = dict
        .get("Title")
        .and_then(|v| v.as_string())
        .unwrap_or("")
        .to_string();
    let identifier = dict
        .get("WebBookmarkIdentifier")
        .and_then(|v| v.as_string())
        .unwrap_or("")
        .to_string();

    match kind {
        "WebBookmarkTypeLeaf" => {
            let url = dict
                .get("URLString")
                .and_then(|v| v.as_string())
                .unwrap_or("")
                .to_string();
            // Some leaves have URI dict instead of plain URLString
            let url = if url.is_empty() {
                dict.get("URIDictionary")
                    .and_then(|v| v.as_dictionary())
                    .and_then(|d| d.get("title"))
                    .and_then(|v| v.as_string())
                    .unwrap_or("")
                    .to_string()
            } else {
                url
            };
            let display_title = if !title.is_empty() {
                title
            } else {
                dict.get("URIDictionary")
                    .and_then(|v| v.as_dictionary())
                    .and_then(|d| d.get("title"))
                    .and_then(|v| v.as_string())
                    .unwrap_or("")
                    .to_string()
            };
            if url.is_empty() && display_title.is_empty() {
                return;
            }

            let is_reading_list = folder_path.iter().any(|s| {
                s == "com.apple.ReadingList" || s.to_lowercase() == "reading list"
            });
            let kind_label = if is_reading_list { "reading_list" } else { "bookmark" };

            let domain = extract_domain(&url);
            out.push(TasteItem {
                id: stable_id(&[SOURCE, kind_label, &url, &display_title]),
                kind: kind_label.to_string(),
                name: display_title,
                creator: if domain.is_empty() { None } else { Some(domain) },
                play_count: 1,
                last_played: None,
                source: SOURCE.into(),
            });
        }
        _ => {
            // List, Proxy, or root — descend into children.
            let mut next_path = folder_path.clone();
            let folder_label = if !title.is_empty() {
                title
            } else if !identifier.is_empty() {
                identifier
            } else {
                String::new()
            };
            if !folder_label.is_empty() {
                next_path.push(folder_label);
            }
            if let Some(Value::Array(children)) = dict.get("Children") {
                for child in children {
                    walk(child, &next_path, out);
                }
            }
        }
    }
}

fn extract_domain(url: &str) -> String {
    let s = url
        .trim_start_matches("https://")
        .trim_start_matches("http://")
        .trim_start_matches("ftp://");
    let host = s.split('/').next().unwrap_or("");
    let host = host.split('?').next().unwrap_or(host);
    let host = host.split('#').next().unwrap_or(host);
    let host = host.trim_start_matches("www.");
    host.to_lowercase()
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
