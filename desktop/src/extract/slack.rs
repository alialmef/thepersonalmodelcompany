//! Slack workspace extractor (metadata only).
//!
//! Reads `~/Library/Application Support/Slack/storage/root-state.json`
//! to enumerate workspaces the user is signed in to. We don't touch
//! message content — Slack's message cache is in a LevelDB that's
//! locked while Slack runs and isn't worth fighting for V1.
//!
//! Workspace metadata alone is high-signal:
//!   - Which workspaces you're in (work life topology)
//!   - Your userId per workspace
//!   - Unread state (a coarse "what's pending" signal)
//!   - Workspace ordering (your preference)
//!
//! Each workspace becomes a `Project` entity (Slack workspaces ARE the
//! container for work-life projects, conceptually). The synthesis
//! layer can later promote them to Themes or merge them with code
//! repos / calendar themes when context warrants.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, Project};
use crate::graph::store::stable_id;
use chrono::Utc;
use serde_json::Value;
use std::path::PathBuf;

const SOURCE: &str = "slack";

fn default_path() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| {
        let mut p = PathBuf::from(h);
        p.push("Library/Application Support/Slack/storage/root-state.json");
        p
    })
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(path) = default_path() else {
        return Ok(skipped("HOME unset"));
    };
    if !path.is_file() {
        return Ok(skipped("Slack not installed (root-state.json absent)"));
    }

    let text = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            return Err(ExtractError::PermissionDenied(
                "Slack root-state.json (Full Disk Access)".into(),
            ));
        }
        Err(_) => return Ok(skipped("couldn't read Slack root-state.json")),
    };
    let json: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return Ok(skipped("Slack root-state.json was not valid JSON")),
    };

    // Workspaces live at the top-level `workspaces` object. Each value
    // has at least: domain, id, name, url, order. We don't require any
    // single field — Slack's shape rotates.
    let Some(workspaces) = json.get("workspaces").and_then(|v| v.as_object()) else {
        return Ok(skipped("no workspaces block in root-state.json"));
    };

    // The webapp.teams block holds per-team user metadata: your userId
    // in that workspace, unreads, locale. Same id keys as workspaces.
    let team_meta = json
        .pointer("/webapp/teams")
        .and_then(|v| v.as_object());

    let now = Utc::now();
    let mut projects: Vec<Project> = Vec::with_capacity(workspaces.len());

    for (team_id, info) in workspaces {
        let name = info
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(team_id)
            .to_string();
        let domain = info
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let order = info.get("order").and_then(|v| v.as_u64()).unwrap_or(0);

        // Pull the user's id in this workspace + unreads if present.
        let (user_id, unreads, unread_highlights) = team_meta
            .and_then(|tm| tm.get(team_id))
            .map(|t| {
                let uid = t
                    .get("userId")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let unread = t
                    .pointer("/unreads/unreads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let highlights = t
                    .pointer("/unreads/unreadHighlights")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                (uid, unread, highlights)
            })
            .unwrap_or((None, 0, 0));

        // Project entity for the workspace. Summary carries the user's
        // workspace-specific userId + domain + unread state in a
        // human-readable form so the synthesis layer can lift them
        // back out by inspection without us needing to add a custom
        // schema for "workspace".
        let _ = order;
        let summary_parts: Vec<String> = {
            let mut v: Vec<String> = Vec::new();
            if !domain.is_empty() {
                v.push(format!("domain={domain}"));
            }
            if let Some(uid) = user_id.as_deref() {
                v.push(format!("user_id={uid}"));
            }
            if unreads > 0 {
                v.push(format!("unreads={unreads}"));
            }
            if unread_highlights > 0 {
                v.push(format!("highlights={unread_highlights}"));
            }
            v
        };

        projects.push(Project {
            id: stable_id(&["slack_workspace", team_id]),
            name: format!("Slack · {}", name),
            // "active" if there are unreads, otherwise we don't claim
            // — Slack root-state doesn't reveal whether the workspace
            // is still being used in any deeper sense than this.
            state: Some(if unreads > 0 || unread_highlights > 0 {
                "active".to_string()
            } else {
                "unknown".to_string()
            }),
            people_ids: Vec::new(),
            last_activity: None,
            summary: if summary_parts.is_empty() {
                None
            } else {
                Some(summary_parts.join(" · "))
            },
            sources: vec![format!("{SOURCE}:{team_id}")],
        });
    }

    let n = projects.len();
    ctx.store
        .upsert_many(EntityKind::Project, &projects, |p| p.id.clone())?;
    ctx.store.flush_kind(EntityKind::Project)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", n as u64);
    }
    ctx.save_watermarks();

    // Silence "now" lint when the workspace block is empty.
    let _ = now;

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: n as u64,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
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
