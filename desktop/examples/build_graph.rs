//! Build the full personal knowledge graph on this Mac.
//!
//! Runs every extractor sequentially against the local data sources,
//! then synthesis, and prints a summary plus a small sample from each
//! resulting file. Mirrors what `graph_run_full` does inside the Tauri
//! app, minus the async scheduler.
//!
//! Usage:
//!   cd desktop && cargo run --example build_graph --release
//!   cd desktop && cargo run --example build_graph --release -- --user 11c7ace3
//!
//! Output lands in `~/.pmc-dev/storage/users/<user_id>/graph/*.jsonl`.

use std::path::PathBuf;
use std::sync::Arc;

use pmc_desktop_lib::extract::{
    self, ExtractCtx,
};
use pmc_desktop_lib::graph::{EntityKind, GraphStore, Watermarks};
use pmc_desktop_lib::synthesis;

fn main() {
    let mut user_id = String::from("11c7ace3-f395-4353-8acb-d6f7a2ec6113");
    let mut sample_n = 3usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--user" => { if i+1 < args.len() { user_id = args[i+1].clone(); i += 1; } }
            "--sample" => { if i+1 < args.len() { sample_n = args[i+1].parse().unwrap_or(3); i += 1; } }
            _ => {}
        }
        i += 1;
    }

    let home = std::env::var_os("HOME").map(PathBuf::from).expect("HOME unset");
    let root = home.join(".pmc-dev/storage/users").join(&user_id).join("graph");
    let wm_path = root.join("_watermarks.json");
    let store = Arc::new(GraphStore::new(&root).expect("graph store"));
    let watermarks = Watermarks::load(&wm_path);
    let ctx = ExtractCtx::new(store.clone(), watermarks);

    println!("graph root: {}", root.display());
    println!();

    let runs: Vec<(&str, fn(&ExtractCtx) -> Result<extract::ExtractSummary, extract::ExtractError>)> = vec![
        ("contacts",         extract::contacts::run),
        ("imessage_enrich",  extract::imessage_enrich::run),
        ("calendar",         extract::calendar::run),
        ("photos",           extract::photos::run),
        ("safari",           extract::safari::run),
        ("call_history",     extract::call_history::run),
        ("music",            extract::music::run),
        ("files",            extract::files::run),
        ("mail_enrich",      extract::mail_enrich::run),
        ("notes_enrich",     extract::notes_enrich::run),
        ("reminders",        extract::reminders::run),
        // Phase 2 wave 1 + 2
        ("chrome",           extract::chrome::run),
        ("screen_time",      extract::screen_time::run),
        ("shell",            extract::shell::run),
        ("locations",        extract::locations::run),
        ("editor_state",     extract::editor_state::run),
        ("notifications",    extract::notifications::run),
        ("voice_memos",      extract::voice_memos::run),
        // Phase 2 wave 3 — third-party comms
        ("slack",            extract::slack::run),
        // Phase 2 wave 4 — explicit curation
        ("bookmarks",        extract::bookmarks::run),
        // Phase 2 wave 5 — money / commerce
        ("wallet",           extract::wallet::run),
        // Phase 2 wave 6 — photo concept index (Apple's photo ML output)
        ("photo_concepts",   extract::photo_concepts::run),
    ];

    for (name, f) in runs {
        let started = std::time::Instant::now();
        match f(&ctx) {
            Ok(s) => {
                let tag = if s.skipped { "SKIP" } else { "OK  " };
                let extra = s.skip_reason.unwrap_or_default();
                println!("{tag} {:<18} items={:<8} wrote={:<6} {:>6} ms  {}",
                    name, s.items_processed, s.entities_written, started.elapsed().as_millis(), extra);
            }
            Err(e) => println!("ERR  {:<18} {}", name, e),
        }
    }

    println!("\n--- synthesis ---");
    match synthesis::run_all(&ctx) {
        Ok(summaries) => {
            for s in summaries {
                println!("OK   {:<24} wrote={:<6} {:>6} ms",
                    s.source, s.entities_written, s.duration_ms);
            }
        }
        Err(e) => println!("ERR  synthesis: {e}"),
    }

    println!("\n--- counts ---");
    for kind in [
        EntityKind::Person, EntityKind::Place, EntityKind::Event,
        EntityKind::Episode, EntityKind::Project, EntityKind::Theme,
        EntityKind::OpenLoop, EntityKind::TasteItem,
        EntityKind::FileSignal, EntityKind::CodeRepo,
        EntityKind::WebSignal, EntityKind::Edge,
    ] {
        println!("  {:<20} {}", kind.filename(), store.count(kind));
    }

    if sample_n > 0 {
        for kind in [
            EntityKind::Person, EntityKind::Theme, EntityKind::OpenLoop,
            EntityKind::Place, EntityKind::Project, EntityKind::Event,
            EntityKind::TasteItem, EntityKind::CodeRepo, EntityKind::WebSignal,
        ] {
            let values: Vec<serde_json::Value> = store.load(kind).unwrap_or_default();
            if values.is_empty() { continue; }
            println!("\n--- sample {} (first {}) ---", kind.filename(), sample_n.min(values.len()));
            for v in values.into_iter().take(sample_n) {
                println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
            }
        }
    }
}
