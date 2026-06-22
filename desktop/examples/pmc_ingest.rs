//! `pmc-ingest` — the engine that populates the personal knowledge graph
//! from the command line. Mac app parity for CLI users.
//!
//! Runs the SAME extractors the Mac app uses, against the SAME on-disk
//! graph store. Emits one JSON line per source as it finishes (with an
//! explicit stdout flush so the Python CLI can render a live checklist),
//! then synthesis at the end.
//!
//! Lives in `examples/` rather than `src/bin/` so Tauri's bundler
//! ignores it — keeps the Mac app build clean.
//!
//! Usage:
//!   cargo run --example pmc_ingest --release -- --user <id>
//!   cargo run --example pmc_ingest --release -- --user <id> --json
//!
//! JSON mode emits, for each source: a `{"phase":"start","source":...}`
//! line, then a `{"phase":"done","source":..., summary fields}` line.
//! Synthesis emits `{"phase":"synth_done", ...}`.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use pmc_desktop_lib::extract::{self, ExtractCtx, ExtractSummary};
use pmc_desktop_lib::graph::{GraphStore, Watermarks};
use pmc_desktop_lib::synthesis;

fn main() {
    let args = parse_args();

    let home = std::env::var_os("HOME").map(PathBuf::from).expect("HOME unset");
    let root: PathBuf = args
        .root
        .unwrap_or_else(|| home.join(".pmc-dev/storage"));
    let graph_root = root.join("users").join(&args.user).join("graph");
    std::fs::create_dir_all(&graph_root).expect("create graph dir");
    let wm_path = graph_root.join("_watermarks.json");

    let store = Arc::new(GraphStore::new(&graph_root).expect("open graph store"));
    let watermarks = Watermarks::load(&wm_path);
    let ctx = ExtractCtx::new(store.clone(), watermarks);

    if !args.json {
        eprintln!("pmc-ingest");
        eprintln!("  user:  {}", args.user);
        eprintln!("  root:  {}", graph_root.display());
        eprintln!();
    }

    // The full extractor list — kept in sync with the Source enum in
    // `desktop/src/schedule/mod.rs::Source::all()`. Same set, same order.
    let runs: Vec<(&str, fn(&ExtractCtx) -> Result<ExtractSummary, extract::ExtractError>)> = vec![
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
        ("chrome",           extract::chrome::run),
        ("screen_time",      extract::screen_time::run),
        ("shell",            extract::shell::run),
        ("locations",        extract::locations::run),
        ("editor_state",     extract::editor_state::run),
        ("notifications",    extract::notifications::run),
        ("voice_memos",      extract::voice_memos::run),
        ("slack",            extract::slack::run),
        ("bookmarks",        extract::bookmarks::run),
        ("wallet",           extract::wallet::run),
        ("photo_concepts",   extract::photo_concepts::run),
        ("icloud_drive",     extract::icloud_drive::run),
    ];

    let mut all_summaries: Vec<ExtractSummary> = Vec::new();

    for (name, run_fn) in runs {
        emit_start(args.json, name);
        match run_fn(&ctx) {
            Ok(summary) => {
                emit_done(args.json, &summary);
                all_summaries.push(summary);
            }
            Err(e) => {
                emit_error(args.json, name, &e.to_string());
                all_summaries.push(ExtractSummary {
                    source: name.into(),
                    skipped: true,
                    skip_reason: Some(e.to_string()),
                    ..Default::default()
                });
            }
        }
    }

    // Synthesis after all extractors.
    emit_start(args.json, "synthesis");
    match synthesis::run_all(&ctx) {
        Ok(sums) => {
            for s in &sums {
                emit_done(args.json, s);
            }
            all_summaries.extend(sums);
        }
        Err(e) => emit_error(args.json, "synthesis", &e.to_string()),
    }

    if !args.json {
        print_summary(&all_summaries);
    }
}

// ---------------------------------------------------------------------------
// JSON-line + human emission
// ---------------------------------------------------------------------------

fn emit_start(json: bool, source: &str) {
    if json {
        println!("{{\"phase\":\"start\",\"source\":\"{}\"}}", source);
        let _ = std::io::stdout().flush();
    } else {
        eprint!("  · {} … ", source);
        let _ = std::io::stderr().flush();
    }
}

fn emit_done(json: bool, s: &ExtractSummary) {
    if json {
        // Pretty-print as a single line so the consumer can read by lines.
        let line = serde_json::to_string(&serde_json::json!({
            "phase": "done",
            "source": s.source,
            "items_processed": s.items_processed,
            "entities_written": s.entities_written,
            "duration_ms": s.duration_ms,
            "skipped": s.skipped,
            "skip_reason": s.skip_reason,
        }))
        .unwrap_or_default();
        println!("{line}");
        let _ = std::io::stdout().flush();
    } else if s.skipped {
        let reason = s.skip_reason.clone().unwrap_or_default();
        eprintln!("skipped ({reason})");
    } else {
        eprintln!(
            "{} entities · {} ms",
            s.entities_written,
            s.duration_ms
        );
    }
}

fn emit_error(json: bool, source: &str, error: &str) {
    if json {
        let line = serde_json::to_string(&serde_json::json!({
            "phase": "error",
            "source": source,
            "error": error,
        }))
        .unwrap_or_default();
        println!("{line}");
        let _ = std::io::stdout().flush();
    } else {
        eprintln!("error: {error}");
    }
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

struct Args {
    user: String,
    root: Option<PathBuf>,
    json: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut user = String::new();
    let mut root: Option<PathBuf> = None;
    let mut json = false;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--user" | "-u" => {
                if i + 1 < argv.len() {
                    user = argv[i + 1].clone();
                    i += 1;
                }
            }
            "--root" | "-r" => {
                if i + 1 < argv.len() {
                    root = Some(PathBuf::from(&argv[i + 1]));
                    i += 1;
                }
            }
            "--json" => json = true,
            "-h" | "--help" => {
                eprintln!("{}", help_text());
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {other}");
                eprintln!("{}", help_text());
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if user.is_empty() {
        eprintln!("error: --user is required");
        eprintln!();
        eprintln!("{}", help_text());
        std::process::exit(2);
    }
    Args { user, root, json }
}

fn help_text() -> &'static str {
    "usage: pmc-ingest --user <id> [--root <path>] [--json]\n\
     \n\
       --user <id>     user_id under storage_root (required)\n\
       --root <path>   override storage root\n\
                       (default: $HOME/.pmc-dev/storage)\n\
       --json          emit one JSON line per extractor start/finish\n\
                       (machine-readable, no human summary)\n\
     "
}

// ---------------------------------------------------------------------------
// Human summary (non-json mode)
// ---------------------------------------------------------------------------

fn print_summary(summaries: &[ExtractSummary]) {
    let mut ran = 0;
    let mut skipped = 0;
    let mut total_entities = 0u64;
    for s in summaries {
        if s.skipped {
            skipped += 1;
        } else {
            ran += 1;
            total_entities += s.entities_written;
        }
    }
    eprintln!();
    eprintln!(
        "done — {ran} extractors ran, {skipped} skipped, {total_entities} entities written"
    );
}
