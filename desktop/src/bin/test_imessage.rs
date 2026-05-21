//! Standalone iMessage ingester smoke test.
//!
//! Runs the same code path the Tauri command uses (snapshot chat.db, open
//! readonly, run the SELECT, decode attributedBody, filter empties) and
//! prints stats so we can iterate on the decoder without rebuilding the
//! whole Tauri app each time.
//!
//! Usage:
//!   cd desktop && cargo run --bin test_imessage
//!   cd desktop && cargo run --bin test_imessage -- --sample 5
//!     (show first 5 decoded message bodies)
//!   cd desktop && cargo run --bin test_imessage -- --limit 1000
//!     (cap the read to 1000 rows for faster iteration)

use pmc_desktop_lib::ingest::imessage::{
    default_chat_db_path, open_chat_db, read_messages,
};

fn main() {
    let mut sample = 0usize;
    let mut limit: Option<usize> = None;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sample" => {
                if i + 1 < args.len() {
                    sample = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--limit" => {
                if i + 1 < args.len() {
                    limit = args[i + 1].parse().ok();
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let path = default_chat_db_path().expect("HOME not set");
    println!("chat.db: {}", path.display());

    let started = std::time::Instant::now();
    let conn = open_chat_db(&path).unwrap_or_else(|e| {
        eprintln!("open_chat_db: {:?}", e);
        std::process::exit(1);
    });
    println!("opened in {:?}", started.elapsed());

    let started = std::time::Instant::now();
    let items = read_messages(&conn, limit).unwrap_or_else(|e| {
        eprintln!("read_messages: {:?}", e);
        std::process::exit(1);
    });
    println!("read {} items in {:?}", items.len(), started.elapsed());

    // Per-author breakdown so we can sanity-check this is "me" too
    let mut from_me = 0usize;
    let mut from_them = 0usize;
    for it in &items {
        if matches!(it.is_user, Some(true)) {
            from_me += 1;
        } else {
            from_them += 1;
        }
    }
    println!("  from me:   {from_me}");
    println!("  from them: {from_them}");

    // Length distribution
    let mut total_chars = 0usize;
    let mut max_len = 0usize;
    let mut over_100 = 0usize;
    for it in &items {
        let n = it.content.chars().count();
        total_chars += n;
        if n > max_len {
            max_len = n;
        }
        if n > 100 {
            over_100 += 1;
        }
    }
    let avg = if items.is_empty() { 0 } else { total_chars / items.len() };
    println!("  total chars: {total_chars}");
    println!("  avg length:  {avg}");
    println!("  max length:  {max_len}");
    println!("  >100 chars:  {over_100}");

    if sample > 0 {
        println!("\n--- first {sample} message bodies ---");
        for (i, it) in items.iter().take(sample).enumerate() {
            let preview: String = it.content.chars().take(160).collect();
            let role = if matches!(it.is_user, Some(true)) { "me" } else { "them" };
            println!("[{i}] ({role}) {preview}");
        }
    }
}
