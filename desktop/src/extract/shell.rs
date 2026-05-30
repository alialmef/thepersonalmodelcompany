//! Shell history extractor.
//!
//! Reads `~/.zsh_history` and `~/.bash_history` — terminal command
//! history. We aggregate by *root command only* (the binary name before
//! arguments) and never persist the full command line. So `git`,
//! `npm`, `claude`, `cd` get counted; the full string with paths,
//! tokens, secrets, etc. is dropped at parse time.
//!
//! This is high-signal for technical users: the tooling someone reaches
//! for daily — and how that mix shifts — says more about what they're
//! actually working on than any project file does.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{EntityKind, ShellCommand};
use crate::graph::store::stable_id;
use chrono::{DateTime, Duration, TimeZone, Utc};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

const SOURCE: &str = "shell";

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let Some(h) = home() else {
        return Ok(skipped("HOME unset"));
    };

    let candidates = [
        h.join(".zsh_history"),
        h.join(".bash_history"),
        // Recent zsh defaults sometimes use the XDG path
        h.join(".local/share/zsh/history"),
    ];

    let mut all: Vec<(Option<DateTime<Utc>>, String)> = Vec::new();
    let mut any_present = false;
    for p in &candidates {
        if !p.is_file() {
            continue;
        }
        any_present = true;
        let bytes = match fs::read(p) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return Err(ExtractError::PermissionDenied(format!(
                    "shell history at {}",
                    p.display()
                )));
            }
            Err(_) => continue,
        };
        // zsh history is mostly UTF-8 but can contain extended chars
        // for metafy-encoded entries; lossy decode is fine for parsing
        // root commands.
        let text = String::from_utf8_lossy(&bytes);
        parse_into(&text, &mut all);
    }

    if !any_present {
        return Ok(skipped("no shell history files present"));
    }

    let now = Utc::now();
    let cutoff_30 = now - Duration::days(30);
    let cutoff_180 = now - Duration::days(180);

    struct Agg {
        count_30d: u64,
        count_180d: u64,
        last: Option<DateTime<Utc>>,
    }
    let mut by_root: HashMap<String, Agg> = HashMap::new();
    let mut total = 0u64;
    for (ts, root) in all {
        // If we have no timestamp (bash w/o HISTTIMEFORMAT), still
        // count it but only in the 180d bucket so undated entries
        // don't inflate the "recent" signal.
        total += 1;
        let recent = matches!(ts, Some(t) if t >= cutoff_30);
        let in_180 = ts.map(|t| t >= cutoff_180).unwrap_or(true);
        if !in_180 {
            continue;
        }
        let a = by_root.entry(root).or_insert(Agg {
            count_30d: 0,
            count_180d: 0,
            last: None,
        });
        a.count_180d += 1;
        if recent {
            a.count_30d += 1;
        }
        if let Some(t) = ts {
            if a.last.map(|l| t > l).unwrap_or(true) {
                a.last = Some(t);
            }
        }
    }

    let mut commands: Vec<ShellCommand> = Vec::with_capacity(by_root.len());
    for (root, agg) in by_root {
        if agg.count_180d < 2 {
            continue; // drop noise
        }
        commands.push(ShellCommand {
            id: stable_id(&["shell_command", &root]),
            command_root: root.clone(),
            count_30d: agg.count_30d,
            count_180d: agg.count_180d,
            last_used: agg.last,
            category: Some(categorize(&root)),
        });
    }

    let n = commands.len();
    ctx.store
        .upsert_many(EntityKind::ShellCommand, &commands, |c| c.id.clone())?;
    ctx.store.flush_kind(EntityKind::ShellCommand)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", total);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: total,
        entities_written: n as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn parse_into(text: &str, out: &mut Vec<(Option<DateTime<Utc>>, String)>) {
    // zsh extended history lines look like:   ": 1717181718:0;git status"
    // bash with HISTTIMEFORMAT lines look like alternating "#1717181718\ngit status"
    // bash plain lines are just the command on its own line.
    let mut pending_ts: Option<DateTime<Utc>> = None;
    for line in text.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }

        // zsh extended format
        if line.starts_with(": ") {
            if let Some(rest) = line.strip_prefix(": ") {
                if let Some((meta, cmd)) = rest.split_once(';') {
                    let ts = meta
                        .split(':')
                        .next()
                        .and_then(|s| s.trim().parse::<i64>().ok())
                        .and_then(|n| Utc.timestamp_opt(n, 0).single());
                    if let Some(root) = root_command(cmd) {
                        out.push((ts, root));
                    }
                    continue;
                }
            }
        }

        // bash HISTTIMEFORMAT marker
        if let Some(rest) = line.strip_prefix('#') {
            if let Ok(n) = rest.trim().parse::<i64>() {
                pending_ts = Utc.timestamp_opt(n, 0).single();
                continue;
            }
        }

        // plain command
        if let Some(root) = root_command(line) {
            out.push((pending_ts.take(), root));
        } else {
            pending_ts = None;
        }
    }
}

fn root_command(cmd: &str) -> Option<String> {
    // Strip shell metas that prefix some history entries.
    let trimmed = cmd.trim_start_matches(['\\', '!', '%']).trim();
    if trimmed.is_empty() {
        return None;
    }
    // sudo X / env A=B X / time X — peel off the wrapper to get the
    // actual command of interest.
    let first = trimmed.split_whitespace().next()?;
    let first = first.trim_matches(|c: char| matches!(c, '"' | '\''));
    if first.is_empty() {
        return None;
    }
    let lowered = first.to_lowercase();
    if matches!(lowered.as_str(), "sudo" | "time" | "env" | "exec" | "nice") {
        // Recurse one level — pretty common in shells
        let rest = trimmed
            .splitn(2, char::is_whitespace)
            .nth(1)
            .unwrap_or("");
        return root_command(rest);
    }
    // Strip path prefixes — /usr/local/bin/git → git
    let basename = lowered.rsplit('/').next().unwrap_or(&lowered).to_string();
    // Drop variable assignments like FOO=bar
    if basename.contains('=') {
        return None;
    }
    Some(basename)
}

fn categorize(root: &str) -> String {
    match root {
        "git" | "gh" | "hg" | "svn" => "vcs",
        "npm" | "yarn" | "pnpm" | "bun" | "pip" | "uv" | "poetry" | "cargo" | "brew"
        | "apt" | "apt-get" | "dnf" | "yum" | "gem" | "go" | "mvn" | "gradle" => "package",
        "cd" | "ls" | "pwd" | "mkdir" | "rm" | "mv" | "cp" | "ln" | "find" | "grep"
        | "rg" | "fd" | "cat" | "less" | "more" | "head" | "tail" | "tree" | "du"
        | "df" | "echo" | "which" | "type" | "history" | "exit" | "clear" | "open" => "shell",
        "vim" | "nvim" | "emacs" | "code" | "subl" | "nano" | "cursor" | "claude" => "editor",
        "curl" | "wget" | "ssh" | "scp" | "rsync" | "ping" | "dig" | "nslookup"
        | "netstat" | "lsof" | "nc" | "telnet" => "network",
        "make" | "bazel" | "ninja" | "docker" | "kubectl" | "helm" | "terraform"
        | "ansible" | "fly" | "railway" | "vercel" => "build",
        _ => "other",
    }
    .to_string()
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
