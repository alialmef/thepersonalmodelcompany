//! Local files + git repos extractor.
//!
//! Walks the user's `~/Documents`, `~/Desktop`, `~/Projects`, and any
//! recognizable code-host directories. We *never read file contents* —
//! the graph captures filename, path, mtime, size, and a coarse `kind`
//! inferred from extension.
//!
//! For directories that look like git repos (have a `.git/`), we read
//! the most recent commit subject and commit count over the last 30
//! days via plain git CLI. This surfaces "active projects" without
//! parsing pack files.

use super::{ExtractCtx, ExtractError, ExtractSummary};
use crate::graph::schema::{CodeRepo, EntityKind, FileSignal};
use crate::graph::store::stable_id;
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = "files";

const ROOTS: &[&str] = &["Documents", "Desktop", "Projects"];

// Filename/path patterns to skip outright.
const SKIP_DIRS: &[&str] = &[
    "node_modules", ".venv", "venv", "target", ".next", "dist", "build",
    "__pycache__", ".cargo", ".npm", ".cache", "Library", ".git",
    "Pods", ".gradle", ".idea", ".vscode-insiders",
];

const FILE_LIMIT: usize = 5000; // bound the walk so we don't melt the disk
const REPO_LIMIT: usize = 200;

pub fn run(ctx: &ExtractCtx) -> Result<ExtractSummary, ExtractError> {
    let started = std::time::Instant::now();
    let home = std::env::var_os("HOME").map(PathBuf::from)
        .ok_or_else(|| ExtractError::Other("HOME unset".into()))?;

    let mut files: Vec<FileSignal> = Vec::new();
    let mut repos: Vec<CodeRepo> = Vec::new();
    let mut walked = 0u64;

    for root in ROOTS {
        let dir = home.join(root);
        if !dir.is_dir() { continue; }
        walk(&dir, &mut files, &mut repos, &mut walked);
        if files.len() >= FILE_LIMIT { break; }
    }

    let n_f = files.len();
    let n_r = repos.len();
    ctx.store.upsert_many(EntityKind::FileSignal, &files, |f| f.id.clone())?;
    ctx.store.upsert_many(EntityKind::CodeRepo,   &repos, |r| r.id.clone())?;
    ctx.store.flush_kind(EntityKind::FileSignal)?;
    ctx.store.flush_kind(EntityKind::CodeRepo)?;

    if let Ok(mut w) = ctx.watermarks.lock() {
        w.set(SOURCE, "full", walked);
    }
    ctx.save_watermarks();

    Ok(ExtractSummary {
        source: SOURCE.into(),
        items_processed: walked,
        entities_written: (n_f + n_r) as u64,
        duration_ms: started.elapsed().as_millis() as u64,
        skipped: false,
        skip_reason: None,
    })
}

fn walk(dir: &Path, files: &mut Vec<FileSignal>, repos: &mut Vec<CodeRepo>, walked: &mut u64) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if repos.len() >= REPO_LIMIT && files.len() >= FILE_LIMIT { return; }

        // Detect a repo at this directory.
        if d.join(".git").is_dir() && repos.len() < REPO_LIMIT {
            if let Some(repo) = inspect_repo(&d) {
                repos.push(repo);
            }
            // Don't recurse into a repo's content — too noisy.
            continue;
        }

        let Ok(read) = std::fs::read_dir(&d) else { continue };
        for entry in read.flatten() {
            let p = entry.path();
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
            if name.starts_with('.') { continue; }
            let meta = match entry.metadata() { Ok(m) => m, Err(_) => continue };
            if meta.is_dir() {
                if SKIP_DIRS.iter().any(|s| s.eq_ignore_ascii_case(name)) { continue; }
                stack.push(p);
                continue;
            }
            if !meta.is_file() { continue; }
            *walked += 1;
            if files.len() >= FILE_LIMIT { continue; }

            let modified: Option<DateTime<Utc>> = meta.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| DateTime::<Utc>::from_timestamp(d.as_secs() as i64, d.subsec_nanos()))
                .flatten();
            let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase());
            let kind = classify_extension(ext.as_deref());

            files.push(FileSignal {
                id: stable_id(&["file", &p.to_string_lossy()]),
                path: p.to_string_lossy().to_string(),
                name: name.to_string(),
                extension: ext,
                modified,
                size_bytes: meta.len(),
                kind: Some(kind),
            });
        }
    }
}

fn classify_extension(ext: Option<&str>) -> String {
    let Some(e) = ext else { return "other".into() };
    match e {
        "md" | "txt" | "rtf" | "pages" | "doc" | "docx" | "pdf" => "document".into(),
        "rs" | "ts" | "tsx" | "js" | "jsx" | "py" | "go" | "rb" | "swift" | "kt" |
        "java" | "c" | "h" | "cpp" | "cs" | "php" | "scala" => "code".into(),
        "jpg" | "jpeg" | "png" | "heic" | "gif" | "webp" | "tiff" => "image".into(),
        "mov" | "mp4" | "m4v" | "avi" | "webm" => "video".into(),
        "mp3" | "m4a" | "wav" | "flac" => "audio".into(),
        "sketch" | "fig" | "psd" | "ai" => "design".into(),
        "zip" | "tar" | "gz" | "tgz" | "7z" | "rar" => "archive".into(),
        "csv" | "tsv" | "xlsx" | "numbers" => "spreadsheet".into(),
        _ => "other".into(),
    }
}

fn inspect_repo(dir: &Path) -> Option<CodeRepo> {
    let name = dir.file_name()?.to_string_lossy().to_string();
    let path = dir.to_string_lossy().to_string();

    let last_commit_ts = run_git(dir, &["log", "-1", "--format=%ct"])
        .and_then(|s| s.trim().parse::<i64>().ok())
        .and_then(|t| DateTime::<Utc>::from_timestamp(t, 0));

    let count_30d = run_git(dir, &["log", "--since=30.days.ago", "--oneline"])
        .map(|s| s.lines().count() as u64)
        .unwrap_or(0);

    let branches = run_git(dir, &["branch", "--format=%(refname:short)"])
        .map(|s| s.lines().take(20).map(|l| l.trim().to_string()).collect::<Vec<_>>())
        .unwrap_or_default();

    let language = infer_repo_language(dir);

    Some(CodeRepo {
        id: stable_id(&["repo", &path]),
        path,
        name,
        language,
        last_commit: last_commit_ts,
        commit_count_30d: count_30d,
        branches,
    })
}

fn run_git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    if !out.status.success() { return None; }
    String::from_utf8(out.stdout).ok()
}

fn infer_repo_language(dir: &Path) -> Option<String> {
    let candidates: &[(&str, &str)] = &[
        ("Cargo.toml", "rust"),
        ("package.json", "javascript"),
        ("pyproject.toml", "python"),
        ("requirements.txt", "python"),
        ("go.mod", "go"),
        ("Gemfile", "ruby"),
        ("pom.xml", "java"),
        ("build.gradle", "kotlin"),
        ("Package.swift", "swift"),
        ("composer.json", "php"),
    ];
    for (name, lang) in candidates {
        if dir.join(name).is_file() { return Some(lang.to_string()); }
    }
    None
}
