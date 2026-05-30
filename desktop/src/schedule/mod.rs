//! Background scheduler.
//!
//! One Tokio task per source, each with its own cadence and watermark.
//! After any extractor finishes with non-zero entities written, a
//! debounced synthesis task fires (10-minute window) so cross-source
//! resolution doesn't thrash.
//!
//! The scheduler is started from `lib.rs::setup()` when the Tauri app
//! launches. It runs as long as the app is alive — quitting the app
//! pauses extraction.

pub mod config;

use crate::extract::{self, ExtractCtx, ExtractSummary};
use crate::graph::{GraphStore, Watermarks};
use crate::synthesis;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio::sync::Mutex;

/// Source identifier — keep names short and stable; they show up in
/// watermarks.json and audit events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Contacts,
    ImessageEnrich,
    Calendar,
    Photos,
    Safari,
    CallHistory,
    Music,
    Files,
    MailEnrich,
    NotesEnrich,
    Reminders,
    // Phase 2 additions
    Chrome,
    ScreenTime,
    Shell,
    Locations,
}

impl Source {
    pub fn cadence(self) -> StdDuration {
        use Source::*;
        match self {
            // Live signal we want fresh:
            ImessageEnrich  => StdDuration::from_secs(2 * 60),
            NotesEnrich     => StdDuration::from_secs(5 * 60),
            Reminders       => StdDuration::from_secs(5 * 60),
            // Frequent:
            Calendar        => StdDuration::from_secs(15 * 60),
            MailEnrich      => StdDuration::from_secs(15 * 60),
            ScreenTime      => StdDuration::from_secs(30 * 60),
            // Periodic:
            Contacts        => StdDuration::from_secs(60 * 60),
            Photos          => StdDuration::from_secs(60 * 60),
            Safari          => StdDuration::from_secs(60 * 60),
            Chrome          => StdDuration::from_secs(60 * 60),
            CallHistory     => StdDuration::from_secs(60 * 60),
            Files           => StdDuration::from_secs(60 * 60),
            Shell           => StdDuration::from_secs(60 * 60),
            Locations       => StdDuration::from_secs(2 * 60 * 60),
            // Slow:
            Music           => StdDuration::from_secs(24 * 60 * 60),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Source::Contacts        => "contacts",
            Source::ImessageEnrich  => "imessage_enrich",
            Source::Calendar        => "calendar",
            Source::Photos          => "photos",
            Source::Safari          => "safari",
            Source::CallHistory     => "call_history",
            Source::Music           => "music",
            Source::Files           => "files",
            Source::MailEnrich      => "mail_enrich",
            Source::NotesEnrich     => "notes_enrich",
            Source::Reminders       => "reminders",
            Source::Chrome          => "chrome",
            Source::ScreenTime      => "screen_time",
            Source::Shell           => "shell",
            Source::Locations       => "locations",
        }
    }

    pub fn all() -> &'static [Source] {
        use Source::*;
        &[Contacts, ImessageEnrich, Calendar, Photos, Safari, CallHistory,
          Music, Files, MailEnrich, NotesEnrich, Reminders,
          Chrome, ScreenTime, Shell, Locations]
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SchedulerEvent {
    pub source: String,
    pub kind: String,           // "started" | "finished" | "skipped" | "error"
    pub at: chrono::DateTime<chrono::Utc>,
    pub summary: Option<ExtractSummary>,
    pub error: Option<String>,
}

pub struct Scheduler {
    ctx: Arc<ExtractCtx>,
    events: UnboundedSender<SchedulerEvent>,
}

impl Scheduler {
    pub fn new(store: Arc<GraphStore>, watermarks: Watermarks) -> (Self, mpsc::UnboundedReceiver<SchedulerEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        let ctx = Arc::new(ExtractCtx::new(store, watermarks));
        (Self { ctx, events: tx }, rx)
    }

    pub fn ctx(&self) -> Arc<ExtractCtx> { self.ctx.clone() }

    /// Kick off all per-source loops. Spawns Tokio tasks; the scheduler
    /// keeps running until the process exits.
    pub fn start(&self) {
        let synth_signal: Arc<Mutex<Option<std::time::Instant>>> = Arc::new(Mutex::new(None));
        for &source in Source::all() {
            let ctx = self.ctx.clone();
            let events = self.events.clone();
            let signal = synth_signal.clone();
            tokio::spawn(async move {
                loop {
                    let started_at = chrono::Utc::now();
                    let _ = events.send(SchedulerEvent {
                        source: source.name().into(), kind: "started".into(),
                        at: started_at, summary: None, error: None,
                    });
                    let res = tokio::task::spawn_blocking({
                        let ctx = ctx.clone();
                        move || dispatch(source, &ctx)
                    }).await;

                    match res {
                        Ok(Ok(summary)) => {
                            let wrote_anything = summary.entities_written > 0;
                            let _ = events.send(SchedulerEvent {
                                source: source.name().into(),
                                kind: if summary.skipped { "skipped".into() } else { "finished".into() },
                                at: chrono::Utc::now(),
                                summary: Some(summary),
                                error: None,
                            });
                            if wrote_anything {
                                let mut g = signal.lock().await;
                                *g = Some(std::time::Instant::now());
                            }
                        }
                        Ok(Err(e)) => {
                            let _ = events.send(SchedulerEvent {
                                source: source.name().into(),
                                kind: "error".into(),
                                at: chrono::Utc::now(),
                                summary: None,
                                error: Some(e.to_string()),
                            });
                        }
                        Err(e) => {
                            let _ = events.send(SchedulerEvent {
                                source: source.name().into(),
                                kind: "error".into(),
                                at: chrono::Utc::now(),
                                summary: None,
                                error: Some(format!("task join: {e}")),
                            });
                        }
                    }

                    tokio::time::sleep(source.cadence()).await;
                }
            });
        }

        // Synthesis worker — runs ~10 minutes after the last extractor
        // reported new entities. Coalesces bursts.
        let ctx = self.ctx.clone();
        let events = self.events.clone();
        let signal = synth_signal.clone();
        tokio::spawn(async move {
            const DEBOUNCE: StdDuration = StdDuration::from_secs(10 * 60);
            loop {
                tokio::time::sleep(StdDuration::from_secs(60)).await;
                let should_run = {
                    let mut g = signal.lock().await;
                    if let Some(t) = *g {
                        if t.elapsed() >= DEBOUNCE { *g = None; true } else { false }
                    } else { false }
                };
                if !should_run { continue; }
                let ctx2 = ctx.clone();
                let r = tokio::task::spawn_blocking(move || synthesis::run_all(&ctx2)).await;
                match r {
                    Ok(Ok(summaries)) => {
                        for s in summaries {
                            let _ = events.send(SchedulerEvent {
                                source: s.source.clone(), kind: "finished".into(),
                                at: chrono::Utc::now(), summary: Some(s), error: None,
                            });
                        }
                    }
                    Ok(Err(e)) => {
                        let _ = events.send(SchedulerEvent {
                            source: "synthesis".into(), kind: "error".into(),
                            at: chrono::Utc::now(), summary: None,
                            error: Some(e.to_string()),
                        });
                    }
                    Err(e) => {
                        let _ = events.send(SchedulerEvent {
                            source: "synthesis".into(), kind: "error".into(),
                            at: chrono::Utc::now(), summary: None,
                            error: Some(format!("task join: {e}")),
                        });
                    }
                }
            }
        });
    }

    /// One-shot full run — every extractor sequentially, then synthesis.
    /// Useful at first-launch to populate the graph from cold.
    pub async fn run_full(&self) -> Vec<ExtractSummary> {
        let mut out = Vec::new();
        for &source in Source::all() {
            let ctx = self.ctx.clone();
            let r = tokio::task::spawn_blocking(move || dispatch(source, &ctx)).await;
            match r {
                Ok(Ok(s)) => out.push(s),
                Ok(Err(e)) => out.push(ExtractSummary {
                    source: source.name().into(),
                    skipped: true,
                    skip_reason: Some(e.to_string()),
                    ..Default::default()
                }),
                Err(e) => out.push(ExtractSummary {
                    source: source.name().into(),
                    skipped: true,
                    skip_reason: Some(format!("task join: {e}")),
                    ..Default::default()
                }),
            }
        }
        let ctx = self.ctx.clone();
        if let Ok(Ok(sums)) = tokio::task::spawn_blocking(move || synthesis::run_all(&ctx)).await {
            out.extend(sums);
        }
        out
    }
}

fn dispatch(source: Source, ctx: &ExtractCtx) -> Result<ExtractSummary, extract::ExtractError> {
    use extract::*;
    match source {
        Source::Contacts        => contacts::run(ctx),
        Source::ImessageEnrich  => imessage_enrich::run(ctx),
        Source::Calendar        => calendar::run(ctx),
        Source::Photos          => photos::run(ctx),
        Source::Safari          => safari::run(ctx),
        Source::CallHistory     => call_history::run(ctx),
        Source::Music           => music::run(ctx),
        Source::Files           => files::run(ctx),
        Source::MailEnrich      => mail_enrich::run(ctx),
        Source::NotesEnrich     => notes_enrich::run(ctx),
        Source::Reminders       => reminders::run(ctx),
        Source::Chrome          => chrome::run(ctx),
        Source::ScreenTime      => screen_time::run(ctx),
        Source::Shell           => shell::run(ctx),
        Source::Locations       => locations::run(ctx),
    }
}
