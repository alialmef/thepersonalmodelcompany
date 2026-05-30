//! Typed entities for the personal knowledge graph.
//!
//! Each entity carries:
//!   * a stable `id` (deterministic from source-id when possible, UUID
//!     when not) so re-extraction is idempotent
//!   * `source` provenance (where this fact came from — every claim is
//!     traceable to a raw input)
//!   * `last_seen` (when this entity was last observed) — drives the
//!     "live vs dormant" computation
//!   * domain-specific structured fields
//!
//! Cross-source identity is resolved separately via `Edge` links — see
//! `synthesis::entity_resolve`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// People
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Person {
    pub id: String,
    /// Preferred display name (e.g. "Sarah Almeflehi"). May be None if all
    /// we know is a raw handle.
    pub display_name: Option<String>,
    /// Names this person has appeared under across sources.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Phone numbers, normalized to E.164 when possible.
    #[serde(default)]
    pub phones: Vec<String>,
    /// Email addresses lowercased.
    #[serde(default)]
    pub emails: Vec<String>,
    /// Relationship label the user has assigned in Contacts ("sister",
    /// "mother", "partner") if any.
    pub relationship: Option<String>,
    /// Inferred role in the user's life — derived from message volume,
    /// recency, and channel. One of: family, close-friend, friend,
    /// colleague, acquaintance, professional, unknown.
    pub inferred_role: Option<String>,
    /// Relationship "temperature" — how active the relationship is right
    /// now. One of: hot (last 7 days), warm (last 30 days), cool
    /// (last 90 days), dormant (older), unknown.
    pub temperature: Option<String>,
    /// Per-channel activity counters. Keys: "imessage", "email",
    /// "facetime", "calendar", "photos".
    #[serde(default)]
    pub channel_counts: HashMap<String, u64>,
    /// First and last observation across all sources.
    pub first_seen: Option<DateTime<Utc>>,
    pub last_seen: Option<DateTime<Utc>>,
    /// Free-form organization affiliations from Contacts or email
    /// domains.
    #[serde(default)]
    pub organizations: Vec<String>,
    /// Birthday from Contacts (MM-DD form so year is optional).
    pub birthday: Option<String>,
    /// Provenance — which raw sources contributed to this person.
    #[serde(default)]
    pub sources: Vec<String>,
}

// ---------------------------------------------------------------------------
// Places
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Place {
    pub id: String,
    pub label: String,
    /// Best-guess coordinates (decimal degrees).
    pub lat: Option<f64>,
    pub lon: Option<f64>,
    /// Place type — one of: home, work, frequent, trip, venue, unknown.
    pub kind: Option<String>,
    /// Number of distinct visits inferred across sources.
    pub visit_count: u64,
    pub first_seen: Option<DateTime<Utc>>,
    pub last_seen: Option<DateTime<Utc>>,
    #[serde(default)]
    pub sources: Vec<String>,
}

// ---------------------------------------------------------------------------
// Events / episodes
// ---------------------------------------------------------------------------

/// An event is something with a definite time and (usually) other people.
/// Calendar entries are events. So are gathering moments inferred from
/// Photos (a cluster of photos at one place on one date with two known
/// faces). So is a long phone call with one person.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub title: String,
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
    /// One of: meeting, dinner, trip, milestone, call, gathering, other.
    pub kind: Option<String>,
    pub place_id: Option<String>,
    #[serde(default)]
    pub attendee_ids: Vec<String>,
    pub notes: Option<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

/// An episode is a higher-level slice of life — "the Vermont weekend,"
/// "the LA decision arc." It binds multiple events + entities together
/// across time. Produced by synthesis, not by extractors directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub label: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    #[serde(default)]
    pub event_ids: Vec<String>,
    #[serde(default)]
    pub people_ids: Vec<String>,
    #[serde(default)]
    pub place_ids: Vec<String>,
    pub summary: Option<String>,
}

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

/// A project is a recurring named concept the user is working on, inferred
/// from named notes, recurring file patterns, recurring messages, git
/// repos, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    /// One of: active, dormant, done, abandoned, unknown.
    pub state: Option<String>,
    #[serde(default)]
    pub people_ids: Vec<String>,
    pub last_activity: Option<DateTime<Utc>>,
    pub summary: Option<String>,
    #[serde(default)]
    pub sources: Vec<String>,
}

// ---------------------------------------------------------------------------
// Themes
// ---------------------------------------------------------------------------

/// A theme is a recurring topic across sources — surfaced by keyword
/// frequency + recency. Trajectory tracks whether the theme is rising
/// (more frequent recently than historically), stable, or falling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub keywords: Vec<String>,
    pub mentions_30d: u64,
    pub mentions_180d: u64,
    /// One of: rising, stable, falling, dormant.
    pub trajectory: Option<String>,
    pub first_seen: Option<DateTime<Utc>>,
    pub last_seen: Option<DateTime<Utc>>,
    #[serde(default)]
    pub source_kinds: Vec<String>,
    #[serde(default)]
    pub example_quotes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Open loops
// ---------------------------------------------------------------------------

/// Open loops are the live things in a person's life: unanswered
/// questions, undecided decisions, unsent drafts, unread invites, plans
/// that never crystallized into action.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenLoop {
    pub id: String,
    /// One of: unanswered_question, undecided, unsent_draft,
    /// missed_followup, planned_unscheduled, recurring_topic_no_action.
    pub kind: String,
    pub summary: String,
    pub related_person_ids: Vec<String>,
    pub related_theme_ids: Vec<String>,
    /// Excerpt of the originating text (truncated).
    pub excerpt: Option<String>,
    pub opened_at: DateTime<Utc>,
    pub last_touched: Option<DateTime<Utc>>,
    /// 0.0-1.0 — how alive this loop still is (decays with age, boosted
    /// by recent related activity).
    pub liveness: f32,
    pub source: String,
}

// ---------------------------------------------------------------------------
// Taste
// ---------------------------------------------------------------------------

/// A music/podcast/book taste signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TasteItem {
    pub id: String,
    /// One of: artist, album, track, podcast, book.
    pub kind: String,
    pub name: String,
    pub creator: Option<String>,
    pub play_count: u64,
    pub last_played: Option<DateTime<Utc>>,
    pub source: String,
}

// ---------------------------------------------------------------------------
// Files / code / browser
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSignal {
    pub id: String,
    pub path: String,
    pub name: String,
    pub extension: Option<String>,
    pub modified: Option<DateTime<Utc>>,
    pub size_bytes: u64,
    /// Inferred kind — code, document, image, design, archive, etc.
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRepo {
    pub id: String,
    pub path: String,
    pub name: String,
    pub language: Option<String>,
    pub last_commit: Option<DateTime<Utc>>,
    pub commit_count_30d: u64,
    pub branches: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSignal {
    pub id: String,
    pub domain: String,
    pub visits_30d: u64,
    pub visits_180d: u64,
    pub last_visit: Option<DateTime<Utc>>,
    /// Best-guess category — research, social, work, news, shopping,
    /// reference, entertainment.
    pub category: Option<String>,
    /// Which browser the visits came from. Lets the agent reason
    /// across browsers without merging conflicting signals at this
    /// layer. "safari" | "chrome" | "arc" | "merged".
    #[serde(default)]
    pub browser: Option<String>,
}

// ---------------------------------------------------------------------------
// Behavioral / attention signal — Phase 2 additions for the
// chief-of-staff agent. Aggregated; raw events never leave the graph.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppUsage {
    pub id: String,
    pub bundle_id: String,
    pub display_name: Option<String>,
    /// Total foreground minutes over the trailing 30 / 180 day windows
    pub minutes_30d: u64,
    pub minutes_180d: u64,
    pub last_used: Option<DateTime<Utc>>,
    /// "social", "work", "entertainment", "productivity",
    /// "communication", "reference", "developer", "other"
    pub category: Option<String>,
    /// Minute distribution across hour-of-day (UTC, 0..24) and
    /// day-of-week (Monday=0). Short fixed arrays so JSONL stays small.
    pub by_hour: [u64; 24],
    pub by_dow: [u64; 7],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCommand {
    pub id: String,
    /// Root command — "git", "npm", "claude", "cd". Arguments are
    /// dropped; we never store the full command line.
    pub command_root: String,
    pub count_30d: u64,
    pub count_180d: u64,
    pub last_used: Option<DateTime<Utc>>,
    /// "vcs" | "package" | "shell" | "editor" | "network" | "build" | "other"
    pub category: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationSignal {
    pub id: String,
    pub bundle_id: String,
    pub display_name: Option<String>,
    pub count_30d: u64,
    pub count_180d: u64,
    pub last_received: Option<DateTime<Utc>>,
    /// Same buckets as AppUsage so the synthesis layer can correlate
    /// (a phone full of social notifications correlates with social
    /// app usage, etc.)
    pub category: Option<String>,
}

// ---------------------------------------------------------------------------
// Edges (cross-source links)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub from_type: String,
    pub from_id: String,
    pub to_type: String,
    pub to_id: String,
    pub kind: String,
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Entity kind tags — used by GraphStore as filenames
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Person,
    Place,
    Event,
    Episode,
    Project,
    Theme,
    OpenLoop,
    TasteItem,
    FileSignal,
    CodeRepo,
    WebSignal,
    AppUsage,
    ShellCommand,
    NotificationSignal,
    Edge,
}

impl EntityKind {
    pub fn filename(self) -> &'static str {
        match self {
            EntityKind::Person => "people.jsonl",
            EntityKind::Place => "places.jsonl",
            EntityKind::Event => "events.jsonl",
            EntityKind::Episode => "episodes.jsonl",
            EntityKind::Project => "projects.jsonl",
            EntityKind::Theme => "themes.jsonl",
            EntityKind::OpenLoop => "open_loops.jsonl",
            EntityKind::TasteItem => "taste.jsonl",
            EntityKind::FileSignal => "files.jsonl",
            EntityKind::CodeRepo => "repos.jsonl",
            EntityKind::WebSignal => "web.jsonl",
            EntityKind::AppUsage => "app_usage.jsonl",
            EntityKind::ShellCommand => "shell.jsonl",
            EntityKind::NotificationSignal => "notifications.jsonl",
            EntityKind::Edge => "edges.jsonl",
        }
    }
}
