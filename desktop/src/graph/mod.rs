//! The personal knowledge graph — typed entities extracted from local Mac
//! sources (Contacts, iMessage, Calendar, Photos, Mail, Notes, Safari,
//! files, etc.) and the edges that link them.
//!
//! Two design choices worth knowing:
//!
//!   * **JSONL on disk, append-only.** Each entity type lives in its own
//!     `.jsonl` file under `<storage_root>/users/<uid>/graph/`. The format
//!     is inspectable with `cat` and recoverable with grep, which is the
//!     right default for a system whose entire promise is "you own this."
//!     We'll move to SQLite when concurrent extractors start contending
//!     for the same file, but JSONL is sufficient through Wave 1.
//!
//!   * **Extractors are reflective, not transcriptive.** A row in
//!     `people.jsonl` doesn't just copy fields from Contacts — it carries
//!     the relationship temperature, message cadence, last-seen topic,
//!     etc., computed from cross-referencing iMessage, Mail, Calendar,
//!     and Photos. The graph is supposed to *think about* the user, not
//!     just list them.

pub mod schema;
pub mod store;
pub mod watermarks;

pub use schema::*;
pub use store::GraphStore;
pub use watermarks::Watermarks;
