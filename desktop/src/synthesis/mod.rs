//! Cross-source synthesis.
//!
//! Extractors run independently and write to the same graph. This module
//! does the work that *can't* belong to any one extractor:
//!
//!   * **Entity resolution** — fold five "Sarahs" across iMessage,
//!     Contacts, Mail, Calendar, and Photos into one canonical Person,
//!     emitting reversible `Edge` records that document the merges.
//!   * **Themes** — discover the topics the user keeps returning to
//!     across sources, with a trajectory (rising / stable / falling).
//!   * **Open-loop scoring** — re-rank the open-loop pile by liveness,
//!     decay old loops, attach related people / themes.
//!
//! Runs after any extractor reports new entities, debounced ~10 min in
//! the scheduler so it doesn't thrash.

pub mod entity_resolve;
pub mod themes;
pub mod open_loops;
pub mod episodes;

use crate::extract::{ExtractCtx, ExtractError, ExtractSummary};

pub fn run_all(ctx: &ExtractCtx) -> Result<Vec<ExtractSummary>, ExtractError> {
    let mut out = Vec::new();
    out.push(entity_resolve::run(ctx)?);
    out.push(themes::run(ctx)?);
    out.push(episodes::run(ctx)?);
    out.push(open_loops::run(ctx)?);
    Ok(out)
}
